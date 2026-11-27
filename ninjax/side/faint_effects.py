"""HP commits and ordered singles faint resolution."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import StatEnum
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.trapping import clear_binding
from ninjax.side.updates import conditional_add_boosts, update_active_fields


def _end_fainted_effects(state, *, target):
    """End represented effects that require the original holder to stay active."""
    state = clear_destiny_bond(state, target)
    state = clear_binding(state, 1 - target)
    v = state.volatile_status
    v = v.replace(
        ability_state=v.ability_state.replace(
            ability_override=v.ability_state.ability_override.at[target, 0].set(-1),
            ability_suppressed=v.ability_state.ability_suppressed.at[target, 0].set(False),
            paradox_stat=v.ability_state.paradox_stat.at[target, 0].set(-1),
            paradox_booster=v.ability_state.paradox_booster.at[target, 0].set(False),
            flash_fire=v.ability_state.flash_fire.at[target, 0].set(False)),
        continuation=v.continuation.replace(
            charging_move_id=v.continuation.charging_move_id.at[target, 0].set(0),
            semi_invulnerability=v.continuation.semi_invulnerability.at[target, 0].set(0),
            recharging=v.continuation.recharging.at[target, 0].set(False)),
        substitute_hp=v.substitute_hp.at[target, 0].set(0))
    return state.replace(volatile_status=v,
        switch_requested=state.switch_requested.at[target].set(False))


def _apply_ko_ability(state, *, source):
    """Award the living move source its after-faint boost."""
    ability = state.ability[source, 0]
    beast = ability == AbilityEnum.BEAST_BOOST
    best_stat = 1 + jnp.argmax(state.active_stats(source)[1:])
    stat = jnp.where(beast, best_stat, StatEnum.ATTACK)
    return conditional_add_boosts(state, source, beast | (ability == AbilityEnum.MOXIE), stat, 1)


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def _resolve_faints_retaliate(s, *, target, safe_source):
    """Commit the direct faint with an explicit Destiny Bond cause."""
    context = s.event_context
    s = s.replace(
        event_context=s.event_context.replace(
            source=jnp.asarray(target, jnp.int32),
            cause=jnp.int32(Cause.DESTINY_BOND),
            move_id=jnp.int32(194)))
    s = direct_faint(s, safe_source, target, Cause.DESTINY_BOND)
    return s.replace(event_context=context)


def _resolve_next_faint(_, s):
    """Consume the oldest record and allow its reactions to queue a new faint."""
    faints = s.pending_faints
    target = jnp.argmin(jnp.where(faints.pending, faints.order, jnp.iinfo(jnp.int32).max))
    return jax.lax.cond(jnp.any(faints.pending), _resolve_faint, _keep_faint_state, s, target)


def _keep_faint_state(state, target):
    """Keep state once the pending records have drained."""
    return state


def _resolve_faint(s, target):
    """Run faint and after-faint effects with their recorded attribution."""
    faints = s.pending_faints
    source = faints.source[target]
    safe_source = jnp.clip(source, 0, 1)
    matches_target = s.active_index[target, 0] == faints.target_slot[target]
    opposing_move = ((faints.cause[target] == Cause.MOVE) & (source >= 0) &
                     (source < 2) & (source != target) &
                     (s.active_index[safe_source, 0] == faints.source_slot[target]))
    bond = (matches_target & faints.destiny_bond[target] & opposing_move &
            (s.active_field('current_hp', safe_source)[0] > 0))
    s = s.replace(pending_faints=faints.replace(
        pending=faints.pending.at[target].set(False),
        order=faints.order.at[target].set(0),
        destiny_bond=faints.destiny_bond.at[target].set(False)))
    s = jax.lax.cond(matches_target, partial(_end_fainted_effects, target=target), _identity, s)
    retaliate = partial(_resolve_faints_retaliate, target=target, safe_source=safe_source)
    s = jax.lax.cond(bond, retaliate, _identity, s)
    # Terminal battles end before the AfterFaint ability event.
    remaining = jnp.any((s.team.current_hp[..., 0] > 0) & s.roster, axis=1)
    boost = opposing_move & (s.active_field('current_hp', safe_source)[0] > 0) & jnp.all(remaining)
    s = jax.lax.cond(boost, partial(_apply_ko_ability, source=safe_source), _identity, s)
    return resolve_terminal_outcome(s, remaining, target)


def resolve_terminal_outcome(state, remaining, last_fainted):
    """Set the singles winner after all faints from this checkpoint resolve.

    A double knockout goes to the side whose final Pokémon fainted last.
    The producer sets the order, including for self-KO moves and Perish Song.
    """
    winner = jnp.where(jnp.any(remaining), jnp.argmax(remaining), last_fainted)
    terminal = ~jnp.all(remaining) & ~jnp.any(state.pending_faints.pending)
    return state.replace(winner=jnp.where(terminal, winner, state.winner).astype(jnp.int32))


def record_faint(state, target, source, cause, enabled):
    """Save a faint's identity and cause independently of observation events."""
    faints = state.pending_faints
    enabled = jnp.asarray(enabled).reshape(()) & ~faints.pending[target]
    source = jnp.asarray(source, jnp.int32)
    source_slot = jnp.where(source >= 0, state.active_index[jnp.clip(source, 0, 1), 0], -1)
    return state.replace(pending_faints=faints.replace(
        pending=faints.pending.at[target].set(faints.pending[target] | enabled),
        order=faints.order.at[target].set(jnp.where(enabled, jnp.max(faints.order) + 1, faints.order[target])),
        source=faints.source.at[target].set(jnp.where(enabled, source, faints.source[target])),
        source_slot=faints.source_slot.at[target].set(jnp.where(enabled, source_slot, faints.source_slot[target])),
        target_slot=faints.target_slot.at[target].set(jnp.where(
            enabled, state.active_index[target, 0], faints.target_slot[target])),
        cause=faints.cause.at[target].set(jnp.where(enabled, cause, faints.cause[target])),
        destiny_bond=faints.destiny_bond.at[target].set(jnp.where(
            enabled, state.volatile_status.destiny_bond[target, 0], faints.destiny_bond[target]))))


@jax.jit
def commit_hp_change(state, player, old_hp, new_hp, max_hp, source, cause, attack=False):
    """Commit HP and record a new faint once, independently of public events."""
    # Ordinary HP changes cannot revive a fainted Pokémon.
    new_hp = jnp.where(old_hp > 0, new_hp, old_hp)
    fainted = (old_hp[0] > 0) & (new_hp[0] == 0)
    state = update_active_fields(state, player, current_hp=new_hp, is_alive=new_hp > 0)
    v = state.volatile_status
    loss = jnp.where(attack, jnp.maximum(old_hp - new_hp, 0), v.move_history.last_damage[player])
    state = state.replace(volatile_status=v.replace(move_history=v.move_history.replace(last_damage=v.move_history.last_damage.at[player].set(loss))))
    delta = new_hp - old_hp
    public_delta = (jnp.ceil(new_hp / max_hp * 100) - jnp.ceil(old_hp / max_hp * 100)) / 100
    state = emit(state, EventKind.HP, player, detail=jnp.asarray(attack, jnp.int32),
                 before=old_hp, after=new_hp, hp_change=delta,
                 hp_fraction_change=public_delta, enabled=delta != 0)
    state = emit(state, EventKind.FAINT, player, enabled=fainted)
    return record_faint(state, player, source, cause, fainted)


def direct_faint(state, player, source, cause):
    """Faint an active Pokémon without damage protections or damage callbacks."""
    hp = state.active_field('current_hp', player)
    previous_source, previous_cause = state.event_context.source, state.event_context.cause
    state = state.replace(
        event_context=state.event_context.replace(
            source=jnp.asarray(source, jnp.int32),
            cause=jnp.asarray(cause, jnp.int32)))
    state = commit_hp_change(state, player, hp, jnp.zeros_like(hp), state.active_stats(player)[0],
                             source, cause)
    return state.replace(
        event_context=state.event_context.replace(
            source=previous_source,
            cause=previous_cause))


def clear_destiny_bond(state, player):
    """End the active volatile when its user acts, is interrupted or switches."""
    v = state.volatile_status
    active = v.destiny_bond[player, 0]
    state = state.replace(volatile_status=v.replace(destiny_bond=v.destiny_bond.at[player, 0].set(False)))
    return emit(state, EventKind.CONDITION, player, detail=16, before=1, after=0, enabled=active)


@jax.jit
def resolve_faints(state):
    """Resolve pending singles faint effects at an action or residual checkpoint.

    Records identify the actual source and team slots. Destiny Bond directly
    faints a surviving opposing attacker; it is neither damage nor another
    opposing move and therefore cannot recurse or invoke damage protections.
    """
    # Each active can faint once at this checkpoint, including Bond retaliation.
    return jax.lax.fori_loop(0, 2, _resolve_next_faint, state)
