"""Status helpers for immutable JAX battle state."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import ability_move
from ninjax.battle.switching import force_switch_target, forced_switch_move, queue_entry_recovery, reflected_switching_source
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.moves import MoveFlags
from ninjax.enums.field import TerrainEnum
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.moves.model import Move
from ninjax.moves.protection import apply_protection, blocked_by_protection
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.restrictions import apply_persistent_restriction, apply_restriction
from ninjax.side.faint_effects import clear_destiny_bond, direct_faint
from ninjax.side.field_conditions import FIELD_CONDITION_MOVES, field_condition_effect, field_status, set_terrain, set_weather
from ninjax.side.hp_effects import take_damage_percent, take_damage_value
from ninjax.side.item_effects import after_hp_change, force_eat_berry, is_berry, recycle_item, status_berry, swap_items
from ninjax.side.property_effects import COPY_MOVES, STAT_MOVES, TYPE_MOVES, copy_move, stat_move, transform, type_move
from ninjax.side.queries import ability_against_move
from ninjax.side.residual_conditions import RESIDUAL_STATUS_MOVES, apply_residual_status
from ninjax.side.stat_effects import stat_changes
from ninjax.side.stat_healing import belly_drum, stockpile, strength_sap, swallow, weather_heal
from ninjax.side.state import BattleState
from ninjax.side.status_effects import apply_confusion, set_status
from ninjax.side.updates import update_active_fields


def _direct_status_move_heal(c, *, player, mon):
    """Use the existing sleep counter's decrement-before-action convention."""

    c = update_active_fields(c, player, sleep_counter=jnp.array([3]))
    c = c.replace(toxic_counter=c.toxic_counter.at[player].set(0))
    c = emit(c, EventKind.STATUS, player, before=mon.status[0], after=Status.SLEEP)

    return status_berry(take_damage_value(c, player, -mon.max_hp, False), player)


def _direct_status_move_restore_state(_, *, s):
    """Direct status move restore state."""
    return s


def _preserve_operand(s):
    """Preserve operand."""
    return s


def _direct_status_move_record_hit(s, *, player):
    """Direct status move record hit."""
    return record_hit(s, 0, player, HitOutcome.FAILED)


def _direct_status_move_rest(args, *, player):
    """Replace an existing major status with two full turns of Rest sleep."""
    key, s = args
    mon = s.active[player]
    eligible = (mon.current_hp[0] < mon.max_hp) & (mon.status[0] != Status.SLEEP) & (mon.current_hp[0] > 0)
    # Reuse terrain and ability sleep checks, but allow Rest to replace a
    # burn/poison/paralysis. Failed sleep attempts restore the original state.
    candidate = update_active_fields(s, player, status=jnp.zeros_like(mon.status))
    key, candidate = set_status(key, candidate, player, jnp.array([Status.SLEEP]), activate_items=False)
    success = eligible & (candidate.active.status[player, 0] == Status.SLEEP)
    heal = partial(_direct_status_move_heal, player=player, mon=mon)
    result = jax.lax.cond(success, heal, partial(_direct_status_move_restore_state, s=s), candidate)
    # Rest's failure needs an explicit move outcome for Stomping Tantrum.
    result = jax.lax.cond(success, _preserve_operand,
        partial(_direct_status_move_record_hit, player=player), result)
    return key, result


def _direct_status_move_assign(p, s, *, average):
    """Assign HP directly: Pain Split is neither damage nor healing."""
    mon = s.active[p]
    hp = jnp.minimum(average, mon.max_hp).reshape((1,))
    s = emit(s, EventKind.HP, p, before=mon.current_hp, after=hp,
        hp_change=hp-mon.current_hp,
        hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
        enabled=hp[0] != mon.current_hp[0])
    return update_active_fields(s, p, current_hp=hp)


def _direct_status_move_after_hp_change(p, s):
    """Direct status move after hp change."""
    return after_hp_change(s, p)


def _direct_status_move_split(args):
    """Set both HP values to their original mean before reactive healing."""


    key, s = args
    average = jnp.maximum(1, jnp.sum(s.active.current_hp) // 2)
    assign = partial(_direct_status_move_assign, average=average)
    s = jax.lax.fori_loop(0, 2, assign, s)
    return key, jax.lax.fori_loop(0, 2, _direct_status_move_after_hp_change, s)


def _pay_substitute_cost(s, *, player, cost):
    """Apply direct HP cost, then announce the new Substitute."""
    s = take_damage_value(s, player, jnp.array([cost]), False)
    v = s.volatile_status
    s = s.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[player, 0].set(cost)))
    return emit(s, EventKind.CONDITION, player, detail=13, before=0, after=1)


def create_substitute(state, player):
    """Spend a quarter of maximum HP to create a switch-cleared Substitute."""
    mon = state.active[player]
    cost = mon.max_hp // 4
    v = state.volatile_status
    eligible = (v.substitute_hp[player, 0] == 0) & (mon.max_hp > 1) & (4 * mon.current_hp[0] > mon.max_hp)
    return jax.lax.cond(eligible, partial(_pay_substitute_cost, player=player, cost=cost), _identity, state)


def create_shed_tail_substitute(state, player):
    """Pay half maximum HP and create a quarter-HP Shed Tail Substitute."""
    max_hp = state.active.max_hp[player]
    cost = (max_hp + 1) // 2
    state = take_damage_value(state, player, jnp.array([cost]), False)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        substitute_hp=v.substitute_hp.at[player, 0].set(max_hp // 4)))
    return emit(state, EventKind.CONDITION, player, detail=13, before=0, after=1)


def do_status_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Execute a status move and commit its drag through the shared continuation."""
    forced = forced_switch_move(move)
    key, state = jax.lax.cond(
        forced,
        _unchanged_key_state,
        partial(_status_move_effects, attacker_idx=attacker_idx, move=move,
                stat_index=stat_index, boost_value=boost_value),
        key, state)
    source = reflected_switching_source(state, attacker_idx, move)
    return force_switch_target(key, state, source, wind=move.identifier[0] == 18,
                               enabled=forced)


def _record_item_swap_blocked(s, *, attacker_idx):
    """Record a blocked Trick or Switcheroo attempt."""
    return record_hit(s, 0, 1-attacker_idx, HitOutcome.BLOCKED)


def _status_swap_items(k, s, *, attacker_idx):
    """Exchange both held items, marking an unsuccessful attempt."""

    s, success = swap_items(s, attacker_idx, 1-attacker_idx)
    s = jax.lax.cond(
        success,
        _identity,
        partial(_record_item_swap_blocked, attacker_idx=attacker_idx),
        s,
    )
    return k, s


def _apply_memento(s, *, attacker_idx):
    """Faint even when abilities or stage limits prevent the drops."""


    s = stat_changes(s, 1-attacker_idx, jnp.array([1, 3]), jnp.array([-2, -2]), True)
    return direct_faint(s, attacker_idx, attacker_idx, Cause.SELF_KO)


def _status_memento(k, s, *, attacker_idx):
    """Lower both offensive stats and directly faint after reaching target."""
    result = s.hit_results[0]
    reached = (result.outcome == HitOutcome.HIT) & (result.recipient == HitRecipient.POKEMON)
    return k, jax.lax.cond(reached, partial(_apply_memento, attacker_idx=attacker_idx), _identity, s)


def _status_destiny_bond(k, s, *, attacker_idx):
    """Activate Destiny Bond without consuming another random key."""
    return (k, use_destiny_bond(s, attacker_idx))


def _status_substitute(k, s, *, attacker_idx):
    """Create Substitute without consuming another random key."""
    return (k, create_substitute(s, attacker_idx))


def _ability_status(key, state, *, player, ident):
    """Dispatch an ability override or suppression."""
    state, success = ability_move(state, player, ident)
    return key, state, success


def _stat_status(key, state, *, player, ident):
    """Dispatch a stage or raw stat replacement."""
    return key, stat_move(state, player, ident), jnp.bool_(True)


def _copy_status(key, state, *, player, ident):
    """Dispatch moveset copying and full Transform."""
    state, success = jax.lax.cond(ident == 144, _transform_move, copy_move, state, player, ident)
    return key, state, success


def _transform_move(state, player, ident):
    """Return the result of a full Transform."""
    return transform(state, player)


def _type_status(key, state, *, player, ident):
    """Dispatch a type change with its explicit PRNG key."""
    return type_move(key, state, player, ident)


def _blocked_property(key, state):
    """Skip a reflected move whose new target blocks it."""
    return key, state, jnp.bool_(False)


def _status_property(key, state, *, attacker_idx, move):
    """Mutate active properties and mark unsuccessful attempts."""
    ident = move.identifier[0]
    branch = jnp.select((jnp.isin(ident, jnp.array(TYPE_MOVES)), jnp.isin(ident, jnp.array(STAT_MOVES)),
                         jnp.isin(ident, jnp.array(COPY_MOVES))), (1, 2, 3), default=0)
    reflectable = (move.move_flags[0] & MoveFlags.REFLECTABLE) != 0
    source = jnp.where(reflectable, reflected_switching_source(state, attacker_idx, move), attacker_idx)
    target = jnp.where(move.target_self[0], source, 1-source)
    blocked = (source != attacker_idx) & (blocked_by_protection(state, move, source) |
        (ability_against_move(state, target, move)[0] == AbilityEnum.GOOD_AS_GOLD))
    handlers = (_ability_status, _type_status, _stat_status, _copy_status)
    key, state, success = jax.lax.switch(jnp.where(blocked, 4, branch),
        tuple(partial(handler, player=source, ident=ident) for handler in handlers) + (_blocked_property,), key, state)
    return key, jax.lax.cond(success, _identity, partial(_record_status_failure, target=target), state)


def _status_direct(k, s, *, attacker_idx, move):
    """Resolve Rest or Pain Split through their direct status implementation."""

    return direct_status_move(k, s, attacker_idx, move)


def _status_persistent_restriction(key, state, *, attacker_idx, move):
    """Apply Torment or Imprison and mark a repeat or immune attempt as failed."""
    state, success = apply_persistent_restriction(state, attacker_idx, move)
    target = jnp.where(move.target_self[0], attacker_idx, 1 - attacker_idx)
    state = jax.lax.cond(success, _identity, partial(_record_status_failure, target=target), state)
    return key, state


def _status_recovery_request(key, state, *, attacker_idx, move):
    """Request revival or sacrifice the user for a later entrant's recovery."""
    player = attacker_idx
    ident = move.identifier[0]
    slots = jnp.arange(6) != state.active_index[player, 0]
    hp = state.team.current_hp[player, :, 0]
    candidates = slots & state.roster[player]
    revive = ident == 863
    eligible = jnp.any(candidates & jnp.where(revive, hp == 0, hp > 0))
    state = state.replace(
        revival_requested=state.revival_requested.at[player].set(revive & eligible))
    state = queue_entry_recovery(state, player, lunar_dance=ident == 461, enabled=~revive & eligible)
    state = jax.lax.cond(~revive & eligible, partial(direct_faint, player=player, source=player, cause=Cause.SELF_KO),
                         _identity, state)
    return key, jax.lax.cond(revive & ~eligible, partial(_record_status_failure, target=player), _identity, state)


def _stuff_cheeks(state, *, player):
    """Gain Defense before force-feeding the user's berry."""
    berry = is_berry(state.active_field('item', player)[0])
    defense = StatEnum.DEFENSE
    before = state.boosts.normal_boosts[player, defense]
    state = stat_changes(state, player, jnp.array([defense]), jnp.array([jnp.where(berry, 2, 0)]))
    boosted = state.boosts.normal_boosts[player, defense] != before
    state = force_eat_berry(state, player, boosted)
    return jax.lax.cond(~boosted, partial(_record_status_failure, target=player), _identity, state)


def _teatime_eater(player, state, *, eligible):
    """Feed one of the eligible active Pokémon."""
    return force_eat_berry(state, player, eligible[player])


def _teatime(state, *, player):
    """Gather eligible holders, then visit them in side order for Teatime."""
    hidden = state.volatile_status.continuation.semi_invulnerability[:, 0] != 0
    ability = state.ability[:, 0]
    own = jnp.arange(2) == player
    reached = ~hidden | own | (ability == AbilityEnum.NO_GUARD) | (ability[player] == AbilityEnum.NO_GUARD)
    eligible = (reached & (own | (ability != AbilityEnum.GOOD_AS_GOLD)) &
                (state.active_field('current_hp')[:, 0] > 0) & is_berry(state.active_field('item')[:, 0]))
    state = jax.lax.fori_loop(0, 2, partial(_teatime_eater, eligible=eligible), state)
    return jax.lax.cond(jnp.any(eligible), _identity, partial(_record_status_failure, target=player), state)


def _recycle(state, *, player):
    """Restore recoverable item memory or record a failed attempt."""
    state, success = recycle_item(state, player)
    return jax.lax.cond(success, _identity, partial(_record_status_failure, target=player), state)


def _status_item_use(key, state, *, attacker_idx, move):
    """Dispatch forced berry use or held-item recovery."""
    ident = move.identifier[0]
    branch = jnp.select((ident == 747, ident == 752), (0, 1), default=2)
    return key, jax.lax.switch(branch,
        (partial(_stuff_cheeks, player=attacker_idx), partial(_teatime, player=attacker_idx),
         partial(_recycle, player=attacker_idx)), state)


def _parting_shot(key, state, *, attacker_idx, move):
    """Apply the drops and request a pivot for the resolved user."""
    source = reflected_switching_source(state, attacker_idx, move)
    reflected = source != attacker_idx
    target = 1 - source
    ability = ability_against_move(state, target, move)[0]
    immune = (ability == AbilityEnum.SOUNDPROOF) | (ability == AbilityEnum.GOOD_AS_GOLD)
    blocked = blocked_by_protection(state, move, attacker_idx) | (
        reflected & blocked_by_protection(state, move, source))
    permitted = ~immune & ~blocked
    mirror = ability == AbilityEnum.MIRROR_ARMOR
    recipient = jnp.where(mirror, source, target)
    before = state.boosts.normal_boosts[recipient]
    context = state.event_context
    state = state.replace(event_context=context.replace(source=jnp.where(mirror, target, source)))
    state = stat_changes(state, recipient, move.boost_stat, jnp.where(permitted, move.boost_stages, 0), True)
    changed = jnp.any(state.boosts.normal_boosts[recipient] != before)
    # Mirror Armor also allows the pivot when the reflected drops cannot land.
    success = permitted & (changed | mirror)
    hit = jnp.where(success, HitOutcome.HIT, jnp.where(blocked, HitOutcome.PROTECTED,
                    jnp.where(immune, HitOutcome.IMMUNE, HitOutcome.FAILED)))
    state = record_hit(state, 0, target, hit, jnp.where(success, HitRecipient.POKEMON, HitRecipient.NONE))
    return key, state.replace(event_context=context, switch_requested=state.switch_requested.at[source].set(success))


def _skipped_residual_application(state, player, move):
    """Skip a reflected residual move that its new target blocks."""
    return state, jnp.bool_(False)


def _status_residual(key, state, *, attacker_idx, move):
    """Route move-created residuals through eligibility and reflection."""
    from_side = reflected_switching_source(state, attacker_idx, move)
    reflected = ((move.move_flags[0] & MoveFlags.REFLECTABLE) != 0) & (from_side != attacker_idx)
    source = jnp.where(reflected, from_side, attacker_idx)
    target = jnp.where(move.target_self[0], source, 1-source)
    grass_seed_target = (move.identifier[0] == 73) & jnp.any(
        state.effective_types(1-attacker_idx, include_added=True) == Type.GRASS)
    blocked = reflected & (blocked_by_protection(state, move, source) |
        (ability_against_move(state, target, move)[0] == AbilityEnum.GOOD_AS_GOLD))
    blocked |= grass_seed_target
    old = state.event_context
    state = state.replace(event_context=old.replace(source=source))
    state, success = jax.lax.cond(blocked, _skipped_residual_application, apply_residual_status, state, source, move)
    state = state.replace(event_context=old)
    return key, jax.lax.cond(success, _identity, partial(_record_status_failure, target=target), state)


def _special_status(key, state, *, attacker_idx, move):
    """Dispatch stat-cost and contextual healing moves through shared helpers."""
    ident = move.identifier[0]
    branch = jnp.select((ident == 187, ident == 254, ident == 256, ident == 668),
                        (0, 1, 2, 3), default=4)
    return key, jax.lax.switch(branch, (
        partial(belly_drum, player=attacker_idx),
        partial(stockpile, player=attacker_idx),
        partial(swallow, player=attacker_idx),
        partial(strength_sap, player=attacker_idx),
        partial(weather_heal, player=attacker_idx, ident=ident)), state)


def _timed_condition_status(key, state, *, attacker_idx, move):
    """Start or toggle a field condition and record rejected reapplication."""
    state, success = field_condition_effect(state, attacker_idx, move)
    target = jnp.where(move.target_self[0], attacker_idx, 1 - attacker_idx)
    return key, jax.lax.cond(success, _identity, partial(_record_status_failure, target=target), state)


def _status_move_effects(key, state, attacker_idx, move, stat_index, boost_value):
    """Select a status handler, leaving forced switching to its dispatch path."""
    ident = move.identifier[0]
    # Special moves bypass generic status metadata. Unmatched identifiers
    # explicitly select the generic handler, rather than relying on clamping.
    branch = jnp.select(
        ((ident == 156) | (ident == 220),  # Rest / Pain Split
         ident == 262,                     # Memento
         (ident == 271) | (ident == 415),   # Trick / Switcheroo
         ident == 194,                     # Destiny Bond
         ident == 164,                     # Substitute
         jnp.isin(ident, jnp.array((272, 285, 380, 388, 493, 494) + TYPE_MOVES + STAT_MOVES + COPY_MOVES)),
         (ident == 259) | (ident == 286),  # Torment / Imprison
         (ident == 361) | (ident == 461) | (ident == 863),
         (ident == 747) | (ident == 752) | (ident == 278),
         ident == 575, jnp.isin(ident, jnp.array(RESIDUAL_STATUS_MOVES)),
         jnp.isin(ident, jnp.array([187, 254, 256, 668, 235, 236, 234, 659])),
         jnp.isin(ident, jnp.array(FIELD_CONDITION_MOVES))),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        default=13,
    )
    return jax.lax.switch(
        branch,
        (
            partial(_status_direct, attacker_idx=attacker_idx, move=move),
            partial(_status_memento, attacker_idx=attacker_idx),
            partial(_status_swap_items, attacker_idx=attacker_idx),
            partial(_status_destiny_bond, attacker_idx=attacker_idx),
            partial(_status_substitute, attacker_idx=attacker_idx),
            partial(_status_property, attacker_idx=attacker_idx, move=move),
            partial(_status_persistent_restriction, attacker_idx=attacker_idx, move=move),
            partial(_status_recovery_request, attacker_idx=attacker_idx, move=move),
            partial(_status_item_use, attacker_idx=attacker_idx, move=move),
            partial(_parting_shot, attacker_idx=attacker_idx, move=move),
            partial(_status_residual, attacker_idx=attacker_idx, move=move),
            partial(_special_status, attacker_idx=attacker_idx, move=move),
            partial(_timed_condition_status, attacker_idx=attacker_idx, move=move),
            partial(_generic_status_move, attacker_idx=attacker_idx, move=move,
                    stat_index=stat_index, boost_value=boost_value),
        ),
        key,
        state,
    )


def _activate_destiny_bond(s, *, player):
    """Keep the bond until the user's next action attempt or departure."""
    v = s.volatile_status
    s = s.replace(volatile_status=v.replace(destiny_bond=v.destiny_bond.at[player, 0].set(True)))
    return emit(s, EventKind.CONDITION, player, detail=16, before=0, after=1)


def _fail_destiny_bond(s, *, player):
    """Clear and mark a consecutive Destiny Bond attempt as blocked."""

    return record_hit(clear_destiny_bond(s, player), 0, player, HitOutcome.BLOCKED)


def use_destiny_bond(state, player):
    """Activate Destiny Bond, or remove it and fail on consecutive execution."""
    active = state.volatile_status.destiny_bond[player, 0]
    return jax.lax.cond(
        active,
        partial(_fail_destiny_bond, player=player),
        partial(_activate_destiny_bond, player=player),
        state,
    )


def _apply_move_status(k, s, *, target, move):
    """Apply the move status effect to its selected target."""
    return set_status(k, s, target, move.status_effect)


def _record_status_failure(s, *, target):
    """Record an unsuccessful status move independently of HP changes."""
    return record_hit(s, 0, target, HitOutcome.FAILED)


def _apply_primary_confusion(key, state, target):
    """Apply a primary confusion attempt through the shared immunity/berry path."""
    return apply_confusion(key, state, target)


def _skip_primary_confusion(key, state, target):
    """Preserve key/state when this status move has no confusion component."""
    return key, state


def _generic_status_move(key, state, attacker_idx, move, stat_index, boost_value):
    """Apply the generic status-move fields and return the key and state.

    The selected target can receive healing, status, stat stages, and hazards;
    weather, terrain, and Perish Song update field-wide state. Effect coverage
    is partial, and field durations are currently fixed at five turns.
    """
    key, protect_key = random.split(key)
    state, guard_failed = apply_protection(state, attacker_idx, move, random.uniform(protect_key))
    target = jnp.where(move.target_self[0], attacker_idx, 1 - attacker_idx)
    # Record these explicit failure conditions independently of HP deltas, so
    # previous-turn move results distinguish failed healing/status attempts.
    heal_failed = (move.healing_fraction[0] > 0) & (state.active.current_hp[target, 0] >= state.active.max_hp[target])
    status_failed = (move.status_effect[0] != Status.NONE) & (
        (state.active.status[target, 0] != Status.NONE) | state.is_status_immune(target, move.status_effect)[0])
    roost_succeeds = (move.identifier[0] == 355) & (state.active.current_hp[attacker_idx, 0] > 0) & (state.active.current_hp[attacker_idx, 0] < state.active.max_hp[attacker_idx])
    state = take_damage_percent(state, target, -move.healing_fraction)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(roost=v.roost.at[attacker_idx, 0].set(
        v.roost[attacker_idx, 0] | (roost_succeeds & ~state.active.is_terastallized[attacker_idx, 0]))))
    key, state = jax.lax.cond(
        move.status_effect[0] != Status.NONE,
        partial(_apply_move_status, target=target, move=move),
        _unchanged_key_state,
        key,
        state,
    )

    old_normal = state.boosts.normal_boosts[target]
    old_accuracy = state.boosts.acc_boosts[target]
    state = stat_changes(state, target, move.boost_stat, move.boost_stages, target != attacker_idx)
    # Pure stat moves fail if every represented stage change was prevented.
    stat_failed = jnp.any(move.boost_stages != 0) & jnp.all(old_normal == state.boosts.normal_boosts[target]) & jnp.all(old_accuracy == state.boosts.acc_boosts[target])
    # A successful confusion attempt counts even if Lum Berry immediately cures
    # it. Conversely Swagger/Flatter can succeed solely through their boosts.
    confusion_eligible = ((state.volatile_status.confusion_turns[target, 0] == 0) &
        (state.ability[target, 0] != AbilityEnum.OWN_TEMPO) &
        ~((state.terrain.terrain == TerrainEnum.MISTY) & state.is_grounded(target)[0]))
    key, state = jax.lax.cond(move.primary_confusion[0],
        _apply_primary_confusion, _skip_primary_confusion, key, state, target)
    stat_success = jnp.any(move.boost_stages != 0) & ~stat_failed
    confusion_failed = move.primary_confusion[0] & ~confusion_eligible & ~stat_success
    stat_failed &= ~move.primary_confusion[0]
    # Reapplying an active field condition fails without refreshing its timer.
    field_failed = ((move.weather_effect[0] != 0) & (state.weather.weather == move.weather_effect[0])) | (
        (move.terrain_effect[0] != 0) & (state.terrain.terrain == move.terrain_effect[0])) | (
        (move.identifier[0] == 115) & (state.side_conditions.reflect[attacker_idx, 0] > 0))
    # Chilly Reception can pivot even when snow was already active.
    field_failed &= move.identifier[0] != 881
    state = field_status(state, attacker_idx, move)
    # Defense Curl's stage boost and its Rollout power effect have separate
    # lifetimes. The power effect persists even if the Defense stage is capped.
    v = state.volatile_status
    curled = v.defense_curl[attacker_idx, 0] | (move.identifier[0] == 111)
    state = state.replace(volatile_status=v.replace(
        defense_curl=v.defense_curl.at[attacker_idx, 0].set(curled)))

    state = apply_restriction(state, attacker_idx, move)
    state = set_weather(state, move.weather_effect[0], 5)
    state = set_terrain(state, move.terrain_effect[0], 5)
    state = state.replace(perish_turns=jnp.where(move.perish_effect[0] & (state.perish_turns < 0) & (state.ability != AbilityEnum.SOUNDPROOF),
                                                4, state.perish_turns))
    # Hazard IDs encode none, Stealth Rock, Sticky Web, Spikes, and Toxic
    # Spikes. Layered hazards are capped at their represented maximum.
    hazard = move.hazard_effect[0]
    state = state.replace(
        hazards=state.hazards.replace(
            stealth_rocks=state.hazards.stealth_rocks.at[target].set(state.hazards.stealth_rocks[target] | (hazard == 1)),
            sticky_webs=state.hazards.sticky_webs.at[target].set(state.hazards.sticky_webs[target] | (hazard == 2)),
            spikes=state.hazards.spikes.at[target].set(jnp.minimum(3, state.hazards.spikes[target] + (hazard == 3))),
            toxic_spikes=state.hazards.toxic_spikes.at[target].set(jnp.minimum(2, state.hazards.toxic_spikes[target] + (hazard == 4)))))
    failed = guard_failed | heal_failed | status_failed | stat_failed | confusion_failed | field_failed
    state = jax.lax.cond(failed, partial(_record_status_failure, target=target), _identity, state)
    return key, state


def direct_status_move(key, state, player, move):
    """Resolve Rest and Pain Split without running ordinary damage callbacks."""
    rest = partial(_direct_status_move_rest, player=player)
    split = _direct_status_move_split
    return jax.lax.cond(move.identifier[0] == 156, rest, split, (key, state))


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)
