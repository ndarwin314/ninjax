"""Status helpers for immutable JAX battle state."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import worry_seed
from ninjax.battle.switching import force_switch_target
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.moves import ProtectionKind
from ninjax.enums.pokemon import Status
from ninjax.moves.model import Move
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.restrictions import apply_restriction
from ninjax.side.faint_effects import clear_destiny_bond, direct_faint
from ninjax.side.field_conditions import field_status, set_terrain, set_weather
from ninjax.side.hp_effects import take_damage_percent, take_damage_value
from ninjax.side.item_effects import after_hp_change, status_berry, swap_items
from ninjax.side.stat_effects import stat_changes
from ninjax.side.state import BattleState
from ninjax.side.status_effects import set_status
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


def do_status_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Execute a status move directly, including immediate forced switching."""
    return jax.lax.cond(
        move.identifier[0] == 18,
        partial(force_switch_target, attacker=attacker_idx),
        partial(
            _status_move_effects,
            attacker_idx=attacker_idx,
            move=move,
            stat_index=stat_index,
            boost_value=boost_value,
        ),
        key,
        state,
    )


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


def _status_worry_seed(k, s, *, attacker_idx):
    """Apply Worry Seed while retaining the current random key."""

    return (k, worry_seed(s, attacker_idx))


def _status_worry_seed_or_generic(k, s, *, move, stat_index, attacker_idx, boost_value):
    """Route Worry Seed before applying ordinary status metadata."""
    return jax.lax.cond(
        move.identifier[0] == 388,
        partial(_status_worry_seed, attacker_idx=attacker_idx),
        partial(
            _generic_status_move,
            attacker_idx=attacker_idx,
            move=move,
            stat_index=stat_index,
            boost_value=boost_value,
        ),
        k,
        s,
    )


def _status_substitute_or_other(k, s, *, move, stat_index, attacker_idx, boost_value):
    """Route Substitute before the remaining status effects."""
    return jax.lax.cond(
        move.identifier[0] == 164,
        partial(_status_substitute, attacker_idx=attacker_idx),
        partial(
            _status_worry_seed_or_generic,
            move=move,
            stat_index=stat_index,
            attacker_idx=attacker_idx,
            boost_value=boost_value,
        ),
        k,
        s,
    )


def _status_destiny_bond_or_other(k, s, *, move, stat_index, attacker_idx, boost_value):
    """Route Destiny Bond before the remaining status effects."""
    return jax.lax.cond(
        move.identifier[0] == 194,
        partial(_status_destiny_bond, attacker_idx=attacker_idx),
        partial(
            _status_substitute_or_other,
            move=move,
            stat_index=stat_index,
            attacker_idx=attacker_idx,
            boost_value=boost_value,
        ),
        k,
        s,
    )


def _status_item_swap_or_other(k, s, *, move, stat_index, attacker_idx, boost_value):
    """Route Trick and Switcheroo before the remaining status effects."""
    return jax.lax.cond(
        (move.identifier[0] == 271) | (move.identifier[0] == 415),
        partial(_status_swap_items, attacker_idx=attacker_idx),
        partial(
            _status_destiny_bond_or_other,
            move=move,
            stat_index=stat_index,
            attacker_idx=attacker_idx,
            boost_value=boost_value,
        ),
        k,
        s,
    )


def _status_other(k, s, *, move, attacker_idx, stat_index, boost_value):
    """Handle a direct ability change or generic status effects."""
    return jax.lax.cond(
        move.identifier[0] == 262,
        partial(_status_memento, attacker_idx=attacker_idx),
        partial(
            _status_item_swap_or_other,
            move=move,
            stat_index=stat_index,
            attacker_idx=attacker_idx,
            boost_value=boost_value,
        ),
        k,
        s,
    )


def _status_direct(k, s, *, attacker_idx, move):
    """Resolve Rest or Pain Split through their direct status implementation."""

    return direct_status_move(k, s, attacker_idx, move)


def _status_move_effects(key, state, attacker_idx, move, stat_index, boost_value):
    """Execute status effects other than forced switching, which dispatch defers."""
    return jax.lax.cond(
        (move.identifier[0] == 156) | (move.identifier[0] == 220),
        partial(_status_direct, attacker_idx=attacker_idx, move=move),
        partial(
            _status_other,
            move=move,
            attacker_idx=attacker_idx,
            stat_index=stat_index,
            boost_value=boost_value,
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


def _generic_status_move(key, state, attacker_idx, move, stat_index, boost_value):
    """Apply the generic status-move fields and return the key and state.

    The selected target can receive healing, status, stat stages, and hazards;
    weather, terrain, and Perish Song update field-wide state. Effect coverage
    is partial, and field durations are currently fixed at five turns.
    """
    # Consecutive Protect attempts succeed with probability 1 / 3**chain.
    key, protect_key = random.split(key)
    v = state.volatile_status
    protect = (move.identifier[0] == 182) | (move.identifier[0] == 596)
    success = protect & (random.uniform(protect_key) < jnp.power(1 / 3., v.protect_chain[attacker_idx, 0]))
    v = v.replace(protected=v.protected.at[attacker_idx, 0].set(success),
                  protection_kind=v.protection_kind.at[attacker_idx, 0].set(jnp.where(success,
                      jnp.where(move.identifier[0] == 596, ProtectionKind.SPIKY_SHIELD, ProtectionKind.PROTECT), ProtectionKind.NONE)),
                  protect_chain=v.protect_chain.at[attacker_idx, 0].set(
                      jnp.where(protect & success, jnp.minimum(v.protect_chain[attacker_idx, 0] + 1, 20), 0)))
    state = state.replace(volatile_status=v)
    target = jnp.where(move.target_self[0], attacker_idx, 1 - attacker_idx)
    # Record these explicit failure conditions independently of HP deltas, so
    # previous-turn move results distinguish failed healing/status attempts.
    heal_failed = ((move.identifier[0] == 105) | (move.identifier[0] == 355)) & (state.active.current_hp[target, 0] >= state.active.max_hp[target])
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
    state = field_status(state, attacker_idx, move)

    state = apply_restriction(state, attacker_idx, move)
    state = set_weather(state, move.weather_effect[0], 5)
    state = set_terrain(state, move.terrain_effect[0], 5)
    state = state.replace(perish_turns=jnp.where(move.perish_effect[0] & (state.perish_turns < 0) & (state.ability != AbilityEnum.SOUNDPROOF),
                                                4, state.perish_turns))
    # Hazard IDs encode none, Stealth Rock, Sticky Web, Spikes, and Toxic
    # Spikes. Layered hazards are capped at their represented maximum.
    hazard = move.hazard_effect[0]
    state = state.replace(
        stealth_rocks=state.stealth_rocks.at[target].set(state.stealth_rocks[target] | (hazard == 1)),
        sticky_webs=state.sticky_webs.at[target].set(state.sticky_webs[target] | (hazard == 2)),
        spikes=state.spikes.at[target].set(jnp.minimum(3, state.spikes[target] + (hazard == 3))),
        toxic_spikes=state.toxic_spikes.at[target].set(jnp.minimum(2, state.toxic_spikes[target] + (hazard == 4))))
    failed = (protect & ~success) | heal_failed | status_failed | stat_failed
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
