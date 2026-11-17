"""Accuracy, immunity and protection routing into shared move mechanics.

State and PRNG keys remain explicit; callbacks are module-level pure functions.
"""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import underlying_ability
from ninjax.enum_types import (
    AbilityEnum,
    MoveFlags,
    MoveType,
    ProtectionKind,
    StatEnum,
    TerrainEnum,
    Type,
)
from ninjax.events import Cause, EventKind, emit
from ninjax.hit import HitOutcome, HitRecipient, record_hit
from ninjax.knowledge import remember_item
from ninjax.moves.model import Move
from ninjax.moves.multi_hit import do_damaging_move
from ninjax.moves.properties import effective_move, move_accuracy, substitute_intercepts
from ninjax.moves.status import _status_move_effects
from ninjax.side.hp_effects import take_damage_percent
from ninjax.side.queries import contact_reactions_allowed
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts
from ninjax.switching import _no_forced_switch, execute_forced_switch, prepare_forced_switch
from ninjax.utils import quad_or, triple_or


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)


def _unchanged_selected_move(k, s, p, i):
    """Skip a selected move while accepting its player and slot operands."""
    return (k, s)


def _ready_move(k, s):
    """Allow the shared hit pipeline without an accuracy draw."""
    return (k, s, jnp.bool_(True))


def _defer_forced_switch(key, state, attacker_idx, move, stat_index, boost_value):
    """Leave forced switching to the common continuation after hit dispatch."""
    return key, state


def do_healing_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Heal the defending active by a quarter of max HP for an absorb ability."""
    state = take_damage_percent(state, 1-attacker_idx, -1/4)
    return key,state


def do_stat_boost_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply an absorption ability's stage change to the defending active."""
    state = add_boosts(state, 1-attacker_idx, stat_index, boost_value)
    return key, state


def do_flash_fire_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Activate the defending active's switch-cleared Flash Fire flag."""
    volatile = state.volatile_status.replace(
        flash_fire=state.volatile_status.flash_fire.at[1-attacker_idx].set(True))
    return key, state.replace(volatile_status=volatile)


def _apply_move_effects(key, state, attacker_index, move_index):
    """Skip Struggle accuracy while sharing the downstream hit executor."""
    # Forced Struggle already uses -1 as its action-local public move marker.
    # Prepare only a routing flag here so the heavy hit executor has one call.
    key, state, ready = jax.lax.cond(
        state.event_move == -1,
        _ready_move,
        partial(_prepare_accuracy, attacker_index=attacker_index, move_index=move_index),
        key,
        state,
    )
    return jax.lax.cond(ready, _move_hits, _unchanged_selected_move,
        key, state, attacker_index, move_index)


def move_not_drawn_in(key: chex.PRNGKey, state: BattleState, attacker_index, move_index: int, unused: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Sample accuracy and dispatch a hit or miss, returning the advanced key.

    The calculation includes represented accuracy stages and abilities. The
    final ``unused`` operand is retained for compatibility with branch calls.
    """
    return _apply_move_effects(key, state, attacker_index, move_index)


def _block_accuracy_target(k, s, *, attacker_index):
    """Stop a forbidden ability change before ordinary hit resolution."""
    return k, record_hit(emit(s, EventKind.IMMUNE, 1-attacker_index), 0, 1-attacker_index, HitOutcome.IMMUNE), jnp.bool_(False)


def _attempt_accuracy(k, s, *, hit, attacker_index, move_index):
    """Retain the accuracy draw and record a miss when the hit is gated out."""
    k, s = jax.lax.cond(
        hit,
        _unchanged_key_state,
        partial(move_misses, attacker_index=attacker_index, move_index=move_index),
        k,
        s,
    )
    return k, s, hit


def _prepare_accuracy(key, state, attacker_index, move_index):
    """Record misses/immunity and return whether normal hit routing is allowed."""
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    move = effective_move(state, attacker_index, state.active[attacker_index].moves[move_index])
    accuracy = move_accuracy(state, attacker_index, move)

    # Worry Seed fails against these identities before accuracy, including
    # while an ability's effects are suppressed.

    old = underlying_ability(state.active.ability[1-attacker_index, 0], state.volatile_status.ability_override[1-attacker_index, 0])
    cannot_target = (move.identifier[0] == 388) & ((old == AbilityEnum.INSOMNIA) | (old == AbilityEnum.TRUANT))

    # Gen 9 protection resolves before accuracy; an inaccurate contact move
    # still strikes Spiky Shield, and Protect blocks before a miss is rolled.
    bypass = (state.ability[attacker_index, 0] == AbilityEnum.UNSEEN_FIST) & move.contact[0]
    protected = state.volatile_status.protected[1-attacker_index, 0] & ~move.target_self[0] & move.protectable[0] & ~bypass

    hit = (jnp.less_equal(r, accuracy)[0] | protected) & move.implemented[0]
    return jax.lax.cond(
        cannot_target,
        partial(_block_accuracy_target, attacker_index=attacker_index),
        partial(_attempt_accuracy, hit=hit, attacker_index=attacker_index, move_index=move_index),
        key,
        state,
    )


def is_immune(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value):
    """Return unchanged state for the currently represented immunity branch."""
    return key, emit(state, EventKind.IMMUNE, 1-attacker_idx)


def move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Dispatch an implemented move after an accuracy check, or do nothing."""
    move = state.active[attacker_index].moves[move_index]
    return jax.lax.cond(move.implemented[0], _move_hits, move_misses,
                        key, state, attacker_index, move_index)


def _move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Join normal and forced routing before one damage/status executor.

    Ability and type immunities, status moves, and damaging moves share one
    branch signature so ``lax.switch`` can trace each possible outcome. Forced
    switches commit afterward, before pivot bookkeeping and attempt finalization.
    """

    state, move, branch, stat, boost_value = jax.lax.cond(state.event_move == -1,
        _prepare_struggle_hit, _prepare_normal_hit, state, attacker_index, move_index)
    # Only Whirlwind routed to the status branch can request a switch. Misses,
    # protection and immunity must never reach the switch preparation/entry path.
    forced = (branch == 1) & (move.identifier[0] == 18)
    key, state = jax.lax.switch(jnp.where(forced, 6, branch),
        [do_damaging_move, _status_move_effects, do_stat_boost_from_move,
         do_flash_fire_from_move, do_healing_from_move, is_immune, _defer_forced_switch],
        key, state, attacker_index, move, stat, boost_value)
    key, state, request = jax.lax.cond(forced, prepare_forced_switch, _no_forced_switch,
                                      key, state, attacker_index)
    state = execute_forced_switch(state, request)
    requested = move.switch_out[0] & (branch <= 1)
    return key, state.replace(switch_requested=state.switch_requested.at[attacker_index].set(requested))


def _match_move_leaf_dtype(value, ref):
    """Match each synthetic Struggle move leaf to the selected move dtype."""
    return value.astype(ref.dtype)


def _prepare_struggle_hit(state, player, index):
    """Supply typeless Struggle to the same damaging branch as ordinary hits."""
    reference = state.active[player].moves[index]
    move = Move(move_type=jnp.array([0]), max_pp=jnp.array([1]), current_pp=jnp.array([1]),
        type=jnp.array([Type.NONE]), base_power=jnp.array([50]), accuracy=jnp.array([1.]),
        priority=jnp.array([0]), offensive_stat=jnp.array([StatEnum.ATTACK]),
        defensive_stat=jnp.array([StatEnum.DEFENSE]), crit_stage=jnp.array([0]))
    move = jax.tree.map(_match_move_leaf_dtype, move, reference)
    ability = state.ability[1-player]
    return state, move, jnp.int32(0), jnp.zeros_like(ability), 1 + (ability == AbilityEnum.WELL_BAKED_BODY)


def _spiky_shield_retaliation(s, *, attacker_index):
    """Attribute Spiky Shield's indirect damage to its defending user."""
    source = s.event_source
    s = s.replace(event_source=jnp.asarray(1-attacker_index, jnp.int32))
    s = take_damage_percent(s, attacker_index, 1/8, cause=Cause.CONTACT)
    return s.replace(event_source=source)


def _prepare_normal_hit(state, attacker_index, move_index):
    """Resolve normal hit routing before the common damage/status executor."""
    # decide branch to execute based on ability immunities
    defender = state.effective_active[1 - attacker_index]
    move = effective_move(state, attacker_index, state.active[attacker_index].moves[move_index])
    v = state.volatile_status
    move = move.replace(priority=jnp.where(v.encore_overridden[attacker_index], v.queued_priority[attacker_index], move.priority))
    ability = defender.ability
    is_flash_fire = jnp.logical_and(ability==AbilityEnum.FLASH_FIRE, move.type==Type.FIRE)
    is_spa_boost = jnp.logical_or(
        jnp.logical_and(ability==AbilityEnum.STORM_DRAIN, move.type==Type.WATER),
        jnp.logical_and(ability==AbilityEnum.LIGHTNING_ROD, move.type==Type.ELECTRIC))

    is_attack_boost = jnp.logical_and(ability == AbilityEnum.SAP_SIPPER, move.type == Type.GRASS)

    # Whirlwind retains its existing forced-switch failure classification.
    wind_absorbed = (ability == AbilityEnum.WIND_RIDER) & ((move.move_flags & MoveFlags.WIND) != 0) & (move.identifier != 18)
    is_attack_boost |= wind_absorbed
    is_speed_boost = jnp.logical_and(ability==AbilityEnum.MOTOR_DRIVE, move.type==Type.ELECTRIC)
    is_def_boost = jnp.logical_and(ability==AbilityEnum.WELL_BAKED_BODY, move.type==Type.FIRE)
    is_heal = quad_or(
        jnp.logical_and(ability == AbilityEnum.WATER_ABSORB, move.type == Type.WATER),
        jnp.logical_and(ability == AbilityEnum.VOLT_ABSORB, move.type == Type.ELECTRIC),
        jnp.logical_and(ability == AbilityEnum.EARTH_EATER, move.type == Type.GROUND),
        jnp.logical_and(ability == AbilityEnum.DRY_SKIN, move.type == Type.WATER)
    )
    # all of this is hacky and inelegant and probably could be simplified
    is_stat_boost = quad_or(is_speed_boost, is_attack_boost, is_spa_boost, is_def_boost)
    not_bypass_move = triple_or(is_flash_fire, is_stat_boost, is_heal)
    immune = state.is_immune_to_move(1-attacker_index, move)
    immune = jnp.logical_and(immune, ~not_bypass_move)
    is_status = jnp.logical_and(move.move_type == MoveType.STATUS, ~not_bypass_move & ~immune)
    # Encode the selected outcome for lax.switch. Self-targeted status moves
    # explicitly select the status branch instead of defender immunity branches.
    branch_index = (
            is_status +
            is_stat_boost * 2 +
            is_flash_fire * 3 +
            is_heal * 4 +
            immune * 5
    )
    branch_index = jnp.where(move.target_self & (move.move_type == MoveType.STATUS), 1, branch_index)
    # Psychic Terrain blocks opposing priority moves before absorption abilities
    # or other hit effects can activate, but only for grounded targets.
    grounded = state.is_grounded(1-attacker_index)[0]
    priority_blocked = (((state.terrain.terrain == TerrainEnum.PSYCHIC) & grounded) |
        (ability[0] == AbilityEnum.QUEENLY_MAJESTY)) & (move.priority[0] > 0) & ~move.target_self[0]
    prankster_dark = ((state.ability[attacker_index, 0] == AbilityEnum.PRANKSTER) &
        (move.move_type[0] == MoveType.STATUS) & jnp.any(defender.type_list == Type.DARK) &
        ~move.target_self[0] & (move.hazard_effect[0] == 0))
    priority_blocked |= prankster_dark
    branch_index = jnp.where(priority_blocked, 5, branch_index)
    # Protect blocks opposing targeted moves, but not field-wide hazards.
    bypass = (state.ability[attacker_index, 0] == AbilityEnum.UNSEEN_FIST) & move.contact[0]
    blocked = state.volatile_status.protected[1-attacker_index, 0] & ~move.target_self[0] & move.protectable[0] & ~bypass
    branch_index = jnp.where(blocked, 5, branch_index)
    # Spiky Shield damages a living contact attacker only when it actually
    # blocks; Long Reach and Magic Guard prevent this retaliation.
    spiky = blocked & (state.volatile_status.protection_kind[1-attacker_index, 0] == ProtectionKind.SPIKY_SHIELD)

    spiky &= move.contact[0] & contact_reactions_allowed(state.ability[attacker_index, 0], state.active.item[attacker_index, 0])
    spiky &= (state.ability[attacker_index, 0] != AbilityEnum.MAGIC_GUARD) & (state.active.current_hp[attacker_index, 0] > 0)
    state = jax.lax.cond(spiky, partial(_spiky_shield_retaliation, attacker_index=attacker_index), _identity, state)
    # Side and field moves do not target the Pokémon behind its Substitute.
    sub_blocked = substitute_intercepts(state.volatile_status.substitute_hp[1-(attacker_index), 0], state.ability[attacker_index, 0],
        move.target_self[0], move.sound[0], move.bypass_substitute[0]) & (move.move_type[0] == MoveType.STATUS) & (move.hazard_effect[0] == 0)
    branch_index = jnp.where(sub_blocked, 5, branch_index)
    failed_endeavor = (move.identifier[0] == 283) & (defender.current_hp[0] <= state.active.current_hp[attacker_index, 0])
    failed_poltergeist = (move.identifier[0] == 809) & (defender.item[0] == 0)
    branch_index = jnp.where(failed_endeavor | failed_poltergeist, 5, branch_index)

    revealed = (move.identifier[0] == 809) & (branch_index[0] == 0)
    state = remember_item(state, 1-attacker_index, defender.item[0], revealed)
    state = emit(state, EventKind.ITEM_REVEALED, 1-attacker_index, detail=defender.item[0], enabled=revealed)
    target = jnp.where(move.target_self[0], attacker_index, 1-attacker_index)
    mon = state.active[attacker_index]
    failed_substitute = (move.identifier[0] == 164) & (
        (state.volatile_status.substitute_hp[attacker_index, 0] > 0) |
        (mon.max_hp <= 1) | (4 * mon.current_hp[0] <= mon.max_hp))
    outcome = jnp.where(blocked, HitOutcome.PROTECTED, jnp.where(sub_blocked | priority_blocked | failed_substitute, HitOutcome.BLOCKED,
                        jnp.where(branch_index[0] >= 2, HitOutcome.IMMUNE, HitOutcome.HIT)))
    recipient = jnp.where(outcome == HitOutcome.HIT, HitRecipient.POKEMON, HitRecipient.NONE)
    state = record_hit(state, 0, target, outcome, recipient)
    boost_value = 1 + (ability==AbilityEnum.WELL_BAKED_BODY)

    return state, move, branch_index[0], is_attack_boost * StatEnum.ATTACK + is_def_boost * StatEnum.DEFENSE + is_spa_boost * StatEnum.SPECIAL_ATTACK + is_speed_boost * StatEnum.SPEED, boost_value


def move_misses(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Return unchanged key and state for a miss or unsupported move."""
    move = state.active[attacker_index].moves[move_index]
    target = jnp.where(move.target_self[0], attacker_index, 1-attacker_index)
    state = record_hit(state, 0, target, HitOutcome.MISS)
    return key, emit(state, EventKind.MISS, target, detail=move.implemented[0])

