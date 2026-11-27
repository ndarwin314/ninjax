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
from ninjax.battle.switching import force_switch_target, forced_switch_move, reflected_switching_source
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import TerrainEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveFlags, MoveType, ProtectionKind, SemiInvulnerability
from ninjax.enums.pokemon import StatEnum, Type, Status
from ninjax.knowledge import remember_item
from ninjax.item_rules import fling_power, item_is_suppressed
from ninjax.moves.model import Move
from ninjax.moves.multi_hit import do_damaging_move
from ninjax.moves.properties import (
    absorption_from_fields, effective_move, is_poison_type_user, move_accuracy,
    resolve_move_category, substitute_intercepts,
)
from ninjax.moves.protection import blocked_by_protection, breaks_protection, remove_protection
from ninjax.moves.status import _status_move_effects, create_shed_tail_substitute
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.faint_effects import resolve_faints
from ninjax.side.field_conditions import clear_target_screens
from ninjax.side.hp_effects import take_damage_percent
from ninjax.side.item_effects import held_item_active, release_fling_item
from ninjax.side.queries import ability_against_move, can_remove_item, contact_reactions_allowed, is_floating_fields, move_immune_from_fields
from ninjax.side.stat_effects import stat_changes
from ninjax.side.state import BattleState
from ninjax.side.status_effects import set_status
from ninjax.side.updates import add_boosts


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)


def _unchanged_selected_move(k, s, p, move, absorption):
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
    volatile = state.volatile_status.replace(ability_state=state.volatile_status.ability_state.replace(flash_fire=state.volatile_status.ability_state.flash_fire.at[1-attacker_idx].set(True)))
    return key, state.replace(volatile_status=volatile)


def _apply_move_effects(key, state, attacker_index, move_index):
    """Skip Struggle accuracy while sharing the downstream hit executor."""
    key, move = prepare_hit_move(key, state, attacker_index, state.active_moves(attacker_index, move_index))
    absorption = absorption_from_fields(move, ability_against_move(state, 1-attacker_index, move))
    # Forced Struggle already uses -1 as its action-local public move marker.
    # Accuracy can write events but does not change the resolved move.
    key, state, ready = jax.lax.cond(
        state.event_context.move_id == -1,
        _ready_move,
        partial(_prepare_accuracy, attacker_index=attacker_index, move=move, absorption=absorption),
        key,
        state,
    )
    key, state = jax.lax.cond(ready, _move_hits, _unchanged_selected_move,
                            key, state, attacker_index, move, absorption)
    return key, release_fling_item(state, attacker_index, move.flung_item[0])


def move_not_drawn_in(key: chex.PRNGKey, state: BattleState, attacker_index, move_index: int, unused: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Sample accuracy and dispatch a hit or miss, returning the advanced key.

    The calculation includes represented accuracy stages and abilities. The
    final ``unused`` operand is retained for compatibility with branch calls.
    """
    return _apply_move_effects(key, state, attacker_index, move_index)


def _block_accuracy_target(k, s, *, attacker_index):
    """Stop a forbidden ability change before ordinary hit resolution."""
    return k, record_hit(emit(s, EventKind.IMMUNE, 1-attacker_index), 0, 1-attacker_index, HitOutcome.IMMUNE), jnp.bool_(False)


def _attempt_accuracy(k, s, *, hit, attacker_index, move):
    """Retain the accuracy draw and record a miss when the hit is gated out."""
    k, s = jax.lax.cond(
        hit,
        _unchanged_key_state,
        partial(_record_move_miss, attacker_index=attacker_index, move=move),
        k,
        s,
    )
    return k, s, hit


def _fling_failed(move):
    """Check the item prerequisite after projecting its transient metadata."""
    return (move.identifier[0] == 374) & (move.flung_item[0] == ItemEnum.NONE)


def _failed_fling(k, state, *, attacker_index):
    """Record a Fling without a usable item as a failed move."""
    return k, record_hit(state, 0, 1 - attacker_index, HitOutcome.FAILED), jnp.bool_(False)


def semi_invulnerable_to_move(state, attacker, move):
    """Check whether a target's charge location avoids this opposing move.

    No Guard on either active Pokémon bypasses this check. The listed moves
    are the Gen 9 exceptions for each represented location.
    """
    target = 1 - attacker
    location = state.volatile_status.continuation.semi_invulnerability[target, 0]
    ident = move.identifier[0]
    airborne_hit = jnp.isin(ident, jnp.array([16, 87, 239, 327, 479, 542, 614]))
    underground_hit = (ident == 89) | (ident == 222)
    underwater_hit = (ident == 57) | (ident == 250)
    avoided = ((location == SemiInvulnerability.AIRBORNE) & ~airborne_hit) | (
        (location == SemiInvulnerability.UNDERGROUND) & ~underground_hit) | (
        (location == SemiInvulnerability.UNDERWATER) & ~underwater_hit) | (
        location == SemiInvulnerability.VANISHED)
    no_guard = (state.ability[attacker, 0] == AbilityEnum.NO_GUARD) | (state.ability[target, 0] == AbilityEnum.NO_GUARD)
    toxic_bypass = (ident == 92) & is_poison_type_user(state, attacker)
    return avoided & ~no_guard & ~toxic_bypass & ~move.target_self[0]


def _prepare_accuracy(key, state, attacker_index, move, absorption=None):
    """Record misses/immunity and return whether normal hit routing is allowed."""
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    accuracy = move_accuracy(state, attacker_index, move, absorption)

    # Worry Seed fails against these identities before accuracy, including
    # while an ability's effects are suppressed.

    old = underlying_ability(state.active_field('ability', 1-attacker_index)[0], state.volatile_status.ability_state.ability_override[1-attacker_index, 0])
    cannot_target = (move.identifier[0] == 388) & ((old == AbilityEnum.INSOMNIA) | (old == AbilityEnum.TRUANT))

    # Gen 9 protection resolves before accuracy; an inaccurate contact move
    # still strikes Spiky Shield, and Protect blocks before a miss is rolled.
    protected = _blocked_helper(state, move, attacker_index)

    hit = (jnp.less_equal(r, accuracy)[0] | protected) & move.implemented[0] & ~semi_invulnerable_to_move(state, attacker_index, move)
    branch = jnp.select((_fling_failed(move), cannot_target), (0, 1), default=2)
    return jax.lax.switch(branch, (
        partial(_failed_fling, attacker_index=attacker_index),
        partial(_block_accuracy_target, attacker_index=attacker_index),
        partial(_attempt_accuracy, hit=hit, attacker_index=attacker_index, move=move)), key, state)


def is_immune(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value):
    """Return unchanged state for the currently represented immunity branch."""
    return key, emit(state, EventKind.IMMUNE, 1-attacker_idx)


def move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Dispatch an accuracy-checked move and resolve its pending faints."""
    move = state.active_moves(attacker_index, move_index)
    key, state = jax.lax.cond(move.implemented[0], _resolve_and_hit, _record_move_miss,
                              key, state, attacker_index, move)
    return key, resolve_faints(state)


def _resolve_and_hit(key, state, attacker_index, move):
    """Resolve a selected move for callers that already checked accuracy."""
    key, move = prepare_hit_move(key, state, attacker_index, move)
    absorption = absorption_from_fields(move, ability_against_move(state, 1-attacker_index, move))
    branch = jnp.select((_fling_failed(move), semi_invulnerable_to_move(state, attacker_index, move)),
                        (0, 1), default=2)
    key, state = jax.lax.switch(branch, (_reject_resolved_fling, _record_move_miss,
        partial(_move_hits, absorption=absorption)), key, state, attacker_index, move)
    return key, release_fling_item(state, attacker_index, move.flung_item[0])


def _reject_resolved_fling(key, state, player, move):
    """Record a failed Fling through the accuracy-free entry point."""
    return key, record_hit(state, 0, 1 - player, HitOutcome.FAILED)


def _resolve_hit_category(key, state, player, move):
    """Gather ordinary staged stats before the move is checked for a hit."""
    stats = state.raw_boosted_stats.astype(jnp.int32)
    user, target = stats[player], stats[1-player]
    return resolve_move_category(key, move, state.active_field('level', player)[0],
        user[StatEnum.ATTACK], user[StatEnum.SPECIAL_ATTACK],
        target[StatEnum.DEFENSE], target[StatEnum.SPECIAL_DEFENSE])


def _keep_hit_category(key, state, player, move):
    """Preserve the move and key for fixed-category attacks."""
    return key, move


def prepare_hit_move(key, state, player, move):
    """Prepare a transient move once for accuracy, hit routing and damage."""
    move = jax.lax.cond(state.event_context.move_id == -1, _struggle_move, effective_move, state, player, move)
    identifier = move.identifier[0]
    move = jax.lax.cond(identifier == 374, _prepare_fling, _keep_fling, state, player, move)
    return jax.lax.cond(move.implemented[0] & ((identifier == 722) | (identifier == 801)),
        _resolve_hit_category, _keep_hit_category, key, state, player, move)


def _keep_fling(state, player, move):
    """Preserve moves with no thrown item."""
    return move


def _prepare_fling(state, player, move):
    """Snapshot the thrown item, intrinsic power and eligible secondary effects."""
    item = state.active_field('item', player)[0]
    power = fling_power(item)
    valid = (power > 0) & held_item_active(state, player) & can_remove_item(state, player, player)
    status = jnp.select((item == ItemEnum.FLAME_ORB, item == ItemEnum.TOXIC_ORB, item == ItemEnum.LIGHT_BALL),
                        (Status.BURN, Status.TOXIC, Status.PARALYZE), default=Status.NONE)
    flinch = item == ItemEnum.KINGS_ROCK
    effects = move.secondaries.replace(status=move.secondaries.status.at[0].set(status),
        flinch=move.secondaries.flinch.at[0].set(flinch),
        probabilities=move.secondaries.probabilities.at[0].set(
            jnp.asarray((status != Status.NONE) | flinch, move.secondaries.probabilities.dtype)))
    return move.replace(base_power=jnp.reshape(power, (1,)), secondaries=effects,
                        flung_item=jnp.reshape(jnp.where(valid, item, ItemEnum.NONE), (1,)))


def apply_pre_damage_move_effects(state, player, move, damaging_hit):
    """Apply move-specific TryHit effects after gates and before Substitute."""
    identifier = move.identifier[0]
    breaks_screens = (identifier == 280) | (identifier == 873)
    return clear_target_screens(state, 1 - player, damaging_hit & breaks_screens)


def _apply_shed_tail(state, *, player):
    """Create the Substitute only after Shed Tail's eligibility checks."""
    return create_shed_tail_substitute(state, player)


def _move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move: Move, absorption) -> Tuple[chex.PRNGKey, BattleState]:
    """Join normal and forced routing before one damage/status executor.

    Ability and type immunities, status moves, and damaging moves share one
    branch signature so ``lax.switch`` can trace each possible outcome. Forced
    switches commit afterward, before pivot bookkeeping and attempt finalization.
    """

    key, state, move, branch, stat, boost_value = _prepare_normal_hit(
        key, state, attacker_index, move, absorption=absorption)
    state = apply_pre_damage_move_effects(state, attacker_index, move, branch == 0)
    # Only status drags routed to the status branch can request a switch. Misses,
    # protection and immunity must never reach the switch preparation/entry path.
    ident = move.identifier[0]
    forced = (branch == 1) & forced_switch_move(move)
    parting = (branch == 1) & (ident == 575)
    key, state = jax.lax.switch(jnp.where(forced, 6, branch),
        [do_damaging_move, _status_move_effects, do_stat_boost_from_move,
         do_flash_fire_from_move, do_healing_from_move, is_immune, _defer_forced_switch],
        key, state, attacker_index, move, stat, boost_value)
    target = 1 - attacker_index
    hit = state.hit_results[0]
    damaging = ((ident == 525) | (ident == 509)) & (branch == 0) & (hit.outcome == HitOutcome.HIT)
    drag_hit = (damaging & (hit.recipient == HitRecipient.POKEMON) &
                (state.active_field('current_hp', target)[0] > 0) &
                (state.active_field('current_hp', attacker_index)[0] > 0))
    drag_source = jnp.where(forced, reflected_switching_source(state, attacker_index, move), attacker_index)
    key, state = force_switch_target(key, state, drag_source, wind=ident == 18,
                                    enabled=forced | drag_hit, damaging=damaging)
    bench = jnp.any(state.roster[attacker_index] & (state.team.current_hp[attacker_index, :, 0] > 0) &
                    (jnp.arange(6) != state.active_index[attacker_index, 0]))
    shed = ident == 880
    hp = state.active.current_hp[attacker_index, 0]
    max_hp = state.active.max_hp[attacker_index]
    shed_ok = bench & (state.volatile_status.substitute_hp[attacker_index, 0] == 0) & (hp > (max_hp + 1) // 2)
    state = jax.lax.cond(shed & (branch == 1) & shed_ok,
        partial(_apply_shed_tail, player=attacker_index), _identity, state)
    failed = (shed & ~shed_ok) | ((ident == 100) & ~bench) | ((ident == 226) & ~bench)
    state = jax.lax.cond((branch == 1) & failed,
        partial(_record_pivot_failure, player=attacker_index), _identity, state)
    requested = jnp.where(parting, state.switch_requested[attacker_index],
                          move.switch_out[0] & (branch <= 1) & ~failed & bench)
    transfer = jnp.where(ident == 226, 1, jnp.where(shed, 2, 0))
    state = state.replace(switch_requested=state.switch_requested.at[attacker_index].set(requested),
        switch_transfer=state.switch_transfer.at[attacker_index].set(jnp.where(requested, transfer, 0)))
    return key, state


def _record_pivot_failure(state, *, player):
    """Record a status pivot that cannot initiate its replacement."""
    return record_hit(state, 0, player, HitOutcome.FAILED)


def _match_move_leaf_dtype(value, ref):
    """Match each synthetic Struggle move leaf to the selected move dtype."""
    return value.astype(ref.dtype)


def _struggle_move(state, player, reference):
    """Build typeless Struggle with the same leaf dtypes as a stored move."""
    move = Move(move_type=jnp.array([0]), max_pp=jnp.array([1]), current_pp=jnp.array([1]),
        type=jnp.array([Type.NONE]), base_power=jnp.array([50]), accuracy=jnp.array([1.]),
        priority=jnp.array([0]), offensive_stat=jnp.array([StatEnum.ATTACK]),
        defensive_stat=jnp.array([StatEnum.DEFENSE]), crit_stage=jnp.array([0]),
        move_flags=jnp.array([int(MoveFlags.CONTACT)]))
    return jax.tree.map(_match_move_leaf_dtype, move, reference)


def _spiky_shield_retaliation(k, s, *, attacker_index):
    """Attribute Spiky Shield's indirect damage to its defending user."""
    source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(1-attacker_index, jnp.int32)))
    s = take_damage_percent(s, attacker_index, 1/8, cause=Cause.CONTACT)
    return k, s.replace(event_context=s.event_context.replace(source=source))


def _baneful_bunker_poison(k, s, *, attacker_index):
    """Attribute Baneful Bunker's Poison affliction to its defending user."""
    source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(1 - attacker_index, jnp.int32)))
    k, s = set_status(k, s, attacker_index, Status.POISON)
    return k, s.replace(event_context=s.event_context.replace(source=source))


def _burning_bulwark_burn(k, s, *, attacker_index):
    """Attribute Burning Bulwark's Burn affliction to its defending user."""
    source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(1 - attacker_index, jnp.int32)))
    k, s = set_status(k, s, attacker_index, Status.BURN)
    return k, s.replace(event_context=s.event_context.replace(source=source))


def _silk_trap_drop(k, s, *, attacker_index):
    """Attribute Silk Trap's speed drop to its defending user."""
    source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(1 - attacker_index, jnp.int32)))
    s = stat_changes(s, attacker_index, jnp.array([StatEnum.SPEED]), jnp.array([-1]), True)
    return k, s.replace(event_context=s.event_context.replace(source=source))


def _kings_shield_drop(k, s, *, attacker_index):
    """Attribute King's Shield's Attack drop to its defending user."""
    source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(1 - attacker_index, jnp.int32)))
    s = stat_changes(s, attacker_index, jnp.array([StatEnum.ATTACK]), jnp.array([-1]), True)
    return k, s.replace(event_context=s.event_context.replace(source=source))


def _blocked_helper(state, move: Move, attacker_index):
    """Share scalar protection eligibility between accuracy and hit routing."""
    return blocked_by_protection(state, move, attacker_index)

def _prepare_normal_hit(key, state, attacker_index, move, *, absorption=None):
    """Resolve normal hit routing before the common damage/status executor."""
    # decide branch to execute based on ability immunities
    defender_idx = 1 - attacker_index
    ability = ability_against_move(state, defender_idx, move)
    defender_types = state.effective_types(defender_idx, include_added=True)
    defender_item = state.active_field('item', defender_idx)
    defender_hp = state.active_field('current_hp', defender_idx)[0]
    gravity = state.gravity_duration > 0
    v = state.volatile_status
    move = move.replace(priority=jnp.where(v.queued_move.encore_overridden[attacker_index], v.queued_move.queued_priority[attacker_index], move.priority))
    if absorption is None:
        absorption = absorption_from_fields(move, ability)
    immune = move_immune_from_fields(defender_types, ability, defender_item, move, gravity)
    # Opposing absorption abilities intercept before ordinary type immunity.
    branch_index = jnp.select(
        (absorption.stat != 0, absorption.flash_fire, absorption.heal, immune,
         move.move_type == MoveType.STATUS), (2, 3, 4, 5, 1), default=0)
    branch_index = jnp.where(move.target_self & (move.move_type == MoveType.STATUS), 1, branch_index)
    # Psychic Terrain blocks opposing priority moves before absorption abilities
    # or other hit effects can activate, but only for grounded targets.
    grounded = (~is_floating_fields(defender_types, ability, defender_item) | gravity)[0]
    priority_blocked = (((state.terrain.terrain == TerrainEnum.PSYCHIC) & grounded) |
        (ability[0] == AbilityEnum.QUEENLY_MAJESTY)) & (move.priority[0] > 0) & ~move.target_self[0]
    prankster_dark = ((state.ability[attacker_index, 0] == AbilityEnum.PRANKSTER) &
        (move.move_type[0] == MoveType.STATUS) & jnp.any(defender_types == Type.DARK) &
        ~move.target_self[0] & (move.hazard_effect[0] == 0))
    priority_blocked |= prankster_dark
    branch_index = jnp.where(priority_blocked, 5, branch_index)
    # Protect blocks opposing targeted moves, but not field-wide hazards.

    blocked = _blocked_helper(state, move, attacker_index) & ~priority_blocked
    branch_index = jnp.where(blocked, 5, branch_index)

    protection_kind = state.volatile_status.protection.protection_kind[1-attacker_index, 0]
    attacker_ability = state.ability[attacker_index, 0]
    item = state.active_field('item', attacker_index)[0]
    item = jnp.where(item_is_suppressed(attacker_ability, state.magic_room_duration,
        v.embargo_turns[attacker_index, 0]), ItemEnum.NONE, item)
    takes_indirect_damage = attacker_ability != AbilityEnum.MAGIC_GUARD
    attacker_alive = (state.active_field('current_hp', attacker_index)[0] > 0)
    blocked_contact = (blocked & move.contact[0] &
                       contact_reactions_allowed(attacker_ability, item) &
                       attacker_alive)

    spiky = blocked_contact & (protection_kind == ProtectionKind.SPIKY_SHIELD) & takes_indirect_damage
    baneful = blocked_contact & (protection_kind == ProtectionKind.BANEFUL_BUNKER)
    bulwark = blocked_contact & (protection_kind == ProtectionKind.BURNING_BULWARK)
    silk = blocked_contact & (protection_kind == ProtectionKind.SILK_TRAP)
    king = blocked_contact & (protection_kind == ProtectionKind.KINGS_SHIELD)

    branch = jnp.select((spiky, baneful, bulwark, silk, king), (1, 2, 3, 4, 5), default=0)

    key, state = jax.lax.switch(
        branch,
        (
            _unchanged_key_state,
            partial(_spiky_shield_retaliation, attacker_index=attacker_index),
            partial(_baneful_bunker_poison, attacker_index=attacker_index),
            partial(_burning_bulwark_burn, attacker_index=attacker_index),
            partial(_silk_trap_drop, attacker_index=attacker_index),
            partial(_kings_shield_drop, attacker_index=attacker_index),
        )
        , key, state)
    # Side and field moves do not target the Pokémon behind its Substitute.
    sub_blocked = substitute_intercepts(state.volatile_status.substitute_hp[1-(attacker_index), 0], state.ability[attacker_index, 0],
        move.target_self[0], move.sound[0], move.bypass_substitute[0]) & (move.move_type[0] == MoveType.STATUS) & (move.hazard_effect[0] == 0)
    branch_index = jnp.where(sub_blocked, 5, branch_index)
    failed_endeavor = (move.identifier[0] == 283) & (defender_hp <= state.active_field('current_hp', attacker_index)[0])
    failed_poltergeist = (move.identifier[0] == 809) & (defender_item[0] == 0)
    branch_index = jnp.where(failed_endeavor | failed_poltergeist, 5, branch_index)

    revealed = (move.identifier[0] == 809) & (branch_index[0] == 0)
    state = remember_item(state, 1-attacker_index, defender_item[0], revealed)
    state = emit(state, EventKind.ITEM_REVEALED, 1-attacker_index, detail=defender_item[0], enabled=revealed)
    target = jnp.where(move.target_self[0], attacker_index, 1-attacker_index)
    max_hp = state.active_stats(attacker_index)[0]
    hp = state.active_field('current_hp', attacker_index)[0]
    failed_substitute = (move.identifier[0] == 164) & (
        (state.volatile_status.substitute_hp[attacker_index, 0] > 0) |
        (max_hp <= 1) | (4 * hp <= max_hp))
    outcome = jnp.where(blocked, HitOutcome.PROTECTED, jnp.where(sub_blocked | priority_blocked | failed_substitute, HitOutcome.BLOCKED,
                        jnp.where(branch_index[0] >= 2, HitOutcome.IMMUNE, HitOutcome.HIT)))
    recipient = jnp.where(outcome == HitOutcome.HIT, HitRecipient.POKEMON, HitRecipient.NONE)
    state = record_hit(state, 0, target, outcome, recipient)
    # A breaker removes protection only after passing the hit gates, before
    # the damaging branch decides whether Substitute intercepts the hit.
    state = remove_protection(state, defender_idx, (branch_index[0] == 0) & breaks_protection(move))
    return key, state, move, branch_index[0], absorption.stat, absorption.stages


def move_misses(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Return unchanged key and state for a miss or unsupported move."""
    return _record_move_miss(key, state, attacker_index, state.active_moves(attacker_index, move_index))


def _record_move_miss(key, state, attacker_index, move):
    """Record a missed attempt from the already selected move's metadata."""
    target = jnp.where(move.target_self[0], attacker_index, 1-attacker_index)
    state = record_hit(state, 0, target, HitOutcome.MISS)
    return key, emit(state, EventKind.MISS, target, detail=move.implemented[0])

