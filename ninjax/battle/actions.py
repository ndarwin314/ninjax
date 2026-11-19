"""Prepare and execute one move, switch or continuation action."""
from functools import partial

import chex
import jax.lax
import jax.numpy as jnp
from jax import lax, random

from ninjax.battle.interruptions import before_move
from ninjax.battle.switching import swap_out
from ninjax.battle.types import ActionType, BattleAction, Transformation
from ninjax.enums.moves import MoveResult
from ninjax.moves.execution import move_interrupted, move_used
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitResults
from ninjax.restrictions import resolve_move
from ninjax.side.faint_effects import clear_destiny_bond, resolve_faints
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


def _prepare_selected_callback(k, s, p, i, *, is_tera):
    """Prepare the selected slot with its bound transformation request."""
    return _prepare_selected_action(k, s, p, i, is_tera)


def _prepare_selected_or_continued(k, s, p, i, *, continued, is_tera):
    """Choose new-selection bookkeeping or release an already charged move."""
    return lax.cond(continued,
                _prepare_charge_release,
                partial(_prepare_selected_callback, is_tera=is_tera),
                k, s, p, i)


def _execute_prepared_move(k, s, p, i, *, continued, struggle):
    """Execute the prepared selection, charge release or forced Struggle."""
    return move_used(k, s, p, i, continued, struggle)


def _execute_ordinary_callback(k, b, *, action, player_index):
    """Dispatch the bound ordinary action after transformation validation."""
    return _ordinary_action(k, b, action, player_index)


def _keep_key_state(k, b):
    """Preserve battle state and PRNG key when the actor does not execute."""
    return (k, b)


def _execute_switch_callback(k, s, *, player_index, action):
    """Execute the selected switch without a transformation."""
    return step_switch(k, s, player_index, action.index, False)


def _spend_recharge_callback(k, s, *, player_index):
    """Consume the bound player's forced recharge action."""
    return _spend_recharge(k, s, player_index)


def _execute_move_callback(k, s, *, player_index, action, release):
    """Attempt the selected move or release using the shared move pipeline."""
    return step_move(k, s, player_index, action.index, False, release)


def decode_action(action):
    # takes int in [0, 15) and returns a tuple of is_move, index, is_tera
    # index is a move index in [0,4) if action is a move, and in [0,6) if its a switch
    # if is_move == False then the third index should be ignored
    """Decode legacy move, switch, transformation and no-op action columns."""
    is_move_action = jnp.less(action, 8)
    move_index = action % 4
    is_tera = (action >= 4) & (action < 8)
    switch_index = action - 8
    index = move_index * is_move_action + switch_index * (1 - is_move_action)
    is_no_op = jnp.equal(action, 14)
    return is_move_action, index, is_tera, is_no_op


def step_move(
    key: chex.PRNGKey,
    state: BattleState,
    player_idx: int | jax.Array,
    index: int | jax.Array,
    is_tera: bool | jax.Array, continued=False) -> tuple[jax.Array, BattleState]:
    """Prepare a selection, charge release or Struggle for one move executor."""
    struggle = ~jnp.asarray(continued) & (index == 4)
    key, state, index, restriction_blocked = lax.cond(struggle,
        _prepare_struggle_action,
        partial(_prepare_selected_or_continued, continued=continued, is_tera=is_tera), key, state, player_idx, index)
    selected_move = state.active_moves(player_idx, index)
    key, state, is_interrupted = before_move(key, state, player_idx, inclusive_paralysis=~struggle, restriction_blocked=restriction_blocked,
        defrost_move=selected_move.defrost & ~struggle, sleep_usable=(selected_move.identifier[0] == 214) & ~jnp.asarray(continued) & ~struggle)
    key, state = jax.lax.cond(
        is_interrupted.reshape(()), move_interrupted,
        partial(_execute_prepared_move, continued=continued, struggle=struggle),
        key, state, player_idx, index)
    # Protect and Spiky Shield return NOT_FAIL in Gen 9; other blocks can fail.
    result = state.hit_results[0].outcome
    failed = is_interrupted | (result == HitOutcome.MISS) | (result == HitOutcome.IMMUNE) | (result == HitOutcome.FAILED) | (result == HitOutcome.BLOCKED)
    failed = jnp.where(struggle, is_interrupted, failed)
    # Sleep Talk succeeds once it selects a move, even if that called move fails.
    failed = jnp.where(~struggle & ~jnp.asarray(continued) & (selected_move.identifier[0] == 214) & state.volatile_status.sleep_talk_succeeded[player_idx, 0] & ~is_interrupted, False, failed)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(move_result=v.move_result.at[player_idx, 0].set(jnp.where(failed, MoveResult.FAILED, MoveResult.SUCCEEDED))))
    return key, resolve_faints(state)


def _prepare_struggle_action(key, state, player, unused_index):
    """Count a forced attempt without applying Encore or paying move-slot PP."""
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        active_move_actions=v.active_move_actions.at[player, 0].add(1)))
    return key, state, jnp.int32(0), jnp.bool_(False)


def _prepare_selected_action(key, state, player_idx, index, is_tera):
    """Resolve Encore, priority and transformation for a newly selected move."""

    original_index = index
    v = state.volatile_status
    index, restriction_blocked = resolve_move(
        state.active_move_field('identifier', player_idx)[:, 0], state.active_move_field('move_type', player_idx)[:, 0],
        state.active_move_field('current_pp', player_idx)[:, 0], index,
        disabled_move_id=v.disabled_move_id[player_idx, 0],
        disable_turns=v.disable_turns[player_idx, 0],
        taunt_turns=v.taunt_turns[player_idx, 0],
        encore_move_id=v.encore_move_id[player_idx, 0],
        encore_turns=v.encore_turns[player_idx, 0],
        gravity_duration=state.gravity_duration)
    v = v.replace(active_move_actions=v.active_move_actions.at[player_idx, 0].add(1))
    # Encore replaces an already ordered action without changing its priority.
    priority = jnp.where(v.queued_priority_valid[player_idx, 0], v.queued_priority[player_idx, 0],
                         state.active_move_field('priority', player_idx)[original_index, 0])
    state = state.replace(volatile_status=v.replace(
        queued_priority=v.queued_priority.at[player_idx, 0].set(priority),
        encore_overridden=v.encore_overridden.at[player_idx, 0].set(index != original_index)))
    state = update_active_fields(state, player_idx,
        is_terastallized=state.active_field('is_terastallized', player_idx) | jnp.bool([is_tera]))
    return key, state, jnp.asarray(index, jnp.int32), restriction_blocked


def _prepare_charge_release(key, state, player, unused_index):
    """Clear the stored charge and ignore Encore/PP while honoring Disable."""

    v = state.volatile_status
    identifier = v.charging_move_id[player, 0]
    matches = (state.active_move_field('identifier', player)[:, 0] == identifier) & (identifier != 0)
    index = jnp.argmax(matches).astype(jnp.int32)
    state = state.replace(hit_results=HitResults.empty(), event_move=identifier,
        volatile_status=v.replace(
            active_move_actions=v.active_move_actions.at[player, 0].add(1),
            charging_move_id=v.charging_move_id.at[player, 0].set(0),
            charge_target_side=v.charge_target_side.at[player, 0].set(0),
            charge_target_position=v.charge_target_position.at[player, 0].set(0),
            protect_chain=v.protect_chain.at[player, 0].set(0),
            encore_overridden=v.encore_overridden.at[player, 0].set(False)))
    state = emit(state, EventKind.CONDITION, player, detail=14, before=identifier, after=0)
    blocked = ~jnp.any(matches) | ((v.disable_turns[player, 0] > 0) & (v.disabled_move_id[player, 0] == identifier))
    return key, state, index, blocked


def step_switch(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int | jax.Array,
    index: int | jax.Array,
    is_tera: bool | jax.Array
) -> tuple[jax.Array, BattleState]:
    # switch needs to access the battle state because opponent switching triggers annoying things
    # TODO: add an opponent switched field somewhere for stakeout + analytic
    """Advance the action key and execute the selected switch with a separate key."""
    key, switch_key = random.split(key)
    return key, swap_out(switch_key, state, player_index, index)


def step_action(
    key: chex.PRNGKey,
    state: BattleState,
    action: int | jax.Array | BattleAction,
    player_index: int | jax.Array
) -> tuple[jax.Array, BattleState]:
    """Execute a selected move, switch, or forced Struggle for one player."""
    if not isinstance(action, BattleAction):
        action = BattleAction.from_legacy(action)
    state = state.replace(event_source=jnp.asarray(player_index, jnp.int32),
        event_cause=jnp.int32(Cause.MOVE), event_move=jnp.int32(0))
    key, state = lax.cond(action.transformation == Transformation.NONE,
        partial(_execute_ordinary_callback, action=action, player_index=player_index),
        _keep_key_state, key, state)
    return key, resolve_faints(state)


def _ordinary_action(key, state, action, player_index):
    """Dispatch an attack, charge release, switch, recharge or explicit no-op."""
    v = state.volatile_status
    continued = action.move_type == ActionType.CONTINUED
    release = continued & ~v.recharging[player_index, 0] & (v.charging_move_id[player_index, 0] != 0)
    # A charge release uses the move path; other continuations only recharge.
    branch = jnp.select(
        ((action.move_type == ActionType.ATTACK) | release,
         action.move_type == ActionType.SWITCH,
         continued & v.recharging[player_index, 0]),
        (0, 1, 2), default=3)
    return lax.switch(branch,
        (partial(_execute_move_callback, player_index=player_index, action=action, release=release),
         partial(_execute_switch_callback, player_index=player_index, action=action),
         partial(_spend_recharge_callback, player_index=player_index), _keep_key_state), key, state)


def step_continued(key, state, player):
    """Spend recharge or attempt the stored charge move without another PP cost."""
    action = BattleAction(jnp.int32(0), jnp.int32(ActionType.CONTINUED))
    return _ordinary_action(key, state, action, player)


def _spend_recharge(key, state, player):
    """Spend recharge before sleep, freeze, flinch, confusion or paralysis."""

    state = clear_destiny_bond(state, player)
    v = state.volatile_status
    state = state.replace(hit_results=HitResults.empty(), volatile_status=v.replace(
        active_move_actions=v.active_move_actions.at[player, 0].add(1),
        move_result=v.move_result.at[player, 0].set(MoveResult.SKIPPED),
        recharging=v.recharging.at[player, 0].set(False),
        protect_chain=v.protect_chain.at[player, 0].set(0)))
    return key, emit(state, EventKind.CONDITION, player, detail=15, before=1, after=0)
