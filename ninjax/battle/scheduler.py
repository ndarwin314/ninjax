"""Queue simultaneous actions and pause or resume replacement decisions."""
from functools import partial

import jax.lax
import jax.numpy as jnp
from jax import lax, random

from ninjax.battle.actions import step_action, step_switch
from ninjax.battle.field import step_field
from ninjax.battle.ordering import action_order
from ninjax.battle.types import ActionType, Phase
from ninjax.enums.moves import MoveType
from ninjax.moves.properties import effective_priority
from ninjax.records.events import Cause, capture_conditions_from_values, condition_values
from ninjax.side.item_effects import refresh_paradox


def _effective_priority(p, i, *, battle):
    """Read the effective move priority for one queued player."""
    return effective_priority(battle, p, battle.active_moves(p, i))[0]


def _queue_actions(current, actions):
    """Order the players' queued choices and preserve their actor slots."""
    key, order_key = random.split(current.key)
    v = current.battle.volatile_status
    continuing = (actions.move_type == ActionType.CONTINUED) & (v.charging_move_id[:, 0] != 0)
    charge_indices = jnp.argmax(current.battle.active_move_field('identifier')[..., 0] == v.charging_move_id, axis=1)
    indices = jnp.where(continuing, charge_indices, jnp.clip(actions.index, 0, 3))

    priorities = jax.vmap(partial(_effective_priority, battle=current.battle))(jnp.arange(2), indices)
    priorities = jnp.where(actions.move_type == ActionType.SWITCH, 7,
        jnp.where(((actions.move_type == ActionType.ATTACK) & (actions.index < 4)) | continuing, priorities, 0))
    first, second = action_order(order_key, current.battle, priorities)
    v = current.battle.volatile_status
    battle = current.battle.replace(volatile_status=v.replace(
        queued_priority=priorities[:, None], queued_priority_valid=(((actions.move_type == ActionType.ATTACK) & (actions.index < 4)) | continuing)[:, None]))
    return current.replace(key=key, battle=battle, pending_actions=actions, queued_actor_slot=current.battle.active_index[:, 0],
                           order=jnp.stack((first, second)), cursor=jnp.int32(0), field_pending=jnp.bool_(True),
                           acting_player_mask=jnp.zeros(2, bool))


def _dispatch_queued_action(k, b, *, chosen, player):
    """Execute the bound action for its original surviving actor."""
    return step_action(k, b, chosen, player)


def _keep_key_state(k, b):
    """Preserve battle state and PRNG key when the actor does not execute."""
    return (k, b)


def _execute_queued_action(current):
    """Execute one actor, reveal spent/called moves and record any pivot request."""
    player = current.order[current.cursor]
    chosen = current.pending_actions[player]
    other = 1-player
    other_action = current.pending_actions.move_type[other]
    pending = (current.cursor == 0) & ((other_action == ActionType.ATTACK) | ((other_action == ActionType.CONTINUED) &
        (current.battle.volatile_status.charging_move_id[other, 0] != 0))) & (current.battle.active_field('current_hp', other)[0] > 0) & (current.queued_actor_slot[other] == current.battle.active_index[other, 0])
    volatile = current.battle.volatile_status
    other_index = current.pending_actions.index[other]
    charged_index = jnp.argmax(current.battle.active_move_field('identifier', other)[:, 0] == volatile.charging_move_id[other, 0])
    other_index = jnp.where(other_action == ActionType.CONTINUED, charged_index, jnp.clip(other_index, 0, 3))
    category = current.battle.active_move_field('move_type', other)[other_index, 0]
    category = jnp.where((other_action == ActionType.ATTACK) & (current.pending_actions.index[other] == 4), MoveType.PHYSICAL, category)
    battle_with_context = current.battle.replace(volatile_status=volatile.replace(
        pending_move=volatile.pending_move.at[:, 0].set(False).at[other, 0].set(pending),
        pending_move_type=volatile.pending_move_type.at[other, 0].set(category),
        pending_actor_slot=volatile.pending_actor_slot.at[other, 0].set(current.queued_actor_slot[other])))
    can_act = (current.battle.active_field('current_hp', player)[0] > 0) & (current.queued_actor_slot[player] == current.battle.active_index[player, 0])
    key, battle = lax.cond(can_act,
        partial(_dispatch_queued_action, chosen=chosen, player=player),
        _keep_key_state, current.key, battle_with_context)

    battle = refresh_paradox(battle)
    battle = capture_conditions_from_values(condition_values(current.battle), battle)
    slot = current.battle.active_index[player, 0]
    spent = battle.team.moves.current_pp[player, slot, :, 0] < current.battle.team.moves.current_pp[player, slot, :, 0]
    called_id = battle.volatile_status.called_move_id[player, 0]
    called = (called_id != 0) & (battle.team.moves.identifier[player, slot, :, 0] == called_id)
    known = current.revealed_moves.at[player, slot].set(current.revealed_moves[player, slot] | spent | called)
    has_bench = jnp.any(current.selected[player] & (battle.team.current_hp[player, :, 0] > 0) &
                       (jnp.arange(6) != battle.active_index[player, 0]))
    pivot = battle.switch_requested[player] & (battle.active_field('current_hp', player)[0] > 0) & has_bench
    battle = battle.replace(switch_requested=jnp.zeros(2, bool))
    requested = jnp.zeros(2, bool).at[player].set(pivot)
    return current.replace(key=key, battle=battle, revealed_moves=known, cursor=current.cursor + 1,
        phase=jnp.where(pivot, Phase.REPLACEMENT, Phase.TURN), acting_player_mask=requested)


def _keep_episode(c):
    """Preserve the episode when a conditional scheduler branch is skipped."""
    return c


def _resolve_queued_action(_, current):
    """Resolve queued action."""
    execute = _execute_queued_action
    return lax.cond((current.cursor < 2) & ~jnp.any(current.acting_player_mask), execute,
                    _keep_episode, current)


def _finish_turn(current):
    """Apply field effects once after the queued actions finish."""
    before = current.battle.replace(event_source=jnp.int32(-1), event_move=jnp.int32(0), event_cause=jnp.int32(Cause.FIELD))
    key, battle = step_field(current.key, before)
    battle = capture_conditions_from_values(condition_values(before), battle)
    return current.replace(key=key, battle=battle, turn=current.turn + 1, field_pending=jnp.bool_(False))


def _execute_replacement(key, battle, player, index):
    """Switch the requested active slot during a replacement phase."""
    return step_switch(key, battle, player, index, False)


def _skip_replacement(key, battle, player, index):
    """Preserve state and key for a player not awaiting replacement."""
    return key, battle


def replace_player(player, carry):
    """Preserve player order, switch keys and condition snapshots."""
    current, actions = carry
    key, battle = lax.cond(current.acting_player_mask[player],
                           _execute_replacement, _skip_replacement,
                           current.key, current.battle, player, actions.index[player])

    battle = refresh_paradox(battle)
    battle = capture_conditions_from_values(condition_values(current.battle), battle)
    return current.replace(key=key, battle=battle), actions


def replace_active(current, actions):
    """Apply replacements sequentially through one compiled switch body."""

    # A Python loop expands swap_out and its entry hooks for each side.
    current, _ = lax.fori_loop(0, 2, replace_player, (current, actions))
    return current.replace(acting_player_mask=jnp.zeros(2, bool))


def advance_turn(state, actions):
    """Execute queued simultaneous choices and pause for replacement decisions."""
    select = _queue_actions


    was_turn = state.phase == Phase.TURN
    state = lax.cond(was_turn, select, replace_active, state, actions)
    # A completed turn awaiting fainted replacements must not run field effects again.
    finished_replacement = ~state.field_pending
    state = state.replace(phase=jnp.int32(Phase.TURN))

    resolve = _resolve_queued_action

    state = lax.fori_loop(0, 2, resolve, state)
    finish = _finish_turn
    state = lax.cond((state.cursor == 2) & ~jnp.any(state.acting_player_mask) & ~finished_replacement,
                     finish, _keep_episode, state)
    needs = (state.battle.active_field('current_hp')[:, 0] <= 0) & jnp.any((state.battle.team.current_hp[..., 0] > 0) & state.selected, axis=1)
    waiting = jnp.any(state.acting_player_mask)
    needs = jnp.where(waiting, state.acting_player_mask, needs)
    state = state.replace(phase=jnp.where(jnp.any(needs), Phase.REPLACEMENT, Phase.TURN),
                          acting_player_mask=jnp.where(jnp.any(needs), needs, jnp.ones(2, bool)))
    revealed = state.revealed.at[jnp.arange(2), state.battle.active_index[:, 0]].set(True)
    return state.replace(revealed=revealed)
