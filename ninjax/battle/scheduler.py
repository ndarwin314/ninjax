"""Queue simultaneous actions and pause or resume replacement decisions."""
from functools import partial

import jax.lax
import jax.numpy as jnp
from jax import lax, random

from ninjax.ability_effects import refresh_trace_player, resume_ability
from ninjax.battle.actions import step_action, step_switch
from ninjax.battle.field import step_field
from ninjax.battle.ordering import action_order
from ninjax.battle.terastallization import tera_eligible, terastallize
from ninjax.battle.types import ActionType, Phase, Transformation
from ninjax.enums.moves import MoveResult, MoveType
from ninjax.moves.properties import effective_priority
from ninjax.records.events import Cause, EventKind, capture_conditions_from_values, condition_values, emit
from ninjax.records.hits import HitOutcome
from ninjax.enums.pokemon import Status
from ninjax.side.form_effects import reactive_form_update
from ninjax.side.item_effects import refresh_paradox


def _effective_priority(p, i, *, battle):
    """Read the effective move priority for one queued player."""
    return effective_priority(battle, p, battle.active_moves(p, i))[0]


def _continuation_id(volatile):
    """Get the move bound to a charge or repeat, excluding recharge."""
    c = volatile.continuation
    return jnp.where(c.locked.move_id != 0, c.locked.move_id, c.charging_move_id)


def _queue_actions(current, actions):
    """Order the players' queued choices and preserve their actor slots."""
    key, order_key = random.split(current.key)
    v = current.battle.volatile_status
    continuation_id = _continuation_id(v)
    continuing = (actions.move_type == ActionType.CONTINUED) & (continuation_id[:, 0] != 0)
    move_ids = current.battle.active_move_field('identifier')[..., 0]
    charge_indices = jnp.argmax(move_ids == continuation_id, axis=1)
    indices = jnp.where(continuing, charge_indices, jnp.clip(actions.index, 0, 3))

    priorities = jax.vmap(partial(_effective_priority, battle=current.battle))(jnp.arange(2), indices)
    priorities = jnp.where(actions.move_type == ActionType.SWITCH, 7,
        jnp.where(((actions.move_type == ActionType.ATTACK) & (actions.index < 4)) | continuing, priorities, 0))
    first, second = action_order(order_key, current.battle, priorities)
    attack = (actions.move_type == ActionType.ATTACK) & (actions.index < 4)
    chosen_ids = move_ids[jnp.arange(2), indices]
    prepared = jnp.where(attack & ((chosen_ids == 264) | (chosen_ids == 690)), chosen_ids, 0)
    counter = jnp.select((attack & (chosen_ids == 68), attack & (chosen_ids == 243)),
                         (MoveType.PHYSICAL, MoveType.SPECIAL), default=-1).astype(jnp.int32)
    battle = current.battle.replace(volatile_status=v.replace(
        queued_move=v.queued_move.replace(
            queued_priority=priorities[:, None],
            queued_priority_valid=(attack | continuing)[:, None],
            preparation=prepared[:, None], focus_lost=jnp.zeros((2, 1), bool))),
        damage_history=current.battle.damage_history.replace(counter_prepared=counter))
    for actor in (first, second):
        battle = emit(battle, EventKind.CONDITION, actor, detail=prepared[actor],
                      after=1, enabled=prepared[actor] != 0)
    return current.replace(key=key, battle=battle, pending_actions=actions, queued_actor_slot=current.battle.active_index[:, 0],
                           order=jnp.stack((first, second)), cursor=jnp.int32(0), field_pending=jnp.bool_(True),
                           acting_player_mask=jnp.zeros(2, bool))


def _dispatch_queued_action(k, b, *, chosen, player):
    """Execute the bound action for its original surviving actor."""
    # The environment validated the switch when both players committed.
    # A queued Tera has already resolved as its own action before the move.
    applied = (chosen.transformation == Transformation.TERA) & b.active_field('is_terastallized', player)[0]
    chosen = chosen.replace(transformation=jnp.where(applied, Transformation.NONE, chosen.transformation))
    return step_action(k, b, chosen, player, switch_committed=True)


def _keep_key_state(k, b):
    """Preserve battle state and PRNG key when the actor does not execute."""
    return (k, b)


def _refresh_action_abilities(before, after):
    """Resume a suppressed ability after the opposing Gas source leaves or faints."""
    for player in (0, 1):
        same = before.active_index[player, 0] == after.active_index[player, 0]
        suppressed = before.ability[player, 0] == 0
        same_identity = before.ability_identity[player, 0] == after.ability_identity[player, 0]
        after = resume_ability(after, player, jnp.where(same & suppressed & same_identity, 0, -1))
        after = refresh_trace_player(player, after)
    return _weather_form_update(before, after)


def _weather_form_update(before, after):
    """Refresh weather-sensitive forms after an actual weather change."""
    changed = before.weather.weather != after.weather.weather
    return lax.cond(changed, _refresh_snow_forms, _keep_battle, after)


def _refresh_snow_forms(state):
    """Restore active Ice Face holders when a new snow field starts."""
    return reactive_form_update(reactive_form_update(state, 0, True), 1, True)


def _keep_battle(state):
    """Preserve forms when the weather itself is unchanged."""
    return state


def _tera_player(index, battle, *, order, declarations):
    """Resolve one declared transformation if its bound actor is present."""
    player = order[index]
    return lax.cond(declarations[player], terastallize, _skip_tera_player, battle, player)


def _skip_tera_player(battle, player):
    """Keep an actor that did not declare a transformation."""
    return battle


def _resolve_teras(current, *, declarations):
    """Resolve both players' transformation actions in Speed order."""
    key, order_key = random.split(current.key)
    order = jnp.stack(action_order(order_key, current.battle, jnp.zeros(2, jnp.int32)))
    battle = lax.fori_loop(0, 2, partial(_tera_player, order=order, declarations=declarations), current.battle)
    return current.replace(key=key, battle=battle)


def _execute_queued_action(current):
    """Execute one actor, reveal spent/called moves and record any pivot request."""
    player = current.order[current.cursor]
    chosen = current.pending_actions[player]
    # Switch actions precede transformations, which precede all move actions.
    declarations = ((current.pending_actions.transformation == Transformation.TERA) &
                    tera_eligible(current.battle) &
                    (current.queued_actor_slot == current.battle.active_index[:, 0]))
    current = lax.cond((chosen.move_type != ActionType.SWITCH) & jnp.any(declarations),
        partial(_resolve_teras, declarations=declarations), _keep_episode, current)
    other = 1-player
    other_action = current.pending_actions.move_type[other]
    volatile = current.battle.volatile_status
    later = ((current.cursor == 0) & (current.battle.active_field('current_hp', other)[0] > 0) &
             (current.queued_actor_slot[other] == current.battle.active_index[other, 0]))
    will_act = later & ((other_action == ActionType.ATTACK) | (other_action == ActionType.SWITCH) |
                        (other_action == ActionType.CONTINUED))
    continuation_id = _continuation_id(volatile)[:, 0]
    pending = later & ((other_action == ActionType.ATTACK) | ((other_action == ActionType.CONTINUED) &
        (continuation_id[other] != 0)))
    other_index = current.pending_actions.index[other]
    charged_index = jnp.argmax(current.battle.active_move_field('identifier', other)[:, 0] == continuation_id[other])
    other_index = jnp.where(other_action == ActionType.CONTINUED, charged_index, jnp.clip(other_index, 0, 3))
    category = current.battle.active_move_field('move_type', other)[other_index, 0]
    category = jnp.where((other_action == ActionType.ATTACK) & (current.pending_actions.index[other] == 4), MoveType.PHYSICAL, category)
    battle_with_context = current.battle.replace(volatile_status=volatile.replace(
        queued_move=volatile.queued_move.replace(
            last_action=volatile.queued_move.last_action.at[player, 0].set(~will_act),
            pending_move=volatile.queued_move.pending_move.at[:, 0].set(False).at[other, 0].set(pending),
            pending_move_type=volatile.queued_move.pending_move_type.at[other, 0].set(category),
            pending_actor_slot=volatile.queued_move.pending_actor_slot.at[other, 0].set(current.queued_actor_slot[other]))))
    can_act = (current.battle.active_field('current_hp', player)[0] > 0) & (current.queued_actor_slot[player] == current.battle.active_index[player, 0])
    # Gravity can cancel a previously committed airborne release.
    can_act &= ((chosen.move_type != ActionType.CONTINUED) | (continuation_id[player] != 0) |
                volatile.continuation.recharging[player, 0])
    key, battle = lax.cond(can_act,
        partial(_dispatch_queued_action, chosen=chosen, player=player),
        _keep_key_state, current.key, battle_with_context)
    v = battle.volatile_status
    completed = can_act & ((chosen.move_type == ActionType.ATTACK) | (continuation_id[player] != 0))
    succeeded = completed & (v.move_history.move_result[player, 0] == MoveResult.SUCCEEDED) & (
        battle.hit_results[0].outcome != HitOutcome.PROTECTED)
    move_id = jnp.where(v.called_move.called_move_id[player, 0] != 0,
                        v.called_move.called_move_id[player, 0], v.move_history.last_move_id[player, 0])
    battle = battle.replace(volatile_status=v.replace(queued_move=v.queued_move.replace(
        preparation=v.queued_move.preparation.at[player, 0].set(0),
        focus_lost=v.queued_move.focus_lost.at[player, 0].set(False))),
        damage_history=battle.damage_history.replace(last_successful_move_this_turn=jnp.where(
            succeeded, move_id, battle.damage_history.last_successful_move_this_turn)))

    battle = _refresh_action_abilities(current.battle, battle)
    battle = refresh_paradox(battle)
    battle = capture_conditions_from_values(condition_values(current.battle), battle)
    slot = current.battle.active_index[player, 0]
    spent = battle.team.moves.current_pp[player, slot, :, 0] < current.battle.team.moves.current_pp[player, slot, :, 0]
    called_id = battle.volatile_status.called_move.called_move_id[player, 0]
    called = (called_id != 0) & (battle.team.moves.identifier[player, slot, :, 0] == called_id)
    known = current.revealed_moves.at[player, slot].set(current.revealed_moves[player, slot] | spent | called)
    has_bench = jnp.any(current.selected & (battle.team.current_hp[..., 0] > 0) &
                       (jnp.arange(6) != battle.active_index), axis=1)
    pivot = (battle.winner < 0) & battle.switch_requested & (battle.active_field('current_hp')[:, 0] > 0) & has_bench
    revival = (battle.winner < 0) & battle.revival_requested
    battle = battle.replace(switch_requested=jnp.zeros(2, bool), revival_requested=jnp.zeros(2, bool),
                            switch_transfer=jnp.where(pivot, battle.switch_transfer, 0))
    requested = pivot | revival
    return current.replace(key=key, battle=battle, revealed_moves=known, cursor=current.cursor + 1,
        phase=jnp.where(jnp.any(revival), Phase.REVIVAL, jnp.where(jnp.any(pivot), Phase.REPLACEMENT, Phase.TURN)),
        acting_player_mask=requested)


def _keep_episode(c):
    """Preserve the episode when a conditional scheduler branch is skipped."""
    return c


def _resolve_queued_action(_, current):
    """Resolve queued action."""
    execute = _execute_queued_action
    return lax.cond((current.cursor < 2) & (current.battle.winner < 0) & ~jnp.any(current.acting_player_mask), execute,
                    _keep_episode, current)


def _finish_turn(current):
    """Apply field effects once after the queued actions finish."""
    before = current.battle.replace(
        event_context=current.battle.event_context.replace(
            source=jnp.int32(-1),
            move_id=jnp.int32(0),
            cause=jnp.int32(Cause.FIELD)))
    key, battle = step_field(current.key, before)
    battle = capture_conditions_from_values(condition_values(before), battle)
    return current.replace(key=key, battle=battle, turn=current.turn + 1, field_pending=jnp.bool_(False))


def _execute_replacement(key, battle, player, index):
    """Switch the requested active slot during a replacement phase."""
    return step_switch(key, battle, player, index, False)


def _skip_replacement(key, battle, player, index):
    """Preserve state and key for a player not awaiting replacement."""
    return key, battle


def _revive_party_member(key, battle, player, index):
    """Restore a selected fainted member without changing the active position."""
    team = battle.team
    hp = jnp.maximum(1, team.max_hp[player, index] // 2)
    team = team.replace(
        current_hp=team.current_hp.at[player, index, 0].set(hp),
        is_alive=team.is_alive.at[player, index, 0].set(True),
        status=team.status.at[player, index, 0].set(Status.NONE),
        sleep_counter=team.sleep_counter.at[player, index, 0].set(0))
    return key, battle.replace(team=team)


def replace_player(player, carry):
    """Preserve player order, switch keys and condition snapshots."""
    current, actions = carry
    active = current.acting_player_mask[player] & (current.battle.winner < 0)
    branch = jnp.where(active, jnp.where(current.phase == Phase.REVIVAL, 2, 1), 0)
    key, battle = lax.switch(branch, (_skip_replacement, _execute_replacement, _revive_party_member),
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
    state = lax.cond((state.cursor == 2) & (state.battle.winner < 0) & ~jnp.any(state.acting_player_mask) & ~finished_replacement,
                     finish, _keep_episode, state)
    needs = (state.battle.active_field('current_hp')[:, 0] <= 0) & jnp.any((state.battle.team.current_hp[..., 0] > 0) & state.selected, axis=1)
    waiting = jnp.any(state.acting_player_mask)
    needs = (state.battle.winner < 0) & jnp.where(waiting, state.acting_player_mask, needs)
    phase = jnp.where(waiting & (state.phase == Phase.REVIVAL), Phase.REVIVAL, Phase.REPLACEMENT)
    state = state.replace(phase=jnp.where(jnp.any(needs), phase, Phase.TURN),
                          acting_player_mask=jnp.where(jnp.any(needs), needs, jnp.ones(2, bool)))
    revealed = state.revealed.at[jnp.arange(2), state.battle.active_index[:, 0]].set(True)
    return state.replace(revealed=revealed)
