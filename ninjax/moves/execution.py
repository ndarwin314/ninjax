"""Move selection bookkeeping, called moves and special execution setup.

State and PRNG keys remain explicit; callbacks are module-level pure functions.
"""

from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import prepare_type_change
from ninjax.enum_types import (
    AbilityEnum,
    MoveType,
    StatEnum,
    Status,
    WeatherEnum,
)
from ninjax.events import Cause, EventKind, emit
from ninjax.hit import HitOutcome, HitResults, record_hit
from ninjax.moves.finalization import _struggle_recoil, finish_attempt
from ninjax.moves.properties import effective_move
from ninjax.moves.routing import _apply_move_effects, move_misses
from ninjax.side.faint_effects import clear_destiny_bond, direct_faint
from ninjax.side.queries import effective_weather, is_choice_item
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts, update_active_fields


def _successful_check(k):
    """Return success without consuming a random key."""
    return (k, jnp.bool_(True))


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)


def _ready_selected_move(k, s, p, i):
    """Allow hit resolution without further selection preparation."""
    return (k, s, jnp.bool_(True))


def _unchanged_selected_move(k, s, p, i):
    """Skip a selected move while accepting its player and slot operands."""
    return (k, s)


def move_interrupted(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """End action-scoped volatiles and report a before-move interruption."""

    state = clear_destiny_bond(state, attacker_index)
    # TODO: eventually we will need to figure out we handle observations and include it here
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protect_chain=v.protect_chain.at[attacker_index, 0].set(0)))
    return key, emit(state, EventKind.INTERRUPTED, attacker_index)


def _release_charged_move(k, s, p, i):
    """Announce a released charge without consuming PP or changing type."""

    s = clear_destiny_bond(s, p)
    return k, emit(s, EventKind.MOVE, p, detail=s.event_move)


def _prepare_selected_or_released_move(k, s, p, i, *, continued):
    """Choose ordinary PP bookkeeping or charged-move release bookkeeping."""
    return jax.lax.cond(continued, _release_charged_move, _prepare_move_use, k, s, p, i)


def move_used(key, state: BattleState, attacker_index, move_index, continued=False, struggle=False):
    """Prepare a selection, release or forced Struggle for one hit pipeline.

    A charge release has already paid PP and established its move context in
    the action scheduler. It skips selection bookkeeping and charge setup.
    """
    key, state = jax.lax.cond(
        struggle,
        _prepare_struggle_use,
        partial(_prepare_selected_or_released_move, continued=continued),
        key,
        state,
        attacker_index,
        move_index,
    )
    move = state.active[attacker_index].moves[move_index]
    key, state = _execute_selected_or_called_move(key, state, attacker_index, move_index,
        ~jnp.asarray(struggle) & ~jnp.asarray(continued) & (move.identifier[0] == 214) & move.implemented[0], continued)
    return key, jax.lax.cond(struggle, partial(_struggle_recoil, player=attacker_index), _identity, state)


def _prepare_struggle_use(key, state, player, unused_index):
    """Announce forced Struggle and invalidate Encore/Disable move history."""

    state = clear_destiny_bond(state, player)
    v = state.volatile_status
    state = state.replace(hit_results=HitResults.empty(), event_move=jnp.int32(-1),
        volatile_status=v.replace(last_move_id=v.last_move_id.at[player, 0].set(0)))
    return key, emit(state, EventKind.MOVE, player, detail=-1)


def _clear_move_destiny_bond(s, *, attacker_index):
    """End the previous Destiny Bond before another move is attempted."""

    return clear_destiny_bond(s, attacker_index)


def _prepare_move_type_change(s, *, attacker_index, move):
    """Apply Protean or Libero using the effective selected move."""

    return prepare_type_change(s, attacker_index, effective_move(s, attacker_index, move))


def _prepare_move_use(key, state: BattleState, attacker_index, move_index):
    """Consume one PP and establish ordinary selected-move bookkeeping.

    Unsupported moves consume PP but otherwise take the no-effect branch. The
    caller is responsible for ensuring the selected move has PP remaining.
    """
    move = state.active[attacker_index].moves[move_index]

    state = jax.lax.cond(
        move.identifier[0] != 194,
        partial(_clear_move_destiny_bond, attacker_index=attacker_index),
        _identity,
        state,
    )

    # Defrost-flagged moves thaw the user only when the move actually executes.
    mon = state.active[attacker_index]
    thaw = move.defrost & (mon.status == Status.FREEZE)
    state = update_active_fields(state, attacker_index, status=jnp.where(thaw, Status.NONE, mon.status))

    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        last_move_id=v.last_move_id.at[attacker_index, 0].set(move.identifier[0]),
        called_move_id=v.called_move_id.at[attacker_index, 0].set(0),
        protect_chain=v.protect_chain.at[attacker_index, 0].set(
            jnp.where((move.identifier[0] == 182) | (move.identifier[0] == 596), v.protect_chain[attacker_index, 0], 0))))

    state = state.replace(hit_results=HitResults.empty(), event_source=jnp.asarray(attacker_index, jnp.int32),
                          event_move=move.identifier[0], event_cause=jnp.int32(Cause.MOVE))
    state = emit(state, EventKind.MOVE, attacker_index, detail=move.identifier[0])

    choice = is_choice_item(state.active.item[attacker_index, 0])
    lock = state.volatile_status.choice_lock
    state = state.replace(volatile_status=state.volatile_status.replace(choice_lock=lock.at[attacker_index, 0].set(
        jnp.where(choice & (lock[attacker_index, 0] < 0), move_index, lock[attacker_index, 0]))))
    # Moves that call other moves do not themselves activate Protean/Libero.
    state = jax.lax.cond(
        move.identifier[0] == 214,
        _identity,
        partial(_prepare_move_type_change, attacker_index=attacker_index, move=move),
        state,
    )
    # decrement pp
    # theoretically the legal action mask should prevent us from using the move if its at 0 so we dont need to clip
    # PP is the only modified move leaf; avoid writing the other metadata back.
    slot = state.active_index[attacker_index, 0]
    moves = state.team.moves
    state = state.replace(team=state.team.replace(moves=moves.replace(
        current_pp=moves.current_pp.at[attacker_index, slot, move_index].add(-1))))
    return key, state


def _detonate_explosion(k, s, *, p):
    """Save source HP for damage calculation and commit its faint."""

    v = s.volatile_status
    s = s.replace(volatile_status=v.replace(explosion_hp=v.explosion_hp.at[p, 0].set(s.active.current_hp[p, 0])))
    s = direct_faint(s, p, p, Cause.SELF_KO)
    return k, s, jnp.bool_(True)


def _block_explosion(k, s, *, damp_users, p):
    """Reveal the Damp ability that prevented the explosion."""
    owner = jnp.argmax(damp_users)
    slot = s.active_index[owner, 0]
    s = s.replace(known_abilities=s.known_abilities.at[owner, slot].set(AbilityEnum.DAMP))
    s = emit(s, EventKind.ABILITY, owner, detail=AbilityEnum.DAMP)
    return k, record_hit(s, 0, 1-p, HitOutcome.BLOCKED), jnp.bool_(False)


def _prepare_explosion(k, s, p, i):
    """Damp prevents execution; otherwise commit self-faint before the hit."""
    damp_users = (s.ability[:, 0] == AbilityEnum.DAMP) & (s.active.current_hp[:, 0] > 0)
    damp = jnp.any(damp_users)
    return jax.lax.cond(
        damp,
        partial(_block_explosion, damp_users=damp_users, p=p),
        partial(_detonate_explosion, p=p),
        k,
        s,
    )


def _fail_selected_move(k, s, p, i, *, move):
    """Record an explicit move failure after PP has been consumed."""
    s = record_hit(s, 0, 1-p, HitOutcome.FAILED)
    return k, emit(s, EventKind.MOVE_FAILED, p, detail=move.identifier[0]), jnp.bool_(False)


def _prepare_charge_or_hit(k, s, p, i, *, electro_shot):
    """Prepare Electro Shot or continue immediately to hit resolution."""
    return jax.lax.cond(electro_shot, _prepare_electro_shot,
                    _ready_selected_move, k, s, p, i)


def _prepare_special_move(k, s, p, i, *, explosion, electro_shot):
    """Special moves only prepare state and decide whether to attempt a hit."""
    return jax.lax.cond(
        explosion,
        _prepare_explosion,
        partial(_prepare_charge_or_hit, electro_shot=electro_shot),
        k,
        s,
        p,
        i,
    )


def _prepare_initial_selection(k, s, p, i, *, fake_fails, sucker_fails, move, explosion, electro_shot):
    """Initial selections can fail or start charging before hit resolution."""
    return jax.lax.cond(
        fake_fails | sucker_fails,
        partial(_fail_selected_move, move=move),
        partial(_prepare_special_move, explosion=explosion, electro_shot=electro_shot),
        k,
        s,
        p,
        i,
    )


def _apply_selected_move(k, s, *, move, struggle, attacker_index, move_index):
    """Keep accuracy, immunity, status and damage resolution in one branch."""
    return jax.lax.cond(move.implemented[0] | struggle, _apply_move_effects, move_misses,
                        k, s, attacker_index, move_index)


def _clear_explosion_hp(s, *, attacker_index):
    """Discard the pre-faint HP after the shared hit pipeline returns."""
    v = s.volatile_status
    return s.replace(volatile_status=v.replace(explosion_hp=v.explosion_hp.at[attacker_index, 0].set(0)))


@jax.jit
def dispatch_selected_move(key, state, attacker_index, move_index, continued=False):
    """Prepare a paid-for/called move or release, then share one hit path.

    The scalar ``continued`` flag bypasses initial charge setup and ordinary
    attempt finalization; release bookkeeping was handled by the scheduler.
    """
    struggle = state.event_move == -1
    move = state.active[attacker_index].moves[move_index]
    explosion = (move.identifier[0] == 153) & move.implemented[0]
    electro_shot = (move.identifier[0] == 905) & move.implemented[0]

    v = state.volatile_status
    target = 1-attacker_index
    fake_fails = (move.identifier[0] == 252) & (v.active_move_actions[attacker_index, 0] > 1)
    sucker_fails = (move.identifier[0] == 389) & (~v.pending_move[target, 0] |
        (v.pending_move_type[target, 0] == MoveType.STATUS) |
        (v.pending_actor_slot[target, 0] != state.active_index[target, 0]))


    key, state, ready = jax.lax.cond(
        continued | struggle,
        _ready_selected_move,
        partial(
            _prepare_initial_selection,
            fake_fails=fake_fails,
            sucker_fails=sucker_fails,
            move=move,
            explosion=explosion,
            electro_shot=electro_shot,
        ),
        key,
        state,
        attacker_index,
        move_index,
    )

    key, state = jax.lax.cond(
        ready,
        partial(
            _apply_selected_move,
            move=move,
            struggle=struggle,
            attacker_index=attacker_index,
            move_index=move_index,
        ),
        _unchanged_key_state,
        key,
        state,
    )
    # Explosion's saved HP is action-local, even when the target was unaffected.
    state = jax.lax.cond(
        explosion & ready & ~struggle,
        partial(_clear_explosion_hp, attacker_index=attacker_index),
        _identity,
        state,
    )
    return key, jax.lax.cond(
        continued | struggle,
        _identity,
        partial(finish_attempt, player=attacker_index, move=move),
        state,
    )


def _fail_sleep_talk(k, s, *, player, move_index):
    """Sleep Talk fails awake or without any callable moves."""
    s = record_hit(s, 0, player, HitOutcome.FAILED)
    return k, emit(s, EventKind.MOVE_FAILED, player, detail=214), jnp.asarray(move_index, jnp.int32), jnp.bool_(False)


def _select_sleep_talk_move(k, s, *, eligible, player):
    """Select a move and establish its priority, sleep and public context."""
    k, select_key = random.split(k)
    # Keep the selected slot aligned with the action's int32 index, also
    # when a caller enables JAX's 64-bit default integer mode.
    selected = random.categorical(select_key, jnp.where(eligible, 0., -jnp.inf)).astype(jnp.int32)
    called = s.active[player].moves[selected]
    v = s.volatile_status
    s = s.replace(event_move=called.identifier[0], volatile_status=v.replace(
        sleep_talk_active=v.sleep_talk_active.at[player, 0].set(True),
        sleep_talk_succeeded=v.sleep_talk_succeeded.at[player, 0].set(True),
        called_move_id=v.called_move_id.at[player, 0].set(called.identifier[0]),
        encore_overridden=v.encore_overridden.at[player, 0].set(True)))

    s = prepare_type_change(s, player, effective_move(s, player, called))
    s = emit(s, EventKind.MOVE, player, detail=called.identifier[0])
    return k, s, selected, jnp.bool_(True)


def _prepare_sleep_talk(key, state, player, move_index):
    """Return a called slot and execution flag without entering the hit pipeline."""
    moves = state.active[player].moves
    asleep = (state.active.status[player, 0] == Status.SLEEP) | (state.ability[player, 0] == AbilityEnum.COMATOSE)
    eligible = moves.sleep_talk_eligible[:, 0] & moves.implemented[:, 0] & (moves.identifier[:, 0] != 0) & (moves.identifier[:, 0] != 214)
    eligible &= jnp.arange(4) != move_index
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(sleep_talk_succeeded=v.sleep_talk_succeeded.at[player, 0].set(False)))
    return jax.lax.cond(
        asleep & jnp.any(eligible),
        partial(_select_sleep_talk_move, eligible=eligible, player=player),
        partial(_fail_sleep_talk, player=player, move_index=move_index),
        key,
        state,
    )


def _use_selected_slot(k, s, p, i):
    """Retain the selected int32 slot when Sleep Talk is not being used."""
    return (k, s, jnp.asarray(i, jnp.int32), jnp.bool_(True))


def _restore_sleep_talk_context(s, *, old_move, player, old_override):
    """Restore Sleep Talk's event and Encore context after success or failure."""
    v = s.volatile_status
    return s.replace(event_move=old_move, volatile_status=v.replace(
        sleep_talk_active=v.sleep_talk_active.at[player, 0].set(False),
        encore_overridden=v.encore_overridden.at[player, 0].set(old_override)))


def _execute_selected_or_called_move(key, state, player, move_index, sleep_talk, continued=False):
    """Select the effective slot, dispatch once and restore called-move context."""
    old_move = state.event_move
    old_override = state.volatile_status.encore_overridden[player, 0]
    key, state, selected, ready = jax.lax.cond(sleep_talk, _prepare_sleep_talk,
        _use_selected_slot,
        key, state, player, move_index)
    key, state = jax.lax.cond(
        ready,
        partial(dispatch_selected_move, continued=continued),
        _unchanged_selected_move,
        key,
        state,
        player,
        selected,
    )

    return key, jax.lax.cond(
        sleep_talk & ready,
        partial(_restore_sleep_talk_context, old_move=old_move, player=player, old_override=old_override),
        _identity,
        state,
    )


def execute_sleep_talk(key, state, player, move_index):
    """Call an eligible move without spending its PP, using the shared executor."""
    return _execute_selected_or_called_move(key, state, player, move_index, jnp.bool_(True))


def _charge_electro_shot(k, s, *, player):
    """Remember the move and original target without attempting a hit."""
    v = s.volatile_status
    s = s.replace(volatile_status=v.replace(
        charging_move_id=v.charging_move_id.at[player, 0].set(905),
        charge_target_side=v.charge_target_side.at[player, 0].set(1-player),
        charge_target_position=v.charge_target_position.at[player, 0].set(0)))
    return k, emit(s, EventKind.CONDITION, player, detail=14, before=0, after=905)


def _prepare_electro_shot(key, state, player, index):
    """Boost SpA and return whether rain permits an immediate hit attempt."""

    state = add_boosts(state, player, StatEnum.SPECIAL_ATTACK, 1)
    ready = effective_weather(state.weather.weather, state.ability, state.active.current_hp) == WeatherEnum.RAIN
    key, state = jax.lax.cond(ready, _unchanged_key_state, partial(_charge_electro_shot, player=player), key, state)
    return key, state, ready


def start_electro_shot(key, state, player, index):
    """Charge or immediately execute Electro Shot without another PP payment."""
    key, state, ready = _prepare_electro_shot(key, state, player, index)
    return jax.lax.cond(ready, _apply_move_effects, _unchanged_selected_move,
                        key, state, player, index)

