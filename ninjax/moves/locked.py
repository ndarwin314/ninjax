"""Repeated-move lifecycle for singles action continuations."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from ninjax.enums.moves import LockedMoveKind
from ninjax.enums.pokemon import Status
from ninjax.records.hits import HitOutcome
from ninjax.side.status_effects import apply_confusion


def _keep_key_state(key, state):
    """Preserve the current state without consuming randomness."""
    return key, state


def clear_lock(state, player):
    """Clear all state owned by the active Pokémon's repeated move."""
    v = state.volatile_status
    lock = v.continuation.locked
    lock = lock.replace(
        kind=lock.kind.at[player, 0].set(0),
        move_id=lock.move_id.at[player, 0].set(0),
        actor_slot=lock.actor_slot.at[player, 0].set(-1),
        turns_left=lock.turns_left.at[player, 0].set(0),
        successful_hits=lock.successful_hits.at[player, 0].set(0))
    return state.replace(volatile_status=v.replace(
        continuation=v.continuation.replace(locked=lock)))


def _finish_rampage(key, state, *, player):
    """Confuse the user after a naturally completed rampage."""
    state = clear_lock(state, player)
    return apply_confusion(key, state, player)


def _sample_rampage(key):
    """Choose a two- or three-turn rampage when the sequence starts."""
    key, sample = random.split(key)
    return key, random.randint(sample, (), 2, 4)


def _fixed_duration(key, *, kind):
    """Use fixed durations for Rollout and Uproar."""
    return key, jnp.where(kind == LockedMoveKind.ROLLOUT, 5, 3)


def _start_lock(key, state, *, player, move, kind):
    """Start an eligible repeated move after its first successful hit."""
    key, duration = jax.lax.cond(kind == LockedMoveKind.RAMPAGE,
        _sample_rampage, partial(_fixed_duration, kind=kind), key)
    v = state.volatile_status
    lock = v.continuation.locked
    lock = lock.replace(
        kind=lock.kind.at[player, 0].set(kind),
        move_id=lock.move_id.at[player, 0].set(move.identifier[0]),
        actor_slot=lock.actor_slot.at[player, 0].set(state.active_index[player, 0]),
        target_side=lock.target_side.at[player, 0].set(1-player),
        target_position=lock.target_position.at[player, 0].set(0),
        turns_left=lock.turns_left.at[player, 0].set(duration - 1),
        successful_hits=lock.successful_hits.at[player, 0].set(1))
    return key, state.replace(volatile_status=v.replace(
        continuation=v.continuation.replace(locked=lock)))


def _advance_lock(key, state, *, player, hit):
    """Advance a continuing sequence and finish or clear it when appropriate."""
    lock = state.volatile_status.continuation.locked
    kind = lock.kind[player, 0]
    remaining = lock.turns_left[player, 0] - 1
    complete = remaining <= 0
    alive = state.active_field('current_hp', player)[0] > 0
    sleeping = state.active_field('status', player)[0] == Status.SLEEP
    # Uproar persists through failed attempts. Rollout breaks immediately.
    # A rampage disrupted on its final turn still ends with confusion.
    cancel = (~hit & (kind != LockedMoveKind.UPROAR)) | ~alive | (
        sleeping & (kind == LockedMoveKind.RAMPAGE))
    lock = lock.replace(
        turns_left=lock.turns_left.at[player, 0].set(jnp.maximum(remaining, 0)),
        successful_hits=lock.successful_hits.at[player, 0].add(hit.astype(jnp.int32)))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        continuation=v.continuation.replace(locked=lock)))
    finish = complete & alive & ~sleeping & (kind == LockedMoveKind.RAMPAGE)
    key, state = jax.lax.cond(finish, partial(_finish_rampage, player=player),
                              _keep_key_state, key, state)
    state = jax.lax.cond((complete | cancel) & ~finish,
                         partial(clear_lock, player=player), _identity, state)
    return key, state


def interrupt_lock(key, state, player):
    """Advance interrupted locks, accounting for sleep and Throat Chop."""
    lock = state.volatile_status.continuation.locked
    uproar_silenced = ((lock.kind[player, 0] == LockedMoveKind.UPROAR) &
                       (state.volatile_status.restrictions.throat_chop_turns[player, 0] > 0))
    key, state = _advance_lock(key, state, player=player, hit=jnp.bool_(False))
    state = jax.lax.cond(uproar_silenced, partial(clear_lock, player=player), _identity, state)
    return key, state


def _identity(state):
    """Preserve state when a sequence remains active."""
    return state


def update_lock(key, state, player, move, continued):
    """Start or advance a repeat policy after a completed move attempt."""
    ident = move.identifier[0]
    kind = jnp.select((jnp.isin(ident, jnp.array([37, 80, 200])), ident == 205, ident == 253),
                       (LockedMoveKind.RAMPAGE, LockedMoveKind.ROLLOUT, LockedMoveKind.UPROAR),
                       default=LockedMoveKind.NONE)
    hit = state.hit_results[0].outcome == HitOutcome.HIT
    previous = state.volatile_status.continuation.locked.move_id[player, 0] != 0
    key, state = jax.lax.cond(previous & continued,
        partial(_advance_lock, player=player, hit=hit), _keep_key_state, key, state)
    start = (kind != LockedMoveKind.NONE) & ~continued & hit & move.implemented[0]
    start &= state.active_field('current_hp', player)[0] > 0
    return jax.lax.cond(start, partial(_start_lock, player=player, move=move, kind=kind),
                        _keep_key_state, key, state)
