"""Major status and confusion application with explicit keys and item reactions."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.calculations.logic import triple_or
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import TerrainEnum
from ninjax.enums.pokemon import Status
from ninjax.records.events import EventKind, emit
from ninjax.side.item_effects import status_berry
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


def _apply_confusion_apply(args, *, player):
    """Sample the volatile duration and publish its start."""
    key, state = args
    key, subkey = random.split(key)
    duration = random.randint(subkey, (), 2, 6)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player, 0].set(duration)))

    return key, status_berry(emit(state, EventKind.CONDITION, player, detail=12, before=0, after=1), player)


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _activate_status_berry(s, *, side_idx):
    """Activate status berry."""
    return status_berry(s, side_idx)


def apply_confusion(key, state, player):
    """Apply a 2–5 counter confusion volatile when the target is eligible.

    Only the presence of confusion is public; its sampled duration is private.
    """
    v = state.volatile_status
    blocked = ((v.confusion_turns[player, 0] > 0) | (state.ability[player, 0] == AbilityEnum.OWN_TEMPO) |
               ((state.terrain.terrain == TerrainEnum.MISTY) & state.is_grounded(player)[0]))
    apply = partial(_apply_confusion_apply, player=player)
    return jax.lax.cond(blocked, _identity, apply, (key, state))


def status_helper(key, active, status: Status):
    """Set an active Pokémon's status and initialize its sleep counter.

    The counter is sampled from 2 through 4 inclusive for sleep and set to zero
    for other statuses. The returned key has been advanced once.
    """
    key, sleep_counter = _roll_sleep_counter(key, status)
    return key, active.replace(status=status, sleep_counter=sleep_counter)


def _roll_sleep_counter(key, status):
    """Advance the status key and sample a counter only used for sleep."""
    key, sub_key = random.split(key, 2)
    # The move transition decrements this counter before checking sleep.
    turns = random.randint(sub_key, (1,), minval=2, maxval=5)[0]
    return key, turns * (status == Status.SLEEP)


def _apply_status_fields(key, old_status, old_sleep, status):
    """Return only the status and sleep counter that the caller writes back."""
    key, sleep = _roll_sleep_counter(key, status)
    new_status = jnp.asarray(status, old_status.dtype).reshape(old_status.shape)
    new_sleep = jnp.asarray(sleep, old_sleep.dtype).reshape(old_sleep.shape)
    return key, new_status, new_sleep


def _unchanged_status(key, old_status, old_sleep, status):
    """Preserve the status fields and key when a status attempt is blocked."""
    return key, old_status, old_sleep


def set_status(key: chex.PRNGKey, state: BattleState, side_idx, status: Status, activate_items=True) -> Tuple[chex.PRNGKey, BattleState]:
    """Attempt a major status, returning the key and updated battle state.

    Existing status, represented terrain/type/ability immunities, and a NONE status
    prevent the update. Rest delays item activation until its healing completes.
    A key is consumed only when the status is applied.
    """
    old_status = state.active_field('status', side_idx)
    old_sleep = state.active_field('sleep_counter', side_idx)
    already_statused = old_status != Status.NONE
    is_immune = state.is_status_immune(side_idx, status)
    fainted = state.active_field('current_hp', side_idx) <= 0
    # Keep the conditional operands limited to the fields changed by status.
    key, new_status, new_sleep = jax.lax.cond(
        (triple_or(already_statused, is_immune, status == Status.NONE) | fainted)[0],
        _unchanged_status,
        _apply_status_fields,
        key, old_status, old_sleep, status)
    state = update_active_fields(state, side_idx, status=new_status, sleep_counter=new_sleep)

    return key, jax.lax.cond(jnp.asarray(activate_items), partial(_activate_status_berry, side_idx=side_idx), _identity, state)
