"""Major status and confusion application with explicit keys and item reactions."""
from functools import partial
import chex
import jax
import jax.numpy as jnp
import jax.random as random
from ninjax.enum_types import AbilityEnum, Status, TerrainEnum
from ninjax.side.state import BattleState
from ninjax.events import EventKind, emit
from typing import Tuple
from ninjax.side.item_effects import status_berry
from ninjax.utils import triple_or
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
    key, sub_key = random.split(key, 2)
    # The move transition decrements this counter before checking sleep.
    turns = random.randint(sub_key, (1,), minval=2, maxval=5)[0]
    active = active.replace(status=status, sleep_counter=turns*(status==Status.SLEEP))
    return key, active


def _unchanged_status(key, active, status):
    """Preserve key and Pokémon when a status attempt is blocked."""
    return key, active


def set_status(key: chex.PRNGKey, state: BattleState, side_idx, status: Status, activate_items=True) -> Tuple[chex.PRNGKey, BattleState]:
    """Attempt a major status, returning the key and updated battle state.

    Existing status, represented terrain/type/ability immunities, and a NONE status
    prevent the update. Rest delays item activation until its healing completes.
    A key is consumed only when ``status_helper`` runs.
    """
    active = state.active[side_idx]
    already_statused = active.status != Status.NONE
    is_immune = state.is_status_immune(side_idx, status)
    key, active = jax.lax.cond(
        (triple_or(already_statused, is_immune, status==Status.NONE) | (active.current_hp <= 0))[0],
        _unchanged_status,
        status_helper,
        key, active, status)
    state = update_active_fields(state, side_idx, status=active.status, sleep_counter=active.sleep_counter)

    return key, jax.lax.cond(jnp.asarray(activate_items), partial(_activate_status_berry, side_idx=side_idx), _identity, state)
