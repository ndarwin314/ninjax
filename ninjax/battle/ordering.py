"""Priority, effective Speed and Trick Room action ordering."""

import chex
import jax.lax
import jax.numpy as jnp
from jax import random

from ninjax.enums.pokemon import StatEnum
from ninjax.side.state import BattleState


def equal_priority(r, speeds, priorities):
    """Choose the faster active Pokémon, breaking equal Speed with the supplied draw."""
    first = jnp.argmax(speeds)
    tie_val = jnp.less_equal(r, 0.5)
    return jax.lax.select(speeds[0]!=speeds[1], jnp.array([first, 1-first]), jnp.array([tie_val, 1-tie_val]))


def action_order(
    key: chex.PRNGKey,
    state: BattleState,
    priorities: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Order priority brackets, reversing effective Speed under Trick Room."""
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)

    priorities = jnp.array(priorities)
    diff_priorities = priorities[0] != priorities[1]
    speeds = state.boosted_stats[:, StatEnum.SPEED]
    speeds = jnp.where(state.trick_room_duration > 0, -speeds, speeds)
    diff_speeds = speeds[0] != speeds[1]
    first_prio = jnp.argmax(priorities)
    first_speed = jnp.argmax(speeds)
    tiebreak = jnp.less_equal(r, 0.5)
    first = (first_prio * diff_priorities +
             first_speed * jnp.logical_and(diff_speeds, 1-diff_priorities) +
             tiebreak * jnp.logical_and(1-diff_priorities, 1-diff_speeds))
    return first, 1-first
