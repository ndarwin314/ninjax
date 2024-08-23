from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from functools import partial

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from flax.nnx.nnx.transforms.transforms import jit_fn

Array = jax.Array
increase_mult = 1.1

range_ = np.array(range(-6, 7))
numerators = 2 * np.ones(13) + np.fmax(range_, 0)
denominators = 2 * np.ones(13) - np.fmin(range_, 0)
STAT_MULTIPLIER_LOOKUP = jnp.array(numerators / denominators)
ACCURACY_MULTIPLIER_LOOKUP = jnp.array((1 + numerators) / (1 + denominators))

TYPE_EFFECTIVENESS = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 0, 1, 1, 1/2, 1],
        [1, 1, 1/2, 1/2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1/2, 1, 1/2, 1, 2, 1],
        [1, 1, 2, 1/2, 1, 1/2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1/2, 1, 1, 1],
        [1, 1, 1, 2, 1/2, 1/2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1/2, 1, 1, 1],
        [1, 1, 1/2, 2, 1, 1/2, 1, 1, 1/2, 2, 1/2, 1, 1/2, 2, 1, 1/2, 1, 1/2, 1],
        [1, 1, 1/2, 1/2, 1, 2, 1/2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1/2, 1],
        [1, 2, 1, 1, 1, 1, 2, 1, 1/2, 1, 1/2, 1/2, 1/2, 2, 0, 1, 2, 2, 1/2],
        [1, 1, 1, 1, 1, 2, 1, 1, 1/2, 1/2, 1, 1, 1, 1/2, 1/2, 1, 1, 0, 2],
        [1, 1, 2, 1, 2, 1/2, 1, 1, 2, 1, 0, 1, 1/2, 2, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1/2, 1],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1/2, 1, 1, 1, 1, 0, 1/2, 1],
        [1, 1, 1/2, 1, 1, 2, 1, 1/2, 1/2, 1, 1/2, 2, 1, 1, 1/2, 1, 2, 1/2, 1/2],
        [1, 1, 2, 1, 1, 1, 2, 1/2, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 1/2, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1/2, 0],
        [1, 1, 1, 1, 1, 1, 1, 1/2, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1/2],
        [1, 1, 1/2, 1/2, 1/2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1/2, 2],
        [1, 1, 1/2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1, 1, 1, 2, 2, 1/2, 1]])

@jax.jit
def calculate_effectiveness_multiplier(attacking_type, defending_types) -> Array:
    return jnp.prod(TYPE_EFFECTIVENESS[attacking_type][defending_types])

@jax.jit
def calculate_stats(level: int, nature: "Nature", base_stats: Array, ivs: Array, evs: Array):
    # initial part of compute
    stats: Array = jnp.floor_divide((2 * base_stats + evs + ivs + jnp.floor(evs)) * level, 100) + 5

    # factor in nature modifiers
    stats_tenth = jnp.floor_divide(stats, 10)
    stats = stats.at[nature.increased].add(stats_tenth[nature.increased])
    stats = stats.at[nature.decreased].add(-stats_tenth[nature.decreased])

    # HP has an extra increase
    stats = stats.at[0].add(level + 5)

    return stats

# computes damage before any multiplicative modifiers
@jax.jit
def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2

@partial(jax.jit, static_argnums=2)
def static_len_array_access(array, index, length):
    return jax.lax.switch(index, [lambda: array[i] for i in range(length)])




