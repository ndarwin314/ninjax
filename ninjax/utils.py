from typing import Union, Tuple, Dict, Any
from collections import namedtuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
increase_mult = 1.1

Nature = namedtuple("Nature", ["increased", "decreased"])
range_ = np.array(range(-6, 7))
numerators = 2 * np.ones(13) + np.fmax(range_, 0)
denominators = 2 * np.ones(13) - np.fmin(range_, 0)
STAT_MULTIPLIER_LOOKUP = numerators / denominators
ACCURACY_MULTIPLIER_LOOKUP = (1 + numerators) / (1 + denominators)

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
def calculate_stats(level: int, nature: Nature, base_stats: Array, ivs: Array, evs: Array):
    # initial part of compute
    stats: Array = jnp.floor((2 * base_stats + evs + ivs + jnp.floor(evs)) * level / 100) + 5

    # factor in nature modifiers
    stats_tenth = stats / 10
    stats.at[nature.increased].add(stats_tenth)
    stats.at[nature.decreased].add(-stats_tenth)
    stats.at[nature.increased].apply(jnp.floor)
    stats.at[nature.decreased].apply(jnp.floor)

    # HP has an extra increase
    stats.at[0].add(level + 5)

    return stats

# computes damage before any multiplicative modifiers
@jax.jit
def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2



