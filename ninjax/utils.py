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
stat_multiplier_lookup = numerators / denominators
accuracy_multiplier_lookup = (1 + numerators) / (1 + denominators)

@jax.jit
def calculate_stats(level: int, nature: Nature, base_stats: Array, ivs: Array, evs: Array):
    # initial part of compute
    stats: Array = jnp.floor((2 * base_stats + evs + ivs + jnp.floor(evs)) * level / 100) + 5

    # factor in nature modifiers
    # probably remove these at some point

    # this handles edge case when same stat is increased and decreased
    stats_tenth = stats / 10
    stats.at[nature.increased].add(stats_tenth)
    stats.at[nature.increased].add(stats_tenth)
    stats.at[nature.increased].apply(jnp.floor)
    stats.at[nature.decreased].apply(jnp.floor)

    # HP has an extra increase
    stats.at[0].add(level + 5)

    return stats

# computes damage before any multiplicative modifiers
def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2
