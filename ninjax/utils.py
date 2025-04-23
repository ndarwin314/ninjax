from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from functools import partial

import chex
from flax import struct
import jax
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
import numpy as np

# i deserve to be tried at the hague for this line of code
# dataclass_array checks if field have a hash as a proxy for being immutable
# so i'm adding a dummy hash function to it here to avoid it throwing errors
# I'm good at programming
ArrayImpl.__hash__ = lambda : 0

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

def calculate_effectiveness_multiplier(attacking_type, defending_types) -> Array:
    return jnp.prod(TYPE_EFFECTIVENESS[attacking_type][defending_types])

def calculate_stats(level: Array, nature: "Nature", base_stats: Array, ivs: Array, evs: Array):
    # initial part of compute
    stats: Array = jnp.floor_divide((2 * base_stats + jnp.floor_divide(evs, 4) + ivs) * level, 100) + 5 + jnp.array([1, 0, 0, 0, 0, 0]) * (level + 5)

    # factor in nature modifiers
    stats_tenth = jnp.floor_divide(stats, 10)
    stats = stats.at[nature.increased].add(stats_tenth[nature.increased])
    stats = stats.at[nature.decreased].add(-stats_tenth[nature.decreased])

    return stats

# computes damage before any multiplicative modifiers

def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2





