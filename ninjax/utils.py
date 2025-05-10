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
one_point_three = 5325/4096
one_point_one = 4506/4096
one_point_three_exact = 1.3
one_third = 1/3


range_ = np.array(range(-6, 7))
numerators = 2 * np.ones(13) + np.fmax(range_, 0)
denominators = 2 * np.ones(13) - np.fmin(range_, 0)
STAT_MULTIPLIER_LOOKUP = jnp.array(numerators / denominators)
ACCURACY_MULTIPLIER_LOOKUP = jnp.array((1 + numerators) / (1 + denominators))
TERRAIN_MULTIPLIER = one_point_three_exact
CRIT_STAGES = jnp.array([1/24, 1/8, 1,2, 1, 1])
COMPOUND_EYES_MULTIPLIER = one_point_three
VICTORY_STAR = one_point_one
TOUGH_CLAWS = one_point_three
WEATHER_VEIL_MODIFIER = 3277/4096
ROUGH_SKIN_DAMAGE = 1/8
IRON_FIST = 1.2
RECKLESS = 1.2

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

def triple_and(a, b, c):
    return jnp.logical_and(jnp.logical_and(a, b), c)

def triple_or(a, b, c):
    return jnp.logical_or(jnp.logical_or(a, b), c)

def quad_or(a, b, c, d):
    return jnp.logical_or(a, jnp.logical_and(jnp.logical_and(b, c), d))

def in_range(lb, value, ub):
    return jnp.logical_and(jnp.less(lb, value), jnp.less_equal(value, ub))

def calculate_effectiveness_multiplier(attacking_type, defending_types) -> Array:
    temp = TYPE_EFFECTIVENESS[attacking_type, defending_types]
    return jnp.prod(temp)

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

def conditional_mult_round(damage, mult, cond):
    # fix correctly rounds down at .5 for all values rather than banker's rounding
    return jnp.fix(conditional_mult(damage, mult, cond)).astype(int)


def conditional_mult(value, mult, cond):
    # TODO, there is a buggy thing happening where multiplying a (2,1) array by a (2,) makes a (2,2) instead of (2,1)
    # this doesnt seem correct to me but i guess its how jax broadcasts so figure out how to deal with that
    return value * jnp.power(mult, cond)





