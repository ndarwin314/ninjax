from typing import Union, Tuple, Dict, Any
from collections import namedtuple

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.enum_types import StatEnum
from ninjax.utils import calculate_stats, STAT_MULTIPLIER_LOOKUP

Array = jax.Array

@struct.dataclass
class Nature:
    increased: StatEnum
    decreased: StatEnum

@struct.dataclass
class StatBoosts:
    normal_boosts: Array = jnp.array([0, 0, 0, 0, 0, 0])
    acc_boosts: Array = jnp.array([0, 0])

@struct.dataclass
class StatTable:
    base_stats: Array = jnp.array([100, 100, 100, 100, 100, 100])
    level: int = 100
    nature: Nature = Nature(1, 1)
    ivs: Array = jnp.array([31, 31, 31, 31, 31, 31])
    evs: Array = jnp.array([84, 84, 84, 84, 84, 84])
    stats: Array = calculate_stats(level, nature, base_stats, ivs, evs)
    current_hp: int = stats[0]



