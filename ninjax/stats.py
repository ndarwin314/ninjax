from typing import Union, Tuple, Dict, Any
from collections import namedtuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray

from ninjax.enum_types import StatEnum
from ninjax.utils import calculate_stats, STAT_MULTIPLIER_LOOKUP

Array = jax.Array

class Nature(DataclassArray):
    increased: IntArray['*batch_size']
    decreased: IntArray['*batch_size']

@struct.dataclass
class StatBoosts:
    normal_boosts: Array = jnp.array([0, 0, 0, 0, 0, 0])
    acc_boosts: Array = jnp.array([0, 0])

class StatTable(DataclassArray):
    level: IntArray['*batch_size'] = 100
    nature: Nature = Nature(increased=1, decreased=1)

    base_stats: IntArray['*batch_size 6'] = jnp.array([100, 100, 100, 100, 100, 100])
    ivs: IntArray['*batch_size 6'] = jnp.array([31, 31, 31, 31, 31, 31])
    evs: IntArray['*batch_size 6'] = jnp.array([84, 84, 84, 84, 84, 84])
    stats: IntArray['*batch_size 6'] = calculate_stats(level, nature, base_stats, ivs, evs)
    current_hp: IntArray['*batch_size'] = stats[0]



