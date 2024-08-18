from typing import Union, Tuple, Dict, Any
from collections import namedtuple

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.enum_types import StatEnum
from ninjax.utils import calculate_stats, stat_multiplier_lookup

Array = jax.Array

# simple way to represent a nature by the stat it increases and decreases
# may need to modify with struct.dataclass
Nature = namedtuple("Nature", ["increased", "decreased"])

@struct.dataclass
class StatBoosts:
    normal_boosts: Array = (0, 0, 0, 0, 0)
    acc_boosts: Array = (0, 0)

@struct.dataclass
class StatTable:
    base_stats: Array = (100, 100, 100, 100, 100, 100)
    level: int = 100
    nature: Nature = (1, 1)
    ivs: Array = (31, 31, 31, 31, 31, 31)
    evs: Array = (84, 84, 84, 84, 84, 84)
    stats: Array = calculate_stats(level, nature, base_stats, ivs, evs)
    current_hp: int = stats[0]



