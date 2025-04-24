from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import field
import dataclass_array as dca


import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray
from flax.core import broadcast

from ninjax.enum_types import StatEnum
from ninjax.utils import calculate_stats, STAT_MULTIPLIER_LOOKUP

Array = jax.Array
int32 = jnp.int32

class Nature(DataclassArray):
    increased: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array([1]))
    decreased: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array([1]))

    def row_update(self, idx: int, new_nature: 'Nature'):
        increased = self.increased.at[idx].set(new_nature.increased)
        decreased = self.decreased.at[idx].set(new_nature.decreased)
        return self.replace(increased=increased, decreased=decreased)


class StatBoosts(DataclassArray):
    normal_boosts: IntArray['*batch_size 6'] = field(default_factory=lambda: jnp.zeros(6, dtype=int32))
    acc_boosts: IntArray['*batch_size 2'] = field(default_factory=lambda: jnp.zeros(2, dtype=int32))


    def __post_init__(self) -> None:
        super().__post_init__()
        # use __setattr__ manually instead of attribute assignment since the class is frozen
        object.__setattr__(self, "normal_boosts", jnp.clip(self.normal_boosts, -6, 6))
        object.__setattr__(self, "acc_boosts", jnp.clip(self.acc_boosts, -6, 6))

    def replace_row(self, idx: int, new_boosts: 'StatBoosts'):
        normal_boosts = self.normal_boosts.at[idx].set(new_boosts.normal_boosts)
        acc_boosts = self.acc_boosts.at[idx].set(new_boosts.acc_boosts)
        return self.replace(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

    def __add__(self, other):
        normal_boosts = self.normal_boosts + other.normal_boosts
        acc_boosts = self.acc_boosts + other.acc_boosts
        return StatBoosts(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

@dca.dataclass_array(cast_dtype=True, broadcast=True)
class StatTable(DataclassArray):
    level: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array([100]))
    nature: Nature = Nature()
    base_stats: IntArray['*batch_size 6'] = field(default_factory=lambda: 100 * jnp.ones(6, dtype=int32))
    ivs: IntArray['*batch_size 6'] = field(default_factory=lambda: 31*jnp.ones(6, dtype=int32))
    evs: IntArray['*batch_size 6'] = field(default_factory=lambda: 84*jnp.ones(6, dtype=int32))

    def row_update(self, idx: int, new_stats: 'StatTable'):
        level = self.level.at[idx].set(new_stats.level)
        ivs = self.ivs.at[idx].set(new_stats.ivs)
        evs = self.evs.at[idx].set(new_stats.evs)
        nature = self.nature.row_update(idx, new_stats.nature)
        # so this is a problem because we calculate the current_hp in the post_init which runs even when we call replace
        return self.replace(level=level, nature=nature, ivs=ivs, evs=evs)

    @property
    def stats(self):
        return calculate_stats(self.level, self.nature, self.base_stats, self.ivs, self.evs)





