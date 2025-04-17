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
    increased: IntArray['*batch_size, 1']
    decreased: IntArray['*batch_size, 1']

    def row_update(self, idx: int, new_nature: 'Nature'):
        increased = self.increased.at[idx].set(new_nature.increased)
        decreased = self.decreased.at[idx].set(new_nature.decreased)
        return self.replace(increased=increased, decreased=decreased)

class StatBoosts(DataclassArray):
    normal_boosts: ['*batch_size 2'] = jnp.array([0, 0, 0, 0, 0, 0])
    acc_boosts: ['*batch_size 2'] = jnp.array([0, 0])
    # TODO: does this work?
    normal_boosts = jnp.clip(normal_boosts, -6, 6)
    acc_boosts = jnp.clip(acc_boosts, -6, 6)

    def replace_row(self, idx: int, new_boosts: 'StatBoosts'):
        normal_boosts = self.normal_boosts.at[idx].set(new_boosts.normal_boosts)
        acc_boosts = self.acc_boosts.at[idx].set(new_boosts.acc_boosts)
        return self.replace(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

    def __add__(self, other):
        normal_boosts = self.normal_boosts + other.normal_boosts
        acc_boosts = self.acc_boosts + other.acc_boosts
        return StatBoosts(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

class StatTable(DataclassArray):
    level: IntArray['*batch_size, 1'] = jnp.array(100)
    nature: Nature = Nature(increased=1, decreased=1)

    base_stats: IntArray['*batch_size 6'] = jnp.array([100, 100, 100, 100, 100, 100])
    ivs: IntArray['*batch_size 6'] = jnp.array([31, 31, 31, 31, 31, 31])
    evs: IntArray['*batch_size 6'] = jnp.array([84, 84, 84, 84, 84, 84])
    stats: IntArray['*batch_size 6'] = calculate_stats(level, nature, base_stats, ivs, evs)
    current_hp: IntArray['*batch_size, 1'] = stats[0]

    def row_update(self, idx: int, new_stats: 'StatTable'):
        level = self.level.at[idx].set(new_stats.level)
        ivs = self.ivs.at[idx].set(new_stats.ivs)
        evs = self.evs.at[idx].set(new_stats.evs)
        stats = self.stats.at[idx].set(new_stats.stats)
        current_hp = self.current_hp.at[idx].set(new_stats.current_hp)
        nature = self.nature.row_update(idx, new_stats.nature)
        return self.replace(level=level, nature=nature, ivs=ivs, evs=evs, stats=stats, current_hp=current_hp)




