from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import field

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray

from ninjax.enum_types import StatEnum
from ninjax.utils import calculate_stats, STAT_MULTIPLIER_LOOKUP

Array = jax.Array

class Nature(DataclassArray):
    increased: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array([1]))
    decreased: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array([1]))

    def row_update(self, idx: int, new_nature: 'Nature'):
        increased = self.increased.at[idx].set(new_nature.increased)
        decreased = self.decreased.at[idx].set(new_nature.decreased)
        return self.replace(increased=increased, decreased=decreased)


class StatBoosts(DataclassArray):
    normal_boosts: IntArray['*batch_size 6'] = field(default_factory=lambda: jnp.zeros(6))
    acc_boosts: IntArray['*batch_size 2'] = field(default_factory=lambda: jnp.zeros(2))


    def __post_init__(self) -> None:
        self.normal_boosts = jnp.clip(self.normal_boosts, -6, 6)
        self.acc_boosts = jnp.clip(self.acc_boosts, -6, 6)

    def replace_row(self, idx: int, new_boosts: 'StatBoosts'):
        normal_boosts = self.normal_boosts.at[idx].set(new_boosts.normal_boosts)
        acc_boosts = self.acc_boosts.at[idx].set(new_boosts.acc_boosts)
        return self.replace(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

    def __add__(self, other):
        normal_boosts = self.normal_boosts + other.normal_boosts
        acc_boosts = self.acc_boosts + other.acc_boosts
        return StatBoosts(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

class StatTable(DataclassArray):
    level: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.array(100))
    nature: Nature = Nature()
    base_stats: IntArray['*batch_size 6'] = field(default_factory=lambda: jnp.zeros(6))
    ivs: IntArray['*batch_size 6'] = field(default_factory=lambda: 31*jnp.ones(6))
    evs: IntArray['*batch_size 6'] = field(default_factory=lambda: 84*jnp.ones(6))

    def __post_init__(self) -> None:
        self.stats = calculate_stats(self.level, self.nature, self.base_stats, self.ivs, self.evs)
        self.current_hp = self.stats[0]

    def row_update(self, idx: int, new_stats: 'StatTable'):
        level = self.level.at[idx].set(new_stats.level)
        ivs = self.ivs.at[idx].set(new_stats.ivs)
        evs = self.evs.at[idx].set(new_stats.evs)
        stats = self.stats.at[idx].set(new_stats.stats)
        current_hp = self.current_hp.at[idx].set(new_stats.current_hp)
        nature = self.nature.row_update(idx, new_stats.nature)
        return self.replace(level=level, nature=nature, ivs=ivs, evs=evs, stats=stats, current_hp=current_hp)





