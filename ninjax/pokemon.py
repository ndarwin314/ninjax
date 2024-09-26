from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum, Type, Status
from dataclass_array import DataclassArray, dataclass_array
from dataclass_array.typing import FloatArray, IntArray, BoolArray
from ninjax.utils import STAT_MULTIPLIER_LOOKUP

@dataclass_array(broadcast=True)
class Pokemon(DataclassArray):
    type_list: IntArray['*batch_size 2']
    moves: Move['*batch_size 4']
    tera_type: IntArray['*batch_size 1'] = jnp.int32([0])
    #species: IntArray['*batch_size'] = jnp.int32([0])
    #name: IntArray['*batch_size'] = jnp.int32([0])
    level: IntArray['*batch_size 1'] = jnp.int32([0])
    is_alive: BoolArray['*batch_size 1'] = jnp.bool([0])
    gender: BoolArray['*batch_size 1'] = jnp.bool([0])
    is_terastallized: BoolArray['*batch_size 1'] = jnp.bool([0])
    status: IntArray['*batch_size 1'] = jnp.int32([0])
    #ability: IntArray['*batch_size'] = jnp.int32([0])
    #item: IntArray['*batch_size'] = jnp.int32([0])
    stat_table: StatTable = StatTable()
    current_hp: IntArray['*batch_size 1'] = jnp.int32([stat_table.current_hp])
    # add stats conditions and volatile status conditions

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def max_hp(self):
        return self.stat_table.stats[0]

    def is_type(self, t: Type):
        return jnp.any(self.type_list==t)

    @property
    def is_floating(self):
        # TODO: add checks for levitate and balloon
        return self.is_type(Type.FLYING)

    @property
    def is_hazard_immune(self):
        # check for boots
        return False

    @property
    def is_sand_immune(self):
        return jnp.logical_or(self.is_type(Type.STEEL), jnp.logical_or(Type.GROUND, Type.ROCK))

    @property
    def is_poison_immune(self):
        return jnp.logical_or(self.is_type(Type.POISON), self.is_type(Type.STEEL))

    @property
    def is_powder_immune(self):
        # add check for goggles
        return self.is_type(Type.GRASS)



