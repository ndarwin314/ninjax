from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum, Type, Status
from ninjax.utils import STAT_MULTIPLIER_LOOKUP

@struct.dataclass
class Pokemon:
    type_list: chex.Array
    moves: list[Move]
    tera_type: Type = 0
    species: int = 0
    name: int = 0
    level: int = 100
    is_alive = True
    gender: bool = False
    # this might need to be changed later for dumb shenanigans with moves adding types
    is_terastallized: bool = False
    ability: int = 0
    item: int = 0
    stat_table: StatTable = StatTable()
    current_hp: int = stat_table.current_hp
    max_hp: int = current_hp
    status: Status = Status.NONE
    # add stats conditions and volatile status conditions

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def max_hp(self):
        return self.stat_table.stats[0]

    def get_move(self, index: int):
        return jax.lax.switch(index, [lambda: self.moves[i] for i in range(4)])

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
        return jnp.any(
            self.type_list==Type.STEEL +
            self.type_list==Type.GROUND +
            self.type_list==Type.ROCK
        )

    @property
    def is_poison_immune(self):
        return jnp.logical_or(self.is_type(Type.POISON), self.is_type(Type.STEEL))

    @property
    def is_powder_immune(self):
        # add check for goggles
        return self.is_type(Type.GRASS)



