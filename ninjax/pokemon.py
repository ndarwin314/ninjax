from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum
from ninjax.utils import STAT_MULTIPLIER_LOOKUP

@struct.dataclass
class VolatileStatus:
    confused: bool = False
    # TODO this is gonna suck, make sure everything as default values

@struct.dataclass
class Pokemon:
    species: str
    name: str
    level: int
    gender: bool # :(
    # this might need to be changed later for dumb shenanigans with moves adding types
    type_list: (str, str)
    tera_type: str # TODO: enum
    is_terastallized: bool
    moves: (Move, Move, Move, Move)
    ability: str
    item: str
    stat_table: StatTable
    boosts: StatBoosts = StatBoosts()
    volatile_status: Any # TODO
    # add stats conditions and volatile status conditions

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def boosted_stats(self):
        return self.stats * STAT_MULTIPLIER_LOOKUP[self.boosts.normal_boosts]

    @property
    def accuracy_boosts(self):
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        return self.stat_table.current_hp

    @property
    def max_hp(self):
        return self.stat_table.stats[0]

def clear_boosts(pokemon: Pokemon):
    return pokemon.replace(boosts=StatBoosts())

def clear_volatile_status(pokemon: Pokemon):
    return pokemon.replace(volatile_status=VolatileStatus())