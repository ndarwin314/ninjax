from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum

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
    # add stats conditions and volatile status conditions

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def boosted_stats(self):
        return self.stat_table.boosted_stats

    @property
    def boosts(self):
        return self.stat_table.boosts.normal_boosts

    @property
    def accuracy_boosts(self):
        return self.stat_table.boosts.acc_boosts
