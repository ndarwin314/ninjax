from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
from gymnax.environments.environment import Environment
import jax.numpy as jnp

from jaxcalibur.move import Move
from jaxcalibur.stats import StatTable

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
    stats: StatTable
