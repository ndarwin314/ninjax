from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax.numpy as jnp

from ninjax.enum_types import MoveType, Type


@struct.dataclass
class Move:
    name: int
    move_type: MoveType
    max_pp: int
    current_pp: int
    type: Type
    base_power: int
    accuracy: int
    priority: int
