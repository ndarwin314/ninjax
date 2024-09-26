from typing import Union, Tuple, Dict, Any

from chex import Array
from flax import struct
import jax.numpy as jnp
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray

from ninjax.enum_types import MoveType, Type


class Move(DataclassArray):
    name: IntArray['*batch_shape']
    move_type: IntArray['*batch_shape']
    max_pp: IntArray['*batch_shape']
    current_pp: IntArray['*batch_shape']
    type: IntArray['*batch_shape']
    base_power: IntArray['*batch_shape']
    accuracy: FloatArray['*batch_shape']
    priority: IntArray['*batch_shape']




    
