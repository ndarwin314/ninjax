from typing import Union, Tuple, Dict, Any

from chex import Array
from flax import struct
import jax.numpy as jnp
from dataclass_array import DataclassArray
import dataclass_array as dca
from dataclass_array.typing import FloatArray, IntArray

from ninjax.enum_types import MoveType, Type

@dca.dataclass_array(cast_dtype=True, broadcast=True)
class Move(DataclassArray):
    #name: IntArray['*batch_shape']
    # e.g. special, physical, status
    move_type: IntArray['*batch_shape 1']
    max_pp: IntArray['*batch_shape 1']
    current_pp: IntArray['*batch_shape 1']
    # e.g. fire, water
    type: IntArray['*batch_shape 1']
    base_power: IntArray['*batch_shape 1']
    accuracy: FloatArray['*batch_shape 1']
    priority: IntArray['*batch_shape 1']
    offensive_stat: IntArray['*batch_shape 1']
    defensive_stat: IntArray['*batch_shape 1']




    
