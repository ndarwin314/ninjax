from dataclasses import field

from chex import Array
from flax import struct
import jax.numpy as jnp
from dataclass_array import DataclassArray
import dataclass_array as dca
from dataclass_array.typing import FloatArray, IntArray, BoolArray

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
    crit_stage: IntArray['*batch_shape 1']
    makes_contact: BoolArray['*batch_shape 1'] = field(default_factory=lambda: jnp.array([0]))

    def reduce_pp(self, index, increment):
        return self.replace(current_pp=self.current_pp.at[index].subtract(increment))

    def set_pp(self, index, value):
        return self.replace(current_pp=self.current_pp.at[index].set(value))






    
