from dataclasses import field

import jax.numpy as jnp
from dataclass_array import DataclassArray
import dataclass_array as dca
from dataclass_array.typing import FloatArray, IntArray, BoolArray

from ninjax.enum_types import MoveType, Type, MoveFlags

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
    move_flags: IntArray['*batch_shape 1'] = field(default_factory=lambda: jnp.array([0]))
    recoil_percent: FloatArray['*batch_shape 1'] = field(default_factory=lambda: jnp.array([0]))

    def reduce_pp(self, index, increment):
        return self.replace(current_pp=self.current_pp.at[index].subtract(increment))

    def set_pp(self, index, value):
        return self.replace(current_pp=self.current_pp.at[index].set(value))

    @property
    def contact(self):
        return (self.move_flags & MoveFlags.CONTACT) == MoveFlags.CONTACT

    @property
    def punching(self):
        return self.move_flags & MoveFlags.PUNCHING == MoveFlags.PUNCHING

    @property
    def recoil(self):
        return self.recoil_percent != 0







    
