import jax
from flax import struct
from ninjax.tree import ArrayTree
from dataclasses import field

import jax.numpy as jnp

from ninjax.enum_types import MoveType, Type, MoveFlags

@struct.dataclass
class Move(ArrayTree):
    """Move arrays have entity axes followed by (4 moves, 1 feature)."""
    #name: jax.Array
    # e.g. special, physical, status
    move_type: jax.Array
    max_pp: jax.Array
    current_pp: jax.Array
    # e.g. fire, water
    type: jax.Array
    base_power: jax.Array
    accuracy: jax.Array
    priority: jax.Array
    offensive_stat: jax.Array
    defensive_stat: jax.Array
    crit_stage: jax.Array
    move_flags: jax.Array = field(default_factory=lambda: jnp.array([0]))
    recoil_percent: jax.Array = field(default_factory=lambda: jnp.array([0]))

    def __post_init__(self):
        shape = jnp.shape(self.current_pp)
        for name in self.__dataclass_fields__:
            dtype = jnp.float32 if name in ('accuracy', 'recoil_percent') else jnp.int32
            object.__setattr__(self, name, jnp.broadcast_to(jnp.asarray(getattr(self, name), dtype), shape))

    def reduce_pp(self, index, increment):
        return self.replace(current_pp=self.current_pp.at[index].subtract(increment))

    def set_pp(self, index, value):
        return self.replace(current_pp=self.current_pp.at[index].set(value))

    @property
    def powder(self):
        return (self.move_flags & MoveFlags.POWDER) == MoveFlags.POWDER

    @property
    def contact(self):
        return (self.move_flags & MoveFlags.CONTACT) == MoveFlags.CONTACT

    @property
    def punching(self):
        return (self.move_flags & MoveFlags.PUNCHING) == MoveFlags.PUNCHING

    @property
    def biting(self):
        return (self.move_flags & MoveFlags.BITING) == MoveFlags.BITING

    @property
    def sound(self):
        return (self.move_flags & MoveFlags.SOUND) == MoveFlags.SOUND

    @property
    def recoil(self):
        return self.recoil_percent != 0

    @property
    def bullet(self):
        return (self.move_flags & MoveFlags.BULLET) == MoveFlags.BULLET

    @property
    def launcher(self):
        return (self.move_flags & MoveFlags.LAUNCHER) == MoveFlags.LAUNCHER

    @property
    def healing(self):
        return (self.move_flags & MoveFlags.HEALING) == MoveFlags.HEALING







    
