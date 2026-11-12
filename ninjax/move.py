import jax
from ninjax.tree import ArrayTree
from dataclasses import field, fields

import jax.numpy as jnp

from ninjax.enum_types import MoveFlags

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
    healing_fraction: jax.Array = field(default_factory=lambda: jnp.array([0.]))
    disabled: jax.Array = field(default_factory=lambda: jnp.array([False]))
    self_ko: jax.Array = field(default_factory=lambda: jnp.array([False]))
    perish_effect: jax.Array = field(default_factory=lambda: jnp.array([False]))
    identifier: jax.Array = field(default_factory=lambda: jnp.array([0]))
    switch_out: jax.Array = field(default_factory=lambda: jnp.array([False]))
    implemented: jax.Array = field(default_factory=lambda: jnp.array([True]))
    target_self: jax.Array = field(default_factory=lambda: jnp.array([False]))
    status_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    boost_stat: jax.Array = field(default_factory=lambda: jnp.array([-1]))
    boost_stages: jax.Array = field(default_factory=lambda: jnp.array([0]))
    weather_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    terrain_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    # 1 = one hit; 2 = Gen 9 two-to-five distribution; 3 = fixed three;
    # 4 = Population Bomb's ten attempts. Accuracy is independently specified.
    hit_rule: jax.Array = field(default_factory=lambda: jnp.array([1]))
    per_hit_accuracy: jax.Array = field(default_factory=lambda: jnp.array([False]))
    escalating_power: jax.Array = field(default_factory=lambda: jnp.array([False]))
    always_crit: jax.Array = field(default_factory=lambda: jnp.array([False]))
    # 0 none, 1 Stealth Rock, 2 Sticky Web, 3 Spikes, 4 Toxic Spikes.
    hazard_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))

    def __post_init__(self):
        shape = jnp.shape(self.current_pp)
        for item in fields(self):
            name = item.name
            dtype = jnp.bool_ if name in ('switch_out', 'implemented', 'target_self', 'disabled', 'self_ko', 'perish_effect', 'per_hit_accuracy', 'escalating_power', 'always_crit') else (jnp.float32 if name in ('accuracy', 'recoil_percent', 'healing_fraction') else jnp.int32)
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







    
