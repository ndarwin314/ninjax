import jax
from ninjax.tree import ArrayTree
from dataclasses import field, fields

import jax.numpy as jnp

from ninjax.enum_types import MoveFlags

class Move(ArrayTree):
    """Move arrays have entity axes followed by move and feature axes.

    Most fields use (4 moves, 1 feature); paired stat-change fields use
    (4 moves, 7 changes), with zero stages marking unused entries.
    """
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
    # Secondary effects describe one chance roll per hit. Stat changes have
    # seven feature slots, while probabilities and other metadata have one.
    secondary_chance: jax.Array = field(default_factory=lambda: jnp.array([0.]))
    secondary_status: jax.Array = field(default_factory=lambda: jnp.array([0]))
    secondary_stat: jax.Array = field(default_factory=lambda: jnp.array([1]))
    secondary_stages: jax.Array = field(default_factory=lambda: jnp.array([0]))
    secondary_flinch: jax.Array = field(default_factory=lambda: jnp.array([False]))
    secondary_confusion: jax.Array = field(default_factory=lambda: jnp.array([False]))
    self_stat: jax.Array = field(default_factory=lambda: jnp.array([1]))
    self_stages: jax.Array = field(default_factory=lambda: jnp.array([0]))
    self_chance: jax.Array = field(default_factory=lambda: jnp.array([1.]))
    drain_fraction: jax.Array = field(default_factory=lambda: jnp.array([0.]))
    ignore_defensive: jax.Array = field(default_factory=lambda: jnp.array([False]))
    always_hit: jax.Array = field(default_factory=lambda: jnp.array([False]))
    protectable: jax.Array = field(default_factory=lambda: jnp.array([True]))
    encore_eligible: jax.Array = field(default_factory=lambda: jnp.array([True]))
    sleep_talk_eligible: jax.Array = field(default_factory=lambda: jnp.array([True]))

    def __post_init__(self):
        """Normalize scalar metadata and pad stat-change arrays to seven slots.

        Callers can supply one change or a vector of up to seven changes.
        Paired stat indices/stages have shape (..., 4, 7), independently of
        scalar metadata's (..., 4, 1). Padding uses zero stages, preserving a
        common shape when different moves are stacked, vmapped or scanned.
        Indices 1..5 name ordinary stats; 6 and 7 are accuracy and evasion.
        """
        shape = jnp.shape(self.current_pp)
        change_fields = ('self_stat', 'self_stages', 'boost_stat', 'boost_stages',
                         'secondary_stat', 'secondary_stages')
        normalized = {}
        for indices_name, amounts_name in (('self_stat', 'self_stages'),
                                           ('boost_stat', 'boost_stages'),
                                           ('secondary_stat', 'secondary_stages')):
            indices, amounts = jnp.broadcast_arrays(
                jnp.atleast_1d(jnp.asarray(getattr(self, indices_name), jnp.int32)),
                jnp.atleast_1d(jnp.asarray(getattr(self, amounts_name), jnp.int32)))
            normalized[indices_name], normalized[amounts_name] = indices, amounts
        for item in fields(self):
            name = item.name
            if name in change_fields:
                value = normalized[name]
                width = value.shape[-1]
                if width > 7:
                    raise ValueError(f'{name} supports at most seven stat changes')
                value = jnp.broadcast_to(value, shape[:-1] + (width,))
                padding = [(0, 0)] * (value.ndim - 1) + [(0, 7-width)]
                object.__setattr__(self, name, jnp.pad(value, padding))
                continue
            dtype = jnp.bool_ if name in ('switch_out', 'implemented', 'target_self', 'disabled', 'self_ko', 'perish_effect', 'per_hit_accuracy', 'escalating_power', 'always_crit', 'secondary_flinch', 'secondary_confusion', 'ignore_defensive', 'always_hit', 'protectable', 'encore_eligible', 'sleep_talk_eligible') else (jnp.float32 if name in ('accuracy', 'recoil_percent', 'healing_fraction', 'secondary_chance', 'drain_fraction', 'self_chance') else jnp.int32)
            object.__setattr__(self, name, jnp.broadcast_to(jnp.asarray(getattr(self, name), dtype), shape))

    def reduce_pp(self, index, increment):
        return self.replace(current_pp=self.current_pp.at[index].subtract(increment))

    def set_pp(self, index, value):
        return self.replace(current_pp=self.current_pp.at[index].set(value))

    @property
    def powder(self):
        return (self.move_flags & MoveFlags.POWDER) == MoveFlags.POWDER

    @property
    def defrost(self):
        """Whether this move can thaw its frozen user when executed."""
        return (self.move_flags & MoveFlags.DEFROST) == MoveFlags.DEFROST

    @property
    def bypass_substitute(self):
        """Whether move metadata explicitly bypasses Substitute."""
        return (self.move_flags & MoveFlags.BYPASS_SUBSTITUTE) == MoveFlags.BYPASS_SUBSTITUTE

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







    
