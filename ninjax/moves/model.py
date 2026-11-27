"""Move PyTree schema and metadata normalization, independent of execution."""
from dataclasses import field, fields

import jax
import jax.numpy as jnp

from ninjax.enums.moves import HitRule, MoveFlags, PowerRule, SecondaryEffectType
from ninjax.tree import ArrayTree, is_tree_placeholder


SECONDARY_SLOTS = 3
STAT_CHANGE_SLOTS = 7


class MoveSecondaries(ArrayTree):
    """Aligned effect slots and probabilities, padded to a fixed JAX shape.

    After the entity axes, effect leaves have shape (3,), stat and stage
    leaves have shape (3, 7), and the sampling mode has shape (1,).
    """

    mode: jax.Array = field(default_factory=lambda: jnp.array([SecondaryEffectType.DEFAULT], jnp.int32))
    probabilities: jax.Array = field(default_factory=lambda: jnp.zeros(SECONDARY_SLOTS, jnp.float32))
    status: jax.Array = field(default_factory=lambda: jnp.zeros(SECONDARY_SLOTS, jnp.int32))
    stats: jax.Array = field(default_factory=lambda: jnp.ones((SECONDARY_SLOTS, STAT_CHANGE_SLOTS), jnp.int32))
    stages: jax.Array = field(default_factory=lambda: jnp.zeros((SECONDARY_SLOTS, STAT_CHANGE_SLOTS), jnp.int32))
    user_stats: jax.Array = field(default_factory=lambda: jnp.ones((SECONDARY_SLOTS, STAT_CHANGE_SLOTS), jnp.int32))
    user_stages: jax.Array = field(default_factory=lambda: jnp.zeros((SECONDARY_SLOTS, STAT_CHANGE_SLOTS), jnp.int32))
    flinch: jax.Array = field(default_factory=lambda: jnp.zeros(SECONDARY_SLOTS, jnp.bool_))
    confusion: jax.Array = field(default_factory=lambda: jnp.zeros(SECONDARY_SLOTS, jnp.bool_))
    restriction: jax.Array = field(default_factory=lambda: jnp.zeros(SECONDARY_SLOTS, jnp.bool_))

    def broadcast_to(self, batch):
        """Normalize effect metadata to the containing move's entity axes."""
        slots = batch + (SECONDARY_SLOTS,)
        changes = slots + (STAT_CHANGE_SLOTS,)
        return self.replace(
            mode=jnp.broadcast_to(jnp.asarray(self.mode, jnp.int32), batch + (1,)),
            probabilities=jnp.broadcast_to(jnp.asarray(self.probabilities, jnp.float32), slots),
            status=jnp.broadcast_to(jnp.asarray(self.status, jnp.int32), slots),
            stats=jnp.broadcast_to(jnp.asarray(self.stats, jnp.int32), changes),
            stages=jnp.broadcast_to(jnp.asarray(self.stages, jnp.int32), changes),
            user_stats=jnp.broadcast_to(jnp.asarray(self.user_stats, jnp.int32), changes),
            user_stages=jnp.broadcast_to(jnp.asarray(self.user_stages, jnp.int32), changes),
            flinch=jnp.broadcast_to(jnp.asarray(self.flinch, jnp.bool_), slots),
            confusion=jnp.broadcast_to(jnp.asarray(self.confusion, jnp.bool_), slots),
            restriction=jnp.broadcast_to(jnp.asarray(self.restriction, jnp.bool_), slots))


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
    power_rule: jax.Array = field(default_factory=lambda: jnp.array([PowerRule.FIXED]))
    move_flags: jax.Array = field(default_factory=lambda: jnp.array([0]))
    recoil_percent: jax.Array = field(default_factory=lambda: jnp.array([0]))
    healing_fraction: jax.Array = field(default_factory=lambda: jnp.array([0.]))
    disabled: jax.Array = field(default_factory=lambda: jnp.array([False]))
    self_ko: jax.Array = field(default_factory=lambda: jnp.array([False]))
    perish_effect: jax.Array = field(default_factory=lambda: jnp.array([False]))
    identifier: jax.Array = field(default_factory=lambda: jnp.array([0]))
    flung_item: jax.Array = field(default_factory=lambda: jnp.array([0]))
    switch_out: jax.Array = field(default_factory=lambda: jnp.array([False]))
    implemented: jax.Array = field(default_factory=lambda: jnp.array([True]))
    target_self: jax.Array = field(default_factory=lambda: jnp.array([False]))
    # Primary confusion composes with target boosts (Swagger and Flatter).
    # Whether type conversion earned its priority-23 power modifier.
    type_power_boost: jax.Array = field(default_factory=lambda: jnp.array([False]))
    primary_confusion: jax.Array = field(default_factory=lambda: jnp.array([False]))
    status_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    boost_stat: jax.Array = field(default_factory=lambda: jnp.array([-1]))
    boost_stages: jax.Array = field(default_factory=lambda: jnp.array([0]))
    weather_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    terrain_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    # Accuracy is independently specified by per_hit_accuracy.
    hit_rule: jax.Array = field(default_factory=lambda: jnp.array([HitRule.SINGLE]))
    per_hit_accuracy: jax.Array = field(default_factory=lambda: jnp.array([False]))
    escalating_power: jax.Array = field(default_factory=lambda: jnp.array([False]))
    always_crit: jax.Array = field(default_factory=lambda: jnp.array([False]))
    # 0 none, 1 Stealth Rock, 2 Sticky Web, 3 Spikes, 4 Toxic Spikes.
    hazard_effect: jax.Array = field(default_factory=lambda: jnp.array([0]))
    secondaries: MoveSecondaries = field(default_factory=MoveSecondaries)
    self_stat: jax.Array = field(default_factory=lambda: jnp.array([1]))
    self_stages: jax.Array = field(default_factory=lambda: jnp.array([0]))
    self_chance: jax.Array = field(default_factory=lambda: jnp.array([1.]))
    drain_fraction: jax.Array = field(default_factory=lambda: jnp.array([0.]))
    ignore_defensive: jax.Array = field(default_factory=lambda: jnp.array([False]))
    always_hit: jax.Array = field(default_factory=lambda: jnp.array([False]))
    protectable: jax.Array = field(default_factory=lambda: jnp.array([True]))
    encore_eligible: jax.Array = field(default_factory=lambda: jnp.array([True]))
    sleep_talk_eligible: jax.Array = field(default_factory=lambda: jnp.array([True]))
    effectiveness_rule: jax.Array = field(default_factory=lambda: jnp.array([0]))

    def __post_init__(self):
        """Normalize scalar metadata and pad stat-change arrays to seven slots.

        Callers can supply one change or a vector of up to seven changes.
        Paired stat indices/stages have shape (..., 4, 7), independently of
        scalar metadata's (..., 4, 1). Padding uses zero stages, preserving a
        common shape when different moves are stacked, vmapped or scanned.
        Indices 1..5 name ordinary stats; 6 and 7 are accuracy and evasion.
        """
        if is_tree_placeholder(self.current_pp):
            return
        shape = jnp.shape(self.current_pp)
        change_fields = ('self_stat', 'self_stages', 'boost_stat', 'boost_stages')
        normalized = {}
        for indices_name, amounts_name in (('self_stat', 'self_stages'),
                                           ('boost_stat', 'boost_stages')):
            indices, amounts = jnp.broadcast_arrays(
                jnp.atleast_1d(jnp.asarray(getattr(self, indices_name), jnp.int32)),
                jnp.atleast_1d(jnp.asarray(getattr(self, amounts_name), jnp.int32)))
            normalized[indices_name], normalized[amounts_name] = indices, amounts
        for item in fields(self):
            name = item.name
            if name == 'secondaries':
                object.__setattr__(self, name, self.secondaries.broadcast_to(shape[:-1]))
                continue
            if name in change_fields:
                value = normalized[name]
                width = value.shape[-1]
                if width > 7:
                    raise ValueError(f'{name} supports at most seven stat changes')
                value = jnp.broadcast_to(value, shape[:-1] + (width,))
                # Most instances are reconstructed normalized PyTrees. A
                # zero-width pad still introduces a traced call on each of
                # these six leaves, so only pad caller-provided short vectors.
                if width < 7:
                    padding = [(0, 0)] * (value.ndim - 1) + [(0, 7-width)]
                    value = jnp.pad(value, padding)
                object.__setattr__(self, name, value)
                continue
            dtype = jnp.bool_ if name in ('switch_out', 'implemented', 'target_self', 'primary_confusion', 'type_power_boost', 'disabled', 'self_ko', 'perish_effect', 'per_hit_accuracy', 'escalating_power', 'always_crit', 'ignore_defensive', 'always_hit', 'protectable', 'encore_eligible', 'sleep_talk_eligible') else (jnp.float32 if name in ('accuracy', 'recoil_percent', 'healing_fraction', 'drain_fraction', 'self_chance') else jnp.int32)
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
    def thaws_target(self):
        """Whether the move thaws a frozen target after secondary effects."""
        return (self.move_flags & MoveFlags.THAWS_TARGET) == MoveFlags.THAWS_TARGET

    @property
    def ignore_ability(self):
        """Whether the move bypasses breakable opposing abilities."""
        return (self.move_flags & MoveFlags.IGNORE_ABILITY) == MoveFlags.IGNORE_ABILITY

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
