"""Persistent and turn-scoped battle facts used by history-dependent moves."""
from dataclasses import field
from functools import partial

import jax.numpy as jnp
from flax import struct


class DamageHistory(struct.PyTreeNode):
    """Counters per team slot and temporary action facts for one singles battle.

    Team leaves have shape (2, 6), side totals shape (2,), and the last
    successful move is scalar. Slot counters persist through switches.
    """

    times_attacked: jnp.ndarray = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.int32))
    total_fainted: jnp.ndarray = field(default_factory=partial(jnp.zeros, (2,), dtype=jnp.int32))
    hurt_this_turn: jnp.ndarray = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.bool_))
    stats_lowered_this_turn: jnp.ndarray = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.bool_))
    last_successful_move_this_turn: jnp.ndarray = field(default_factory=partial(jnp.int32, 0))
    counter_damage: jnp.ndarray = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.int32))
    counter_category: jnp.ndarray = field(default_factory=partial(jnp.full, (2, 6), -1, dtype=jnp.int32))
    counter_source_slot: jnp.ndarray = field(default_factory=partial(jnp.full, (2, 6), -1, dtype=jnp.int32))
    counter_prepared: jnp.ndarray = field(default_factory=partial(jnp.full, (2,), -1, dtype=jnp.int32))

    def reset_turn(self):
        """Clear temporary records while retaining per-Pokémon and side counters."""
        return self.replace(
            hurt_this_turn=jnp.zeros_like(self.hurt_this_turn),
            stats_lowered_this_turn=jnp.zeros_like(self.stats_lowered_this_turn),
            last_successful_move_this_turn=jnp.zeros_like(self.last_successful_move_this_turn),
            counter_damage=jnp.zeros_like(self.counter_damage),
            counter_category=jnp.full_like(self.counter_category, -1),
            counter_source_slot=jnp.full_like(self.counter_source_slot, -1),
            counter_prepared=jnp.full_like(self.counter_prepared, -1),
        )
