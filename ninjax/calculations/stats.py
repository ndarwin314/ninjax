from dataclasses import field

import jax
import jax.numpy as jnp

from ninjax.calculations.stat_formulas import STAT_MULTIPLIER_LOOKUP, calculate_stats
from ninjax.enums.moves import OffensiveStatOwner
from ninjax.tree import ArrayTree, broadcast_tree, is_tree_placeholder


Array = jax.Array
int32 = jnp.int32

class Nature(ArrayTree):
    """Nature indices controlling the increased and decreased Pokémon stats."""
    increased: jax.Array = field(default_factory=lambda: jnp.array([1]))
    decreased: jax.Array = field(default_factory=lambda: jnp.array([1]))

    def __post_init__(self):
        """Normalize nature indices while preserving JAX metadata placeholders."""
        if is_tree_placeholder(self.increased):
            return
        object.__setattr__(self, 'increased', jnp.asarray(self.increased, jnp.int32))
        object.__setattr__(self, 'decreased', jnp.asarray(self.decreased, jnp.int32))

    def row_update(self, idx: int, new_nature: 'Nature'):
        """Row update."""
        increased = self.increased.at[idx].set(new_nature.increased)
        decreased = self.decreased.at[idx].set(new_nature.decreased)
        return self.replace(increased=increased, decreased=decreased)

class StatBoosts(ArrayTree):
    """Clamped six-stat and accuracy/evasion stage PyTrees."""
    normal_boosts: jax.Array = field(default_factory=lambda: jnp.zeros(6, dtype=int32))
    acc_boosts: jax.Array = field(default_factory=lambda: jnp.zeros(2, dtype=int32))


    def __post_init__(self) -> None:
        """Clamp real stages; leave diagnostic PyTree placeholders untouched."""
        if is_tree_placeholder(self.normal_boosts):
            return
        # use __setattr__ manually instead of attribute assignment since the class is frozen
        object.__setattr__(self, "normal_boosts", jnp.clip(self.normal_boosts, -6, 6))
        object.__setattr__(self, "acc_boosts", jnp.clip(self.acc_boosts, -6, 6))

    def replace_row(self, idx: int, new_boosts: 'StatBoosts'):
        """Replace row."""
        normal_boosts = self.normal_boosts.at[idx].set(new_boosts.normal_boosts)
        acc_boosts = self.acc_boosts.at[idx].set(new_boosts.acc_boosts)
        return self.replace(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

    def __add__(self, other):
        """Add."""
        normal_boosts = self.normal_boosts + other.normal_boosts
        acc_boosts = self.acc_boosts + other.acc_boosts
        return StatBoosts(normal_boosts=normal_boosts, acc_boosts=acc_boosts)

class StatTable(ArrayTree):
    """Level, nature, base stats, IVs and EVs used to construct battle stats."""
    level: jax.Array = field(default_factory=lambda: jnp.array([100]))
    nature: Nature = field(default_factory=Nature)
    base_stats: jax.Array = field(default_factory=lambda: 100 * jnp.ones(6, dtype=int32))
    ivs: jax.Array = field(default_factory=lambda: 31*jnp.ones(6, dtype=int32))
    evs: jax.Array = field(default_factory=lambda: 84*jnp.ones(6, dtype=int32))

    def __post_init__(self):
        """Broadcast real stat inputs and preserve metadata-only tree leaves."""
        if is_tree_placeholder(self.base_stats):
            return
        batch = jnp.broadcast_shapes(jnp.asarray(self.base_stats).shape[:-1], jnp.asarray(self.level).shape[:-1])
        for name in ('level', 'base_stats', 'ivs', 'evs'):
            size = 1 if name == 'level' else 6
            object.__setattr__(self, name, jnp.broadcast_to(jnp.asarray(getattr(self, name), jnp.int32), batch + (size,)))
        object.__setattr__(self, 'nature', broadcast_tree(self.nature, batch))

    def row_update(self, idx: int, new_stats: 'StatTable'):
        """Row update."""
        level = self.level.at[idx].set(new_stats.level)
        ivs = self.ivs.at[idx].set(new_stats.ivs)
        evs = self.evs.at[idx].set(new_stats.evs)
        nature = self.nature.row_update(idx, new_stats.nature)
        return self.replace(level=level, nature=nature, ivs=ivs, evs=evs)

    @property
    def stats(self):
        """Stats."""
        return calculate_stats(self.level, self.nature, self.base_stats, self.ivs, self.evs)


def boosted_stats_helper(stats, boosts, ignore_drops, ignore_boosts):
    """Apply stages after optionally ignoring negative or positive stages."""
    boosts = boosts.normal_boosts
    boosts = jnp.minimum(boosts, jnp.where(ignore_boosts, 0, 6))
    boosts = jnp.maximum(boosts, jnp.where(ignore_drops, 0, -6))
    return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + boosts])


def select_offensive_stats(raw_stats, boosts, attacker_idx, owner, defender_unaware, is_crit):
    """Select an active owner's raw stats and apply offensive-stage rules.

    ``raw_stats`` and ``boosts`` have leading player axes of length two.
    ``owner`` selects the user or target. The returned stat vector excludes
    item and ability modifiers; callers apply those using the move user.
    """
    owner_idx = jnp.where(owner == OffensiveStatOwner.TARGET, 1 - attacker_idx, attacker_idx)
    return boosted_stats_helper(raw_stats[owner_idx], boosts[owner_idx], is_crit | defender_unaware, defender_unaware)


def raw_boosted_stats(raw_stats, boosts, attacker_idx, attacker_unaware, defender_unaware, is_crit,
                      offensive_owner=OffensiveStatOwner.USER):
    """Return offensive-owner and defender stats with represented stage rules.

    Critical hits and Unaware alter which stages are applied. The offensive
    owner defaults to the user; callers apply ability and item modifiers later.
    """
    defender_stats = raw_stats[1 - attacker_idx]
    defender_boosts = boosts[1 - attacker_idx]
    attacker_stats = select_offensive_stats(raw_stats, boosts, attacker_idx, offensive_owner, defender_unaware, is_crit)
    defender_stats = boosted_stats_helper(
        defender_stats,
        defender_boosts,
        attacker_unaware,
        jnp.logical_or(is_crit, attacker_unaware)
    )
    return attacker_stats, defender_stats
