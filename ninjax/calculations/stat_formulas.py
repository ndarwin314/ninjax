"""Stat construction and stage multiplier lookup tables."""

import jax.numpy as jnp
import numpy as np
from chex import Array


range_ = np.array(range(-6, 7))


numerators = 2 * np.ones(13) + np.fmax(range_, 0)


denominators = 2 * np.ones(13) - np.fmin(range_, 0)


STAT_MULTIPLIER_LOOKUP = jnp.array(numerators / denominators)


ACCURACY_MULTIPLIER_LOOKUP = jnp.array((1 + numerators) / (1 + denominators))


def calculate_stats(level: Array, nature: "Nature", base_stats: Array, ivs: Array, evs: Array):
    # initial part of compute
    """Compute the six stats from level, nature, base stats, IVs and EVs."""
    stats: Array = jnp.floor_divide((2 * base_stats + jnp.floor_divide(evs, 4) + ivs) * level, 100) + 5 + jnp.array([1, 0, 0, 0, 0, 0]) * (level + 5)

    # factor in nature modifiers
    stats_tenth = jnp.floor_divide(stats, 10)
    stats = stats + stats_tenth * (jnp.arange(6) == nature.increased)
    stats = stats - stats_tenth * (jnp.arange(6) == nature.decreased)

    return stats
