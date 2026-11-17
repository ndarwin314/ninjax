"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp

from ninjax.enum_types import OffensiveStatOwner as Owner
from ninjax.stats import select_offensive_stats
from unit_tests.test_battle import neutral_type


def test_foul_play_selection_traces(neutral_type):
    """The target-owner selector supports JAX batching without compiling a battle."""
    _, state = neutral_type


    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    result = jax.eval_shape(jax.vmap(lambda s: select_offensive_stats(s.active.stats, s.boosts, 0, Owner.TARGET, False, False)), batch)
    assert result.shape == (2, 6)
