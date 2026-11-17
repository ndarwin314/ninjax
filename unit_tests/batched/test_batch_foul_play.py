"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_foul_play import (
    jax,
    jnp,
    neutral_type,
)
from ninjax.enum_types import OffensiveStatOwner as Owner
from ninjax.stats import select_offensive_stats


def test_foul_play_selection_traces(neutral_type):
    """The target-owner selector supports JAX batching without compiling a battle."""
    _, state = neutral_type


    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    result = jax.eval_shape(jax.vmap(lambda s: select_offensive_stats(s.active.stats, s.boosts, 0, Owner.TARGET, False, False)), batch)
    assert result.shape == (2, 6)
