"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_stat_owner import (
    Owner,
    StatBoosts,
    jax,
    jnp,
    select_offensive_stats,
)


def test_stat_owner_traces_with_dynamic_owner():
    """The owner selector supports traced values under vmap and scan."""
    stats = jnp.full((2, 6), 100, jnp.int32)
    boosts = StatBoosts(normal_boosts=jnp.zeros((2, 6), jnp.int32), acc_boosts=jnp.zeros((2, 2), jnp.int32))
    owners = jnp.array([Owner.USER, Owner.TARGET])
    def select(owner):
        """Select a traced stat owner."""
        return select_offensive_stats(stats, boosts, 0, owner, False, False)
    assert jax.eval_shape(jax.vmap(select), owners).shape == (2, 6)
    assert jax.eval_shape(lambda: jax.lax.scan(lambda _, x: (None, select(x)), None, owners)[1]).shape == (2, 6)
