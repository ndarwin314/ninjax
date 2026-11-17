"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_prototype import (
    jax,
    jnp,
    np,
    random_actions,
)


def test_policy_classes_and_uniform_choices():
    """Keep ordinary action probabilities and honor forced continuation masks."""
    mask = jnp.zeros((2, 15), bool).at[:, :4].set(True).at[:, 8:14].set(True)
    choices = jax.vmap(random_actions, in_axes=(0, None))(jax.random.split(jax.random.key(7), 10000), mask)
    switching = np.asarray(choices >= 8)
    assert .08 < switching.mean() < .12
    counts = np.bincount(np.asarray(choices).ravel(), minlength=15)
    assert np.all(counts[:4] > 4000)
    assert np.all(counts[8:14] > 240)
    forced = random_actions(jax.random.key(2), mask.at[:, :4].set(False))
    assert jnp.all((forced >= 8) & (forced < 14))
    forced_mask = jnp.zeros((2, 15), bool).at[0, 5].set(True).at[1, 14].set(True)
    forced = random_actions(jax.random.key(2), forced_mask)
    assert jnp.array_equal(forced, jnp.array([5, 14]))
