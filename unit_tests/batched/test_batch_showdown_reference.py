"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_showdown_reference import (
    GAMES,
    Type,
    calculate_effectiveness_multiplier,
    jax,
    jnp,
    np,
    pytest,
)


@pytest.mark.parametrize('game', GAMES, ids=lambda g: str(g['seed']))
def test_types_of_moves_seen_in_trace(game):
    """Check compatibility with batched JAX transforms."""
    probes = game['typeProbes']
    attacking = jnp.array([int(Type[p['attackType'].upper()]) for p in probes])
    defending = jnp.array([[int(Type[t.upper()]) for t in p['defenderTypes']] +
                           [0] * (2-len(p['defenderTypes'])) for p in probes])
    actual = jax.jit(jax.vmap(calculate_effectiveness_multiplier))(attacking, defending)
    np.testing.assert_array_equal(actual, [p['multiplier'] for p in probes])
