"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_before_move import (
    before_move,
    jax,
    jnp,
    neutral_type,
    pytest,
)


@pytest.mark.jit
def test_before_move_jit_vmap_scan(neutral_type):
    """The shared gate works with explicit keys under JAX transformations."""
    key, state = neutral_type
    keys = jax.random.split(key, 2)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    _, _, blocked = jax.jit(jax.vmap(lambda k, s: before_move(k, s, 0)))(keys, batch)
    assert not jnp.any(blocked)
    def step(carry, _):
        """Carry both state and randomness through successive attempts."""
        key, state = carry
        key, state, blocked = before_move(key, state, 0)
        return (key, state), blocked
    _, blocked = jax.jit(lambda k, s: jax.lax.scan(step, (k, s), None, length=2))(key, state)
    assert not jnp.any(blocked)
