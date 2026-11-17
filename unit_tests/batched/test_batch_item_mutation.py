"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_item_mutation import (
    I,
    items,
    jax,
    jnp,
    neutral_type,
    swap_items,
)


def test_item_mutation_traces_under_jax(neutral_type):
    """Pure mutation helpers retain fixed shapes under vmap and scan tracing."""
    _, state = neutral_type
    state = items(state, I.LIFE_ORB, I.LEFTOVERS)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    shape = jax.eval_shape(jax.vmap(lambda s: swap_items(s, 0, 1)[0].active.item), batch)
    assert shape.shape == (2, 2, 1)
    def scan(s):
        """Carry immutable item state through repeated exchanges."""
        return jax.lax.scan(lambda s, _: (swap_items(s, 0, 1)[0], None), s, None, length=2)[0].active.item
    assert jax.eval_shape(scan, state).shape == (2, 1)
