"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_roost import (
    jax,
    jnp,
    neutral_type,
)


def test_effective_roost_types_trace(neutral_type):
    """Roost type and grounded checks trace under vmap and scan."""
    _, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(roost=v.roost.at[0, 0].set(True)))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.eval_shape(jax.vmap(lambda s: s.effective_active.type_list), batch).shape == (2, 2, 2)
    def scan(s):
        """Observe effective types across a fixed scan."""
        return jax.lax.scan(lambda s, _: (s, s.effective_active.type_list), s, None, length=2)[1]
    assert jax.eval_shape(scan, state).shape == (2, 2, 2)
