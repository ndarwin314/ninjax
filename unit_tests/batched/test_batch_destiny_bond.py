"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_destiny_bond import (
    bonded,
    jax,
    jnp,
    neutral_type,
    resolve_faints,
    take_damage_value,
)


def test_faint_resolution_trace(neutral_type):
    """Faint records and direct retaliation trace under batching without compilation."""
    key, state = neutral_type
    state = bonded(key, state)
    pending = take_damage_value(state, 1, jnp.array([9999]), True, source=0)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), pending)
    shape = jax.eval_shape(jax.vmap(lambda s: resolve_faints(s).active.current_hp), batch)
    assert shape.shape == (2, 2, 1)
