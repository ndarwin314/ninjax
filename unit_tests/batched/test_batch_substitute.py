"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_substitute import (
    HitOutcome,
    HitRecipient,
    jnp,
    neutral_type,
)
import jax
from ninjax.hit import record_hit


def test_hit_record_shapes_under_jax_transforms(neutral_type):
    """Tracing batched and scanned hits retains a fixed PyTree shape."""


    _, state = neutral_type
    def one(s, target):
        """Record one attempted hit for a traced target index."""
        return record_hit(s, 0, target, HitOutcome.HIT, HitRecipient.POKEMON, 3)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    shaped = jax.eval_shape(jax.vmap(lambda s, target: one(s, target).hit_results.data), batch, jnp.array([0, 1]))
    assert shaped.shape == (2, 10, 7)
    def scan(s):
        """Trace a fixed scan without compiling a full battle."""
        return jax.lax.scan(lambda carry, i: (one(carry, i), None), s, jnp.array([0, 1]))[0].hit_results.data
    assert jax.eval_shape(scan, state).shape == (10, 7)
