"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_action_dependent_moves import (
    effective_move,
    jnp,
    make_move,
    neutral_type,
)
import jax


def test_stomping_result_transform_shapes(neutral_type):
    """Previous-result power selection traces under vmap and scan."""

    _, state = neutral_type
    move = make_move('Stomping Tantrum')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.eval_shape(jax.vmap(lambda s: effective_move(s, 0, move).base_power), batch).shape == (2, 1)
    def scan(s):
        """Carry move-result state while observing power over two turns."""
        return jax.lax.scan(lambda s, _: (s, effective_move(s, 0, move).base_power), s, None, length=2)[1]
    assert jax.eval_shape(scan, state).shape == (2, 1)
