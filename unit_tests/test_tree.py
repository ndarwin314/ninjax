"""Array-tree indexing preserves dynamic selection and static slice semantics."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.prototype import make_move
from ninjax.tree import stack


@pytest.mark.jit
def test_move_indexing_under_jit_vmap_scan():
    """Shared indexing keeps scalar metadata and padded stat vectors aligned."""
    bank = stack([make_move(name) for name in ('Dragon Pulse', 'Calm Mind', 'Close Combat')])
    indices = jnp.array([2, 0, 1], jnp.int32)
    def read(index):
        """Return both feature widths from a dynamically chosen move."""
        move = bank[index]
        return move.identifier, move.self_stat, move.self_stages
    batched = jax.jit(jax.vmap(read))(indices)
    _, scanned = jax.jit(lambda rows: jax.lax.scan(
        lambda carry, row: (carry, read(row)), None, rows))(indices)
    for actual, sequence, expected in zip(batched, scanned,
            (bank.identifier[indices], bank.self_stat[indices], bank.self_stages[indices])):
        assert jnp.array_equal(actual, expected)
        assert jnp.array_equal(sequence, expected)


def test_static_tree_slices():
    """Static slices and Ellipsis retain normal array indexing behavior."""
    bank = stack([make_move(name) for name in ('Dragon Pulse', 'Calm Mind', 'Close Combat')])
    assert jnp.array_equal(bank[1:].identifier, bank.identifier[1:])
    assert jnp.array_equal(bank[..., :].self_stat, bank.self_stat[..., :])
