"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.teams.construction import make_move
from ninjax.tree import stack


@pytest.mark.jit
def test_move_indexing_under_jit_vmap_scan():
    """Shared indexing keeps move metadata and secondary slots aligned."""
    bank = stack([make_move(name) for name in ('Fire Fang', 'Charge Beam', 'Close Combat')])
    indices = jnp.array([2, 0, 1], jnp.int32)
    def read(index):
        """Return primary and secondary features from a dynamically chosen move."""
        move = bank[index]
        return (move.identifier, move.self_stat, move.self_stages,
                move.secondaries.probabilities, move.secondaries.user_stages)
    batched = jax.jit(jax.vmap(read))(indices)
    _, scanned = jax.jit(lambda rows: jax.lax.scan(
        lambda carry, row: (carry, read(row)), None, rows))(indices)
    for actual, sequence, expected in zip(batched, scanned,
            (bank.identifier[indices], bank.self_stat[indices], bank.self_stages[indices],
             bank.secondaries.probabilities[indices], bank.secondaries.user_stages[indices])):
        assert jnp.array_equal(actual, expected)
        assert jnp.array_equal(sequence, expected)
