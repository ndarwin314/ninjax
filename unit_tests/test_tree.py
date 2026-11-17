"""Array-tree indexing preserves dynamic selection and static slice semantics."""
import jax
import jax.numpy as jnp

from ninjax.team_builder import make_move
from ninjax.tree import stack
from unit_tests.test_battle import neutral_type


def test_tree_reconstruction_preserves_diagnostic_placeholders(neutral_type):
    """JAX cache diagnostics can reconstruct state with opaque metadata leaves."""
    _, state = neutral_type
    leaves, definition = jax.tree.flatten(state)
    placeholder = object()
    rebuilt = jax.tree.unflatten(definition, [placeholder] * len(leaves))
    assert all(value is placeholder for value in jax.tree.leaves(rebuilt))




def test_static_tree_slices():
    """Static slices and Ellipsis retain normal array indexing behavior."""
    bank = stack([make_move(name) for name in ('Dragon Pulse', 'Calm Mind', 'Close Combat')])
    assert jnp.array_equal(bank[1:].identifier, bank.identifier[1:])
    assert jnp.array_equal(bank[..., :].self_stat, bank.self_stat[..., :])
