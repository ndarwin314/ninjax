"""Field updates preserve full replacement behavior and public status events."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enum_types import ItemEnum, Status, Type
from ninjax.side import update_active, update_active_fields
from unit_tests.test_battle import neutral_type


@pytest.mark.parametrize('updates', [
    {'current_hp': jnp.array([17.9]), 'is_alive': jnp.array([True])},
    {'item': jnp.array([ItemEnum.SITRUS_BERRY])},
    {'status': jnp.array([Status.SLEEP]), 'sleep_counter': jnp.array([3])},
    {'status': jnp.array([Status.NONE])},
    {'type_list': jnp.array([Type.WATER, Type.NONE])},
])
def test_field_update_matches_full_replacement(neutral_type, updates):
    """Only requested fields change, retaining constructor casts and event order."""
    _, state = neutral_type
    state = state.replace(active_index=jnp.array([[2], [4]], jnp.int32))
    expected = update_active(state, 1, state.active[1].replace(**updates))
    actual = update_active_fields(state, 1, **updates)
    assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))


@pytest.mark.jit
def test_field_updates_compile_with_dynamic_player(neutral_type):
    """Field names are static while player indices and values batch and scan."""
    _, state = neutral_type
    state = state.replace(active_index=jnp.array([[2], [4]], jnp.int32))
    def update(player):
        """Modify only the selected active slot's HP."""
        return update_active_fields(state, player, current_hp=jnp.array([19])).team.current_hp
    players = jnp.arange(2)
    batched = jax.jit(jax.vmap(update))(players)
    _, scanned = jax.jit(lambda rows: jax.lax.scan(
        lambda carry, row: (carry, update(row)), None, rows))(players)
    assert jnp.array_equal(batched, scanned)
    for player in range(2):
        expected = state.team.current_hp.at[player, state.active_index[player, 0], 0].set(19)
        assert jnp.array_equal(batched[player], expected)
