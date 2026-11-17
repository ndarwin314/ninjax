"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_active_updates import (
    jax,
    jnp,
    neutral_type,
    pytest,
    update_active_fields,
)


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
