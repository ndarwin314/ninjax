"""Compiled move callbacks preserve effects outside their mechanical scope."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.field import Terrain, TerrainEnum
from ninjax.moves.effects import after_damage
from ninjax.side.field_conditions import field_status
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.fixture(scope='module')
def compiled_after_damage():
    """Reuse one compiled callback with dynamic move metadata and player."""
    return jax.jit(after_damage)


@pytest.fixture(scope='module')
def compiled_field_status():
    """Reuse one compiled field callback across move identifiers."""
    return jax.jit(field_status)


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('name', ['Clear Smog', 'Ice Spinner', 'Rapid Spin', 'Tackle'])
def test_hit_callback_scope(neutral_type, compiled_after_damage, player, name):
    """Stage, terrain and hazard callbacks change only their intended state."""
    _, state = neutral_type
    state = state.replace(
        terrain=Terrain(TerrainEnum.GRASSY, 5),
        spikes=jnp.full((2, 1), 2, jnp.int32),
        boosts=state.boosts.replace(normal_boosts=jnp.ones_like(state.boosts.normal_boosts),
                                    acc_boosts=jnp.ones_like(state.boosts.acc_boosts)))
    out = compiled_after_damage(state, jnp.int32(player), make_move(name))
    expected_normal = state.boosts.normal_boosts
    expected_accuracy = state.boosts.acc_boosts
    expected_spikes = state.spikes
    if name == 'Clear Smog':
        expected_normal = expected_normal.at[1-player].set(0)
        expected_accuracy = expected_accuracy.at[1-player].set(0)
    if name == 'Rapid Spin':
        expected_spikes = expected_spikes.at[player].set(0)
        expected_normal = expected_normal.at[player, 5].add(1)
    assert jnp.array_equal(out.boosts.normal_boosts, expected_normal)
    assert jnp.array_equal(out.boosts.acc_boosts, expected_accuracy)
    assert jnp.array_equal(out.spikes, expected_spikes)
    assert out.terrain.terrain == (TerrainEnum.NONE if name == 'Ice Spinner' else TerrainEnum.GRASSY)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)


@pytest.mark.parametrize('player', [0, 1])
def test_haze_preserves_field_conditions(neutral_type, compiled_field_status, player):
    """Haze clears both sides' ordinary and accuracy stages without clearing hazards."""
    _, state = neutral_type
    state = state.replace(
        terrain=Terrain(TerrainEnum.PSYCHIC, 5),
        spikes=jnp.full((2, 1), 2, jnp.int32),
        boosts=state.boosts.replace(normal_boosts=jnp.ones_like(state.boosts.normal_boosts),
                                    acc_boosts=jnp.ones_like(state.boosts.acc_boosts)))
    out = compiled_field_status(state, jnp.int32(player), make_move('Haze'))
    assert jnp.all(out.boosts.normal_boosts == 0)
    assert jnp.all(out.boosts.acc_boosts == 0)
    assert jnp.array_equal(out.spikes, state.spikes)
    assert out.terrain.terrain == TerrainEnum.PSYCHIC


def test_unknown_field_callback_preserves_state(neutral_type, compiled_field_status):
    """An unmatched identifier must select the fallback instead of Court Change."""
    _, state = neutral_type
    state = state.replace(spikes=state.spikes.at[0, 0].set(2))
    move = make_move('Tackle').replace(identifier=jnp.array([999999], jnp.int32))
    out = compiled_field_status(state, jnp.int32(0), move)
    assert all(jnp.array_equal(before, after)
               for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(out)))
