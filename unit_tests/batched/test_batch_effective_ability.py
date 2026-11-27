"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_effective_ability import read_abilities


@pytest.mark.jit
def test_ability_resolution_batches(neutral_type):
    """The ability property has a stable player axis under vmap and scan."""
    _, state = neutral_type
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.vmap(read_abilities)(batched).shape == (2, 2, 1)
    _, values = jax.lax.scan(lambda s, _: (s, read_abilities(s)), state, None, length=2)
    assert values.shape == (2, 2, 1)


@pytest.mark.jit
def test_dynamic_active_gather_transforms(neutral_type):
    """Shared gathers retain slots and effective abilities under jit/vmap/scan."""
    _, state = neutral_type
    team = state.team.replace(
        current_hp=jnp.arange(100, 112, dtype=jnp.int32).reshape(2, 6, 1),
        ability=jnp.arange(12, dtype=jnp.int32).reshape(2, 6, 1),
        moves=state.team.moves.replace(identifier=jnp.arange(48, dtype=jnp.int32).reshape(2, 6, 4, 1)))
    state = state.replace(team=team)
    slots = jnp.array([[2, 4], [5, 1]], jnp.int32)
    def read(indices):
        """Read only selected leaves from a dynamic active snapshot."""
        current = state.replace(active_index=indices[:, None])
        active = current.active
        return active.current_hp, active.moves.identifier, current.ability
    expected = jax.vmap(lambda indices: (
        team.current_hp[jnp.arange(2), indices],
        team.moves.identifier[jnp.arange(2), indices],
        team.ability[jnp.arange(2), indices]))(slots)
    actual = jax.jit(jax.vmap(read))(slots)
    _, scanned = jax.jit(lambda indices: jax.lax.scan(
        lambda carry, row: (carry, read(row)), None, indices))(slots)
    for wanted, batched, sequence in zip(expected, actual, scanned):
        assert jnp.array_equal(wanted, batched)
        assert jnp.array_equal(wanted, sequence)


@pytest.mark.jit
def test_effective_immunities_transform(neutral_type):
    """Effective status, grounding and move immunities compile under vmap/scan."""

    _, state = neutral_type
    move = make_move('Earthquake')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[1, 0].set(A.LEVITATE))))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    def check(s):
        """Check Ground immunity from the active effective ability."""
        return s.is_immune_to_move(1, move), s.is_grounded(1)
    immunity, grounded = jax.jit(jax.vmap(check))(batch)
    assert jnp.all(immunity) and not jnp.any(grounded)
    def step(s, _):
        """Carry the same immutable battle through repeated checks."""
        return s, check(s)
    _, (immunity, grounded) = jax.jit(lambda s: jax.lax.scan(step, s, None, length=2))(state)
    assert jnp.all(immunity) and not jnp.any(grounded)
