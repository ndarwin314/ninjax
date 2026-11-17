"""Checkpoint coverage for local forced-switch requests under transformations."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enum_types import AbilityEnum
from ninjax.switching import prepare_forced_switch, execute_forced_switch
from unit_tests.test_battle import neutral_type


def prepared_switch(key, state):
    """Carry the explicit battle key through switch preparation and commitment."""
    key, state, request = prepare_forced_switch(key, state, jnp.int32(0))
    return key, execute_forced_switch(state, request)


@pytest.mark.jit
def test_forced_switch_requests_vmap_and_scan(neutral_type):
    """Mixed blocked/successful requests batch and repeated requests scan."""
    key, state = neutral_type
    state = state.replace(roster=state.roster.at[1].set(jnp.array([True, True, False, False, False, False])))
    blocked = state.replace(team=state.team.replace(
        ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.GUARD_DOG)))
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), blocked, state)
    keys = jax.random.split(key, 2)
    next_keys, out = jax.jit(jax.vmap(prepared_switch))(keys, batch)
    assert jnp.array_equal(out.active_index[:, 1, 0], jnp.array([0, 1]))
    assert jnp.array_equal(jax.random.key_data(next_keys[0]), jax.random.key_data(keys[0]))

    def step(carry, unused):
        """Alternate between the two eligible team slots with explicit state."""
        return prepared_switch(*carry), None

    (_, out), _ = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=2))((key, state))
    assert out.active_index[1, 0] == 0
