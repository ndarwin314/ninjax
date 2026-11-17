"""Environment transformation checks for the extracted side-state package."""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.types import BattleParams
from unit_tests.test_battle import neutral_type


def _step_episode(episode, unused, *, env):
    """Carry the extracted state through one simultaneous move decision."""
    next_episode, timestep = env.step(episode, jnp.zeros(2, jnp.int32))
    return next_episode, timestep.observation


def _rollout(key, *, env):
    """Reset and scan an episode using module-level callbacks."""
    episode, _ = env.reset(key)
    final, observations = jax.lax.scan(
        partial(_step_episode, env=env), episode, None, length=2,
    )
    # Keep the full state in the scan carry, but inspect array outputs rather
    # than rebuilding Pokémon constructors from eval_shape metadata leaves.
    return final.battle.team.current_hp, observations


@pytest.mark.jit
def test_side_state_reset_compiles_and_batches(neutral_type):
    """State factories create correctly shaped leaves under compiled resets."""
    key, state = neutral_type
    env = Battle(state.team, BattleParams(max_steps_in_episode=1))
    episodes, timesteps = jax.jit(jax.vmap(env.reset))(jax.random.split(key, 2))
    assert episodes.battle.team.current_hp.shape == (2, 2, 6, 1)
    assert episodes.battle.volatile_status.ability_override.shape == (2, 2, 1)
    assert jnp.all(episodes.battle.volatile_status.ability_override == -1)
    assert not jnp.any(timesteps.last())


@pytest.mark.jit
def test_side_state_rollout_traces_under_jit_vmap_scan(neutral_type):
    """Trace full environment composition without compiling a large rollout."""
    key, state = neutral_type
    env = Battle(state.team, BattleParams(max_steps_in_episode=1))
    rollout = jax.jit(jax.vmap(partial(_rollout, env=env)))
    final_hp, observations = jax.eval_shape(rollout, jax.random.split(key, 2))
    assert final_hp.shape == (2, 2, 6, 1)
    assert observations.team['species'].shape == (2, 2, 2, 2, 6)
