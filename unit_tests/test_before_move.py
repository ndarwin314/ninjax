"""Shared before-action sleep, paralysis and flinch handling."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.interruptions import before_move
from ninjax.enums.pokemon import Status
from ninjax.side.updates import update_active
from unit_tests.test_battle import neutral_type


@pytest.mark.parametrize('status,counter,flinch', [
    (Status.NONE, 0, False), (Status.SLEEP, 2, False), (Status.SLEEP, 1, False),
    (Status.PARALYZE, 0, False), (Status.NONE, 0, True)])
def test_before_move_preserves_status_and_key_semantics(neutral_type, status, counter, flinch):
    """The shared gate advances one key and resolves the existing conditions."""
    key, state = neutral_type
    actor = state.active[0].replace(status=jnp.array([status]), sleep_counter=jnp.array([counter]), is_terastallized=jnp.array([True]))
    state = update_active(state, 0, actor)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(flinch)))
    expected_key, subkey = jax.random.split(key)
    expected_status = Status.NONE if counter == 1 else status
    expected_interrupted = flinch or expected_status == Status.SLEEP or (
        expected_status == Status.PARALYZE and float(jax.random.uniform(subkey)) < .25)
    out_key, out, interrupted = before_move(key, state, 0)
    assert jnp.array_equal(jax.random.key_data(out_key), jax.random.key_data(expected_key))
    assert out.active.status[0, 0] == expected_status
    assert out.active.sleep_counter[0, 0] == max(0, counter - 1)
    assert out.active.is_terastallized[0, 0]
    assert bool(interrupted) == bool(expected_interrupted)
