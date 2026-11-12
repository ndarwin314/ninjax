"""Shared before-action sleep, paralysis and flinch handling."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.battle import before_move
from ninjax.enum_types import Status
from ninjax.side import update_active


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
        expected_status == Status.PARALYZE and float(jax.random.uniform(subkey)) <= .25)
    out_key, out, interrupted = before_move(key, state, 0)
    assert jnp.array_equal(jax.random.key_data(out_key), jax.random.key_data(expected_key))
    assert out.active.status[0, 0] == expected_status
    assert out.active.sleep_counter[0, 0] == max(0, counter - 1)
    assert out.active.is_terastallized[0, 0]
    assert bool(interrupted) == bool(expected_interrupted)


@pytest.mark.jit
def test_before_move_jit_vmap_scan(neutral_type):
    """The shared gate works with explicit keys under JAX transformations."""
    key, state = neutral_type
    keys = jax.random.split(key, 2)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    _, _, blocked = jax.jit(jax.vmap(lambda k, s: before_move(k, s, 0)))(keys, batch)
    assert not jnp.any(blocked)
    def step(carry, _):
        """Carry both state and randomness through successive attempts."""
        key, state = carry
        key, state, blocked = before_move(key, state, 0)
        return (key, state), blocked
    _, blocked = jax.jit(lambda k, s: jax.lax.scan(step, (k, s), None, length=2))(key, state)
    assert not jnp.any(blocked)
