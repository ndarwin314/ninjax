"""Singles locked move continuations and PP behavior."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_move
from ninjax.enums.moves import LockedMoveKind
from ninjax.side.updates import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('name, kind', [
    ('Outrage', LockedMoveKind.RAMPAGE),
    ('Thrash', LockedMoveKind.RAMPAGE),
    ('Petal Dance', LockedMoveKind.RAMPAGE),
    ('Rollout', LockedMoveKind.ROLLOUT),
    ('Uproar', LockedMoveKind.UPROAR),
])
def test_locked_move_starts_and_continues_without_pp(neutral_type, name, kind):
    """A successful first hit starts a forced continuation with no extra PP."""
    key, state = neutral_type
    state = equip_move(state, name)
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(
        accuracy=mon.moves.accuracy.at[0, 0].set(1.))))
    initial_pp = state.active.moves.current_pp[0, 0, 0]
    step = jax.jit(step_move)
    key, state = step(key, state, 0, 0, False, False)
    assert state.volatile_status.continuation.locked.kind[0, 0] == kind
    assert state.active.moves.current_pp[0, 0, 0] == initial_pp - 1
    _, state = step(key, state, 0, 0, False, True)
    assert state.active.moves.current_pp[0, 0, 0] == initial_pp - 1


def test_rollout_miss_clears_lock(neutral_type):
    """A missed Rollout attempt does not retain the continuation."""
    key, state = neutral_type
    state = equip_move(state, 'Rollout')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(
        accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    _, state = jax.jit(step_move)(key, state, 0, 0, False, False)
    assert state.volatile_status.continuation.locked.move_id[0, 0] == 0
