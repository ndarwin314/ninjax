"""Represented field conditions and their shared accuracy and status effects."""
import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import Status
from ninjax.moves.properties import lock_on_matches, move_accuracy
from ninjax.records.events import Cause
from ninjax.side.field_conditions import field_condition_effect
from ninjax.side.status_effects import set_status
from ninjax.side.updates import update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def test_gravity_applies_accuracy_and_cancels_airborne(neutral_type):
    """Gravity grounds a charging Fly user and raises ordinary accuracy."""
    _, state = neutral_type
    v = state.volatile_status
    continuation = v.continuation.replace(
        charging_move_id=v.continuation.charging_move_id.at[1, 0].set(19),
        semi_invulnerability=v.continuation.semi_invulnerability.at[1, 0].set(1))
    state = state.replace(volatile_status=v.replace(continuation=continuation))
    before = move_accuracy(state, 0, make_move('Thunder'))
    state, success = field_condition_effect(state, 0, make_move('Gravity'))
    assert success and state.gravity_duration == 5
    assert state.volatile_status.continuation.charging_move_id[1, 0] == 0
    assert move_accuracy(state, 0, make_move('Thunder')) > before
    _, success = field_condition_effect(state, 1, make_move('Gravity'))
    assert not success


def test_rooms_toggle_without_extending_duration(neutral_type):
    """Magic Room and Wonder Room each stop on a repeated use."""
    _, state = neutral_type
    for name, field in [('Magic Room', 'magic_room_duration'), ('Wonder Room', 'wonder_room_duration')]:
        state, success = field_condition_effect(state, 0, make_move(name))
        assert success and getattr(state, field) == 5
        state, success = field_condition_effect(state, 1, make_move(name))
        assert success and getattr(state, field) == 0


def test_lock_on_follows_target_identity(neutral_type):
    """Lock-On guarantees accuracy for the chosen target until it switches."""
    _, state = neutral_type
    state, success = field_condition_effect(state, 0, make_move('Lock-On'))
    assert success and lock_on_matches(state, 0, 1)
    assert move_accuracy(state, 0, make_move('Thunder')) == 1
    switched = state.replace(active_index=state.active_index.at[1, 0].set(1))
    assert not lock_on_matches(switched, 0, 1)
    assert move_accuracy(switched, 0, make_move('Thunder')) < 1


def test_safeguard_blocks_opponent_but_not_infiltrator(neutral_type):
    """Safeguard blocks opposing status and yields to Infiltrator."""
    key, state = neutral_type
    state, _ = field_condition_effect(state, 1, make_move('Safeguard'))
    state = state.replace(event_context=state.event_context.replace(
        source=jnp.int32(0), cause=jnp.int32(Cause.MOVE)))
    _, blocked = set_status(key, state, 1, Status.BURN)
    assert blocked.active.status[1, 0] == Status.NONE
    state = update_active_fields(state, 0, ability=jnp.array([AbilityEnum.INFILTRATOR]))
    _, burned = set_status(key, state, 1, Status.BURN)
    assert burned.active.status[1, 0] == Status.BURN
