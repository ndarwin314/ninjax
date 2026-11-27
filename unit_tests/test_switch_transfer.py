"""Switch transfer policies for represented active conditions."""

import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.switching import reset_switch_effects, swap_out
from ninjax.moves.routing import move_hits
from ninjax.enums.field import WeatherEnum
from ninjax.records.hits import HitOutcome
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('policy,substitute,confusion,taunt,stage,perish', [
    (0, 0, 0, 0, 0, -1),
    (1, 12, 3, 2, 2, 3),
    (2, 12, 0, 0, 0, -1),
])
def test_switch_transfer(neutral_type, policy, substitute, confusion, taunt, stage, perish):
    """Only passable effects and stages survive their matching policy."""
    _, state = neutral_type
    v = state.volatile_status
    v = v.replace(
        substitute_hp=v.substitute_hp.at[0, 0].set(12),
        confusion_turns=v.confusion_turns.at[0, 0].set(3),
        restrictions=v.restrictions.replace(
            taunt_turns=v.restrictions.taunt_turns.at[0, 0].set(2),
            disable_turns=v.restrictions.disable_turns.at[0, 0].set(2)))
    boosts = state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 1].set(2))
    state = state.replace(volatile_status=v, boosts=boosts,
                          perish_turns=state.perish_turns.at[0].set(3))
    out = jax.jit(reset_switch_effects)(state, 0, jnp.int32(policy))
    assert out.volatile_status.substitute_hp[0, 0] == substitute
    assert out.volatile_status.confusion_turns[0, 0] == confusion
    assert out.volatile_status.restrictions.taunt_turns[0, 0] == taunt
    assert out.volatile_status.restrictions.disable_turns[0, 0] == 0
    assert out.boosts.normal_boosts[0, 1] == stage
    assert out.perish_turns[0, 0] == perish


@pytest.mark.parametrize('name,policy', [('Baton Pass', 1), ('Shed Tail', 2)])
def test_transfer_move_requests_switch(neutral_type, name, policy):
    """The replacement receives the requested transfer policy and Substitute."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = state.replace(boosts=state.boosts.replace(
        normal_boosts=state.boosts.normal_boosts.at[0, 1].set(2)))
    if policy == 1:
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[0, 0].set(12)))
    hp = state.active.current_hp[0, 0]
    maximum = state.active.max_hp[0]
    _, out = move_hits(key, state, 0, 0)
    assert out.switch_requested[0]
    assert out.switch_transfer[0] == policy
    expected = 12 if policy == 1 else maximum // 4
    if policy == 2:
        assert out.active.current_hp[0, 0] == hp - (maximum + 1) // 2
    out = swap_out(key, out, 0, 1, out.switch_transfer[0])
    assert out.volatile_status.substitute_hp[0, 0] == expected
    assert out.boosts.normal_boosts[0, 1] == (2 if policy == 1 else 0)


@pytest.mark.parametrize('name', ['Baton Pass', 'Shed Tail'])
def test_transfer_move_fails_without_bench(neutral_type, name):
    """A failed transfer does not pay HP or create a Substitute."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = state.replace(roster=state.roster.at[0, 1:].set(False))
    hp = state.active.current_hp[0, 0]
    _, out = move_hits(key, state, 0, 0)
    assert not out.switch_requested[0]
    assert out.hit_results[0].outcome == HitOutcome.FAILED
    assert out.active.current_hp[0, 0] == hp
    assert out.volatile_status.substitute_hp[0, 0] == 0


@pytest.mark.parametrize('name', ['Teleport', 'Chilly Reception'])
def test_status_pivot_requests_switch(neutral_type, name):
    """Status pivots request replacement and Chilly Reception sets snow."""
    key, state = neutral_type
    _, out = move_hits(key, equip_move(state, name), 0, 0)
    assert out.switch_requested[0]
    assert out.switch_transfer[0] == 0
    if name == 'Chilly Reception':
        assert out.weather.weather == WeatherEnum.SNOW
