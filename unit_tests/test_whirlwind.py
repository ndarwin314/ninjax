"""Whirlwind's random forced-switch behavior."""
import jax
import jax.numpy as jnp
import pytest
from ninjax.enum_types import AbilityEnum as A, WeatherEnum
from ninjax.moves.status import do_status_move
from ninjax.moves.routing import move_hits
from ninjax.switching import prepare_forced_switch, execute_forced_switch
from ninjax.hit import HitOutcome
from ninjax.team_builder import make_move
from ninjax.side.updates import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move
from ninjax.battle import ActionType, Battle, BattleAction


@pytest.fixture(scope='module')
def compiled_hit_dispatch():
    """Reuse one hit dispatcher for both players and all forced-switch outcomes."""
    return jax.jit(move_hits)


@pytest.mark.parametrize('player', [0, 1])
def test_forced_switch_request_defers_entry_effects(neutral_type, player):
    """Preparation leaves the entrant untouched; commit applies hazards before Drought."""
    key, state = neutral_type
    target = 1-player
    roster = state.roster.at[target].set(jnp.array([True, False, True, False, False, False]))
    team = state.team.replace(current_hp=state.team.current_hp.at[target, 2, 0].set(1),
                              ability=state.team.ability.at[target, 2, 0].set(A.DROUGHT))
    state = state.replace(team=team, roster=roster,
                          stealth_rocks=state.stealth_rocks.at[target, 0].set(True))
    next_key, prepared, request = prepare_forced_switch(key, state, jnp.int32(player))
    assert request.enabled and request.player == target and request.slot == 2
    assert prepared.active_index[target, 0] == 0
    assert prepared.team.current_hp[target, 2, 0] == 1
    assert prepared.events.count == state.events.count
    assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(jax.random.split(key, 3)[0]))
    out = execute_forced_switch(prepared, request)
    assert out.active_index[target, 0] == 2
    assert out.active.current_hp[target, 0] == 0
    assert out.weather.weather == WeatherEnum.NONE


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('case', ['success', 'protected', 'immune', 'no_bench', 'anchored', 'wind_rider'])
def test_hit_dispatch_forced_switch_outcomes(neutral_type, compiled_hit_dispatch, player, case):
    """Only an unblocked status hit switches; blocked paths preserve key consumption."""

    key, state = neutral_type
    target = 1-player
    state = equip_move(state, 'Whirlwind', player)
    roster = state.roster.at[target].set(jnp.array([True, False, True, False, False, False]))
    state = state.replace(roster=roster)
    if case == 'protected':
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(protected=v.protected.at[target, 0].set(True)))
    if case == 'no_bench':
        state = state.replace(roster=roster.at[target, 2].set(False))
    if case in ('anchored', 'wind_rider', 'immune'):
        ability = {'anchored': A.GUARD_DOG, 'wind_rider': A.WIND_RIDER, 'immune': A.GOOD_AS_GOLD}[case]
        state = update_active(state, target, state.active[target].replace(ability=jnp.array([ability])))
    next_key, out = compiled_hit_dispatch(key, state, jnp.int32(player), jnp.int32(0))
    # Whirlwind bypasses Protect in the pinned move metadata.
    switches = case in ('success', 'protected')
    assert out.active_index[target, 0] == (2 if switches else 0)
    expected_key = jax.random.split(key, 3)[0] if switches else key
    assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(expected_key))
    expected = {'success': HitOutcome.HIT, 'protected': HitOutcome.HIT, 'immune': HitOutcome.IMMUNE,
                'no_bench': HitOutcome.FAILED, 'anchored': HitOutcome.HIT,
                'wind_rider': HitOutcome.FAILED}[case]
    assert out.hit_results[0].outcome == expected
    if case in ('anchored', 'wind_rider'):
        assert out.known_abilities[target, 0] == ability
    assert out.boosts.normal_boosts[target, 1] == (1 if case == 'wind_rider' else 0)


def test_whirlwind_chooses_legal_replacement_and_resets(neutral_type):
    """Only selected living bench slots are eligible and volatiles reset."""
    key, state = neutral_type
    roster = state.roster.at[1].set(jnp.array([True, False, True, False, False, False]))
    v = state.volatile_status
    state = state.replace(roster=roster, volatile_status=v.replace(taunt_turns=v.taunt_turns.at[1, 0].set(3)))
    next_key, out = do_status_move(key, state, 0, make_move('Whirlwind'), 0, 0)
    assert out.active_index[1, 0] == 2
    assert out.volatile_status.taunt_turns[1, 0] == 0
    assert not jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))


@pytest.mark.parametrize('ability', [A.SUCTION_CUPS, A.GUARD_DOG, A.WIND_RIDER])
def test_forced_switch_abilities(neutral_type, ability):
    """Represented drag and wind immunities keep the current target active."""
    key, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    _, out = do_status_move(key, state, 0, make_move('Whirlwind'), 0, 0)
    assert out.active_index[1, 0] == state.active_index[1, 0]
    assert out.hit_results[0].outcome == (HitOutcome.FAILED if ability == A.WIND_RIDER else HitOutcome.HIT)
    if ability == A.WIND_RIDER:
        assert out.boosts.normal_boosts[1, 1] == 1


def test_whirlwind_no_bench_fails(neutral_type):
    """No legal replacement leaves the target and key untouched."""
    key, state = neutral_type
    state = state.replace(roster=state.roster.at[1, 1:].set(False))
    next_key, out = do_status_move(key, state, 0, make_move('Whirlwind'), 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.FAILED
    assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))


def test_whirlwind_cancels_displaced_pending_action(neutral_type):
    """A forced replacement cannot execute its predecessor's queued action."""


    key, state = neutral_type
    state = equip_move(equip_move(state, 'Whirlwind', 0), 'Whirlwind', 1)
    for player, speed in ((0, 200), (1, 30)):
        mon = state.active[player]
        state = update_active(state, player, mon.replace(stat_table=mon.stat_table.replace(base_stats=mon.stat_table.base_stats.at[5].set(speed))))
    env = Battle(state.team)
    episode, _ = env.reset(key)
    roster = jnp.tile(jnp.array([[True, True, False, False, False, False]]), (2, 1))
    episode = episode.replace(selected=roster, battle=episode.battle.replace(roster=roster))
    action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.ATTACK))
    out, ts = env.step(episode, action)
    assert not ts.extras['invalid_action']
    assert out.battle.active_index[0, 0] == 0 and out.battle.active_index[1, 0] == 1
    assert out.battle.team.moves.current_pp[1, 0, 0, 0] == episode.battle.team.moves.current_pp[1, 0, 0, 0]
