"""Whirlwind's random forced-switch behavior."""
import jax
import jax.numpy as jnp
import pytest
from ninjax.enum_types import AbilityEnum as A
from ninjax.game_logic import do_status_move
from ninjax.hit import HitOutcome
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type


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
    from ninjax.battle import ActionType, Battle, BattleAction
    from unit_tests.test_curated_moves import equip_move
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
