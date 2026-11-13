"""Sleep Talk calls an eligible move without consuming its PP."""
import jax
import jax.numpy as jnp
import pytest
from ninjax.battle import step_move
from ninjax.enum_types import AbilityEnum as A, MoveResult, Status
from ninjax.hit import HitOutcome
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def equip_second(state, name, pp=None, disabled=False):
    """Install one callable move in the second slot."""
    mon = state.active[0]
    move = make_move(name)
    if pp is not None:
        move = move.replace(current_pp=jnp.array([pp]))
    move = move.replace(disabled=jnp.array([disabled]))
    moves = jax.tree.map(lambda bank, value: bank.at[1].set(value), mon.moves, move)
    return update_active(state, 0, mon.replace(moves=moves))


def asleep(state, counter=3):
    """Set a sleeping user with a known sleep counter."""
    return update_active(state, 0, state.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([counter])))


def test_sleep_talk_calls_move_without_spending_called_pp(neutral_type):
    """Sleep Talk can call a zero-PP disabled move and spends only its own PP."""
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Dragon Pulse', 0, True))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1
    assert out.active.moves.current_pp[0, 1, 0] == 0
    assert out.active.sleep_counter[0, 0] == 2
    assert out.volatile_status.called_move_id[0, 0] == make_move('Dragon Pulse').identifier[0]
    assert out.volatile_status.move_result[0, 0] == MoveResult.SUCCEEDED


def test_sleep_talk_wakes_or_no_eligible_move(neutral_type):
    """Awake users and sleepers without callable moves fail after paying PP."""
    key, state = neutral_type
    state = equip_move(state, 'Sleep Talk')
    for candidate in (state, asleep(state), asleep(equip_second(state, 'Electro Shot'), 3), asleep(equip_second(state, 'Dragon Pulse'), 1)):
        _, out = step_move(key, candidate, 0, 0, False)
        assert out.hit_results[0].outcome == HitOutcome.FAILED
        assert out.active.moves.current_pp[0, 0, 0] == candidate.active.moves.current_pp[0, 0, 0] - 1


def test_comatose_can_use_sleep_talk(neutral_type):
    """Comatose permits Sleep Talk even without a sleep status."""
    key, state = neutral_type
    state = equip_second(equip_move(state, 'Sleep Talk'), 'Calm Mind')
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.COMATOSE])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.boosts.normal_boosts[0, 3] == 1 and out.boosts.normal_boosts[0, 4] == 1


def test_sleep_talk_succeeds_when_called_move_fails(neutral_type):
    """A failed called move does not make Sleep Talk itself fail."""
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Rest'))
    _, out = step_move(key, state, 0, 0, False)
    assert out.hit_results[0].outcome == HitOutcome.FAILED
    assert out.volatile_status.move_result[0, 0] == MoveResult.SUCCEEDED
    assert out.active.moves.current_pp[0, 1, 0] == state.active.moves.current_pp[0, 1, 0]


def test_sleep_talk_beat_up_remains_asleep_between_hits(neutral_type):
    """Sleep Talk permits every Beat Up hit despite the caller's sleep status."""
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Beat Up'))
    state = state.replace(roster=state.roster.at[0, 2:].set(False))
    _, out = step_move(key, state, 0, 0, False)
    assert jnp.sum(out.hit_results.data[:, 4] != 0) == 2
    assert out.active.status[0, 0] == Status.SLEEP


def test_sleep_talk_reveals_called_move_in_environment(neutral_type):
    """Called moves become publicly known even though their PP is unchanged."""
    from ninjax.battle import ActionType, Battle, BattleAction
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Dragon Pulse'))
    env = Battle(state.team)
    episode, _ = env.reset(key)
    action = BattleAction(jnp.array([0, 0]), jnp.full(2, ActionType.ATTACK))
    out, ts = env.step(episode, action)
    assert not ts.extras['invalid_action']
    assert out.revealed_moves[0, 0, 0] and out.revealed_moves[0, 0, 1]
    assert out.battle.team.moves.current_pp[0, 0, 1, 0] == episode.battle.team.moves.current_pp[0, 0, 1, 0]


def test_sleep_talk_libero_uses_called_move_type(neutral_type):
    """Libero ignores the calling move and changes to the called move's type."""
    from ninjax.enum_types import Type
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Dragon Pulse'))
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.LIBERO]), type_list=jnp.array([Type.ROCK, Type.NONE])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.type_list[0, 0] == Type.DRAGON
    assert out.volatile_status.libero_used[0, 0]


@pytest.mark.parametrize('old_override', [False, True])
def test_called_crash_restores_context(neutral_type, old_override):
    """A called miss applies crash once, then restores Sleep Talk's context."""
    from ninjax.game_logic import execute_sleep_talk
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'High Jump Kick', 0))
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(
        accuracy=mon.moves.accuracy.at[1, 0].set(0.))))
    v = state.volatile_status
    state = state.replace(event_move=jnp.int32(214), volatile_status=v.replace(
        encore_overridden=v.encore_overridden.at[0, 0].set(old_override)))
    _, out = execute_sleep_talk(key, state, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.MISS
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - state.active.max_hp[0] // 2
    assert jnp.array_equal(out.active.moves.current_pp, state.active.moves.current_pp)
    assert out.event_move == 214
    assert not out.volatile_status.sleep_talk_active[0, 0]
    assert out.volatile_status.encore_overridden[0, 0] == old_override
    assert out.volatile_status.sleep_talk_succeeded[0, 0]


@pytest.mark.parametrize('damp', [False, True])
def test_called_explosion_prepares_and_cleans_up(neutral_type, damp):
    """Called Explosion respects Damp, saved HP cleanup and parent PP payment."""
    key, state = neutral_type
    state = asleep(equip_second(equip_move(state, 'Sleep Talk'), 'Explosion'))
    if damp:
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.DAMP])))
    _, out = step_move(key, state, 0, 0, False)
    assert bool(out.active.current_hp[0, 0] == 0) != damp
    assert out.volatile_status.explosion_hp[0, 0] == 0
    assert not out.volatile_status.sleep_talk_active[0, 0]
    assert out.event_move == 214
    assert out.volatile_status.move_result[0, 0] == MoveResult.SUCCEEDED
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1
    assert out.active.moves.current_pp[0, 1, 0] == state.active.moves.current_pp[0, 1, 0]


def test_failed_call_does_not_enter_hit_pipeline(neutral_type):
    """An awake Sleep Talk failure consumes no selection or accuracy key."""
    from ninjax.game_logic import execute_sleep_talk
    key, state = neutral_type
    state = equip_second(equip_move(state, 'Sleep Talk'), 'Dragon Pulse')
    out_key, out = execute_sleep_talk(key, state, 0, 0)
    assert jnp.array_equal(jax.random.key_data(out_key), jax.random.key_data(key))
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert out.hit_results[0].outcome == HitOutcome.FAILED


@pytest.mark.jit
def test_compiled_called_move_branches_match_eager(neutral_type):
    """One executable handles called damage, self-faint, status and failed calls."""
    from ninjax.game_logic import move_used
    key, initial = neutral_type
    def use(k, state):
        """Keep team metadata dynamic while executing the Sleep Talk slot."""
        return move_used(k, state, jnp.int32(0), jnp.int32(0))
    compiled = jax.jit(use)
    for name, sleeping in (('High Jump Kick', True), ('Explosion', True),
                           ('Calm Mind', True), ('Rest', True), ('Calm Mind', False)):
        state = equip_second(equip_move(initial, 'Sleep Talk'), name)
        if sleeping:
            state = asleep(state)
        if name == 'High Jump Kick':
            mon = state.active[0]
            state = update_active(state, 0, mon.replace(moves=mon.moves.replace(
                accuracy=mon.moves.accuracy.at[1, 0].set(0.))))
        with jax.disable_jit():
            expected = use(key, state)
        actual = compiled(key, state)
        for wanted, result in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            if not hasattr(wanted, 'dtype'):
                wanted = jnp.asarray(wanted)
            if jax.dtypes.issubdtype(result.dtype, jax.dtypes.prng_key):
                wanted, result = jax.random.key_data(wanted), jax.random.key_data(result)
            assert result.shape == wanted.shape and result.dtype == wanted.dtype
            if jnp.issubdtype(result.dtype, jnp.inexact):
                assert jnp.allclose(result, wanted, rtol=1e-5, atol=1e-5)
            else:
                assert jnp.array_equal(result, wanted)
