"""Charge and recharge lifecycle with forced continued actions."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle import ActionType, Battle, BattleAction, Phase, step_action, step_move
from ninjax.enum_types import AbilityEnum as A, MoveResult, Status, Weather, WeatherEnum as W
from ninjax.game_logic import resolve_damaging_hit, resolve_multi_hit, move_hits
from ninjax.prototype import make_move
from ninjax.side import clear_volatile_status, update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def continued():
    """Build a scalar forced action for a single active player."""
    return BattleAction(jnp.int32(0), jnp.int32(ActionType.CONTINUED))


def test_electro_shot_charge_and_last_pp(neutral_type):
    """Charge boosts once, forces continuation, and spends only the first PP."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(current_pp=mon.moves.current_pp.at[0, 0].set(1))))
    key, charged = step_move(key, state, 0, 0, False)
    assert charged.volatile_status.charging_move_id[0, 0] == 905
    assert charged.boosts.normal_boosts[0, 3] == 1
    assert charged.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert charged.active.moves.current_pp[0, 0, 0] == 0
    assert not jnp.any(charged.hit_results.data[:, 4])
    env = Battle(state.team)
    episode, _ = env.reset(key)
    episode = episode.replace(battle=charged)
    mask = env.legal_action_mask(episode)
    assert jnp.array_equal(mask[0], jnp.arange(15) == 5)
    _, released = step_action(key, charged, continued(), 0)
    assert released.active.current_hp[1, 0] < charged.active.current_hp[1, 0]
    assert released.active.moves.current_pp[0, 0, 0] == 0
    assert released.boosts.normal_boosts[0, 3] == 1
    assert released.volatile_status.charging_move_id[0, 0] == 0


@pytest.mark.parametrize('suppressed', [False, True])
def test_rain_skip_uses_effective_weather(neutral_type, suppressed):
    """Rain boosts and attacks immediately unless Cloud Nine suppresses it."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot').replace(weather=Weather(W.RAIN, 5))
    if suppressed:
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.CLOUD_NINE])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.boosts.normal_boosts[0, 3] == 1
    assert bool(out.volatile_status.charging_move_id[0, 0] != 0) == suppressed
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) != suppressed


@pytest.mark.parametrize('rain', [False, True])
def test_charge_preparation_gates_accuracy(neutral_type, rain):
    """A dry charge skips accuracy; immediate rain execution draws it once."""
    from ninjax.game_logic import dispatch_selected_move
    from ninjax.hit import HitOutcome
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot').replace(weather=Weather(W.RAIN if rain else W.NONE, 5))
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(
        accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    out_key, out = dispatch_selected_move(key, state, 0, 0)
    expected_key = jax.random.split(key)[0] if rain else key
    assert jnp.array_equal(jax.random.key_data(out_key), jax.random.key_data(expected_key))
    assert out.boosts.normal_boosts[0, 3] == 1
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert jnp.array_equal(out.active.moves.current_pp, state.active.moves.current_pp)
    assert (out.volatile_status.charging_move_id[0, 0] == 905) != rain
    assert out.hit_results[0].outcome == (HitOutcome.MISS if rain else HitOutcome.UNUSED)


def test_charge_interruption_and_switch_cleanup(neutral_type):
    """Flinch cancels a release; switching clears both forced-turn states."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot')
    key, state = step_move(key, state, 0, 0, False)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True)))
    _, out = step_action(key, state, continued(), 0)
    assert out.volatile_status.charging_move_id[0, 0] == 0
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0]
    cleared = clear_volatile_status(state.replace(volatile_status=v.replace(recharging=v.recharging.at[0, 0].set(True))), 0)
    assert not cleared.volatile_status.recharging[0, 0]
    assert cleared.volatile_status.charging_move_id[0, 0] == 0


def test_hyper_beam_recharge_precedes_status_checks(neutral_type):
    """A successful beam forces recharge without ticking sleep or confusion."""
    key, state = neutral_type
    _, hit = resolve_multi_hit(key, state, 0, make_move('Hyper Beam'))
    assert hit.volatile_status.recharging[0, 0]
    hit = update_active(hit, 0, hit.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    v = hit.volatile_status
    hit = hit.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[0, 0].set(3)))
    _, out = step_action(key, hit, continued(), 0)
    assert not out.volatile_status.recharging[0, 0]
    assert out.volatile_status.move_result[0, 0] == MoveResult.SKIPPED
    assert out.active.sleep_counter[0, 0] == 2
    assert out.volatile_status.confusion_turns[0, 0] == 3
    assert jnp.array_equal(out.active.current_hp, hit.active.current_hp)


def test_blocked_hyper_beam_needs_no_recharge(neutral_type):
    """Protect blocks the beam before its recharge effect is applied."""
    key, state = neutral_type
    state = equip_move(state, 'Hyper Beam')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    _, out = move_hits(key, state, 0, 0)
    assert not out.volatile_status.recharging[0, 0]


def test_missed_beam_and_substitute_recharge(neutral_type):
    """A missed beam needs no recharge, but a Substitute hit does."""
    key, state = neutral_type
    state = equip_move(state, 'Hyper Beam')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    _, missed = step_move(key, state, 0, 0, False)
    assert not missed.volatile_status.recharging[0, 0]
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(1)))
    _, hit = resolve_multi_hit(key, state, 0, make_move('Hyper Beam'))
    assert hit.volatile_status.recharging[0, 0]
    assert hit.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_forced_mask_transforms(neutral_type):
    """Charge/recharge masks trace under batching and scanning without compilation."""
    key, state = neutral_type
    env = Battle(state.team)
    episode, _ = env.reset(key)
    v = state.volatile_status
    episode = episode.replace(battle=state.replace(volatile_status=v.replace(recharging=v.recharging.at[0, 0].set(True))))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), episode)
    shape = jax.eval_shape(jax.vmap(env.legal_action_mask), batch)
    assert shape.shape == (2, 2, 15)
    def scan(s):
        """Keep fixed action-mask output over a scan of forced turns."""
        return jax.lax.scan(lambda s, _: (s, env.legal_action_mask(s)), s, None, length=2)[1]
    assert jax.eval_shape(scan, episode).shape == (2, 2, 15)


def test_environment_continues_charge_and_recharge(neutral_type):
    """The scheduler dispatches both continued kinds in a simultaneous turn."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot')
    env = Battle(state.team)
    episode, _ = env.reset(key)
    state = episode.battle
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(current_pp=mon.moves.current_pp.at[0, 0].set(0))))
    state = update_active(state, 1, state.active[1].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(charging_move_id=v.charging_move_id.at[0, 0].set(905), recharging=v.recharging.at[1, 0].set(True)))
    episode = episode.replace(battle=state)
    action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.CONTINUED))
    out, ts = env.step(episode, action)
    assert not ts.extras['invalid_action']
    assert out.turn == 1
    assert not jnp.any(out.battle.volatile_status.recharging)
    assert not jnp.any(out.battle.volatile_status.charging_move_id)
    assert out.battle.active.moves.current_pp[0, 0, 0] == 0
    assert out.battle.active.sleep_counter[1, 0] == 2
    env.observation_spec.validate(ts.observation)
