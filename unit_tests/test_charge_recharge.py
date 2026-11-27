"""Charge and recharge lifecycle with forced continued actions."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_action, step_move
from ninjax.battle.environment import Battle
from ninjax.battle.types import ActionType, BattleAction
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Weather, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import MoveResult, SemiInvulnerability
from ninjax.enums.pokemon import Status
from ninjax.moves.execution import dispatch_selected_move, prepare_charge_move
from ninjax.calculations.damage import compute_base_power
from ninjax.moves.multi_hit import resolve_multi_hit
from ninjax.moves.routing import move_hits
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import clear_volatile_status, update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def continued():
    """Build a scalar forced action for a single active player."""
    return BattleAction(jnp.int32(0), jnp.int32(ActionType.CONTINUED))


@pytest.mark.parametrize('name,weather,herb,ready,boost', [
    ('Solar Beam', W.SUN, True, True, 0),
    ('Solar Blade', W.NONE, True, True, 0),
    ('Solar Beam', W.RAIN, False, False, 0),
    ('Meteor Beam', W.NONE, False, False, 1),
    ('Electro Shot', W.RAIN, False, True, 1),
    ('Freeze Shock', W.NONE, True, True, 0),
])
def test_charge_preparation(neutral_type, name, weather, herb, ready, boost):
    """Weather, herb and initial boosts compose before a charge is stored."""
    key, state = neutral_type
    state = equip_move(state, name).replace(weather=Weather(weather, 5))
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.POWER_HERB if herb else I.NONE])))
    _, out, actual_ready = jax.jit(prepare_charge_move)(key, state, 0, 0)
    assert bool(actual_ready) == ready
    assert out.boosts.normal_boosts[0, 3] == boost
    assert (out.active.item[0, 0] == I.POWER_HERB) == (herb and name == 'Solar Beam' and weather == W.SUN)
    assert (out.volatile_status.continuation.charging_move_id[0, 0] != 0) == (not ready)


@pytest.mark.parametrize('name', ['Solar Beam', 'Solar Blade', 'Meteor Beam', 'Sky Attack', 'Freeze Shock', 'Ice Burn'])
def test_shared_charge_release(neutral_type, name):
    """Each represented charging attack spends PP once and releases next turn."""
    key, state = neutral_type
    state = equip_move(state, name)
    key, charged = step_move(key, state, 0, 0, False)
    assert charged.volatile_status.continuation.charging_move_id[0, 0] == make_move(name).identifier[0]
    assert charged.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1
    assert charged.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    v = charged.volatile_status
    charged = charged.replace(volatile_status=v.replace(continuation=v.continuation.replace(semi_invulnerability=
        v.continuation.semi_invulnerability.at[0, 0].set(SemiInvulnerability.AIRBORNE))))
    _, released = step_action(key, charged, continued(), 0)
    assert released.volatile_status.continuation.charging_move_id[0, 0] == 0
    assert released.volatile_status.continuation.semi_invulnerability[0, 0] == SemiInvulnerability.NONE
    assert released.active.moves.current_pp[0, 0, 0] == charged.active.moves.current_pp[0, 0, 0]
    if name == 'Meteor Beam':
        assert released.boosts.normal_boosts[0, 3] == 1


@pytest.mark.parametrize('name', ['Solar Beam', 'Solar Blade', 'Electro Shot', 'Meteor Beam'])
def test_power_herb_skips_charge(neutral_type, name):
    """Power Herb is consumed once when a charge would otherwise be needed."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.POWER_HERB])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.item[0, 0] == I.NONE
    assert out.volatile_status.continuation.charging_move_id[0, 0] == 0
    if name in ('Meteor Beam', 'Electro Shot'):
        assert out.boosts.normal_boosts[0, 3] == 1


@pytest.mark.parametrize('weather,expected', [(W.NONE, 120), (W.SUN, 120), (W.RAIN, 60), (W.SANDSTORM, 60), (W.SNOW, 60)])
def test_solar_weather_power(neutral_type, weather, expected):
    """Weakening weather modifies solar power at the hit-time power stage."""
    _, state = neutral_type
    move = make_move('Solar Beam')
    assert compute_base_power(state.active[0], state.active[1], move, state.terrain, weather)[0] == expected


def test_sun_preserves_power_herb(neutral_type):
    """Sun makes Solar Beam immediate without consuming the held herb."""
    key, state = neutral_type
    state = equip_move(state, 'Solar Beam').replace(weather=Weather(W.SUN, 5))
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.POWER_HERB])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.item[0, 0] == I.POWER_HERB
    assert out.volatile_status.continuation.charging_move_id[0, 0] == 0


def test_electro_shot_charge_and_last_pp(neutral_type):
    """Charge boosts once, forces continuation, and spends only the first PP."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(current_pp=mon.moves.current_pp.at[0, 0].set(1))))
    key, charged = step_move(key, state, 0, 0, False)
    assert charged.volatile_status.continuation.charging_move_id[0, 0] == 905
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
    assert released.volatile_status.continuation.charging_move_id[0, 0] == 0


@pytest.mark.parametrize('suppressed', [False, True])
def test_rain_skip_uses_effective_weather(neutral_type, suppressed):
    """Rain boosts and attacks immediately unless Cloud Nine suppresses it."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot').replace(weather=Weather(W.RAIN, 5))
    if suppressed:
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.CLOUD_NINE])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.boosts.normal_boosts[0, 3] == 1
    assert bool(out.volatile_status.continuation.charging_move_id[0, 0] != 0) == suppressed
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) != suppressed


@pytest.mark.parametrize('rain', [False, True])
def test_charge_preparation_gates_accuracy(neutral_type, rain):
    """A dry charge skips accuracy; immediate rain execution draws it once."""


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
    assert (out.volatile_status.continuation.charging_move_id[0, 0] == 905) != rain
    assert out.hit_results[0].outcome == (HitOutcome.MISS if rain else HitOutcome.UNUSED)


def test_charge_interruption_and_switch_cleanup(neutral_type):
    """Flinch cancels a release; switching clears both forced-turn states."""
    key, state = neutral_type
    state = equip_move(state, 'Electro Shot')
    key, state = step_move(key, state, 0, 0, False)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True)))
    _, out = step_action(key, state, continued(), 0)
    assert out.volatile_status.continuation.charging_move_id[0, 0] == 0
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0]
    cleared = clear_volatile_status(state.replace(volatile_status=v.replace(continuation=v.continuation.replace(recharging=v.continuation.recharging.at[0, 0].set(True)))), 0)
    assert not cleared.volatile_status.continuation.recharging[0, 0]
    assert cleared.volatile_status.continuation.charging_move_id[0, 0] == 0


@pytest.mark.parametrize('disabled', [False, True])
def test_shared_release_ignores_encore_and_selection_bookkeeping(neutral_type, disabled):
    """A release finds its stored slot and checks Disable without paying again."""
    key, state = neutral_type
    state = equip_move(state, 'Calm Mind')
    mon = state.active[0]
    moves = jax.tree.map(lambda bank, move: bank.at[2].set(move), mon.moves, make_move('Electro Shot'))
    state = update_active(state, 0, mon.replace(moves=moves))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        continuation=v.continuation.replace(charging_move_id=v.continuation.charging_move_id.at[0, 0].set(905)),
        restrictions=v.restrictions.replace(
            encore_turns=v.restrictions.encore_turns.at[0, 0].set(2),
            encore_move_id=v.restrictions.encore_move_id.at[0, 0].set(347),
            disable_turns=v.restrictions.disable_turns.at[0, 0].set(2 if disabled else 0),
            disabled_move_id=v.restrictions.disabled_move_id.at[0, 0].set(905),
            choice_lock=v.restrictions.choice_lock.at[0, 0].set(2)),
        move_history=v.move_history.replace(last_move_id=v.move_history.last_move_id.at[0, 0].set(905))))
    _, out = step_action(key, state, continued(), 0)
    assert jnp.array_equal(out.team.moves.current_pp, state.team.moves.current_pp)
    assert out.volatile_status.move_history.last_move_id[0, 0] == 905
    assert out.volatile_status.restrictions.choice_lock[0, 0] == 2
    assert out.volatile_status.continuation.charging_move_id[0, 0] == 0
    assert out.boosts.normal_boosts[0, 3] == 0
    assert (out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) != disabled


def test_hyper_beam_recharge_precedes_status_checks(neutral_type):
    """A successful beam forces recharge without ticking sleep or confusion."""
    key, state = neutral_type
    _, hit = resolve_multi_hit(key, state, 0, make_move('Hyper Beam'))
    assert hit.volatile_status.continuation.recharging[0, 0]
    hit = update_active(hit, 0, hit.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    v = hit.volatile_status
    hit = hit.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[0, 0].set(3)))
    _, out = step_action(key, hit, continued(), 0)
    assert not out.volatile_status.continuation.recharging[0, 0]
    assert out.volatile_status.move_history.move_result[0, 0] == MoveResult.SKIPPED
    assert out.active.sleep_counter[0, 0] == 2
    assert out.volatile_status.confusion_turns[0, 0] == 3
    assert jnp.array_equal(out.active.current_hp, hit.active.current_hp)


@pytest.mark.parametrize('name', ['Giga Impact', 'Prismatic Laser'])
def test_catalogue_recharge_after_hit(neutral_type, name):
    """The shared finishing path forces recharge for physical and special moves."""
    key, state = neutral_type
    _, hit = resolve_multi_hit(key, state, 0, make_move(name))
    assert hit.volatile_status.continuation.recharging[0, 0]


def test_blocked_hyper_beam_needs_no_recharge(neutral_type):
    """Protect blocks the beam before its recharge effect is applied."""
    key, state = neutral_type
    state = equip_move(state, 'Hyper Beam')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protection=v.protection.replace(protected=v.protection.protected.at[1, 0].set(True))))
    _, out = move_hits(key, state, 0, 0)
    assert not out.volatile_status.continuation.recharging[0, 0]


def test_missed_beam_and_substitute_recharge(neutral_type):
    """A missed beam needs no recharge, but a Substitute hit does."""
    key, state = neutral_type
    state = equip_move(state, 'Hyper Beam')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    _, missed = step_move(key, state, 0, 0, False)
    assert not missed.volatile_status.continuation.recharging[0, 0]
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(1)))
    _, hit = resolve_multi_hit(key, state, 0, make_move('Hyper Beam'))
    assert hit.volatile_status.continuation.recharging[0, 0]
    assert hit.active.current_hp[1, 0] == state.active.current_hp[1, 0]


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
    state = state.replace(volatile_status=v.replace(
        continuation=v.continuation.replace(
            charging_move_id=v.continuation.charging_move_id.at[0, 0].set(905),
            recharging=v.continuation.recharging.at[1, 0].set(True))))
    episode = episode.replace(battle=state)
    action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.CONTINUED))
    out, ts = env.step(episode, action)
    assert not ts.extras['invalid_action']
    assert out.turn == 1
    assert not jnp.any(out.battle.volatile_status.continuation.recharging)
    assert not jnp.any(out.battle.volatile_status.continuation.charging_move_id)
    assert out.battle.active.moves.current_pp[0, 0, 0] == 0
    assert out.battle.active.sleep_counter[1, 0] == 2
    env.observation_spec.validate(ts.observation)
