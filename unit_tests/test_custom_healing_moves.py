"""Focused checks for special stat and healing status moves."""
import jax
import jax.numpy as jnp

from ninjax.battle.actions import step_move
from ninjax.enums.field import Weather, WeatherEnum
from ninjax.enums.pokemon import StatEnum
from ninjax.side.updates import update_active_fields
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def test_belly_drum_pays_hp_and_maximizes_attack(neutral_type):
    """A valid Belly Drum spends half max HP and sets Attack to +6."""
    key, state = neutral_type
    state = equip_move(state, 'Belly Drum')
    _, result = jax.jit(step_move)(key, state, 0, 0, False, False)
    assert result.active.current_hp[0, 0] == state.active.current_hp[0, 0] - state.active.max_hp[0] // 2
    assert result.boosts.normal_boosts[0, StatEnum.ATTACK] == 6


def test_stockpile_and_swallow(neutral_type):
    """Stockpile boosts defenses and Swallow consumes their stored changes."""
    key, state = neutral_type
    state = equip_move(state, 'Stockpile')
    compiled = jax.jit(step_move)
    key, state = compiled(key, state, 0, 0, False, False)
    assert state.volatile_status.stockpile_count[0, 0] == 1
    assert state.boosts.normal_boosts[0, StatEnum.DEFENSE] == 1
    state = equip_move(state, 'Swallow')
    state = update_active_fields(state, 0, current_hp=jnp.array([30]))
    _, result = compiled(key, state, 0, 0, False, False)
    assert result.active.current_hp[0, 0] > 30
    assert result.volatile_status.stockpile_count[0, 0] == 0
    assert result.boosts.normal_boosts[0, StatEnum.DEFENSE] == 0


def test_synthesis_heals_more_in_sun(neutral_type):
    """Synthesis uses the live weather to choose its recovery."""
    key, state = neutral_type
    state = equip_move(state, 'Synthesis')
    state = update_active_fields(state, 0, current_hp=jnp.array([1]))
    compiled = jax.jit(step_move)
    _, clear = compiled(key, state, 0, 0, False, False)
    state = state.replace(weather=Weather(WeatherEnum.SUN, 3))
    _, sunny = compiled(key, state, 0, 0, False, False)
    assert sunny.active.current_hp[0, 0] > clear.active.current_hp[0, 0]


def test_strength_sap_heals_and_lowers_attack(neutral_type):
    """Strength Sap restores HP and applies an opposing Attack drop."""
    key, state = neutral_type
    state = equip_move(state, 'Strength Sap')
    state = update_active_fields(state, 0, current_hp=jnp.array([1]))
    _, result = jax.jit(step_move)(key, state, 0, 0, False, False)
    assert result.active.current_hp[0, 0] > 1
    assert result.boosts.normal_boosts[1, StatEnum.ATTACK] == -1
