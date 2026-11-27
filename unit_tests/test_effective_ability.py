"""Effective ability state and switch-reset behavior."""
import jax
import jax.numpy as jnp

from ninjax.battle.switching import swap_out
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Weather, WeatherEnum as W
from ninjax.enums.pokemon import StatEnum, Status
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.queries import effective_weather
from ninjax.side.status_effects import set_status
from ninjax.side.updates import add_boosts, clear_volatile_status
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def test_ability_override_suppression_and_switch_reset(neutral_type):
    """An override is temporary; suppression does not erase either ability."""
    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.LEVITATE)))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[0, 0].set(A.INSOMNIA))))
    assert state.ability[0, 0] == A.INSOMNIA
    assert state.team.ability[0, 0, 0] == A.LEVITATE
    v = state.volatile_status
    suppressed = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_suppressed=v.ability_state.ability_suppressed.at[0, 0].set(True))))
    assert suppressed.ability[0, 0] == A.NONE
    assert clear_volatile_status(suppressed, 0).ability[0, 0] == A.LEVITATE


@jax.jit
def read_abilities(state):
    """Exercise effective ability resolution as a compiled pure function."""
    return state.ability


def test_status_immunity_uses_override(neutral_type):
    """A temporary Insomnia grants sleep immunity without changing the team."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[0, 0].set(A.INSOMNIA))))
    _, blocked = set_status(key, state, 0, jnp.array([Status.SLEEP]))
    assert blocked.active.status[0, 0] == Status.NONE
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_suppressed=v.ability_state.ability_suppressed.at[0, 0].set(True))))
    _, asleep = set_status(key, state, 0, jnp.array([Status.SLEEP]))
    assert asleep.active.status[0, 0] == Status.SLEEP


def test_override_is_used_by_move_immunity_and_survives_hp_updates(neutral_type):
    """Levitate and Sturdy use overrides without rewriting original abilities."""


    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.STURDY)))
    v = state.volatile_status
    levitating = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[1, 0].set(A.LEVITATE))))
    assert levitating.is_immune_to_move(1, make_move('Earthquake'))
    assert not levitating.is_grounded(1)[0]
    out = take_damage_value(levitating, 1, jnp.array([10000]), True)
    assert out.active.current_hp[1, 0] == 0
    assert out.team.ability[1, 0, 0] == A.STURDY
    v = state.volatile_status
    suppressed = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_suppressed=v.ability_state.ability_suppressed.at[1, 0].set(True))))
    assert take_damage_value(suppressed, 1, jnp.array([10000]), True).active.current_hp[1, 0] == 0
    assert take_damage_value(state, 1, jnp.array([10000]), True).active.current_hp[1, 0] == 1


def test_effective_ability_drives_weather_and_boosts(neutral_type):
    """Cloud Nine and Contrary use the volatile ability, not the team value."""


    _, state = neutral_type
    state = state.replace(weather=Weather(W.SUN, 5))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[0, 0].set(A.CLOUD_NINE))))
    assert effective_weather(state.weather.weather, state.ability, state.active.current_hp) == W.NONE
    state = state.replace(volatile_status=state.volatile_status.replace(ability_state=state.volatile_status.ability_state.replace(ability_override=state.volatile_status.ability_state.ability_override.at[0, 0].set(A.CONTRARY))))
    assert add_boosts(state, 0, StatEnum.ATTACK, 1).boosts.normal_boosts[0, 1] == -1


def test_hit_effects_use_override_and_preserve_original(neutral_type):
    """Stamina from an override responds to a hit without changing the team."""


    key, state = neutral_type
    original = state.team.ability[1, 0, 0]
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[1, 0].set(A.STAMINA))))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Dragon Pulse'))
    assert out.boosts.normal_boosts[1, 2] == 1
    assert out.team.ability[1, 0, 0] == original
    assert out.ability[1, 0] == A.STAMINA


def test_switch_uses_override_then_restores_incoming_ability(neutral_type):
    """Regenerator uses the outgoing override and the incoming ability is original."""

    key, state = neutral_type
    team = state.team.replace(current_hp=state.team.current_hp.at[0, 0, 0].set(20),
        ability=state.team.ability.at[0, 1, 0].set(A.STURDY))
    v = state.volatile_status
    state = state.replace(team=team, volatile_status=v.replace(ability_state=v.ability_state.replace(ability_override=v.ability_state.ability_override.at[0, 0].set(A.REGENERATOR))))
    maximum = state.active.max_hp[0]
    out = swap_out(key, state, 0, 1)
    assert out.team.current_hp[0, 0, 0] == 20 + maximum // 3
    assert out.team.ability[0, 0, 0] == A.NONE
    assert out.ability[0, 0] == A.STURDY
    assert out.volatile_status.ability_state.ability_override[0, 0] == -1
