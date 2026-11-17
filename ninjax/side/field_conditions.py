"""Weather and terrain transitions for immutable battle state."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.enum_types import (
    AbilityEnum as A,
    ItemEnum as I,
    Terrain,
    TerrainEnum as T,
    TerrainEnum,
    Weather,
    WeatherEnum as W,
    WeatherEnum,
)
from ninjax.side.queries import effective_weather
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts, clear_boosts, set_boosts
from ninjax.stats import StatBoosts


def _field_status_add_boosts(s, *, player):
    """Field status add boosts."""
    return add_boosts(s, player, 1, 1)


def _preserve_operand(s):
    """Preserve operand."""
    return s


def _field_status_clear_boosts(s):
    """Field status clear boosts."""
    return clear_boosts(clear_boosts(s, 0), 1)


def _field_status_defog(s, *, player):
    """Lower target evasion, clear both sides' hazards and target screens."""


    target = 1-player
    amount = jnp.where(s.ability[target, 0] == A.CONTRARY, 1, -1)
    amount = jnp.where(s.ability[target, 0] == A.CLEAR_BODY, 0, amount)
    boosts = StatBoosts(normal_boosts=s.boosts.normal_boosts[target],
        acc_boosts=s.boosts.acc_boosts[target].at[1].add(amount))
    s = set_boosts(s, target, boosts)
    s = s.replace(**{name: jnp.zeros_like(getattr(s, name)) for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs')})
    s = s.replace(**{name: getattr(s, name).at[target].set(0) for name in ('reflect', 'light_screen', 'aurora_veil')})
    return s.replace(terrain=Terrain(T.NONE, 0))


def _field_status_update_field(s):
    """Field status update field."""
    return s.replace(**{
            name: getattr(s, name)[::-1] for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs',
                                                    'reflect', 'light_screen', 'aurora_veil', 'tailwind')})


def set_terrain(
    state: BattleState, new_terrain: TerrainEnum, duration: int
    ):
    """Set a non-NONE terrain, retaining its duration if it already matches."""
    current_terrain = state.terrain.terrain
    requested = jnp.asarray(new_terrain, jnp.int32).reshape(())
    new_terrain = jnp.where(requested == TerrainEnum.NONE, current_terrain, requested)
    matching_terrain = current_terrain == new_terrain
    new_duration_terrain = duration * (1 - matching_terrain) + state.terrain.duration * matching_terrain
    state = state.replace(terrain=Terrain(new_terrain, new_duration_terrain))
    return state


def set_weather(
    state: BattleState, new_weather: WeatherEnum, duration: int):
    """Set non-NONE weather, retaining its duration if it already matches."""
    current_weather = state.weather.weather
    requested = jnp.asarray(new_weather, jnp.int32).reshape(())
    new_weather = jnp.where(requested == WeatherEnum.NONE, current_weather, requested)
    matching_weather = current_weather == new_weather
    # TODO: add item check for the weather stones
    new_duration_weather = duration * (1 - matching_weather) + state.weather.duration * matching_weather
    return state.replace(weather=Weather(new_weather, new_duration_weather))


def field_status(state, player, move):
    """Set screens and timers, swap side conditions, or clear stages field-wide."""
    ident = move.identifier[0]
    old_tailwind = state.tailwind[player, 0]
    duration = jnp.where(state.active.item[player, 0] == I.LIGHT_CLAY, 8, 5)
    for name, enabled in [('light_screen', ident == 113),
                          ('aurora_veil', (ident == 694) & (effective_weather(state.weather.weather, state.ability, state.active.current_hp) == W.SNOW)),
                          ('tailwind', ident == 366)]:
        current = getattr(state, name)
        value = jnp.where(enabled & (current[player, 0] == 0), jnp.where(ident == 366, 4, duration), current[player, 0])
        state = state.replace(**{name: current.at[player, 0].set(value)})
    state = jax.lax.cond((old_tailwind == 0) & (state.tailwind[player, 0] > 0) & (state.ability[player, 0] == A.WIND_RIDER),
        partial(_field_status_add_boosts, player=player), _preserve_operand, state)
    state = state.replace(trick_room_duration=jnp.where(ident == 433, jnp.where(state.trick_room_duration > 0, 0, 5), state.trick_room_duration))
    state = jax.lax.cond(ident == 114, _field_status_clear_boosts, _preserve_operand, state)
    defog = partial(_field_status_defog, player=player)
    state = jax.lax.cond(ident == 432, defog, _preserve_operand, state)
    return jax.lax.cond(ident == 756, _field_status_update_field, _preserve_operand, state)
