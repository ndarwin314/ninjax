"""Weather and terrain transitions for immutable battle state."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.stats import StatBoosts
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import (
    Terrain,
    TerrainEnum as T,
    TerrainEnum,
    Weather,
    WeatherEnum as W,
    WeatherEnum,
)
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import SemiInvulnerability
from ninjax.side.queries import effective_weather
from ninjax.side.state import BattleState, Hazards
from ninjax.side.updates import add_boosts, clear_boosts, set_boosts


FIELD_CONDITION_MOVES = (54, 116, 170, 199, 219, 356, 393, 472, 478)


def field_condition_effect(state, player, move):
    """Apply a supported field or active condition and return its success."""
    ident = move.identifier[0]
    v = state.volatile_status
    side = state.side_conditions
    target = 1 - player
    gravity = ident == 356
    magic = ident == 478
    wonder = ident == 472
    safeguard = ident == 219
    mist = ident == 54
    magnet = ident == 393
    focus = ident == 116
    lock = (ident == 199) | (ident == 170)
    # Gravity cannot be restarted, whereas Magic and Wonder Room toggle.
    gravity_ok = gravity & (state.gravity_duration == 0)
    safeguard_ok = safeguard & (side.safeguard[player, 0] == 0)
    mist_ok = mist & (side.mist[player, 0] == 0)
    magnet_ok = magnet & (state.gravity_duration == 0) & (
        v.magnet_rise_turns[player, 0] == 0) & (v.residuals.ingrain[player, 0] == 0)
    focus_ok = focus & ~v.focus_energy[player, 0]
    lock_ok = lock & (v.lock_on_turns[player, 0] == 0)
    success = gravity_ok | magic | wonder | safeguard_ok | mist_ok | magnet_ok | focus_ok | lock_ok

    # Gravity grounds both sides and cancels airborne charge volatiles.
    continuation = v.continuation
    airborne = gravity_ok & (continuation.semi_invulnerability == SemiInvulnerability.AIRBORNE)
    continuation = continuation.replace(
        charging_move_id=jnp.where(airborne, 0, continuation.charging_move_id),
        semi_invulnerability=jnp.where(airborne, SemiInvulnerability.NONE, continuation.semi_invulnerability))
    v = v.replace(
        continuation=continuation,
        magnet_rise_turns=v.magnet_rise_turns.at[player, 0].set(
            jnp.where(magnet_ok, 5, v.magnet_rise_turns[player, 0])),
        focus_energy=v.focus_energy.at[player, 0].set(v.focus_energy[player, 0] | focus_ok),
        lock_on_target_slot=v.lock_on_target_slot.at[player, 0].set(
            jnp.where(lock_ok, state.active_index[target, 0], v.lock_on_target_slot[player, 0])),
        lock_on_turns=v.lock_on_turns.at[player, 0].set(
            jnp.where(lock_ok, 2, v.lock_on_turns[player, 0])))
    v = v.replace(magnet_rise_turns=jnp.where(gravity_ok, 0, v.magnet_rise_turns))
    side = side.replace(
        safeguard=side.safeguard.at[player, 0].set(jnp.where(safeguard_ok, 5, side.safeguard[player, 0])),
        mist=side.mist.at[player, 0].set(jnp.where(mist_ok, 5, side.mist[player, 0])))
    state = state.replace(
        volatile_status=v, side_conditions=side,
        gravity_duration=jnp.where(gravity_ok, 5, state.gravity_duration),
        magic_room_duration=jnp.where(magic, jnp.where(state.magic_room_duration > 0, 0, 5), state.magic_room_duration),
        wonder_room_duration=jnp.where(wonder, jnp.where(state.wonder_room_duration > 0, 0, 5), state.wonder_room_duration))
    return state, success


def clear_target_screens(state, target, enabled):
    """Clear the selected side's three damage-reducing screens."""
    return state.replace(
        side_conditions=state.side_conditions.replace(
            reflect=state.side_conditions.reflect.at[target].set(jnp.where(enabled, 0, state.side_conditions.reflect[target])),
            light_screen=state.side_conditions.light_screen.at[target].set(jnp.where(enabled, 0, state.side_conditions.light_screen[target])),
            aurora_veil=state.side_conditions.aurora_veil.at[target].set(jnp.where(enabled, 0, state.side_conditions.aurora_veil[target]))))


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
    s = clear_target_screens(s.replace(hazards=Hazards()), target, True)
    return s.replace(terrain=Terrain(T.NONE, 0))


def _field_status_update_field(s):
    """Swap both sides' hazards and timed side conditions for Court Change."""
    hazards, side_conditions = jax.tree.map(jnp.flipud, (s.hazards, s.side_conditions))
    return s.replace(hazards=hazards, side_conditions=side_conditions)


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
    conditions = state.side_conditions
    old_tailwind = conditions.tailwind[player, 0]
    duration = jnp.where(state.active.item[player, 0] == I.LIGHT_CLAY, 8, 5)
    for name, enabled in [('reflect', ident == 115),
                          ('light_screen', ident == 113),
                          ('aurora_veil', (ident == 694) & (effective_weather(state.weather.weather, state.ability, state.active.current_hp) == W.SNOW)),
                          ('tailwind', ident == 366)]:
        current = getattr(conditions, name)
        value = jnp.where(enabled & (current[player, 0] == 0), jnp.where(ident == 366, 4, duration), current[player, 0])
        conditions = conditions.replace(**{name: current.at[player, 0].set(value)})
    state = state.replace(side_conditions=conditions)
    state = jax.lax.cond((old_tailwind == 0) & (state.side_conditions.tailwind[player, 0] > 0) & (state.ability[player, 0] == A.WIND_RIDER),
        partial(_field_status_add_boosts, player=player), _preserve_operand, state)
    state = state.replace(trick_room_duration=jnp.where(ident == 433, jnp.where(state.trick_room_duration > 0, 0, 5), state.trick_room_duration))
    # Haze, Defog and Court Change are mutually exclusive move callbacks.
    branch = jnp.select((ident == 114, ident == 432, ident == 756), (0, 1, 2), default=3)
    return jax.lax.switch(branch,
        (_field_status_clear_boosts, partial(_field_status_defog, player=player),
         _field_status_update_field, _preserve_operand), state)
