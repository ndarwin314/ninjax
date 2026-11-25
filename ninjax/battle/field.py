"""Order residual phases, tick timers and resolve end-of-turn faints."""

import chex
import jax.lax
import jax.numpy as jnp
from jax import lax, random

from ninjax.battle.residuals import (
    ResidualHPPhase,
    residual_speed_order,
    run_hp_residuals,
    leftovers,
    residual_abilities,
    residual_orbs,
    step_moody,
    step_side_conditions,
)
from ninjax.enums.field import Terrain, Weather
from ninjax.records.events import Cause
from ninjax.side.faint_effects import direct_faint, resolve_faints
from ninjax.side.item_effects import refresh_paradox, update_items
from ninjax.side.state import BattleState


def _perish_faint(state, player):
    """Faint an active Pokémon whose Perish Song counter reached zero."""
    return direct_faint(state, player, -1, Cause.PERISH)


def _unchanged_player_state(state, player):
    """Keep the state when this player has no expiring counter."""
    return state


def _perish_residual(key, state):
    """Queue the order-24 Perish Song faints in Speed order."""
    speeds = state.effective_speeds
    effective = jnp.where(state.trick_room_duration > 0, -speeds, speeds)
    key, tie_key = random.split(key)
    slowest = jnp.where(effective[0] == effective[1], random.randint(tie_key, (), 0, 2), jnp.argmin(effective))
    expiring = state.perish_turns[:, 0] == 1
    for player in (1 - slowest, slowest):
        state = lax.cond(expiring[player], _perish_faint, _unchanged_player_state, state, player)
    state = state.replace(perish_turns=jnp.maximum(0, state.perish_turns - 1))
    return key, resolve_faints(state)


def _before_status(key, state, order):
    """Run order-five status cures and Leftovers before poison and burn."""
    key, state = residual_abilities(key, state, speed_boost=False, order=order)
    return key, leftovers(state, order)


def _skip_ordered_group(key, state, order):
    """Skip a residual group after a terminal checkpoint."""
    return key, state


def _late_residuals(key, state, order):
    """Run order-28 abilities, followed by status orbs and item updates."""
    key, state = step_moody(key, state, order)
    key, state = residual_abilities(key, state, status_cures=False, order=order)
    state = run_hp_residuals(state, (ResidualHPPhase.BAD_DREAMS,), order)
    return lax.cond(state.winner < 0, _finish_residual_items,
                    _skip_ordered_group, key, state, order)


def _finish_residual_items(key, state, order):
    """Run late status orbs and then update held items."""
    key, state = residual_orbs(key, state, order)
    return key, update_items(state)


def _skip_residual_effects(key, state):
    """Keep a finished battle at the residual checkpoint that ended it."""
    return key, state


def step_field(
    key: chex.PRNGKey,
    state: BattleState,
) -> tuple[jax.Array, BattleState]:
    """Run the residual schedule, resolving faints between HP phases."""
    weather_duration = jnp.maximum(state.weather.duration - 1, 0)
    new_weather = jnp.where(weather_duration > 0, state.weather.weather, 0)
    terrain_duration = jnp.maximum(state.terrain.duration - 1, 0)
    new_terrain = jnp.where(terrain_duration > 0, state.terrain.terrain, 0)

    order = residual_speed_order(state)
    key, state = step_side_conditions(key, state)
    state = run_hp_residuals(state, (ResidualHPPhase.WEATHER, ResidualHPPhase.GRASSY_TERRAIN), order)
    key, state = lax.cond(state.winner < 0, _before_status, _skip_ordered_group, key, state, order)
    state = run_hp_residuals(state, (ResidualHPPhase.POISON, ResidualHPPhase.BURN), order)
    key, state = lax.cond(state.winner < 0, _perish_residual, _skip_residual_effects, key, state)
    key, state = lax.cond(state.winner < 0, _late_residuals, _skip_ordered_group, key, state, order)

    state = state.replace(
        turn_number=state.turn_number + 1,
        weather=Weather(new_weather, weather_duration),
        terrain=Terrain(new_terrain, terrain_duration),
        trick_room_duration=jnp.maximum(state.trick_room_duration - 1, 0),
        gravity_duration=jnp.maximum(state.gravity_duration - 1, 0),
        magic_room_duration=jnp.maximum(state.magic_room_duration - 1, 0),
        volatile_status=state.volatile_status.replace(roost=jnp.zeros_like(state.volatile_status.roost),
            previous_move_result=state.volatile_status.move_result, move_result=jnp.zeros_like(state.volatile_status.move_result)),
    )

    return key, refresh_paradox(state)
