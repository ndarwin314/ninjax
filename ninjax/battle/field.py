"""Order residual phases, tick timers and resolve end-of-turn faints."""

import chex
import jax.lax
import jax.numpy as jnp
from jax import lax, random

from ninjax.battle.residuals import (
    end_turn_damage,
    leftovers,
    residual_abilities,
    residual_orbs,
    step_moody,
    step_side_conditions,
)
from ninjax.enums.field import Terrain, Weather
from ninjax.side.faint_effects import resolve_faints
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import refresh_paradox, update_items
from ninjax.side.state import BattleState


def _perish_faint(state, player):
    """Resolve one expiring Perish Song counter as indirect HP loss."""
    return take_damage_value(state, player, state.active.current_hp[player], False)


def _unchanged_player_state(state, player):
    """Keep the state when this player has no expiring counter."""
    return state


def step_field(
    key: chex.PRNGKey,
    state: BattleState,
) -> tuple[jax.Array, BattleState]:
    # there is probably some reason we need rng or actions but idk rn
    """Apply residual phases once, then decrement timers and resolve deferred faints."""
    weather_duration = jnp.maximum(state.weather.duration - 1, 0)
    new_weather = jnp.where(weather_duration > 0, state.weather.weather, 0)
    terrain_duration = jnp.maximum(state.terrain.duration - 1, 0)
    new_terrain = jnp.where(terrain_duration > 0, state.terrain.terrain, 0)

    key, state = step_side_conditions(key, state)

    # Hydration and Shed Skin resolve at residual order 5, before poison/burn.
    key, state = residual_abilities(key, state, speed_boost=False)

    # weather, terrain, status, items (leftovers etc)
    state = end_turn_damage(state)


    state = leftovers(state)
    key, state = step_moody(key, state)
    key, state = residual_abilities(key, state, status_cures=False)
    key, state = residual_orbs(key, state)
    state = update_items(state)
    # Perish Song resolves at the end of the turn, using effective speed order.
    speeds = state.effective_speeds
    effective = jnp.where(state.trick_room_duration > 0, -speeds, speeds)
    expiring = state.perish_turns[:, 0] == 1
    for player in range(2):
        state = lax.cond(expiring[player], _perish_faint, _unchanged_player_state, state, player)
    key, tie_key = random.split(key)
    slowest = jnp.where(effective[0] == effective[1], random.randint(tie_key, (), 0, 2), jnp.argmin(effective))
    both_out = ~jnp.any((state.team.current_hp[..., 0] > 0) & state.roster)
    state = state.replace(perish_turns=jnp.where(state.perish_turns > 0, state.perish_turns-1, state.perish_turns),
                          winner=jnp.where(jnp.all(expiring) & both_out, slowest, state.winner))

    state = state.replace(
        turn_number=state.turn_number + 1,
        weather=Weather(new_weather, weather_duration),
        terrain=Terrain(new_terrain, terrain_duration),
        trick_room_duration=jnp.maximum(state.trick_room_duration - 1, 0),
        gravity_duration=jnp.maximum(state.gravity_duration - 1, 0),
        volatile_status=state.volatile_status.replace(roost=jnp.zeros_like(state.volatile_status.roost),
            previous_move_result=state.volatile_status.move_result, move_result=jnp.zeros_like(state.volatile_status.move_result)),
    )

    return key, resolve_faints(refresh_paradox(state))
