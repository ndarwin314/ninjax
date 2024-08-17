from typing import Union, Tuple, Dict, Any, Optional
from collections import namedtuple

import chex
from flax import struct
from jax import lax
from gymnax.environments import environment
import gymnax.environments.spaces as spaces
import jax.numpy as jnp

from ninjax.side import SideState, step_side
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])


Binary = (0,1)

@struct.dataclass
class BattleState(environment.EnvState):
    turn: int
    agents: (SideState, SideState)
    weather: Weather = (WeatherEnum.NONE, 0)
    terrain: Terrain = (TerrainEnum.NONE, 0)
    trick_room_duration: int = 0
    gravity_duration: int = 0


@struct.dataclass
class BattleParams(environment.EnvParams):
    max_steps_in_episode: int = 1

class Battle(environment.Environment[BattleState, BattleParams]):

    def __init__(self):
        self.action_set = jnp.array(range(10))
        # idk
        #self.obs_shape = (1, 1)

    @property
    def name(self) -> str:
        return "Pokemon"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: Optional[BattleParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def step_env(
        self,
        key: chex.PRNGKey,
        state: BattleState,
        actions: (int, int),
        params: BattleParams,
    ) -> Tuple[chex.Array, BattleState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # the fun part :))))
        first, second = action_order(key, state, actions)
        key, state.agents[first] = step_action(key, BattleState.agents[first], state, actions[first])
        key, state.agents[second] = step_action(key, BattleState.agents[second], state, actions[second])

        key, state = step_field(key, state)


    def reset_env(
        self, key: chex.PRNGKey, params: BattleParams
    ) -> Tuple[chex.Array, BattleState]:
        pass



def action_order(
    key: chex.PRNGKey,
    state: BattleState,
    actions: (int, int),
) -> (int, int):
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    # 1. breaking ties
    # 2. getting priority
    speeds = jnp.zeros(2)
    priorities = jnp.zeros(2)
    for i in range(2):
        # TODO: will eventually need to account for other sources of speed boost
        speeds[i] = state.agents[i].active.boosted_stats[StatEnum.SPEED]
    first = jnp.argmax(speeds)
    return first, 1 - first

def step_action(
    key: chex.PRNGKey,
    side: SideState,
    state: BattleState,
    action: int,
) -> (chex.PRNGKey, SideState):
    # this will execute whatever move is selected
    return key, side

def step_field(
    key: chex.PRNGKey,
    state: "BattleState",
) -> (chex.PRNGKey, "BattleState"):
    # there is probably som reason we need rng or actions but idk rn
    weather_duration = max(state.weather.duration-1, 0)
    new_weather = state.weather * weather_duration
    # TODO: add damage from sand at some point and resulting switches
    terrain_duration = max(state.terrain.duration-1, 0)
    new_terrain = state.terrain * terrain_duration
    key, agent0 = step_side(key, state.agents[0])
    key, agent1 = step_side(key, state.agents[0])
    state = state.replace(
        turn=max(state.turn-1, 0),
        agents=(agent0, agent1),
        weather=(new_weather, weather_duration),
        terrain=(new_terrain, terrain_duration),
        trick_room_duration=max(state.trick_room_duration - 1, 0),
        gravity_duration=max(state.gravity_duration - 1, 0),
    )

    return key, state
