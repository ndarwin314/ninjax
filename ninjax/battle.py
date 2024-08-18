from typing import Union, Tuple, Dict, Any, Optional
from collections import namedtuple
from copy import deepcopy

import chex
from flax import struct
from jax import lax, random
from gymnax.environments import environment
import gymnax.environments.spaces as spaces
import jax.numpy as jnp

from ninjax.side import SideState, step_side, swap_out, take_damage_percent, take_damage_value
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum
from ninjax.move import Move, MoveType
from ninjax.utils import base_damage_compute, calculate_effectiveness_multiplier

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])


Binary = (0,1)

@struct.dataclass
class BattleState(environment.EnvState):
    turn: int = 0
    sides: list[SideState]  # always length 2
    weather: Weather = (WeatherEnum.NONE, 0)
    terrain: Terrain = (TerrainEnum.NONE, 0)
    trick_room_duration: int = 0
    gravity_duration: int = 0

    def get_active(self, index):
        return self.sides[index].active


@struct.dataclass
class BattleParams(environment.EnvParams):
    max_steps_in_episode: int = 1

class Battle(environment.Environment[BattleState, BattleParams]):

    def __init__(self):
        # this is a bad way to represent actions but i cant think of a better way
        # we have 4 actions for moves, 4 for move + tera, and 6 for switching
        # one of these actions is still illegal, switching to self but that makes it way worse
        self.action_set = jnp.array(range(14))
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
        key, state.sides[first] = step_action(key, state, actions, first)
        key, state.sides[second] = step_action(key, state, actions, second)

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
        speeds[i] = state.sides[i].active.boosted_stats[StatEnum.SPEED]
    first = jnp.argmax(speeds)
    return first, 1 - first

def decode_action(action: int) -> (bool, int, bool):
    # takes int in [0, 14) and returns a tuple of is_move, index, is_tera
    # index is a move index in [0,4) if action is a move, and in [0,6) if its a switch
    # if is_move == False then the third index should be ignored
    is_move_action = action < 8
    move_index = (action - 4) % 14
    is_tera = action >= 4
    switch_index = action - 8
    index = move_index * is_move_action + switch_index * (1 - is_move_action)
    return is_move_action, index, is_tera


def conditional_mult_round(damage, mult, cond):
    return jnp.floor(damage * mult ** cond + 1 / 2)


def step_move(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int,
    index: int,
    is_tera: bool):
    # this, or something this calls is probably going to be the most complex function
    # for now im just going to implement a simplistic version
    # TODO: add the tera part of move
    # TODO: if condition for status moves before we do damage stuff

    # TODO: stuff to add
    # 1. glaive rush mult
    # 2. burn mult
    # 3. unaware for both
    # 4. guts/facade
    # 5. weather
    # 6. tera + adaptability stab modifiers
    # 7. various crit damage and rate multipliers
    attacker = state.get_active(player_index)
    defender = state.get_active(1 - player_index)
    move = attacker.moves[index]
    # some moves will deviate this, examples psyshock/strike, secret sword, photon geyser, body press
    offensive_stat = attacker.boosted_stats[1 + 3 * move.move_type == MoveType.SPECIAL]
    defensive_stat = defender.boosted_stats[2 + 3 * move.move_type == MoveType.SPECIAL]
    base_damage = base_damage_compute(attacker.level, offensive_stat, defensive_stat, move.bp)

    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    damage = base_damage
    key, one, two = random.split(key, num=3)
    # crit multiplier
    crit_chance = 1 / 24
    is_crit = random.uniform(one) < crit_chance
    crit_multiplier = 1.5
    # damage roll, idc about preserving the in game RNG
    damage = conditional_mult_round(damage, crit_multiplier, is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100
    damage = conditional_mult_round(damage, damage_roll, 1)
    # idk if jit likes the "in" operator, this might be challenging to write
    # stab multiplier
    is_stab = move.type in attacker.type_list
    stab_multiplier = 1.5
    damage = conditional_mult_round(damage, stab_multiplier, is_stab)
    # Type effectiveness, when we get around to implementing observations
    # it should include does not effect, not very effective, or super effective
    effectiveness = calculate_effectiveness_multiplier(move.type, defender.type_list)
    damage = conditional_mult_round(damage, effectiveness, 1)

    # dealing damage
    defending_side = take_damage_value(state.sides[1 - player_index], damage)
    new_sides = deepcopy(state.sides)
    new_sides[1 - player_index] = defending_side
    return state.replace(agents=new_sides)


def step_switch(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int,
    index: int,
    *args):
    # switch needs to access the battle state because opponent switching triggers annoying things
    # TODO: add an opponent switched field somewhere for stakeout + analytic
    new_side = swap_out(key, state.sides[player_index], index)
    new_agents = deepcopy(state.sides)
    new_agents[player_index] = new_side
    return key, state.replace(agents=new_agents)

def step_action(
    key: chex.PRNGKey,
    state: BattleState,
    action: int,
    player_index: int
) -> (chex.PRNGKey, BattleState):
    # this will execute whatever move is selected
    is_move_action, index, is_tera = decode_action(action)
    # i think this is the best way to implement this conditional in jax
    lax.cond(is_move_action, step_move, step_switch, )
    return key, state


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
    key, agent0 = step_side(key, state.sides[0])
    key, agent1 = step_side(key, state.sides[0])
    state = state.replace(
        turn=max(state.turn-1, 0),
        agents=(agent0, agent1),
        weather=(new_weather, weather_duration),
        terrain=(new_terrain, terrain_duration),
        trick_room_duration=max(state.trick_room_duration - 1, 0),
        gravity_duration=max(state.gravity_duration - 1, 0),
    )

    return key, state
