from typing import Union, Tuple, Dict, Any, Optional
from collections import namedtuple
from functools import partial

import chex
import jax.lax
import dataclass_array as dca
from flax import struct
from jax import lax, random, jit
from gymnax.environments import environment
import gymnax.environments.spaces as spaces
import jax.numpy as jnp
import numpy as np

from ninjax.side import SideState, step_side, swap_out, take_damage_percent, take_damage_value
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType
from ninjax.move import Move, MoveType
from ninjax.utils import base_damage_compute, calculate_effectiveness_multiplier, static_len_array_access

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])


Binary = (0,1)

@struct.dataclass
class BattleState(environment.EnvState):
    sides: (SideState, SideState)
    weather: Weather = Weather(WeatherEnum.NONE, 0)
    terrain: Terrain = Terrain(TerrainEnum.NONE, 0)
    trick_room_duration: int = 0
    gravity_duration: int = 0
    turn_type: TurnType = TurnType.STANDARD
    legal_action_mask: jax.Array = jnp.ones((2, 15))
    can_tera: jax.Array = jnp.ones((2,))


    def get_side(self, index):
        # are you having fun yet?
        return lax.cond(index, lambda: self.sides[1], lambda: self.sides[0])

    def get_active(self, index):
        # are you having fun yet?
        return self.get_side(index).active



@struct.dataclass
class BattleParams(environment.EnvParams):
    max_steps_in_episode: int = 100

class Battle(environment.Environment[BattleState, BattleParams]):

    def __init__(self):
        # this is a bad way to represent actions but i cant think of a better way
        # we have 4 actions for moves, 4 for move + tera, and 6 for switching, 1 for no-op
        # one of these actions is still illegal, switching to self but that makes it way worse
        self.action_set = jnp.array(range(15))
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
        act1, act2 = actions
        first, second = action_order(state, actions)
        key, state = step_action(key, state, act1, first)
        key, state = step_action(key, state, act2, second)

        key, state = step_field(key, state)
        # should return obs, state, reward, done, info
        return jnp.array([0]), state, jnp.array([0]), jnp.array([0]), {}


    def reset_env(
        self, key: chex.PRNGKey, params: BattleParams
    ) -> Tuple[chex.Array, BattleState]:
        pass

def standard_turn_step(
    key: chex.PRNGKey,
    state: BattleState,
    actions: (int, int)
) -> Tuple[chex.Array, BattleState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
    act1, act2 = actions
    first, second = action_order(state, actions)
    key, state = step_action(key, state, act1, first)
    # also check for like is flinched here, and check for sleep for both or something
    key, state = jax.lax.cond(
        second.active.is_alive,
        step_action,
        lambda k, s, a, _: (key, state), key, state, act2, second)

    key, state = step_field(key, state)
    # set legal action masks here
    mask = jnp.zeros((2, 15))
    alive = (state.sides[0].active.is_alive, state.sides[1].active.is_alive)
    bad = jnp.zeros((2, 6))
    for i in range(2):
        bad = bad.at[i].set(state.sides[i].legal_switch_mask())
    # make sure this broadcast works correctly
    bad = bad * alive
    mask = mask.at[:,8:14].set(bad)
    # make sure this axis is the right way
    mask = mask.at[:,15].set(1 - jnp.any(mask[:,8:14], axis=0))
    state = state.replace(legal_action_mask=mask)

    return jnp.array([0]), state, jnp.array([0]), jnp.array([0]), {}

no_op_func = lambda k, s, a, b, c: (k, s)


def switch_move_step(
    key: chex.PRNGKey,
    state: BattleState,
    actions: (int, int)
):
    bad = True
    mask = jnp.ones((2, 15))
    for i in range(2):
        # TODO: add assertions to verify action is legal
        is_move_action, index, is_tera, is_no_op = decode_action(actions[i])
        key, state = jax.lax.cond(is_no_op, no_op_func, step_switch, key, state, i, index, is_tera)
        is_alive = state.sides[i].active.is_alive
        mask = mask.at[i, 8:14].set(state.sides[i].legal_switch_mask())
        bad = jnp.logical_or(bad, is_alive)
    mask = mask.at[:, 0:8].mul(bad)
    state = state.replace(legal_action_mask=mask)
    # TODO: ugggghhhhhh, run it back if not bad, return the correct stuff


def update_side_at_index(state, index, new_side):
    new_sides = state.sides[index].replace(
        team=new_side.team,
        active_index=new_side.active_index,
        stealth_rocks=new_side.stealth_rocks,
        spikes=new_side.spikes,
        toxic_spikes=new_side.toxic_spikes,
        sticky_webs=new_side.sticky_webs,
        reflect=new_side.reflect,
        light_screen=new_side.light_screen,
        aurora_veil=new_side.aurora_veil,
        tailwind=new_side.tailwind,
        toxic_counter=new_side.toxic_counter,
        boosts=new_side.boosts,
        volatile_status=new_side.volatile_status
    )
    return state.replace(sides=new_sides)


@jit
def action_order(
    state: BattleState,
    actions: (int, int),
) -> (int, int):
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    # 1. breaking ties
    # 2. getting priority
    s1 = state.sides[0].boosted_stats[StatEnum.SPEED]
    s2 = state.sides[1].boosted_stats[StatEnum.SPEED]
    priorities = []
    first = s1 < s2
    return first + 0, 1 - first

def decode_action(action: int) -> (bool, int, bool):
    # takes int in [0, 14) and returns a tuple of is_move, index, is_tera
    # index is a move index in [0,4) if action is a move, and in [0,6) if its a switch
    # if is_move == False then the third index should be ignored
    is_move_action = action < 8
    move_index = (action - 4) % 14
    is_tera = action >= 4
    switch_index = action - 8
    index = move_index * is_move_action + switch_index * (1 - is_move_action)
    is_no_op = action==15
    return is_move_action, index, is_tera, is_no_op


def conditional_mult_round(damage, mult, cond):
    return jnp.floor(damage * mult ** cond + 1 / 2)

@partial(jit, static_argnums=2)
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
    attacking_side = state.sides[player_index]
    defending_side = state.sides[1 - player_index]
    attacker = attacking_side.active
    defender = defending_side.active
    move = attacker.moves[index]
    # do tera stuff
    can_tera = state.can_tera.at[player_index].set(1 - is_tera)
    attacker = attacker.replace(is_terastallized=jnp.bool([is_tera]))
    # some moves will deviate this, examples psyshock/strike, secret sword, photon geyser, body press
    # TODO i put a sum here to make jax stop complainign even though this should always be a scalar
    test = 3 * jnp.sum(move.move_type == MoveType.SPECIAL)
    offensive_stat = attacking_side.boosted_stats[1 + test]
    defensive_stat = defending_side.boosted_stats[2 + test]
    base_damage = base_damage_compute(attacker.level, offensive_stat, defensive_stat, move.base_power)

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
    # stab multiplier
    is_stab = np.any(attacker.type_list==move.type)
    stab_multiplier = 1.5
    damage = conditional_mult_round(damage, stab_multiplier, is_stab)
    # Type effectiveness, when we get around to implementing observations
    # it should include does not affect, not very effective, or super effective
    effectiveness = calculate_effectiveness_multiplier(move.type, defender.type_list)
    damage = conditional_mult_round(damage, effectiveness, 1).astype(int)

    # dealing damage
    defending_side = take_damage_value(defending_side, damage)

    new_state = dca.stack([attacking_side, defending_side])
    return key, new_state


def step_switch(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int,
    index: int,
    is_tera: bool):
    # switch needs to access the battle state because opponent switching triggers annoying things
    # TODO: add an opponent switched field somewhere for stakeout + analytic
    new_side = swap_out(state.get_side(player_index), index)
    new_state = update_side_at_index(state, player_index, new_side)
    return key, new_state

@jit
def step_action(
    key: chex.PRNGKey,
    state: BattleState,
    action: int,
    player_index: int
) -> (chex.PRNGKey, BattleState):
    # this will execute whatever move is selected
    is_move_action, index, is_tera, is_no_op = decode_action(action)
    # i think this is the best way to implement this conditional in jax
    lax.cond(is_move_action, step_move, step_switch, key, state, player_index, index, is_tera)
    return key, state


def end_turn_damage(state: BattleState, side: SideState) -> (BattleState, SideState):
    # TODO: the order of all these updates is probably incorrect
    # i think it should be like sand, sand, grass, grass
    # whereas this is currently sand, grass, sand grass,
    # and order should also depend on speed
    # that being said idk if that is high priority
    active = side.active
    is_floating = active.is_floating
    is_sand_immune = active.is_sand_immune
    # sand damage
    sand_damage = (1 - is_sand_immune) / 16 * state.weather.weather == WeatherEnum.SANDSTORM
    take_damage_percent(side, sand_damage)
    # grassy terrain healing
    grass_healing = (is_floating - 1) / 16 * state.terrain.terrain == TerrainEnum.GRASSY
    take_damage_percent(side, grass_healing)
    # status damage
    status_damage = (1 / 8 * (active.status==Status.POISON) +
                     1 / 16 * (active.status==Status.BURN) +
                     side.toxic_counter / 16 * (active.status==Status.TOXIC))
    side = take_damage_percent(side, status_damage)
    return state, side

def step_field(
    key: chex.PRNGKey,
    state: BattleState,
) -> (chex.PRNGKey, BattleState):
    # there is probably some reason we need rng or actions but idk rn
    weather_duration = jnp.maximum(state.weather.duration - 1, 0)
    new_weather = state.weather * weather_duration
    terrain_duration = jnp.maximum(state.terrain.duration - 1, 0)
    new_terrain = state.terrain * terrain_duration

    key, side0 = step_side(key, state.sides[0])
    key, side1 = step_side(key, state.sides[1])
    state, side0 = end_turn_damage(state, side0)
    state, side1 = end_turn_damage(state, side1)
    # terrain and weather damage
    # TODO: check the order of these since it matters, if something dies to weather it cant then be healed
    state = state.replace(
        time=state.time + 1,
        sides=(side0, side1),
        weather=Weather(new_weather, weather_duration),
        terrain=Terrain(new_terrain, terrain_duration),
        trick_room_duration=jnp.maximum(state.trick_room_duration - 1, 0),
        gravity_duration=jnp.maximum(state.gravity_duration - 1, 0),
    )

    return key, state
