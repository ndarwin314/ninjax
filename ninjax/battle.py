from typing import Union, Tuple, Dict, Any, Optional

import chex
import jax.lax
from flax import struct
from jax import lax, random, jit
from gymnax.environments import environment
import gymnax.environments.spaces as spaces
import jax.numpy as jnp


from ninjax.side import BattleState, update_active
from ninjax.utils import triple_and
from ninjax.enum_types import StatEnum, Type, AbilityEnum, Weather, Terrain, Status
from ninjax.game_logic import (step_side_conditions, swap_out, end_turn_damage, do_move_damage, do_healing_from_move,
                               do_stat_boost_from_move, do_status_move, do_flash_fire_from_move, move_interrupted,
                               move_used, step_moody)


Binary = (0,1)

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
        act1, act2 = actions
        # TODO: stupid action order code probably needs to be rewritten to be more jax-y
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
) -> (chex.PRNGKey, BattleState):
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



def action_order(
    state: BattleState,
    actions: (int, int),
) -> (int, int):
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    # 1. breaking ties
    # 2. getting priority
    speeds = state.boosted_stats[StatEnum.SPEED]
    # priorities = []
    first = speeds[0] < speeds[1]
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

def step_move(
    key: chex.PRNGKey,
    state: BattleState,
    player_idx: int,
    index: int,
    is_tera: bool) -> (chex.PRNGKey, BattleState):
    # this, or something this calls is probably going to be the most complex function
    # for now im just going to implement a simplistic version
    # TODO: add the tera part of move
    # TODO: if condition for status moves before we do damage stuff

    # TODO: stuff to add
    # 1. glaive rush mult
    # 3. unaware for both
    # 4. guts/facade
    # 7. various crit damage and rate multipliers
    attacker = state.active[player_idx]

    # do tera stuff
    can_tera = state.can_tera.at[player_idx].set(1 - is_tera)

    # do sleep stuff
    # more hacks to avoid branching
    old_sleep_counter = attacker.sleep_counter
    sleep_counter = jnp.maximum(0, old_sleep_counter - 1)
    # if we aren't asleep, sleep counter is already 0, so the diff is 0
    woken_up = jnp.logical_and(sleep_counter==0, old_sleep_counter-sleep_counter==1)
    new_status = attacker.status * (1-woken_up)

    attacker = attacker.replace(
        status=new_status,
        sleep_counter=sleep_counter,
        is_terastallized=jnp.bool([is_tera]))

    state = update_active(state, player_idx, attacker)
    # add check for if move happens because of flinch, sleep, paralysis, etc here
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)
    is_paralyzed = attacker.status==Status.PARALYZE
    is_fully_paralyzed = jnp.logical_and(jnp.less_equal(r, 0.25), is_paralyzed)
    is_sleeping = attacker.status==Status.SLEEP
    is_flinched = False
    is_interrupted = triple_and(is_flinched, is_sleeping, is_fully_paralyzed)
    key, state = jax.lax.cond(is_interrupted, move_interrupted, move_used, key, state, player_idx, index)
    return key, state


def step_switch(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int,
    index: int,
    is_tera: bool
) -> (chex.PRNGKey, BattleState):
    # switch needs to access the battle state because opponent switching triggers annoying things
    # TODO: add an opponent switched field somewhere for stakeout + analytic
    return swap_out(state, player_index, index), key


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

def step_field(
    key: chex.PRNGKey,
    state: BattleState,
) -> (chex.PRNGKey, BattleState):
    # there is probably some reason we need rng or actions but idk rn
    weather_duration = jnp.maximum(state.weather.duration - 1, 0)
    new_weather = state.weather * weather_duration
    terrain_duration = jnp.maximum(state.terrain.duration - 1, 0)
    new_terrain = state.terrain * terrain_duration

    key, state = step_side_conditions(key, state)

    # weather, terrain, status, items (leftovers etc)
    state = end_turn_damage(state)

    key, state = step_moody(key, state)

    state = state.replace(
        turn_number=state.turn_number + 1,
        weather=Weather(new_weather, weather_duration),
        terrain=Terrain(new_terrain, terrain_duration),
        trick_room_duration=jnp.maximum(state.trick_room_duration - 1, 0),
        gravity_duration=jnp.maximum(state.gravity_duration - 1, 0),
    )

    return key, state
