from typing import Union, Tuple, Dict, Any, Optional

import chex
import jax.lax
from flax import struct
from jax import lax, random, jit
from functools import cached_property
from jumanji.env import Environment
from jumanji import specs
from jumanji.types import StepType, restart, transition
import jax.numpy as jnp


from ninjax.side import BattleState, update_active
from ninjax.utils import triple_and, triple_or
from ninjax.enum_types import StatEnum, Type, AbilityEnum, Weather, Terrain, Status, MoveType
from ninjax.game_logic import (step_side_conditions, swap_out, end_turn_damage, do_healing_from_move,
                               do_stat_boost_from_move, do_status_move, do_flash_fire_from_move, move_interrupted,
                               move_used, step_moody)


Binary = (0,1)

@struct.dataclass
class BattleParams:
    max_steps_in_episode: int = 100

@struct.dataclass
class EpisodeState:
    """Jumanji state; the existing battle representation is retained for now."""
    key: chex.PRNGKey
    battle: BattleState
    turn: chex.Array


class Battle(Environment):

    def __init__(self, teams=None, params=None):
        # this is a bad way to represent actions but i cant think of a better way
        # we have 4 actions for moves, 4 for move + tera, and 6 for switching, 1 for no-op
        # one of these actions is still illegal, switching to self but that makes it way worse
        self.action_set = jnp.array(range(15))
        self.teams = teams
        self.params = params or BattleParams()
        super().__init__()
        # idk
        #self.obs_shape = (1, 1)

    @property
    def name(self) -> str:
        return "Pokemon"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    @cached_property
    def action_spec(self):
        return specs.MultiDiscreteArray(jnp.array([15, 15], dtype=jnp.int32), name="actions")

    @cached_property
    def observation_spec(self):
        return specs.Array(shape=(2, 1), dtype=jnp.int32, name="observation")

    def reset(self, key):
        if self.teams is None:
            raise ValueError("Battle requires caller-provided teams for reset")
        state = EpisodeState(key=key, battle=BattleState(team=self.teams), turn=jnp.int32(0))
        return state, restart(jnp.zeros((2, 1), dtype=jnp.int32))

    def step(self, state, action):
        # Battle transitions are not yet connected to this interface.
        key, _ = random.split(state.key)
        state = state.replace(key=key, turn=state.turn + jnp.int32(1))
        timestep = transition(jnp.float32(0), jnp.zeros((2, 1), dtype=jnp.int32))
        timeout = state.turn >= self.params.max_steps_in_episode
        timestep = timestep.replace(step_type=jnp.where(timeout, StepType.LAST, StepType.MID))
        return state, timestep


    def step_env(
        self,
        key: chex.PRNGKey,
        state: BattleState,
        actions: (int, int),
        params: BattleParams,
    ) -> Tuple[chex.Array, BattleState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        act1, act2 = actions
        is_move, index, is_tera, is_no_op = decode_action(actions)
        active = state.active
        moves = active.moves[(0, 1), index]
        is_gale_wings = triple_and(
            moves.type==Type.FLYING,
            active.ability==AbilityEnum.GALE_WINGS,
            active.hp_percent==1
        )
        is_prankster = jnp.logical_and(
            moves.move_type==MoveType.STATUS,
            active.ability==AbilityEnum.PRANKSTER
        )
        is_triage = jnp.logical_and(
            moves.healing,
            active.ability==AbilityEnum.TRIAGE
        )
        priorities = moves.priority + is_triage + is_prankster + is_gale_wings
        priorities = priorities * is_move + (1-is_move) * 7
        first, second = action_order(key, state, priorities)
        key, state = step_action(key, state, act1, first)
        key, state = step_action(key, state, act2, second)

        key, state = step_field(key, state)
        # should return obs, state, reward, done, info
        return jnp.array([0]), state, jnp.array([0]), jnp.array([0]), {}


    def reset_env(
        self, key: chex.PRNGKey, params: BattleParams
    ) -> Tuple[chex.Array, BattleState]:
        pass

def equal_priority(r, speeds, priorities):
    first = jnp.argmax(speeds)
    tie_val = jnp.less_equal(r, 0.5)
    return jax.lax.select(speeds[0]!=speeds[1], jnp.array([first, 1-first]), jnp.array([tie_val, 1-tie_val]))

def action_order(
    key: chex.PRNGKey,
    state: BattleState,
    priorities: (int, int),
) -> (int, int):
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)

    priorities = jnp.array(priorities)
    diff_priorities = priorities[0] != priorities[1]
    speeds = state.boosted_stats[StatEnum.SPEED]
    diff_speeds = speeds[0] != speeds[1]
    first_prio = jnp.argmax(priorities)
    first_speed = jnp.argmax(speeds)
    tiebreak = jnp.less_equal(r, 0.5)
    first = (first_prio * diff_priorities +
             first_speed * jnp.logical_and(diff_speeds, 1-diff_priorities) +
             tiebreak * jnp.logical_and(1-diff_priorities, 1-diff_speeds))
    return first, 1-first

def decode_action(action):
    # takes int in [0, 15) and returns a tuple of is_move, index, is_tera
    # index is a move index in [0,4) if action is a move, and in [0,6) if its a switch
    # if is_move == False then the third index should be ignored
    is_move_action = jnp.less(action, 8)
    move_index = (action - 4) % 14
    is_tera = jnp.greater(action, 4)
    switch_index = action - 8
    index = move_index * is_move_action + switch_index * (1 - is_move_action)
    is_no_op = jnp.equal(action, 15)
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
    is_interrupted = triple_or(is_flinched, is_sleeping, is_fully_paralyzed)
    key, state = jax.lax.cond(
        is_interrupted, move_interrupted, move_used,
        key, state, player_idx, index)
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
