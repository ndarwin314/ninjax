from typing import Union, Tuple, Dict, Any
from functools import partial
from collections import namedtuple
from dataclasses import field

import jax.lax
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray, BoolArray
import dataclass_array as dca
from flax import struct
import jax.numpy as jnp
from jax import jit

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType, Type, AbilityEnum
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, calculate_effectiveness_multiplier, conditional_mult_round, quad_or

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])


class VolatileStatus(DataclassArray):
    confused: bool = False
    # TODO this is gonna suck, make sure everything has default values
    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        return self

@dca.dataclass_array(broadcast=True, cast_dtype=True)
class BattleState(DataclassArray):
    team: Pokemon['*batch_size 6']
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    stealth_rocks: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.bool([0]))
    sticky_webs: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.bool([0]))
    spikes: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    toxic_spikes: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    reflect: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    light_screen: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    aurora_veil: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    tailwind: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    toxic_counter: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    boosts: StatBoosts = StatBoosts(
        normal_boosts=jnp.zeros((2, 6) ,dtype='int32'),
        acc_boosts=jnp.zeros((2,2), dtype='int32'))
    legal_action_mask: BoolArray['*batch_size 15'] = field(default_factory=lambda: jnp.ones((2, 15)))
    can_tera: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.ones(1))
    # notably volatile status needs like wish, healing wish, and future sight things
    # but those are lowish priority
    volatile_status: VolatileStatus = field(default_factory=VolatileStatus) # TODO

    weather: Weather = Weather(WeatherEnum.NONE, 0)
    terrain: Terrain = Terrain(TerrainEnum.NONE, 0)
    turn_number: int = 0
    trick_room_duration: int = 0
    gravity_duration: int = 0
    turn_type: TurnType = TurnType.STANDARD


    def replace_row(self, idx: int, new_side: 'BattleState'):
        team = self.team.replace_row(idx, new_side.team)
        active_index = self.active_index.at[idx].set(new_side.active_index)
        stealth_rocks = self.stealth_rocks.at[idx].set(new_side.stealth_rocks)
        sticky_webs = self.sticky_webs.at[idx].set(new_side.sticky_webs)
        spikes = self.spikes.at[idx].set(new_side.spikes)
        toxic_spikes = self.toxic_spikes.at[idx].set(new_side.toxic_spikes)
        reflect = self.reflect.at[idx].set(new_side.reflect)
        light_screen = self.light_screen.at[idx].set(new_side.light_screen)
        aurora_veil = self.aurora_veil.at[idx].set(new_side.aurora_veil)
        tailwind = self.tailwind.at[idx].set(new_side.tailwind)
        toxic_counter = self.toxic_counter.at[idx].set(new_side.toxic_counter)
        boosts = self.boosts.replace_row(idx, new_side.boosts)
        volatile_status = self.volatile_status.replace_row(idx, new_side.volatile_status)
        return self.replace(
            team=team, active_index=active_index, stealth_rocks=stealth_rocks, sticky_webs=sticky_webs, spikes=spikes,
            toxic_spikes=toxic_spikes, reflect=reflect, light_screen=light_screen, aurora_veil=aurora_veil,
            tailwind=tailwind, toxic_counter=toxic_counter, boosts=boosts, volatile_status=volatile_status
        )


    @property
    def active(self) -> Pokemon:
        flattened_idx = jnp.squeeze(self.active_index)
        return self.team[[0,1], flattened_idx]

    @property
    def raw_boosted_stats(self):
        stats = self.active.stats
        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + self.boosts.normal_boosts])

    def attack_multiplier(self):
        ability = self.active.ability
        status = self.active.status
        # guts check
        is_guts = jnp.logical_and(ability == AbilityEnum.GUTS, status == Status.BURN).squeeze()
        # huge power
        is_huge_power = (ability == AbilityEnum.HUGE_POWER).squeeze()
        # defeatist
        is_defeatist = jnp.logical_and((ability == AbilityEnum.DEFEATIST).squeeze(), self.active.hp_less_than(0.5))
        # hustle
        is_hustle = (ability == AbilityEnum.DEFEATIST).squeeze()
        # gorilla tactics
        is_gorilla_tactics = (ability == AbilityEnum.GORILLA_TACTICS).squeeze()
        # supreme overlord
        dead_count = jnp.sum(1 - self.team.is_alive)
        is_supreme_overlord = (ability == AbilityEnum.SUPREME_OVERLORD).squeeze()
        return jnp.power(
            jnp.array([1.5, 1.5, 1.5, 2, 0.5, 1 + dead_count / 10]),
            jnp.array([is_guts, is_hustle, is_gorilla_tactics, is_huge_power, is_defeatist, is_huge_power, is_defeatist, is_supreme_overlord]))

    def special_attack_multiplier(self):
        ability = self.active.ability
        is_defeatist = jnp.logical_and((ability == AbilityEnum.DEFEATIST).squeeze(), self.active.hp_less_than(0.5))
        # supreme overlord
        dead_count = jnp.sum(1 - self.team.is_alive)
        is_supreme_overlord = (ability == AbilityEnum.SUPREME_OVERLORD).squeeze()
        return jnp.power(jnp.array([1/2, 1+dead_count/10]), jnp.array([is_defeatist, is_supreme_overlord]))

    @property
    def boosted_stats(self):
        active = self.active
        ability = active.ability
        status = active.status
        stats = self.active.stats
        # squeeze removes dimensions with length 1 which makes this broadcast correctly
        # its probably going to be helpful to use this in other places

        # marvel scale
        temp = conditional_mult_round(stats[...,StatEnum.SPECIAL_DEFENSE], 1.5,
                                      jnp.logical_and(ability==AbilityEnum.MARVEL_SCALE, active.has_status).squeeze())
        stats = stats.at[..., StatEnum.SPECIAL_DEFENSE].set(temp)

        # surge surfer
        terrain = self.terrain.terrain
        is_surge_surfer = jnp.logical_and(
            ability==AbilityEnum.SURGE_SURFER,
            terrain==TerrainEnum.ELECTRIC).squeeze()

        # speed boosting weather abilities
        weather = self.weather.weather
        is_chlorophyll  = jnp.logical_and(
            ability==AbilityEnum.CHLOROPHYLL,
            weather==WeatherEnum.SUN).squeeze()
        is_swift_swim = jnp.logical_and(
            ability == AbilityEnum.SWIFT_SWIM,
            weather == WeatherEnum.RAIN).squeeze()
        is_slush_rush = jnp.logical_and(
            ability == AbilityEnum.SLUSH_RUSH,
            weather == WeatherEnum.SNOW).squeeze()
        is_sand_rush = jnp.logical_and(
            ability == AbilityEnum.SAND_RUSH,
            weather == WeatherEnum.SANDSTORM).squeeze()
        arr = jnp.array([is_chlorophyll, is_slush_rush, is_sand_rush, is_swift_swim, is_surge_surfer])
        temp = conditional_mult_round(
            stats[..., StatEnum.SPEED],
            2,
            jnp.any(arr))


        # paralyzed
        is_paralyzed = (status==Status.PARALYZE).squeeze()
        # quick feet
        is_quick_feet = jnp.logical_and(ability==AbilityEnum.QUICK_FEET, active.has_status).squeeze()
        temp = conditional_mult_round(temp, 1.5, is_quick_feet)
        # apply paralyze speed drop if paralyzed and not quick feet
        temp = conditional_mult_round(temp, 0.5, jnp.logical_and(is_paralyzed, 1-is_quick_feet))

        stats = stats.at[..., StatEnum.SPEED].set(temp)


        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6+self.boosts.normal_boosts])

    @property
    def accuracy_boosts(self):
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        return self.active.current_hp

    def legal_switch_mask(self):
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]


def set_boosts(state: BattleState, side_idx, new_boosts):
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    return state.replace(boosts=new_boosts)

def clear_boosts(state: BattleState, side_idx) -> BattleState:
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idx, val) -> BattleState:
    # contrary
    val = val * (2 * (state.active.ability[side_idx]!=AbilityEnum.CONTRARY)[0]-1)
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idx].add(val)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def reduce_boosts(state: BattleState, side_idx, idx, val, bounceable=False) -> BattleState:
    ability = state.active.ability
    not_clear_body = ability[side_idx]!=AbilityEnum.CLEAR_BODY
    # TODO: this doesn't account for the functionality of idx being an array
    # in which case it should apply len(idx) times but that is hard to implement in JAX
    is_defiant = ability[side_idx]==AbilityEnum.DEFIANT
    is_competitive = ability[side_idx]==AbilityEnum.COMPETITIVE
    jax.lax.cond(is_defiant,
                 add_boosts,
                 lambda b, s, i, v: s,
                 state, side_idx, StatEnum.ATTACK, 2)
    jax.lax.cond(is_competitive,
                 add_boosts,
                 lambda b, s, i, v: s,
                 state, side_idx, StatEnum.SPECIAL_ATTACK, 2)
    is_mirror_armor = ability[side_idx] = AbilityEnum.MIRROR_ARMOR
    return add_boosts(state, side_idx, idx, -val * not_clear_body)

@partial(jit, static_argnames=['cond'])
def conditional_set_boosts(state: BattleState, side_idx, new_boosts, cond):
    if cond:
        return set_boosts(state, side_idx, new_boosts)
    return state

@partial(jit, static_argnames=['cond'])
def conditional_add_boosts(state: BattleState, side_idx, cond, idx, val) -> BattleState:
    if cond:
        return add_boosts(state, side_idx, idx, val)
    return state

@partial(jit, static_argnames=['cond'])
def conditional_reduce_boosts(state: BattleState, side_idx, cond, idx, val, bounceable) -> BattleState:
    if cond:
        return reduce_boosts(state, side_idx, idx, val, bounceable)
    return state

def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    new_team = state.team.replace_row((side_idx, state[side_idx].active_index), new_mon)
    return state.replace(team=new_team)

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    # TODO
    return state





