from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import field

import jax.lax
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray, BoolArray
import dataclass_array as dca
from flax import struct
import jax.numpy as jnp

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType, Type, AbilityEnum
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, calculate_effectiveness_multiplier, conditional_mult_round

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])


@struct.dataclass
class VolatileStatus:
    confused: bool = False
    # TODO this is gonna suck, make sure everything has default values
    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        return self

@dca.dataclass_array(broadcast=True)
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
    legal_action_mask: jax.Array = field(default_factory=lambda: jnp.ones((2, 15)))
    can_tera: jax.Array = field(default_factory=lambda: jnp.ones(2))
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
        flattened_idx = (jnp.reshape(self.active_index, (2,)))
        return self.team[[0,1], flattened_idx]

    @property
    def boosted_stats(self):
        active = self.active
        stats = self.active.stats
        # squeeze removes dimensions with length 1 which makes this broadcast correctly
        # its probably going to be helpful to use this in other places
        is_guts = jnp.logical_and(active.ability==AbilityEnum.GUTS, active.status==Status.BURN).squeeze()
        temp = conditional_mult_round(stats[...,StatEnum.ATTACK], 1.5, is_guts)
        stats = stats.at[..., StatEnum.ATTACK].set(temp)
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


# TODO: these all need to be rewritten i think
def set_boosts(state: BattleState, side_idx, new_boosts):
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    return state.replace(boosts=new_boosts)

def clear_boosts(state: BattleState, side_idx) -> BattleState:
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idxs, vals) -> BattleState:
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idxs].add(vals)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    new_team = state.team.replace_row((side_idx, state[side_idx].active_index), new_mon)
    return state.replace(team=new_team)

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    # TODO
    return state





