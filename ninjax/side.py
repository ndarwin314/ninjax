from typing import Union, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import field

import chex
import jax.lax
from dataclass_array import DataclassArray
from dataclass_array.typing import FloatArray, IntArray, BoolArray
import dataclass_array as dca
from flax import struct
import jax.numpy as jnp

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType, Type
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, calculate_effectiveness_multiplier

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
        return self.team[self.active_index]

    @property
    def boosted_stats(self):
        return self.active.stats * STAT_MULTIPLIER_LOOKUP[self.boosts.normal_boosts]

    @property
    def accuracy_boosts(self):
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        return self.active.stat_table.current_hp

    def legal_switch_mask(self):
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]


# TODO: these all need to be rewritten i think
def set_boosts(state: BattleState, side_idx, new_boosts):
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    return state.replace(stat_boosts=new_boosts)

def clear_boosts(state: BattleState, side_idx) -> BattleState:
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idxs, vals) -> BattleState:
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idxs].add(vals)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    new_team = state[side_idx].team.replace_row(state[side_idx].active_index, new_mon)
    return state.replace(team=new_team)

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    # TODO
    return state


# TODO: at some point probably factor out part of this into like
# just swapping out to implement baton pass idk
def swap_out(
    state: BattleState,
    side_idx,
    new_active: int
) -> (chex.PRNGKey, BattleState):
    # swaps the active pokemon and does appropriate things like
    # 1. clearing volatile statuses
    # 2. resting boosts
    # 3. probably stuff im forgetting
    # 4. ahhh palafin, ahhh regenerator

    # clear boosts and volatile status from active pokemon, maybe add baton pass check here?
    state = clear_volatile_status(clear_boosts(state, side_idx), side_idx)

    # update active pokemon index
    # this update could be saved since we need an update later but idk if it will matter
    active = state.active_index.at[side_idx].set(new_active)
    toxic_counter = state.toxic_counter.at[side_idx].set(0)
    state = state.replace(active=active, toxic_counter=toxic_counter)

    # hazards
    active = state[side_idx].team[new_active]
    is_not_flying = 1 - active.is_floating
    is_not_hazard_immune = 1 - active.is_hazard_immune
    no_status = active.status != Status.NONE
    is_poison = active.is_type(Type.POISON)
    is_poison_immune = jnp.any(active.type_list == Type.STEEL)
    # stealth rocks
    state = take_damage_percent(
        state,
        side_idx,
        state[side_idx].stealth_rocks * calculate_effectiveness_multiplier(Type.ROCK, side_idx, active.type_list) / 8
    )

    # spikes
    state = take_damage_percent(
        state, side_idx,
        (state[side_idx].spikes != 0) / (10 - 2 * state[side_idx].spikes) * is_not_flying
    )

    # sticky webs
    state = add_boosts(state, side_idx, StatEnum.SPEED, -1 * is_not_flying * state[side_idx].sticky_webs)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state[side_idx].toxic_spikes * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(toxic_spikes=toxic_spikes)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state[side_idx].toxic_spikes) * (state[side_idx].toxic_spikes != 0)
    state = set_status(state, side_idx, status_ * no_status * is_poison_immune * is_not_flying)

    #check if switch in is still alive
    active_hp = state[side_idx].active.stat_table.current_hp[side_idx]
    is_alive = active_hp != 0
    active = state.active[side_idx].replace(is_alive=is_alive)
    state = update_active(state, side_idx, active)
    return state



def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> (chex.PRNGKey, BattleState):
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state.replace(
        reflect=jnp.maximum(state.reflect - 1, 0),
        light_screen=jnp.maximum(state.light_screen - 1, 0),
        aurora_veil=jnp.maximum(state.aurora_veil - 1, 0),
        tailwind=jnp.maximum(state.tailwind - 1, 0),
        toxic_counter=toxic_counter
    )
    return key, state

def take_damage_value(state: BattleState, defender_idx: int, damage: chex.Array) -> BattleState:
    defending_side = state[defender_idx]
    active = defending_side.active
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    new_health = jax.lax.clamp(0, (active.stat_table.current_hp-damage), active.max_hp)
    # check to make sure we don't accidentally revive a pokemon
    alive = jnp.logical_and(jnp.bool([new_health != 0]), active.is_alive)
    active = active.replace(current_hp=new_health, is_alive=alive)
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    return update_active(state, defender_idx, active)

def take_damage_percent(side: BattleState, defender_idx , percent: chex.Array) -> BattleState:
    damage = jnp.round(side.active.max_hp * percent).astype(int)
    return take_damage_value(side, defender_idx, damage)

def set_status(state: BattleState, side_idx, status: Status):
    active = state[side_idx].active
    active = active.replace(status=status)
    return update_active(state, side_idx, active)

