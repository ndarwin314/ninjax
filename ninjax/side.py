from typing import Union, Tuple, Dict, Any
from copy import deepcopy

import chex
import jax.lax
from flax import struct
import jax.numpy as jnp

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import Type, StatEnum, Status
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, calculate_effectiveness_multiplier

@struct.dataclass
class VolatileStatus:
    confused: bool = False
    # TODO this is gonna suck, make sure everything as default values

@struct.dataclass
class SideState:
    team: list[Pokemon]
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: int = 0
    stealth_rocks: bool = False
    sticky_webs: bool = False
    spikes: int = 0
    toxic_spikes: int = 0
    reflect: int = 0
    light_screen: int = 0
    aurora_veil: int = 0
    tailwind: int = 0
    toxic_counter: int = 0
    boosts: StatBoosts = StatBoosts()
    volatile_status: VolatileStatus = VolatileStatus() # TODO
    # notably this needs like wish, healing wish, and future sight things
    # but those are lowish priority


    @property
    def active(self) -> Pokemon:
        return jax.lax.switch(self.active_index, [lambda : self.team[i] for i in range(6)])

    @property
    def boosted_stats(self):
        return self.active.stats * STAT_MULTIPLIER_LOOKUP[self.boosts.normal_boosts]

    @property
    def accuracy_boosts(self):
        return self.boosts.acc_boosts

def clear_boosts(side: SideState) -> SideState:
    return side.replace(boosts=StatBoosts())

def clear_volatile_status(side: SideState) -> SideState:
    return side.replace(volatile_status=VolatileStatus())

def update_boosts(side: SideState, idxs, vals) -> SideState:
    return side.replace(boosts=jax.lax.clamp(-6, side.boosts.normal_boosts.at[idxs].set(vals), 6))

def add_boosts(side: SideState, idxs, vals) -> SideState:
    return update_boosts(side, idxs, vals+side.boosts.normal_boosts)

def update_pokemon_at_index(side: SideState, index: int, new_mon: Pokemon) -> SideState:
    new_team = deepcopy(side.team)
    for i in range(6):
        new_team[i] = jax.lax.cond(i==index, lambda:  new_mon, lambda: new_team[i])
    return side.replace(team=new_team)

def update_active(side: SideState, new_mon: Pokemon) -> SideState:
    return update_pokemon_at_index(side, side.active_index, new_mon)


# TODO: at some point probably factor out part of this into like
# just swapping out to implement baton pass idk
def swap_out(
    side: SideState,
    new_active: int
) -> (chex.PRNGKey, SideState):
    # swaps the active pokemon and does appropriate things like
    # 1. clearing volatile statuses
    # 2. resting boosts
    # 3. probably stuff im forgetting
    # 4. ahhh palafin, ahhh regenerator
    side = clear_volatile_status(clear_boosts(side))
    side = side.replace(active_index=new_active, toxic_counter=0)
    # hazards
    active = side.active
    is_not_flying = 1 - active.is_floating
    is_not_hazard_immune = 1 - active.is_hazard_immune
    side = take_damage_percent(
        side,
        side.stealth_rocks * calculate_effectiveness_multiplier(Type.ROCK, side.active.type_list) / 8)
    side = take_damage_percent(
        side,
        (side.spikes != 0) / (10 - 2 * side.spikes) * is_not_flying
    )
    side = add_boosts(side, StatEnum.SPEED, -1 * is_not_flying)
    no_status = active.status != Status.NONE
    # only remove is poison type and not floating
    is_poison = active.is_type(Type.POISON)
    toxic_spikes = side.toxic_spikes * (1 - jnp.logical_and(is_poison, is_not_flying))
    side = side.replace(toxic_spikes=toxic_spikes)
    is_poison_immune = jnp.any(side.active.type_list == Type.STEEL)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - side.toxic_spikes) * (side.toxic_spikes != 0)
    new_mon = active.replace(status=status_ * no_status * is_poison_immune * is_not_flying)
    side = update_active(side, new_mon)
    return side



def step_side(
    key: chex.PRNGKey,
    side: SideState,
) -> (chex.PRNGKey, SideState):
    toxic_counter = (side.toxic_counter + 1) * (side.active.status==Status.TOXIC)
    side.replace(
        reflect=jnp.maximum(side.reflect-1, 0),
        light_screen=jnp.maximum(side.light_screen-1, 0),
        aurora_veil=jnp.maximum(side.aurora_veil-1, 0),
        tailwind=jnp.maximum(side.tailwind-1, 0),
        toxic_counter=toxic_counter
    )
    return key, side

def take_damage_value(side: SideState, damage: chex.Array) -> SideState:
    active = side.active
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    # i guess i need to figure out that bullshit later but that is also low priority
    new_health = jax.lax.clamp(0, active.current_hp-damage, active.max_hp)
    is_alive = new_health != 0
    active.replace(current_hp=new_health, is_alive=is_alive)
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    new_active = (side.active_index + 1) * (active.current_hp == 0) - 1
    side = update_pokemon_at_index(side, side.active_index, active)
    return side.replace(active_index=new_active)

def take_damage_percent(side: SideState, percent: chex.Array) -> SideState:
    damage = jnp.round(side.active.max_hp * percent)
    return take_damage_value(side, damage)
