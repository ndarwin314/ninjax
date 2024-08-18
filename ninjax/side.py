from typing import Union, Tuple, Dict, Any
from copy import deepcopy

import chex
from flax import struct
import jax.numpy as jnp

from ninjax.pokemon import Pokemon, clear_boosts, clear_volatile_status

@struct.dataclass
class SideState:
    # incredible code
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
    # notably this needs like wish, healing wish, and future sight things
    # but those are lowish priority


    @property
    def active(self) -> Pokemon:
        return self.team[self.active_index]

def update_pokemon_at_index(side: SideState, index: int, new_mon:Pokemon) -> SideState:
    new_team = deepcopy(side.team)
    new_team[index] = new_mon
    return side.replace(team=new_team)


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
    active = side.active
    active = clear_volatile_status(clear_boosts(active))
    side = update_pokemon_at_index(side, side.active_index, active)
    return side.replace(active_index=new_active)


def step_side(
    key: chex.PRNGKey,
    side: SideState,
) -> (chex.PRNGKey, SideState):
    # hazards should always be changed mid-turn, not end
    side.replace(
        reflect=max(side.reflect-1, 0),
        light_screen=max(side.light_screen-1, 0),
        aurora_veil=max(side.aurora_veil-1, 0),
        tailwind=max(side.tailwind-1, 0)
    )
    return key, side

def take_damage_value(side: SideState, damage: chex.Array) -> SideState:
    active = side.active
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    # i guess i need to figure out that bullshit later but that is also low priority
    active.replace(current_hp=max(active.current_hp-damage, 0))
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    new_active = (side.active_index + 1) * (active.current_hp == 0) - 1
    side = update_pokemon_at_index(side, side.active_index, active)
    return side.replace(active_index=new_active)

def take_damage_percent(side: SideState, percent: chex.Array) -> SideState:
    damage = jnp.round(side.active.max_hp * percent)
    return take_damage_value(side, damage)
