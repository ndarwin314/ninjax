from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax.numpy as jnp

from ninjax.pokemon import Pokemon

@struct.dataclass
class SideState:
    # incredible code
    team: (Pokemon, Pokemon, Pokemon, Pokemon, Pokemon, Pokemon)
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
