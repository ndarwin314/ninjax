from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
from gymnax.environments.environment import Environment
import jax.numpy as jnp

from jaxcalibur.pokemon import Pokemon

@struct.dataclass
class SideState:
    # incredible code
    team: (Pokemon, Pokemon, Pokemon, Pokemon, Pokemon, Pokemon)
    stealth_rocks: bool
    sticky_webs: bool
    spikes: int
    toxic_spikes: int
    reflect: int
    light_screen: int
    aurora_veil: int



