from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
from gymnax.environments.environment import Environment
import jax.numpy as jnp

from jaxcalibur.side import SideState


@struct.dataclass
class BattleState:
    turn: int
    agents: (SideState, SideState)
    field_state: "FieldState"



@struct.dataclass
class BattleParams:
    max_steps_in_episode: int = 1

class Battle(Environment[BattleState, BattleParams]):

    def __init__(self):
        pass


    def step_env(
        self,
        key: chex.PRNGKey,
        state: BattleState,
        action: Union[int, float, chex.Array],
        params: BattleParams,
    ) -> Tuple[chex.Array, BattleState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # the fun part :))))
        pass

    def reset_env(
        self, key: chex.PRNGKey, params: BattleParams
    ) -> Tuple[chex.Array, BattleState]:
        pass

