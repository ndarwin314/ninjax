from flax import struct
from pettingzoo.utils.env import ParallelEnv
import jax.numpy as jnp


@struct.dataclass
class EnvState:
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 1

class Battle(ParallelEnv):

    def __init__(self):
        pass



    def render(self) -> None | jnp.ndarray | str | list:
        raise NotImplemented