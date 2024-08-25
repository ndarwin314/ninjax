from typing import Union, Tuple, Dict, Any

import chex
from flax import struct
import jax
import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum, Type, Status
from ninjax.utils import STAT_MULTIPLIER_LOOKUP
from .data.load_json import load_json

@struct.dataclass
class Pokemon:
    type_list: chex.Array
    moves: list[Move]
    tera_type: Type = 0
    species: int = 0
    name: int = 0
    level: int = 100
    is_alive = True
    gender: bool = False
    # this might need to be changed later for dumb shenanigans with moves adding types
    is_terastallized: bool = False
    ability: int = 0
    item: int = 0
    stat_table: StatTable = StatTable()
    current_hp: int = stat_table.current_hp
    max_hp: int = current_hp
    status: Status = Status.NONE
    # add stats conditions and volatile status conditions

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def max_hp(self):
        return self.stat_table.stats[0]

    def get_move(self, index: int):
        return jax.lax.switch(index, [lambda: self.moves[i] for i in range(4)])

    def is_type(self, t: Type):
        return jnp.any(self.type_list==t)

    @property
    def is_floating(self):
        # TODO: add checks for levitate and balloon
        return self.is_type(Type.FLYING)

    @property
    def is_hazard_immune(self):
        # check for boots
        return False

    @property
    def is_sand_immune(self):
        return jnp.any(
            self.type_list==Type.STEEL +
            self.type_list==Type.GROUND +
            self.type_list==Type.ROCK
        )

    @property
    def is_poison_immune(self):
        return jnp.logical_or(self.is_type(Type.POISON), self.is_type(Type.STEEL))

    @property
    def is_powder_immune(self):
        # add check for goggles
        return self.is_type(Type.GRASS)

def update_pokemon_at_index(pokemon: Pokemon, index: int, new_mon: Pokemon):
    return jax.lax.switch(index, [lambda: new_mon, lambda: pokemon])
    
def create_pokemon(species: str, moves: list, level: int):
    data = load_json('pokedex.json')
    move_list = load_json('gen9moves.json')
    if species not in data:
        raise ValueError(f"Species {species} not found in pokedex")
    species_data = data[species]
    type_list = chex.Array([species_data['types'][0], species_data['types'][1]])
    
    for move in moves:
        if move not in move_list:
            raise ValueError(f"Move {move} not found in move list")
    
    return Pokemon(
        type_list=type_list,
        moves=[Move(
            name=move_list[move]['name'],
            move_type=move_list[move]['type'],
            max_pp=move_list[move]['pp'],
            current_pp=move_list[move]['pp'],
            type=move_list[move]['category'],
            base_power=move_list[move]['basePower'],
            accuracy=move_list[move]['accuracy'],
            priority=move_list[move]['priority']
        ) for move in moves],
        species=species,
        level=level
    )



    


    


