from dataclasses import field
from dataclass_array import DataclassArray, dataclass_array
from dataclass_array.typing import IntArray, BoolArray

import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum, Type, Status, AbilityEnum
from ninjax.utils import triple_or

@dataclass_array(broadcast=True, cast_list=True, cast_dtype=True)
class Pokemon(DataclassArray):
    type_list: IntArray['*batch_size 2']
    moves: Move['*batch_size 4']
    tera_type: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    #species: IntArray['*batch_size'] = jnp.int32([0])
    #name: IntArray['*batch_size'] = jnp.int32([0])
    #level: IntArray['*batch_size 1'] = jnp.int32([0])
    is_alive: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.bool([1]))
    gender: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.bool([0]))
    is_terastallized: BoolArray['*batch_size 1'] = field(default_factory=lambda: jnp.bool([0]))
    status: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    ability: IntArray['*batch_size 1'] = field(default_factory=lambda: jnp.int32([0]))
    #item: IntArray['*batch_size'] = jnp.int32([0])
    stat_table: StatTable = StatTable()
    # this is a hack. we
    current_hp: IntArray['*batch_size 1'] = stat_table.stats[...,0].reshape(1)
    # add stats conditions and volatile status conditions

    def replace_row(self, idx, new_pokemon: "Pokemon"):
        # unfortunately this is literally the best way i can think of to do this
        # it saves ugliness everywhere else
        # with the cost of needing to update all fields manually like this
        type = self.type_list.at[idx].set(new_pokemon.type_list)
        # probably need to call a similar function for move to update it properly
        # the only time moves should need to be changed is transform and i dont need to deal with that yet
        #new_moves = self.moves.at[idx].set(new_pokemon.moves)
        #level = self.level.at[idx].set(new_pokemon.level)
        is_alive = self.is_alive.at[idx].set(new_pokemon.is_alive)
        gender = self.gender.at[idx].set(new_pokemon.gender)
        is_terastallized = self.is_terastallized.at[idx].set(new_pokemon.is_terastallized)
        status = self.status.at[idx].set(new_pokemon.status)
        ability = self.ability.at[idx].set(new_pokemon.ability)
        stat_table = self.stat_table.row_update(idx, new_pokemon.stat_table)
        current_hp = self.current_hp.at[idx].set(new_pokemon.current_hp)
        moves = self.moves.set_pp(idx, new_pokemon.moves.current_pp)
        new_obj = self.replace(
            type_list=type, is_alive=is_alive, gender=gender, is_terastallized=is_terastallized, status=status,
            stat_table=stat_table, current_hp=current_hp, ability=ability, moves=moves
        )
        return new_obj

    @property
    def stats(self):
        return self.stat_table.stats

    @property
    def max_hp(self):
        # TODO: idk if this works long term but im doing this as a hack right now
        return self.stat_table.stats[...,0]

    @property
    def hp_percent(self):
        return self.current_hp / self.max_hp

    def hp_less_than(self, percent):
        return jnp.less_equal(self.hp_percent, percent)

    # TODO: this method doesnt seem to work properly for stacked pokemon
    def is_type(self, t: Type):
        return jnp.any(self.type_list==t)

    @property
    def is_floating(self):
        # TODO: add check for balloon
        return jnp.logical_or(self.is_type(Type.FLYING), self.ability==AbilityEnum.LEVITATE)

    @property
    def is_hazard_immune(self):
        # check for boots
        return self.ability == AbilityEnum.MAGIC_GUARD

    @property
    def is_sand_immune(self):
        return triple_or(self.is_type(Type.STEEL), Type.GROUND, Type.ROCK)

    @property
    def is_poison_immune(self):
        return triple_or(self.is_type(Type.POISON), self.is_type(Type.STEEL), self.ability==AbilityEnum.IMMUNITY)

    @property
    def is_paralyze_immune(self):
        return jnp.logical_or(self.is_type(Type.ELECTRIC), self.ability==AbilityEnum.LIMBER)

    @property
    def is_burn_immune(self):
        return jnp.logical_or(self.is_type(Type.FIRE), self.ability==AbilityEnum.WATER_VEIL)

    @property
    def is_freeze_immune(self):
        return jnp.logical_or(self.is_type(Type.ICE), self.ability==AbilityEnum.MAGMA_ARMOR)

    @property
    def is_sleep_immune(self):
        return self.ability==AbilityEnum.INSOMNIA

    @property
    def is_powder_immune(self):
        # add check for goggles
        return jnp.logical_or(self.is_type(Type.GRASS), self.ability==AbilityEnum.OVERCOAT)

    @property
    def is_sound_immune(self):
        return self.ability==AbilityEnum.SOUNDPROOF

    @property
    def has_status(self):
        return self.status!=Status.NONE




