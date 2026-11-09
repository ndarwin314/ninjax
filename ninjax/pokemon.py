import jax
from flax import struct
from ninjax.tree import ArrayTree, broadcast_tree
from dataclasses import field

import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable, StatBoosts
from ninjax.enum_types import StatEnum, Type, Status, AbilityEnum, MoveType
from ninjax.utils import triple_or, quad_or, calculate_effectiveness_multiplier

BURN_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.WATER_VEIL, AbilityEnum.WATER_BUBBLE, AbilityEnum.THERMAL_EXCHANGE])
PARALYSIS_IMMUNE_ABILITIES = jnp.array([AbilityEnum.LIMBER, AbilityEnum.PURIFYING_SALT])
POISON_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.IMMUNITY])
FREEZE_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.MAGMA_ARMOR])
SLEEP_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.INSOMNIA])

@struct.dataclass
class Pokemon(ArrayTree):
    """Team leaves use (2 players, 6 slots, features); moves add a move axis."""
    type_list: jax.Array
    moves: Move
    tera_type: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    #species: jax.Array = jnp.int32([0])
    #name: jax.Array = jnp.int32([0])
    #level: jax.Array = jnp.int32([0])
    is_alive: jax.Array = field(default_factory=lambda: jnp.bool([1]))
    gender: jax.Array = field(default_factory=lambda: jnp.bool([0]))
    is_terastallized: jax.Array = field(default_factory=lambda: jnp.bool([0]))
    status: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    ability: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    sleep_counter: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    #item: jax.Array = jnp.int32([0])
    stat_table: StatTable = field(default_factory=StatTable)
    current_hp: jax.Array = None
    # TODO add stats conditions and volatile status conditions

    def __post_init__(self):
        batch = jnp.shape(self.type_list)[:-1]
        object.__setattr__(self, 'type_list', jnp.asarray(self.type_list, jnp.int32))
        object.__setattr__(self, 'stat_table', broadcast_tree(self.stat_table, batch))
        for name in ('tera_type', 'is_alive', 'gender', 'is_terastallized', 'status', 'ability', 'sleep_counter', 'current_hp'):
            value = getattr(self, name)
            if value is None:
                value = self.stat_table.stats[..., :1]
            dtype = jnp.bool_ if name in ('is_alive', 'gender', 'is_terastallized') else jnp.int32
            object.__setattr__(self, name, jnp.broadcast_to(jnp.asarray(value, dtype), batch + (1,)))

    def replace_row(self, idx, new_pokemon):
        return jax.tree.map(lambda old, new: old.at[idx].set(new), self, new_pokemon)

    @property
    def stats(self):
        return self.stat_table.stats.squeeze()

    @property
    def max_hp(self):
        # TODO: idk if this works long term but im doing this as a hack right now
        return self.stat_table.stats[...,0]

    @property
    def hp_percent(self):
        return self.current_hp.squeeze() / self.max_hp

    def hp_less_than(self, percent):
        return jnp.less_equal(self.hp_percent, percent)

    def hp_greater_than(self, percent):
        return jnp.greater(self.hp_percent, percent)

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
        return triple_or(self.is_type(Type.POISON), self.is_type(Type.STEEL), jnp.any(POISON_IMMUNE_ABILITIES==self.ability))

    @property
    def is_paralyze_immune(self):
        return jnp.logical_or(self.is_type(Type.ELECTRIC),jnp.any(PARALYSIS_IMMUNE_ABILITIES==self.ability))

    @property
    def is_burn_immune(self):
        return jnp.logical_or(self.is_type(Type.FIRE), jnp.any(BURN_IMMUNE_ABILITIES==self.ability))

    @property
    def is_freeze_immune(self):
        return jnp.logical_or(self.is_type(Type.ICE), jnp.any(FREEZE_IMMUNE_ABILITIES==self.ability))

    @property
    def is_sleep_immune(self):
        return jnp.any(SLEEP_IMMUNE_ABILITIES==self.ability)

    @property
    def is_powder_immune(self):
        # add check for goggles
        return jnp.logical_or(self.is_type(Type.GRASS), self.ability==AbilityEnum.OVERCOAT)

    @property
    def is_sound_immune(self):
        return self.ability==AbilityEnum.SOUNDPROOF

    @property
    def is_bulletproof(self):
        return self.ability==AbilityEnum.BULLET_PROOF

    @property
    def is_status_immune(self):
        return self.ability==AbilityEnum.GOOD_AS_GOLD

    @property
    def has_status(self):
        return self.status!=Status.NONE

    @property
    def is_poisoned(self):
        return jnp.logical_and(self.status==Status.POISON, self.status==Status.TOXIC)

    @property
    def level(self):
        return self.stat_table.level

    def is_immune_to_move(self, move):
        type_immunity = calculate_effectiveness_multiplier(move.type, self.type_list)
        return jnp.any(jnp.array([
            jnp.logical_and(self.is_powder_immune, move.powder),
            jnp.logical_and(self.is_sound_immune, move.sound),
            jnp.logical_and(self.is_bulletproof, move.bullet),
            jnp.logical_and(self.is_status_immune, move.move_type==MoveType.STATUS),
            type_immunity]))




