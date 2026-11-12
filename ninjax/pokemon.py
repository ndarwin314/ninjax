import jax
from ninjax.tree import ArrayTree, broadcast_tree
from dataclasses import field

import jax.numpy as jnp

from ninjax.move import Move
from ninjax.stats import StatTable
from ninjax.enum_types import Type, Status

class Pokemon(ArrayTree):
    """Team leaves use (2 players, 6 slots, features); moves add a move axis."""
    type_list: jax.Array
    moves: Move
    base_types: jax.Array = None
    species: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    # Form identity is independent of species number and mutable typing.
    form: jax.Array = field(default_factory=lambda: jnp.int32([0]))
    # Integer hectograms avoid floating threshold drift in weight-based moves.
    weight_hg: jax.Array = field(default_factory=lambda: jnp.int32([100]))
    item: jax.Array = field(default_factory=lambda: jnp.int32([0]))
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
        """Broadcast Pokémon features across its entity axes."""
        batch = jnp.shape(self.type_list)[:-1]
        object.__setattr__(self, 'type_list', jnp.asarray(self.type_list, jnp.int32))
        object.__setattr__(self, 'base_types', jnp.asarray(self.type_list if self.base_types is None else self.base_types, jnp.int32))
        object.__setattr__(self, 'stat_table', broadcast_tree(self.stat_table, batch))
        for name in ('species', 'form', 'weight_hg', 'item', 'tera_type', 'is_alive', 'gender', 'is_terastallized', 'status', 'ability', 'sleep_counter', 'current_hp'):
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
        return jnp.any(self.type_list==t, axis=-1, keepdims=True)

    @property
    def has_status(self):
        return self.status!=Status.NONE

    @property
    def is_poisoned(self):
        return jnp.logical_and(self.status==Status.POISON, self.status==Status.TOXIC)

    @property
    def level(self):
        return self.stat_table.level
