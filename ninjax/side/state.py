"""Immutable battle PyTrees and read-only views; no reactive effect imports."""
from dataclasses import field
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from ninjax.calculations.modifiers import conditional_mult_round
from ninjax.calculations.stat_formulas import STAT_MULTIPLIER_LOOKUP
from ninjax.calculations.stats import StatBoosts
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Terrain, TerrainEnum, TurnType, Weather, WeatherEnum
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.pokemon import Pokemon
from ninjax.records.events import EventBuffer
from ninjax.records.hits import HitResults
from ninjax.side.queries import (
    BURN_IMMUNE_ABILITIES,
    FREEZE_IMMUNE_ABILITIES,
    PARALYSIS_IMMUNE_ABILITIES,
    POISON_IMMUNE_ABILITIES,
    SLEEP_IMMUNE_ABILITIES,
    effective_weather,
    is_floating_fields,
    move_immune_from_fields,
    modify_speed,
    modify_stats,
)


Array = jax.Array

def _roost_types(original_types, remove):
    """Remove Flying during Roost, preserving primary and secondary order."""
    types = jnp.where(remove & (original_types == Type.FLYING), Type.NONE, original_types)
    first, second = types[..., 0], types[..., 1]
    promote = first == Type.NONE
    primary = jnp.where(promote, second, first)
    primary = jnp.where(primary == Type.NONE, Type.NORMAL, primary)
    secondary = jnp.where(promote, Type.NONE, second)
    return jnp.where(remove, jnp.stack((primary, secondary), axis=-1), original_types)


def _negative_ones(shape, dtype):
    """Create negative sentinel arrays when a new battle state is constructed."""
    return -jnp.ones(shape, dtype=dtype)


def _initial_boosts():
    """Create fresh normal and accuracy/evasion stage arrays for both players."""
    return StatBoosts(normal_boosts=jnp.zeros((2, 6), dtype='int32'),
                      acc_boosts=jnp.zeros((2, 2), dtype='int32'))


def _select_volatile_row(value, *, index):
    """Select one player's row from a volatile leaf."""
    return value[index]


def _replace_volatile_row(old, new, *, index):
    """Replace one player's row in a volatile leaf."""
    return old.at[index].set(new)


def _modify_stats_row(player, values, *, state):
    """Apply the read-only item and ability stat modifiers for one player."""
    return modify_stats(state, player, values)


class VolatileStatus(struct.PyTreeNode):
    """Switch-reset effects for each active side, with shape (2, 1).

    Confusion counts down on eligible action attempts; Transform remains reserved.
    """
    paradox_stat: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    # -1 means the active Pokémon retains its caller-provided ability.
    ability_override: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    ability_suppressed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    paradox_booster: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    choice_lock: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    protected: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    protection_kind: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    flinched: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    # Actual Pokémon or Substitute HP lost on the latest attack, before healing.
    last_damage: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    protect_chain: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    last_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    disabled_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    disable_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    encore_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    encore_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    taunt_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    entry_turn: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    pending_move: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    pending_move_type: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    pending_actor_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    active_move_actions: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # 0 = not attempted, 1 = failed, 2 = succeeded, 3 = skipped.
    move_result: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    previous_move_result: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    queued_priority: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    queued_priority_valid: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    encore_overridden: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    libero_used: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    flash_fire: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    confusion_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # Remaining Substitute HP; zero means no Substitute is present.
    substitute_hp: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # A stored move ID forces its second turn, independently of remaining PP.
    charging_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    charge_target_side: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    charge_target_position: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    recharging: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    destiny_bond: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    roost: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    # Pre-faint HP for calculating an Explosion already committed this action.
    explosion_hp: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    sleep_talk_active: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    sleep_talk_succeeded: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    called_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    transformed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))

    def __getitem__(self, idx):
        """Select one side of every volatile-state leaf."""
        return jax.tree.map(partial(_select_volatile_row, index=idx), self)

    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        """Return a copy with one side's volatile leaves replaced."""
        return jax.tree.map(partial(_replace_volatile_row, index=idx), self, new_status)


@jax.jit
def _active_pokemon(team, active_index):
    """Gather active slots once in a shared graph instead of at every read.

    The returned Pokémon remains stored state, with no ability/type overrides.
    Keeping this gather outlined avoids duplicating every team leaf's indexing
    operations throughout move, item and status branches during tracing.
    """
    return team[jnp.arange(2), jnp.squeeze(active_index)]


class BattleState(struct.PyTreeNode):
    """State for one singles battle, without an outer environment batch axis.

    ``team`` has player and team-slot axes ``(2, 6)``; most side conditions
    and ``active_index`` have shape ``(2, 1)``. Normal stat stages have shape
    ``(2, 6)``, accuracy/evasion stages ``(2, 2)``, and the legacy action mask
    ``(2, 15)``. ``roster`` selects participating team slots. Weather, terrain,
    and turn counters describe the shared field. Independent battles can be
    batched by applying ``jax.vmap`` outside these player axes.

    ``winner`` is -1 until set, otherwise a player index; ``perish_turns`` uses
    -1 for an inactive counter. Mechanics and public item-reveal memory live
    here; player-relative observations are constructed separately.
    """
    team: Pokemon
    # Publicly revealed held items by owner and team slot; -1 means unknown.
    known_items: Array = field(default_factory=partial(_negative_ones, (2, 6), dtype=jnp.int32))
    # Public current ability knowledge; cleared when a temporary change ends.
    known_abilities: Array = field(default_factory=partial(_negative_ones, (2, 6), dtype=jnp.int32))
    events: EventBuffer = field(default_factory=EventBuffer.empty)
    hit_results: HitResults = field(default_factory=HitResults.empty)
    # Pending faint records: target side is the row; slots preserve identity.
    faint_pending: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    faint_source: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    faint_source_slot: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    faint_target_slot: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    faint_cause: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.int32))
    faint_destiny_bond: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    event_source: Array = field(default_factory=partial(jnp.int32, -1))
    event_cause: Array = field(default_factory=partial(jnp.int32, 0))
    event_move: Array = field(default_factory=partial(jnp.int32, 0))
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    stealth_rocks: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    sticky_webs: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    spikes: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    toxic_spikes: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    reflect: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    light_screen: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    aurora_veil: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    tailwind: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    toxic_counter: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    boosts: StatBoosts = field(default_factory=_initial_boosts)
    legal_action_mask: Array = field(default_factory=partial(jnp.ones, (2, 15), dtype=jnp.bool_))
    can_tera: Array = field(default_factory=partial(jnp.ones, (2, 1), dtype=jnp.bool_))
    switch_requested: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    roster: Array = field(default_factory=partial(jnp.ones, (2, 6), dtype=jnp.bool_))
    winner: Array = field(default_factory=partial(jnp.int32, -1))
    perish_turns: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    # notably volatile status needs like wish, healing wish, and future sight things
    # but those are lowish priority
    volatile_status: VolatileStatus = field(default_factory=VolatileStatus) # TODO

    weather: Weather = Weather(WeatherEnum.NONE, 0)
    terrain: Terrain = Terrain(TerrainEnum.NONE, 0)
    turn_number: int = 0
    trick_room_duration: int = 0
    gravity_duration: int = 0
    turn_type: TurnType = TurnType.STANDARD

    def __getitem__(self, idx):
        """Select side-indexed fields while retaining shared field values.

        This is a convenience view for side helpers, not a complete independent
        battle: fields not explicitly indexed here retain their original shape.
        """
        return self.replace(
            team=self.team[idx], active_index=self.active_index[idx],
            stealth_rocks=self.stealth_rocks[idx], sticky_webs=self.sticky_webs[idx],
            spikes=self.spikes[idx], toxic_spikes=self.toxic_spikes[idx],
            reflect=self.reflect[idx], light_screen=self.light_screen[idx],
            aurora_veil=self.aurora_veil[idx], tailwind=self.tailwind[idx],
            toxic_counter=self.toxic_counter[idx], boosts=self.boosts[idx],
            legal_action_mask=self.legal_action_mask[idx], can_tera=self.can_tera[idx],
            volatile_status=self.volatile_status[idx])


    def replace_row(self, idx: int, new_side: 'BattleState'):
        """Replace the side fields listed below; shared field values are retained.

        Fields not listed in the replacement, including roster and pending switch
        information, remain unchanged.
        """
        team = self.team.replace_row(idx, new_side.team)
        active_index = self.active_index.at[idx].set(new_side.active_index)
        stealth_rocks = self.stealth_rocks.at[idx].set(new_side.stealth_rocks)
        sticky_webs = self.sticky_webs.at[idx].set(new_side.sticky_webs)
        spikes = self.spikes.at[idx].set(new_side.spikes)
        toxic_spikes = self.toxic_spikes.at[idx].set(new_side.toxic_spikes)
        reflect = self.reflect.at[idx].set(new_side.reflect)
        light_screen = self.light_screen.at[idx].set(new_side.light_screen)
        aurora_veil = self.aurora_veil.at[idx].set(new_side.aurora_veil)
        tailwind = self.tailwind.at[idx].set(new_side.tailwind)
        toxic_counter = self.toxic_counter.at[idx].set(new_side.toxic_counter)
        boosts = self.boosts.replace_row(idx, new_side.boosts)
        volatile_status = self.volatile_status.replace_row(idx, new_side.volatile_status)
        return self.replace(
            team=team, active_index=active_index, stealth_rocks=stealth_rocks, sticky_webs=sticky_webs, spikes=spikes,
            toxic_spikes=toxic_spikes, reflect=reflect, light_screen=light_screen, aurora_veil=aurora_veil,
            tailwind=tailwind, toxic_counter=toxic_counter, boosts=boosts, volatile_status=volatile_status
        )


    @property
    def active(self) -> Pokemon:
        """Gather stored active Pokémon; their ability fields are unmodified.

        Battle mechanics should use ``ability`` or the read-only
        ``effective_active`` snapshot when evaluating ability effects.
        """
        return _active_pokemon(self.team, self.active_index)

    def active_field(self, name, player=None):
        """Gather one stored team array for a player or both active slots."""
        field = getattr(self.team, name)
        if player is None:
            return field[jnp.arange(2), self.active_index[:, 0]]
        return field[player, self.active_index[player, 0]]

    def active_move_field(self, name, player=None):
        """Read a stored move field for the requested active bank or both banks."""
        field = getattr(self.team.moves, name)
        if player is None:
            return field[jnp.arange(2), self.active_index[:, 0]]
        return field[player, self.active_index[player, 0]]

    def active_moves(self, player, move_index=None):
        """Read the active move bank or one move directly from the team."""
        slot = self.active_index[player, 0]
        if move_index is None:
            return self.team.moves[player, slot]
        return self.team.moves[player, slot, move_index]

    def active_stats(self, player=None):
        """Calculate stats after selecting the active slots' stat inputs."""
        if player is None:
            index = (jnp.arange(2), self.active_index[:, 0])
        else:
            index = (player, self.active_index[player, 0])
        return self.team.stat_table[index].stats

    @property
    def ability(self):
        """Effective active abilities, shape (2, 1), including volatile changes.

        Suppression removes the effect without erasing the underlying ability
        or override, so clearing it can restore the appropriate ability.
        """
        override = self.volatile_status.ability_override
        # This hot read needs one leaf, not the entire active Pokémon tree.
        stored = self.team.ability[jnp.arange(2), self.active_index[:, 0]]
        ability = jnp.where(override >= 0, override, stored)
        return jnp.where(self.volatile_status.ability_suppressed, AbilityEnum.NONE, ability)

    @property
    def effective_active(self):
        """Read-only active snapshots with effective abilities and Roost types.

        These snapshots must not be written back to the team: the stored team
        ability remains the original, including while an override is active.
        """
        active = self.active
        remove = self.volatile_status.roost & ~active.is_terastallized
        return active.replace(ability=self.ability, type_list=_roost_types(active.type_list, remove))

    def effective_types(self, player=None):
        """Read active types with Roost applied unless the Pokémon has Terastallized."""
        original_types = self.active_field('type_list', player)
        roost = self.volatile_status.roost
        remove = (roost if player is None else roost[player]) & ~self.active_field('is_terastallized', player)
        return _roost_types(original_types, remove)

    def is_status_immune(self, player, status):
        """Check status immunity using active types, ability, item and field."""
        types = self.effective_types(player)
        abilities = self.ability
        ability = abilities[player]
        item = self.active_field('item', player)
        sunny = effective_weather(self.weather.weather, abilities, self.active_field('current_hp')) == WeatherEnum.SUN

        immune = (((status == Status.POISON) | (status == Status.TOXIC)) &
                  (jnp.any((types == Type.POISON) | (types == Type.STEEL)) |
                   jnp.any(jnp.asarray(POISON_IMMUNE_ABILITIES) == ability)))
        immune |= (status == Status.PARALYZE) & (jnp.any(types == Type.ELECTRIC) | jnp.any(jnp.asarray(PARALYSIS_IMMUNE_ABILITIES) == ability))
        immune |= (status == Status.BURN) & (jnp.any(types == Type.FIRE) | jnp.any(jnp.asarray(BURN_IMMUNE_ABILITIES) == ability))
        immune |= (status == Status.FREEZE) & (jnp.any(types == Type.ICE) | jnp.any(jnp.asarray(FREEZE_IMMUNE_ABILITIES) == ability) | sunny)
        immune |= (status == Status.SLEEP) & jnp.any(jnp.asarray(SLEEP_IMMUNE_ABILITIES) == ability)
        immune |= (ability == AbilityEnum.LEAF_GUARD) & sunny
        grounded = ~is_floating_fields(types, ability, item) | (self.gravity_duration > 0)
        immune |= grounded & ((self.terrain.terrain == TerrainEnum.MISTY) |
                              ((self.terrain.terrain == TerrainEnum.ELECTRIC) & (status == Status.SLEEP)))
        return immune | (ability == AbilityEnum.COMATOSE)

    def is_grounded(self, player):
        """Check grounding of an active player, including Gravity."""
        airborne = is_floating_fields(self.effective_types(player), self.ability[player], self.active_field('item', player))
        return ~airborne | (self.gravity_duration > 0)

    def is_immune_to_move(self, player, move):
        """Check an active player's move immunity in the current field."""
        return move_immune_from_fields(self.effective_types(player), self.ability[player],
                                       self.active_field('item', player), move, self.gravity_duration > 0)

    @property
    def raw_boosted_stats(self):
        """Apply stage multipliers to active stats without ability modifiers."""
        stats = self.active_stats()
        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + self.boosts.normal_boosts])

    @property
    def boosted_stats(self):
        """Return active stats with the ability and status modifiers handled here.

        The result has player and stat axes ``(2, 6)``. This property is used
        for action ordering; the damage path separately applies its modifiers.
        Ability coverage is incomplete.
        """
        active = self.active
        ability = self.ability
        status = active.status
        stats = self.active.stats
        # squeeze removes dimensions with length 1 which makes this broadcast correctly
        # its probably going to be helpful to use this in other places

        # Marvel Scale changes the displayed defensive stat; the damage path
        # applies its own defensive-stat modifiers separately.
        temp = conditional_mult_round(stats[...,StatEnum.DEFENSE], 1.5,
                                      jnp.logical_and(ability==AbilityEnum.MARVEL_SCALE, active.has_status).squeeze())
        stats = stats.at[..., StatEnum.DEFENSE].set(temp)


        stats = jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6+self.boosts.normal_boosts])
        # Chain Speed ability/item/field modifiers after stage rounding.
        return jax.vmap(partial(_modify_stats_row, state=self))(jnp.arange(2), stats)

    @property
    def effective_speeds(self):
        """Calculate only the stage-adjusted Speed needed for battle order."""
        base = self.active_stats()[:, StatEnum.SPEED]
        stages = self.boosts.normal_boosts[:, StatEnum.SPEED]
        speed = jnp.floor(base * STAT_MULTIPLIER_LOOKUP[6 + stages])
        ability = self.ability
        weather = effective_weather(self.weather.weather, ability, self.active_field('current_hp'))
        return modify_speed(speed, self.active_field('item')[:, 0], ability[:, 0],
                            self.active_field('status')[:, 0], weather, self.terrain.terrain,
                            self.volatile_status.paradox_stat[:, 0], self.tailwind[:, 0])

    @property
    def accuracy_boosts(self):
        """Return accuracy and evasion stages for both players, shape ``(2, 2)``."""
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        """Return the active Pokémon's current HP for each player."""
        return self.active_field('current_hp')

    def legal_switch_mask(self):
        """Build the legacy six-slot switch mask for a side view.

        This helper returns a Python list and does not apply roster selection;
        the environment's action mask is the authoritative decision mask.
        """
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]
