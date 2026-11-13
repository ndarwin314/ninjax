"""Immutable singles battle state and helpers for side and field updates.

The leading axis of a full battle is the two-player axis. Helpers return new
PyTrees, and random helpers take and return an explicit JAX key. The mechanics
implemented here are a subset of the intended battle rules.
"""

from typing import Tuple
from dataclasses import field

import jax.lax
import jax.random as random
import chex
from flax import struct
import jax.numpy as jnp

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType, AbilityEnum, Weather, Terrain, Type, MoveType, OffensiveStatOwner
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, triple_or, conditional_mult_round, triple_and, calculate_effectiveness_multiplier

from ninjax.events import EventBuffer, EventKind, Cause, emit
from ninjax.hit import HitResults

Array = jax.Array

BURN_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.WATER_VEIL, AbilityEnum.WATER_BUBBLE, AbilityEnum.THERMAL_EXCHANGE])
PARALYSIS_IMMUNE_ABILITIES = jnp.array([AbilityEnum.LIMBER, AbilityEnum.PURIFYING_SALT])
POISON_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.IMMUNITY])
FREEZE_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.MAGMA_ARMOR])
SLEEP_IMMUNE_ABILITIES = jnp.array([AbilityEnum.PURIFYING_SALT, AbilityEnum.INSOMNIA, AbilityEnum.VITAL_SPIRIT])


def is_floating(mon):
    """Check Flying type and effective Levitate before field grounding effects."""
    return mon.is_type(Type.FLYING) | (mon.ability == AbilityEnum.LEVITATE)


def is_sand_immune(mon):
    """Check represented type and effective ability immunities to sandstorm."""
    return triple_or(mon.is_type(Type.STEEL), mon.is_type(Type.GROUND), mon.is_type(Type.ROCK)) | (mon.ability == AbilityEnum.SAND_FORCE)


def is_powder_immune(mon):
    """Check represented Grass and Overcoat immunities to powder effects."""
    return mon.is_type(Type.GRASS) | (mon.ability == AbilityEnum.OVERCOAT)


def is_immune_to_move(mon, move, gravity_active=False):
    """Check represented move immunities using a resolved active snapshot."""
    types = jnp.where(jnp.asarray(gravity_active) & (move.type == Type.GROUND) &
                      (mon.type_list == Type.FLYING), Type.NONE, mon.type_list)
    checks_type = (move.move_type != MoveType.STATUS) | (move.identifier == 86)
    type_immunity = (calculate_effectiveness_multiplier(move.type, types) == 0) & checks_type
    levitate = (mon.ability == AbilityEnum.LEVITATE) & (move.type == Type.GROUND) & ~jnp.asarray(gravity_active) & checks_type
    return jnp.any(jnp.stack(jnp.broadcast_arrays(
        is_powder_immune(mon) & move.powder,
        (mon.ability == AbilityEnum.SOUNDPROOF) & move.sound,
        (mon.ability == AbilityEnum.BULLET_PROOF) & move.bullet,
        (mon.ability == AbilityEnum.GOOD_AS_GOLD) & (move.move_type == MoveType.STATUS),
        type_immunity, levitate)))


def apply_confusion(key, state, player):
    """Apply a 2–5 counter confusion volatile when the target is eligible.

    Only the presence of confusion is public; its sampled duration is private.
    """
    v = state.volatile_status
    blocked = ((v.confusion_turns[player, 0] > 0) | (state.ability[player, 0] == AbilityEnum.OWN_TEMPO) |
               ((state.terrain.terrain == TerrainEnum.MISTY) & state.is_grounded(player)[0]))
    def apply(args):
        """Sample the volatile duration and publish its start."""
        key, state = args
        key, subkey = random.split(key)
        duration = random.randint(subkey, (), 2, 6)
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player, 0].set(duration)))
        return key, emit(state, EventKind.CONDITION, player, detail=12, before=0, after=1)
    return jax.lax.cond(blocked, lambda x: x, apply, (key, state))



class VolatileStatus(struct.PyTreeNode):
    """Switch-reset effects for each active side, with shape (2, 1).

    Confusion counts down on eligible action attempts; Transform remains reserved.
    """
    paradox_stat: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
    # -1 means the active Pokémon retains its caller-provided ability.
    ability_override: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
    ability_suppressed: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    paradox_booster: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    choice_lock: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
    protected: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    protection_kind: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    flinched: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    # Actual Pokémon or Substitute HP lost on the latest attack, before healing.
    last_damage: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    protect_chain: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    last_move_id: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    disabled_move_id: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    disable_turns: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    encore_move_id: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    encore_turns: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    taunt_turns: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    entry_turn: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    pending_move: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    pending_move_type: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    pending_actor_slot: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
    active_move_actions: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    # 0 = not attempted, 1 = failed, 2 = succeeded, 3 = skipped.
    move_result: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    previous_move_result: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    queued_priority: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    queued_priority_valid: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    encore_overridden: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    libero_used: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    flash_fire: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    confusion_turns: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    # Remaining Substitute HP; zero means no Substitute is present.
    substitute_hp: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    # A stored move ID forces its second turn, independently of remaining PP.
    charging_move_id: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    charge_target_side: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    charge_target_position: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    recharging: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    destiny_bond: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    roost: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    # Pre-faint HP for calculating an Explosion already committed this action.
    explosion_hp: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    sleep_talk_active: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    sleep_talk_succeeded: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    called_move_id: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    transformed: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))

    def __getitem__(self, idx):
        """Select one side of every volatile-state leaf."""
        return jax.tree.map(lambda value: value[idx], self)

    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        """Return a copy with one side's volatile leaves replaced."""
        return jax.tree.map(lambda old, new: old.at[idx].set(new), self, new_status)

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
    known_items: Array = field(default_factory=lambda: -jnp.ones((2, 6), dtype=jnp.int32))
    # Public current ability knowledge; cleared when a temporary change ends.
    known_abilities: Array = field(default_factory=lambda: -jnp.ones((2, 6), dtype=jnp.int32))
    events: EventBuffer = field(default_factory=EventBuffer.empty)
    hit_results: HitResults = field(default_factory=HitResults.empty)
    # Pending faint records: target side is the row; slots preserve identity.
    faint_pending: Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.bool_))
    faint_source: Array = field(default_factory=lambda: -jnp.ones(2, dtype=jnp.int32))
    faint_source_slot: Array = field(default_factory=lambda: -jnp.ones(2, dtype=jnp.int32))
    faint_target_slot: Array = field(default_factory=lambda: -jnp.ones(2, dtype=jnp.int32))
    faint_cause: Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.int32))
    faint_destiny_bond: Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.bool_))
    event_source: Array = field(default_factory=lambda: jnp.int32(-1))
    event_cause: Array = field(default_factory=lambda: jnp.int32(0))
    event_move: Array = field(default_factory=lambda: jnp.int32(0))
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    stealth_rocks: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    sticky_webs: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    spikes: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    toxic_spikes: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    reflect: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    light_screen: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    aurora_veil: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    tailwind: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    toxic_counter: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    boosts: StatBoosts = field(default_factory=lambda: StatBoosts(
        normal_boosts=jnp.zeros((2, 6) ,dtype='int32'),
        acc_boosts=jnp.zeros((2,2), dtype='int32')))
    legal_action_mask: Array = field(default_factory=lambda: jnp.ones((2, 15), dtype=jnp.bool_))
    can_tera: Array = field(default_factory=lambda: jnp.ones((2, 1), dtype=jnp.bool_))
    switch_requested: Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.bool_))
    roster: Array = field(default_factory=lambda: jnp.ones((2, 6), dtype=jnp.bool_))
    winner: Array = field(default_factory=lambda: jnp.int32(-1))
    perish_turns: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
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
        flattened_idx = jnp.squeeze(self.active_index)
        return self.team[[0,1], flattened_idx]

    @property
    def ability(self):
        """Effective active abilities, shape (2, 1), including volatile changes.

        Suppression removes the effect without erasing the underlying ability
        or override, so clearing it can restore the appropriate ability.
        """
        override = self.volatile_status.ability_override
        ability = jnp.where(override >= 0, override, self.active.ability)
        return jnp.where(self.volatile_status.ability_suppressed, AbilityEnum.NONE, ability)

    @property
    def effective_active(self):
        """Read-only active snapshots with effective abilities and Roost types.

        These snapshots must not be written back to the team: the stored team
        ability remains the original, including while an override is active.
        """
        active = self.active
        original_types = active.type_list
        types = original_types
        remove = self.volatile_status.roost & ~active.is_terastallized
        types = jnp.where(remove & (types == Type.FLYING), Type.NONE, types)
        # Preserve primary-type ordering for effects such as Revelation Dance.
        types = jnp.stack((jnp.where(types[:, 0] == Type.NONE, types[:, 1], types[:, 0]),
                           jnp.where(types[:, 0] == Type.NONE, Type.NONE, types[:, 1])), axis=1)
        types = types.at[:, 0].set(jnp.where(remove[:, 0] & (types[:, 0] == Type.NONE), Type.NORMAL, types[:, 0]))
        types = jnp.where(remove, types, original_types)
        return active.replace(ability=self.ability, type_list=types)

    def is_status_immune(self, player, status):
        """Check represented active status immunities using the effective ability."""
        from ninjax.held_effects import effective_weather
        mon = self.effective_active[player]
        ability = self.ability[player]
        immune = (((status == Status.POISON) | (status == Status.TOXIC)) &
                  (mon.is_type(Type.POISON) | mon.is_type(Type.STEEL) | jnp.any(POISON_IMMUNE_ABILITIES == ability)))
        immune |= (status == Status.PARALYZE) & (mon.is_type(Type.ELECTRIC) | jnp.any(PARALYSIS_IMMUNE_ABILITIES == ability))
        immune |= (status == Status.BURN) & (mon.is_type(Type.FIRE) | jnp.any(BURN_IMMUNE_ABILITIES == ability))
        immune |= (status == Status.FREEZE) & (mon.is_type(Type.ICE) | jnp.any(FREEZE_IMMUNE_ABILITIES == ability) |
                                                (effective_weather(self) == WeatherEnum.SUN))
        immune |= (status == Status.SLEEP) & jnp.any(SLEEP_IMMUNE_ABILITIES == ability)
        grounded = (~(mon.is_type(Type.FLYING) | (ability == AbilityEnum.LEVITATE))) | (self.gravity_duration > 0)
        immune |= grounded & ((self.terrain.terrain == TerrainEnum.MISTY) |
                              ((self.terrain.terrain == TerrainEnum.ELECTRIC) & (status == Status.SLEEP)))
        return immune | (ability == AbilityEnum.COMATOSE)

    def is_grounded(self, player):
        """Check grounding of an active player, including Gravity."""
        return ~is_floating(self.effective_active[player]) | (self.gravity_duration > 0)

    def is_immune_to_move(self, player, move):
        """Check an active player's move immunity in the current field."""
        return is_immune_to_move(self.effective_active[player], move, self.gravity_duration > 0)

    @property
    def raw_boosted_stats(self):
        """Apply stage multipliers to active stats without ability modifiers."""
        stats = self.active.stats
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

        # marvel scale
        temp = conditional_mult_round(stats[...,StatEnum.DEFENSE], 1.5,
                                      jnp.logical_and(ability==AbilityEnum.MARVEL_SCALE, active.has_status).squeeze())
        stats = stats.at[..., StatEnum.DEFENSE].set(temp)

        # surge surfer
        terrain = self.terrain.terrain
        is_surge_surfer = jnp.logical_and(
            ability==AbilityEnum.SURGE_SURFER,
            terrain==TerrainEnum.ELECTRIC).squeeze()

        # speed boosting weather abilities
        weather = self.weather.weather
        is_chlorophyll  = jnp.logical_and(
            ability==AbilityEnum.CHLOROPHYLL,
            weather==WeatherEnum.SUN).squeeze()
        is_swift_swim = jnp.logical_and(
            ability == AbilityEnum.SWIFT_SWIM,
            weather == WeatherEnum.RAIN).squeeze()
        is_slush_rush = jnp.logical_and(
            ability == AbilityEnum.SLUSH_RUSH,
            weather == WeatherEnum.SNOW).squeeze()
        is_sand_rush = jnp.logical_and(
            ability == AbilityEnum.SAND_RUSH,
            weather == WeatherEnum.SANDSTORM).squeeze()
        # Reduce over ability predicates, preserving the player axis so one
        # Pokémon's weather ability cannot affect its opponent's speed.
        arr = jnp.array([is_chlorophyll, is_slush_rush, is_sand_rush, is_swift_swim, is_surge_surfer])
        temp = conditional_mult_round(
            stats[..., StatEnum.SPEED],
            2,
            jnp.any(arr, axis=0))


        # paralyzed
        is_paralyzed = (status==Status.PARALYZE).squeeze()
        # quick feet
        is_quick_feet = jnp.logical_and(ability==AbilityEnum.QUICK_FEET, active.has_status).squeeze()
        temp = conditional_mult_round(temp, 1.5, is_quick_feet)
        # apply paralyze speed drop if paralyzed and not quick feet
        temp = conditional_mult_round(temp, 0.5, jnp.logical_and(is_paralyzed, 1-is_quick_feet))

        stats = stats.at[..., StatEnum.SPEED].set(temp)


        from ninjax.held_effects import modify_stats
        stats = jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6+self.boosts.normal_boosts])
        # Tailwind modifies stage-adjusted Speed; applying it before stages
        # loses Scarlet/Violet's intermediate rounding on odd Speed values.
        stats = stats.at[..., StatEnum.SPEED].multiply(jnp.where(self.tailwind[:, 0] > 0, 2, 1))
        return jax.vmap(lambda i, values: modify_stats(self, i, values))(jnp.arange(2), stats)

    @property
    def accuracy_boosts(self):
        """Return accuracy and evasion stages for both players, shape ``(2, 2)``."""
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        """Return the active Pokémon's current HP for each player."""
        return self.active.current_hp

    def legal_switch_mask(self):
        """Build the legacy six-slot switch mask for a side view.

        This helper returns a Python list and does not apply roster selection;
        the environment's action mask is the authoritative decision mask.
        """
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]


def set_boosts(state: BattleState, side_idx, new_boosts):
    """Replace one side's stat-stage record and return the updated state."""
    old = jnp.concatenate((state.boosts.normal_boosts[side_idx], state.boosts.acc_boosts[side_idx]))
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    state = state.replace(boosts=new_boosts)
    new = jnp.concatenate((new_boosts.normal_boosts[side_idx], new_boosts.acc_boosts[side_idx]))
    def record(stat, state):
        a, b = old[stat], new[stat]
        return emit(state, EventKind.BOOST, side_idx, stat, a, b, enabled=a != b)
    return jax.lax.fori_loop(0, 8, record, state)


def clear_boosts(state: BattleState, side_idx) -> BattleState:
    """Reset one side's normal and accuracy/evasion stages."""
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idx, val) -> BattleState:
    """Add normal stat stages, reversing their sign for Contrary.

    ``idx`` may be a scalar or an array of stat indices. ``StatBoosts`` clips
    the resulting stages to the supported range when constructed.
    """
    # contrary
    idx = jnp.asarray(idx)
    val = jnp.asarray(val) * (2 * (state.ability[side_idx]!=AbilityEnum.CONTRARY)[0]-1)
    if idx.ndim == 0:
        val = val.reshape(())
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idx].add(val)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def reduce_boosts(state: BattleState, side_idx, idx, val, bounceable=False) -> BattleState:
    """Apply the currently implemented stage-reduction and ability handling.

    Clear Body, Defiant, and Competitive handling is incomplete; in particular,
    the caller's source and ``bounceable`` are not yet used to distinguish all
    opposing and self-inflicted reductions.
    """
    ability = state.ability
    not_clear_body = (ability[side_idx]!=AbilityEnum.CLEAR_BODY)[0]
    # TODO: this doesn't account for the functionality of idx being an array
    # in which case it should apply len(idx) times but that is hard to implement in JAX
    is_defiant = ability[side_idx]==AbilityEnum.DEFIANT
    is_competitive = ability[side_idx]==AbilityEnum.COMPETITIVE
    state = jax.lax.cond(is_defiant.reshape(()),
                 add_boosts,
                 lambda b, s, i, v: b,
                 state, side_idx, StatEnum.ATTACK, 2)
    state = jax.lax.cond(is_competitive.reshape(()),
                 add_boosts,
                 lambda b, s, i, v: b,
                 state, side_idx, StatEnum.SPECIAL_ATTACK, 2)
    return add_boosts(state, side_idx, idx, -val * not_clear_body)

def conditional_set_boosts(state: BattleState, side_idx, new_boosts, cond):
    """Set stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), set_boosts,
                        lambda state, side_idx, new_boosts: state,
                        state, side_idx, new_boosts)

def conditional_add_boosts(state: BattleState, side_idx, cond, idx, val) -> BattleState:
    """Add stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), add_boosts,
                        lambda state, side_idx, idx, val: state,
                        state, side_idx, idx, val)

def conditional_reduce_boosts(state: BattleState, side_idx, cond, idx, val, bounceable) -> BattleState:
    """Reduce stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), reduce_boosts,
                        lambda state, side_idx, idx, val, bounceable: state,
                        state, side_idx, idx, val, bounceable)

# Reuse this update graph across hit, item, and status branches during tracing.
@jax.jit
def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    """Write a modified active Pokémon back into its team's current slot."""
    # Index only the needed leaves, rather than constructing a complete side
    # view and gathering every active Pokémon field on every update.
    slot = state.active_index[side_idx]
    new_team = state.team.replace_row((side_idx, slot), new_mon)
    old = state.team.status[side_idx, state.active_index[side_idx, 0], 0]
    state = state.replace(team=new_team)
    return emit(state, EventKind.STATUS, side_idx, before=old, after=new_mon.status[0], enabled=old != new_mon.status[0])

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    """Reset all represented switch-cleared effects for one active side."""
    volatile = jax.tree.map(lambda value: value.at[side_idx].set(jnp.zeros_like(value[side_idx])), state.volatile_status)
    volatile = volatile.replace(paradox_stat=volatile.paradox_stat.at[side_idx].set(-1),
                                choice_lock=volatile.choice_lock.at[side_idx].set(-1),
                                pending_actor_slot=volatile.pending_actor_slot.at[side_idx].set(-1),
                                ability_override=volatile.ability_override.at[side_idx].set(-1))
    return state.replace(volatile_status=volatile)


def set_terrain(
    state: BattleState, new_terrain: TerrainEnum, duration: int
    ):
    """Set a non-NONE terrain, retaining its duration if it already matches."""
    current_terrain = state.terrain.terrain
    requested = jnp.asarray(new_terrain, jnp.int32).reshape(())
    new_terrain = jnp.where(requested == TerrainEnum.NONE, current_terrain, requested)
    matching_terrain = current_terrain == new_terrain
    new_duration_terrain = duration * (1 - matching_terrain) + state.terrain.duration * matching_terrain
    state = state.replace(terrain=Terrain(new_terrain, new_duration_terrain))
    return state

def set_weather(
    state: BattleState, new_weather: WeatherEnum, duration: int):
    """Set non-NONE weather, retaining its duration if it already matches."""
    current_weather = state.weather.weather
    requested = jnp.asarray(new_weather, jnp.int32).reshape(())
    new_weather = jnp.where(requested == WeatherEnum.NONE, current_weather, requested)
    matching_weather = current_weather == new_weather
    # TODO: add item check for the weather stones
    new_duration_weather = duration * (1 - matching_weather) + state.weather.duration * matching_weather
    return state.replace(weather=Weather(new_weather, new_duration_weather))

def status_helper(key, active, status: Status):
    """Set an active Pokémon's status and initialize its sleep counter.

    The counter is sampled from 2 through 4 inclusive for sleep and set to zero
    for other statuses. The returned key has been advanced once.
    """
    key, sub_key = random.split(key, 2)
    # The move transition decrements this counter before checking sleep.
    turns = random.randint(sub_key, (1,), minval=2, maxval=5)[0]
    active = active.replace(status=status, sleep_counter=turns*(status==Status.SLEEP))
    return key, active



def set_status(key: chex.PRNGKey, state: BattleState, side_idx, status: Status) -> Tuple[chex.PRNGKey, BattleState]:
    """Attempt a major status, returning the key and updated battle state.

    Existing status, represented terrain/type/ability immunities, and a NONE status
    prevent the update. A key is consumed only when ``status_helper`` runs.
    """
    active = state.active[side_idx]
    already_statused = active.status != Status.NONE
    is_immune = state.is_status_immune(side_idx, status)
    key, active = jax.lax.cond(
        triple_or(already_statused, is_immune, status==Status.NONE)[0],
        lambda k, a, s: (k, a),
        status_helper,
        key, active, status)
    return key, update_active(state, side_idx, active)

# Damage is invoked by many residual/contact branches. Outlining the shared
# operation avoids retracing its complete ability/item/event graph at each call.
@jax.jit
def take_damage_value(state: BattleState, defender_idx: int, damage: Array, is_attack_damage, fixed_damage=False, source=-1) -> BattleState:
    """Apply signed HP damage to an active Pokémon and return updated state.

    Negative values heal. ``is_attack_damage`` controls the represented Sturdy,
    Multiscale, and Innards Out behavior; other damage-triggered effects remain
    incomplete. Fixed damage bypasses Multiscale but retains Sturdy/Sash.
    Fainted Pokémon are not marked alive again by healing. Direct attacks pass
    their source side explicitly for deferred faint effects.
    """
    return _take_damage_value(state, defender_idx, damage, is_attack_damage, True, fixed_damage, source)

def _take_damage_value(state, defender_idx, damage, is_attack_damage, allow_innards, fixed_damage=False, source=-1):
    """Implement HP updates; optionally allow a single Innards Out response."""
    active = state.active
    defender = active[defender_idx]
    attacker = active[1-defender_idx]
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    health_full = defender.current_hp==defender.max_hp
    is_sturdy = state.ability[defender_idx]==AbilityEnum.STURDY
    is_multiscale = triple_and(health_full, state.ability[defender_idx]==AbilityEnum.MULTISCALE, is_attack_damage) & ~jnp.asarray(fixed_damage)
    damage = conditional_mult_round(damage, 0.5, is_multiscale)
    old_health = defender.current_hp
    over_half = jnp.greater_equal(old_health/defender.max_hp, 0.5)
    new_health = jax.lax.clamp(0, old_health-damage, jnp.asarray(defender.max_hp).reshape(()))
    sturdy_survives = health_full & is_sturdy & jnp.asarray(is_attack_damage) & (damage > 0) & (new_health == 0)
    from ninjax.enum_types import ItemEnum
    sash = health_full & (defender.item == ItemEnum.FOCUS_SASH) & jnp.asarray(is_attack_damage) & (damage > 0) & (new_health == 0) & ~sturdy_survives
    new_health = jnp.maximum(new_health, (sturdy_survives | sash).astype(new_health.dtype))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(last_damage=v.last_damage.at[defender_idx].set(
        jnp.where(jnp.asarray(is_attack_damage), jnp.maximum(old_health-new_health, 0), v.last_damage[defender_idx]))))
    state = emit(state, EventKind.ITEM, defender_idx, detail=ItemEnum.FOCUS_SASH, enabled=sash)
    from ninjax.ability_effects import remember_item
    state = remember_item(state, defender_idx, ItemEnum.NONE, sash)
    defender = defender.replace(item=jnp.where(sash, ItemEnum.NONE, defender.item))
    under_half = jnp.less(new_health/defender.max_hp, 0.5)

    # check to make sure we don't accidentally revive a pokemon
    alive = jnp.logical_and(new_health != 0, defender.is_alive)
    defender = defender.replace(current_hp=new_health, is_alive=alive)

    new_health = defender.current_hp
    delta = (new_health - old_health).reshape(())
    # Public HP endpoints use the same rounding as observations.
    public_delta = (jnp.ceil(new_health / defender.max_hp * 100) - jnp.ceil(old_health / defender.max_hp * 100)) / 100
    state = emit(state, EventKind.HP, defender_idx, detail=jnp.asarray(is_attack_damage, jnp.int32),
                 before=old_health, after=new_health, hp_change=delta,
                 hp_fraction_change=public_delta, enabled=delta != 0)
    state = emit(state, EventKind.FAINT, defender_idx, enabled=(old_health > 0) & (new_health == 0))

    # run moxie type abilities
    is_soul_heart = state.ability[1-defender_idx]==AbilityEnum.SOUL_HEART
    is_moxie = state.ability[1-defender_idx]==AbilityEnum.MOXIE
    is_beast_boost = state.ability[1-defender_idx]==AbilityEnum.BEAST_BOOST
    is_grim_neigh = state.ability[1-defender_idx]==AbilityEnum.GRIM_NEIGH
    beast_boost_index = (1+jnp.argmax(attacker.stats))
    index = (beast_boost_index*is_beast_boost +
             is_moxie * StatEnum.ATTACK +
             jnp.logical_or(is_soul_heart, is_grim_neigh) * StatEnum.SPECIAL_ATTACK)
    # yeah idk why this needs double index thingy here
    cond = (jnp.logical_or(is_moxie, is_beast_boost) & ~alive & (attacker.current_hp > 0))[0]
    state = conditional_add_boosts(state, 1-defender_idx, cond, index, 1)

    # innards out
    is_innards_out = triple_and(1-alive, state.ability[defender_idx]==AbilityEnum.INNARDS_OUT, is_attack_damage) & (attacker.current_hp > 0)
    # This flag is a static Python value: tracing the response with it disabled
    # prevents an unbounded recursive JAX trace.
    if allow_innards:
        state = jax.lax.cond(
            is_innards_out[0],
            lambda b, i, d: _take_damage_value(b, i, d, False, False),
            lambda b, i, d: b,
            state, 1-defender_idx, old_health
        )
    # TODO: consider making separate functions for take damage value and take damage from attack
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    from ninjax.held_effects import after_hp_change
    state = after_hp_change(update_active(state, defender_idx, defender), defender_idx)
    return record_faint(state, defender_idx, source,
        jnp.where(is_attack_damage, Cause.MOVE,
            jnp.where(state.event_cause == Cause.MOVE, Cause.INDIRECT, state.event_cause)),
        (old_health[0] > 0) & (new_health[0] == 0))


def record_faint(state, target, source, cause, enabled):
    """Save a faint's identity and cause independently of observation events."""
    enabled = jnp.asarray(enabled).reshape(()) & ~state.faint_pending[target]
    source = jnp.asarray(source, jnp.int32)
    source_slot = jnp.where(source >= 0, state.active_index[jnp.maximum(source, 0), 0], -1)
    return state.replace(
        faint_pending=state.faint_pending.at[target].set(state.faint_pending[target] | enabled),
        faint_source=state.faint_source.at[target].set(jnp.where(enabled, source, state.faint_source[target])),
        faint_source_slot=state.faint_source_slot.at[target].set(jnp.where(enabled, source_slot, state.faint_source_slot[target])),
        faint_target_slot=state.faint_target_slot.at[target].set(jnp.where(enabled, state.active_index[target, 0], state.faint_target_slot[target])),
        faint_cause=state.faint_cause.at[target].set(jnp.where(enabled, cause, state.faint_cause[target])),
        faint_destiny_bond=state.faint_destiny_bond.at[target].set(jnp.where(enabled, state.volatile_status.destiny_bond[target, 0], state.faint_destiny_bond[target])))


def direct_faint(state, player, source, cause):
    """Faint an active Pokémon without damage protections or damage callbacks."""
    mon = state.active[player]
    alive = mon.current_hp[0] > 0
    previous_source, previous_cause = state.event_source, state.event_cause
    state = state.replace(event_source=jnp.asarray(source, jnp.int32), event_cause=jnp.asarray(cause, jnp.int32))
    state = emit(state, EventKind.HP, player, before=mon.current_hp, after=0, hp_change=-mon.current_hp,
        hp_fraction_change=-jnp.ceil(mon.hp_percent * 100) / 100, enabled=alive)
    state = emit(state, EventKind.FAINT, player, enabled=alive)
    state = update_active(state, player, mon.replace(current_hp=jnp.zeros_like(mon.current_hp), is_alive=jnp.zeros_like(mon.is_alive)))
    state = record_faint(state, player, source, cause, alive)
    return state.replace(event_source=previous_source, event_cause=previous_cause)


def clear_destiny_bond(state, player):
    """End the active volatile when its user acts, is interrupted or switches."""
    v = state.volatile_status
    active = v.destiny_bond[player, 0]
    state = state.replace(volatile_status=v.replace(destiny_bond=v.destiny_bond.at[player, 0].set(False)))
    return emit(state, EventKind.CONDITION, player, detail=16, before=1, after=0, enabled=active)


@jax.jit
def resolve_faints(state):
    """Resolve pending singles faint effects after the move's HP effects finish.

    Records identify the actual source and team slots. Destiny Bond directly
    faints a surviving opposing attacker; it is neither damage nor another
    opposing move and therefore cannot recurse or invoke damage protections.
    """
    def resolve(target, s):
        """Consume one pending faint and its captured Destiny Bond state."""
        source = s.faint_source[target]
        safe_source = jnp.maximum(source, 0)
        triggers = (s.faint_pending[target] & s.faint_destiny_bond[target] &
            (s.faint_cause[target] == Cause.MOVE) & (source >= 0) & (source != target) &
            (s.active_index[target, 0] == s.faint_target_slot[target]) &
            (s.active_index[safe_source, 0] == s.faint_source_slot[target]) &
            (s.active.current_hp[safe_source, 0] > 0))
        s = jax.lax.cond(s.faint_pending[target], lambda s: clear_destiny_bond(s, target), lambda s: s, s)
        s = s.replace(faint_pending=s.faint_pending.at[target].set(False),
                      faint_destiny_bond=s.faint_destiny_bond.at[target].set(False))
        def retaliate(s):
            """Commit the direct faint with an explicit Destiny Bond cause."""
            old_source, old_cause, old_move = s.event_source, s.event_cause, s.event_move
            s = s.replace(event_source=jnp.asarray(target, jnp.int32), event_cause=jnp.int32(Cause.DESTINY_BOND), event_move=jnp.int32(194))
            mon = s.active[safe_source]
            s = emit(s, EventKind.HP, safe_source, before=mon.current_hp, after=0,
                hp_change=-mon.current_hp, hp_fraction_change=-jnp.ceil(mon.hp_percent * 100)/100)
            s = emit(s, EventKind.FAINT, safe_source)
            s = update_active(s, safe_source, mon.replace(current_hp=jnp.zeros_like(mon.current_hp), is_alive=jnp.zeros_like(mon.is_alive)))
            s = clear_destiny_bond(s, target)
            s = clear_destiny_bond(s, safe_source)
            # In Gen 9 the last Pokémon to faint wins a simultaneous final KO.
            both_out = ~jnp.any((s.team.current_hp[..., 0] > 0) & s.roster)
            return s.replace(event_source=old_source, event_cause=old_cause, event_move=old_move,
                             winner=jnp.where(both_out, source, s.winner))
        return jax.lax.cond(triggers, retaliate, lambda s: s, s)
    return jax.lax.fori_loop(0, 2, resolve, state)

def boosted_stats_helper(stats, boosts, ignore_drops, ignore_boosts):
    """Apply stages after optionally ignoring negative or positive stages."""
    boosts = boosts.normal_boosts
    boosts = jnp.minimum(boosts, jnp.where(ignore_boosts, 0, 6))
    boosts = jnp.maximum(boosts, jnp.where(ignore_drops, 0, -6))
    return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + boosts])

def select_offensive_stats(raw_stats, boosts, attacker_idx, owner, defender_unaware, is_crit):
    """Select an active owner's raw stats and apply offensive-stage rules.

    ``raw_stats`` and ``boosts`` have leading player axes of length two.
    ``owner`` selects the user or target. The returned stat vector excludes
    item and ability modifiers; callers apply those using the move user.
    """
    owner_idx = jnp.where(owner == OffensiveStatOwner.TARGET, 1 - attacker_idx, attacker_idx)
    return boosted_stats_helper(raw_stats[owner_idx], boosts[owner_idx], is_crit | defender_unaware, defender_unaware)


def raw_boosted_stats(raw_stats, boosts, attacker_idx, attacker_unaware, defender_unaware, is_crit,
                      offensive_owner=OffensiveStatOwner.USER):
    """Return offensive-owner and defender stats with represented stage rules.

    Critical hits and Unaware alter which stages are applied. The offensive
    owner defaults to the user; callers apply ability and item modifiers later.
    """
    defender_stats = raw_stats[1 - attacker_idx]
    defender_boosts = boosts[1 - attacker_idx]
    attacker_stats = select_offensive_stats(raw_stats, boosts, attacker_idx, offensive_owner, defender_unaware, is_crit)
    defender_stats = boosted_stats_helper(
        defender_stats,
        defender_boosts,
        attacker_unaware,
        jnp.logical_or(is_crit, attacker_unaware)
    )
    return attacker_stats, defender_stats

def take_damage_percent(state: BattleState, defender_idx, percent: chex.Array, cause=None) -> BattleState:
    """Apply a signed fraction of max HP as indirect damage or healing.

    Nonzero fractions use at least one HP after rounding the magnitude down.
    An array of indices invokes a fixed JAX loop over both sides, sharing the
    HP-update graph instead of expanding it twice during compilation.
    """
    if cause is not None:
        original_cause = state.event_cause
        state = take_damage_percent(state.replace(event_cause=jnp.int32(cause)), defender_idx, percent)
        return state.replace(event_cause=original_cause)
    if jnp.ndim(defender_idx) > 0:
        fractions = jnp.broadcast_to(percent, (2, 1))
        return jax.lax.fori_loop(0, 2,
            lambda i, s: take_damage_percent(s, i, fractions[i]), state)
    amount = state.active.max_hp[defender_idx] * percent
    damage = (jnp.sign(amount) * jnp.maximum(1, jnp.floor(jnp.abs(amount)))).astype(jnp.int32)
    return take_damage_value(state, defender_idx, damage, False)





