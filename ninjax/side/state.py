"""Immutable battle PyTrees and read-only views; no reactive effect imports."""
from dataclasses import field
from functools import partial
from math import prod
import warnings

import jax
import jax.numpy as jnp
from flax import struct

from ninjax.ability_rules import TRANSFORM_SUPPRESSED_ABILITIES, UNCHANGEABLE_ABILITIES
from ninjax.calculations.stat_formulas import STAT_MULTIPLIER_LOOKUP
from ninjax.calculations.stats import StatBoosts
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Terrain, TerrainEnum, TurnType, Weather, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import LockedMoveKind
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.item_rules import ability_shield_active, item_is_suppressed, plate_type
from ninjax.pokemon import Pokemon
from ninjax.records.battle_history import DamageHistory
from ninjax.records.events import EventBuffer
from ninjax.records.hits import HitResults
from ninjax.side.queries import (
    ability_against_move,
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


@jax.jit
def _effective_abilities(ability, hp, item, magic_room, embargo, acid, transformed):
    """Apply field and local suppression to two active ability identities."""
    immune = jnp.isin(ability, jnp.asarray(UNCHANGEABLE_ABILITIES))
    shield = ability_shield_active(item, magic_room, embargo)
    gas = (ability == AbilityEnum.NEUTRALIZING_GAS) & ~acid & ~transformed
    gas = gas & (hp > 0)
    suppressed = ~immune & (acid |
        (gas[::-1] & ~shield & (ability != AbilityEnum.NEUTRALIZING_GAS)))
    suppressed |= transformed & jnp.isin(ability, jnp.asarray(TRANSFORM_SUPPRESSED_ABILITIES))
    return jnp.where(suppressed, AbilityEnum.NONE, ability)


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


class MoveRestrictions(struct.PyTreeNode):
    """Switch-cleared move locks and restriction timers, each with shape (2, 1)."""

    choice_lock: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    disabled_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    disable_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    encore_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    encore_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    taunt_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    throat_chop_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    heal_block_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    tormented: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    imprisoning: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class TrappingState(struct.PyTreeNode):
    """Source-linked traps on each active target, with leaves of shape (2, 1)."""

    # Opposing party slot in singles. Negative means no source.
    binding_source_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    binding_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    binding_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    binding_divisor: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    lock_source_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    no_retreat: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class ActiveResiduals(struct.PyTreeNode):
    """Per-active condition lifetimes, shape (2, 1). Zero is inactive and -1 is indefinite."""

    aqua_ring: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    ingrain: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    leech_seed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    curse: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    salt_cure: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    syrup_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    syrup_source_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    yawn_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class SlotResiduals(struct.PyTreeNode):
    """Delayed recovery by active position, with leaves of shape (2, 1)."""

    wish_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    wish_hp: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class LockedMove(struct.PyTreeNode):
    """Persistent payload for repeated moves, with leaves of shape (2, 1)."""

    # Zero move ID means no lock. The slot binds the sequence to its actor.
    # The policy uses LockedMoveKind and determines how the sequence advances.
    kind: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    actor_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    target_side: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    target_position: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    turns_left: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    successful_hits: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class MoveContinuation(struct.PyTreeNode):
    """Charge, recharge and repeat state for each active side."""

    # A stored move ID forces its second turn, independently of remaining PP.
    charging_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # SemiInvulnerability enum, cleared on release, interruption or departure.
    semi_invulnerability: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    charge_target_side: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    charge_target_position: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    recharging: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    locked: LockedMove = field(default_factory=LockedMove)


class MoveProtection(struct.PyTreeNode):
    """Current Protect effect and consecutive-use counter, each with shape (2, 1)."""

    protected: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    protection_kind: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    protect_chain: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    stall_refreshed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class SideProtection(struct.PyTreeNode):
    """One-turn guards that stay with a side through switching."""

    quick_guard: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    wide_guard: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    mat_block: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class ActiveAbilityState(struct.PyTreeNode):
    """Temporary ability changes and activation state, each with shape (2, 1)."""

    # -1 means the active Pokémon retains its caller-provided ability.
    ability_override: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    ability_suppressed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    paradox_stat: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    paradox_booster: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    libero_used: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    tera_shell: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    flash_fire: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class QueuedMove(struct.PyTreeNode):
    """Pending action identity and priority, each with shape (2, 1)."""

    pending_move: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    pending_move_type: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    pending_actor_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    queued_priority: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    queued_priority_valid: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    last_action: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    # A redirected or called move keeps the queued action's priority.
    encore_overridden: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    preparation: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    focus_lost: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))


class MoveHistory(struct.PyTreeNode):
    """Most recent move and action results by active side, each with shape (2, 1)."""

    # Actual Pokémon or Substitute HP lost on the latest attack, before healing.
    last_damage: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    last_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    last_move_type: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    active_move_actions: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # 0 = not attempted, 1 = failed, 2 = succeeded, 3 = skipped.
    move_result: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    previous_move_result: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class CalledMove(struct.PyTreeNode):
    """Execution context for Sleep Talk, each with shape (2, 1)."""

    sleep_talk_active: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    sleep_talk_succeeded: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    called_move_id: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class ActiveProperties(struct.PyTreeNode):
    """Switch-cleared identity overlay by player, with raw stats shaped (2, 6)."""
    added_type: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    form: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    species: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    weight_hg: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    stats: Array = field(default_factory=partial(_negative_ones, (2, 6), dtype=jnp.int32))
    power_trick: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    copied_slots: Array = field(default_factory=partial(jnp.zeros, (2, 4), dtype=jnp.bool_))


class VolatileStatus(struct.PyTreeNode):
    """Switch-reset effects for each active side, with shape (2, 1).

    Confusion counts down on eligible action attempts.
    """
    restrictions: MoveRestrictions = field(default_factory=MoveRestrictions)
    trapping: TrappingState = field(default_factory=TrappingState)
    residuals: ActiveResiduals = field(default_factory=ActiveResiduals)
    continuation: MoveContinuation = field(default_factory=MoveContinuation)
    protection: MoveProtection = field(default_factory=MoveProtection)
    ability_state: ActiveAbilityState = field(default_factory=ActiveAbilityState)
    properties: ActiveProperties = field(default_factory=ActiveProperties)
    queued_move: QueuedMove = field(default_factory=QueuedMove)
    move_history: MoveHistory = field(default_factory=MoveHistory)
    called_move: CalledMove = field(default_factory=CalledMove)
    flinched: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    embargo_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    entry_turn: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    confusion_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # Remaining Substitute HP; zero means no Substitute is present.
    substitute_hp: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    destiny_bond: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    roost: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    # Pre-faint HP for calculating an Explosion already committed this action.
    explosion_hp: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    transformed: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    stockpile_count: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    # Net Defense and Special Defense stages credited to Stockpile.
    stockpile_stages: Array = field(default_factory=partial(jnp.zeros, (2, 2), dtype=jnp.int32))
    magnet_rise_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    focus_energy: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    lock_on_turns: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    lock_on_target_slot: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    defense_curl: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))

    def __getitem__(self, idx):
        """Select one side of every volatile-state leaf (deprecated)."""
        warnings.warn('VolatileStatus indexing is deprecated.', DeprecationWarning, stacklevel=2)
        return jax.tree.map(partial(_select_volatile_row, index=idx), self)

    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        """Return a copy with one side's volatile leaves replaced (deprecated)."""
        warnings.warn('VolatileStatus.replace_row is deprecated.', DeprecationWarning, stacklevel=2)
        return jax.tree.map(partial(_replace_volatile_row, index=idx), self, new_status)


@jax.jit
def _active_pokemon(team, active_index):
    """Gather active slots once in a shared graph instead of at every read.

    The returned Pokémon remains stored state, with no ability/type overrides.
    Keeping this gather outlined avoids duplicating every team leaf's indexing
    operations throughout move, item and status branches during tracing.
    """
    return team[jnp.arange(2), jnp.squeeze(active_index)]


class Hazards(struct.PyTreeNode):
    """Entry hazards by affected side, each with shape (2, 1)."""

    stealth_rocks: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    sticky_webs: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.bool_))
    spikes: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    toxic_spikes: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class SideConditions(struct.PyTreeNode):
    """Timed side conditions, each with shape (2, 1)."""

    reflect: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    light_screen: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    aurora_veil: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    tailwind: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    safeguard: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    mist: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))


class PendingFaints(struct.PyTreeNode):
    """Deferred faint records, with one row per target side."""

    # Pending faint records: target side is the row; slots preserve identity.
    pending: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    # Positive sequence within the pending faint batch; zero means empty.
    order: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.int32))
    source: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    source_slot: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    target_slot: Array = field(default_factory=partial(_negative_ones, 2, dtype=jnp.int32))
    cause: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.int32))
    destiny_bond: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))


class EventContext(struct.PyTreeNode):
    """Scalar attribution for events emitted by the current action or effect."""

    source: Array = field(default_factory=partial(jnp.int32, -1))
    cause: Array = field(default_factory=partial(jnp.int32, 0))
    move_id: Array = field(default_factory=partial(jnp.int32, 0))


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
    # Original move banks of the two active slots while their moves are copied.
    original_moves: Array | None = None
    last_executed_move: Array | None = None
    hazards: Hazards = field(default_factory=Hazards)
    side_conditions: SideConditions = field(default_factory=SideConditions)
    side_protection: SideProtection = field(default_factory=SideProtection)
    slot_residuals: SlotResiduals = field(default_factory=SlotResiduals)
    pending_faints: PendingFaints = field(default_factory=PendingFaints)
    event_context: EventContext = field(default_factory=EventContext)
    # Publicly revealed held items by owner and team slot; -1 means unknown.
    known_items: Array = field(default_factory=partial(_negative_ones, (2, 6), dtype=jnp.int32))
    # Public current ability knowledge; cleared when a temporary change ends.
    known_abilities: Array = field(default_factory=partial(_negative_ones, (2, 6), dtype=jnp.int32))
    events: EventBuffer = field(default_factory=EventBuffer.empty)
    hit_results: HitResults = field(default_factory=HitResults.empty)
    damage_history: DamageHistory = field(default_factory=DamageHistory)
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    toxic_counter: Array = field(default_factory=partial(jnp.zeros, (2, 1), dtype=jnp.int32))
    boosts: StatBoosts = field(default_factory=_initial_boosts)
    legal_action_mask: Array = field(default_factory=partial(jnp.ones, (2, 15), dtype=jnp.bool_))
    can_tera: Array = field(default_factory=partial(jnp.ones, (2, 1), dtype=jnp.bool_))
    switch_requested: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    revival_requested: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.bool_))
    # Pending EntryRecovery flags by active position, including application order.
    entry_recovery: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.int32))
    # Recycle memory is carried by the party slot, independent of its held item.
    ability_once_used: Array = field(default_factory=partial(jnp.zeros, (2, 6, 2), dtype=jnp.bool_))
    consumed_items: Array = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.int32))
    # Belch and Last Resort prerequisites by party slot and move slot.
    ate_berry: Array = field(default_factory=partial(jnp.zeros, (2, 6), dtype=jnp.bool_))
    used_move_ids: Array = field(default_factory=partial(jnp.zeros, (2, 6, 4), dtype=jnp.int32))
    original_used_move_ids: Array = field(default_factory=partial(jnp.zeros, (2, 4), dtype=jnp.int32))
    # Replacement continuation: 0 normal, 1 Baton Pass, 2 Shed Tail.
    switch_transfer: Array = field(default_factory=partial(jnp.zeros, 2, dtype=jnp.int32))
    roster: Array = field(default_factory=partial(jnp.ones, (2, 6), dtype=jnp.bool_))
    winner: Array = field(default_factory=partial(jnp.int32, -1))
    perish_turns: Array = field(default_factory=partial(_negative_ones, (2, 1), dtype=jnp.int32))
    volatile_status: VolatileStatus = field(default_factory=VolatileStatus)

    weather: Weather = Weather(WeatherEnum.NONE, 0)
    terrain: Terrain = Terrain(TerrainEnum.NONE, 0)
    turn_number: int = 0
    trick_room_duration: int = 0
    gravity_duration: int = 0
    magic_room_duration: int = 0
    wonder_room_duration: int = 0
    fairy_lock_turns: int = 0
    turn_type: TurnType = TurnType.STANDARD

    def __post_init__(self):
        """Allocate shape-compatible backup banks for temporary moves."""
        if self.original_moves is None:
            # One buffer keeps this backup out of the per-move PyTree leaf count.
            width = sum(prod(leaf.shape[2:]) for leaf in jax.tree.leaves(self.team.moves))
            object.__setattr__(self, 'original_moves', jnp.zeros((2, width), jnp.int32))
        if self.last_executed_move is None:
            object.__setattr__(self, 'last_executed_move', jnp.zeros(self.original_moves.shape[-1] // 4, jnp.int32))

    def __getitem__(self, idx):
        """Select side-indexed fields while retaining shared field values (deprecated).

        This is a convenience view for side helpers, not a complete independent
        battle: fields not explicitly indexed here retain their original shape.
        """
        warnings.warn('BattleState indexing is deprecated.', DeprecationWarning, stacklevel=2)
        select_row = partial(_select_volatile_row, index=idx)
        return self.replace(
            team=self.team[idx], active_index=self.active_index[idx],
            hazards=jax.tree.map(select_row, self.hazards),
            side_conditions=jax.tree.map(select_row, self.side_conditions),
            side_protection=jax.tree.map(select_row, self.side_protection),
            toxic_counter=self.toxic_counter[idx], boosts=self.boosts[idx],
            legal_action_mask=self.legal_action_mask[idx], can_tera=self.can_tera[idx],
            volatile_status=jax.tree.map(select_row, self.volatile_status))


    def replace_row(self, idx: int, new_side: 'BattleState'):
        """Replace the listed side fields, retaining shared values (deprecated).

        Fields not listed in the replacement, including roster and pending switch
        information, remain unchanged.
        """
        warnings.warn('BattleState.replace_row is deprecated.', DeprecationWarning, stacklevel=2)
        team = self.team.replace_row(idx, new_side.team)
        active_index = self.active_index.at[idx].set(new_side.active_index)
        replace_row = partial(_replace_volatile_row, index=idx)
        hazards = jax.tree.map(replace_row, self.hazards, new_side.hazards)
        side_conditions = jax.tree.map(replace_row, self.side_conditions, new_side.side_conditions)
        side_protection = jax.tree.map(replace_row, self.side_protection, new_side.side_protection)
        toxic_counter = self.toxic_counter.at[idx].set(new_side.toxic_counter)
        boosts = self.boosts.replace_row(idx, new_side.boosts)
        volatile_status = jax.tree.map(replace_row, self.volatile_status, new_side.volatile_status)
        return self.replace(
            team=team, active_index=active_index, hazards=hazards, side_conditions=side_conditions,
            side_protection=side_protection,
            toxic_counter=toxic_counter, boosts=boosts, volatile_status=volatile_status
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
        """Read raw stats after applying temporary stat replacements."""
        if player is None:
            index = (jnp.arange(2), self.active_index[:, 0])
        else:
            index = (player, self.active_index[player, 0])
        overrides = self.volatile_status.properties.stats
        overrides = overrides if player is None else overrides[player]
        return jnp.where(overrides >= 0, overrides, self.team.stat_table[index].stats)

    def active_identity(self, name, player=None):
        """Read effective form, species or weight, retaining immutable team identity."""
        overrides = getattr(self.volatile_status.properties, name)
        overrides = overrides if player is None else overrides[player]
        return jnp.where(overrides >= 0, overrides, self.active_field(name, player))

    @property
    def ability_identity(self):
        """Read current ability IDs before effect suppression."""
        override = self.volatile_status.ability_state.ability_override
        return jnp.where(override >= 0, override, self.active_field('ability'))

    @property
    def ability(self):
        """Effective active abilities, shape (2, 1), including volatile changes.

        Suppression removes the effect without erasing the underlying ability
        or override, so clearing it can restore the appropriate ability.
        """
        v = self.volatile_status
        return _effective_abilities(self.ability_identity, self.active_field('current_hp'),
            self.active_field('item'), self.magic_room_duration, v.embargo_turns,
            v.ability_state.ability_suppressed, v.transformed)

    def effective_item(self, player=None):
        """Read held items after Magic Room, Embargo and Klutz suppression."""
        ability = self.ability if player is None else self.ability[player]
        embargo = self.volatile_status.embargo_turns
        embargo = embargo if player is None else embargo[player]
        suppressed = item_is_suppressed(ability, self.magic_room_duration, embargo)
        return jnp.where(suppressed, ItemEnum.NONE, self.active_field('item', player))

    @property
    def effective_active(self):
        """Read-only active snapshots with effective abilities and Roost types.

        These snapshots must not be written back to the team: the stored team
        ability remains the original, including while an override is active.
        """
        return self.active.replace(ability=self.ability, type_list=self.effective_types(),
            species=self.active_identity('species'), form=self.active_identity('form'),
            weight_hg=self.active_identity('weight_hg'))

    @property
    def combat_active(self):
        """Read active snapshots with the added-type column used by mechanics."""
        return self.effective_active.replace(type_list=self.effective_types(include_added=True))

    def effective_types(self, player=None, *, include_added=False, apply_roost=True, ignore_tera=False):
        """Read current defensive types or the original types used for STAB."""
        original_types = self.active_field('type_list', player)
        v = self.volatile_status
        transformed = v.transformed if player is None else v.transformed[player]
        abilities = self.ability_identity if player is None else self.ability_identity[player]
        plate = plate_type(self.active_field('item', player))
        arceus = (self.active_identity('species', player) == 493) & (abilities == AbilityEnum.MULTITYPE) & ~transformed
        mapped = jnp.concatenate((jnp.where(plate == Type.NONE, Type.NORMAL, plate), jnp.zeros_like(plate)), axis=-1)
        original_types = jnp.where(arceus, mapped, original_types)
        roost = self.volatile_status.roost
        remove = (roost if player is None else roost[player]) & ~self.active_field('is_terastallized', player)
        remove &= apply_roost
        types = _roost_types(original_types, remove)
        if include_added:
            extra = self.volatile_status.properties.added_type
            extra = extra if player is None else extra[player]
            # Roost promotes a surviving added type instead of creating Normal.
            only_added = remove & (extra != Type.NONE) & jnp.all(
                (original_types == Type.NONE) | (original_types == Type.FLYING), axis=-1, keepdims=True)
            types = jnp.where(only_added, jnp.zeros_like(types), types)
            types = jnp.concatenate((types, extra), axis=-1)
        if not ignore_tera:
            tera = self.active_field('tera_type', player)
            transformed = self.active_field('is_terastallized', player) & (tera != Type.NONE)
            types = jnp.where(transformed, jnp.zeros_like(types).at[..., 0].set(tera[..., 0]), types)
        return types

    def is_status_immune(self, player, status, *, ability=None, check_terrain=True):
        """Check status immunity using active types, ability, item and field."""
        types = self.effective_types(player, include_added=True)
        abilities = self.ability
        if ability is None:
            ability = abilities[player]
        item = self.active_field('item', player)
        suppressed = item_is_suppressed(ability, self.magic_room_duration, self.volatile_status.embargo_turns[player])
        item = jnp.where(suppressed, ItemEnum.NONE, item)
        sunny = effective_weather(self.weather.weather, abilities, self.active_field('current_hp')) == WeatherEnum.SUN

        immune = (((status == Status.POISON) | (status == Status.TOXIC)) &
                  (jnp.any((types == Type.POISON) | (types == Type.STEEL)) |
                   jnp.any(jnp.asarray(POISON_IMMUNE_ABILITIES) == ability)))
        immune |= (status == Status.PARALYZE) & (jnp.any(types == Type.ELECTRIC) | jnp.any(jnp.asarray(PARALYSIS_IMMUNE_ABILITIES) == ability))
        immune |= (status == Status.BURN) & (jnp.any(types == Type.FIRE) | jnp.any(jnp.asarray(BURN_IMMUNE_ABILITIES) == ability))
        immune |= (status == Status.FREEZE) & (jnp.any(types == Type.ICE) | jnp.any(jnp.asarray(FREEZE_IMMUNE_ABILITIES) == ability) | sunny)
        uproar = jnp.any((self.volatile_status.continuation.locked.kind[:, 0] == LockedMoveKind.UPROAR) &
                         (self.active_field('current_hp')[:, 0] > 0))
        immune |= (status == Status.SLEEP) & (uproar | jnp.any(jnp.asarray(SLEEP_IMMUNE_ABILITIES) == ability))
        immune |= (ability == AbilityEnum.LEAF_GUARD) & sunny
        rising = self.volatile_status.magnet_rise_turns[player] > 0
        grounded = ~(is_floating_fields(types, ability, item) | rising) | self.grounding_override(player)
        immune |= check_terrain & grounded & ((self.terrain.terrain == TerrainEnum.MISTY) |
                              ((self.terrain.terrain == TerrainEnum.ELECTRIC) & (status == Status.SLEEP)))
        meteor = (ability == AbilityEnum.SHIELDS_DOWN) & (self.active_identity('form', player)[0] == FormEnum.MINIOR_METEOR)
        return immune | meteor | (ability == AbilityEnum.COMATOSE)

    def grounding_override(self, player=None):
        """Return grounding imposed independently of type, item and ability."""
        ingrain = self.volatile_status.residuals.ingrain
        rooted = ingrain if player is None else ingrain[player]
        return (rooted != 0) | (self.gravity_duration > 0)

    def is_grounded(self, player):
        """Check grounding from field overrides and airborne effects."""
        ability = self.ability[player]
        item = self.active_field('item', player)
        suppressed = item_is_suppressed(ability, self.magic_room_duration, self.volatile_status.embargo_turns[player])
        airborne = is_floating_fields(self.effective_types(player, include_added=True), ability,
                                      jnp.where(suppressed, ItemEnum.NONE, item))
        rising = self.volatile_status.magnet_rise_turns[player] > 0
        return ~(airborne | rising) | self.grounding_override(player)

    def is_immune_to_move(self, player, move):
        """Check an active player's move immunity in the current field."""
        override = self.grounding_override(player)
        rising = self.volatile_status.magnet_rise_turns[player, 0] > 0
        return move_immune_from_fields(self.effective_types(player, include_added=True), ability_against_move(self, player, move),
                                       self.effective_item(player), move, override) | (
            rising & (move.type[0] == Type.GROUND) & ~override[0])

    @property
    def raw_boosted_stats(self):
        """Apply stage multipliers to active stats without ability modifiers."""
        stats = self.active_stats()
        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + self.boosts.normal_boosts])

    @property
    def boosted_stats(self):
        """Return the stage-adjusted stat view with shared battle factors."""
        return jax.vmap(partial(_modify_stats_row, state=self))(jnp.arange(2), self.raw_boosted_stats)

    @property
    def effective_speeds(self):
        """Calculate only the stage-adjusted Speed needed for battle order."""
        base = self.active_stats()[:, StatEnum.SPEED]
        stages = self.boosts.normal_boosts[:, StatEnum.SPEED]
        speed = jnp.floor(base * STAT_MULTIPLIER_LOOKUP[6 + stages])
        ability = self.ability
        weather = effective_weather(self.weather.weather, ability, self.active_field('current_hp'))
        return modify_speed(speed, self.effective_item()[:, 0], ability[:, 0],
                            self.active_field('status')[:, 0], weather, self.terrain.terrain,
                            self.volatile_status.ability_state.paradox_stat[:, 0], self.side_conditions.tailwind[:, 0])

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
