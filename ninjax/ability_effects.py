"""Ability transitions using explicit lower-level state operations."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.ability_rules import (ENTRAINMENT_BLOCKED_ABILITIES, ROLE_PLAY_BLOCKED_ABILITIES,
    SWAP_BLOCKED_ABILITIES, TRACE_BLOCKED_ABILITIES, UNCHANGEABLE_ABILITIES)
from ninjax.calculations.logic import quad_or
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import TerrainEnum, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.item_rules import ability_shield_active
from ninjax.knowledge import remember_item
from ninjax.records.events import EventKind, emit
from ninjax.side.field_conditions import set_terrain, set_weather
from ninjax.side.item_effects import cure_status, held_item_active, refresh_paradox
from ninjax.side.queries import contact_reactions_allowed
from ninjax.side.stat_effects import conditional_reduce_boosts
from ninjax.side.updates import conditional_add_boosts, update_active_fields


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def underlying_ability(base_ability, override):
    """Resolve scalar ability identity, ignoring effect suppression.

    A negative override means the active Pokémon's stored ability is used.
    """
    return jnp.where(override >= 0, override, base_ability)


def frisk(state, source):
    """Reveal a living opponent's held item; an empty item slot reveals nothing."""
    target = 1-source
    item = state.active.item[target, 0]
    enabled = (state.ability[source, 0] == AbilityEnum.FRISK) & (
        state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & (item != ItemEnum.NONE)
    state = remember_item(state, target, item, enabled)
    previous_source = state.event_context.source
    state = state.replace(event_context=state.event_context.replace(source=jnp.asarray(source, jnp.int32)))
    state = emit(state, EventKind.ABILITY, source, detail=AbilityEnum.FRISK, enabled=enabled)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=item, enabled=enabled)
    return state.replace(event_context=state.event_context.replace(source=previous_source))


def prepare_type_change(state, player, move):
    """Apply Gen 9 Protean/Libero once per switch, only on an actual type change."""

    mon = state.active[player]
    used = state.volatile_status.ability_state.libero_used[player, 0]
    # A monotype matching the move does not consume the activation; a dual
    # type containing it still changes to a monotype and does consume it.
    extra = state.volatile_status.properties.added_type
    matching = (mon.type_list[0] == move.type[0]) & (mon.type_list[1] == Type.NONE) & (extra[player, 0] == Type.NONE)
    trigger = (state.ability[player, 0] == AbilityEnum.PROTEAN) & ~used & ~matching & (
        move.type[0] != Type.NONE) & ~mon.is_terastallized[0] & move.implemented[0]
    types = jnp.where(trigger, jnp.array([move.type[0], Type.NONE]), mon.type_list)
    needs_type = (move.identifier[0] == 892) | (move.identifier[0] == 682)
    required = jnp.where(move.identifier[0] == 892, Type.ELECTRIC, Type.FIRE)
    trigger &= ~needs_type | jnp.any(state.effective_types(player, include_added=True) == required)
    state = update_active_fields(state, player, type_list=jnp.where(trigger, types, mon.type_list))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(properties=v.properties.replace(
        added_type=extra.at[player, 0].set(jnp.where(trigger, Type.NONE, extra[player, 0])))))
    state = emit(state, EventKind.ABILITY, player, detail=state.ability[player, 0], enabled=trigger)
    return state.replace(volatile_status=state.volatile_status.replace(ability_state=state.volatile_status.ability_state.replace(libero_used=state.volatile_status.ability_state.libero_used.at[player, 0].set(used | trigger))))


@jax.jit
def start_ability(state, side_idx, entering=False):
    """Start represented switch-in and newly acquired ability effects."""
    ability = state.ability[side_idx, 0]
    defender_ability = state.ability[1-side_idx]
    # activate weather abilities if target is alive
    # we abuse the choice to WeatherEnum.NONE=0 to simplify the logic

    previous_weather = state.weather.weather
    previous_terrain = state.terrain.terrain
    new_weather = (jnp.logical_or(ability==AbilityEnum.DROUGHT, ability==AbilityEnum.ORICHALCUM_PULSE) * WeatherEnum.SUN +
                   (ability==AbilityEnum.DRIZZLE) * WeatherEnum.RAIN +
                   (ability==AbilityEnum.SAND_STREAM) * WeatherEnum.SANDSTORM +
                   (ability==AbilityEnum.SNOW_WARNING) * WeatherEnum.SNOW)
    state = set_weather(state, new_weather, 5)

    # terrain, same deal as weather
    new_terrain = (jnp.logical_or(ability==AbilityEnum.ELECTRIC_SURGE, ability==AbilityEnum.HADRON_ENGINE) * TerrainEnum.ELECTRIC +
                   (ability==AbilityEnum.PSYCHIC_SURGE) * TerrainEnum.PSYCHIC +
                   (ability==AbilityEnum.GRASSY_SURGE) * TerrainEnum.GRASSY +
                   (ability==AbilityEnum.MISTY_SURGE) * TerrainEnum.MISTY)
    state = set_terrain(state, new_terrain, 5)

    # intimidate
    is_intimidate = ability == AbilityEnum.INTIMIDATE
    is_intimidate_immune = quad_or(
        defender_ability==AbilityEnum.OBLIVIOUS,
        defender_ability==AbilityEnum.OWN_TEMPO,
        defender_ability==AbilityEnum.INNER_FOCUS,
        defender_ability==AbilityEnum.SCRAPPY,
    )
    intimidate_activated = jnp.logical_and(is_intimidate, 1-is_intimidate_immune)
    old_attack = state.boosts.normal_boosts[1-side_idx, StatEnum.ATTACK]
    state = conditional_reduce_boosts(state, 1-side_idx, intimidate_activated, StatEnum.ATTACK, 1, True)
    rattled = ((ability == AbilityEnum.INTIMIDATE) & (defender_ability[0] == AbilityEnum.RATTLED) &
               (state.active.current_hp[1-side_idx, 0] > 0) &
               (state.boosts.normal_boosts[1-side_idx, StatEnum.ATTACK] != old_attack))
    state = conditional_add_boosts(state, 1-side_idx, rattled, StatEnum.SPEED, 1)
    # Download compares stage-adjusted defenses without item/ability modifiers.
    defenses = state.raw_boosted_stats[1-side_idx]
    download_stat = jnp.where(defenses[StatEnum.DEFENSE] < defenses[StatEnum.SPECIAL_DEFENSE], StatEnum.ATTACK, StatEnum.SPECIAL_ATTACK)
    state = conditional_add_boosts(state, side_idx, (ability == AbilityEnum.DOWNLOAD) & (state.active.current_hp[1-side_idx, 0] > 0), download_stat, 1)
    state = conditional_add_boosts(state, side_idx,
        (entering & (ability == AbilityEnum.WIND_RIDER)) & (state.side_conditions.tailwind[side_idx, 0] > 0), StatEnum.ATTACK, 1)
    old_source = state.event_context.source
    state = state.replace(event_context=state.event_context.replace(source=jnp.asarray(side_idx, jnp.int32)))
    state = emit(state, EventKind.ABILITY, side_idx, detail=ability,
                 enabled=(state.weather.weather != previous_weather) | (state.terrain.terrain != previous_terrain))
    state = state.replace(event_context=state.event_context.replace(source=old_source))
    state = _royal_ability_start(state, side_idx, ability)
    return frisk(state, side_idx)


def _contains(abilities, identity):
    """Test a current identity against one of the static mechanic lists."""
    return jnp.any(jnp.asarray(abilities, jnp.int32) == identity)


def _shield_blocks(state, player, eligible):
    """Recognize and reveal an active Ability Shield on a valid mutation."""
    shield = eligible & ability_shield_active(state.active_field('item', player)[0],
        state.magic_room_duration, state.volatile_status.embargo_turns[player, 0])
    state = remember_item(state, player, ItemEnum.ABILITY_SHIELD, shield)
    return emit(state, EventKind.ITEM_REVEALED, player, detail=ItemEnum.ABILITY_SHIELD, enabled=shield), shield


def _clear_ability_activation(state, player):
    """Clear state belonging to an ended or replaced ability."""
    v = state.volatile_status
    a = v.ability_state
    return state.replace(volatile_status=v.replace(ability_state=a.replace(
        paradox_stat=a.paradox_stat.at[player, 0].set(-1),
        paradox_booster=a.paradox_booster.at[player, 0].set(False),
        flash_fire=a.flash_fire.at[player, 0].set(False),
        libero_used=a.libero_used.at[player, 0].set(False))))


def install_ability(state, player, identity):
    """Install an already validated identity without running start hooks."""
    state = _clear_ability_activation(state, player)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(
        ability_override=v.ability_state.ability_override.at[player, 0].set(identity))))
    slot = state.active_index[player, 0]
    state = state.replace(known_abilities=state.known_abilities.at[player, slot].set(identity))
    return emit(state, EventKind.ABILITY, player, detail=identity, after=identity)


def _start_new_ability(state, player):
    """Run start hooks and cure existing sleep for a new sleep immunity."""
    identity = state.ability[player, 0]
    wake = ((identity == AbilityEnum.INSOMNIA) | (identity == AbilityEnum.VITAL_SPIRIT)) & (
        state.active_field('status', player)[0] == Status.SLEEP)
    state = jax.lax.cond(wake, partial(cure_status, player=player), _identity, state)
    return refresh_paradox(start_ability(state, player))


def _install_and_start(state, *, player, identity):
    """Replace an ability and immediately start its effective replacement."""
    return _start_new_ability(install_ability(state, player, identity), player)


def _swap_abilities(state, *, player, own, other):
    """Commit both identities before either participant runs a start hook."""
    target = 1-player
    state = install_ability(install_ability(state, player, other), target, own)
    return _start_new_ability(_start_new_ability(state, target), player)


def swap_abilities(state, player):
    """Apply Skill Swap or Wandering Spirit as one atomic ability exchange."""
    own, other = state.ability_identity[player, 0], state.ability_identity[1-player, 0]
    eligible = (~_contains(SWAP_BLOCKED_ABILITIES, own) & ~_contains(SWAP_BLOCKED_ABILITIES, other)
                & jnp.all(state.active_field('current_hp') > 0))
    state, shield = _shield_blocks(state, 1-player, eligible)
    state, own_shield = _shield_blocks(state, player, eligible & ~shield)
    success = eligible & ~shield & ~own_shield
    return jax.lax.cond(success, partial(_swap_abilities, player=player, own=own, other=other),
                        _identity, state), success


def override_ability(state, target, identity, eligible):
    """Replace one eligible mutable ability unless Ability Shield prevents it."""
    old = state.ability_identity[target, 0]
    allowed = eligible & (old != identity) & ~_contains(UNCHANGEABLE_ABILITIES, old) & (
        state.active_field('current_hp', target)[0] > 0)
    state, shield = _shield_blocks(state, target, allowed)
    success = allowed & ~shield
    return jax.lax.cond(success, partial(_install_and_start, player=target, identity=identity),
                        _identity, state), success


@jax.jit
def suppress_ability(state, target, enabled=True):
    """Apply Gastro Acid without erasing the ability that will be restored."""
    a = state.volatile_status.ability_state
    identity = state.ability_identity[target, 0]
    allowed = enabled & ~a.ability_suppressed[target, 0] & ~_contains(UNCHANGEABLE_ABILITIES, identity) & (
        state.active_field('current_hp', target)[0] > 0)
    state, shield = _shield_blocks(state, target, allowed)
    success = allowed & ~shield
    state = jax.lax.cond(success, partial(_clear_ability_activation, player=target), _identity, state)
    v = state.volatile_status
    return state.replace(volatile_status=v.replace(ability_state=v.ability_state.replace(
        ability_suppressed=a.ability_suppressed.at[target, 0].set(a.ability_suppressed[target, 0] | success)))), success


@jax.jit
def ability_move(state, player, ident):
    """Resolve ability identity moves through common eligibility and lifecycle."""
    target = 1-player
    own, other = state.ability_identity[player, 0], state.ability_identity[target, 0]
    role_play = ident == 272
    entrainment = ident == 494
    receiver = jnp.where(role_play, player, target)
    replacement = jnp.select((role_play, entrainment, ident == 493),
                             (other, own, AbilityEnum.SIMPLE), default=AbilityEnum.INSOMNIA)
    eligible = jnp.where(role_play, ~_contains(ROLE_PLAY_BLOCKED_ABILITIES, other),
                         (other != AbilityEnum.TRUANT) &
                         (~entrainment | ~_contains(ENTRAINMENT_BLOCKED_ABILITIES, own)))
    branch = jnp.select((ident == 285, ident == 380), (1, 2), default=0)
    state, success = jax.lax.switch(branch, (
        partial(_ability_override, player=receiver, ability=replacement, allowed=eligible),
        partial(_ability_swap, player=player), partial(_ability_suppress, target=target)), state)
    wake = (ident == 388) & success & (state.active_field('status', target)[0] == Status.SLEEP)
    state = jax.lax.cond(wake, partial(cure_status, player=target), _identity, state)
    return state, success


def _ability_override(state, *, player, ability, allowed):
    """Adapt an ability replacement to the move dispatcher."""
    return override_ability(state, player, ability, allowed)


def _ability_swap(state, *, player):
    """Adapt an ability swap to the move dispatcher."""
    return swap_abilities(state, player)


def _ability_suppress(state, *, target):
    """Adapt Gastro Acid to the move dispatcher."""
    return suppress_ability(state, target)


def worry_seed(state, attacker):
    """Apply Worry Seed via the shared override lifecycle."""
    return ability_move(state, attacker, jnp.int32(388))[0]


@jax.jit
def refresh_trace_player(player, state):
    """Retry Trace when a living foe has a copyable ability."""
    target = 1-player
    identity = state.ability_identity[target, 0]
    eligible = (state.ability[player, 0] == AbilityEnum.TRACE) & (identity != AbilityEnum.NONE) & (
        state.active_field('current_hp', target)[0] > 0) & ~_contains(TRACE_BLOCKED_ABILITIES, identity)
    return jax.lax.cond(eligible, partial(_trace, player=player, identity=identity), _identity, state)


def _trace(state, *, player, identity):
    """Copy a valid foe ability, accounting for the wearer's Ability Shield."""
    return override_ability(state, player, identity, True)[0]


@jax.jit
def resume_ability(state, player, original):
    """Restart a surviving unchanged holder's ability after suppression ends."""
    ready = (original == AbilityEnum.NONE) & (state.ability[player, 0] != AbilityEnum.NONE) & (
        state.active_field('current_hp', player)[0] > 0)
    return jax.lax.cond(ready, partial(_start_new_ability, player=player), _identity, state)


@jax.jit
def ability_contact_reaction(state, attacker, defender_ability, contact):
    """Apply ability-changing contact reactions on a direct damaging hit."""
    item = jnp.where(held_item_active(state, attacker), state.active_field('item', attacker)[0], ItemEnum.NONE)
    allowed = contact & contact_reactions_allowed(state.ability[attacker, 0], item)
    aroma = (defender_ability == AbilityEnum.LINGERING_AROMA) | (defender_ability == AbilityEnum.MUMMY)
    branch = jnp.select((allowed & aroma, allowed & (defender_ability == AbilityEnum.WANDERING_SPIRIT)),
                        (1, 2), default=0)
    return jax.lax.switch(branch, (_identity,
        partial(_contact_override, player=attacker, identity=defender_ability),
        partial(_contact_swap, player=attacker)), state)


def _contact_override(state, *, player, identity):
    """Give a contact attacker the defender's ability."""
    return override_ability(state, player, identity, True)[0]


def _contact_swap(state, *, player):
    """Apply Wandering Spirit's atomic exchange."""
    return swap_abilities(state, player)[0]


def _royal_ability_start(state, player, identity):
    """Intrepid Sword and Dauntless Shield activate once per Pokémon each battle."""
    sword = identity == AbilityEnum.INTREPID_SWORD
    shield = identity == AbilityEnum.DAUNTLESS_SHIELD
    ability = jnp.where(sword, 0, 1)
    stat = jnp.where(sword, StatEnum.ATTACK, StatEnum.DEFENSE)
    slot = state.active_index[player, 0]
    used = state.ability_once_used[player, slot, ability]
    activate = (sword | shield) & ~used & (state.active_field('current_hp', player)[0] > 0)
    state = state.replace(ability_once_used=state.ability_once_used.at[player, slot, ability].set(used | activate))
    state = conditional_add_boosts(state, player, activate, stat, 1)
    return emit(state, EventKind.ABILITY, player, detail=identity, enabled=activate)
