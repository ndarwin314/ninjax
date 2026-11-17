"""Ability transitions using explicit lower-level state operations."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.ability_rules import UNCHANGEABLE_ABILITIES
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import Status, Type
from ninjax.knowledge import remember_item
from ninjax.records.events import EventKind, emit
from ninjax.side.item_effects import refresh_paradox
from ninjax.side.updates import update_active_fields


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def _worry_seed_change(s, *, old, target):
    """End represented old-ability state before applying the new identity."""
    v = s.volatile_status
    paradox = (old == AbilityEnum.PROTOSYNTHESIS) | (old == AbilityEnum.QUARK_DRIVE)
    v = v.replace(ability_override=v.ability_override.at[target, 0].set(AbilityEnum.INSOMNIA),
        paradox_stat=v.paradox_stat.at[target, 0].set(jnp.where(paradox, -1, v.paradox_stat[target, 0])),
        paradox_booster=v.paradox_booster.at[target, 0].set(jnp.where(paradox, False, v.paradox_booster[target, 0])),
        flash_fire=v.flash_fire.at[target, 0].set(jnp.where(old == AbilityEnum.FLASH_FIRE, False, v.flash_fire[target, 0])),
        libero_used=v.libero_used.at[target, 0].set(jnp.where(old == AbilityEnum.PROTEAN, False, v.libero_used[target, 0])))
    s = s.replace(volatile_status=v)
    slot = s.active_index[target, 0]
    s = s.replace(known_abilities=s.known_abilities.at[target, slot].set(AbilityEnum.INSOMNIA))
    # The old ability may never have been public in hidden-team formats.
    s = emit(s, EventKind.ABILITY, target, detail=AbilityEnum.INSOMNIA, after=AbilityEnum.INSOMNIA)
    mon = s.active[target]
    asleep = mon.status == Status.SLEEP
    s = update_active_fields(s, target, status=jnp.where(asleep, Status.NONE, mon.status),
                             sleep_counter=jnp.where(asleep, 0, mon.sleep_counter))
    return refresh_paradox(s)


def underlying_ability(base_ability, override):
    """Resolve scalar ability identity, ignoring effect suppression.

    A negative override means the active Pokémon's stored ability is used.
    """
    return jnp.where(override >= 0, override, base_ability)


def worry_seed(state, attacker):
    """Replace an eligible target ability with Insomnia and wake the target.

    Preserve the team ability, clear represented old-ability state, refresh
    field-dependent Paradox effects, and record the publicly revealed ability.
    """

    target = 1-attacker
    old = underlying_ability(state.active.ability[target, 0], state.volatile_status.ability_override[target, 0])
    eligible = ((old != AbilityEnum.INSOMNIA) & (old != AbilityEnum.TRUANT) &
        ~jnp.any(jnp.asarray(UNCHANGEABLE_ABILITIES, dtype=jnp.int32) == old))
    shield = eligible & (state.active.item[target, 0] == ItemEnum.ABILITY_SHIELD)
    state = remember_item(state, target, ItemEnum.ABILITY_SHIELD, shield)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=ItemEnum.ABILITY_SHIELD, enabled=shield)
    allowed = eligible & ~shield
    change = partial(_worry_seed_change, old=old, target=target)
    return jax.lax.cond(allowed, change, _identity, state)


def frisk(state, source):
    """Reveal a living opponent's held item; an empty item slot reveals nothing."""
    target = 1-source
    item = state.active.item[target, 0]
    enabled = (state.ability[source, 0] == AbilityEnum.FRISK) & (
        state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & (item != ItemEnum.NONE)
    state = remember_item(state, target, item, enabled)
    previous_source = state.event_source
    state = state.replace(event_source=jnp.asarray(source, jnp.int32))
    state = emit(state, EventKind.ABILITY, source, detail=AbilityEnum.FRISK, enabled=enabled)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=item, enabled=enabled)
    return state.replace(event_source=previous_source)


def prepare_type_change(state, player, move):
    """Apply Gen 9 Protean/Libero once per switch, only on an actual type change."""

    mon = state.active[player]
    used = state.volatile_status.libero_used[player, 0]
    # A monotype matching the move does not consume the activation; a dual
    # type containing it still changes to a monotype and does consume it.
    matching = (mon.type_list[0] == move.type[0]) & (mon.type_list[1] == Type.NONE)
    trigger = (state.ability[player, 0] == AbilityEnum.PROTEAN) & ~used & ~matching & (
        move.type[0] != Type.NONE) & ~mon.is_terastallized[0] & move.implemented[0]
    types = jnp.where(trigger, jnp.array([move.type[0], Type.NONE]), mon.type_list)
    state = update_active_fields(state, player, type_list=types)
    state = emit(state, EventKind.ABILITY, player, detail=state.ability[player, 0], enabled=trigger)
    return state.replace(volatile_status=state.volatile_status.replace(
        libero_used=state.volatile_status.libero_used.at[player, 0].set(used | trigger)))
