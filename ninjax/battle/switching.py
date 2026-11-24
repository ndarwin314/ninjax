"""Switch requests, departure, hazards and entry effects for immutable battles.

Preparation preserves the explicit selection/entry key sequence. A prepared
forced switch is committed after move routing, before the next actor proceeds."""
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import frisk
from ninjax.calculations.logic import quad_or
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import TerrainEnum, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import StatEnum, Type
from ninjax.knowledge import remember_item
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.field_conditions import set_terrain, set_weather
from ninjax.side.hp_effects import take_damage_percent
from ninjax.side.item_effects import cure_status, update_items
from ninjax.side.queries import is_floating
from ninjax.side.stat_effects import conditional_reduce_boosts
from ninjax.side.state import BattleState
from ninjax.side.status_effects import set_status
from ninjax.side.updates import (
    add_boosts,
    clear_boosts,
    clear_volatile_status,
    conditional_add_boosts,
    update_active_fields,
)


__all__ = (
    'ForcedSwitchRequest',
    'prepare_forced_switch',
    'execute_forced_switch',
    'force_switch_target',
    'swap_out',
    'swap_is_alive',
)


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


class ForcedSwitchRequest(NamedTuple):
    """Local switch operands: scalar enabled/player/slot arrays and a PRNG key.

    This PyTree is consumed within one hit attempt, never stored in battle state.
    The entry key is a placeholder when ``enabled`` is false.
    """
    enabled: jax.Array
    player: jax.Array
    slot: jax.Array
    entry_key: chex.PRNGKey


def _no_forced_switch(key, state, attacker):
    """Return an inactive request without advancing the battle key."""
    return key, state, ForcedSwitchRequest(jnp.bool_(False), jnp.asarray(1-attacker, jnp.int32),
                                          jnp.int32(0), key)


def forced_switch_move(move):
    """Recognize moves that request the shared status drag continuation."""
    identifier = move.identifier[0]
    return (identifier == 18) | (identifier == 46)


def force_switch_target(key, state, attacker, *, wind=True, enabled=True):
    """Prepare and commit one drag request for either move entry point."""
    key, state, request = jax.lax.cond(
        enabled, partial(prepare_forced_switch, wind=wind), _no_forced_switch,
        key, state, attacker)
    return key, execute_forced_switch(state, request)


def _wind_rider_boost(s, *, target):
    """Raise Attack when Wind Rider blocks a forced switch."""
    return add_boosts(s, target, StatEnum.ATTACK, 1)


def _fail_forced_switch(k, s, *, wind_rider, anchored, soundproof, allowed, target, ability, attacker):
    """Record failure and reveal represented ability-based prevention."""
    s = jax.lax.cond(wind_rider, partial(_wind_rider_boost, target=target), _identity, s)
    reveal = wind_rider | soundproof | (anchored & jnp.any(allowed))
    slot = s.active_index[target, 0]
    s = s.replace(known_abilities=s.known_abilities.at[target, slot].set(jnp.where(reveal, ability, s.known_abilities[target, slot])))
    s = emit(s, EventKind.ABILITY, target, detail=ability, enabled=reveal)
    # Drag immunity returns NOT_FAIL when a replacement exists; unlike a
    # missing bench or Wind Rider, it does not boost Stomping Tantrum.
    drag_prevented = anchored & jnp.any(allowed) & ~wind_rider
    s = record_hit(s, 0, target,
        jnp.where(soundproof, HitOutcome.IMMUNE,
                  jnp.where(drag_prevented, HitOutcome.HIT, HitOutcome.FAILED)),
        jnp.where(drag_prevented, HitRecipient.POKEMON, HitRecipient.NONE))
    return _no_forced_switch(k, s, attacker)


def _sample_forced_switch(k, s, *, allowed, target):
    """Uniformly select a legal replacement and retain its independent entry key."""
    k, select_key, entry_key = random.split(k, 3)
    slot = random.categorical(select_key, jnp.where(allowed, 0., -jnp.inf)).astype(jnp.int32)
    request = ForcedSwitchRequest(jnp.bool_(True), jnp.asarray(target, jnp.int32), slot, entry_key)
    return k, s, request


def prepare_forced_switch(key, state, attacker, *, wind=True):
    """Resolve drag immunity and sample a replacement without applying entry effects.

    Only a successful request splits the key, preserving the selection/entry
    key sequence and blocked-attempt outcomes of immediate forced switching.
    ``wind`` distinguishes Whirlwind's Wind Rider check from Roar's Soundproof
    check; ordinary hit routing also checks the move's sound flag.
    """
    target = 1-attacker
    ability = state.ability[target, 0]
    wind_rider = (ability == AbilityEnum.WIND_RIDER) & wind
    anchored = (ability == AbilityEnum.SUCTION_CUPS) | (ability == AbilityEnum.GUARD_DOG)
    soundproof = ~jnp.asarray(wind) & (ability == AbilityEnum.SOUNDPROOF)
    allowed = state.roster[target] & (state.team.current_hp[target, :, 0] > 0) & (jnp.arange(6) != state.active_index[target, 0])
    blocked = wind_rider | anchored | soundproof | ~jnp.any(allowed)
    return jax.lax.cond(
        blocked,
        partial(
            _fail_forced_switch,
            wind_rider=wind_rider,
            anchored=anchored,
            soundproof=soundproof,
            allowed=allowed,
            target=target,
            ability=ability,
            attacker=attacker,
        ),
        partial(_sample_forced_switch, allowed=allowed, target=target),
        key,
        state,
    )


def _commit_forced_switch(s, *, request):
    """Apply the selected replacement with its saved entry key."""
    return swap_out(request.entry_key, s, request.player, request.slot)


def execute_forced_switch(state, request: ForcedSwitchRequest):
    """Commit a prepared switch, including departure, hazards and entry hooks."""
    return jax.lax.cond(
        request.enabled,
        partial(_commit_forced_switch, request=request),
        _identity,
        state,
    )


def _unchanged_switch_state(state, player):
    """Preserve state when a switch entrant is already fainted."""
    return state


def _cure_switch_status(s, *, side_idx):
    """Cure status on departure for Natural Cure."""

    return cure_status(s, side_idx)


def reset_switch_effects(state, side_idx, transfer=0):
    """Clear departure effects, preserving Baton Pass (1) or Shed Tail (2).

    Baton Pass copies represented passable conditions and stat stages. Shed
    Tail copies only Substitute. Apply this before hazards and entry abilities.
    """
    old = state.volatile_status
    boosts = state.boosts
    perish = state.perish_turns[side_idx]
    baton = transfer == 1
    state = jax.lax.cond(baton, _identity, partial(clear_boosts, side_idx=side_idx), state)
    state = clear_volatile_status(state, side_idx)
    substitute = (transfer == 1) | (transfer == 2)
    volatile = state.volatile_status
    # Disable, Encore, Destiny Bond and ability-specific effects are noCopy;
    # Taunt, confusion, Substitute and Perish Song are passable.
    volatile = volatile.replace(
        substitute_hp=volatile.substitute_hp.at[side_idx].set(jnp.where(substitute, old.substitute_hp[side_idx], 0)),
        confusion_turns=volatile.confusion_turns.at[side_idx].set(jnp.where(baton, old.confusion_turns[side_idx], 0)),
        taunt_turns=volatile.taunt_turns.at[side_idx].set(jnp.where(baton, old.taunt_turns[side_idx], 0)))
    current = state.boosts
    current = current.replace(
        normal_boosts=current.normal_boosts.at[side_idx].set(jnp.where(baton, boosts.normal_boosts[side_idx], current.normal_boosts[side_idx])),
        acc_boosts=current.acc_boosts.at[side_idx].set(jnp.where(baton, boosts.acc_boosts[side_idx], current.acc_boosts[side_idx])))
    return state.replace(volatile_status=volatile, boosts=current,
                         perish_turns=state.perish_turns.at[side_idx].set(jnp.where(baton, perish, -1)))


def swap_out(
    key,
    state: BattleState,
    side_idx,
    new_active: int | jax.Array,
    transfer=0,
) -> BattleState:
    """Switch one side to a team slot and apply represented entry effects.

    Stages, volatile state, toxic count, and Perish Song count are reset before
    hazards and switch-in abilities run. A Pokémon fainting to entry hazards
    does not run ``swap_is_alive``. Switch and hazard coverage is incomplete.

    The supplied key is used for status effects; this helper currently returns
    only state, so its caller must manage key consumption separately.
    """
    # swaps the active pokemon and does appropriate things like
    # 1. clearing volatile statuses
    # 2. resting boosts
    # 3. probably stuff im forgetting
    # 4. ahhh palafin, ahhh regenerator

    state = state.replace(event_source=jnp.asarray(side_idx, jnp.int32), event_move=jnp.int32(0), event_cause=jnp.int32(Cause.SWITCH))
    state = emit(state, EventKind.SWITCH, side_idx, before=state.active_index[side_idx, 0], after=new_active)
    outgoing = state.active[side_idx]
    state = jax.lax.cond(
        (state.ability[side_idx, 0] == AbilityEnum.NATURAL_CURE) & (outgoing.current_hp[0] > 0),
        partial(_cure_switch_status, side_idx=side_idx),
        _identity,
        state,
    )
    # A temporary public reveal no longer identifies the restored base ability.
    slot = state.active_index[side_idx, 0]
    known = state.known_abilities
    state = state.replace(known_abilities=known.at[side_idx, slot].set(jnp.where(
        state.volatile_status.ability_override[side_idx, 0] >= 0, -1, known[side_idx, slot])))
    state = update_active_fields(state, side_idx, type_list=outgoing.base_types)
    state = take_damage_percent(state, side_idx, jnp.where(
        (state.ability[side_idx] == AbilityEnum.REGENERATOR) & (outgoing.current_hp > 0), -1/3, 0))
    # clear boosts and volatile status from active pokemon, maybe add baton pass check here?
    state = reset_switch_effects(state, side_idx, transfer)
    state = state.replace(volatile_status=state.volatile_status.replace(
        entry_turn=state.volatile_status.entry_turn.at[side_idx, 0].set(state.turn_number)))

    # update active pokemon index
    # this update could be saved since we need an update later but idk if it will matter
    active = state.active_index.at[side_idx].set(new_active)
    toxic_counter = state.toxic_counter.at[side_idx].set(0)
    state = state.replace(active_index=active, toxic_counter=toxic_counter)

    # hazards
    active = state.effective_active[side_idx]

    boots = active.item == ItemEnum.HEAVY_DUTY_BOOTS
    is_not_flying = (1 - is_floating(active)) * ~boots
    is_not_hazard_immune = (1 - (active.ability == AbilityEnum.MAGIC_GUARD)) * ~boots
    is_poison = active.is_type(Type.POISON)
    is_poison_immune = jnp.any(active.type_list == Type.STEEL)
    # stealth rocks
    state = take_damage_percent(
        state,
        side_idx,
        state.stealth_rocks[side_idx] * calculate_effectiveness_multiplier(Type.ROCK, active.type_list) / 8 * is_not_hazard_immune
    )

    # spikes
    state = take_damage_percent(
        state, side_idx,
        (state.spikes[side_idx] != 0) / (10 - 2 * state.spikes[side_idx]) * is_not_flying * is_not_hazard_immune
    )

    # sticky webs
    # so apparently if a pokemon sets webs and a pokemon with mirror armor is effected by them
    # then it applies the stat reduction to that mon, but this only happens if they are on field
    state = conditional_reduce_boosts(state, side_idx,
        (is_not_flying * state.sticky_webs[side_idx]).reshape(()) != 0, StatEnum.SPEED, 1, False)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state.toxic_spikes[side_idx] * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(toxic_spikes=toxic_spikes)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state.toxic_spikes[side_idx]) * (state.toxic_spikes[side_idx] != 0)
    key, state = set_status(key, state, side_idx, status_ * (1-is_poison_immune) * is_not_flying)

    #check if switch in is still alive
    active_hp = state.active.current_hp[side_idx, 0]
    is_alive = active_hp != 0

    state = jax.lax.cond(is_alive, swap_is_alive, _unchanged_switch_state, state, side_idx)


    state = update_active_fields(state, side_idx, is_alive=is_alive)
    return state


def swap_is_alive(
    state: BattleState,
    side_idx) -> BattleState:
    """Apply represented weather, terrain, Intimidate and Frisk entry abilities."""
    active = state.effective_active
    attacker = active[side_idx]
    defender_ability = active[1-side_idx].ability

    # activate weather abilities if target is alive
    # we abuse the choice to WeatherEnum.NONE=0 to simplify the logic

    ability = attacker.ability[0]
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
    is_intimidate = attacker.ability == AbilityEnum.INTIMIDATE
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
        (ability == AbilityEnum.WIND_RIDER) & (state.tailwind[side_idx, 0] > 0), StatEnum.ATTACK, 1)
    old_source = state.event_source
    state = state.replace(event_source=jnp.asarray(side_idx, jnp.int32))
    state = emit(state, EventKind.ABILITY, side_idx, detail=ability,
                 enabled=(state.weather.weather != previous_weather) | (state.terrain.terrain != previous_terrain))
    state = state.replace(event_source=old_source)


    balloon = attacker.item[0] == ItemEnum.AIR_BALLOON

    state = remember_item(state, side_idx, ItemEnum.AIR_BALLOON, balloon)
    state = emit(state, EventKind.ITEM_REVEALED, side_idx, detail=ItemEnum.AIR_BALLOON, enabled=balloon)
    return update_items(frisk(state, side_idx))
