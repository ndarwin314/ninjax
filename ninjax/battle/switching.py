"""Switch requests, departure, hazards and entry effects for immutable battles.

Preparation preserves the explicit selection/entry key sequence. A prepared
forced switch is committed after move routing, before the next actor proceeds."""
from enum import IntFlag
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.ability_effects import refresh_trace_player, resume_ability, start_ability, suppress_ability
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import StatEnum, Type
from ninjax.knowledge import remember_item
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.form_effects import held_crowned_form, held_origin_form, palafin_departure, reactive_form_update, tera_shift
from ninjax.side.hp_effects import take_damage_percent
from ninjax.side.item_effects import cure_status, update_items
from ninjax.side.property_effects import pass_power_trick, restore_move_bank, transform
from ninjax.side.queries import ability_against_move
from ninjax.side.trapping import clear_source_linked
from ninjax.side.stat_effects import conditional_reduce_boosts
from ninjax.side.state import BattleState
from ninjax.side.status_effects import set_status
from ninjax.side.updates import (
    add_boosts,
    clear_boosts,
    clear_volatile_status,
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


class EntryRecovery(IntFlag):
    """Pending recovery effects and the order to use when both are present."""
    HEALING_WISH = 1
    LUNAR_DANCE = 2
    LUNAR_DANCE_FIRST = 4


def queue_entry_recovery(state, player, lunar_dance, enabled):
    """Queue recovery at the active position in the order it was created."""
    before = state.entry_recovery[player]
    new_wish = ~lunar_dance & ((before & EntryRecovery.HEALING_WISH) == 0)
    dance_first = new_wish & ((before & EntryRecovery.LUNAR_DANCE) != 0)
    effect = jnp.where(lunar_dance, EntryRecovery.LUNAR_DANCE, EntryRecovery.HEALING_WISH)
    after = before | effect | jnp.where(dance_first, EntryRecovery.LUNAR_DANCE_FIRST, 0)
    return state.replace(entry_recovery=state.entry_recovery.at[player].set(jnp.where(enabled, after, before)))


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


def reflected_switching_source(state, attacker, move):
    """Return the resolved user of a reflectable switching move."""
    target = 1 - attacker
    reflects = (ability_against_move(state, target, move)[0] == AbilityEnum.MAGIC_BOUNCE) & (
        state.volatile_status.continuation.semi_invulnerability[target, 0] == 0)
    return jnp.where(reflects, target, attacker)


def force_switch_target(key, state, attacker, *, wind=True, enabled=True, damaging=False):
    """Prepare and commit one drag request for either move entry point."""
    key, state, request = jax.lax.cond(
        enabled, partial(prepare_forced_switch, wind=wind, damaging=damaging), _no_forced_switch,
        key, state, attacker)
    return key, execute_forced_switch(state, request)


def _wind_rider_boost(s, *, target):
    """Raise Attack when Wind Rider blocks a forced switch."""
    return add_boosts(s, target, StatEnum.ATTACK, 1)


def _fail_forced_switch(k, s, *, wind_rider, anchored, immune, allowed, target, ability, attacker, damaging):
    """Record failure and reveal represented ability-based prevention."""
    s = jax.lax.cond(wind_rider, partial(_wind_rider_boost, target=target), _identity, s)
    anchoring_ability = (ability == AbilityEnum.SUCTION_CUPS) | (ability == AbilityEnum.GUARD_DOG)
    reveal = wind_rider | immune | (anchoring_ability & anchored & jnp.any(allowed))
    slot = s.active_index[target, 0]
    s = s.replace(known_abilities=s.known_abilities.at[target, slot].set(jnp.where(reveal, ability, s.known_abilities[target, slot])))
    s = emit(s, EventKind.ABILITY, target, detail=ability, enabled=reveal)
    # Drag immunity returns NOT_FAIL when a replacement exists; unlike a
    # missing bench or Wind Rider, it does not boost Stomping Tantrum.
    drag_prevented = anchored & jnp.any(allowed) & ~wind_rider
    recorded = record_hit(s, 0, target,
        jnp.where(immune, HitOutcome.IMMUNE,
                  jnp.where(drag_prevented, HitOutcome.HIT, HitOutcome.FAILED)),
        jnp.where(drag_prevented, HitRecipient.POKEMON, HitRecipient.NONE))
    s = s.replace(hit_results=jax.tree.map(partial(jnp.where, damaging), s.hit_results, recorded.hit_results))
    return _no_forced_switch(k, s, attacker)


def _sample_forced_switch(k, s, *, allowed, target, damaging):
    """Uniformly select a legal replacement and retain its independent entry key."""
    k, select_key, entry_key = random.split(k, 3)
    slot = random.categorical(select_key, jnp.where(allowed, 0., -jnp.inf)).astype(jnp.int32)
    request = ForcedSwitchRequest(jnp.bool_(True), jnp.asarray(target, jnp.int32), slot, entry_key)
    status_hit = record_hit(s, 0, target, HitOutcome.HIT, HitRecipient.POKEMON)
    s = s.replace(hit_results=jax.tree.map(partial(jnp.where, damaging), s.hit_results, status_hit.hit_results))
    return k, s, request


def prepare_forced_switch(key, state, attacker, *, wind=True, damaging=False):
    """Resolve drag immunity and sample a replacement without applying entry effects.

    Only a successful request splits the key, preserving the selection/entry
    key sequence and blocked-attempt outcomes of immediate forced switching.
    ``wind`` distinguishes Whirlwind's Wind Rider check from Roar's Soundproof
    check; ordinary hit routing also checks the move's sound flag.
    """
    target = 1-attacker
    ability = state.ability[target, 0]
    wind_rider = (ability == AbilityEnum.WIND_RIDER) & wind
    rooted = state.volatile_status.residuals.ingrain[target, 0] != 0
    anchored = rooted | (ability == AbilityEnum.SUCTION_CUPS) | (ability == AbilityEnum.GUARD_DOG)
    immune = ~jnp.asarray(damaging) & ((ability == AbilityEnum.GOOD_AS_GOLD) |
                                      (~jnp.asarray(wind) & (ability == AbilityEnum.SOUNDPROOF)))
    allowed = state.roster[target] & (state.team.current_hp[target, :, 0] > 0) & (jnp.arange(6) != state.active_index[target, 0])
    blocked = wind_rider | anchored | immune | ~jnp.any(allowed)
    return jax.lax.cond(
        blocked,
        partial(
            _fail_forced_switch,
            wind_rider=wind_rider,
            anchored=anchored,
            immune=immune,
            allowed=allowed,
            target=target,
            ability=ability,
            attacker=attacker,
            damaging=damaging,
        ),
        partial(_sample_forced_switch, allowed=allowed, target=target, damaging=damaging),
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
    residuals = volatile.residuals
    previous = old.residuals
    # Only these indefinite residuals follow Baton Pass's recipient.
    residuals = residuals.replace(
        aqua_ring=residuals.aqua_ring.at[side_idx].set(jnp.where(baton, previous.aqua_ring[side_idx], 0)),
        ingrain=residuals.ingrain.at[side_idx].set(jnp.where(baton, previous.ingrain[side_idx], 0)),
        leech_seed=residuals.leech_seed.at[side_idx].set(jnp.where(baton, previous.leech_seed[side_idx], 0)),
        curse=residuals.curse.at[side_idx].set(jnp.where(baton, previous.curse[side_idx], 0)))
    volatile = volatile.replace(
        residuals=residuals,
        # Focus Energy and Magnet Rise pass with Baton Pass. Lock-On's source
        # and its target must both stay in battle.
        focus_energy=volatile.focus_energy.at[side_idx].set(baton & old.focus_energy[side_idx]),
        magnet_rise_turns=volatile.magnet_rise_turns.at[side_idx].set(
            jnp.where(baton, old.magnet_rise_turns[side_idx], 0)),
        lock_on_turns=volatile.lock_on_turns.at[1-side_idx].set(0),
        lock_on_target_slot=volatile.lock_on_target_slot.at[1-side_idx].set(-1),
        substitute_hp=volatile.substitute_hp.at[side_idx].set(jnp.where(substitute, old.substitute_hp[side_idx], 0)),
        confusion_turns=volatile.confusion_turns.at[side_idx].set(jnp.where(baton, old.confusion_turns[side_idx], 0)),
        restrictions=volatile.restrictions.replace(taunt_turns=volatile.restrictions.taunt_turns.at[side_idx].set(jnp.where(baton, old.restrictions.taunt_turns[side_idx], 0))))
    current = state.boosts
    current = current.replace(
        normal_boosts=current.normal_boosts.at[side_idx].set(jnp.where(baton, boosts.normal_boosts[side_idx], current.normal_boosts[side_idx])),
        acc_boosts=current.acc_boosts.at[side_idx].set(jnp.where(baton, boosts.acc_boosts[side_idx], current.acc_boosts[side_idx])))
    slot = state.active_index[side_idx, 0]
    history = state.damage_history
    history = history.replace(
        hurt_this_turn=history.hurt_this_turn.at[side_idx, slot].set(False),
        stats_lowered_this_turn=history.stats_lowered_this_turn.at[side_idx, slot].set(False),
        counter_prepared=history.counter_prepared.at[side_idx].set(-1),
        counter_damage=history.counter_damage.at[side_idx, slot].set(0),
        counter_category=history.counter_category.at[side_idx, slot].set(-1),
        counter_source_slot=history.counter_source_slot.at[side_idx, slot].set(-1))
    return state.replace(volatile_status=volatile, boosts=current, damage_history=history,
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

    state = state.replace(
        event_context=state.event_context.replace(
            source=jnp.asarray(side_idx, jnp.int32),
            move_id=jnp.int32(0),
            cause=jnp.int32(Cause.SWITCH)))
    state = emit(state, EventKind.SWITCH, side_idx, before=state.active_index[side_idx, 0], after=new_active)
    outgoing = state.active[side_idx]
    other_ability = state.ability[1-side_idx, 0]
    prior = state.volatile_status
    baton = transfer == 1
    carry_trick = baton & prior.properties.power_trick[side_idx, 0]
    carry_acid = baton & prior.ability_state.ability_suppressed[side_idx, 0]
    state = palafin_departure(restore_move_bank(state, side_idx), side_idx)
    state = clear_source_linked(state, side_idx)
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
        state.volatile_status.ability_state.ability_override[side_idx, 0] >= 0, -1, known[side_idx, slot])))
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
    # Last Resort counts only uses since this entry. Belch's Berry history is
    # independent and remains with the Pokémon for the rest of the battle.
    state = state.replace(active_index=active, toxic_counter=toxic_counter,
        used_move_ids=state.used_move_ids.at[side_idx, new_active].set(0))
    state = jax.lax.cond(carry_trick, partial(pass_power_trick, player=side_idx), _identity, state)
    state, _ = suppress_ability(state, side_idx, carry_acid)
    state = resume_ability(state, 1-side_idx, other_ability)
    state = jax.lax.cond(state.entry_recovery[side_idx] != 0,
                         partial(_apply_entry_recovery, player=side_idx), _identity, state)

    state = tera_shift(held_origin_form(held_crowned_form(state, side_idx), side_idx), side_idx)

    # hazards
    active = state.combat_active[side_idx]

    boots = active.item == ItemEnum.HEAVY_DUTY_BOOTS
    is_not_flying = state.is_grounded(side_idx) & ~boots
    is_not_hazard_immune = (1 - (active.ability == AbilityEnum.MAGIC_GUARD)) * ~boots
    is_poison = active.is_type(Type.POISON)
    is_poison_immune = jnp.any(active.type_list == Type.STEEL)
    # stealth rocks
    state = take_damage_percent(
        state,
        side_idx,
        state.hazards.stealth_rocks[side_idx] * calculate_effectiveness_multiplier(Type.ROCK, active.type_list) / 8 * is_not_hazard_immune
    )

    # spikes
    state = take_damage_percent(
        state, side_idx,
        (state.hazards.spikes[side_idx] != 0) / (10 - 2 * state.hazards.spikes[side_idx]) * is_not_flying * is_not_hazard_immune
    )

    # sticky webs
    # so apparently if a pokemon sets webs and a pokemon with mirror armor is effected by them
    # then it applies the stat reduction to that mon, but this only happens if they are on field
    state = conditional_reduce_boosts(state, side_idx,
        (is_not_flying * state.hazards.sticky_webs[side_idx]).reshape(()) != 0, StatEnum.SPEED, 1, False)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state.hazards.toxic_spikes[side_idx] * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.hazards.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(hazards=state.hazards.replace(toxic_spikes=toxic_spikes))
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state.hazards.toxic_spikes[side_idx]) * (state.hazards.toxic_spikes[side_idx] != 0)
    key, state = set_status(key, state, side_idx, status_ * (1-is_poison_immune) * is_not_flying)

    #check if switch in is still alive
    active_hp = state.active.current_hp[side_idx, 0]
    is_alive = active_hp != 0

    state = jax.lax.cond(is_alive, swap_is_alive, _unchanged_switch_state, state, side_idx)


    state = update_active_fields(state, side_idx, is_alive=is_alive)
    return state


def _apply_entry_recovery(state, *, player):
    """Spend a Healing Wish or Lunar Dance when a living entrant needs it."""
    hp = state.active_field('current_hp', player)[0]
    maximum = state.active_stats(player)[0]
    flag = state.entry_recovery[player]
    slot = state.active_index[player, 0]
    moves = state.team.moves
    pp = moves.current_pp[player, slot]
    max_pp = moves.max_pp[player, slot]
    restore_hp = (hp < maximum) | (state.active_field('status', player)[0] != 0)
    healing = (flag & EntryRecovery.HEALING_WISH) != 0
    dancing = (flag & EntryRecovery.LUNAR_DANCE) != 0
    dance_first = (flag & EntryRecovery.LUNAR_DANCE_FIRST) != 0
    use_wish = (hp > 0) & healing & restore_hp & ~(dancing & dance_first)
    use_dance = (hp > 0) & dancing & (jnp.any(pp < max_pp) | (restore_hp & ~use_wish))
    use = use_wish | use_dance
    state = take_damage_percent(state, player, jnp.where(use & (hp < maximum), -1., 0.))
    state = jax.lax.cond(use, partial(cure_status, player=player), _identity, state)
    moves = state.team.moves.replace(current_pp=state.team.moves.current_pp.at[player, slot].set(
        jnp.where(use_dance, max_pp, pp)))
    used = jnp.where(use_wish, EntryRecovery.HEALING_WISH, 0) | jnp.where(use_dance, EntryRecovery.LUNAR_DANCE, 0)
    used |= jnp.where(use, EntryRecovery.LUNAR_DANCE_FIRST, 0)
    return state.replace(team=state.team.replace(moves=moves),
                         entry_recovery=state.entry_recovery.at[player].set(flag & ~used))


def swap_is_alive(
    state: BattleState,
    side_idx) -> BattleState:
    """Apply represented weather, terrain, Intimidate and Frisk entry abilities."""
    state = tera_shift(state, side_idx)
    active = state.combat_active
    attacker = active[side_idx]

    state = start_ability(state, side_idx, entering=True)
    state, _ = transform(state, side_idx, attacker.ability[0] == AbilityEnum.IMPOSTER)
    state = refresh_trace_player(side_idx, state)
    state = reactive_form_update(state, side_idx, True)
    balloon = attacker.item[0] == ItemEnum.AIR_BALLOON

    state = remember_item(state, side_idx, ItemEnum.AIR_BALLOON, balloon)
    state = emit(state, EventKind.ITEM_REVEALED, side_idx, detail=ItemEnum.AIR_BALLOON, enabled=balloon)
    return update_items(state)
