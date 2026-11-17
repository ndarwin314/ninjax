"""Damage and healing with protections, HP reactions and deferred faint records."""
from functools import partial

import chex
import jax
import jax.numpy as jnp

from ninjax.calculations.logic import triple_and
from ninjax.calculations.modifiers import conditional_mult_round
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import StatEnum
from ninjax.knowledge import remember_item
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.faint_effects import record_faint
from ninjax.side.item_effects import after_hp_change
from ninjax.side.state import BattleState
from ninjax.side.updates import conditional_add_boosts, update_active_fields


Array = jax.Array


def _damage_fraction_for_player(i, s, *, fractions):
    """Damage fraction for player."""
    return take_damage_percent(s, i, fractions[i])


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


def _innards_response(state, player, damage):
    """Apply Innards Out once without recursively invoking the same ability."""
    return _take_damage_value(state, player, damage, False, False)


def _unchanged_damage_state(state, player, damage):
    """Preserve state when Innards Out does not trigger."""
    return state


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

    sash = health_full & (defender.item == ItemEnum.FOCUS_SASH) & jnp.asarray(is_attack_damage) & (damage > 0) & (new_health == 0) & ~sturdy_survives
    new_health = jnp.maximum(new_health, (sturdy_survives | sash).astype(new_health.dtype))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(last_damage=v.last_damage.at[defender_idx].set(
        jnp.where(jnp.asarray(is_attack_damage), jnp.maximum(old_health-new_health, 0), v.last_damage[defender_idx]))))
    state = emit(state, EventKind.ITEM, defender_idx, detail=ItemEnum.FOCUS_SASH, enabled=sash)

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
            _innards_response,
            _unchanged_damage_state,
            state, 1-defender_idx, old_health
        )
    # TODO: consider making separate functions for take damage value and take damage from attack
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere

    state = after_hp_change(update_active_fields(state, defender_idx,
        current_hp=defender.current_hp, is_alive=defender.is_alive, item=defender.item), defender_idx)
    return record_faint(state, defender_idx, source,
        jnp.where(is_attack_damage, Cause.MOVE,
            jnp.where(state.event_cause == Cause.MOVE, Cause.INDIRECT, state.event_cause)),
        (old_health[0] > 0) & (new_health[0] == 0))


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
            partial(_damage_fraction_for_player, fractions=fractions), state)
    amount = state.active.max_hp[defender_idx] * percent
    damage = (jnp.sign(amount) * jnp.maximum(1, jnp.floor(jnp.abs(amount)))).astype(jnp.int32)
    return take_damage_value(state, defender_idx, damage, False)
