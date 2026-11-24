"""Damage and healing with protections, HP reactions and deferred faint records."""
from functools import partial

import chex
import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.knowledge import remember_item
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.faint_effects import commit_hp_change
from ninjax.side.item_effects import after_hp_change
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


Array = jax.Array


def _damage_fraction_for_player(i, s, *, fractions):
    """Damage fraction for player."""
    return take_damage_percent(s, i, fractions[i])


def take_damage_value(state: BattleState, defender_idx: int, damage: Array, is_attack_damage, source=-1) -> BattleState:
    """Apply signed damage, with negative amounts healing the active Pokémon.

    Attack damage must already include all calculation modifiers. Callers
    passing Python False use the indirect path. An array flag remains
    dynamic under JIT and gates attack protections at runtime.
    """
    if is_attack_damage is False:
        return apply_indirect_hp_change(state, defender_idx, damage, source)
    return _apply_attack_hp_change(state, defender_idx, damage, is_attack_damage, source)


def _active_hp_fields(state, player):
    """Read the active slot and the HP fields used by both damage paths."""
    slot = state.active_index[player, 0]
    return slot, state.team.current_hp[player, slot], state.active_stats(player)[0]


def _indirect_cause(state):
    """Keep indirect effects distinct from the move that initiated them."""
    return jnp.where(state.event_cause == Cause.MOVE, Cause.INDIRECT, state.event_cause)


@jax.jit
def apply_indirect_hp_change(state, player, damage, source=-1):
    """Apply indirect damage or healing with the usual item and faint reactions."""
    _, old_hp, max_hp = _active_hp_fields(state, player)
    # HP changes truncate fractional damage toward zero before clamping.
    damage = jnp.asarray(damage, dtype=old_hp.dtype)
    new_hp = jax.lax.clamp(0, old_hp - damage, max_hp.reshape(()))
    state = commit_hp_change(state, player, old_hp, new_hp, max_hp, source, _indirect_cause(state))
    return after_hp_change(state, player)


@jax.jit
def _apply_attack_hp_change(state, player, damage, attack, source):
    """Apply attack protections, commit HP and check reactive items."""
    slot, old_hp, max_hp = _active_hp_fields(state, player)
    ability = state.ability[player]
    item = state.team.item[player, slot]
    full = old_hp == max_hp
    # Damage modifiers, including Multiscale, are resolved before HP commits.
    new_hp = jax.lax.clamp(0, old_hp - damage, max_hp.reshape(()))
    protected_ko = full & attack & (damage > 0) & (new_hp == 0)
    sturdy = protected_ko & (ability == AbilityEnum.STURDY)
    sash = protected_ko & (item == ItemEnum.FOCUS_SASH) & ~sturdy
    new_hp = jnp.maximum(new_hp, (sturdy | sash).astype(new_hp.dtype))
    state = emit(state, EventKind.ITEM, player, detail=ItemEnum.FOCUS_SASH, enabled=sash)
    state = remember_item(state, player, ItemEnum.NONE, sash)
    state = update_active_fields(state, player, item=jnp.where(sash, ItemEnum.NONE, item))
    state = commit_hp_change(state, player, old_hp, new_hp, max_hp, source,
                             jnp.where(attack, Cause.MOVE, _indirect_cause(state)), attack)
    return after_hp_change(state, player)


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
    amount = state.active_stats(defender_idx)[0] * percent
    damage = (jnp.sign(amount) * jnp.maximum(1, jnp.floor(jnp.abs(amount)))).astype(jnp.int32)
    return apply_indirect_hp_change(state, defender_idx, damage)
