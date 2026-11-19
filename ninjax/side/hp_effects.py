"""Damage and healing with protections, HP reactions and deferred faint records."""
from functools import partial

import chex
import jax
import jax.numpy as jnp

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


def take_damage_value(state: BattleState, defender_idx: int, damage: Array, is_attack_damage, fixed_damage=False, source=-1) -> BattleState:
    """Apply signed damage, with negative amounts healing the active Pokémon.

    Callers passing Python False use the indirect path. An array flag remains
    dynamic under JIT and gates attack protections and Innards Out at runtime.
    """
    if is_attack_damage is False:
        return apply_indirect_hp_change(state, defender_idx, damage, source)
    return _apply_attack_hp_change(state, defender_idx, damage, is_attack_damage, fixed_damage, source)


def _active_hp_fields(state, player):
    """Read the active slot and the HP fields used by both damage paths."""
    slot = state.active_index[player, 0]
    return slot, state.team.current_hp[player, slot], state.active_stats(player)[0]


@jax.jit
def _record_hp_change(state, player, old_hp, new_hp, max_hp, alive, attack):
    """Emit HP and faint events, then apply the opposing side's KO boost."""
    delta = (new_hp - old_hp).reshape(())
    # Public HP endpoints use the same rounding as observations.
    public_delta = (jnp.ceil(new_hp / max_hp * 100) - jnp.ceil(old_hp / max_hp * 100)) / 100
    state = emit(state, EventKind.HP, player, detail=jnp.asarray(attack, jnp.int32),
                 before=old_hp, after=new_hp, hp_change=delta,
                 hp_fraction_change=public_delta, enabled=delta != 0)
    state = emit(state, EventKind.FAINT, player, enabled=(old_hp > 0) & (new_hp == 0))

    opponent = 1 - player
    slot = state.active_index[opponent, 0]
    ability = state.ability[opponent]
    moxie = ability == AbilityEnum.MOXIE
    beast_boost = ability == AbilityEnum.BEAST_BOOST
    beast_stat = 1 + jnp.argmax(state.active_stats(opponent))
    index = jnp.where(beast_boost, beast_stat, StatEnum.ATTACK)
    eligible = ((moxie | beast_boost) & ~alive & (state.team.current_hp[opponent, slot] > 0))[0]
    return conditional_add_boosts(state, opponent, eligible, index, 1)


@jax.jit
def _finish_hp_change(state, player, old_hp, new_hp, alive, source, attack):
    """Commit HP, run reactive items and queue the original faint if needed."""
    state = update_active_fields(state, player, current_hp=new_hp, is_alive=alive)
    state = after_hp_change(state, player)
    indirect_cause = jnp.where(state.event_cause == Cause.MOVE, Cause.INDIRECT, state.event_cause)
    return record_faint(state, player, source, jnp.where(attack, Cause.MOVE, indirect_cause),
                        (old_hp[0] > 0) & (new_hp[0] == 0))


@jax.jit
def apply_indirect_hp_change(state, player, damage, source=-1):
    """Apply indirect damage or healing with the usual item and faint reactions."""
    slot, old_hp, max_hp = _active_hp_fields(state, player)
    # HP changes truncate fractional damage toward zero before clamping.
    damage = jnp.asarray(damage, dtype=old_hp.dtype)
    new_hp = jax.lax.clamp(0, old_hp - damage, max_hp.reshape(()))
    alive = (new_hp != 0) & state.team.is_alive[player, slot]
    state = _record_hp_change(state, player, old_hp, new_hp, max_hp, alive, jnp.bool_(False))
    return _finish_hp_change(state, player, old_hp, new_hp, alive, jnp.asarray(source, jnp.int32),
                             jnp.bool_(False))


def _innards_response(state, player, damage):
    """Apply Innards Out damage through the indirect HP path."""
    return apply_indirect_hp_change(state, player, damage)


def _unchanged_damage_state(state, player, damage):
    """Preserve state when Innards Out does not trigger."""
    return state


@jax.jit
def _apply_attack_hp_change(state, player, damage, attack, fixed_damage, source):
    """Apply attack protections and resolve Innards Out before committing HP."""
    slot, old_hp, max_hp = _active_hp_fields(state, player)
    ability = state.ability[player]
    item = state.team.item[player, slot]
    full = old_hp == max_hp
    multiscale = full & (ability == AbilityEnum.MULTISCALE) & attack & ~jnp.asarray(fixed_damage)
    damage = conditional_mult_round(damage, 0.5, multiscale)
    new_hp = jax.lax.clamp(0, old_hp - damage, max_hp.reshape(()))
    protected_ko = full & attack & (damage > 0) & (new_hp == 0)
    sturdy = protected_ko & (ability == AbilityEnum.STURDY)
    sash = protected_ko & (item == ItemEnum.FOCUS_SASH) & ~sturdy
    new_hp = jnp.maximum(new_hp, (sturdy | sash).astype(new_hp.dtype))
    alive = (new_hp != 0) & state.team.is_alive[player, slot]

    # Direct hits store actual damage before healing items have a chance to act.
    v = state.volatile_status
    last_damage = jnp.where(attack, jnp.maximum(old_hp - new_hp, 0), v.last_damage[player])
    state = state.replace(volatile_status=v.replace(last_damage=v.last_damage.at[player].set(last_damage)))
    state = emit(state, EventKind.ITEM, player, detail=ItemEnum.FOCUS_SASH, enabled=sash)
    state = remember_item(state, player, ItemEnum.NONE, sash)
    state = _record_hp_change(state, player, old_hp, new_hp, max_hp, alive, jnp.asarray(attack))

    # Preserve event order: retaliate after the original faint event, but
    # before committing the original target's HP and running its item hooks.
    opponent = 1 - player
    opponent_hp = state.team.current_hp[opponent, state.active_index[opponent, 0]]
    innards = (~alive & (ability == AbilityEnum.INNARDS_OUT) & attack & (opponent_hp > 0))[0]
    state = jax.lax.cond(innards, _innards_response, _unchanged_damage_state, state, opponent, old_hp)
    state = update_active_fields(state, player, item=jnp.where(sash, ItemEnum.NONE, item))
    return _finish_hp_change(state, player, old_hp, new_hp, alive, jnp.asarray(source, jnp.int32),
                             jnp.asarray(attack))


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
