"""Immutable leaf and stage writes without reactive item or ability hooks."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.stats import StatBoosts
from ninjax.enums.abilities import AbilityEnum
from ninjax.pokemon import Pokemon
from ninjax.records.events import EventKind, emit
from ninjax.side.state import BattleState


def _clear_volatile_row(value, *, side_idx):
    """Clear volatile row."""
    return value.at[side_idx].set(jnp.zeros_like(value[side_idx]))


def _record_boost(stat, carry):
    """Record a changed stage while carrying loop-invariant event values."""
    state, old, new, side_idx = carry
    a, b = old[stat], new[stat]
    return emit(state, EventKind.BOOST, side_idx, stat, a, b, enabled=a != b), old, new, side_idx


def _unchanged_boost_state(state, *unused):
    """Keep state unchanged when a conditional boost does not apply."""
    return state


def set_boosts(state: BattleState, side_idx, new_boosts):
    """Replace one side's stat-stage record and return the updated state."""
    old = jnp.concatenate((state.boosts.normal_boosts[side_idx], state.boosts.acc_boosts[side_idx]))
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    state = state.replace(boosts=new_boosts)
    new = jnp.concatenate((new_boosts.normal_boosts[side_idx], new_boosts.acc_boosts[side_idx]))
    return jax.lax.fori_loop(0, 8, _record_boost, (state, old, new, side_idx))[0]


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


def conditional_set_boosts(state: BattleState, side_idx, new_boosts, cond):
    """Set stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), set_boosts,
                        _unchanged_boost_state,
                        state, side_idx, new_boosts)


def conditional_add_boosts(state: BattleState, side_idx, cond, idx, val) -> BattleState:
    """Add stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), add_boosts,
                        _unchanged_boost_state,
                        state, side_idx, idx, val)


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


@jax.jit
def update_active_fields(state: BattleState, side_idx, **updates) -> BattleState:
    """Write named active array fields without scattering unrelated team data.

    Keyword names are static PyTree structure; values remain dynamic arrays.
    This is for local HP, status, item and type mutations. Full Pokémon
    replacement, including move/stat trees, still uses ``update_active``.
    """
    slot = state.active_index[side_idx, 0]
    fields = {name: getattr(state.team, name).at[side_idx, slot].set(
                  jnp.asarray(value, getattr(state.team, name).dtype))
              for name, value in updates.items()}
    old_status = state.team.status[side_idx, slot, 0]
    state = state.replace(team=state.team.replace(**fields))
    if 'status' in updates:
        new_status = state.team.status[side_idx, slot, 0]
        state = emit(state, EventKind.STATUS, side_idx, before=old_status,
                     after=new_status, enabled=old_status != new_status)
    return state


def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    """Reset all represented switch-cleared effects for one active side."""
    volatile = jax.tree.map(partial(_clear_volatile_row, side_idx=side_idx), state.volatile_status)
    volatile = volatile.replace(paradox_stat=volatile.paradox_stat.at[side_idx].set(-1),
                                choice_lock=volatile.choice_lock.at[side_idx].set(-1),
                                pending_actor_slot=volatile.pending_actor_slot.at[side_idx].set(-1),
                                ability_override=volatile.ability_override.at[side_idx].set(-1))
    return state.replace(volatile_status=volatile)
