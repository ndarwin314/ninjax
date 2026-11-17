"""Faint records, direct fainting and deferred Destiny Bond resolution."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.updates import update_active_fields


def _clear_fainted_destiny_bond(s, *, target):
    """Clear fainted destiny bond."""
    return clear_destiny_bond(s, target)


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def _resolve_faints_retaliate(s, *, target, safe_source, source):
    """Commit the direct faint with an explicit Destiny Bond cause."""
    old_source, old_cause, old_move = s.event_source, s.event_cause, s.event_move
    s = s.replace(event_source=jnp.asarray(target, jnp.int32), event_cause=jnp.int32(Cause.DESTINY_BOND), event_move=jnp.int32(194))
    mon = s.active[safe_source]
    s = emit(s, EventKind.HP, safe_source, before=mon.current_hp, after=0,
        hp_change=-mon.current_hp, hp_fraction_change=-jnp.ceil(mon.hp_percent * 100)/100)
    s = emit(s, EventKind.FAINT, safe_source)
    s = update_active_fields(s, safe_source, current_hp=jnp.zeros_like(mon.current_hp), is_alive=jnp.zeros_like(mon.is_alive))
    s = clear_destiny_bond(s, target)
    s = clear_destiny_bond(s, safe_source)
    # In Gen 9 the last Pokémon to faint wins a simultaneous final KO.
    both_out = ~jnp.any((s.team.current_hp[..., 0] > 0) & s.roster)
    return s.replace(event_source=old_source, event_cause=old_cause, event_move=old_move,
                     winner=jnp.where(both_out, source, s.winner))


def _resolve_faints_resolve(target, s):
    """Consume one pending faint and its captured Destiny Bond state."""
    source = s.faint_source[target]
    safe_source = jnp.maximum(source, 0)
    triggers = (s.faint_pending[target] & s.faint_destiny_bond[target] &
        (s.faint_cause[target] == Cause.MOVE) & (source >= 0) & (source != target) &
        (s.active_index[target, 0] == s.faint_target_slot[target]) &
        (s.active_index[safe_source, 0] == s.faint_source_slot[target]) &
        (s.active.current_hp[safe_source, 0] > 0))
    s = jax.lax.cond(s.faint_pending[target], partial(_clear_fainted_destiny_bond, target=target), _identity, s)
    s = s.replace(faint_pending=s.faint_pending.at[target].set(False),
                  faint_destiny_bond=s.faint_destiny_bond.at[target].set(False))
    retaliate = partial(_resolve_faints_retaliate, target=target, safe_source=safe_source, source=source)
    return jax.lax.cond(triggers, retaliate, _identity, s)


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
    state = update_active_fields(state, player, current_hp=jnp.zeros_like(mon.current_hp), is_alive=jnp.zeros_like(mon.is_alive))
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
    resolve = _resolve_faints_resolve
    return jax.lax.fori_loop(0, 2, resolve, state)
