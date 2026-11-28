"""Application and duration handling for move-created residual conditions."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.modifiers import apply_modifier
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import TerrainEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Status, StatEnum, Type
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import held_item_active
from ninjax.side.stat_effects import stat_changes
from ninjax.side.status_effects import safeguard_blocks, set_status
from ninjax.side.trapping import apply_linked_trap


RESIDUAL_STATUS_MOVES = (73, 174, 212, 273, 275, 281, 335, 392, 587, 748)


def _keep_state(state, *unused):
    """Leave state unchanged when the condition has no effect."""
    return state


def _keep_key_state(key, state, player):
    """Leave key and state unchanged while no Yawn is due."""
    return key, state


def _write_condition(state, target, name, value, enabled):
    """Update one active timer with its existing value on a failed application."""
    volatile = state.volatile_status
    residuals = volatile.residuals
    field = getattr(residuals, name)
    field = field.at[target, 0].set(jnp.where(enabled, value, field[target, 0]))
    return state.replace(volatile_status=volatile.replace(residuals=residuals.replace(**{name: field})))


def clear_leech_seed(state, target):
    """Clear Leech Seed when Rapid Spin or Mortal Spin frees the holder."""
    seeded = state.volatile_status.residuals.leech_seed[target, 0] != 0
    state = _write_condition(state, target, 'leech_seed', 0, seeded)
    return emit(state, EventKind.CONDITION, target, detail=73, before=1, after=0, enabled=seeded)


def root_healing(state, player, amount):
    """Apply the represented Big Root recovery modifier to a healing amount."""
    active = held_item_active(state, player) & (state.active_field('item', player)[0] == I.BIG_ROOT)
    return apply_modifier(amount, jnp.where(active, 5324, 4096))


def ghost_curse_user(state, player):
    """Choose the Ghost Curse effect using the user's current typing."""
    types = state.effective_types(player, include_added=True)
    tera = state.active_field('is_terastallized', player)[0]
    type_id = state.active_field('tera_type', player)[0]
    return jnp.where(tera & (type_id != Type.NONE), type_id == Type.GHOST,
                     jnp.any(types == Type.GHOST))


def _self_healing_condition(state, player, move):
    """Install an indefinite Aqua Ring or Ingrain without refreshing it."""
    ident = move.identifier[0]
    r = state.volatile_status.residuals
    ring = ident == 392
    success = jnp.where(ring, r.aqua_ring[player, 0] == 0, r.ingrain[player, 0] == 0)
    state = _write_condition(state, player, 'aqua_ring', -1, ring & success)
    return _write_condition(state, player, 'ingrain', -1, ~ring & success), success


def _opposing_condition(state, player, move):
    """Apply Leech Seed or Yawn after their type or sleep eligibility gate."""
    target = 1-player
    r = state.volatile_status.residuals
    seed = move.identifier[0] == 73
    grass = jnp.any(state.effective_types(target, include_added=True) == Type.GRASS)
    # Electric Terrain stops drowsiness itself. Misty Terrain stops the sleep
    # attempt at expiry if it is still active then.
    sleep_prevented = state.is_status_immune(target, jnp.array([Status.SLEEP]), check_terrain=False)[0]
    sleep_prevented |= (state.terrain.terrain == TerrainEnum.ELECTRIC) & state.is_grounded(target)[0]
    sleep_prevented |= (state.active_field('status', target)[0] != Status.NONE) | (state.ability[target, 0] == A.PURIFYING_SALT)
    sleep_prevented |= safeguard_blocks(state, target)
    success = jnp.where(seed, (r.leech_seed[target, 0] == 0) & ~grass,
                        (r.yawn_turns[target, 0] == 0) & ~sleep_prevented)
    success &= state.active_field('current_hp', target)[0] > 0
    state = _write_condition(state, target, 'leech_seed', -1, seed & success)
    return _write_condition(state, target, 'yawn_turns', 2, ~seed & success), success


def _wish(state, player, move):
    """Snapshot the caster's HP into a Wish that follows the active position."""
    wish = state.slot_residuals
    success = wish.wish_turns[player, 0] == 0
    hp = jnp.maximum(1, state.active_stats(player)[0] // 2)
    return state.replace(slot_residuals=wish.replace(
        wish_turns=wish.wish_turns.at[player, 0].set(jnp.where(success, 2, wish.wish_turns[player, 0])),
        wish_hp=wish.wish_hp.at[player, 0].set(jnp.where(success, hp, wish.wish_hp[player, 0])))), success


def _curse_cost(state, player):
    """Pay Ghost Curse's direct HP cost, which can faint the user."""
    cost = jnp.maximum(1, state.active_stats(player)[0] // 2)
    return take_damage_value(state, player, jnp.array([cost]), False, source=player)


def _curse(state, player, move):
    """Apply Ghost Curse or its ordinary Attack, Defense and Speed changes."""
    ghost = ghost_curse_user(state, player)
    target = 1-player
    success = ~ghost | ((state.volatile_status.residuals.curse[target, 0] == 0) &
                        (state.active_field('current_hp', target)[0] > 0))
    state = _write_condition(state, target, 'curse', -1, ghost & success)
    state = jax.lax.cond(ghost & success, _curse_cost, _keep_state, state, player)
    stats = jnp.array([StatEnum.ATTACK, StatEnum.DEFENSE, StatEnum.SPEED])
    stages = jnp.where(ghost, jnp.zeros(3, jnp.int32), jnp.array([1, 1, -1]))
    return stat_changes(state, player, stats, stages), success


def _linked_trap(state, player, move):
    """Attach Mean Look or Block to the current source slot."""
    return apply_linked_trap(state, 1-player, player)


def _no_retreat(state, player, move):
    """Boost all stats and add self-trapping when no linked lock exists."""
    volatile = state.volatile_status
    traps = volatile.trapping
    success = ~traps.no_retreat[player, 0]
    already_linked = traps.lock_source_slot[player, 0] >= 0
    state = state.replace(volatile_status=volatile.replace(trapping=traps.replace(
        no_retreat=traps.no_retreat.at[player, 0].set(~success | ~already_linked))))
    return stat_changes(state, player, move.boost_stat, jnp.where(success, move.boost_stages, 0)), success


def _fairy_lock(state, player, move):
    """Set the field-wide switch lock through the following turn."""
    success = state.fairy_lock_turns == 0
    return state.replace(fairy_lock_turns=jnp.where(success, 2, state.fairy_lock_turns)), success


@jax.jit
def apply_residual_status(state, player, move):
    """Apply a primary move-created residual condition and report success."""
    ident = move.identifier[0]
    branch = jnp.select(((ident == 392) | (ident == 275), (ident == 73) | (ident == 281),
                         ident == 273, ident == 174, (ident == 212) | (ident == 335),
                         ident == 748), (0, 1, 2, 3, 4, 5), default=6)
    state, success = jax.lax.switch(branch, (_self_healing_condition, _opposing_condition,
        _wish, _curse, _linked_trap, _no_retreat, _fairy_lock), state, player, move)
    self_effect = (branch == 0) | (branch == 2) | (branch >= 5) | ((ident == 174) & ~ghost_curse_user(state, player))
    target = jnp.where(self_effect, player, 1-player)
    state = emit(state, EventKind.CONDITION, target, detail=ident, after=1, enabled=success)
    return state, success


def apply_residual_secondary(state, target, move):
    """Install Salt Cure, Syrup Bomb or Spirit Shackle after a valid secondary."""
    ident = move.identifier[0]
    source = 1-target
    r = state.volatile_status.residuals
    salt = (ident == 864) & (r.salt_cure[target, 0] == 0)
    syrup = (ident == 903) & (r.syrup_turns[target, 0] == 0)
    state = _write_condition(state, target, 'salt_cure', -1, salt)
    state = _write_condition(state, target, 'syrup_turns', 4, syrup)
    state = _write_condition(state, target, 'syrup_source_slot', state.active_index[source, 0], syrup)
    state, linked = apply_linked_trap(state, target, source, enabled=ident == 662)
    return emit(state, EventKind.CONDITION, target, detail=ident, after=1, enabled=salt | syrup | linked)


def _syrup_tick(index, state, *, order):
    """Apply a Speed drop in residual order while the original source remains."""
    player = order[index]
    source = 1-player
    r = state.volatile_status.residuals
    turns = r.syrup_turns[player, 0]
    active = (turns > 0) & (r.syrup_source_slot[player, 0] == state.active_index[source, 0]) & (
        state.active_field('current_hp', source)[0] > 0) & (state.active_field('current_hp', player)[0] > 0)
    left = jnp.where(active, jnp.maximum(0, turns - 1), 0)
    state = _write_condition(state, player, 'syrup_turns', left, True)
    state = _write_condition(state, player, 'syrup_source_slot', -1, left == 0)
    old = state.event_context
    state = state.replace(event_context=old.replace(source=source, move_id=jnp.int32(903), cause=jnp.int32(Cause.INDIRECT)))
    state = stat_changes(state, player, jnp.array([StatEnum.SPEED]), jnp.array([jnp.where(left > 0, -1, 0)]), True)
    state = emit(state, EventKind.CONDITION, player, detail=903, before=1, after=0, enabled=(turns > 0) & (left == 0))
    return state.replace(event_context=old)


@jax.jit
def tick_syrup_bomb(state, order):
    """Tick Syrup Bomb after order-13 HP damage."""
    return jax.lax.fori_loop(0, 2, partial(_syrup_tick, order=order), state)


def _expire_yawn(key, state, player):
    """Try sleep with current immunities when the drowsiness expires."""
    old = state.event_context
    state = state.replace(event_context=old.replace(source=jnp.int32(-1), move_id=jnp.int32(281), cause=jnp.int32(Cause.STATUS)))
    state = emit(state, EventKind.CONDITION, player, detail=281, before=1, after=0)
    status = jnp.where(state.ability[player, 0] == A.PURIFYING_SALT, Status.NONE, Status.SLEEP)
    key, state = set_status(key, state, player, jnp.array([status]))
    return key, state.replace(event_context=old)


def _yawn_tick(index, carry, *, order):
    """Expire one active position's Yawn at residual order 23."""
    key, state = carry
    player = order[index]
    turns = state.volatile_status.residuals.yawn_turns[player, 0]
    state = _write_condition(state, player, 'yawn_turns', jnp.maximum(0, turns - 1), turns > 0)
    ready = (turns == 1) & (state.active_field('current_hp', player)[0] > 0)
    return jax.lax.cond(ready, _expire_yawn, _keep_key_state, key, state, player)


@jax.jit
def tick_yawn(key, state, order):
    """Tick Yawn after ordinary residual status interference and before perish."""
    return jax.lax.fori_loop(0, 2, partial(_yawn_tick, order=order), (key, state))
