"""Public item knowledge and switch-in ability effects for singles battles."""
import jax.numpy as jnp
from ninjax.enum_types import AbilityEnum, ItemEnum
from ninjax.events import EventKind, emit


def remember_item(state, owner, item, enabled=True):
    """Retain a publicly revealed item for the owner's active team slot.

    `known_items` has shape (2 owners, 6 slots); -1 means unknown. Store zero
    after a publicly observed consumption so a previous Frisk reveal stays
    accurate without exposing unrelated private fields.
    """
    slot = state.active_index[owner, 0]
    known = state.known_items
    value = jnp.where(jnp.asarray(enabled).reshape(()), jnp.asarray(item).reshape(()), known[owner, slot])
    return state.replace(known_items=known.at[owner, slot].set(value))


def frisk(state, source):
    """Reveal a living opponent's held item; an empty item slot reveals nothing."""
    target = 1-source
    item = state.active.item[target, 0]
    enabled = (state.active.ability[source, 0] == AbilityEnum.FRISK) & (
        state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & (item != ItemEnum.NONE)
    state = remember_item(state, target, item, enabled)
    previous_source = state.event_source
    state = state.replace(event_source=jnp.asarray(source, jnp.int32))
    state = emit(state, EventKind.ABILITY, source, detail=AbilityEnum.FRISK, enabled=enabled)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=item, enabled=enabled)
    return state.replace(event_source=previous_source)


def prepare_type_change(state, player, move):
    """Apply Gen 9 Protean/Libero once per switch, only on an actual type change."""
    from ninjax.side import update_active
    from ninjax.enum_types import Type
    mon = state.active[player]
    used = state.volatile_status.libero_used[player, 0]
    # A monotype matching the move does not consume the activation; a dual
    # type containing it still changes to a monotype and does consume it.
    matching = (mon.type_list[0] == move.type[0]) & (mon.type_list[1] == Type.NONE)
    trigger = (mon.ability[0] == AbilityEnum.PROTEAN) & ~used & ~matching & (
        move.type[0] != Type.NONE) & ~mon.is_terastallized[0] & move.implemented[0]
    types = jnp.where(trigger, jnp.array([move.type[0], Type.NONE]), mon.type_list)
    state = update_active(state, player, mon.replace(type_list=types))
    state = emit(state, EventKind.ABILITY, player, detail=mon.ability[0], enabled=trigger)
    return state.replace(volatile_status=state.volatile_status.replace(
        libero_used=state.volatile_status.libero_used.at[player, 0].set(used | trigger)))
