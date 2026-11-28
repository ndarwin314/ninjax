"""Gen 9 singles trapping eligibility and source-linked binding state."""
import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Type
from ninjax.item_rules import item_is_suppressed
from ninjax.side.queries import is_floating_fields
from ninjax.records.events import EventKind, emit


def _item_suppressed(state):
    """Read suppression for each active holder."""
    return item_is_suppressed(state.ability, state.magic_room_duration,
                              state.volatile_status.embargo_turns)


def active_binding(state):
    """Return binding records whose original opposing source remains active."""
    traps = state.volatile_status.trapping
    hp = state.active_field('current_hp')
    return ((traps.binding_turns > 0) & (traps.binding_source_slot == state.active_index[::-1]) &
            (hp[::-1] > 0) & (hp > 0))


def active_linked_trap(state):
    """Read source-linked switching locks independently of binding damage."""
    traps = state.volatile_status.trapping
    hp = state.active_field('current_hp')
    return (traps.lock_source_slot == state.active_index[::-1]) & (hp[::-1] > 0) & (hp > 0)


def apply_linked_trap(state, target, source, enabled=True):
    """Latch a trap to both participants' active lifetimes."""
    traps = state.volatile_status.trapping
    hp = state.active_field('current_hp')
    installed = enabled & ~active_linked_trap(state)[target, 0] & ~traps.no_retreat[target, 0] & (
        (hp[source, 0] > 0) & (hp[target, 0] > 0))
    slot = jnp.where(installed, state.active_index[source, 0], traps.lock_source_slot[target, 0])
    return _replace_trapping(state, traps.replace(
        lock_source_slot=traps.lock_source_slot.at[target, 0].set(slot))), installed


def voluntary_switch_blocked(state):
    """Return one flag per side for a new voluntary switch choice."""
    types = state.effective_types(include_added=True)
    ability = state.ability
    opposing = ability[::-1]
    item = jnp.where(_item_suppressed(state), I.NONE, state.active_field('item'))
    hp = state.active_field('current_hp')
    grounded = ~is_floating_fields(types, ability, item) | state.grounding_override()
    ghost = jnp.any(types == Type.GHOST, axis=-1, keepdims=True)
    steel = jnp.any(types == Type.STEEL, axis=-1, keepdims=True)
    ability_trap = ((opposing == A.SHADOW_TAG) & (ability != A.SHADOW_TAG) |
                    (opposing == A.ARENA_TRAP) & grounded |
                    (opposing == A.MAGNET_PULL) & steel)
    own = state.volatile_status.trapping.no_retreat | (state.volatile_status.residuals.ingrain != 0)
    trapped = (ability_trap & (hp[::-1] > 0)) | active_binding(state) | active_linked_trap(state) | own | (
        state.fairy_lock_turns > 0)
    return (trapped & ~ghost & (item != I.SHED_SHELL) & (hp > 0))[:, 0]


def _replace_trapping(state, traps):
    """Write the trapping subtree without touching other volatile effects."""
    return state.replace(volatile_status=state.volatile_status.replace(trapping=traps))


def apply_binding(key, state, *, source, move_id):
    """Install a fresh binding effect after a direct hit on a living target."""
    target = 1 - source
    traps = state.volatile_status.trapping
    already_bound = active_binding(state)[target, 0]
    enabled = ~already_bound & (state.active_field('current_hp', target)[0] > 0)
    key, duration_key = jax.random.split(key)
    item = jnp.where(_item_suppressed(state)[source, 0], I.NONE, state.active_field('item', source)[0])
    # Expiry precedes the residual callback, so five or six timer counts
    # produce four or five ticks including the turn on which the move hits.
    duration = jnp.where(item == I.GRIP_CLAW, 8, jax.random.randint(duration_key, (), 5, 7))
    divisor = jnp.where(item == I.BINDING_BAND, 6, 8)
    slot = state.active_index[source, 0]
    updated = traps.replace(
        binding_source_slot=traps.binding_source_slot.at[target, 0].set(slot),
        binding_move_id=traps.binding_move_id.at[target, 0].set(move_id),
        binding_turns=traps.binding_turns.at[target, 0].set(duration),
        binding_divisor=traps.binding_divisor.at[target, 0].set(divisor))
    return key, jax.lax.cond(enabled, _replace_trapping, _keep_trapping, state, updated)


def _keep_trapping(state, traps):
    """Keep the previous trap when another cannot be installed."""
    return state


def clear_binding(state, target):
    """Remove the chosen target's binding effect on escape or source departure."""
    traps = state.volatile_status.trapping
    return _replace_trapping(state, traps.replace(
        binding_source_slot=traps.binding_source_slot.at[target, 0].set(-1),
        binding_move_id=traps.binding_move_id.at[target, 0].set(0),
        binding_turns=traps.binding_turns.at[target, 0].set(0),
        binding_divisor=traps.binding_divisor.at[target, 0].set(0)))


def clear_source_linked(state, source):
    """Remove opponent effects that expire when their original source leaves."""
    target = 1-source
    state = clear_binding(state, target)
    v = state.volatile_status
    r = v.residuals
    state = emit(state, EventKind.CONDITION, target, detail=903, before=1, after=0,
                 enabled=r.syrup_turns[target, 0] > 0)
    return state.replace(volatile_status=v.replace(
        trapping=v.trapping.replace(lock_source_slot=v.trapping.lock_source_slot.at[target, 0].set(-1)),
        residuals=r.replace(syrup_source_slot=r.syrup_source_slot.at[target, 0].set(-1),
                             syrup_turns=r.syrup_turns.at[target, 0].set(0))))


def tick_binding(state):
    """Expire bindings before the order-13 residual damage phase."""
    traps = state.volatile_status.trapping
    turns = jnp.where(active_binding(state), jnp.maximum(traps.binding_turns - 1, 0), 0)
    return _replace_trapping(state, traps.replace(binding_turns=turns,
        binding_source_slot=jnp.where(turns > 0, traps.binding_source_slot, -1),
        binding_move_id=jnp.where(turns > 0, traps.binding_move_id, 0),
        binding_divisor=jnp.where(turns > 0, traps.binding_divisor, 0)))
