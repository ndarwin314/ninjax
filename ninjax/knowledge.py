"""Public item knowledge updates for immutable battle state."""
import jax.numpy as jnp

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
