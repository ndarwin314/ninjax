"""Switch-cleared restrictions on move selection and execution."""
import jax
import jax.numpy as jnp
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, MoveType
from ninjax.held_effects import consume

# Scarlet/Violet move durations, separated from residual timing adjustments so
# a future ruleset can supply different values at a single boundary.
DISABLE_TURNS = 4
ENCORE_TURNS = 3
TAUNT_TURNS = 3


def allowed_moves(move_ids, move_types, *, disabled_move_id, disable_turns,
                  taunt_turns, encore_move_id, encore_turns, gravity_duration):
    """Return restriction eligibility (2, 4) from active move/volatile arrays.

    Move IDs/categories have shape (2, 4), restriction arrays (2, 1), and
    gravity duration is scalar. PP and Choice/item restrictions are separate.
    """
    return (~((disable_turns > 0) & (move_ids == disabled_move_id)) &
            ~((taunt_turns > 0) & (move_types == MoveType.STATUS)) &
            ~((gravity_duration > 0) & (move_ids == 136)) &
            ((encore_turns <= 0) | (move_ids == encore_move_id)))


def resolve_move(move_ids, move_types, current_pp, index, *, disabled_move_id,
                 disable_turns, taunt_turns, encore_move_id, encore_turns,
                 gravity_duration):
    """Resolve a queued slot from (4,) move arrays and scalar restrictions.

    Encore redirects only to a matching move with PP. Invalid/exhausted Encore
    does not itself block the original selection; Disable/Taunt/gravity still
    apply to the resolved slot.
    """
    match = (move_ids == encore_move_id) & (current_pp > 0)
    use_encore = (encore_turns > 0) & jnp.any(match)
    index = jnp.where(use_encore, jnp.argmax(match), index)
    identifier = move_ids[index]
    blocked = ((disable_turns > 0) & (identifier == disabled_move_id)) | (
        (taunt_turns > 0) & (move_types[index] == MoveType.STATUS))
    blocked |= (gravity_duration > 0) & (identifier == 136)
    return index, blocked


def apply_restriction(state, attacker, move):
    """Apply a valid effect, including duration timing and Mental Herb cure."""
    target = 1-attacker
    v = state.volatile_status
    ident = move.identifier[0]
    last = v.last_move_id[target, 0]
    mon = state.active[target]
    matching = (mon.moves.identifier[:, 0] == last) & (mon.moves.current_pp[:, 0] > 0)
    eligible = jnp.any(matching)
    selected = jnp.argmax(matching)
    encore_ok = eligible & mon.moves.encore_eligible[selected, 0]
    protected = (state.ability[target, 0] == A.AROMA_VEIL) | ((ident == 269) & (state.ability[target, 0] == A.OBLIVIOUS))
    disable = (ident == 50) & eligible & (last != 0) & (v.disable_turns[target, 0] == 0) & ~protected
    encore = (ident == 227) & encore_ok & (last != 0) & (v.encore_turns[target, 0] == 0) & ~protected
    taunt = (ident == 269) & (v.taunt_turns[target, 0] == 0) & ~protected
    pending = v.pending_move[target, 0]
    # Showdown's residual countdown depends on whether the target can still act.
    disable_duration = DISABLE_TURNS + (~pending).astype(jnp.int32)
    encore_duration = ENCORE_TURNS + (~pending).astype(jnp.int32)
    taunt_duration = TAUNT_TURNS + (~pending & (v.entry_turn[target, 0] < state.turn_number)).astype(jnp.int32)
    v = v.replace(disabled_move_id=v.disabled_move_id.at[target, 0].set(jnp.where(disable, last, v.disabled_move_id[target, 0])),
        disable_turns=v.disable_turns.at[target, 0].set(jnp.where(disable, disable_duration, v.disable_turns[target, 0])),
        encore_move_id=v.encore_move_id.at[target, 0].set(jnp.where(encore, last, v.encore_move_id[target, 0])),
        encore_turns=v.encore_turns.at[target, 0].set(jnp.where(encore, encore_duration, v.encore_turns[target, 0])),
        taunt_turns=v.taunt_turns.at[target, 0].set(jnp.where(taunt, taunt_duration, v.taunt_turns[target, 0])))
    state = state.replace(volatile_status=v)
    herb = (disable | encore | taunt) & (mon.item[0] == I.MENTAL_HERB)
    state = consume(state, target, herb)
    v = state.volatile_status
    return state.replace(volatile_status=v.replace(
        disabled_move_id=v.disabled_move_id.at[target, 0].set(jnp.where(herb, 0, v.disabled_move_id[target, 0])),
        disable_turns=v.disable_turns.at[target, 0].set(jnp.where(herb, 0, v.disable_turns[target, 0])),
        encore_move_id=v.encore_move_id.at[target, 0].set(jnp.where(herb, 0, v.encore_move_id[target, 0])),
        encore_turns=v.encore_turns.at[target, 0].set(jnp.where(herb, 0, v.encore_turns[target, 0])),
        taunt_turns=v.taunt_turns.at[target, 0].set(jnp.where(herb, 0, v.taunt_turns[target, 0]))))


def tick(state):
    """Expire restrictions at residual time, including exhausted Encore PP."""
    v = state.volatile_status
    ids = state.active.moves.identifier[..., 0]
    pp = state.active.moves.current_pp[..., 0]
    valid = jnp.any((ids == v.encore_move_id) & (pp > 0), axis=1, keepdims=True)
    disable_turns = jnp.maximum(v.disable_turns-1, 0)
    encore_turns = jnp.where(valid, jnp.maximum(v.encore_turns-1, 0), 0)
    return state.replace(volatile_status=v.replace(
        disabled_move_id=jnp.where(disable_turns > 0, v.disabled_move_id, 0),
        disable_turns=disable_turns,
        taunt_turns=jnp.maximum(v.taunt_turns-1, 0),
        encore_move_id=jnp.where(encore_turns > 0, v.encore_move_id, 0),
        encore_turns=encore_turns,
        pending_move=jnp.zeros_like(v.pending_move),
        queued_priority_valid=jnp.zeros_like(v.queued_priority_valid),
        encore_overridden=jnp.zeros_like(v.encore_overridden)))
