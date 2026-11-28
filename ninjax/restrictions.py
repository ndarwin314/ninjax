"""Switch-cleared restrictions on move selection and execution."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import MoveFlags, MoveType
from ninjax.item_rules import item_is_suppressed
from ninjax.side.item_effects import after_hp_change, consume, is_berry, mental_herb


# Scarlet/Violet move durations, separated from residual timing adjustments so
# a future ruleset can supply different values at a single boundary.
THROAT_CHOP_TURNS = 2
HEAL_BLOCK_TURNS = 2
DISABLE_TURNS = 4
ENCORE_TURNS = 3
TAUNT_TURNS = 3


def gravity_blocks_move(identifier, gravity_duration):
    """Enforce the catalogue's Gen 9 Gravity flags at selection and execution."""
    return (gravity_duration > 0) & jnp.isin(identifier, jnp.array([19, 136, 150, 340, 393, 560]))


def blocked_by_hit_restrictions(is_sound, is_healing, throat_chop_turns, heal_block_turns):
    """Check whether a move's flags are blocked by either hit restriction."""
    return ((jnp.asarray(throat_chop_turns) > 0) & is_sound) | ((jnp.asarray(heal_block_turns) > 0) & is_healing)


def move_blocked_at_execution(state, player, move, *, called=False):
    """Check the live restrictions on the move about to execute."""
    restrictions = state.volatile_status.restrictions
    foe = 1 - player
    # Sleep Talk calls its move outside the BeforeMove event Imprison uses.
    imprisoned = False
    if not called:
        imprisoned = (restrictions.imprisoning[foe, 0] & (state.active_field('current_hp', foe)[0] > 0) &
                      jnp.any(state.active_move_field('identifier', foe)[:, 0] == move.identifier[0]))
    return imprisoned | blocked_by_hit_restrictions(
        move.sound[0], move.healing[0], restrictions.throat_chop_turns[player, 0],
        restrictions.heal_block_turns[player, 0])


def allowed_moves(
        move_ids, move_types, is_sound=False, is_healing=False, *, disabled_move_id, disable_turns,
        taunt_turns, encore_move_id, encore_turns,
        throat_chop_turns=0, heal_block_turns=0, gravity_duration):
    """Return restriction eligibility (2, 4) from active move/volatile arrays.

    Move IDs/categories have shape (2, 4), restriction arrays (2, 1), and
    gravity duration is scalar. PP and Choice/item restrictions are separate.
    """
    return (~((disable_turns > 0) & (move_ids == disabled_move_id)) &
            ~((taunt_turns > 0) & (move_types == MoveType.STATUS)) &
            ~gravity_blocks_move(move_ids, gravity_duration) &
            ((encore_turns <= 0) | (move_ids == encore_move_id)) &
            ~blocked_by_hit_restrictions(is_sound, is_healing, throat_chop_turns, heal_block_turns))


def active_restriction_mask(state):
    """Read active move flags and restriction timers for masks and observations."""
    restrictions = state.volatile_status.restrictions
    flags = state.active_move_field('move_flags')[..., 0]
    ids = state.active_move_field('identifier')[..., 0]
    shared = jnp.any(ids[:, :, None] == ids[::-1, None, :], axis=-1)
    imprisoned = restrictions.imprisoning[::-1] & (state.active_field('current_hp')[::-1] > 0) & shared
    repeated = ids == state.volatile_status.move_history.last_move_id
    cannot_repeat = restrictions.tormented | ((flags & MoveFlags.CANNOT_REPEAT) != 0)
    legal = allowed_moves(ids, state.active_move_field('move_type')[..., 0],
        (flags & MoveFlags.SOUND) != 0, (flags & MoveFlags.HEALING) != 0,
        disabled_move_id=restrictions.disabled_move_id, disable_turns=restrictions.disable_turns,
        taunt_turns=restrictions.taunt_turns, encore_move_id=restrictions.encore_move_id,
        encore_turns=restrictions.encore_turns,
        throat_chop_turns=restrictions.throat_chop_turns, heal_block_turns=restrictions.heal_block_turns,
        gravity_duration=state.gravity_duration)
    stuff_cheeks_blocked = (ids == 747) & ~is_berry(state.active_field('item'))
    eaten = state.ate_berry[jnp.arange(2), state.active_index[:, 0]][:, None]
    belch_blocked = (ids == 562) & ~eaten
    return legal & ~imprisoned & ~(cannot_repeat & repeated) & ~stuff_cheeks_blocked & ~belch_blocked


def last_resort_fails(state, player, move):
    """Check Last Resort against the currently held moves and their use history.

    The move is selectable before this prerequisite is met. It fails during
    execution after paying PP, including when called from another move.
    """
    slot = state.active_index[player, 0]
    move_ids = state.active_move_field('identifier', player)[:, 0]
    used = state.used_move_ids[player, slot]
    present = move_ids != 0
    last_resort = move_ids == 387
    others_used = jnp.all(~present | last_resort | (used == move_ids))
    return (move.identifier[0] == 387) & (
        (jnp.sum(present) < 2) | ~jnp.any(last_resort) | ~others_used)


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
    blocked |= gravity_blocks_move(identifier, gravity_duration)
    return index, blocked


def apply_restriction(state, attacker, move):
    """Apply a valid effect, including duration timing and Mental Herb cure."""
    target = 1-attacker
    v = state.volatile_status
    ident = move.identifier[0]
    last = v.move_history.last_move_id[target, 0]
    ids = state.active_move_field('identifier', target)[:, 0]
    pp = state.active_move_field('current_pp', target)[:, 0]
    matching = (ids == last) & (pp > 0)
    eligible = jnp.any(matching)
    selected = jnp.argmax(matching)
    encore_ok = eligible & state.active_move_field('encore_eligible', target)[selected, 0]
    protected = (state.ability[target, 0] == A.AROMA_VEIL) | ((ident == 269) & (state.ability[target, 0] == A.OBLIVIOUS))
    disable = (ident == 50) & eligible & (last != 0) & (v.restrictions.disable_turns[target, 0] == 0) & ~protected
    encore = (ident == 227) & encore_ok & (last != 0) & (v.restrictions.encore_turns[target, 0] == 0) & ~protected
    taunt = (ident == 269) & (v.restrictions.taunt_turns[target, 0] == 0) & ~protected
    pending = v.queued_move.pending_move[target, 0]
    # Showdown's residual countdown depends on whether the target can still act.
    disable_duration = DISABLE_TURNS + (~pending).astype(jnp.int32)
    encore_duration = ENCORE_TURNS + (~pending).astype(jnp.int32)
    taunt_duration = TAUNT_TURNS + (~pending & (v.entry_turn[target, 0] < state.turn_number)).astype(jnp.int32)
    v = v.replace(
        restrictions=v.restrictions.replace(
            disabled_move_id=v.restrictions.disabled_move_id.at[target, 0].set(jnp.where(disable, last, v.restrictions.disabled_move_id[target, 0])),
            disable_turns=v.restrictions.disable_turns.at[target, 0].set(jnp.where(disable, disable_duration, v.restrictions.disable_turns[target, 0])),
            encore_move_id=v.restrictions.encore_move_id.at[target, 0].set(jnp.where(encore, last, v.restrictions.encore_move_id[target, 0])),
            encore_turns=v.restrictions.encore_turns.at[target, 0].set(jnp.where(encore, encore_duration, v.restrictions.encore_turns[target, 0])),
            taunt_turns=v.restrictions.taunt_turns.at[target, 0].set(jnp.where(taunt, taunt_duration, v.restrictions.taunt_turns[target, 0]))))
    state = state.replace(volatile_status=v)
    herb = (disable | encore | taunt) & (state.active_field('item', target)[0] == I.MENTAL_HERB)
    state = consume(state, target, herb)
    v = state.volatile_status
    return state.replace(volatile_status=v.replace(
        restrictions=v.restrictions.replace(
            disabled_move_id=v.restrictions.disabled_move_id.at[target, 0].set(jnp.where(herb, 0, v.restrictions.disabled_move_id[target, 0])),
            disable_turns=v.restrictions.disable_turns.at[target, 0].set(jnp.where(herb, 0, v.restrictions.disable_turns[target, 0])),
            encore_move_id=v.restrictions.encore_move_id.at[target, 0].set(jnp.where(herb, 0, v.restrictions.encore_move_id[target, 0])),
            encore_turns=v.restrictions.encore_turns.at[target, 0].set(jnp.where(herb, 0, v.restrictions.encore_turns[target, 0])),
            taunt_turns=v.restrictions.taunt_turns.at[target, 0].set(jnp.where(herb, 0, v.restrictions.taunt_turns[target, 0])))))


def apply_persistent_restriction(state, attacker, move):
    """Apply Torment to its target or Imprison to its user until departure."""
    v = state.volatile_status
    restrictions = v.restrictions
    target = 1 - attacker
    torment = move.identifier[0] == 259
    success = jnp.where(torment,
        ~restrictions.tormented[target, 0] & (state.ability[target, 0] != A.AROMA_VEIL),
        ~restrictions.imprisoning[attacker, 0])
    restrictions = restrictions.replace(
        tormented=restrictions.tormented.at[target, 0].set(
            restrictions.tormented[target, 0] | (torment & success)),
        imprisoning=restrictions.imprisoning.at[attacker, 0].set(
            restrictions.imprisoning[attacker, 0] | (~torment & success)))
    state = state.replace(volatile_status=v.replace(restrictions=restrictions))
    return mental_herb(state, target, torment & success), success


def apply_hit_restriction(state, target, move, target_ability):
    """Apply the two-turn restriction on a successful secondary without refreshing it."""
    v = state.volatile_status
    restrictions = v.restrictions
    ident = move.identifier[0]
    throat_chop = (ident == 675) & (restrictions.throat_chop_turns[target, 0] == 0)
    heal_block = (ident == 917) & (restrictions.heal_block_turns[target, 0] == 0) & (target_ability != A.AROMA_VEIL)
    herb = heal_block & (state.active_field('item', target)[0] == I.MENTAL_HERB) & ~item_is_suppressed(
        state.ability[target, 0], state.magic_room_duration, v.embargo_turns[target, 0])
    state = consume(state, target, herb)
    restrictions = restrictions.replace(
        throat_chop_turns=restrictions.throat_chop_turns.at[target, 0].set(
            jnp.where(throat_chop, THROAT_CHOP_TURNS, restrictions.throat_chop_turns[target, 0])),
        heal_block_turns=restrictions.heal_block_turns.at[target, 0].set(
            jnp.where(heal_block & ~herb, HEAL_BLOCK_TURNS, restrictions.heal_block_turns[target, 0])))
    return state.replace(volatile_status=v.replace(restrictions=restrictions))


def _check_expired_heal_block(index, state, *, expired, order):
    """Recheck a holder's reactive berry once Heal Block ends."""
    player = order[index]
    return jax.lax.cond(expired[player], after_hp_change, _keep_hp, state, player)


def _keep_hp(state, player):
    """Skip the reactive item check for an unaffected holder."""
    return state


def tick_hit_restrictions(state, order):
    """Expire hit restrictions after residual healing and recheck eligible items."""
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        restrictions=v.restrictions.replace(
            throat_chop_turns=jnp.maximum(v.restrictions.throat_chop_turns - 1, 0),
            heal_block_turns=jnp.maximum(v.restrictions.heal_block_turns - 1, 0))))
    return jax.lax.fori_loop(0, 2,
        partial(_check_expired_heal_block, expired=v.restrictions.heal_block_turns[:, 0] == 1, order=order), state)


def tick(state):
    """Expire restrictions at residual time, including exhausted Encore PP."""
    v = state.volatile_status
    ids = state.active_move_field('identifier')[..., 0]
    pp = state.active_move_field('current_pp')[..., 0]
    valid = jnp.any((ids == v.restrictions.encore_move_id) & (pp > 0), axis=1, keepdims=True)
    disable_turns = jnp.maximum(v.restrictions.disable_turns-1, 0)
    encore_turns = jnp.where(valid, jnp.maximum(v.restrictions.encore_turns-1, 0), 0)
    return state.replace(volatile_status=v.replace(
        restrictions=v.restrictions.replace(
            disabled_move_id=jnp.where(disable_turns > 0, v.restrictions.disabled_move_id, 0),
            disable_turns=disable_turns,
            taunt_turns=jnp.maximum(v.restrictions.taunt_turns-1, 0),
            encore_move_id=jnp.where(encore_turns > 0, v.restrictions.encore_move_id, 0),
            encore_turns=encore_turns),
        queued_move=v.queued_move.replace(
            pending_move=jnp.zeros_like(v.queued_move.pending_move),
            queued_priority_valid=jnp.zeros_like(v.queued_move.queued_priority_valid),
            encore_overridden=jnp.zeros_like(v.queued_move.encore_overridden))))
