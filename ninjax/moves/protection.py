"""Activation, hit eligibility and removal for turn-scoped guards."""
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.moves import MoveFlags, MoveType, ProtectionKind as Guard


def protection_kind(identifier):
    """Map catalogue move IDs to the protection they install."""
    return jnp.select(
        ((identifier == 182) | (identifier == 197), identifier == 596,
         identifier == 661, identifier == 908, identifier == 852,
         identifier == 588, identifier == 561, identifier == 203,
         identifier == 501, identifier == 469),
        (Guard.PROTECT, Guard.SPIKY_SHIELD, Guard.BANEFUL_BUNKER, Guard.BURNING_BULWARK,
         Guard.SILK_TRAP, Guard.KINGS_SHIELD, Guard.MAT_BLOCK, Guard.ENDURE,
         Guard.QUICK_GUARD, Guard.WIDE_GUARD), default=Guard.NONE)


def apply_protection(state, player, move, roll):
    """Raise a guard and refresh shared stall odds on a successful use."""
    kind = protection_kind(move.identifier[0])
    v = state.volatile_status
    guard = v.protection
    quick, wide, mat = kind == Guard.QUICK_GUARD, kind == Guard.WIDE_GUARD, kind == Guard.MAT_BLOCK
    personal = (kind != Guard.NONE) & ~(quick | wide | mat)
    chain = guard.protect_chain[player, 0]
    # The reference stall denominator caps at 729 (3**6). Quick and Wide
    # Guard refresh stall without an odds check. Mat Block has neither.
    odds_pass = roll < jnp.power(1 / 3., jnp.minimum(chain, 6))
    can_act = ~v.queued_move.last_action[player, 0]
    mat_eligible = v.move_history.active_move_actions[player, 0] <= 1
    success = can_act & ((personal & odds_pass) | quick | wide | (mat & mat_eligible))
    side = state.side_protection
    side = side.replace(
        quick_guard=side.quick_guard.at[player, 0].set(side.quick_guard[player, 0] | (quick & success)),
        wide_guard=side.wide_guard.at[player, 0].set(side.wide_guard[player, 0] | (wide & success)),
        mat_block=side.mat_block.at[player, 0].set(side.mat_block[player, 0] | (mat & success)))
    refresh = success & ~mat
    stall_failed = personal & can_act & ~odds_pass
    guard = guard.replace(
        protection_kind=guard.protection_kind.at[player, 0].set(
            jnp.where(personal & success, kind, guard.protection_kind[player, 0])),
        protected=guard.protected.at[player, 0].set(
            guard.protected[player, 0] | (personal & success & (kind != Guard.ENDURE))),
        protect_chain=guard.protect_chain.at[player, 0].set(
            jnp.where(refresh, jnp.minimum(chain + 1, 6), jnp.where(stall_failed, 0, chain))),
        stall_refreshed=guard.stall_refreshed.at[player, 0].set(
            (guard.stall_refreshed[player, 0] | refresh) & ~stall_failed))
    state = state.replace(side_protection=side, volatile_status=v.replace(protection=guard))
    return state, (kind != Guard.NONE) & ~success


def breaks_protection(move):
    """Check moves that remove shields after reaching the protected side."""
    ident = move.identifier[0]
    return (ident == 364) | (ident == 467) | (ident == 566) | (ident == 593) | (ident == 621)


def blocked_by_protection(state, move, attacker):
    """Share a guard decision between accuracy and hit routing."""
    defender = 1 - attacker
    v = state.volatile_status
    kind = v.protection.protection_kind[defender, 0]
    status = move.move_type[0] == MoveType.STATUS
    damage_only = ((kind == Guard.SILK_TRAP) | (kind == Guard.BURNING_BULWARK) |
                   (kind == Guard.KINGS_SHIELD) | (kind == Guard.MAT_BLOCK))
    personal = v.protection.protected[defender, 0] & (kind != Guard.ENDURE) & ~(status & damage_only)
    priority = jnp.where(v.queued_move.encore_overridden[attacker, 0],
                         v.queued_move.queued_priority[attacker, 0], move.priority[0])
    sides = state.side_protection
    side = ((sides.quick_guard[defender, 0] & (priority > 0)) |
            (sides.wide_guard[defender, 0] & ((move.move_flags[0] & MoveFlags.SPREAD) != 0)) |
            (sides.mat_block[defender, 0] & ~status))
    unseen_fist = (state.ability[attacker, 0] == A.UNSEEN_FIST) & move.contact[0]
    return (personal | side) & ~move.target_self[0] & move.protectable[0] & ~unseen_fist & ~breaks_protection(move)


def remove_protection(state, player, enabled):
    """Break shields and their stall chain, preserving Endure."""
    v = state.volatile_status
    guard, sides = v.protection, state.side_protection
    personal = enabled & guard.protected[player, 0] & (guard.protection_kind[player, 0] != Guard.ENDURE)
    broken = personal | (enabled & (sides.quick_guard[player, 0] |
                         sides.wide_guard[player, 0] | sides.mat_block[player, 0]))
    return state.replace(
        side_protection=sides.replace(
            quick_guard=sides.quick_guard.at[player, 0].set(sides.quick_guard[player, 0] & ~enabled),
            wide_guard=sides.wide_guard.at[player, 0].set(sides.wide_guard[player, 0] & ~enabled),
            mat_block=sides.mat_block.at[player, 0].set(sides.mat_block[player, 0] & ~enabled)),
        volatile_status=v.replace(protection=guard.replace(
            protected=guard.protected.at[player, 0].set(guard.protected[player, 0] & ~personal),
            protection_kind=guard.protection_kind.at[player, 0].set(
                jnp.where(personal, Guard.NONE, guard.protection_kind[player, 0])),
            protect_chain=guard.protect_chain.at[player, 0].set(
                jnp.where(broken, 0, guard.protect_chain[player, 0])),
            stall_refreshed=guard.stall_refreshed.at[player, 0].set(
                guard.stall_refreshed[player, 0] & ~broken))))
