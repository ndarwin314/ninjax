"""Finalization helpers for immutable JAX battle state."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.enum_types import AbilityEnum, ItemEnum as I, StatEnum as S, StatEnum
from ninjax.events import Cause, EventKind, emit
from ninjax.hit import HitOutcome
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.side.hp_effects import take_damage_percent, take_damage_value
from ninjax.side.item_effects import consume, update_items
from ninjax.side.stat_effects import stat_changes
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts, conditional_add_boosts, update_active_fields
from ninjax.utils import quad_and


def _apply_self_stat_changes(s, *, attacker_idx, move):
    """Apply the self-stage vector gated by the once-per-move roll."""
    return stat_changes(s, attacker_idx, move.self_stat, move.self_stages)


def _apply_self_ko(s, *, attacker_idx):
    """Remove the remaining user HP for a self-KO move."""
    return take_damage_value(s, attacker_idx, s.active[attacker_idx].current_hp, False)


def finish_damaging_move(state: BattleState, attacker_idx, move: Move, damage, defender: Pokemon, self_effect=True) -> BattleState:
    """Run existing once-per-move effects and resolve simultaneous knockouts.

    ``damage`` is the sum of calculated damage, not HP actually lost. Effects
    requiring actual HP lost need a separate total before they can be added.
    ``defender`` is the snapshot from before the first hit. ``self_effect`` is
    the caller's once-per-move chance outcome, shared by all self stat changes.
    """

    # Hyper Beam requires recharge after a successful hit, including a hit
    # on Substitute or a knockout. Misses and blocked attempts never reach here.
    v = state.volatile_status
    recharge = (move.identifier[0] == 63) & (state.active.current_hp[attacker_idx, 0] > 0)
    state = state.replace(volatile_status=v.replace(recharging=v.recharging.at[attacker_idx, 0].set(
        v.recharging[attacker_idx, 0] | recharge)))
    state = emit(state, EventKind.CONDITION, attacker_idx, detail=15, before=0, after=1, enabled=recharge)
    state = jax.lax.cond(
        (state.active.current_hp[attacker_idx, 0] > 0) & self_effect,
        partial(_apply_self_stat_changes, attacker_idx=attacker_idx, move=move),
        _identity,
        state,
    )
    state = after_move(state, attacker_idx, move, damage[0] > 0)
    state = after_move_finished(state, defender, 1-attacker_idx, defender.current_hp)
    state = jax.lax.cond(
        move.self_ko[0],
        partial(_apply_self_ko, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    # Retain the cause-dependent winner for a simultaneous final knockout;
    # the environment uses this value when it constructs the terminal reward.
    both_out = ~jnp.any((state.team.current_hp[..., 0] > 0) & state.roster)
    winner = jnp.where(move.self_ko[0] | (move.identifier[0] == 153), 1-attacker_idx, attacker_idx)
    state = state.replace(winner=jnp.where(both_out, winner, state.winner))

    return state


def _apply_crash_damage(s, *, player):
    """Apply half-max-HP recoil after a failed crash-damage move."""
    return take_damage_percent(s, player, .5, cause=Cause.RECOIL)


def finish_attempt(state, player, move):
    """Resolve effects that depend on the recorded outcome of a move attempt."""
    outcome = state.hit_results[0].outcome
    failed = (outcome == HitOutcome.MISS) | (outcome == HitOutcome.BLOCKED) | (outcome == HitOutcome.IMMUNE) | (outcome == HitOutcome.PROTECTED)
    crash = (move.identifier[0] == 136) & failed & (state.ability[player, 0] != AbilityEnum.MAGIC_GUARD) & (state.active.current_hp[player, 0] > 0)

    return update_items(
        jax.lax.cond(
            crash,
            partial(_apply_crash_damage, player=player),
            _identity,
            state,
        ),
    )


def _struggle_recoil(b, player):
    """Apply Struggle's existing quarter-max-HP recoil after shared damage."""
    actor = b.active[player]
    hp = jnp.maximum(0, actor.current_hp - jnp.maximum(1, jnp.round(actor.max_hp / 4)))
    b = b.replace(event_cause=jnp.int32(Cause.RECOIL))
    b = emit(b, EventKind.HP, player, before=actor.current_hp, after=hp,
             hp_change=hp-actor.current_hp,
             hp_fraction_change=(jnp.ceil(hp / actor.max_hp * 100) -
                                 jnp.ceil(actor.current_hp / actor.max_hp * 100)) / 100,
             enabled=hp != actor.current_hp)
    b = emit(b, EventKind.FAINT, player, enabled=(actor.current_hp > 0) & (hp == 0))
    b = update_active_fields(b, player, current_hp=hp, is_alive=hp > 0)
    b = b.replace(winner=jnp.where(~jnp.any((b.team.current_hp[..., 0] > 0) & b.roster), player, b.winner))
    return b


def _apply_throat_spray(s, *, player):
    """Apply throat spray."""
    return add_boosts(s, player, S.SPECIAL_ATTACK, 1)


def after_move(state, player, move, dealt_damage):
    """Apply Life Orb recoil before the surviving user's Throat Spray effect."""
    mon = state.active[player]
    alive = mon.current_hp[0] > 0
    state = take_damage_percent(state, player, jnp.where(alive & dealt_damage & (mon.item[0] == I.LIFE_ORB), .1, 0))
    spray = alive & move.sound[0] & (state.active.item[player, 0] == I.THROAT_SPRAY)
    state = consume(state, player, spray)
    return jax.lax.cond(spray, partial(_apply_throat_spray, player=player), _identity, state)


def after_move_finished(state: BattleState, defender: Pokemon, defender_idx, hp_start):
    """Apply the existing Berserk check to the saved defender snapshot."""
    # idk if stamina should actually be here or somewhere else, idk

    # berserk
    is_berserk = quad_and(
        defender.hp_greater_than(hp_start),
        defender.hp_less_than(0.5),
        defender.ability == AbilityEnum.BERSERK,
        defender.is_alive
    )
    state = conditional_add_boosts(state, defender_idx, is_berserk, StatEnum.SPECIAL_ATTACK, 1)
    return state


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x
