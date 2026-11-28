"""Finalization helpers for immutable JAX battle state."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.logic import quad_and
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import MoveFlags, MoveHpCost
from ninjax.enums.pokemon import StatEnum as S, StatEnum
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome
from ninjax.side.faint_effects import direct_faint
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import consume, update_items
from ninjax.side.stat_effects import stat_changes
from ninjax.side.stat_healing import consume_stockpile
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts, conditional_add_boosts


def _apply_self_stat_changes(s, *, attacker_idx, move):
    """Apply the self-stage vector gated by the once-per-move roll."""
    return stat_changes(s, attacker_idx, move.self_stat, move.self_stages)


def _apply_self_ko(s, *, attacker_idx):
    """Remove the remaining user HP for a self-KO move."""
    return direct_faint(s, attacker_idx, attacker_idx, Cause.SELF_KO)


def _apply_recoil_damage(state, *, player, damage):
    """Commit recoil with the user's attribution and restore the action context."""
    source, cause = state.event_context.source, state.event_context.cause
    state = state.replace(
        event_context=state.event_context.replace(
            source=jnp.asarray(player, jnp.int32),
            cause=jnp.int32(Cause.RECOIL)))
    state = take_damage_value(state, player, damage, False, source=player)
    return state.replace(
        event_context=state.event_context.replace(
            source=source,
            cause=cause))


def _apply_move_recoil(state, player, move, hp_lost):
    """Apply damage recoil or a fixed HP cost once after all damaging hits."""
    ability = state.ability[player, 0]
    cost = move.hp_cost[0] != MoveHpCost.NONE
    # Showdown treats Chloroblast as ordinary recoil and Steel Beam as a
    # named damage condition. Rock Head only prevents the former.
    rock_head_bypassed = move.hp_cost[0] == MoveHpCost.ON_DAMAGE_OR_FAILURE
    enabled = ((move.recoil[0] | cost) & (hp_lost > 0) &
               ((ability != AbilityEnum.ROCK_HEAD) | rock_head_bypassed) &
               (ability != AbilityEnum.MAGIC_GUARD) &
               (state.active_field('current_hp', player)[0] > 0))
    # Ordinary recoil uses total damage across all recipients. Fixed costs use
    # max HP and round half up independently of how much damage the move dealt.
    damage_recoil = jnp.floor(hp_lost * move.recoil_percent[0] + .5).astype(jnp.int32)
    amount = jnp.maximum(1, jnp.where(cost, (state.active_stats(player)[0] + 1) // 2, damage_recoil))
    return jax.lax.cond(enabled, partial(_apply_recoil_damage, player=player, damage=amount),
                        _identity, state)


def finish_damaging_move(state: BattleState, attacker_idx, move: Move, hp_lost, defender: Pokemon, self_effect=True) -> BattleState:
    """Run once-per-move effects before the action faint checkpoint.

    ``hp_lost`` is the total HP removed from Pokémon and Substitutes.
    ``defender`` is the snapshot from before the first hit. ``self_effect`` is
    the caller's once-per-move chance outcome, shared by all self stat changes.
    """

    state = _apply_move_recoil(state, attacker_idx, move, hp_lost)

    # Recharge moves require a successful hit, including a hit
    # on Substitute or a knockout. Misses and blocked attempts never reach here.
    v = state.volatile_status
    recharge = ((move.move_flags[0] & MoveFlags.RECHARGE) != 0) & (state.active.current_hp[attacker_idx, 0] > 0)
    state = state.replace(volatile_status=v.replace(continuation=v.continuation.replace(recharging=v.continuation.recharging.at[attacker_idx, 0].set(
        v.continuation.recharging[attacker_idx, 0] | recharge))))
    state = emit(state, EventKind.CONDITION, attacker_idx, detail=15, before=0, after=1, enabled=recharge)
    state = jax.lax.cond(
        (state.active.current_hp[attacker_idx, 0] > 0) & self_effect,
        partial(_apply_self_stat_changes, attacker_idx=attacker_idx, move=move),
        _identity,
        state,
    )
    state = after_move(state, attacker_idx, move, hp_lost > 0)
    state = after_move_finished(state, defender, 1-attacker_idx, defender.current_hp)
    state = jax.lax.cond(
        move.self_ko[0],
        partial(_apply_self_ko, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    return state


def finish_attempt(state, player, move):
    """Resolve effects that depend on the recorded outcome of a move attempt."""
    outcome = state.hit_results[0].outcome
    failed = (outcome == HitOutcome.MISS) | (outcome == HitOutcome.BLOCKED) | (outcome == HitOutcome.IMMUNE) | (outcome == HitOutcome.PROTECTED)
    # Interrupted or rejected selections never reach this hook with a failed
    # hit result. Steel Beam rounds up here; crash damage retains its floor.
    fixed_cost = move.hp_cost[0] == MoveHpCost.ON_DAMAGE_OR_FAILURE
    cost = (move.implemented[0] & (move.crashes[0] | fixed_cost) & failed &
            (state.ability[player, 0] != AbilityEnum.MAGIC_GUARD) &
            (state.active_field('current_hp', player)[0] > 0))
    amount = jnp.maximum(1, (state.active_stats(player)[0] + fixed_cost.astype(jnp.int32)) // 2)
    # Spit Up consumes the stockpile even on a miss or blocked hit. Actions
    # interrupted before execution never enter this finalization path.
    spent = (move.identifier[0] == 255) & (state.volatile_status.stockpile_count[player, 0] > 0)
    state = jax.lax.cond(spent, partial(consume_stockpile, player=player), _identity, state)

    return update_items(
        jax.lax.cond(
            cost,
            partial(_apply_recoil_damage, player=player, damage=amount),
            _identity,
            state,
        ),
    )


def _struggle_recoil(state, *, player):
    """Apply quarter-max-HP Struggle recoil and record any resulting faint."""
    max_hp = state.active_stats(player)[0]
    damage = jnp.maximum(1, (max_hp + 2) // 4)
    return _apply_recoil_damage(state, player=player, damage=damage)


def _apply_throat_spray(s, *, player):
    """Apply throat spray."""
    return add_boosts(s, player, S.SPECIAL_ATTACK, 1)


def after_move(state, player, move, dealt_damage):
    """Apply Life Orb recoil before the surviving user's Throat Spray effect."""
    mon = state.active[player]
    item = state.effective_item(player)[0]
    orb = ((mon.current_hp[0] > 0) & dealt_damage & (item == I.LIFE_ORB) &
           (state.ability[player, 0] != AbilityEnum.MAGIC_GUARD))
    damage = jnp.maximum(1, mon.max_hp // 10)
    state = jax.lax.cond(orb, partial(_apply_recoil_damage, player=player, damage=damage),
                         _identity, state)
    alive = state.active_field('current_hp', player)[0] > 0
    spray = alive & move.sound[0] & (item == I.THROAT_SPRAY)
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
