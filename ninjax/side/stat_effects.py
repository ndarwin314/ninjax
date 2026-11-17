"""Opposing stat reductions and their represented ability reactions."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.enum_types import AbilityEnum as A, AbilityEnum, StatEnum
from ninjax.side.state import BattleState
from ninjax.side.updates import add_boosts, set_boosts
from ninjax.stats import StatBoosts


def _apply_opposing_drop_boost(s, *, player, stat, lowered_count):
    """Apply opposing drop boost."""
    return add_boosts(s, player, stat, 2*lowered_count)


def _identity(s, *unused):
    """Keep state unchanged while accepting either conditional branch's operands."""
    return s


def opposing_drop_response(state, player, lowered_count):
    """React to actual opposing stage reductions, including accuracy/evasion."""

    ability = state.ability[player, 0]
    stat = jnp.where(ability == AbilityEnum.DEFIANT, StatEnum.ATTACK, StatEnum.SPECIAL_ATTACK)
    eligible = ((ability == AbilityEnum.DEFIANT) | (ability == AbilityEnum.COMPETITIVE)) & (lowered_count > 0)
    return jax.lax.cond(eligible, partial(_apply_opposing_drop_boost, player=player, stat=stat, lowered_count=lowered_count), _identity, state)


def reduce_boosts(state: BattleState, side_idx, idx, val, bounceable=False) -> BattleState:
    """Apply an opposing normal-stat drop and react only to actual reductions.

    Self-inflicted drops use ``add_boosts`` instead. Mirror Armor reflection
    and arbitrary source-sensitive ability interactions remain unrepresented.
    """
    ability = state.ability
    not_clear_body = (ability[side_idx]!=AbilityEnum.CLEAR_BODY)[0]
    not_clear_body = not_clear_body & ~((ability[side_idx, 0] == AbilityEnum.BIG_PECKS) & (jnp.asarray(idx) == StatEnum.DEFENSE))
    old = state.boosts.normal_boosts[side_idx]
    state = add_boosts(state, side_idx, idx, -val * not_clear_body)

    return opposing_drop_response(state, side_idx, jnp.sum(state.boosts.normal_boosts[side_idx] < old))


def conditional_reduce_boosts(state: BattleState, side_idx, cond, idx, val, bounceable) -> BattleState:
    """Reduce stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), reduce_boosts,
                        _identity,
                        state, side_idx, idx, val, bounceable)


def _unchanged_stages(state, player, stat, amount, opposing):
    """Keep state unchanged for an unused stage-vector entry."""
    return state


def _reduce_stage(state, player, stat, amount, opposing):
    """Invoke opposing-drop hooks for a normal stat."""
    return reduce_boosts(state, player, stat, -amount)


def _add_stage(state, player, stat, amount, opposing):
    """Apply a normal stat increase or self-inflicted decrease."""
    return add_boosts(state, player, stat, amount)


def _ordinary_stage(state, player, stat, amount, opposing):
    """Route normal stages through the existing ability hooks."""
    return jax.lax.cond((amount < 0) & opposing, _reduce_stage, _add_stage,
                        state, player, stat, amount, opposing)


def _accuracy_stage(state, player, stat, amount, opposing):
    """Apply accuracy/evasion stages in their separate two-column array."""


    delta = amount * jnp.where(state.ability[player, 0] == A.CONTRARY, -1, 1)
    immune = (state.ability[player, 0] == A.CLEAR_BODY) | ((state.ability[player, 0] == A.KEEN_EYE) & (stat == 6))
    delta = jnp.where(opposing & (amount < 0) & immune, 0, delta)
    old = state.boosts.acc_boosts[player]
    state = set_boosts(state, player, StatBoosts(normal_boosts=state.boosts.normal_boosts[player],
        acc_boosts=state.boosts.acc_boosts[player].at[stat-6].add(delta)))

    lowered = opposing & jnp.any(state.boosts.acc_boosts[player] < old)
    return opposing_drop_response(state, player, lowered.astype(jnp.int32))


def _valid_stage(state, player, stat, amount, opposing):
    """Select the normal-stat or accuracy/evasion handler."""
    return jax.lax.cond(stat <= 5, _ordinary_stage, _accuracy_stage,
                        state, player, stat, amount, opposing)


def stages(state, player, stat, amount, opposing=False):
    """Apply one stat change, invoking opposing-drop abilities only on drops."""
    return jax.lax.cond((amount != 0) & (stat >= 1) & (stat <= 7),
                        _valid_stage, _unchanged_stages, state, player, stat, amount, opposing)


def stat_changes(state, player, indices, amounts, opposing=False):
    """Apply a padded stat-change vector sequentially in a fixed JAX scan."""
    return jax.lax.scan(_apply_stat_change, (state, player, opposing), (indices, amounts))[0][0]


def _apply_stat_change(carry, change):
    """Consume one index/stage pair while retaining the player and source."""
    state, player, opposing = carry
    return (stages(state, player, change[0], change[1], opposing), player, opposing), None
