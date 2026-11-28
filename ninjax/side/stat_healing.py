"""Special status moves that combine stat changes and HP recovery."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.stat_formulas import STAT_MULTIPLIER_LOOKUP
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import WeatherEnum
from ninjax.enums.pokemon import StatEnum
from ninjax.records.hits import HitOutcome, record_hit
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.queries import effective_weather
from ninjax.side.residual_conditions import root_healing
from ninjax.side.stat_effects import stat_changes


def _identity(state):
    """Leave battle state unchanged for a failed prerequisite."""
    return state


def _fail(state, *, player):
    """Record a failed status action."""
    return record_hit(state, 0, player, HitOutcome.FAILED)


def _belly_drum_apply(state, *, player, cost):
    """Pay the direct HP cost before the twelve-stage boost."""
    state = take_damage_value(state, player, jnp.reshape(cost, (1,)), False)
    return stat_changes(state, player, jnp.array([StatEnum.ATTACK]), jnp.array([12]))


def belly_drum(state, player):
    """Pay half max HP if the user survives and Attack is below its cap."""
    hp = state.active_field('current_hp', player)[0]
    max_hp = state.active_stats(player)[0]
    eligible = (max_hp > 1) & (2 * hp > max_hp) & (state.boosts.normal_boosts[player, 1] < 6)
    return jax.lax.cond(eligible, partial(_belly_drum_apply, player=player, cost=max_hp // 2),
                        partial(_fail, player=player), state)


def _stockpile_apply(state, *, player):
    """Accumulate one layer and remember each successful defensive boost."""
    before = state.boosts.normal_boosts[player, jnp.array([2, 4])]
    state = stat_changes(state, player, jnp.array([2, 4]), jnp.array([1, 1]))
    after = state.boosts.normal_boosts[player, jnp.array([2, 4])]
    # Showdown remembers one removal per changed stat, including Contrary.
    v = state.volatile_status
    return state.replace(volatile_status=v.replace(
        stockpile_count=v.stockpile_count.at[player, 0].add(1),
        stockpile_stages=v.stockpile_stages.at[player].add((before != after).astype(jnp.int32))))


def stockpile(state, player):
    """Add a Stockpile layer up to the three-layer limit."""
    return jax.lax.cond(state.volatile_status.stockpile_count[player, 0] < 3,
                        partial(_stockpile_apply, player=player),
                        partial(_fail, player=player), state)


def consume_stockpile(state, player):
    """Remove accumulated layers and reverse their recorded defensive boosts."""
    v = state.volatile_status
    amounts = -v.stockpile_stages[player]
    state = state.replace(volatile_status=v.replace(
        stockpile_count=v.stockpile_count.at[player, 0].set(0),
        stockpile_stages=v.stockpile_stages.at[player].set(jnp.zeros(2, jnp.int32))))
    return stat_changes(state, player, jnp.array([2, 4]), amounts)


def _swallow_apply(state, *, player, count):
    """Heal from Stockpile layers and remove their defensive effects."""
    max_hp = state.active_stats(player)[0]
    amount = jnp.select((count == 1, count == 2), ((max_hp + 2) // 4, (max_hp + 1) // 2), default=max_hp)
    state = take_damage_value(state, player, -jnp.reshape(amount, (1,)), False)
    return consume_stockpile(state, player)


def swallow(state, player):
    """Consume stored layers and restore HP based on their number."""
    count = state.volatile_status.stockpile_count[player, 0]
    return jax.lax.cond(count > 0, partial(_swallow_apply, player=player, count=count),
                        partial(_fail, player=player), state)


def weather_heal(state, player, ident):
    """Heal according to live weather for Synthesis and Shore Up families."""
    weather = effective_weather(state.weather.weather, state.ability, state.active_field('current_hp'))
    max_hp = state.active_stats(player)[0]
    boosted = jnp.where(ident == 659, weather == WeatherEnum.SANDSTORM, weather == WeatherEnum.SUN)
    reduced = (ident != 659) & (weather != WeatherEnum.NONE) & ~boosted
    # Showdown's fixed-point modifier for 0.667 is 2732/4096.
    amount = jnp.where(boosted, (max_hp * 2732 + 2048) // 4096,
                       jnp.where(reduced, (max_hp + 2) // 4, (max_hp + 1) // 2))
    old = state.active_field('current_hp', player)[0]
    state = take_damage_value(state, player, -jnp.reshape(amount, (1,)), False)
    return jax.lax.cond(state.active_field('current_hp', player)[0] > old,
                        _identity, partial(_fail, player=player), state)


def strength_sap(state, player):
    """Heal the user from the target's staged Attack and attempt its drop."""
    target = 1 - player
    stage = state.boosts.normal_boosts[target, StatEnum.ATTACK]
    amount = jnp.floor(state.active_stats(target)[StatEnum.ATTACK] *
                       STAT_MULTIPLIER_LOOKUP[jnp.clip(stage + 6, 0, 12)]).astype(jnp.int32)
    return jax.lax.cond(stage > -6, partial(_strength_sap_apply, player=player, target=target, amount=amount),
                        partial(_fail, player=target), state)


def _strength_sap_apply(state, *, player, target, amount):
    """Apply the opposing Attack drop and the user's recovery."""
    before = state.boosts.normal_boosts[target, StatEnum.ATTACK]
    state = stat_changes(state, target, jnp.array([StatEnum.ATTACK]), jnp.array([-1]), True)
    old_hp = state.active_field('current_hp', player)[0]
    amount = root_healing(state, player, amount)
    ooze = state.ability[target, 0] == AbilityEnum.LIQUID_OOZE
    damage = jnp.where(ooze, jnp.where(state.ability[player, 0] == AbilityEnum.MAGIC_GUARD, 0, amount), -amount)
    state = take_damage_value(state, player, jnp.reshape(damage, (1,)), False)
    success = (state.boosts.normal_boosts[target, StatEnum.ATTACK] != before) | (
        state.active_field('current_hp', player)[0] > old_hp)
    return jax.lax.cond(success, _identity, partial(_fail, player=target), state)
