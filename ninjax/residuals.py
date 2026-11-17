"""End-of-turn effects called in the order scheduled by battle.step_field.

Functions retain current player ordering, event attribution and PRNG policy.
Residual ordering coverage remains partial; extraction does not change rules."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random
from ninjax.enum_types import AbilityEnum, ItemEnum as I, StatEnum, Status, TerrainEnum, WeatherEnum
from ninjax.side.state import BattleState
from ninjax.events import Cause
from ninjax.side.updates import add_boosts, conditional_add_boosts
from ninjax.side.item_effects import cure_status
from ninjax.side.queries import effective_weather, is_floating, is_sand_immune
from ninjax.side.status_effects import set_status
from ninjax.side.hp_effects import take_damage_percent
from ninjax.restrictions import tick

__all__ = (
    'end_turn_damage',
    'step_side_conditions',
    'step_moody',
    'bad_dreams',
    'residual_abilities',
    'leftovers',
    'residual_orbs',
)


def _apply_end_turn_effect(s, effect, *, idx):
    """Apply one weather or terrain HP fraction to both players in order."""
    percent, cause = effect
    return take_damage_percent(s, idx, percent, cause=cause), None


def end_turn_damage(state: BattleState) -> BattleState:
    """Apply the represented weather, terrain, ability, and status HP effects.

    Both sides are updated through ``take_damage_percent``. The exact ordering,
    including speed-dependent ordering and simultaneous knockouts, is incomplete.
    """
    # TODO: the order of all these updates is probably incorrect
    # i think it should be like sand, sand, grass, grass
    # whereas this is currently sand, grass, sand grass,
    # and order should also depend on speed
    # that being said idk if that is high priority
    # find something for gen 9 https://www.smogon.com/forums/threads/sword-shield-battle-mechanics-research.3655528/page-64#post-9244179
    # also another wrinkle is that effects take effect based on speed order
    # i.e it would go sand fast, sand slow, grass fast, grass slow
    # this matters a lot in vgc but usually less in singles
    # it also matters for determining winner if the last pokemon for both players faints on the same turn

    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    active = state.effective_active
    idx = jnp.array([0,1])
    floating = is_floating(active)
    sand_immune = is_sand_immune(active)

    # sand damage
    sand_damage = (1 - sand_immune) / 16 * (weather == WeatherEnum.SANDSTORM)

    # rain abilities
    rain_healing = ((active.ability==AbilityEnum.RAIN_DISH) + 2 * (active.ability==AbilityEnum.DRY_SKIN)) / 16 * (weather == WeatherEnum.RAIN)
    # why is this one an int but the others arent

    # sun abilities
    sun_damage = ((active.ability==AbilityEnum.DRY_SKIN) + (active.ability==AbilityEnum.SOLAR_POWER)) / 8 * (weather == WeatherEnum.SUN)

    # snow abilities
    snow_healing = (active.ability==AbilityEnum.ICE_BODY) / 16 * (weather == WeatherEnum.SNOW)


    # grassy terrain healing
    grass_healing = (floating - 1) / 16 * (state.terrain.terrain == TerrainEnum.GRASSY)

    # status damage
    # technically this should be factored out to multiple bits since it goes burn poison toxic in priority
    poison_heal = (active.ability == AbilityEnum.POISON_HEAL) & active.is_poisoned
    status_damage = (1 / 8 * (active.status==Status.POISON) +
                     1 / 16 * (active.status==Status.BURN) * jnp.where(active.ability == AbilityEnum.HEATPROOF, .5, 1.) +
                     state.toxic_counter / 16 * (active.status==Status.TOXIC))
    status_damage = jnp.where(poison_heal, -1/8, status_damage)
    # Preserve effect-major, then player-major ordering while compiling one
    # shared HP-update body. All fractions use the original active snapshot.
    fractions = jnp.stack((sand_damage, -rain_healing, sun_damage,
                           -snow_healing, grass_healing))
    causes = jnp.array([Cause.WEATHER] * 4 + [Cause.TERRAIN], jnp.int32)
    state = jax.lax.scan(partial(_apply_end_turn_effect, idx=idx), state, (fractions, causes))[0]

    state = bad_dreams(state)
    return take_damage_percent(state, idx, status_damage, cause=Cause.STATUS)


def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    """Decrement side timers and update toxic counters; return the unchanged key."""
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state = state.replace(
        reflect=jnp.maximum(state.reflect - 1, 0),
        light_screen=jnp.maximum(state.light_screen - 1, 0),
        aurora_veil=jnp.maximum(state.aurora_veil - 1, 0),
        tailwind=jnp.maximum(state.tailwind - 1, 0),
        toxic_counter=toxic_counter
    )
    state = state.replace(volatile_status=state.volatile_status.replace(
        protected=jnp.zeros_like(state.volatile_status.protected),
        protection_kind=jnp.zeros_like(state.volatile_status.protection_kind),
        flinched=jnp.zeros_like(state.volatile_status.flinched)))

    state = tick(state)
    return key, state


def step_moody(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply represented Moody stage changes, splitting one key per side.

    The fixed two-iteration loop is unrolled when tracing a singles battle.
    """
    for i in range(2):
        key, plus_key, minus_key = random.split(key, 3)
        stages = state.boosts.normal_boosts[i, 1:]
        can_raise = stages < 6
        plus = random.categorical(plus_key, jnp.where(can_raise, 0., -jnp.inf))
        can_lower = (stages > -6) & ((jnp.arange(5) != plus) | ~jnp.any(can_raise))
        minus = random.categorical(minus_key, jnp.where(can_lower, 0., -jnp.inf))
        active = (state.ability[i, 0] == AbilityEnum.MOODY) & (state.active.current_hp[i, 0] > 0)
        state = conditional_add_boosts(state, i, active & jnp.any(can_raise), 1+plus, 2)
        state = conditional_add_boosts(state, i, active & jnp.any(can_lower), 1+minus, -1)
    return key, state


def _apply_orb_status(k, s, *, player, status):
    """Apply orb status."""
    return set_status(k, s, player, jnp.asarray(status).reshape((1,)))


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)


def _residual_orbs_apply(player, carry):
    """Use existing status immunity and explicit-key helpers for each holder."""
    key, s = carry
    item = s.active.item[player, 0]
    status = jnp.where(item == I.FLAME_ORB, Status.BURN, jnp.where(item == I.TOXIC_ORB, Status.TOXIC, Status.NONE))
    return jax.lax.cond((status != Status.NONE) & (s.active.current_hp[player, 0] > 0),
        partial(_apply_orb_status, player=player, status=status),
        _unchanged_key_state, key, s)


def _leftovers_heal(i, s):
    """Apply one player's residual item healing."""
    mon = s.active[i]
    return take_damage_percent(s, i, jnp.where((mon.item[0] == I.LEFTOVERS) & (mon.current_hp[0] > 0), -1/16, 0))


def residual_orbs(key, state):
    """Inflict orb statuses after this turn's status damage; orbs are not consumed."""

    apply = _residual_orbs_apply
    return jax.lax.fori_loop(0, 2, apply, (key, state))


def leftovers(state):
    """Heal living Leftovers holders in player order."""
    heal = _leftovers_heal
    return jax.lax.fori_loop(0, 2, heal, state)


def _bad_dreams_damage(player, s):
    """Skip fainted sources, and attribute indirect damage to the opponent."""
    other = 1-player
    sleeping = (s.active.status[player, 0] == Status.SLEEP) | (s.ability[player, 0] == AbilityEnum.COMATOSE)
    eligible = sleeping & (s.ability[other, 0] == AbilityEnum.BAD_DREAMS) & (s.active.current_hp[other, 0] > 0)
    eligible &= s.ability[player, 0] != AbilityEnum.MAGIC_GUARD
    source = s.event_source
    s = s.replace(event_source=jnp.asarray(other, jnp.int32))
    s = take_damage_percent(s, player, jnp.where(eligible, 1/8, 0), cause=Cause.INDIRECT)
    return s.replace(event_source=source)


def _residual_abilities_roll(k):
    """Use the pinned Scarlet/Violet 33-in-100 Shed Skin probability."""
    k, subkey = jax.random.split(k)
    return k, jax.random.randint(subkey, (), 0, 100) < 33


def _skip_status_cure_roll(k):
    """Skip status cure roll."""
    return (k, jnp.bool_(False))


def _cure_residual_status(s, *, player):
    """Cure residual status."""
    return cure_status(s, player)


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def _raise_residual_speed(s, *, player):
    """Raise residual speed."""
    return add_boosts(s, player, StatEnum.SPEED, 1)


def _residual_abilities_apply(player, carry, *, status_cures, weather, speed_boost):
    """Only roll Shed Skin for a living, statused holder."""
    key, s = carry
    mon = s.active[player]
    ability = s.ability[player, 0]
    living = mon.current_hp[0] > 0
    roll = _residual_abilities_roll
    key, shed = jax.lax.cond(jnp.asarray(status_cures) & living & (mon.status[0] != Status.NONE) & (ability == AbilityEnum.SHED_SKIN),
        roll, _skip_status_cure_roll, key)
    cure = jnp.asarray(status_cures) & living & (mon.status[0] != Status.NONE) & (shed | ((ability == AbilityEnum.HYDRATION) & (weather == WeatherEnum.RAIN)))
    s = jax.lax.cond(cure, partial(_cure_residual_status, player=player), _identity, s)
    speed = jnp.asarray(speed_boost) & living & (ability == AbilityEnum.SPEED_BOOST) & (s.volatile_status.entry_turn[player, 0] < s.turn_number)
    return key, jax.lax.cond(speed, partial(_raise_residual_speed, player=player), _identity, s)


def bad_dreams(state):
    """Damage sleeping opposing singles Pokémon at residual order eight."""

    damage = _bad_dreams_damage
    return jax.lax.fori_loop(0, 2, damage, state)


def residual_abilities(key, state, status_cures=True, speed_boost=True):
    """Resolve represented status cures and Speed Boost with explicit randomness.

    The scheduler runs status cures before poison/burn damage, then Speed Boost
    late in the turn. It does not activate on a normal switch-in's first turn.
    """


    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    apply = partial(_residual_abilities_apply, status_cures=status_cures, weather=weather, speed_boost=speed_boost)
    return jax.lax.fori_loop(0, 2, apply, (key, state))
