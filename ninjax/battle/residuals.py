"""Schedule represented end-of-turn effects using live battle state."""
from enum import IntEnum
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import TerrainEnum, WeatherEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import SemiInvulnerability
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.records.events import Cause
from ninjax.restrictions import tick
from ninjax.side.faint_effects import resolve_faints
from ninjax.side.hp_effects import take_damage_percent, take_damage_value
from ninjax.side.item_effects import cure_status
from ninjax.side.queries import effective_weather, is_sand_immune
from ninjax.side.residual_conditions import root_healing
from ninjax.side.state import BattleState, SideProtection
from ninjax.side.status_effects import set_status
from ninjax.side.updates import add_boosts, conditional_add_boosts


__all__ = (
    'end_turn_damage',
    'step_side_conditions',
    'step_moody',
    'bad_dreams',
    'residual_abilities',
    'leftovers',
    'residual_orbs',
)


class ResidualHPPhase(IntEnum):
    """HP effects that share the ordered per-holder update loop."""

    WEATHER = 0
    GRASSY_TERRAIN = 1
    POISON = 2
    BURN = 3
    BAD_DREAMS = 4
    BINDING = 5
    WISH = 6
    AQUA_RING = 7
    INGRAIN = 8
    LEECH_SEED = 9
    CURSE = 10
    SALT_CURE = 11
    BINDING_AND_SALT = 12


def residual_speed_order(state):
    """Return fastest first, using side order for equal Speed."""
    return jnp.argsort(-state.effective_speeds, stable=True)


def _weather_fraction(state, mon):
    """Read the effective weather and the holder's current weather reactions."""
    weather = effective_weather(state.weather.weather, state.ability, state.active_field('current_hp'))
    ability = mon.ability[0]
    return jnp.select(
        (weather == WeatherEnum.SANDSTORM, weather == WeatherEnum.RAIN,
         weather == WeatherEnum.SUN, weather == WeatherEnum.SNOW),
        ((~is_sand_immune(mon)[0]) / 16,
         -((ability == AbilityEnum.RAIN_DISH) + 2 * (ability == AbilityEnum.DRY_SKIN)) / 16,
         ((ability == AbilityEnum.DRY_SKIN) | (ability == AbilityEnum.SOLAR_POWER)) / 8,
         -(ability == AbilityEnum.ICE_BODY).astype(jnp.int32) / 16),
        default=0.)


def _residual_fraction(state, player, phase):
    """Resolve one holder's eligibility and HP fraction at this phase."""
    mon = state.combat_active[player]
    ability, status = mon.ability[0], mon.status[0]
    weather = _weather_fraction(state, mon)
    # Dig and Dive shelter the user from sandstorm, but not other weather effects.
    hidden = state.volatile_status.continuation.semi_invulnerability[player, 0]
    sand = effective_weather(state.weather.weather, state.ability, state.active_field('current_hp')) == WeatherEnum.SANDSTORM
    weather = jnp.where(sand & ((hidden == SemiInvulnerability.UNDERGROUND) | (hidden == SemiInvulnerability.UNDERWATER)), 0., weather)
    grassy = (state.terrain.terrain == TerrainEnum.GRASSY) & state.is_grounded(player)[0]
    poison = jnp.where(status == Status.POISON, 1 / 8,
                        jnp.where(status == Status.TOXIC, state.toxic_counter[player, 0] / 16, 0.))
    poison = jnp.where((ability == AbilityEnum.POISON_HEAL) & (poison > 0), -1 / 8, poison)
    burn = (status == Status.BURN) * jnp.where(ability == AbilityEnum.HEATPROOF, 1 / 32, 1 / 16)
    source = 1 - player
    dreams = ((state.ability[source, 0] == AbilityEnum.BAD_DREAMS) &
              (state.active_field('current_hp', source)[0] > 0) &
              ((status == Status.SLEEP) | (ability == AbilityEnum.COMATOSE)))
    traps = state.volatile_status.trapping
    binding = jnp.where(traps.binding_turns[player, 0] > 0,
                        1 / jnp.maximum(1, traps.binding_divisor[player, 0]), 0.)
    r = state.volatile_status.residuals
    seed = (r.leech_seed[player, 0] != 0) & (state.active_field('current_hp', source)[0] > 0)
    salt_vulnerable = jnp.any((mon.type_list == Type.WATER) | (mon.type_list == Type.STEEL))
    salt = (r.salt_cure[player, 0] != 0) * jnp.where(salt_vulnerable, 1 / 4, 1 / 8)
    wish = state.slot_residuals
    wish_due = (wish.wish_turns[player, 0] == 1) & (wish.wish_hp[player, 0] > 0)
    fractions = jnp.stack((weather, jnp.where(grassy, -1 / 16, 0.), poison, burn, dreams / 8, binding,
        -wish_due.astype(jnp.float32), -(r.aqua_ring[player, 0] != 0).astype(jnp.float32) / 16,
        -(r.ingrain[player, 0] != 0).astype(jnp.float32) / 16, seed / 8, (r.curse[player, 0] != 0) / 4, salt))
    fraction = fractions[phase]
    return jnp.where((mon.current_hp[0] > 0) &
                     ((ability != AbilityEnum.MAGIC_GUARD) | (fraction < 0)), fraction, 0.)


def _apply_residual_hp(state, player, fraction, phase):
    """Attribute and commit a single residual HP change."""
    causes = jnp.array([Cause.WEATHER, Cause.TERRAIN, Cause.STATUS, Cause.STATUS] + [Cause.INDIRECT] * 8)
    context = state.event_context
    binding = phase == ResidualHPPhase.BINDING
    seed = phase == ResidualHPPhase.LEECH_SEED
    wish = phase == ResidualHPPhase.WISH
    passive_healing = (phase == ResidualHPPhase.AQUA_RING) | (phase == ResidualHPPhase.INGRAIN)
    opposing = binding | seed | (phase == ResidualHPPhase.BAD_DREAMS) | (
        phase == ResidualHPPhase.CURSE) | (phase == ResidualHPPhase.SALT_CURE)
    source = jnp.where(opposing, 1-player, jnp.where(passive_healing | wish, player, context.source))
    ids = jnp.array([0, 0, 0, 0, 0, 0, 273, 392, 275, 73, 174, 864])
    move_id = jnp.where(binding, state.volatile_status.trapping.binding_move_id[player, 0], ids[phase])
    state = state.replace(event_context=context.replace(source=source, move_id=move_id, cause=causes[phase]))
    amount = jnp.maximum(1, jnp.floor(jnp.abs(fraction) * state.active_stats(player)[0]).astype(jnp.int32))
    amount = jnp.where(passive_healing, root_healing(state, player, amount), amount)
    amount = jnp.where(wish, state.slot_residuals.wish_hp[player, 0], amount)
    # Seed drains actual damage before reactive healing, including on a KO.
    drained = jnp.minimum(amount, state.active_field('current_hp', player)[0])
    signed = jnp.where(fraction < 0, -amount, amount)
    state = take_damage_value(state, player, jnp.array([signed]), False, source=source)
    state = jax.lax.cond(seed, _leech_heal, _skip_leech_heal, state, player, drained)
    return state.replace(event_context=context)


def _leech_heal(state, victim, drained):
    """Heal the opposing position or reverse the transfer with Liquid Ooze."""
    holder = 1-victim
    amount = root_healing(state, holder, drained)
    ooze = state.ability[victim, 0] == AbilityEnum.LIQUID_OOZE
    damage = jnp.where(state.ability[holder, 0] == AbilityEnum.MAGIC_GUARD, 0, amount)
    old = state.event_context
    state = state.replace(event_context=old.replace(source=victim))
    state = take_damage_value(state, holder, jnp.array([jnp.where(ooze, damage, -amount)]), False, source=victim)
    return state.replace(event_context=old)


def _skip_leech_heal(state, victim, drained):
    """Preserve HP for residuals that do not transfer it."""
    return state


def _skip_residual_hp(state, player, fraction, phase):
    """Leave HP and events untouched when the holder is ineligible."""
    return state


def _apply_hp_phase_to_player(index, state, *, phase, order):
    """Run the current phase for one holder in the Speed order."""
    grouped = phase == ResidualHPPhase.BINDING_AND_SALT
    player = order[jnp.where(grouped, index // 2, index)]
    phase = jnp.where(grouped, jnp.where(index % 2 == 0, ResidualHPPhase.BINDING, ResidualHPPhase.SALT_CURE), phase)
    fraction = _residual_fraction(state, player, phase)
    return jax.lax.cond((state.winner < 0) & (fraction != 0),
                        _apply_residual_hp, _skip_residual_hp, state, player, fraction, phase)


def _run_hp_phase(state, phase, *, order):
    """Apply a phase and resolve its faints before the next phase."""
    count = jnp.where(phase == ResidualHPPhase.BINDING_AND_SALT, 4, 2)
    state = jax.lax.fori_loop(0, count, partial(_apply_hp_phase_to_player, phase=phase, order=order), state)
    return resolve_faints(state), None


@jax.jit
def run_hp_residuals(state, phases, order=None):
    """Scan a fixed list of residual phases with one shared HP update body."""
    if order is None:
        order = residual_speed_order(state)
    return jax.lax.scan(partial(_run_hp_phase, order=order), state, jnp.asarray(phases, jnp.int32))[0]


def end_turn_damage(state: BattleState) -> BattleState:
    """Run all represented HP phases for callers excluding timer and item effects."""
    phase = ResidualHPPhase
    return run_hp_residuals(state, (phase.WEATHER, phase.WISH, phase.GRASSY_TERRAIN, phase.AQUA_RING,
        phase.INGRAIN, phase.LEECH_SEED, phase.POISON, phase.BURN, phase.CURSE, phase.BINDING_AND_SALT,
        phase.BAD_DREAMS))


def _tick_side_timer(turns):
    """Decrement one side condition duration without going below zero."""
    return jnp.maximum(turns - 1, 0)


def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    """Decrement side timers and update toxic counters; return the unchanged key."""
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state = state.replace(
        side_conditions=jax.tree.map(_tick_side_timer, state.side_conditions),
        side_protection=SideProtection(),
        toxic_counter=toxic_counter)
    v = state.volatile_status
    guard = v.protection
    state = state.replace(volatile_status=v.replace(
        protection=guard.replace(
            protected=jnp.zeros_like(guard.protected),
            protection_kind=jnp.zeros_like(guard.protection_kind),
            protect_chain=jnp.where(guard.stall_refreshed, guard.protect_chain, 0),
            stall_refreshed=jnp.zeros_like(guard.stall_refreshed)),
        queued_move=v.queued_move.replace(last_action=jnp.zeros_like(v.queued_move.last_action)),
        flinched=jnp.zeros_like(v.flinched),
        embargo_turns=jnp.maximum(v.embargo_turns - 1, 0)))

    state = tick(state)
    return key, state


def _step_moody_for_player(index, carry, *, order):
    """Roll Moody for one holder using the shared residual Speed order."""
    key, state = carry
    player = order[index]
    key, plus_key, minus_key = random.split(key, 3)
    stages = state.boosts.normal_boosts[player, 1:]
    can_raise = stages < 6
    plus = random.categorical(plus_key, jnp.where(can_raise, 0., -jnp.inf))
    can_lower = (stages > -6) & ((jnp.arange(5) != plus) | ~jnp.any(can_raise))
    minus = random.categorical(minus_key, jnp.where(can_lower, 0., -jnp.inf))
    active = ((state.ability[player, 0] == AbilityEnum.MOODY) &
              (state.active_field('current_hp', player)[0] > 0))
    state = conditional_add_boosts(state, player, active & jnp.any(can_raise), 1 + plus, 2)
    state = conditional_add_boosts(state, player, active & jnp.any(can_lower), 1 + minus, -1)
    return key, state


def step_moody(key, state, order=None):
    """Apply represented Moody changes, splitting one key per side."""
    if order is None:
        order = residual_speed_order(state)
    return jax.lax.fori_loop(0, 2, partial(_step_moody_for_player, order=order), (key, state))


def _apply_orb_status(k, s, *, player, status):
    """Apply orb status."""
    return set_status(k, s, player, jnp.asarray(status).reshape((1,)))


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)


def _residual_orbs_apply(index, carry, *, order):
    """Use existing status immunity and explicit-key helpers for each holder."""
    key, s = carry
    player = order[index]
    item = s.active.item[player, 0]
    status = jnp.where(item == I.FLAME_ORB, Status.BURN, jnp.where(item == I.TOXIC_ORB, Status.TOXIC, Status.NONE))
    return jax.lax.cond((status != Status.NONE) & (s.active.current_hp[player, 0] > 0),
        partial(_apply_orb_status, player=player, status=status),
        _unchanged_key_state, key, s)


def _leftovers_heal(index, s, *, order):
    """Apply one player's residual item healing."""
    player = order[index]
    mon = s.active[player]
    return take_damage_percent(s, player, jnp.where((mon.item[0] == I.LEFTOVERS) & (mon.current_hp[0] > 0), -1/16, 0))


def residual_orbs(key, state, order=None):
    """Inflict held-orb status in Speed order after status damage."""
    if order is None:
        order = residual_speed_order(state)
    return jax.lax.fori_loop(0, 2, partial(_residual_orbs_apply, order=order), (key, state))


def leftovers(state, order=None):
    """Heal living Leftovers holders in the shared Speed order."""
    if order is None:
        order = residual_speed_order(state)
    return jax.lax.fori_loop(0, 2, partial(_leftovers_heal, order=order), state)


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


def _residual_abilities_apply(index, carry, *, status_cures, weather, speed_boost, order):
    """Only roll Shed Skin for a living, statused holder."""
    key, s = carry
    player = order[index]
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
    """Apply the late Bad Dreams residual with normal HP attribution."""
    return run_hp_residuals(state, (ResidualHPPhase.BAD_DREAMS,))


def residual_abilities(key, state, status_cures=True, speed_boost=True, *, order=None):
    """Resolve represented status cures and Speed Boost with explicit randomness.

    The scheduler runs status cures before poison/burn damage, then Speed Boost
    late in the turn. It does not activate on a normal switch-in's first turn.
    """


    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    if order is None:
        order = residual_speed_order(state)
    apply = partial(_residual_abilities_apply, status_cures=status_cures, weather=weather, speed_boost=speed_boost, order=order)
    return jax.lax.fori_loop(0, 2, apply, (key, state))
