"""Properties helpers for immutable JAX battle state."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ninjax.calculations.modifiers import (
    COMPOUND_EYES_MULTIPLIER,
    VICTORY_STAR,
    WEATHER_VEIL_MODIFIER,
)
from ninjax.calculations.stat_formulas import ACCURACY_MULTIPLIER_LOOKUP
from ninjax.enums.abilities import AbilityEnum, AbilityEnum as A
from ninjax.enums.field import TerrainEnum as T, WeatherEnum, WeatherEnum as W
from ninjax.enums.items import ItemEnum, ItemEnum as I
from ninjax.enums.moves import MoveFlags, MoveType, PowerRule
from ninjax.enums.pokemon import FormEnum, StatEnum, Type
from ninjax.side.queries import (
    effective_weather,
    is_floating_fields,
)


def effective_move(state, attacker_index, move):
    """Return a transient type/power override without mutating the move slot."""
    ability = state.ability[attacker_index]
    excluded = _excludes_type_conversion(move.identifier)
    # Normalize runs before move-specific type callbacks in the pinned rules.
    normalize = (ability == AbilityEnum.NORMALIZE) & ~excluded
    move = move.replace(type=jnp.where(normalize, Type.NORMAL, move.type))
    move = move_overrides(state, attacker_index, move)
    move = move.replace(priority=_ability_priority(state, attacker_index, move, move.type, move.priority, ability))
    normal = (move.type == Type.NORMAL) & ~excluded
    changed = normal & ((ability == AbilityEnum.PIXILATE) | (ability == AbilityEnum.AERILATE) |
                        (ability == AbilityEnum.REFRIGERATE) | (ability == AbilityEnum.GALVANIZE))
    move_type = jnp.where(normal & (ability == AbilityEnum.PIXILATE), Type.FAIRY, move.type)
    move_type = jnp.where(normal & (ability == AbilityEnum.AERILATE), Type.FLYING, move_type)
    move_type = jnp.where(normal & (ability == AbilityEnum.REFRIGERATE), Type.ICE, move_type)
    move_type = jnp.where(normal & (ability == AbilityEnum.GALVANIZE), Type.ELECTRIC, move_type)
    move_type = jnp.where((ability == AbilityEnum.LIQUID_VOICE) & move.sound, Type.WATER, move_type)
    return move.replace(type=move_type, type_power_boost=normalize | changed)


def _excludes_type_conversion(identifier):
    """Check move IDs that keep their own typing through type conversion."""
    # Judgment, Multi-Attack, Natural Gift, Revelation Dance, Techno Blast,
    # Terrain Pulse and Weather Ball keep their own typing.
    return jnp.any(identifier[0] == jnp.array([449, 718, 363, 686, 546, 805, 311]))


def _ability_priority(state, player, move, move_type, priority, ability):
    """Apply ability priority before the final -ate and Liquid Voice changes."""
    priority += jnp.where((ability == AbilityEnum.PRANKSTER) & (move.move_type == MoveType.STATUS), 1, 0)
    hp = state.active_field('current_hp', player)[0]
    priority += jnp.where((ability == AbilityEnum.GALE_WINGS) & (move_type == Type.FLYING) &
                          (hp == state.active_stats(player)[0]), 1, 0)
    return priority + jnp.where((ability == AbilityEnum.TRIAGE) & move.healing, 3, 0)


def effective_priority(state, player, move):
    """Resolve queued priority without computing base-power overrides."""
    abilities = state.ability
    ability = abilities[player]
    normalize = (ability == A.NORMALIZE) & ~_excludes_type_conversion(move.identifier)
    weather = effective_weather(state.weather.weather, abilities, state.active_field('current_hp'))
    types = state.effective_types(player)
    item = state.active_field('item', player)
    move_type = _override_type(state, player, move.identifier, jnp.where(normalize, Type.NORMAL, move.type), weather, types)
    priority = _terrain_priority(state, move.identifier, move.priority, types, ability, item)
    return _ability_priority(state, player, move, move_type, priority, ability)


class Absorption(NamedTuple):
    """The single defender response shared by accuracy and hit dispatch."""

    stat: jax.Array
    stages: jax.Array
    flash_fire: jax.Array
    heal: jax.Array

    @property
    def active(self):
        """Return whether this response bypasses the accuracy roll."""
        return (self.stat != 0) | self.flash_fire | self.heal


def absorption_from_fields(move, ability):
    """Classify the represented opposing ability intercepts before accuracy."""
    water = move.type == Type.WATER
    electric = move.type == Type.ELECTRIC
    fire = move.type == Type.FIRE
    defense = (ability == A.WELL_BAKED_BODY) & fire
    # Whirlwind lets the drag request handle Wind Rider and its failure outcome.
    attack = ((ability == A.SAP_SIPPER) & (move.type == Type.GRASS)) | (
        (ability == A.WIND_RIDER) & ((move.move_flags & MoveFlags.WIND) != 0) &
        (move.identifier != 18))
    special_attack = ((ability == A.STORM_DRAIN) & water) | ((ability == A.LIGHTNING_ROD) & electric)
    speed = (ability == A.MOTOR_DRIVE) & electric
    stat = jnp.select((defense, attack, special_attack, speed),
                      (StatEnum.DEFENSE, StatEnum.ATTACK, StatEnum.SPECIAL_ATTACK, StatEnum.SPEED),
                      default=0)
    heal = (water & ((ability == A.WATER_ABSORB) | (ability == A.DRY_SKIN))) | (
        electric & (ability == A.VOLT_ABSORB)) | ((move.type == Type.GROUND) & (ability == A.EARTH_EATER))
    opposing = ~move.target_self
    return Absorption(jnp.where(opposing, stat, 0), 1 + defense,
                      opposing & (ability == A.FLASH_FIRE) & fire, opposing & heal)


def move_accuracy(state, attacker_index, move, absorption=None):
    """Calculate the currently represented accuracy for an individual attempt."""
    return move_accuracy_from_fields(move, *accuracy_fields(state, attacker_index), absorption=absorption)


def accuracy_fields(state, attacker_index):
    """Read the current values needed to check a move's accuracy."""
    abilities = state.ability
    target = 1 - attacker_index
    return (abilities[attacker_index], abilities[target], state.active_field('item', attacker_index),
            state.weather.weather, abilities, state.active_field('current_hp'),
            state.boosts.acc_boosts[attacker_index, 0], state.boosts.acc_boosts[target, 1],
            state.volatile_status.confusion_turns[target])


def move_accuracy_from_fields(move, attacker_ability, defender_ability, item, field_weather,
                              abilities, hp, accuracy_stage, evasion_stage, confusion_turns, *, absorption=None):
    """Calculate accuracy from the current move and relevant battle fields."""
    weather = effective_weather(field_weather, abilities, hp)
    is_hustle = jnp.logical_and((attacker_ability == AbilityEnum.HUSTLE), move.move_type==MoveType.PHYSICAL)

    base_accuracy = jnp.where((move.identifier == 542) & (weather == WeatherEnum.SUN), .5, move.accuracy)
    ignore_evasion = move.ignore_defensive[0] | (attacker_ability[0] == AbilityEnum.KEEN_EYE)
    accuracy = (base_accuracy *
                ACCURACY_MULTIPLIER_LOOKUP[6 + accuracy_stage] *
                ACCURACY_MULTIPLIER_LOOKUP[6 - jnp.where(ignore_evasion, 0, evasion_stage)] *
                jnp.power(0.8, is_hustle))
    veil_active = jnp.logical_or(
        jnp.logical_and(defender_ability == AbilityEnum.SAND_VEIL, weather == WeatherEnum.SANDSTORM),
        jnp.logical_and(defender_ability == AbilityEnum.SNOW_CLOAK, weather == WeatherEnum.SNOW))
    conditions = jnp.array(
        [jnp.equal(attacker_ability, AbilityEnum.COMPOUND_EYES),
         attacker_ability==AbilityEnum.VICTORY_STAR,
         veil_active]).reshape((3,))
    modifiers = jnp.array([COMPOUND_EYES_MULTIPLIER, VICTORY_STAR, WEATHER_VEIL_MODIFIER])
    accuracy = accuracy * jnp.prod(jnp.power(modifiers, conditions))
    accuracy *= jnp.where(item == ItemEnum.WIDE_LENS, 1.1, 1.)
    # The simulator samples an integer percentage after accuracy modifiers.
    accuracy = jnp.floor(accuracy * 100 + 1e-5) / 100
    no_guard_active = jnp.logical_or(
        jnp.equal(defender_ability, AbilityEnum.NO_GUARD),
        jnp.equal(attacker_ability, AbilityEnum.NO_GUARD))
    accuracy = jnp.clip(accuracy, no_guard_active, 1)
    if absorption is None:
        absorption = absorption_from_fields(move, defender_ability)
    # Blizzard in snow and Hurricane in rain bypass accuracy/evasion checks.
    weather_always = ((move.identifier == 59) & (weather == WeatherEnum.SNOW)) | ((move.identifier == 542) & (weather == WeatherEnum.RAIN))
    accuracy = jnp.where((defender_ability == AbilityEnum.TANGLED_FEET) & (confusion_turns > 0), accuracy * .5, accuracy)
    accuracy = jnp.where(move.target_self | move.always_hit | no_guard_active | weather_always | absorption.active, 1., accuracy)

    return accuracy


def substitute_intercepts(substitute_hp, attacker_ability, target_self, sound, bypass_substitute):
    """Check interception from scalar HP, effective ability and move flags."""
    return ((substitute_hp > 0) & ~target_self & ~sound & ~bypass_substitute &
            (attacker_ability != AbilityEnum.INFILTRATOR))


def move_overrides(state, player, move):
    """Resolve property-time power/type changes and grounded terrain priority."""
    ident = move.identifier
    abilities = state.ability
    weather = effective_weather(state.weather.weather, abilities, state.active_field('current_hp'))
    types = state.effective_types(player)
    item = state.active_field('item', player)
    # Weather Ball's power is a ModifyMove effect, coupled to its type change.
    # All BasePower callbacks are evaluated later from live hit-time inputs.
    power = jnp.where((move.power_rule == PowerRule.DOUBLE_IN_WEATHER) &
                      (weather != W.NONE), move.base_power * 2, move.base_power)
    move_type = _override_type(state, player, ident, move.type, weather, types)
    priority = _terrain_priority(state, ident, move.priority, types, abilities[player], item)
    return move.replace(base_power=power, type=move_type, priority=priority)


def _override_type(state, player, ident, move_type, weather, types):
    """Resolve Weather Ball, Ivy Cudgel and Revelation Dance typing."""
    weather_ball = (ident == 311) & (weather != W.NONE)
    move_type = jnp.where(weather_ball & (weather == W.SUN), Type.FIRE, move_type)
    move_type = jnp.where(weather_ball & (weather == W.RAIN), Type.WATER, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SANDSTORM), Type.ROCK, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SNOW), Type.ICE, move_type)
    form = state.active_field('form', player)
    ivy_type = jnp.where(form == FormEnum.OGERPON_WELLSPRING, Type.WATER,
               jnp.where(form == FormEnum.OGERPON_HEARTHFLAME, Type.FIRE,
               jnp.where(form == FormEnum.OGERPON_CORNERSTONE, Type.ROCK, move_type)))
    move_type = jnp.where(ident == 904, ivy_type, move_type)
    # Revelation Dance uses the first current type; ordinary Tera overrides it.
    first_type = jnp.where(types[0] != Type.NONE, types[0], jnp.where(types[1] != Type.NONE, types[1], Type.NORMAL))
    tera_type = state.active_field('tera_type', player)[0]
    tera = state.active_field('is_terastallized', player)[0]
    dance_type = jnp.where(tera & (tera_type != Type.NONE), tera_type, first_type)
    return jnp.where(ident == 686, dance_type, move_type)


def _terrain_priority(state, ident, priority, types, ability, item):
    """Raise Grassy Glide's priority for a grounded user on Grassy Terrain."""
    grounded = ~is_floating_fields(types, ability, item) | (state.gravity_duration > 0)
    return jnp.where((ident == 803) & grounded & (state.terrain.terrain == T.GRASSY), 1, priority)
