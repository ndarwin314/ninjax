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


def is_poison_type_user(state, attacker_index):
    """Resolve the user's current Poison typing, including ordinary Tera."""
    types = state.effective_types(attacker_index)
    tera = state.active_field('is_terastallized', attacker_index)[0]
    tera_type = state.active_field('tera_type', attacker_index)[0]
    return jnp.where(tera & (tera_type != Type.NONE), tera_type == Type.POISON, jnp.any(types == Type.POISON))


def accuracy_fields(state, attacker_index):
    """Read the current values needed to check a move's accuracy."""
    abilities = state.ability
    target = 1 - attacker_index
    poison_user = is_poison_type_user(state, attacker_index)
    return (abilities[attacker_index], abilities[target], state.active_field('item', attacker_index),
            state.weather.weather, abilities, state.active_field('current_hp'),
            state.boosts.acc_boosts[attacker_index, 0], state.boosts.acc_boosts[target, 1],
            state.volatile_status.confusion_turns[target], poison_user)


def move_accuracy_from_fields(move, attacker_ability, defender_ability, item, field_weather,
                              abilities, hp, accuracy_stage, evasion_stage, confusion_turns, poison_user=False, *, absorption=None):
    """Calculate accuracy from the current move and relevant battle fields."""
    weather = effective_weather(field_weather, abilities, hp)
    is_hustle = jnp.logical_and((attacker_ability == AbilityEnum.HUSTLE), move.move_type==MoveType.PHYSICAL)

    # Thunder and Hurricane use 50% base accuracy in sun, before stages and
    # other accuracy modifiers. The three Forces of Nature storms do not.
    sun_sensitive = (move.identifier == 87) | (move.identifier == 542)
    base_accuracy = jnp.where(sun_sensitive & (weather == WeatherEnum.SUN), .5, move.accuracy)
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
    # These weather effects bypass the accuracy roll and its modifiers.
    rain_sensitive = sun_sensitive | (move.identifier == 846) | (move.identifier == 847) | (move.identifier == 848)
    weather_always = ((move.identifier == 59) & (weather == WeatherEnum.SNOW)) | (rain_sensitive & (weather == WeatherEnum.RAIN))
    accuracy = jnp.where((defender_ability == AbilityEnum.TANGLED_FEET) & (confusion_turns > 0), accuracy * .5, accuracy)
    toxic_always = (move.identifier == 92) & poison_user
    accuracy = jnp.where(move.target_self | move.always_hit | no_guard_active | weather_always | absorption.active | toxic_always, 1., accuracy)

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
    terrain = state.terrain.terrain
    types = state.effective_types(player)
    item = state.active_field('item', player)
    # Weather Ball's power is a ModifyMove effect, coupled to its type change.
    # All BasePower callbacks are evaluated later from live hit-time inputs.
    double_in_weather = (move.power_rule == PowerRule.DOUBLE_IN_WEATHER) & (weather != W.NONE)
    double_in_terrain = (move.power_rule == PowerRule.DOUBLE_IN_TERRAIN) & (terrain != W.NONE)
    power = jnp.where(double_in_weather | double_in_terrain, move.base_power * 2, move.base_power)
    move_type = _override_type(state, player, ident, move.type, weather, terrain, types, item)
    priority = _terrain_priority(state, ident, move.priority, types, abilities[player], item)
    return move.replace(base_power=power, type=move_type, priority=priority)


# Match on form identity so temporary type changes never change these moves.
FORM_MOVE_TYPES = (
    (904, FormEnum.OGERPON_WELLSPRING, Type.WATER),
    (904, FormEnum.OGERPON_HEARTHFLAME, Type.FIRE),
    (904, FormEnum.OGERPON_CORNERSTONE, Type.ROCK),
    (873, FormEnum.TAUROS_PALDEA_COMBAT, Type.FIGHTING),
    (873, FormEnum.TAUROS_PALDEA_BLAZE, Type.FIRE),
    (873, FormEnum.TAUROS_PALDEA_AQUA, Type.WATER),
    (783, FormEnum.MORPEKO_HANGRY, Type.DARK),
)


def form_move_type(identifier, default_type, form):
    """Apply form callbacks to the transient type shared by planning and hits."""
    # Aura Wheel explicitly sets Electric outside Hangry Mode, even after Normalize.
    move_type = jnp.where(identifier == 783, Type.ELECTRIC, default_type)
    for move_id, required_form, resolved_type in FORM_MOVE_TYPES:
        move_type = jnp.where((identifier == move_id) & (form == required_form),
                              resolved_type, move_type)
    return move_type

def judgement_move_type(item):
    return jnp.select([
        item == ItemEnum.FIST_PLATE,
        item == ItemEnum.SKY_PLATE,
        item == ItemEnum.TOXIC_PLATE,
        item == ItemEnum.EARTH_PLATE,
        item == ItemEnum.STONE_PLATE,
        item == ItemEnum.INSECT_PLATE,
        item == ItemEnum.SPOOKY_PLATE,
        item == ItemEnum.IRON_PLATE,
        item == ItemEnum.FLAME_PLATE,
        item == ItemEnum.SPLASH_PLATE,
        item == ItemEnum.MEADOW_PLATE,
        item == ItemEnum.ZAP_PLATE,
        item == ItemEnum.MIND_PLATE,
        item == ItemEnum.ICICLE_PLATE,
        item == ItemEnum.DRACO_PLATE,
        item == ItemEnum.DREAD_PLATE,
        item == ItemEnum.PIXIE_PLATE,
    ], [
        Type.FIGHTING,
        Type.FLYING,
        Type.POISON,
        Type.GROUND,
        Type.ROCK,
        Type.BUG,
        Type.GHOST,
        Type.STEEL,
        Type.FIRE,
        Type.WATER,
        Type.GRASS,
        Type.ELECTRIC,
        Type.PSYCHIC,
        Type.ICE,
        Type.DRAGON,
        Type.DARK,
        Type.FAIRY
    ], Type.NORMAL)


def _override_type(
        state,
        player,
        ident,
        move_type,
        weather, terrain,
        types, item):
    """Resolve weather, terrain, form and current-type callbacks."""
    weather_ball = (ident == 311) & (weather != W.NONE)
    move_type = jnp.where(weather_ball & (weather == W.SUN), Type.FIRE, move_type)
    move_type = jnp.where(weather_ball & (weather == W.RAIN), Type.WATER, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SANDSTORM), Type.ROCK, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SNOW), Type.ICE, move_type)
    terrain_pulse = (ident == 805) & (terrain != T.NONE)
    move_type = jnp.where(terrain_pulse & (terrain == T.ELECTRIC), Type.ELECTRIC, move_type)
    move_type = jnp.where(terrain_pulse & (terrain == T.GRASSY), Type.GRASS, move_type)
    move_type = jnp.where(terrain_pulse & (terrain == T.MISTY), Type.FAIRY, move_type)
    move_type = jnp.where(terrain_pulse & (terrain == T.PSYCHIC), Type.PSYCHIC, move_type)

    move_type = form_move_type(ident, move_type, state.active_field('form', player))
    move_type = jnp.where(ident == 449, judgement_move_type(item), move_type)
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
