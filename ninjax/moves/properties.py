"""Properties helpers for immutable JAX battle state."""
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
from ninjax.enums.moves import MoveFlags, MoveResult, MoveType
from ninjax.enums.pokemon import FormEnum, Type
from ninjax.side.queries import (
    conditional_mult_round as fixed_modifier,
    conditional_mult_round,
    effective_weather,
    item_transfer_allowed,
    is_floating_fields,
)


def _move_overrides_weight(weight, ability):
    """Apply represented weight abilities in integer hectograms."""
    return jnp.maximum(1, weight * jnp.where(ability == A.HEAVY_METAL, 2, 1) // jnp.where(ability == A.LIGHT_METAL, 2, 1))


def effective_move(state, attacker_index, move):
    """Return a transient type/power override without mutating the move slot."""
    ability = state.ability[attacker_index]
    excluded = _excludes_type_conversion(move.identifier)
    # Normalize runs before move-specific type callbacks in the pinned rules.
    normalize = (ability == AbilityEnum.NORMALIZE) & ~excluded
    move = move.replace(type=jnp.where(normalize, Type.NORMAL, move.type),
        base_power=fixed_modifier(move.base_power, 4915/4096, normalize))
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
    return move.replace(type=move_type, base_power=fixed_modifier(move.base_power, 4915/4096,
                        changed))


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


def move_accuracy(state, attacker_index, move):
    """Calculate the currently represented accuracy for an individual attempt."""
    return move_accuracy_from_fields(move, *accuracy_fields(state, attacker_index))


def accuracy_fields(state, attacker_index):
    """Read the current values needed to check a move's accuracy."""
    abilities = state.ability
    target = 1 - attacker_index
    return (abilities[attacker_index], abilities[target], state.active_field('item', attacker_index),
            state.weather.weather, abilities, state.active_field('current_hp'),
            state.boosts.acc_boosts[attacker_index, 0], state.boosts.acc_boosts[target, 1],
            state.volatile_status.confusion_turns[target])


def move_accuracy_from_fields(move, attacker_ability, defender_ability, item, field_weather,
                              abilities, hp, accuracy_stage, evasion_stage, confusion_turns):
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
    flash_fire = (defender_ability == AbilityEnum.FLASH_FIRE) & (move.type == Type.FIRE) & ~move.target_self
    accuracy = jnp.where(flash_fire, 1., accuracy)
    # Storm Drain and Water Absorb intercept incoming Water moves in TryHit,
    # before Showdown's accuracy stage. Dispatch their branch even on a move
    # that would otherwise miss; self-targeted moves are handled separately.
    absorbed = (((move.type == Type.WATER) & ((defender_ability == AbilityEnum.STORM_DRAIN) |
        (defender_ability == AbilityEnum.WATER_ABSORB) | (defender_ability == AbilityEnum.DRY_SKIN))) |
        ((move.type == Type.ELECTRIC) & ((defender_ability == AbilityEnum.LIGHTNING_ROD) |
        (defender_ability == AbilityEnum.VOLT_ABSORB) | (defender_ability == AbilityEnum.MOTOR_DRIVE))) |
        ((move.type == Type.FIRE) & (defender_ability == AbilityEnum.WELL_BAKED_BODY)) |
        ((move.type == Type.GRASS) & (defender_ability == AbilityEnum.SAP_SIPPER)) |
        ((move.type == Type.GROUND) & (defender_ability == AbilityEnum.EARTH_EATER))) & ~move.target_self

    absorbed |= (defender_ability == AbilityEnum.WIND_RIDER) & ((move.move_flags & MoveFlags.WIND) != 0) & ~move.target_self
    accuracy = jnp.where(absorbed, 1., accuracy)
    # Blizzard in snow and Hurricane in rain bypass accuracy/evasion checks.
    weather_always = ((move.identifier == 59) & (weather == WeatherEnum.SNOW)) | ((move.identifier == 542) & (weather == WeatherEnum.RAIN))
    accuracy = jnp.where((defender_ability == AbilityEnum.TANGLED_FEET) & (confusion_turns > 0), accuracy * .5, accuracy)
    accuracy = jnp.where(move.target_self | move.always_hit | no_guard_active | weather_always, 1., accuracy)

    return accuracy


def substitute_intercepts(substitute_hp, attacker_ability, target_self, sound, bypass_substitute):
    """Check interception from scalar HP, effective ability and move flags."""
    return ((substitute_hp > 0) & ~target_self & ~sound & ~bypass_substitute &
            (attacker_ability != AbilityEnum.INFILTRATOR))


def move_overrides(state, player, move):
    """Resolve represented move power, typing and grounded terrain priority."""
    ident = move.identifier
    abilities = state.ability
    weather = effective_weather(state.weather.weather, abilities, state.active_field('current_hp'))
    types = state.effective_types(player)
    item = state.active_field('item', player)
    power = jnp.where((ident == 512) & (item == I.NONE), 110, move.base_power)
    power = jnp.where((ident == 707) & (state.volatile_status.previous_move_result[player] == MoveResult.FAILED), power * 2, power)

    opponent = 1-player
    target_item = state.active_field('item', opponent)[0]
    species = state.active_field('species', opponent)[0]
    form = state.active_field('form', opponent)[0]
    knock_boost = (ident == 282) & (target_item != I.NONE) & item_transfer_allowed(species, form, target_item)
    power = conditional_mult_round(power, 1.5, knock_boost)
    positive = jnp.sum(jnp.maximum(state.boosts.normal_boosts[player, 1:], 0)) + jnp.sum(jnp.maximum(state.boosts.acc_boosts[player], 0))
    power = jnp.where(ident == 500, 20 + 20 * positive, power)
    weight = _move_overrides_weight(state.active_field('weight_hg', player), abilities[player])
    target_weight = _move_overrides_weight(state.active_field('weight_hg', opponent), abilities[opponent])
    low_kick = 20 + 20 * sum((target_weight >= threshold).astype(jnp.int32) for threshold in (100, 250, 500, 1000, 2000))
    heavy_slam = jnp.where(weight >= 5*target_weight, 120,
                 jnp.where(weight >= 4*target_weight, 100,
                 jnp.where(weight >= 3*target_weight, 80,
                 jnp.where(weight >= 2*target_weight, 60, 40))))
    power = jnp.where(ident == 67, low_kick, power)
    power = jnp.where(ident == 484, heavy_slam, power)
    power = jnp.where((ident == 311) & (weather != W.NONE), 100, power)
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
