"""Properties helpers for immutable JAX battle state."""
import jax.numpy as jnp
from ninjax.enum_types import (
    AbilityEnum,
    MoveType,
    Type,
    ItemEnum,
    MoveFlags,
    WeatherEnum,
    AbilityEnum as A,
    FormEnum,
    ItemEnum as I,
    MoveResult,
    TerrainEnum as T,
    WeatherEnum as W,
)
from ninjax.side.queries import (
    conditional_mult_round as fixed_modifier,
    effective_weather,
    conditional_mult_round,
    item_transfer_allowed,
)
from ninjax.utils import (
    ACCURACY_MULTIPLIER_LOOKUP,
    COMPOUND_EYES_MULTIPLIER,
    VICTORY_STAR,
    WEATHER_VEIL_MODIFIER,
)



def _move_overrides_weight(mon):
    """Apply represented weight abilities in integer hectograms."""
    return jnp.maximum(1, mon.weight_hg * jnp.where(mon.ability == A.HEAVY_METAL, 2, 1) // jnp.where(mon.ability == A.LIGHT_METAL, 2, 1))


def effective_move(state, attacker_index, move):
    """Return a transient type/power override without mutating the move slot."""
    ability = state.ability[attacker_index]
    # Showdown excludes Judgment (449), Multi-Attack (718), Natural Gift (363),
    # Revelation Dance (686), Techno Blast (546), Terrain Pulse (805), and
    # Weather Ball (311) from -ate type conversion.
    excluded = jnp.any(move.identifier[0] == jnp.array([449, 718, 363, 686, 546, 805, 311]))
    # Normalize runs before move-specific type callbacks in the pinned rules.
    normalize = (ability == AbilityEnum.NORMALIZE) & ~excluded
    move = move.replace(type=jnp.where(normalize, Type.NORMAL, move.type),
        base_power=fixed_modifier(move.base_power, 4915/4096, normalize))
    move = move_overrides(state, attacker_index, move)
    # These abilities change the queued priority bracket, independently of Speed.
    priority = move.priority + jnp.where((ability == AbilityEnum.PRANKSTER) & (move.move_type == MoveType.STATUS), 1, 0)
    priority += jnp.where((ability == AbilityEnum.GALE_WINGS) & (move.type == Type.FLYING) &
                         (state.active.current_hp[attacker_index, 0] == state.active.max_hp[attacker_index]), 1, 0)
    priority += jnp.where((ability == AbilityEnum.TRIAGE) & move.healing, 3, 0)
    move = move.replace(priority=priority)
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


def move_accuracy(state, attacker_index, move):
    """Calculate the currently represented accuracy for an individual attempt."""
    active = state.effective_active
    defender_ability = active[1 - attacker_index].ability
    attacker_ability = active[attacker_index].ability


    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    is_hustle = jnp.logical_and((attacker_ability == AbilityEnum.HUSTLE), move.move_type==MoveType.PHYSICAL)

    base_accuracy = jnp.where((move.identifier == 542) & (weather == WeatherEnum.SUN), .5, move.accuracy)
    ignore_evasion = move.ignore_defensive[0] | (attacker_ability[0] == AbilityEnum.KEEN_EYE)
    accuracy = (base_accuracy *
                ACCURACY_MULTIPLIER_LOOKUP[6 + state.boosts.acc_boosts[attacker_index, 0]] *
                ACCURACY_MULTIPLIER_LOOKUP[6 - jnp.where(ignore_evasion, 0, state.boosts.acc_boosts[1 - attacker_index, 1])] *
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
    accuracy *= jnp.where(active.item[attacker_index] == ItemEnum.WIDE_LENS, 1.1, 1.)
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
    accuracy = jnp.where((defender_ability == AbilityEnum.TANGLED_FEET) & (state.volatile_status.confusion_turns[1-attacker_index] > 0), accuracy * .5, accuracy)
    accuracy = jnp.where(move.target_self | move.always_hit | no_guard_active | weather_always, 1., accuracy)

    return accuracy


def substitute_intercepts(substitute_hp, attacker_ability, target_self, sound, bypass_substitute):
    """Check interception from scalar HP, effective ability and move flags."""
    return ((substitute_hp > 0) & ~target_self & ~sound & ~bypass_substitute &
            (attacker_ability != AbilityEnum.INFILTRATOR))


def move_overrides(state, player, move):
    """Resolve represented move power, typing and grounded terrain priority."""
    mon = state.effective_active[player]
    ident = move.identifier
    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    grounded = state.is_grounded(player)
    power = jnp.where((ident == 512) & (mon.item == I.NONE), 110, move.base_power)

    power = jnp.where((ident == 707) & (state.volatile_status.previous_move_result[player] == MoveResult.FAILED), power * 2, power)

    target_item = state.active.item[1-player, 0]
    knock_boost = (ident == 282) & (target_item != I.NONE) & item_transfer_allowed(state.active.species[1-player, 0], state.active.form[1-player, 0], target_item)
    power = conditional_mult_round(power, 1.5, knock_boost)
    positive = jnp.sum(jnp.maximum(state.boosts.normal_boosts[player, 1:], 0)) + jnp.sum(jnp.maximum(state.boosts.acc_boosts[player], 0))
    power = jnp.where(ident == 500, 20 + 20 * positive, power)
    target = state.effective_active[1-player]
    weight = _move_overrides_weight
    target_weight = weight(target)
    low_kick = 20 + 20 * sum((target_weight >= threshold).astype(jnp.int32) for threshold in (100, 250, 500, 1000, 2000))
    heavy_slam = jnp.where(weight(mon) >= 5*target_weight, 120,
                 jnp.where(weight(mon) >= 4*target_weight, 100,
                 jnp.where(weight(mon) >= 3*target_weight, 80,
                 jnp.where(weight(mon) >= 2*target_weight, 60, 40))))
    power = jnp.where(ident == 67, low_kick, power)
    power = jnp.where(ident == 484, heavy_slam, power)
    weather_ball = (ident == 311) & (weather != W.NONE)
    move_type = jnp.where(weather_ball & (weather == W.SUN), Type.FIRE, move.type)
    move_type = jnp.where(weather_ball & (weather == W.RAIN), Type.WATER, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SANDSTORM), Type.ROCK, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SNOW), Type.ICE, move_type)
    # Ivy Cudgel follows the user's current form, including when copied.
    ivy_type = jnp.where(mon.form == FormEnum.OGERPON_WELLSPRING, Type.WATER,
               jnp.where(mon.form == FormEnum.OGERPON_HEARTHFLAME, Type.FIRE,
               jnp.where(mon.form == FormEnum.OGERPON_CORNERSTONE, Type.ROCK, move_type)))
    move_type = jnp.where(ident == 904, ivy_type, move_type)
    # Revelation Dance uses the first current type; ordinary Tera overrides it.
    first_type = jnp.where(mon.type_list[0] != Type.NONE, mon.type_list[0],
                           jnp.where(mon.type_list[1] != Type.NONE, mon.type_list[1], Type.NORMAL))
    dance_type = jnp.where(mon.is_terastallized[0] & (mon.tera_type[0] != Type.NONE), mon.tera_type[0], first_type)
    move_type = jnp.where(ident == 686, dance_type, move_type)
    power = jnp.where(weather_ball, 100, power)
    priority = jnp.where((ident == 803) & grounded & (state.terrain.terrain == T.GRASSY), 1, move.priority)
    return move.replace(base_power=power, type=move_type, priority=priority)
