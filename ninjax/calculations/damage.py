"""Pure intrinsic-power formulas and ordered battle damage calculations."""
import chex
import jax.lax
import jax.numpy as jnp

from ninjax.calculations.logic import triple_and, triple_or
from ninjax.calculations.modifiers import (
    apply_modifier,
    chain_modifier,
)
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier, type_effectiveness_factors
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Terrain, TerrainEnum, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveFlags, MoveType, PowerRule
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.side.queries import chain_defense_ability, is_floating, item_transfer_allowed


Array = chex.Array

def status_base_power(base_power, target_status, target_ability):
    """Resolve status-dependent base power multiplier for Hex and Infernal Parade"""
    has_status = (target_status != Status.NONE) | (target_ability == AbilityEnum.COMATOSE)
    return jnp.where(has_status, base_power * 2, base_power)

def proportional_hp_base_power(scale, current_hp, max_hp):
    """Resolve base power for moves that deal damage based on proportion of HP remaining"""
    return jnp.maximum(scale * current_hp // jnp.maximum(max_hp, 1), 1)


def target_hp_base_power(scale, current_hp, max_hp):
    """Resolve base power for moves that deal damage based on the proportion of opponent's hp remaining"""
    hp_ratio = (current_hp * 4096) // jnp.maximum(max_hp, 1)
    scaled = (scale * 100 * hp_ratio + 2047) // 4096
    return jnp.maximum(scaled // 100, 1)


def electro_ball_base_power(user_speed, target_speed):
    """Resolve base power of Electro Ball"""
    ratio = user_speed // jnp.maximum(target_speed, 1)
    ratio = jnp.where(target_speed == 0, 0, ratio)
    return jnp.select((ratio >= 4, ratio >= 3, ratio >= 2, ratio >= 1),
                      (150, 120, 80, 60), default=40)


def gyro_ball_base_power(user_speed, target_speed):
    """Resolve capped integer Gyro Ball power with a safe zero-Speed fallback."""
    power = 25 * target_speed // jnp.maximum(user_speed, 1) + 1
    return jnp.where(user_speed == 0, 1, jnp.minimum(150, power))


def effective_weight(weight, ability):
    """Apply Heavy/Light Metal to integer hectograms for hit-time weight rules."""
    return jnp.maximum(1, weight * jnp.where(ability == AbilityEnum.HEAVY_METAL, 2, 1) //
                       jnp.where(ability == AbilityEnum.LIGHT_METAL, 2, 1))


def resolve_hit_base_power(base_power, rule, user_hp=1, user_max_hp=1, target_hp=1,
                           target_max_hp=1, target_status=Status.NONE, target_ability=AbilityEnum.NONE,
                           user_speed=1, target_speed=1, *, no_item=False,
                           previous_failed=False, positive_stages=0,
                           user_weight=1, target_weight=1,
                           terrain_type=TerrainEnum.NONE, grounded_target=False):
    """Resolve all represented hit-time intrinsic callbacks before modifiers.

    Weather Ball's ModifyMove power and hit-plan contributions are already
    supplied in base_power. Modifier-only rules leave that input unchanged.
    All arithmetic uses safe denominators for traced, unselected branches.
    """
    target_weight = jnp.maximum(1, target_weight)
    weight_power = 20 + 20 * sum((target_weight >= threshold).astype(jnp.int32)
                               for threshold in (100, 250, 500, 1000, 2000))
    ratio_power = jnp.where(user_weight >= 5 * target_weight, 120,
                  jnp.where(user_weight >= 4 * target_weight, 100,
                  jnp.where(user_weight >= 3 * target_weight, 80,
                  jnp.where(user_weight >= 2 * target_weight, 60, 40))))
    return jnp.select(
        (rule == PowerRule.TARGET_STATUS, rule == PowerRule.USER_HP,
         rule == PowerRule.TARGET_HP_120, rule == PowerRule.TARGET_HP_100,
         rule == PowerRule.ELECTRO_BALL, rule == PowerRule.GYRO_BALL,
         (rule == PowerRule.DOUBLE_WITHOUT_ITEM) & no_item,
         (rule == PowerRule.DOUBLE_AFTER_FAILURE) & previous_failed,
         rule == PowerRule.POSITIVE_STAGES, rule == PowerRule.TARGET_WEIGHT,
         rule == PowerRule.WEIGHT_RATIO,
         (rule == PowerRule.DOUBLE_ON_GROUNDED_ELECTRIC_TARGET) &
         (terrain_type == TerrainEnum.ELECTRIC) & grounded_target),
        (status_base_power(base_power, target_status, target_ability),
         proportional_hp_base_power(base_power, user_hp, user_max_hp),
         target_hp_base_power(120, target_hp, target_max_hp),
         target_hp_base_power(100, target_hp, target_max_hp),
         electro_ball_base_power(user_speed, target_speed),
         gyro_ball_base_power(user_speed, target_speed), base_power * 2,
         base_power * 2, base_power + 20 * positive_stages, weight_power,
         ratio_power, base_power * 2), default=base_power).astype(jnp.int32)


def compute_base_power(
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        terrain: Terrain, weather=WeatherEnum.NONE, gravity_active=False, *, raw_effectiveness=None,
        user_speed=1, target_speed=1, positive_stages=0, previous_failed=False,
        target_charging_move=0) -> chex.Array:
    """Resolve live intrinsic power, then apply ordered power modifiers.

    BasePower event factors are chained as 4096-based integers in descending
    Showdown priority, then applied once with half-down rounding.
    Property-time Weather Ball power and fixed hit-plan powers pass through.
    HP, status and speed callbacks use this hit's effective snapshots before
    Technician. Callers must supply effective user/target speeds for speed rules,
    independently of critical-hit and Unaware damage-stat adjustments.
    """
    if raw_effectiveness is None:
        raw_effectiveness = calculate_effectiveness_multiplier(move.type, defender.type_list)
    grounded_user = ~is_floating(attacker) | jnp.asarray(gravity_active)
    grounded_target = ~is_floating(defender) | jnp.asarray(gravity_active)
    # Rising Voltage is a basePowerCallback, before Technician's power check.
    power = resolve_hit_base_power(move.base_power, move.power_rule, attacker.current_hp,
        attacker.max_hp, defender.current_hp, defender.max_hp, defender.status,
        defender.ability, user_speed, target_speed,
        no_item=attacker.item == ItemEnum.NONE, previous_failed=previous_failed,
        positive_stages=positive_stages,
        user_weight=effective_weight(attacker.weight_hg, attacker.ability),
        target_weight=effective_weight(defender.weight_hg, defender.ability),
        terrain_type=terrain.terrain, grounded_target=grounded_target)
    # Supported effects have a fixed priority sequence. Equal-priority attacker
    # abilities are mutually exclusive; matching auras contribute only once.
    modifier = jnp.ones_like(power, dtype=jnp.int32) * 4096
    modifier = chain_attacker_power_modifiers(modifier, power, move, attacker.ability,
        weather)
    modifier = chain_aura_power_modifiers(modifier, move.type, attacker.ability, defender.ability)
    modifier = chain_status_and_defender_power_modifiers(modifier, move, attacker.ability,
        defender.ability, attacker.is_poisoned, attacker.status)
    modifier = chain_item_power_modifiers(modifier, move.type, attacker.item, attacker.form, attacker.species)
    # Punk Rock priority 7 precedes terrain priority 6 and move priority 0.
    modifier = chain_modifier(modifier, 5325, (attacker.ability == AbilityEnum.PUNK_ROCK) & move.sound)
    modifier = chain_terrain_power_modifiers(modifier, move, terrain.terrain, grounded_user, grounded_target)
    # Solar moves are halved by rain, sand or snow at the BasePower stage.
    solar = (move.identifier == 76) | (move.identifier == 669)
    modifier = chain_modifier(modifier, 2048, solar & ((weather == WeatherEnum.RAIN) | (weather == WeatherEnum.SANDSTORM) | (weather == WeatherEnum.SNOW)))
    modifier = chain_move_power_modifiers(modifier, move, terrain.terrain, grounded_user, raw_effectiveness, defender, attacker.status)
    # Bounce doubles Gust/Twister at BasePower; Fly does so later at damage.
    modifier = chain_modifier(modifier, 8192, (target_charging_move == 340) & ((move.identifier == 16) | (move.identifier == 239)))
    return jnp.maximum(1, apply_modifier(power, modifier))


def chain_attacker_power_modifiers(modifier, power, move, ability, weather):
    """Chain attacker effects at priorities 30, 23 and 21, before auras."""
    modifier = chain_modifier(modifier, 6144,
        (ability == AbilityEnum.TECHNICIAN) & (apply_modifier(power, modifier) <= 60))
    # Priority 23: move-property abilities and type-conversion boosts.
    property_boost = (((ability == AbilityEnum.IRON_FIST) & move.punching) |
        ((ability == AbilityEnum.RECKLESS) & (move.recoil | (move.identifier == 136))) |
        move.type_power_boost)
    modifier = chain_modifier(modifier, 4915, property_boost)
    # Priority 21; these abilities are mutually exclusive on the user.
    sand = (ability == AbilityEnum.SAND_FORCE) & (weather == WeatherEnum.SANDSTORM) & (
        (move.type == Type.ROCK) | (move.type == Type.GROUND) | (move.type == Type.STEEL))
    return chain_modifier(modifier, 5325,
        sand | ((ability == AbilityEnum.TOUGH_CLAWS) & move.contact))


def chain_aura_power_modifiers(modifier, move_type, attacker_ability, defender_ability):
    """Chain one matching aura at priority 20, including Aura Break inversion."""
    aura = ((move_type == Type.FAIRY) & ((attacker_ability == AbilityEnum.FAIRY_AURA) |
        (defender_ability == AbilityEnum.FAIRY_AURA))) | (
        (move_type == Type.DARK) & ((attacker_ability == AbilityEnum.DARK_AURA) |
        (defender_ability == AbilityEnum.DARK_AURA)))
    broken = (attacker_ability == AbilityEnum.AURA_BREAK) | (defender_ability == AbilityEnum.AURA_BREAK)
    return chain_modifier(modifier, jnp.where(broken, 3072, 5448), aura)


def chain_status_and_defender_power_modifiers(modifier, move, ability, defender_ability, poisoned, status):
    """Chain priority-19 attacker effects followed by priority-17 Dry Skin."""
    boosted = (((ability == AbilityEnum.TOXIC_BOOST) & poisoned & (move.move_type == MoveType.PHYSICAL)) |
        ((ability == AbilityEnum.FLARE_BOOST) & (status == Status.BURN) & (move.move_type == MoveType.SPECIAL)) |
        ((ability == AbilityEnum.STRONG_JAW) & move.biting) |
        ((ability == AbilityEnum.MEGA_LAUNCHER) & move.launcher) |
        ((ability == AbilityEnum.SHARPNESS) & ((move.move_flags & MoveFlags.SLICING) != 0)))
    modifier = chain_modifier(modifier, 6144, boosted)
    # Fire vulnerability applies in all weather; weather HP effects are residuals.
    return chain_modifier(modifier, 5120,
        (defender_ability == AbilityEnum.DRY_SKIN) & (move.type == Type.FIRE))


def chain_item_power_modifiers(modifier, move_type, item, form, species):
    """Chain represented held-item power bonuses at priority 15."""
    mask = (((item == ItemEnum.WELLSPRING_MASK) & (form == FormEnum.OGERPON_WELLSPRING)) |
        ((item == ItemEnum.HEARTHFLAME_MASK) & (form == FormEnum.OGERPON_HEARTHFLAME)) |
        ((item == ItemEnum.CORNERSTONE_MASK) & (form == FormEnum.OGERPON_CORNERSTONE)))
    type_boost = (((item == ItemEnum.MAGNET) & (move_type == Type.ELECTRIC)) |
        ((item == ItemEnum.SILK_SCARF) & (move_type == Type.NORMAL)) |
        ((item == ItemEnum.SOUL_DEW) & ((species == 380) | (species == 381)) &
            ((move_type == Type.DRAGON) | (move_type == Type.PSYCHIC))) |
        ((item == ItemEnum.LUSTROUS_ORB) & (species == 484) &
            ((move_type == Type.DRAGON) | (move_type == Type.WATER))))
    return chain_modifier(modifier, 4915, mask | type_boost)


def chain_terrain_power_modifiers(modifier, move, terrain_type, grounded_user, grounded_target):
    """Chain priority-6 terrain effects, leaving intrinsic callbacks separate."""
    bonus = grounded_user & (((move.type == Type.GRASS) & (terrain_type == TerrainEnum.GRASSY)) |
        ((move.type == Type.PSYCHIC) & (terrain_type == TerrainEnum.PSYCHIC)) |
        ((move.type == Type.ELECTRIC) & (terrain_type == TerrainEnum.ELECTRIC)))
    modifier = chain_modifier(modifier, 5325, bonus)
    weakened = grounded_target & (((terrain_type == TerrainEnum.GRASSY) & (move.identifier == 89)) |
        ((terrain_type == TerrainEnum.MISTY) & (move.type == Type.DRAGON)))
    return chain_modifier(modifier, 2048, weakened)


def chain_move_power_modifiers(modifier, move, terrain_type, grounded_user, effectiveness, defender, user_status=Status.NONE):
    """Chain represented priority-zero move effects without rounding power early."""
    modifier = chain_modifier(modifier, 6144,
        (move.identifier == 797) & (terrain_type == TerrainEnum.PSYCHIC) & grounded_user)
    modifier = chain_conditional_move_power_modifiers(modifier, move.power_rule,
        user_status, defender.status, defender.current_hp, defender.max_hp)
    modifier = chain_field_matchup_move_modifiers(modifier, move.power_rule, terrain_type, effectiveness)
    eligible = (defender.item != ItemEnum.NONE) & item_transfer_allowed(defender.species, defender.form, defender.item)
    return chain_modifier(modifier, 6144, (move.power_rule == PowerRule.REMOVABLE_ITEM_BOOST) & eligible)


def chain_conditional_move_power_modifiers(modifier, rule, user_status, target_status, target_hp, target_max_hp):
    """Chain Facade, poison-target moves and Brine at priority zero."""
    eligible = ((rule == PowerRule.USER_STATUS_MODIFIER) &
        (user_status != Status.NONE) & (user_status != Status.SLEEP)) | (
        (rule == PowerRule.TARGET_POISON_MODIFIER) &
        ((target_status == Status.POISON) | (target_status == Status.TOXIC))) | (
        (rule == PowerRule.TARGET_HALF_HP_MODIFIER) & (2 * target_hp <= target_max_hp))
    return chain_modifier(modifier, 8192, eligible)


def chain_field_matchup_move_modifiers(modifier, rule, terrain_type, effectiveness):
    """Chain Psyblade's terrain bonus and the shared super-effective move bonus."""
    modifier = chain_modifier(modifier, 6144,
        (rule == PowerRule.ELECTRIC_TERRAIN_MODIFIER) & (terrain_type == TerrainEnum.ELECTRIC))
    return chain_modifier(modifier, 5461,
        (rule == PowerRule.SUPER_EFFECTIVE_MODIFIER) & (effectiveness > 1))


def compute_base_damage(
        attacker_ability: AbilityEnum,
        defender_ability: AbilityEnum,
        hp_percent,
        level,
        attacker_stats,
        defender_stats,
        move: Move,
        power,
        attacker_status,
        defender_status,
        flash_fire_boosted=False,
        *, offensive_modifier=4096, defensive_modifier=4096,
) -> Array:
    """Select the stage-adjusted stats and apply each stat's complete chain."""
    offense = _chain_offensive_ability(offensive_modifier, attacker_ability,
        defender_ability, move, hp_percent, attacker_status, flash_fire_boosted)
    defense = chain_defense_ability(defensive_modifier, defender_ability,
                                    defender_status, move.defensive_stat)
    sword = (attacker_ability == AbilityEnum.SWORD_OF_RUIN) & (defender_ability != AbilityEnum.SWORD_OF_RUIN)
    beads = (attacker_ability == AbilityEnum.BEADS_OF_RUIN) & (defender_ability != AbilityEnum.BEADS_OF_RUIN)
    defense = chain_modifier(defense, 3072,
        (sword & (move.defensive_stat == StatEnum.DEFENSE)) |
        (beads & (move.defensive_stat == StatEnum.SPECIAL_DEFENSE)))

    tablets = (defender_ability == AbilityEnum.TABLETS_OF_RUIN) & (attacker_ability != AbilityEnum.TABLETS_OF_RUIN)
    vessel = (defender_ability == AbilityEnum.VESSEL_OF_RUIN) & (attacker_ability != AbilityEnum.VESSEL_OF_RUIN)
    # Body Press reads Defense for offense, so an opposing Sword of Ruin applies.
    body_press = ((defender_ability == AbilityEnum.SWORD_OF_RUIN) &
                  (attacker_ability != AbilityEnum.SWORD_OF_RUIN) &
                  (move.offensive_stat == StatEnum.DEFENSE))
    offense = chain_modifier(offense, 3072,
        (tablets & (move.move_type == MoveType.PHYSICAL)) |
        (vessel & (move.move_type == MoveType.SPECIAL)) | body_press)
    offensive_stat = jnp.maximum(1, apply_modifier(attacker_stats[move.offensive_stat], offense))
    defensive_stat = jnp.maximum(1, apply_modifier(defender_stats[move.defensive_stat], defense))

    level_factor = jnp.floor(2 * level / 5) + 2
    base_damage = jnp.floor(jnp.floor(level_factor * power * offensive_stat / defensive_stat) / 50) + 2
    return base_damage


def hit_type_effectiveness(move_type, defender_types, defender_ability, gravity_active):
    """Return the raw item matchup and the damage matchup from one type lookup."""
    factors = type_effectiveness_factors(move_type, defender_types)
    raw = jnp.prod(factors)
    # Gravity neutralizes Flying's type factor and suppresses Levitate.
    gravity = jnp.asarray(gravity_active)
    ground = move_type == Type.GROUND
    damage_factors = jnp.where(gravity & ground & (defender_types == Type.FLYING), 1, factors)
    levitate = (defender_ability == AbilityEnum.LEVITATE) & ground & ~gravity
    return raw, jnp.prod(damage_factors) * (1 - levitate)


def apply_weather_damage(damage, move_type, weather):
    """Apply the weather event before critical hits and the random roll."""
    sun = weather == WeatherEnum.SUN
    rain = weather == WeatherEnum.RAIN
    boosted = (sun & (move_type == Type.FIRE)) | (rain & (move_type == Type.WATER))
    weakened = (sun & (move_type == Type.WATER)) | (rain & (move_type == Type.FIRE))
    return apply_modifier(damage, jnp.where(boosted, 6144, jnp.where(weakened, 2048, 4096)))


def stab_damage_modifier(types, tera_type, terastallized, ability, move_type):
    """Return the represented ordinary and matching-Tera STAB factor."""
    tera = terastallized & (tera_type == move_type)
    matching_tera = tera & jnp.any(types == tera_type)
    stab = (jnp.any(types == move_type) | tera) & (move_type != Type.NONE)
    adaptability = (ability == AbilityEnum.ADAPTABILITY) & stab
    modifier = 6144 + 2048 * (matching_tera | adaptability) + 1024 * (matching_tera & adaptability)
    return jnp.where(stab, modifier, 4096)


def apply_burn_penalty(damage, status, ability, category, identifier):
    """Halve physical damage after effectiveness, except Guts and Gen 6+ Facade."""
    burned = ((status == Status.BURN) & (category == MoveType.PHYSICAL) &
              (ability != AbilityEnum.GUTS) & (identifier != 263))
    return apply_modifier(damage, jnp.where(burned, 2048, 4096))


def attacker_damage_modifier(ability, effectiveness, is_crit):
    """Return the user's represented priority-zero ModifyDamage ability factor."""
    return jnp.select(
        ((ability == AbilityEnum.SNIPER) & is_crit,
         (ability == AbilityEnum.TINTED_LENS) & (effectiveness > 0) & (effectiveness < 1),
         (ability == AbilityEnum.NEUROFORCE) & (effectiveness > 1)),
        (6144, 8192, 5120), default=4096)


def defender_damage_modifier(ability, move_type, category, contact, sound, effectiveness, full_hp):
    """Return one defensive ability factor, combining Fluffy's two conditions."""
    fluffy = jnp.where(move_type == Type.FIRE, 8192, 4096) // jnp.where(contact, 2, 1)
    return jnp.select(
        ((ability == AbilityEnum.FILTER) & (effectiveness > 1),
         ability == AbilityEnum.FLUFFY,
         ((ability == AbilityEnum.PUNK_ROCK) & sound) |
         ((ability == AbilityEnum.ICE_SCALES) & (category == MoveType.SPECIAL)) |
         ((ability == AbilityEnum.MULTISCALE) & full_hp)),
        (3072, fluffy, 2048), default=4096)


def final_damage_modifier(user_ability, target_ability, item, move, effectiveness,
                          is_crit, full_hp, screen, user_speed, target_speed, semi_damage_boost=False):
    """Chain singles ModifyDamage effects by Speed, then effect type.

    Represented callbacks have priority zero. Faster holders execute first;
    tied abilities precede items. The two ability factors commute when tied,
    so their tie needs no extra random draw. Side conditions have zero Speed
    and follow Pokémon callbacks. Damage is rounded only after this chain.
    """
    user = attacker_damage_modifier(user_ability, effectiveness, is_crit)
    target = defender_damage_modifier(target_ability, move.type, move.move_type,
        move.contact, move.sound, effectiveness, full_hp)
    # A target's Fly/Dig/Dive callback follows its ability at the same speed.
    target = chain_modifier(target, 8192, semi_damage_boost)
    held = jnp.where(item == ItemEnum.LIFE_ORB, 5324,
        jnp.where((item == ItemEnum.EXPERT_BELT) & (effectiveness > 1), 4915, 4096))
    target_first = target_speed > user_speed
    user_first = user_speed > target_speed
    # Faster user: user ability/item, then target ability. Faster target:
    # target ability, then user ability/item. Ties put both abilities first.
    factors = (jnp.where(target_first, target, user),
               jnp.where(target_first, user, jnp.where(user_first, held, target)),
               jnp.where(user_first, target, held))
    modifier = jnp.ones_like(user, dtype=jnp.int32) * 4096
    for factor in factors:
        modifier = chain_modifier(modifier, factor)
    protected = screen & jnp.logical_not(is_crit) & (user_ability != AbilityEnum.INFILTRATOR)
    return chain_modifier(modifier, 2048, protected)


def compute_damage_multipliers(attacker, defender, weather, move, base_damage,
                               is_crit, damage_roll, gravity_active=False, *,
                               effectiveness=None, screen=False,
                               user_speed=1, target_speed=1, semi_damage_boost=False) -> Array:
    """Apply weather, crit, random, STAB, types, burn and final modifiers.

    Critical hits, random rolls and type resistance truncate independently.
    Weather, STAB and burn use half-down rounding. Final ability, item and
    screen factors form one fixed-point chain applied once. Immunity remains
    zero; otherwise ordinary Gen 9 damage is at least one before 16-bit wrap.
    PRNG sampling belongs to the caller; damage_roll is an explicit input.
    """
    if effectiveness is None:
        _, effectiveness = hit_type_effectiveness(move.type, defender.type_list, defender.ability, gravity_active)
    damage = apply_weather_damage(base_damage, move.type, weather)
    damage = jnp.where(is_crit, damage * 3 // 2, damage)
    damage = jnp.floor(damage * damage_roll).astype(jnp.int32)
    damage = apply_modifier(damage, stab_damage_modifier(attacker.type_list,
        attacker.tera_type, attacker.is_terastallized, attacker.ability, move.type))
    damage = jnp.floor(damage * effectiveness).astype(jnp.int32)
    damage = apply_burn_penalty(damage, attacker.status, attacker.ability,
        move.move_type, move.identifier)
    modifier = final_damage_modifier(attacker.ability, defender.ability, attacker.item,
        move, effectiveness, is_crit, defender.current_hp >= defender.max_hp,
        screen, user_speed, target_speed, semi_damage_boost)
    damage = apply_modifier(damage, modifier)
    return jnp.where(effectiveness > 0, jnp.maximum(1, damage) & 65535, 0)


def _chain_offensive_ability(modifier, user, target, move, hp, status, flash_fire):
    """Add move-dependent attacking-stat factors before the stat is rounded."""
    physical = move.move_type == MoveType.PHYSICAL
    kind = move.type
    low_hp = (hp <= 1 / 3) & (
        ((user == AbilityEnum.OVERGROW) & (kind == Type.GRASS)) |
        ((user == AbilityEnum.BLAZE) & (kind == Type.FIRE)) |
        ((user == AbilityEnum.TORRENT) & (kind == Type.WATER)) |
        ((user == AbilityEnum.SWARM) & (kind == Type.BUG)))
    boosted = (low_hp | ((user == AbilityEnum.HUSTLE) & physical) |
        ((user == AbilityEnum.GUTS) & physical & (status != Status.NONE)) |
        ((user == AbilityEnum.STEELWORKER) & (kind == Type.STEEL)) |
        ((user == AbilityEnum.ROCKY_PAYLOAD) & (kind == Type.ROCK)) |
        ((user == AbilityEnum.DRAGONS_MAW) & (kind == Type.DRAGON)) |
        ((user == AbilityEnum.FLASH_FIRE) & flash_fire & (kind == Type.FIRE)))
    modifier = chain_modifier(modifier, 6144, boosted)
    modifier = chain_modifier(modifier, 8192,
        ((user == AbilityEnum.HUGE_POWER) & physical) |
        ((user == AbilityEnum.WATER_BUBBLE) & (kind == Type.WATER)))
    # The defender's resistance abilities halve the attacking stat.
    resisted = (((target == AbilityEnum.WATER_BUBBLE) | (target == AbilityEnum.HEATPROOF)) & (kind == Type.FIRE)) | (
        (target == AbilityEnum.THICK_FAT) & ((kind == Type.FIRE) | (kind == Type.ICE))) | (
        (target == AbilityEnum.PURIFYING_SALT) & (kind == Type.GHOST))
    modifier = chain_modifier(modifier, 2048, resisted)
    modifier = chain_modifier(modifier, 2048, (user == AbilityEnum.DEFEATIST) & (hp <= .5))
    return chain_modifier(modifier, 5325, (user == AbilityEnum.TRANSISTOR) & (kind == Type.ELECTRIC))


def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    """Compute the unmodified level, power, attack and defense damage expression."""
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2
