"""Pure intrinsic-power formulas and ordered battle damage calculations."""
import chex
import jax.lax
import jax.numpy as jnp

from ninjax.calculations.logic import triple_and, triple_or
from ninjax.calculations.modifiers import (
    conditional_mult,
    conditional_mult_prod,
    conditional_mult_prod_round,
    conditional_mult_round,
    one_point_three,
)
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier, type_effectiveness_factors
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Terrain, TerrainEnum, Weather, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveFlags, MoveType, PowerRule
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.side.queries import conditional_mult_round as fixed_modifier, is_floating, item_transfer_allowed


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


def resolve_hit_base_power(base_power, rule, user_hp, user_max_hp, target_hp,
                           target_max_hp, target_status, target_ability,
                           user_speed, target_speed):
    """Resolve live intrinsic formulas before Technician and other modifiers.

    Previously resolved property-time rules and per-hit plan powers pass through.
    Numeric branches use safe denominators so unselected formulas remain finite.
    """
    return jnp.select(
        (rule == PowerRule.TARGET_STATUS, rule == PowerRule.USER_HP,
         rule == PowerRule.TARGET_HP_120, rule == PowerRule.TARGET_HP_100,
         rule == PowerRule.ELECTRO_BALL, rule == PowerRule.GYRO_BALL),
        (status_base_power(base_power, target_status, target_ability),
         proportional_hp_base_power(base_power, user_hp, user_max_hp),
         target_hp_base_power(120, target_hp, target_max_hp),
         target_hp_base_power(100, target_hp, target_max_hp),
         electro_ball_base_power(user_speed, target_speed),
         gyro_ball_base_power(user_speed, target_speed)), default=base_power).astype(jnp.int32)


def compute_base_power(
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        terrain: Terrain, weather=WeatherEnum.NONE, gravity_active=False, *, raw_effectiveness=None,
        user_speed=1, target_speed=1) -> chex.Array:
    """Resolve live intrinsic power, then apply ordered power modifiers.

    BasePower event factors are chained as 4096-based integers in descending
    Showdown priority, then applied once with half-down rounding.
    Existing property-time rules and hit-plan powers pass through unchanged.
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
        defender.ability, user_speed, target_speed)
    power = resolve_terrain_base_power(power, move.power_rule, terrain.terrain, grounded_target)
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
    modifier = chain_power_modifier(modifier, 5325, (attacker.ability == AbilityEnum.PUNK_ROCK) & move.sound)
    modifier = chain_terrain_power_modifiers(modifier, move, terrain.terrain, grounded_user, grounded_target)
    modifier = chain_move_power_modifiers(modifier, move, terrain.terrain, grounded_user, raw_effectiveness, defender, attacker.status)
    return jnp.maximum(1, apply_power_modifier(power, modifier))


def resolve_variable_base_power(base_power, rule, *, no_item=False, previous_failed=False,
                                positive_stages=0, user_weight=1, target_weight=1,
                                weather_active=False):
    """Resolve represented intrinsic formulas from explicit scalar/array inputs.

    Weights are effective integer hectograms. The caller supplies stage/history
    summaries and item eligibility; this pure function never reads battle state.
    Knock Off eligibility is evaluated at hit time by the modifier chain.
    Unknown rules preserve the supplied power, as does PowerRule.FIXED.
    """
    power = jnp.where((rule == PowerRule.DOUBLE_WITHOUT_ITEM) & no_item, base_power * 2, base_power)
    power = jnp.where((rule == PowerRule.DOUBLE_AFTER_FAILURE) & previous_failed, power * 2, power)
    power = jnp.where(rule == PowerRule.POSITIVE_STAGES, 20 + 20 * positive_stages, power)
    target_weight = jnp.maximum(1, target_weight)
    weight_power = 20 + 20 * sum((target_weight >= threshold).astype(jnp.int32)
                               for threshold in (100, 250, 500, 1000, 2000))
    ratio_power = jnp.where(user_weight >= 5 * target_weight, 120,
                  jnp.where(user_weight >= 4 * target_weight, 100,
                  jnp.where(user_weight >= 3 * target_weight, 80,
                  jnp.where(user_weight >= 2 * target_weight, 60, 40))))
    power = jnp.where(rule == PowerRule.TARGET_WEIGHT, weight_power, power)
    power = jnp.where(rule == PowerRule.WEIGHT_RATIO, ratio_power, power)
    power = jnp.where((rule == PowerRule.DOUBLE_IN_WEATHER) & weather_active, base_power * 2, power)
    # Knock Off is a BasePower event modifier, not an intrinsic formula.
    return power


def resolve_terrain_base_power(power, rule, terrain_type, grounded_target):
    """Double Rising Voltage's intrinsic power for a grounded Electric Terrain target.

    The user's grounding controls the separate ordinary terrain bonus. This
    callback precedes power modifiers, including Technician's eligibility check.
    """
    return jnp.where((rule == PowerRule.DOUBLE_ON_GROUNDED_ELECTRIC_TARGET) &
        (terrain_type == TerrainEnum.ELECTRIC) & grounded_target, power * 2, power)


def chain_power_modifier(modifier, numerator, enabled):
    """Chain a 4096-based modifier, rounding the chain half up like Showdown."""
    next_modifier = jnp.where(enabled, numerator, 4096).astype(jnp.int32)
    return (modifier * next_modifier + 2048) >> 12


def apply_power_modifier(power, modifier):
    """Apply the completed chain once, rounding base-power halves down."""
    return (jnp.asarray(power, jnp.int32) * modifier + 2047) >> 12


def chain_attacker_power_modifiers(modifier, power, move, ability, weather):
    """Chain attacker effects at priorities 30, 23 and 21, before auras."""
    modifier = chain_power_modifier(modifier, 6144,
        (ability == AbilityEnum.TECHNICIAN) & (apply_power_modifier(power, modifier) <= 60))
    # Priority 23: move-property abilities and type-conversion boosts.
    property_boost = (((ability == AbilityEnum.IRON_FIST) & move.punching) |
        ((ability == AbilityEnum.RECKLESS) & (move.recoil | (move.identifier == 136))) |
        move.type_power_boost)
    modifier = chain_power_modifier(modifier, 4915, property_boost)
    # Priority 21; these abilities are mutually exclusive on the user.
    sand = (ability == AbilityEnum.SAND_FORCE) & (weather == WeatherEnum.SANDSTORM) & (
        (move.type == Type.ROCK) | (move.type == Type.GROUND) | (move.type == Type.STEEL))
    return chain_power_modifier(modifier, 5325,
        sand | ((ability == AbilityEnum.TOUGH_CLAWS) & move.contact))


def chain_aura_power_modifiers(modifier, move_type, attacker_ability, defender_ability):
    """Chain one matching aura at priority 20, including Aura Break inversion."""
    aura = ((move_type == Type.FAIRY) & ((attacker_ability == AbilityEnum.FAIRY_AURA) |
        (defender_ability == AbilityEnum.FAIRY_AURA))) | (
        (move_type == Type.DARK) & ((attacker_ability == AbilityEnum.DARK_AURA) |
        (defender_ability == AbilityEnum.DARK_AURA)))
    broken = (attacker_ability == AbilityEnum.AURA_BREAK) | (defender_ability == AbilityEnum.AURA_BREAK)
    return chain_power_modifier(modifier, jnp.where(broken, 3072, 5448), aura)


def chain_status_and_defender_power_modifiers(modifier, move, ability, defender_ability, poisoned, status):
    """Chain priority-19 attacker effects followed by priority-17 Dry Skin."""
    boosted = (((ability == AbilityEnum.TOXIC_BOOST) & poisoned & (move.move_type == MoveType.PHYSICAL)) |
        ((ability == AbilityEnum.FLARE_BOOST) & (status == Status.BURN) & (move.move_type == MoveType.SPECIAL)) |
        ((ability == AbilityEnum.STRONG_JAW) & move.biting) |
        ((ability == AbilityEnum.MEGA_LAUNCHER) & move.launcher) |
        ((ability == AbilityEnum.SHARPNESS) & ((move.move_flags & MoveFlags.SLICING) != 0)))
    modifier = chain_power_modifier(modifier, 6144, boosted)
    # Fire vulnerability applies in all weather; weather HP effects are residuals.
    return chain_power_modifier(modifier, 5120,
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
    return chain_power_modifier(modifier, 4915, mask | type_boost)


def chain_terrain_power_modifiers(modifier, move, terrain_type, grounded_user, grounded_target):
    """Chain priority-6 terrain effects, leaving intrinsic callbacks separate."""
    bonus = grounded_user & (((move.type == Type.GRASS) & (terrain_type == TerrainEnum.GRASSY)) |
        ((move.type == Type.PSYCHIC) & (terrain_type == TerrainEnum.PSYCHIC)) |
        ((move.type == Type.ELECTRIC) & (terrain_type == TerrainEnum.ELECTRIC)))
    modifier = chain_power_modifier(modifier, 5325, bonus)
    weakened = grounded_target & (((terrain_type == TerrainEnum.GRASSY) & (move.identifier == 89)) |
        ((terrain_type == TerrainEnum.MISTY) & (move.type == Type.DRAGON)))
    return chain_power_modifier(modifier, 2048, weakened)


def chain_move_power_modifiers(modifier, move, terrain_type, grounded_user, effectiveness, defender, user_status=Status.NONE):
    """Chain represented priority-zero move effects without rounding power early."""
    modifier = chain_power_modifier(modifier, 6144,
        (move.identifier == 797) & (terrain_type == TerrainEnum.PSYCHIC) & grounded_user)
    modifier = chain_power_modifier(modifier, 5461, (move.identifier == 879) & (effectiveness > 1))
    modifier = chain_conditional_move_power_modifiers(modifier, move.power_rule,
        user_status, defender.status, defender.current_hp, defender.max_hp)
    modifier = chain_field_matchup_move_modifiers(modifier, move.power_rule, terrain_type, effectiveness)
    eligible = (defender.item != ItemEnum.NONE) & item_transfer_allowed(defender.species, defender.form, defender.item)
    return chain_power_modifier(modifier, 6144, (move.power_rule == PowerRule.REMOVABLE_ITEM_BOOST) & eligible)


def chain_conditional_move_power_modifiers(modifier, rule, user_status, target_status, target_hp, target_max_hp):
    """Chain Facade, poison-target moves and Brine at priority zero."""
    eligible = ((rule == PowerRule.USER_STATUS_MODIFIER) &
        (user_status != Status.NONE) & (user_status != Status.SLEEP)) | (
        (rule == PowerRule.TARGET_POISON_MODIFIER) &
        ((target_status == Status.POISON) | (target_status == Status.TOXIC))) | (
        (rule == PowerRule.TARGET_HALF_HP_MODIFIER) & (2 * target_hp <= target_max_hp))
    return chain_power_modifier(modifier, 8192, eligible)


def chain_field_matchup_move_modifiers(modifier, rule, terrain_type, effectiveness):
    """Chain Psyblade's terrain presence and Collision Course's matchup bonus."""
    modifier = chain_power_modifier(modifier, 6144,
        (rule == PowerRule.ELECTRIC_TERRAIN_MODIFIER) & (terrain_type == TerrainEnum.ELECTRIC))
    return chain_power_modifier(modifier, 5461,
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
        flash_fire_boosted=False
) -> Array:
    """Compute base damage using selected stats and represented Ruin modifiers."""
    offensive_stat = compute_offensive_stat(
        attacker_stats[move.offensive_stat],
        attacker_ability,
        defender_ability,
        move,
        hp_percent,
        attacker_status
    )

    offensive_stat = conditional_mult_round(offensive_stat, 1.5,
        triple_and(jnp.asarray(flash_fire_boosted), attacker_ability == AbilityEnum.FLASH_FIRE, move.type == Type.FIRE))
    defensive_stat = compute_defensive_stat(defender_stats, defender_ability, defender_status, move.defensive_stat)

    # ruin abilities
    sword_of_ruin = (attacker_ability==AbilityEnum.SWORD_OF_RUIN) & (defender_ability != AbilityEnum.SWORD_OF_RUIN)
    beads_of_ruin = (attacker_ability == AbilityEnum.BEADS_OF_RUIN) & (defender_ability != AbilityEnum.BEADS_OF_RUIN)
    defensive_stat = fixed_modifier(
        defensive_stat,
        0.75,
        jnp.logical_or(
            jnp.logical_and(sword_of_ruin, move.defensive_stat==StatEnum.DEFENSE),
            jnp.logical_and(beads_of_ruin, move.defensive_stat==StatEnum.SPECIAL_DEFENSE)
        ))
    # i hate body press so much why is it so stupid
    vessel_of_ruin = (defender_ability==AbilityEnum.VESSEL_OF_RUIN) & (attacker_ability != AbilityEnum.VESSEL_OF_RUIN)
    tablets_of_ruin = (defender_ability==AbilityEnum.TABLETS_OF_RUIN) & (attacker_ability != AbilityEnum.TABLETS_OF_RUIN)
    offensive_stat = fixed_modifier(
        offensive_stat,
        0.75,
        jnp.logical_or(
            jnp.logical_and(tablets_of_ruin, move.move_type==MoveType.PHYSICAL),
            jnp.logical_and(vessel_of_ruin, move.move_type==MoveType.SPECIAL)
        )
    )


    # A defending Sword of Ruin also lowers the attacker's Defense when
    # Body Press selects Defense as its offensive stat. Holders are exempt.
    offensive_stat = fixed_modifier(offensive_stat, .75,
        (defender_ability == AbilityEnum.SWORD_OF_RUIN) &
        (attacker_ability != AbilityEnum.SWORD_OF_RUIN) &
        (move.offensive_stat == StatEnum.DEFENSE))

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


def compute_damage_multipliers(
        key: chex.PRNGKey,
        attacker,
        defender,
        weather: Weather,
        move: Move,
        base_damage,
        is_crit,
        damage_roll,
        gravity_active=False,
        *, effectiveness=None,
) -> Array:
    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way

    # sun modifier
    """Compute damage multipliers."""
    is_sun = weather.weather == WeatherEnum.SUN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_sun, move.type == Type.FIRE))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_sun, move.type == Type.WATER))
    # rain modifier
    is_rain = weather.weather == WeatherEnum.RAIN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_rain, move.type == Type.WATER))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_rain, move.type == Type.FIRE))

    crit_multiplier = 1.5
    # damage roll, idc about preserving the in game RNG generation
    base_damage = conditional_mult_round(base_damage, crit_multiplier, is_crit)
    base_damage = conditional_mult_round(base_damage, damage_roll, 1)
    # stab multiplier
    # TODO: this logic can probably be simplified
    is_tera_boosted = jnp.logical_and(attacker.is_terastallized, attacker.tera_type == move.type)
    is_matching_tera = jnp.logical_and(is_tera_boosted, jnp.any(attacker.type_list == attacker.tera_type))
    is_stab = jnp.logical_or(jnp.any(attacker.type_list == move.type), is_tera_boosted) & (move.type != Type.NONE)
    is_adaptability_boosted = jnp.logical_and(attacker.ability==AbilityEnum.ADAPTABILITY, is_stab)
    stab_multiplier = (
            1.5 +
            0.5 * jnp.logical_or(is_matching_tera, is_adaptability_boosted) +
            0.25 * jnp.logical_and(is_matching_tera, is_adaptability_boosted))
    base_damage = conditional_mult_round(base_damage, stab_multiplier, is_stab)
    if effectiveness is None:
        _, effectiveness = hit_type_effectiveness(move.type, defender.type_list, defender.ability, gravity_active)
    is_tinted_lens = (effectiveness > 0) & (effectiveness < 1) & (attacker.ability==AbilityEnum.TINTED_LENS)
    is_filter = (effectiveness > 1) & (defender.ability==AbilityEnum.FILTER)
    is_neuroforce = (effectiveness > 1) & (attacker.ability==AbilityEnum.NEUROFORCE)
    base_damage = jnp.trunc(base_damage * effectiveness)
    # burn
    is_burned = attacker.status == Status.BURN
    is_physical = move.move_type == MoveType.PHYSICAL
    is_guts = attacker.ability == AbilityEnum.GUTS
    base_damage = conditional_mult_round(base_damage, 0.5, triple_and(1-is_guts, is_physical, is_burned) & (move.identifier != 263))
    # Sniper is a final damage modifier, after the ordinary critical and random
    # rolls. Folding it into the critical multiplier changes integer rounding.
    base_damage = conditional_mult_round(base_damage, 1.5, is_crit & (attacker.ability == AbilityEnum.SNIPER))
    base_damage = conditional_mult_round(base_damage, 2, is_tinted_lens)
    base_damage = conditional_mult_round(base_damage, 0.75, is_filter)
    base_damage = conditional_mult_round(base_damage, 1.25, is_neuroforce)

    return base_damage

def damage_post_modifiers(
        damage: Array,
        defender_ability: AbilityEnum,
        move: Move) -> Array:
    """Damage post modifiers."""
    damage = damage.astype(int)
    # so these might function the same as water bubble defensively which modifies the attackers attack but idk for sure
    is_fire_move = move.type == Type.FIRE
    # fluffy is the same
    is_fluffy = defender_ability == AbilityEnum.FLUFFY
    fluffy_increase = jnp.logical_and(is_fire_move, is_fluffy)
    fluffy_decrease = jnp.logical_and(move.contact, is_fluffy)
    # punk rock
    is_punk_rock = defender_ability == AbilityEnum.PUNK_ROCK
    damage = conditional_mult_round(damage, 1 / 2, jnp.logical_and(is_punk_rock, move.sound))
    # ice scales
    is_ice_scales = defender_ability == AbilityEnum.ICE_SCALES
    damage = conditional_mult_round(damage, 1 / 2, jnp.logical_and(is_ice_scales, move.move_type == MoveType.SPECIAL))
    damage = conditional_mult_prod_round(
        damage,
        jnp.array([2, 1 / 2]),
        jnp.array([fluffy_increase, fluffy_decrease]).squeeze())
    return damage

def compute_offensive_stat(
        starting_stat: Array,
        attacker_ability: AbilityEnum,
        defender_ability: AbilityEnum,
        move,
        hp_percent: Array,
        status: Status):

    """Apply represented offensive and opposing attack-stat modifiers."""
    # I'm copying this implementation from what i can find of DaWoblefet's damage dissertation
    # ignoring the rounding

    is_physical = move.move_type == MoveType.PHYSICAL
    is_special = move.move_type == MoveType.SPECIAL
    hp_under_half = jnp.less_equal(hp_percent, 0.5)
    hp_low = jnp.less_equal(hp_percent, 1/3)
    move_type = move.type

    # hustle: for whatever reason this is separate from anything else
    stat = conditional_mult_prod(
        starting_stat,
        1.5,
        jnp.logical_and(attacker_ability==AbilityEnum.HUSTLE, is_physical)
    )

    # 0.5x abilities
    # TODO: slow start
    stat = conditional_mult_prod(stat, 0.5, jnp.logical_and(attacker_ability==AbilityEnum.DEFEATIST, hp_under_half))

    # 1.5x abilities
    is_guts = jnp.logical_and(status!=Status.NONE, jnp.logical_and(attacker_ability==AbilityEnum.GUTS, is_physical))
    is_overgrow = triple_and(move_type==Type.GRASS, hp_low, attacker_ability==AbilityEnum.OVERGROW)
    is_blaze = triple_and(move_type==Type.FIRE, hp_low, attacker_ability==AbilityEnum.BLAZE)
    is_torrent = triple_and(move_type==Type.WATER, hp_low, attacker_ability==AbilityEnum.TORRENT)
    is_swarm = triple_and(move_type==Type.BUG, hp_low, attacker_ability==AbilityEnum.SWARM)
    is_steel_worker = jnp.logical_and(move_type==Type.STEEL, attacker_ability==AbilityEnum.STEELWORKER)
    is_rocky_payload = jnp.logical_and(move_type==Type.ROCK, attacker_ability==AbilityEnum.ROCKY_PAYLOAD)
    is_maw = jnp.logical_and(move_type==Type.DRAGON, attacker_ability==AbilityEnum.DRAGONS_MAW)

    arr = jnp.array([is_swarm, is_torrent, is_blaze, is_overgrow, is_steel_worker, is_rocky_payload, is_maw, is_guts])

    stat = conditional_mult(stat, 1.5, jnp.any(arr))

    # 2x abilities
    # TODO: stakeout
    huge_power = jnp.logical_and(attacker_ability==AbilityEnum.HUGE_POWER, is_physical)
    water_bubble = jnp.logical_and(attacker_ability==AbilityEnum.WATER_BUBBLE, move_type==Type.WATER)
    arr = jnp.array([huge_power, water_bubble])

    stat = conditional_mult(stat, 2, jnp.any(arr))

    # 0.5x defensive abilities
    water_bubble = jnp.logical_and(defender_ability==AbilityEnum.WATER_BUBBLE, move_type==Type.FIRE)
    thick_fat = jnp.logical_and(
        defender_ability==AbilityEnum.THICK_FAT,
        jnp.logical_or(move_type==Type.FIRE, move_type==Type.ICE)
    )

    heatproof = (defender_ability == AbilityEnum.HEATPROOF) & (move_type == Type.FIRE)
    salt = (defender_ability == AbilityEnum.PURIFYING_SALT) & (move_type == Type.GHOST)
    # Purifying Salt halves the attacking stat, not final damage; the +2 in
    # the base-damage formula makes these observably different at low powers.
    stat = conditional_mult(stat, 0.5, water_bubble | thick_fat | heatproof | salt)

    # Transistor was reduced from 1.5x in generation 8 to 1.3x in generation 9.
    is_transistor = jnp.logical_and(move_type==Type.ELECTRIC, attacker_ability==AbilityEnum.TRANSISTOR)
    stat = conditional_mult_round(stat, one_point_three, is_transistor)

    #choice items

    # pokemon specific boosting items, eg thick club or ogerpon mask

    return jnp.trunc(stat)

def compute_defensive_stat(starting_stats, defender_ability, status, stat_idx):
    """Apply status-dependent defense and Fur Coat to the selected defensive stat."""
    # marvel scale
    stat = conditional_mult_round(
        starting_stats[stat_idx], 1.5,
        triple_and(defender_ability == AbilityEnum.MARVEL_SCALE,
                   status!=Status.NONE, stat_idx == StatEnum.DEFENSE))

    # eviolite and assault vest
    return conditional_mult_round(stat, 2, (defender_ability == AbilityEnum.FUR_COAT) & (stat_idx == StatEnum.DEFENSE))


def base_damage_compute(
        attacker_level: int,
        attack_stat: int,
        defence_stat: int,
        base_power: int):
    """Compute the unmodified level, power, attack and defense damage expression."""
    return (2 * attacker_level / 5 + 2) * base_power * attack_stat / defence_stat / 50 + 2
