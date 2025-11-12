from typing import Tuple

import jax
import jax.random as random
import jax.numpy as jnp
import chex


from ninjax.utils import STAT_MULTIPLIER_LOOKUP, ACCURACY_MULTIPLIER_LOOKUP

from ninjax.enum_types import AbilityEnum, Status, Type, TerrainEnum, WeatherEnum, MoveType, Weather, Terrain, StatEnum
from ninjax.side import boosted_stats_helper
from ninjax.pokemon import Pokemon
from ninjax.move import Move
from ninjax.utils import (
    conditional_mult_round, TERRAIN_MULTIPLIER, TYPE_EFFECTIVENESS, CRIT_STAGES,calculate_effectiveness_multiplier,
    COMPOUND_EYES_MULTIPLIER, conditional_mult, WEATHER_VEIL_MODIFIER, triple_and, triple_or, quad_or, ROUGH_SKIN_DAMAGE,
    in_range, IRON_FIST, TOUGH_CLAWS, one_third, RECKLESS, VICTORY_STAR, four_thirds, quad_and,
    conditional_mult_prod_round, conditional_mult_prod, one_point_three)

jax.config.update("jax_disable_jit", True)

Array = chex.Array

def compute_base_power(
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        terrain: Terrain) -> chex.Array:
    power = move.base_power
    ability = attacker.ability
    #TODO: tera boost
    # TODO at some point test if this can be optimized by putting all the values in arrays and doing one conditional mult round

    # dark and fairy aura
    # both of these  apply to both sides
    fairy_aura = jnp.logical_and(move.type==Type.FAIRY,
                                 jnp.logical_or(ability==AbilityEnum.FAIRY_AURA, defender.ability==AbilityEnum.FAIRY_AURA))
    dark_aura = jnp.logical_and(move.type==Type.DARK,
                                 jnp.logical_or(ability==AbilityEnum.DARK_AURA, defender.ability==AbilityEnum.DARK_AURA))
    aura = jnp.logical_or(fairy_aura, dark_aura)
    aura_break = jnp.logical_or(attacker.ability==AbilityEnum.AURA_BREAK, defender.ability==AbilityEnum.AURA_BREAK)

    power = conditional_mult_round(power, four_thirds, jnp.logical_and(aura, 1-aura_break))
    power = conditional_mult_round(power, 3/4, jnp.logical_and(aura, aura_break))


    #technician
    is_technician = ability==AbilityEnum.TECHNICIAN
    is_technician_boosted = jnp.logical_and(
        is_technician,
        jnp.less_equal(power, 60)
    )
    power = conditional_mult(power, 1.5, is_technician_boosted)
    # toxic boost
    is_toxic_boosted = triple_and(
        ability==AbilityEnum.TOXIC_BOOST,
        attacker.is_poisoned,
        move.move_type==MoveType.PHYSICAL
    )
    power = conditional_mult(power, 1.5, is_toxic_boosted)
    # flare boost
    is_flare_boosted = triple_and(
        ability==AbilityEnum.FLARE_BOOST,
        attacker.status==Status.BURN,
        move.move_type==MoveType.SPECIAL
    )
    power = conditional_mult(power, 1.5, is_flare_boosted)

    is_grounded = 1 - attacker.is_floating
    t = terrain.terrain
    terrain_boosted = triple_or(
        jnp.logical_and(move.type==Type.GRASS, t==TerrainEnum.GRASSY),
        jnp.logical_and(move.type == Type.PSYCHIC, t == TerrainEnum.PSYCHIC),
        jnp.logical_and(move.type==Type.ELECTRIC, t==TerrainEnum.ELECTRIC),
    )
    terrain_boosted = jnp.logical_and(terrain_boosted, is_grounded)
    power = conditional_mult_round(power, TERRAIN_MULTIPLIER, terrain_boosted)

    ability = attacker.ability
    # TODO: similar to with the low health abilities, we could put the conditions in an array and put mults in array
    # that is probably faster than this
    # iron fist
    power = conditional_mult_round(power, IRON_FIST, jnp.logical_and(move.punching, ability==AbilityEnum.IRON_FIST))
    # tough claws
    power = conditional_mult_round(power, TOUGH_CLAWS, jnp.logical_and(move.contact, ability==AbilityEnum.TOUGH_CLAWS))
    # reckless
    power = conditional_mult_round(power, RECKLESS, jnp.logical_and(move.recoil, ability==AbilityEnum.RECKLESS))
    # strong jaw
    power = conditional_mult_round(power, 1.5, jnp.logical_and(move.biting, ability==AbilityEnum.STRONG_JAW))
    # mega launcher
    power = conditional_mult_round(power, 1.5, jnp.logical_and(move.launcher, ability==AbilityEnum.MEGA_LAUNCHER))
    # punk rock
    power = conditional_mult_round(power, 1.3, jnp.logical_and(move.sound, ability==AbilityEnum.PUNK_ROCK))
    return power

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
        defender_status
) -> Array:

    offensive_stat = compute_offensive_stat(
        attacker_stats[move.offensive_stat],
        attacker_ability,
        defender_ability,
        move,
        hp_percent,
        attacker_status
    )

    defensive_stat = compute_defensive_stat(defender_stats, defender_ability, defender_status, move.defensive_stat)
    # ruin abilities
    sword_of_ruin = attacker_ability==AbilityEnum.SWORD_OF_RUIN
    beads_of_ruin = attacker_ability == AbilityEnum.BEADS_OF_RUIN
    defensive_stat = conditional_mult_round(
        defensive_stat,
        0.75,
        jnp.logical_or(
            jnp.logical_and(sword_of_ruin, defensive_stat==StatEnum.DEFENSE),
            jnp.logical_and(beads_of_ruin, defensive_stat==StatEnum.SPECIAL_DEFENSE)
        ))
    # i hate body press so much why is it so stupid
    vessel_of_ruin = defender_ability==AbilityEnum.VESSEL_OF_RUIN
    tablets_of_ruin = defender_ability==AbilityEnum.TABLETS_OF_RUIN
    offensive_stat = conditional_mult_round(
        offensive_stat,
        0.75,
        jnp.logical_or(
            jnp.logical_and(tablets_of_ruin, move.move_type==MoveType.PHYSICAL),
            jnp.logical_and(vessel_of_ruin, move.move_type==MoveType.SPECIAL)
        )
    )


    base_damage = jnp.floor(((2 * level / 5 + 2) * power * offensive_stat) / (defensive_stat * 50) + 2)
    return base_damage


def compute_damage_multipliers(
        key: chex.PRNGKey,
        attacker,
        defender,
        weather: Weather,
        move: Move,
        base_damage,
        is_crit,
        damage_roll
) -> Array:
    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way

    # sun modifier
    is_sun = weather.weather == WeatherEnum.SUN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_sun, move.type == Type.FIRE))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_sun, move.type == Type.WATER))
    # rain modifier
    is_rain = weather.weather == WeatherEnum.RAIN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_rain, move.type == Type.WATER))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_rain, move.type == Type.FIRE))

    crit_multiplier = 1.5 + 0.75 * (attacker.ability==AbilityEnum.SNIPER)[0]
    # damage roll, idc about preserving the in game RNG generation
    base_damage = conditional_mult_round(base_damage, crit_multiplier, is_crit)
    base_damage = conditional_mult_round(base_damage, damage_roll, 1)
    # stab multiplier
    # TODO: this logic can probably be simplified
    is_tera_boosted = jnp.logical_and(attacker.is_terastallized, attacker.tera_type == move.type)
    is_matching_tera = jnp.logical_and(is_tera_boosted, jnp.any(attacker.type_list == attacker.tera_type))
    is_stab = jnp.logical_or(jnp.any(attacker.type_list == move.type), is_tera_boosted)
    is_adaptability_boosted = jnp.logical_and(attacker.ability==AbilityEnum.ADAPTABILITY, is_stab)
    stab_multiplier = (
            1.5 +
            0.5 * jnp.logical_or(is_matching_tera, is_adaptability_boosted) +
            0.25 * jnp.logical_and(is_matching_tera, is_adaptability_boosted))
    base_damage = conditional_mult_round(base_damage, stab_multiplier, is_stab)
    # Type effectiveness, when we get around to implementing observations
    # it should include does not affect, not very effective, or super effective
    effectiveness = calculate_effectiveness_multiplier(move.type, defender.type_list)
    is_levitate = defender.ability==AbilityEnum.LEVITATE
    effectiveness = effectiveness * 1-jnp.logical_and(is_levitate, move.type==Type.GROUND)
    is_tinted_lens = jnp.logical_and(jnp.less_equal(effectiveness, 1), attacker.ability==AbilityEnum.TINTED_LENS)
    is_filter = jnp.logical_and(jnp.greater_equal(effectiveness, 1), defender.ability==AbilityEnum.FILTER)
    is_neuroforce = jnp.logical_and(jnp.greater_equal(effectiveness, 1), defender.ability==AbilityEnum.NEUROFORCE)
    effectiveness = conditional_mult_prod(
        effectiveness,
        jnp.array([2, 3/4, 1.25]),
        jnp.array([is_tinted_lens, is_filter, is_neuroforce]).squeeze()
    )
    base_damage = jnp.fix(base_damage * effectiveness)
    # burn
    is_burned = attacker.status == Status.BURN
    is_physical = move.move_type == MoveType.PHYSICAL
    is_guts = attacker.ability == AbilityEnum.GUTS
    base_damage = conditional_mult_round(base_damage, 0.5, triple_and(1-is_guts, is_physical, is_burned))

    return base_damage

def damage_post_modifiers(
        damage: Array,
        defender_ability: AbilityEnum,
        move: Move) -> Array:
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
    # purifying salt
    is_purifying_salt = defender_ability == AbilityEnum.PURIFYING_SALT
    damage = conditional_mult_round(damage, 1 / 2, jnp.logical_and(is_purifying_salt, move.type == Type.GHOST))
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

    stat = conditional_mult(stat, 0.5, jnp.logical_or(water_bubble, thick_fat))

    # transistor is the only 1.3x
    is_transistor = jnp.logical_and(move_type==Type.ELECTRIC, attacker_ability==AbilityEnum.TRANSISTOR)
    stat = conditional_mult_round(stat, one_point_three, is_transistor)

    #choice items

    # pokemon specific boosting items, eg thick club or ogerpon mask

    return jnp.fix(stat)

def compute_defensive_stat(starting_stats, defender_ability, status, stat_idx):

    # marvel scale
    stats = conditional_mult_round(
        starting_stats[StatEnum.SPECIAL_DEFENSE], 1.5,
        jnp.logical_and(defender_ability == AbilityEnum.MARVEL_SCALE, status!=Status.NONE))

    # eviolite and assault vest
    return stats[stat_idx]
