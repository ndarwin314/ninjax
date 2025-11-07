from typing import Tuple

import jax
import jax.random as random
import jax.numpy as jnp
import chex
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, ACCURACY_MULTIPLIER_LOOKUP

from ninjax.enum_types import AbilityEnum, Status, Type, TerrainEnum, WeatherEnum, MoveType, Weather, Terrain, StatEnum
from ninjax.side import (BattleState, update_active, clear_volatile_status, clear_boosts, add_boosts,reduce_boosts,
                         conditional_add_boosts, conditional_reduce_boosts, set_weather, set_terrain)
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

    # water bubble
    water_bubble = jnp.logical_and(ability==AbilityEnum.WATER_BUBBLE, move.type==Type.WATER)
    power = conditional_mult_round(power, 2, water_bubble)

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
        ability: AbilityEnum,
        defender_ability: AbilityEnum,
        hp_percent,
        level,
        attack_multiplier,
        boosted_stats,
        move: Move,
        attacker_idx,
        power) -> Array:

    # this is a hack but it should work i think probably
    # TODO i want to write this better actually doing repeated mult rounds but whatever
    offensive_stat = jnp.floor(boosted_stats[attacker_idx][move.offensive_stat] * attack_multiplier)
    defensive_stat = boosted_stats[1-attacker_idx][move.defensive_stat]
    # ruin abilities
    sword_of_ruin = ability==AbilityEnum.SWORD_OF_RUIN
    beads_of_ruin = ability == AbilityEnum.BEADS_OF_RUIN
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

    # we need some additional conditional stat changes here
    # for example, guts always effects attack when its active but overgrow boosts attack only for grass moves when its active
    type_ = move.type
    low_hp = jnp.less_equal(hp_percent, 1/3)
    is_overgrow = triple_and(type_==Type.GRASS, low_hp, ability==AbilityEnum.OVERGROW)
    is_blaze = triple_and(type_==Type.FIRE, low_hp, ability==AbilityEnum.BLAZE)
    is_torrent = triple_and(type_==Type.WATER, low_hp, ability==AbilityEnum.TORRENT)
    is_swarm = triple_and(type_==Type.BUG, low_hp, ability==AbilityEnum.SWARM)
    is_steel_worker = jnp.logical_and(type_==Type.STEEL, ability==AbilityEnum.STEELWORKER)
    is_rocky_payload = jnp.logical_and(type_==Type.ROCK, ability==AbilityEnum.ROCKY_PAYLOAD)
    arr = jnp.array([is_swarm, is_torrent, is_blaze, is_overgrow, is_steel_worker, is_rocky_payload])
    offensive_stat = conditional_mult_round(offensive_stat, 1.5,
                                            jnp.any(arr))
    is_transistor = jnp.logical_and(type_==Type.ELECTRIC, ability==AbilityEnum.TRANSISTOR)
    is_maw = jnp.logical_and(type_==Type.DRAGON, ability==AbilityEnum.DRAGONS_MAW)
    offensive_stat = conditional_mult_round(offensive_stat, one_point_three, jnp.any(jnp.array([is_maw, is_transistor])))

    base_damage = jnp.floor(((2 * level / 5 + 2) * power * offensive_stat) / (defensive_stat * 50) + 2)
    return base_damage


def compute_damage_multipliers(
        key: chex.PRNGKey,
        attacker,
        defender,
        weather: Weather,
        move: Move,
        base_damage) -> Tuple[chex.PRNGKey, Array, bool]:
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

    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_stage = (move.crit_stage+
                  (attacker.ability==AbilityEnum.SUPER_LUCK)[0] +
                  3*jnp.logical_and(attacker.ability==AbilityEnum.MERCILESS, defender.is_poisoned))
    crit_chance = CRIT_STAGES[crit_stage] * (defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = random.uniform(one) < crit_chance
    crit_multiplier = 1.5 + 0.75 * (attacker.ability==AbilityEnum.SNIPER)[0]
    # damage roll, idc about preserving the in game RNG generation
    base_damage = conditional_mult_round(base_damage, crit_multiplier, is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100
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

    return key, base_damage, is_crit

def damage_post_modifiers(
        damage: Array,
        defender_ability: AbilityEnum,
        move: Move) -> Array:
    damage = damage.astype(int)
    # bulbapedia says water bubble "halves damage" so here we are
    is_fire_move = move.type == Type.FIRE
    water_bubble = jnp.logical_and(is_fire_move, defender_ability == AbilityEnum.WATER_BUBBLE)
    damage = conditional_mult_round(damage, 1 / 2, water_bubble)
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