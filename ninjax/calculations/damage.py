


import chex
import jax.numpy as jnp

from ninjax.calculations.logic import triple_and, triple_or
from ninjax.calculations.modifiers import (
    IRON_FIST,
    RECKLESS,
    TERRAIN_MULTIPLIER,
    TOUGH_CLAWS,
    conditional_mult,
    conditional_mult_prod,
    conditional_mult_prod_round,
    conditional_mult_round,
    four_thirds,
    one_point_three,
)
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Terrain, TerrainEnum, Weather, WeatherEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveFlags, MoveType
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.side.queries import conditional_mult_round as fixed_modifier, is_floating


Array = chex.Array

def compute_base_power(
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        terrain: Terrain, weather=WeatherEnum.NONE, gravity_active=False) -> chex.Array:
    """Calculate represented power modifiers, including grounded terrain effects."""

    power = fixed_modifier(move.base_power, 1.5,
        (move.identifier == 797) & (terrain.terrain == TerrainEnum.PSYCHIC) &
        (~is_floating(attacker) | jnp.asarray(gravity_active)))
    power = fixed_modifier(power, 1.25, (defender.ability == AbilityEnum.DRY_SKIN) & (move.type == Type.FIRE))
    power = fixed_modifier(power, 5461/4096,
        (move.identifier == 879) & (calculate_effectiveness_multiplier(move.type, defender.type_list) > 1))
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


    power = fixed_modifier(power, 5325/4096,
        (ability == AbilityEnum.SAND_FORCE) & (weather == WeatherEnum.SANDSTORM) &
        ((move.type == Type.ROCK) | (move.type == Type.GROUND) | (move.type == Type.STEEL)))
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

    is_grounded = ~is_floating(attacker) | jnp.asarray(gravity_active)
    t = terrain.terrain
    terrain_boosted = triple_or(
        jnp.logical_and(move.type==Type.GRASS, t==TerrainEnum.GRASSY),
        jnp.logical_and(move.type == Type.PSYCHIC, t == TerrainEnum.PSYCHIC),
        jnp.logical_and(move.type==Type.ELECTRIC, t==TerrainEnum.ELECTRIC),
    )
    terrain_boosted = jnp.logical_and(terrain_boosted, is_grounded)
    power = conditional_mult_round(power, TERRAIN_MULTIPLIER, terrain_boosted)
    grounded_target = ~is_floating(defender) | jnp.asarray(gravity_active)
    # Earthquake (Showdown move ID 89) is weakened by Grassy Terrain.
    weakened = ((t == TerrainEnum.GRASSY) & (move.identifier == 89)) | (
        (t == TerrainEnum.MISTY) & (move.type == Type.DRAGON))
    power = conditional_mult_round(power, .5, grounded_target & weakened)

    ability = attacker.ability
    # TODO: similar to with the low health abilities, we could put the conditions in an array and put mults in array
    # that is probably faster than this
    # iron fist
    power = conditional_mult_round(power, IRON_FIST, jnp.logical_and(move.punching, ability==AbilityEnum.IRON_FIST))
    # tough claws
    power = conditional_mult_round(power, TOUGH_CLAWS, jnp.logical_and(move.contact, ability==AbilityEnum.TOUGH_CLAWS))
    # reckless
    power = conditional_mult_round(power, RECKLESS, (move.recoil | (move.identifier == 136)) & (ability==AbilityEnum.RECKLESS))
    # strong jaw
    power = conditional_mult_round(power, 1.5, jnp.logical_and(move.biting, ability==AbilityEnum.STRONG_JAW))
    # mega launcher
    power = conditional_mult_round(power, 1.5, jnp.logical_and(move.launcher, ability==AbilityEnum.MEGA_LAUNCHER))
    # punk rock
    power = fixed_modifier(power, 5325/4096, jnp.logical_and(move.sound, ability==AbilityEnum.PUNK_ROCK))

    power = fixed_modifier(power, 1.5, (ability == AbilityEnum.SHARPNESS) & ((move.move_flags & MoveFlags.SLICING) != 0))
    # Masks boost every damaging type for their matching stored Ogerpon form.
    # Apply after the earlier power hooks (Showdown item priority 15).

    mask = (((attacker.item == ItemEnum.WELLSPRING_MASK) & (attacker.form == FormEnum.OGERPON_WELLSPRING)) |
            ((attacker.item == ItemEnum.HEARTHFLAME_MASK) & (attacker.form == FormEnum.OGERPON_HEARTHFLAME)) |
            ((attacker.item == ItemEnum.CORNERSTONE_MASK) & (attacker.form == FormEnum.OGERPON_CORNERSTONE)))
    species = attacker.species
    type_boost = (((attacker.item == ItemEnum.MAGNET) & (move.type == Type.ELECTRIC)) |
                  ((attacker.item == ItemEnum.SILK_SCARF) & (move.type == Type.NORMAL)) |
                  ((attacker.item == ItemEnum.SOUL_DEW) & ((species == 380) | (species == 381)) &
                   ((move.type == Type.DRAGON) | (move.type == Type.PSYCHIC))) |
                  ((attacker.item == ItemEnum.LUSTROUS_ORB) & (species == 484) &
                   ((move.type == Type.DRAGON) | (move.type == Type.WATER))))
    return fixed_modifier(power, 4915/4096, mask | type_boost)

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


def compute_damage_multipliers(
        key: chex.PRNGKey,
        attacker,
        defender,
        weather: Weather,
        move: Move,
        base_damage,
        is_crit,
        damage_roll,
        gravity_active=False
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
    # Type effectiveness, when we get around to implementing observations
    # it should include does not affect, not very effective, or super effective
    defending_types = jnp.where(jnp.asarray(gravity_active) & (move.type == Type.GROUND) &
                                (defender.type_list == Type.FLYING), Type.NONE, defender.type_list)
    effectiveness = calculate_effectiveness_multiplier(move.type, defending_types)
    is_levitate = defender.ability==AbilityEnum.LEVITATE
    effectiveness = effectiveness * (1-jnp.logical_and(jnp.logical_and(is_levitate, move.type==Type.GROUND), ~jnp.asarray(gravity_active)))
    is_tinted_lens = (effectiveness > 0) & (effectiveness < 1) & (attacker.ability==AbilityEnum.TINTED_LENS)
    is_filter = (effectiveness > 1) & (defender.ability==AbilityEnum.FILTER)
    is_neuroforce = (effectiveness > 1) & (attacker.ability==AbilityEnum.NEUROFORCE)
    base_damage = jnp.trunc(base_damage * effectiveness)
    # burn
    is_burned = attacker.status == Status.BURN
    is_physical = move.move_type == MoveType.PHYSICAL
    is_guts = attacker.ability == AbilityEnum.GUTS
    base_damage = conditional_mult_round(base_damage, 0.5, triple_and(1-is_guts, is_physical, is_burned))
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
