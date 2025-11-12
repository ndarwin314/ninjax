from typing import Tuple

import jax
import jax.random as random
import jax.numpy as jnp
import chex
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, ACCURACY_MULTIPLIER_LOOKUP

from ninjax.enum_types import AbilityEnum, Status, Type, TerrainEnum, WeatherEnum, MoveType, Weather, Terrain, StatEnum
from ninjax.side import (BattleState, update_active, clear_volatile_status, clear_boosts, add_boosts, reduce_boosts,
                         conditional_add_boosts, conditional_reduce_boosts, set_weather, set_terrain,
                         take_damage_percent, set_status, take_damage_value, raw_boosted_stats)
from ninjax.pokemon import Pokemon
from ninjax.move import Move
from ninjax.move_effects import after_move_finished, after_every_hit
from ninjax.damage import damage_post_modifiers, compute_damage_multipliers, compute_base_damage, compute_base_power
from ninjax.utils import (
    conditional_mult_round, TERRAIN_MULTIPLIER, TYPE_EFFECTIVENESS, CRIT_STAGES,calculate_effectiveness_multiplier,
    COMPOUND_EYES_MULTIPLIER, conditional_mult, WEATHER_VEIL_MODIFIER, triple_and, triple_or, quad_or, ROUGH_SKIN_DAMAGE,
    in_range, IRON_FIST, TOUGH_CLAWS, one_third, RECKLESS, VICTORY_STAR, four_thirds, quad_and,
    conditional_mult_prod_round, conditional_mult_prod, one_point_three)

jax.config.update("jax_disable_jit", True)

Array = chex.Array



def do_damaging_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    active = state.active
    attacker = active[attacker_idx]
    defender = active[1-attacker_idx]
    hp_start = defender.current_hp

    # base power modifications, technician, tera, terrain etc
    power = compute_base_power(
        attacker,
        defender,
        move,
        state.terrain
    )
    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_stage = (move.crit_stage+
                  (attacker.ability==AbilityEnum.SUPER_LUCK)[0] +
                  3*jnp.logical_and(attacker.ability==AbilityEnum.MERCILESS, defender.is_poisoned))
    crit_chance = CRIT_STAGES[crit_stage] * (defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = random.uniform(one) < crit_chance
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100

    # base damage pre multipliers
    # so the problem is that body press which does damage based on defence, ignores the defence reduction of sword of ruin
    # when it calculated the offensive stat, but doesn't ignore it as a modifier of defensive stats
    # but it also does get boosted by modifiers like choice band, huge power, and guts because this game is made with spaghetti code
    # so i have to implement that and i think this is probably the least stupid way to do that
    raw_stats = state.active.stats
    boosts = state.boosts
    attacker_stats, defender_stats = raw_boosted_stats(
        raw_stats,
        boosts,
        attacker_idx,
        attacker.ability==AbilityEnum.UNAWARE,
        defender.ability==AbilityEnum.UNAWARE,
        is_crit
    )


    base_damage = compute_base_damage(
        attacker.ability,
        defender.ability,
        attacker.hp_percent,
        attacker.level,
        attacker_stats,
        defender_stats,
        move,
        power,
        attacker.status,
        defender.status
    )

    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    damage = compute_damage_multipliers(
        key,
        attacker,
        defender,
        state.weather,
        move,
        base_damage,
        is_crit,
        damage_roll
    )

    damage = damage_post_modifiers(damage, defender.ability, move)

    # dealing damage
    state = take_damage_value(state, 1 - attacker_idx, damage, True)


    # recoil
    state = jax.lax.cond(
        jnp.logical_and(move.recoil, attacker.ability!=AbilityEnum.ROCK_HEAD)[0],
        do_recoil, lambda s, p, d: s,
        state, attacker_idx, jnp.fix(damage*move.recoil_percent))

    # TODO: this is actually really important and really hard
    # i have no clue how to implement multi-hit moves well, it might just need to be hard coded or something stupid
    # once that is done move around these to appropriate places, this is fine for now ig
    key, state = after_every_hit(key, state, attacker_idx, move, is_crit)
    state = after_move_finished(state, defender, 1-attacker_idx, hp_start)

    return key, state

def do_recoil(state: BattleState, attacker_idx: int, damage) -> BattleState:
    state = take_damage_value(state, attacker_idx, damage, False)
    return state

def do_status_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    # this is gonna be a pain
    return key, state

# this is for when water absorb or volt absorb is triggered
def do_healing_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    state = take_damage_percent(state, 1-attacker_idx, -1/4)
    return key,state

def do_stat_boost_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    state = add_boosts(state, 1-attacker_idx, stat_index, boost_value)
    return key, state


def do_flash_fire_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    # TODO: i dont want to do volatile status
    return key, state

def end_turn_damage(state: BattleState) -> BattleState:
    # TODO: the order of all these updates is probably incorrect
    # i think it should be like sand, sand, grass, grass
    # whereas this is currently sand, grass, sand grass,
    # and order should also depend on speed
    # that being said idk if that is high priority
    # find something for gen 9 https://www.smogon.com/forums/threads/sword-shield-battle-mechanics-research.3655528/page-64#post-9244179
    # also another wrinkle is that effects take effect based on speed order
    # i.e it would go sand fast, sand slow, grass fast, grass slow
    # this matters a lot in vgc but usually less in singles
    # it also matters for determining winner if the last pokemon for both players faints on the same turn
    active = state.active
    idx = jnp.array([0,1])
    is_floating = active.is_floating
    is_sand_immune = active.is_sand_immune

    # sand damage
    sand_damage = (1 - is_sand_immune) / 16 * state.weather.weather == WeatherEnum.SANDSTORM
    state = take_damage_percent(state, idx, sand_damage)

    # rain abilities
    rain_healing = (active.ability==AbilityEnum.RAIN_DISH + 2 * (active.ability==AbilityEnum.DRY_SKIN)) / 16 * state.weather.weather == WeatherEnum.RAIN
    # why is this one an int but the others arent
    state = take_damage_percent(state, idx, -rain_healing)

    # sun abilities
    sun_damage = (active.ability==AbilityEnum.DRY_SKIN + active.ability==AbilityEnum.SOLAR_POWER) / 8 *state.weather.weather == WeatherEnum.SUN
    state = take_damage_percent(state, idx, sun_damage)

    # snow abilities
    snow_healing = active.ability==AbilityEnum.ICE_BODY / 16 *state.weather.weather == WeatherEnum.SNOW
    state = take_damage_percent(state, idx, -snow_healing)


    # grassy terrain healing
    grass_healing = (is_floating - 1) / 16 * state.terrain.terrain == TerrainEnum.GRASSY
    state = take_damage_percent(state, idx, grass_healing)

    # status damage
    # technically this should be factored out to multiple bits since it goes burn poison toxic in priority
    status_damage = (1 / 8 * (active.status==Status.POISON) +
                     1 / 16 * (active.status==Status.BURN) +
                     state.toxic_counter / 16 * (active.status==Status.TOXIC))
    state = take_damage_percent(state, idx, status_damage)
    return state

# TODO: at some point probably factor out part of this into like
# just swapping out to implement baton pass idk
def swap_out(
    key,
    state: BattleState,
    side_idx,
    new_active: int
) -> BattleState:
    # swaps the active pokemon and does appropriate things like
    # 1. clearing volatile statuses
    # 2. resting boosts
    # 3. probably stuff im forgetting
    # 4. ahhh palafin, ahhh regenerator

    # clear boosts and volatile status from active pokemon, maybe add baton pass check here?
    state = clear_volatile_status(clear_boosts(state, side_idx), side_idx)

    # update active pokemon index
    # this update could be saved since we need an update later but idk if it will matter
    active = state.active_index.at[side_idx].set(new_active)
    toxic_counter = state.toxic_counter.at[side_idx].set(0)
    state = state.replace(active=active, toxic_counter=toxic_counter)

    # hazards
    active = state[side_idx].team[new_active]
    is_not_flying = 1 - active.is_floating
    is_not_hazard_immune = 1 - active.is_hazard_immune
    is_poison = active.is_type(Type.POISON)
    is_poison_immune = jnp.any(active.type_list == Type.STEEL)
    # stealth rocks
    state = take_damage_percent(
        state,
        side_idx,
        state[side_idx].stealth_rocks * calculate_effectiveness_multiplier(Type.ROCK, active.type_list) / 8
    )

    # spikes
    state = take_damage_percent(
        state, side_idx,
        (state[side_idx].spikes != 0) / (10 - 2 * state[side_idx].spikes) * is_not_flying
    )

    # sticky webs
    # so apparently if a pokemon sets webs and a pokemon with mirror armor is effected by them
    # then it applies the stat reduction to that mon, but this only happens if they are on field
    state = reduce_boosts(state, side_idx, StatEnum.SPEED, 1 * is_not_flying * state[side_idx].sticky_webs, False)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state[side_idx].toxic_spikes * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(toxic_spikes=toxic_spikes)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state[side_idx].toxic_spikes) * (state[side_idx].toxic_spikes != 0)
    key, state = set_status(key, state, side_idx, status_ * is_poison_immune * is_not_flying)

    #check if switch in is still alive
    active_hp = state[side_idx].active.current_hp[side_idx]
    is_alive = active_hp != 0

    state = jax.lax.cond(is_alive, swap_is_alive, lambda a, b: a, state, side_idx)


    active = state.active[side_idx].replace(is_alive=is_alive)
    state = update_active(state, side_idx, active)
    return state

def swap_is_alive(
    state: BattleState,
    side_idx) -> BattleState:
    active = state.active
    attacker = active[side_idx]
    defender_ability = active[1-side_idx].ability

    # activate weather abilities if target is alive
    # we abuse the choice to WeatherEnum.NONE=0 to simplify the logic

    new_weather = (jnp.logical_or(active.ability==AbilityEnum.DROUGHT, active.ability==AbilityEnum.ORICHALCUM_PULSE) * WeatherEnum.SUN +
                   active.ability==AbilityEnum.DRIZZLE * WeatherEnum.RAIN +
                   active.ability==AbilityEnum.SAND_STREAM * WeatherEnum.SANDSTORM +
                   active.ability==AbilityEnum.SNOW_WARNING * WeatherEnum.SNOW)
    state = set_terrain(state, new_weather, 5)

    # terrain, same deal as weather
    new_terrain = (jnp.logical_and(active.ability==AbilityEnum.ELECTRIC_SURGE, active.ability==AbilityEnum.HADRON_ENGINE) * TerrainEnum.ELECTRIC +
                   active.ability==AbilityEnum.PSYCHIC_SURGE * TerrainEnum.PSYCHIC +
                   active.ability==AbilityEnum.GRASSY_SURGE * TerrainEnum.GRASSY +
                   active.ability==AbilityEnum.MISTY_SURGE * TerrainEnum.MISTY)
    state = set_terrain(state, new_terrain, 5)

    # intimidate
    is_intimidate = attacker.ability == AbilityEnum.INTIMIDATE
    is_intimidate_immune = quad_or(
        defender_ability==AbilityEnum.OBLIVIOUS,
        defender_ability==AbilityEnum.OWN_TEMPO,
        defender_ability==AbilityEnum.INNER_FOCUS,
        defender_ability==AbilityEnum.SCRAPPY,
    )
    intimidate_activated = jnp.logical_and(is_intimidate, 1-is_intimidate_immune)
    state = conditional_reduce_boosts(state, 1-side_idx, StatEnum.ATTACK, 1, intimidate_activated, True)
    return state


def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state.replace(
        reflect=jnp.maximum(state.reflect - 1, 0),
        light_screen=jnp.maximum(state.light_screen - 1, 0),
        aurora_veil=jnp.maximum(state.aurora_veil - 1, 0),
        tailwind=jnp.maximum(state.tailwind - 1, 0),
        toxic_counter=toxic_counter
    )
    return key, state

def step_moody(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    # i think this needs to be a loop to make the random choices function work
    abilities = state.active.ability
    for i in range(2):
        key, sub_key = random.split(key, 2)
        key, state = conditional_add_boosts(
            state,
            i,
            abilities[i]==AbilityEnum.MOODY,
            1 + random.choice(sub_key, 5, (2,), replace=False),
            (2, -1)
        )
    return key, state



def move_interrupted(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    # TODO: eventually we will need to figure out we handle observations and include it here
    return key, state

def move_used(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    move = state.active[attacker_index].moves[move_index]

    # decrement pp
    # theoretically the legal action mask should prevent us from using the move if its at 0 so we dont need to clip
    move = move.replace(current_pp=move.current_pp-1)
    active = state.active[attacker_index]
    new_pp = active.moves.current_pp.at[move_index].subtract(1)
    new_moves = active.moves.replace(current_pp=new_pp)
    active = active.replace(moves=new_moves)
    state = update_active(state, attacker_index, active)
    attacker_ability = state.active[attacker_index].ability

    # do modifications to the move
    # check for -ate abilities that change move type
    # also add ion deluge at some point
    is_normal = move.type==Type.NORMAL
    aerilate = jnp.logical_and(attacker_ability==AbilityEnum.AERILATE, is_normal)
    refrigerate = jnp.logical_and(attacker_ability==AbilityEnum.REFRIGERATE, is_normal)
    pixilate = jnp.logical_and(attacker_ability==AbilityEnum.PIXILATE, is_normal)
    galvanize = jnp.logical_and(attacker_ability==AbilityEnum.GALVANIZE, is_normal)
    normalize = attacker_ability==AbilityEnum.NORMALIZE
    liquid_voice = jnp.logical_and(attacker_ability==AbilityEnum.LIQUID_VOICE, move.sound)
    override_boost = jnp.logical_or(quad_or(aerilate, refrigerate, pixilate, galvanize), normalize)
    new_type = (move.type * jnp.logical_or(override_boost, liquid_voice) +
                aerilate * Type.FLYING +
                refrigerate * Type.ICE +
                pixilate * Type.FAIRY +
                galvanize * Type.FAIRY +
                liquid_voice * Type.WATER +
                normalize * Type.NORMAL)
    new_power = conditional_mult_round(move.base_power, 1.2, override_boost)
    move = move.replace(base_power=new_power, type=new_type)

    # check for accuracy bypasses
    storm_drain = jnp.logical_and(AbilityEnum.STORM_DRAIN == attacker_ability, Type.WATER == move.type)
    lighting_rod = jnp.logical_and(attacker_ability == AbilityEnum.LIGHTNING_ROD, move.type == Type.ELECTRIC)
    draw_in = jnp.logical_or(storm_drain, lighting_rod)


    key, state = jax.lax.cond(
        draw_in,
        do_stat_boost_from_move, move_not_drawn_in,
        key, state, attacker_index, move_index, StatEnum.SPECIAL_ATTACK
    )
    return key, state


def move_not_drawn_in(key: chex.PRNGKey, state: BattleState, attacker_index, move_index: int, unused: int) -> Tuple[chex.PRNGKey, BattleState]:
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    active = state.active
    move = active[attacker_index].moves[move_index]
    defender_ability = active[1 - attacker_index].ability
    attacker_ability = active[attacker_index].ability

    weather = state.weather.weather
    is_hustle = jnp.logical_and((attacker_ability == AbilityEnum.HUSTLE), move.move_type==MoveType.PHYSICAL)

    accuracy = (move.accuracy *
                ACCURACY_MULTIPLIER_LOOKUP[6 + state.boosts.acc_boosts[attacker_index, 0]] *
                ACCURACY_MULTIPLIER_LOOKUP[6 - state.boosts.acc_boosts[1 - attacker_index, 1]] *
                jnp.power(0.8, is_hustle))
    veil_active = jnp.logical_or(
        jnp.logical_and(defender_ability == AbilityEnum.SAND_VEIL, weather == WeatherEnum.SANDSTORM),
        jnp.logical_and(defender_ability == AbilityEnum.SNOW_CLOAK, weather == WeatherEnum.SNOW))
    conditions = jnp.array(
        [jnp.equal(attacker_ability, AbilityEnum.COMPOUND_EYES),
         attacker_ability==AbilityEnum.VICTORY_STAR,
         veil_active])
    modifiers = jnp.array([COMPOUND_EYES_MULTIPLIER, WEATHER_VEIL_MODIFIER, VICTORY_STAR])
    accuracy = accuracy * jnp.prod(jnp.power(modifiers, conditions))
    no_guard_active = jnp.logical_or(
        jnp.equal(defender_ability, AbilityEnum.NO_GUARD),
        jnp.equal(attacker_ability, AbilityEnum.NO_GUARD))
    accuracy = jnp.clip(accuracy, no_guard_active, 1)

    # choose function based on if move hits
    return jax.lax.cond(
        jnp.less_equal(r, accuracy)[0],
        move_hits, move_misses,
        key, state, attacker_index, move_index)

def is_immune(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value):
    return key, state

def move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:

    # decide branch to execute based on ability immunities
    defender = state.active[1 - attacker_index]
    move = state.active[attacker_index].moves[move_index]
    ability = defender.ability
    branches = [do_damaging_move,
                do_status_move,
                do_stat_boost_from_move,
                do_flash_fire_from_move,
                do_healing_from_move,
                is_immune]
    is_flash_fire = jnp.logical_and(ability==AbilityEnum.FLASH_FIRE, move.type==Type.FIRE)
    is_spa_boost = False
    # this is redundant
    """is_spa_boost = jnp.logical_or(
        jnp.logical_and(AbilityEnum.STORM_DRAIN==ability, Type.WATER==move.type),
        jnp.logical_and(ability == AbilityEnum.LIGHTNING_ROD, move.type == Type.ELECTRIC))"""
    is_attack_boost = jnp.logical_and(ability == AbilityEnum.SAP_SIPPER, move.type == Type.GRASS)
    is_speed_boost = jnp.logical_and(ability==AbilityEnum.MOTOR_DRIVE, move.type==Type.ELECTRIC)
    is_def_boost = jnp.logical_and(ability==AbilityEnum.WELL_BAKED_BODY, move.type==Type.FIRE)
    is_heal = quad_or(
        jnp.logical_and(ability == AbilityEnum.WATER_ABSORB, move.type == Type.WATER),
        jnp.logical_and(ability == AbilityEnum.VOLT_ABSORB, move.type == Type.ELECTRIC),
        jnp.logical_and(ability == AbilityEnum.EARTH_EATER, move.type == Type.GROUND),
        jnp.logical_and(ability == AbilityEnum.DRY_SKIN, move.type == Type.WATER)
    )
    # all of this is hacky and inelegant and probably could be simplified
    is_stat_boost = quad_or(is_speed_boost, is_attack_boost, is_spa_boost, is_def_boost)
    not_bypass_move = triple_or(is_flash_fire, is_stat_boost, is_heal)
    immune = defender.is_immune_to_move(move)
    immune = jnp.logical_and(immune, not_bypass_move)
    is_status = jnp.logical_and(move.move_type == MoveType.STATUS, not_bypass_move)
    # this feels really hacky way to compute this but :shrug:
    branch_index = (
            is_status +
            is_stat_boost * 2 +
            is_flash_fire * 3 +
            is_heal * 4 +
            immune * 5
    )
    boost_value = 1 + ability==AbilityEnum.WELL_BAKED_BODY

    # i think using a switch means we skip evaluating the branches we don't need
    # the stat_index only is used in the stat_boost branch so its value doesnt matter the rest of the time
    return jax.lax.switch(
        branch_index[0],
        branches,
        key,
        state,
        attacker_index,
        move,
        is_attack_boost + 2 * is_def_boost + 4 * is_spa_boost + 6 * is_speed_boost,
        boost_value
    )

def move_misses(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    return key, state



