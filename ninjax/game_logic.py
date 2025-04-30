import jax
import jax.random as random
import jax.numpy as jnp
import chex
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, ACCURACY_MULTIPLIER_LOOKUP

from ninjax.enum_types import AbilityEnum, Status, Type, TerrainEnum, WeatherEnum, MoveType, Weather, Terrain, StatEnum
from ninjax.side import BattleState, update_active, clear_volatile_status, clear_boosts, add_boosts
from ninjax.pokemon import Pokemon
from ninjax.move import Move
from ninjax.utils import (
    conditional_mult_round, TERRAIN_MULTIPLIER, TYPE_EFFECTIVENESS, CRIT_STAGES,calculate_effectiveness_multiplier,
    COMPOUND_EYES_MULTIPLIER, conditional_mult, WEATHER_VEIL_MODIFIER, triple_and, triple_or, quad_or, ROUGH_SKIN_DAMAGE,
    in_range
)


def take_damage_value(state: BattleState, defender_idx: int, damage: chex.Array ,is_attack_damage) -> BattleState:
    active = state.active[defender_idx]
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    health_full = active.current_hp==active.max_hp
    is_sturdy = active.ability==AbilityEnum.STURDY
    new_health = jax.lax.clamp(0, (active.current_hp-damage), active.max_hp - health_full*is_sturdy*is_attack_damage)
    # check to make sure we don't accidentally revive a pokemon
    alive = jnp.logical_and(jnp.bool([new_health != 0]), active.is_alive)
    active = active.replace(current_hp=new_health, is_alive=alive)
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    return update_active(state, defender_idx, active)

def take_damage_percent(state: BattleState, defender_idx, percent: chex.Array) -> BattleState:
    damage = jnp.round(state.active.max_hp[defender_idx] * percent).astype(int)
    return take_damage_value(state, defender_idx, damage, False)

def status_helper(active, status: Status):
    active = active.replace(status=status)
    return active

def set_status(state: BattleState, side_idx, status: Status):
    active = state.active[side_idx]
    already_statused = active.status != Status.NONE
    is_immune = (
        jnp.logical_and(jnp.logical_or(status==Status.POISON, status==Status.TOXIC), active.is_poison_immune) +
        jnp.logical_and(status==Status.PARALYZE, active.is_paralyze_immune) +
        jnp.logical_and(status==Status.BURN, active.is_burn_immune) +
        jnp.logical_and(status==Status.FREEZE, active.is_freeze_immune) +
        jnp.logical_and(status==Status.SLEEP, active.is_sleep_immune)
    )
    active = jax.lax.cond(
        triple_or(already_statused, is_immune, status==Status.NONE)[0],
        lambda a, s: a,
        status_helper, active, status)
    return update_active(state, side_idx, active)

def compute_base_power(state: BattleState, attacker: Pokemon, move: Move):
    power = move.base_power
    is_grounded = 1 - attacker.is_floating
    terrain = state.terrain.terrain
    # grassy terrain
    power = conditional_mult_round(power, TERRAIN_MULTIPLIER, triple_and(is_grounded, move.type == Type.GRASS, terrain==TerrainEnum.GRASSY))
    # psychic terrain
    power = conditional_mult_round(power, TERRAIN_MULTIPLIER, triple_and(is_grounded, move.type == Type.PSYCHIC, terrain==TerrainEnum.PSYCHIC))
    # electric terrain
    power = conditional_mult_round(power, TERRAIN_MULTIPLIER, triple_and(is_grounded, move.type == Type.ELECTRIC, terrain==TerrainEnum.ELECTRIC))
    return power

def compute_base_damage(state: BattleState, move: Move, attacker_idx, power):
    boosted_stats = state.boosted_stats
    offensive_stat = boosted_stats[attacker_idx][move.offensive_stat]
    defensive_stat = boosted_stats[1-attacker_idx][move.defensive_stat]
    level = state.active.stat_table.level[attacker_idx]
    base_damage = jnp.floor(((2 * level / 5 + 2) * power * offensive_stat) / (defensive_stat * 50) + 2)
    return base_damage

def effect_spore_status(r):
    return (jnp.less(r, 0.09) * Status.POISON +
            in_range(0.09, r, 0.19) * Status.PARALYZE +
            in_range(0.19, r, 0.3) * Status.SLEEP)


def do_contact(key: chex.PRNGKey, state: BattleState, attacker_idx) -> (chex.PRNGKey, BattleState):
    active = state.active
    defender = active[1-attacker_idx]
    attacker = active[attacker_idx]
    is_rough_skin = defender.ability == AbilityEnum.ROUGH_SKIN
    state = take_damage_percent(state, attacker_idx, ROUGH_SKIN_DAMAGE * is_rough_skin)
    key, sub_key = random.split(key, 2)
    is_static = defender.ability==AbilityEnum.STATIC
    is_flame = defender.ability==AbilityEnum.FLAME_BODY
    is_effect_spore = jnp.logical_and(defender.ability==AbilityEnum.EFFECT_SPORE, 1-defender.is_powder_immune)
    r = random.uniform(sub_key)
    triggered = jnp.less_equal(r, 0.3)
    status = (effect_spore_status(r) * is_effect_spore + is_flame * Status.BURN + is_static * Status.PARALYZE) * triggered
    state = set_status(state, attacker_idx, status)
    return key, state

def compute_damage_multipliers(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, base_damage) -> (chex.PRNGKey, BattleState):
    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    attacker = state.active[attacker_idx]
    defender = state.active[1-attacker_idx]

    # sun modifier
    is_sun = state.weather.weather == WeatherEnum.SUN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_sun, move.type == Type.FIRE))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_sun, move.type == Type.WATER))
    # rain modifier
    is_rain = state.weather.weather == WeatherEnum.RAIN
    base_damage = conditional_mult_round(base_damage, 1.5, jnp.logical_and(is_rain, move.type == Type.WATER))
    base_damage = conditional_mult_round(base_damage, 0.5, jnp.logical_and(is_rain, move.type == Type.FIRE))

    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_chance = CRIT_STAGES[move.crit_stage] * (defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = random.uniform(one) < crit_chance
    crit_multiplier = 1.5
    # damage roll, idc about preserving the in game RNG generation
    base_damage = conditional_mult_round(base_damage, crit_multiplier, is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100
    base_damage = conditional_mult_round(base_damage, damage_roll, 1)
    # stab multiplier
    # TODO: this logic can probably be simplified
    is_tera_boosted = jnp.logical_and(attacker.is_terastallized, attacker.tera_type == move.type)
    is_matching_tera = jnp.logical_and(is_tera_boosted, jnp.any(attacker.type_list == attacker.tera_type))
    is_stab = jnp.logical_or(jnp.any(attacker.type_list == move.type), is_tera_boosted)
    is_adaptability_boosted = attacker.ability==AbilityEnum.ADAPTABILITY * is_stab
    stab_multiplier = (
            1.5 +
            0.5 * jnp.logical_or(is_matching_tera, is_adaptability_boosted) +
            0.25 * jnp.logical_and(is_matching_tera, is_adaptability_boosted))
    base_damage = conditional_mult_round(base_damage, stab_multiplier, is_stab)
    # Type effectiveness, when we get around to implementing observations
    # it should include does not affect, not very effective, or super effective
    effectiveness = calculate_effectiveness_multiplier(move.type, defender.type_list)
    is_levitate = defender.ability==AbilityEnum.LEVITATE
    base_damage = jnp.fix(base_damage * effectiveness) * (1-jnp.logical_and(is_levitate, move.type==Type.GROUND))
    # burn
    is_burned = attacker.status == Status.BURN
    is_physical = move.move_type == MoveType.PHYSICAL
    is_guts = attacker.ability == AbilityEnum.GUTS

    # do on contact effects
    makes_contact = move.makes_contact
    key, state = jax.lax.cond(makes_contact[0], do_contact, lambda k, s, a: (k, s), key, state, attacker_idx)

    base_damage = conditional_mult_round(base_damage, 0.5, triple_and(1-is_guts, is_physical, is_burned))
    return key, base_damage

def do_move_damage(key: chex.PRNGKey, state: BattleState, player_idx, move: Move, stat_index) -> (BattleState, chex.PRNGKey):
    attacker = state.active[player_idx]

    # base power modifications, technician, tera, terrain etc
    power = compute_base_power(state, attacker, move)

    # base damage pre multipliers
    base_damage = compute_base_damage(state, move, player_idx, power)

    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    key, damage = compute_damage_multipliers(key, state, player_idx, move, base_damage)

    # dealing damage
    damage = damage.astype(int)
    state = take_damage_value(state, 1 - player_idx, damage, True)

    return key, state

def do_status_move(key: chex.PRNGKey, state: BattleState, player_idx, move: Move, stat_index) -> (BattleState, chex.PRNGKey):
    # this is gonna be a pain
    return key, state

# this is for when water absorb or volt absorb is triggered
def do_healing_from_move(key: chex.PRNGKey, state: BattleState, player_idx, move: Move, stat_index) -> (BattleState, chex.PRNGKey):
    state = take_damage_percent(state, 1-player_idx, -1/4)
    return key,state

def do_stat_boost_from_move(key: chex.PRNGKey, state: BattleState, player_idx, move: Move, stat_index) -> (BattleState, chex.PRNGKey):
    state = add_boosts(state, 1-player_idx, stat_index, 1)
    return key, state


def do_flash_fire_from_move(key: chex.PRNGKey, state: BattleState, player_idx, move: Move, stat_index) -> (BattleState, chex.PRNGKey):
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
    take_damage_percent(state, idx, sand_damage)

    # grassy terrain healing
    grass_healing = (is_floating - 1) / 16 * state.terrain.terrain == TerrainEnum.GRASSY
    take_damage_percent(state, idx, grass_healing)

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
    no_status = active.status != Status.NONE
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
    state = add_boosts(state, side_idx, StatEnum.SPEED, -1 * is_not_flying * state[side_idx].sticky_webs)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state[side_idx].toxic_spikes * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(toxic_spikes=toxic_spikes)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state[side_idx].toxic_spikes) * (state[side_idx].toxic_spikes != 0)
    state = set_status(state, side_idx, status_ * no_status * is_poison_immune * is_not_flying)

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
    active = state.active[side_idx]

    # activate weather abilities if target is alive
    # we abuse the choice to WeatherEnum.NONE=0 to simplify the logic
    current_weather = state.weather.weather
    new_weather = (active.ability==AbilityEnum.DROUGHT * WeatherEnum.SUN +
                   active.ability==AbilityEnum.DRIZZLE * WeatherEnum.RAIN +
                   active.ability==AbilityEnum.SAND_STREAM * WeatherEnum.SANDSTORM +
                   active.ability==AbilityEnum.SNOW_WARNING * WeatherEnum.SNOW)
    new_weather = current_weather * new_weather==WeatherEnum.NONE + new_weather
    matching_weather = current_weather==new_weather
    # TODO: add item check for the weather stones
    new_duration = 5 * (1 - matching_weather) + state.weather.duration * matching_weather
    state = state.replace(weather=Weather(new_weather, new_duration))

    # intimidate
    is_intimidate = active.ability == AbilityEnum.INTIMIDATE
    state = add_boosts(state, 1-side_idx, StatEnum.ATTACK, -1*is_intimidate)
    return state



def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> (chex.PRNGKey, BattleState):
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state.replace(
        reflect=jnp.maximum(state.reflect - 1, 0),
        light_screen=jnp.maximum(state.light_screen - 1, 0),
        aurora_veil=jnp.maximum(state.aurora_veil - 1, 0),
        tailwind=jnp.maximum(state.tailwind - 1, 0),
        toxic_counter=toxic_counter
    )
    return key, state



def move_interrupted(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> (chex.PRNGKey, BattleState):
    # TODO: eventually we will need to figure out we handle observations and include it here
    return key, state

def move_used(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> (chex.PRNGKey, BattleState):
    move = state.active[attacker_index].moves[move_index]

    # decrement pp
    # theoretically the legal action mask should prevent us from using the move if its at 0 so we dont need to clip
    move = move.replace(current_pp=move.current_pp-1)
    active = state.active[attacker_index]
    new_pp = active.moves.current_pp.at[move_index].subtract(1)
    new_moves = active.moves.replace(current_pp=new_pp)
    active = active.replace(moves=new_moves)
    state = update_active(state, attacker_index, active)


    # check if move hits
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    active = state.active
    defender_ability = active[1 - attacker_index].ability
    attacker_ability = active[attacker_index].ability
    weather = state.weather.weather
    accuracy = (move.accuracy *
                ACCURACY_MULTIPLIER_LOOKUP[6 + state.boosts.acc_boosts[attacker_index, 0]] *
                ACCURACY_MULTIPLIER_LOOKUP[6 - state.boosts.acc_boosts[1 - attacker_index, 1]])
    veil_active = jnp.logical_or(
        jnp.logical_and(defender_ability==AbilityEnum.SAND_VEIL, weather==WeatherEnum.SANDSTORM),
        jnp.logical_and(defender_ability==AbilityEnum.SNOW_CLOAK, weather==WeatherEnum.SNOW))
    conditions = jnp.array(
        [jnp.equal(attacker_ability,AbilityEnum.COMPOUND_EYES),
         veil_active])
    modifiers = jnp.array([COMPOUND_EYES_MULTIPLIER, WEATHER_VEIL_MODIFIER])
    accuracy = accuracy * jnp.prod(jnp.power(modifiers, conditions))
    no_guard_active = jnp.logical_or(
        jnp.equal(defender_ability, AbilityEnum.NO_GUARD),
        jnp.equal(attacker_ability, AbilityEnum.NO_GUARD))
    accuracy = jnp.clip(accuracy, no_guard_active, 1)

    # choose function based on if move hits
    return jax.lax.cond(jnp.less_equal(r, accuracy)[0], move_hits, move_misses, key, state, attacker_index, move_index)


def move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> (chex.PRNGKey, BattleState):
    # decide branch to execute based on ability immunities
    defender = state.active[1 - attacker_index]
    move = state.active[attacker_index].moves[move_index]
    ability = defender.ability
    branches = [do_move_damage,
                do_status_move,
                do_stat_boost_from_move,
                do_flash_fire_from_move,
                do_healing_from_move]
    is_flash_fire = jnp.logical_and(ability==AbilityEnum.FLASH_FIRE, move.type==Type.FIRE)
    is_spa_boost = jnp.logical_or(
        jnp.logical_and(AbilityEnum.STORM_DRAIN==ability, Type.WATER==move.type),
        jnp.logical_and(ability == AbilityEnum.LIGHTNING_ROD, move.type == Type.ELECTRIC))
    is_attack_boost = jnp.logical_and(ability == AbilityEnum.SAP_SIPPER, move.type == Type.GRASS)
    is_heal = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_and(ability == AbilityEnum.WATER_ABSORB, move.type == Type.WATER),
            jnp.logical_and(ability == AbilityEnum.VOLT_ABSORB, move.type == Type.ELECTRIC)),
            jnp.logical_and(ability == AbilityEnum.EARTH_EATER, move.type == Type.GROUND))
    is_status = move.move_type == MoveType.STATUS * (1 - quad_or(is_flash_fire, is_spa_boost, is_attack_boost, is_heal))
    # this feels really hacky way to compute this but :shrug:
    branch_index = is_status + jnp.logical_or(is_spa_boost, is_attack_boost) * 2 + is_flash_fire * 3 + is_heal * 4

    # i think using a switch means we skip evaluating the branches we don't need
    # the stat_index only is used in the stat_boost branch so its value doesnt matter the rest of the time
    return jax.lax.switch(branch_index[0], branches, key, state, attacker_index, move, 1 + 3 * is_spa_boost)

def move_misses(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> (chex.PRNGKey, BattleState):
    return key, state



