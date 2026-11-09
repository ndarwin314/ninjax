from typing import Tuple
from dataclasses import field

import jax.lax
import jax.random as random
import chex
from flax import struct
import jax.numpy as jnp

from ninjax.stats import StatBoosts
from ninjax.pokemon import Pokemon
from ninjax.enum_types import StatEnum, WeatherEnum, TerrainEnum, Status, TurnType, AbilityEnum, Weather, Terrain
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, triple_or, conditional_mult_round, triple_and

Array = jax.Array



class VolatileStatus(struct.PyTreeNode):
    confused: bool = False
    # TODO this is gonna suck, make sure everything has default values
    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        return self

class BattleState(struct.PyTreeNode):
    """A single battle: side fields have shape (2, 1), team shape (2, 6)."""
    team: Pokemon
    # figure out how to represent no pokemon on field, maybe active_index=-1?
    # or make a flag variable?
    active_index: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    stealth_rocks: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    sticky_webs: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    spikes: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    toxic_spikes: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    reflect: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    light_screen: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    aurora_veil: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    tailwind: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    toxic_counter: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    boosts: StatBoosts = field(default_factory=lambda: StatBoosts(
        normal_boosts=jnp.zeros((2, 6) ,dtype='int32'),
        acc_boosts=jnp.zeros((2,2), dtype='int32')))
    legal_action_mask: Array = field(default_factory=lambda: jnp.ones((2, 15), dtype=jnp.bool_))
    can_tera: Array = field(default_factory=lambda: jnp.ones((2, 1), dtype=jnp.bool_))
    switch_requested: Array = field(default_factory=lambda: jnp.zeros(2, dtype=jnp.bool_))
    roster: Array = field(default_factory=lambda: jnp.ones((2, 6), dtype=jnp.bool_))
    winner: Array = field(default_factory=lambda: jnp.int32(-1))
    perish_turns: Array = field(default_factory=lambda: -jnp.ones((2, 1), dtype=jnp.int32))
    # notably volatile status needs like wish, healing wish, and future sight things
    # but those are lowish priority
    volatile_status: VolatileStatus = field(default_factory=VolatileStatus) # TODO

    weather: Weather = Weather(WeatherEnum.NONE, 0)
    terrain: Terrain = Terrain(TerrainEnum.NONE, 0)
    turn_number: int = 0
    trick_room_duration: int = 0
    gravity_duration: int = 0
    turn_type: TurnType = TurnType.STANDARD

    def __getitem__(self, idx):
        return self.replace(
            team=self.team[idx], active_index=self.active_index[idx],
            stealth_rocks=self.stealth_rocks[idx], sticky_webs=self.sticky_webs[idx],
            spikes=self.spikes[idx], toxic_spikes=self.toxic_spikes[idx],
            reflect=self.reflect[idx], light_screen=self.light_screen[idx],
            aurora_veil=self.aurora_veil[idx], tailwind=self.tailwind[idx],
            toxic_counter=self.toxic_counter[idx], boosts=self.boosts[idx],
            legal_action_mask=self.legal_action_mask[idx], can_tera=self.can_tera[idx])


    def replace_row(self, idx: int, new_side: 'BattleState'):
        team = self.team.replace_row(idx, new_side.team)
        active_index = self.active_index.at[idx].set(new_side.active_index)
        stealth_rocks = self.stealth_rocks.at[idx].set(new_side.stealth_rocks)
        sticky_webs = self.sticky_webs.at[idx].set(new_side.sticky_webs)
        spikes = self.spikes.at[idx].set(new_side.spikes)
        toxic_spikes = self.toxic_spikes.at[idx].set(new_side.toxic_spikes)
        reflect = self.reflect.at[idx].set(new_side.reflect)
        light_screen = self.light_screen.at[idx].set(new_side.light_screen)
        aurora_veil = self.aurora_veil.at[idx].set(new_side.aurora_veil)
        tailwind = self.tailwind.at[idx].set(new_side.tailwind)
        toxic_counter = self.toxic_counter.at[idx].set(new_side.toxic_counter)
        boosts = self.boosts.replace_row(idx, new_side.boosts)
        volatile_status = self.volatile_status.replace_row(idx, new_side.volatile_status)
        return self.replace(
            team=team, active_index=active_index, stealth_rocks=stealth_rocks, sticky_webs=sticky_webs, spikes=spikes,
            toxic_spikes=toxic_spikes, reflect=reflect, light_screen=light_screen, aurora_veil=aurora_veil,
            tailwind=tailwind, toxic_counter=toxic_counter, boosts=boosts, volatile_status=volatile_status
        )


    @property
    def active(self) -> Pokemon:
        flattened_idx = jnp.squeeze(self.active_index)
        return self.team[[0,1], flattened_idx]

    @property
    def raw_boosted_stats(self):
        stats = self.active.stats
        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + self.boosts.normal_boosts])

    @property
    def boosted_stats(self):
        active = self.active
        ability = active.ability
        status = active.status
        stats = self.active.stats
        # squeeze removes dimensions with length 1 which makes this broadcast correctly
        # its probably going to be helpful to use this in other places

        # marvel scale
        temp = conditional_mult_round(stats[...,StatEnum.SPECIAL_DEFENSE], 1.5,
                                      jnp.logical_and(ability==AbilityEnum.MARVEL_SCALE, active.has_status).squeeze())
        stats = stats.at[..., StatEnum.SPECIAL_DEFENSE].set(temp)

        # surge surfer
        terrain = self.terrain.terrain
        is_surge_surfer = jnp.logical_and(
            ability==AbilityEnum.SURGE_SURFER,
            terrain==TerrainEnum.ELECTRIC).squeeze()

        # speed boosting weather abilities
        weather = self.weather.weather
        is_chlorophyll  = jnp.logical_and(
            ability==AbilityEnum.CHLOROPHYLL,
            weather==WeatherEnum.SUN).squeeze()
        is_swift_swim = jnp.logical_and(
            ability == AbilityEnum.SWIFT_SWIM,
            weather == WeatherEnum.RAIN).squeeze()
        is_slush_rush = jnp.logical_and(
            ability == AbilityEnum.SLUSH_RUSH,
            weather == WeatherEnum.SNOW).squeeze()
        is_sand_rush = jnp.logical_and(
            ability == AbilityEnum.SAND_RUSH,
            weather == WeatherEnum.SANDSTORM).squeeze()
        arr = jnp.array([is_chlorophyll, is_slush_rush, is_sand_rush, is_swift_swim, is_surge_surfer])
        temp = conditional_mult_round(
            stats[..., StatEnum.SPEED],
            2,
            jnp.any(arr))


        # paralyzed
        is_paralyzed = (status==Status.PARALYZE).squeeze()
        # quick feet
        is_quick_feet = jnp.logical_and(ability==AbilityEnum.QUICK_FEET, active.has_status).squeeze()
        temp = conditional_mult_round(temp, 1.5, is_quick_feet)
        # apply paralyze speed drop if paralyzed and not quick feet
        temp = conditional_mult_round(temp, 0.5, jnp.logical_and(is_paralyzed, 1-is_quick_feet))

        stats = stats.at[..., StatEnum.SPEED].set(temp)


        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6+self.boosts.normal_boosts])

    @property
    def accuracy_boosts(self):
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        return self.active.current_hp

    def legal_switch_mask(self):
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]


def set_boosts(state: BattleState, side_idx, new_boosts):
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    return state.replace(boosts=new_boosts)

def clear_boosts(state: BattleState, side_idx) -> BattleState:
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idx, val) -> BattleState:
    # contrary
    idx = jnp.asarray(idx)
    val = jnp.asarray(val) * (2 * (state.active.ability[side_idx]!=AbilityEnum.CONTRARY)[0]-1)
    if idx.ndim == 0:
        val = val.reshape(())
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idx].add(val)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def reduce_boosts(state: BattleState, side_idx, idx, val, bounceable=False) -> BattleState:
    ability = state.active.ability
    not_clear_body = (ability[side_idx]!=AbilityEnum.CLEAR_BODY)[0]
    # TODO: this doesn't account for the functionality of idx being an array
    # in which case it should apply len(idx) times but that is hard to implement in JAX
    is_defiant = ability[side_idx]==AbilityEnum.DEFIANT
    is_competitive = ability[side_idx]==AbilityEnum.COMPETITIVE
    state = jax.lax.cond(is_defiant.reshape(()),
                 add_boosts,
                 lambda b, s, i, v: b,
                 state, side_idx, StatEnum.ATTACK, 2)
    state = jax.lax.cond(is_competitive.reshape(()),
                 add_boosts,
                 lambda b, s, i, v: b,
                 state, side_idx, StatEnum.SPECIAL_ATTACK, 2)
    return add_boosts(state, side_idx, idx, -val * not_clear_body)

def conditional_set_boosts(state: BattleState, side_idx, new_boosts, cond):
    return jax.lax.cond(jnp.asarray(cond).reshape(()), set_boosts,
                        lambda state, side_idx, new_boosts: state,
                        state, side_idx, new_boosts)

def conditional_add_boosts(state: BattleState, side_idx, cond, idx, val) -> BattleState:
    return jax.lax.cond(jnp.asarray(cond).reshape(()), add_boosts,
                        lambda state, side_idx, idx, val: state,
                        state, side_idx, idx, val)

def conditional_reduce_boosts(state: BattleState, side_idx, cond, idx, val, bounceable) -> BattleState:
    return jax.lax.cond(jnp.asarray(cond).reshape(()), reduce_boosts,
                        lambda state, side_idx, idx, val, bounceable: state,
                        state, side_idx, idx, val, bounceable)

def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    new_team = state.team.replace_row((side_idx, state[side_idx].active_index), new_mon)
    return state.replace(team=new_team)

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    # TODO
    return state


def set_terrain(
    state: BattleState, new_terrain: TerrainEnum, duration: int
    ):
    current_terrain = state.terrain.terrain
    requested = jnp.asarray(new_terrain, jnp.int32).reshape(())
    new_terrain = jnp.where(requested == TerrainEnum.NONE, current_terrain, requested)
    matching_terrain = current_terrain == new_terrain
    new_duration_terrain = duration * (1 - matching_terrain) + state.terrain.duration * matching_terrain
    state = state.replace(terrain=Terrain(new_terrain, new_duration_terrain))
    return state

def set_weather(
    state: BattleState, new_weather: WeatherEnum, duration: int):
    current_weather = state.weather.weather
    requested = jnp.asarray(new_weather, jnp.int32).reshape(())
    new_weather = jnp.where(requested == WeatherEnum.NONE, current_weather, requested)
    matching_weather = current_weather == new_weather
    # TODO: add item check for the weather stones
    new_duration_weather = duration * (1 - matching_weather) + state.weather.duration * matching_weather
    return state.replace(weather=Weather(new_weather, new_duration_weather))

def status_helper(key, active, status: Status):
    key, sub_key = random.split(key, 2)
    # makes sleep turns between 1 and 3 equally likely
    turns = random.randint(sub_key, (1,), minval=2, maxval=5)[0]
    active = active.replace(status=status, sleep_counter=turns*(status==Status.SLEEP))
    return key, active



def set_status(key: chex.PRNGKey, state: BattleState, side_idx, status: Status) -> Tuple[chex.PRNGKey, BattleState]:
    active = state.active[side_idx]
    already_statused = active.status != Status.NONE
    is_comatose =  active.ability==AbilityEnum.COMATOSE
    is_immune = jnp.any(jnp.array([
        jnp.logical_and(jnp.logical_or(status==Status.POISON, status==Status.TOXIC), active.is_poison_immune),
        jnp.logical_and(status==Status.PARALYZE, active.is_paralyze_immune),
        jnp.logical_and(status==Status.BURN, active.is_burn_immune),
        jnp.logical_and(status==Status.FREEZE, active.is_freeze_immune),
        jnp.logical_and(status==Status.SLEEP, active.is_sleep_immune),
        is_comatose
    ]))
    key, active = jax.lax.cond(
        triple_or(already_statused, is_immune, status==Status.NONE)[0],
        lambda k, a, s: (k, a),
        status_helper,
        key, active, status)
    return key, update_active(state, side_idx, active)

def take_damage_value(state: BattleState, defender_idx: int, damage: Array, is_attack_damage) -> BattleState:
    return _take_damage_value(state, defender_idx, damage, is_attack_damage, True)

def _take_damage_value(state, defender_idx, damage, is_attack_damage, allow_innards):
    active = state.active
    defender = active[defender_idx]
    attacker = active[1-defender_idx]
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    health_full = defender.current_hp==defender.max_hp
    is_sturdy = defender.ability==AbilityEnum.STURDY
    is_multiscale = jnp.logical_and(health_full, defender.ability==AbilityEnum.MULTISCALE)
    damage = conditional_mult_round(damage, 0.5, is_multiscale)
    old_health = defender.current_hp
    over_half = jnp.greater_equal(old_health/defender.max_hp, 0.5)
    new_health = jax.lax.clamp(0, old_health-damage, (defender.max_hp - health_full*is_sturdy*is_attack_damage)[0])
    under_half = jnp.less(new_health/defender.max_hp, 0.5)

    # check to make sure we don't accidentally revive a pokemon
    alive = jnp.logical_and(new_health != 0, defender.is_alive)
    defender = defender.replace(current_hp=new_health, is_alive=alive)

    # run moxie type abilities
    is_soul_heart = attacker.ability==AbilityEnum.SOUL_HEART
    is_moxie = attacker.ability==AbilityEnum.MOXIE
    is_beast_boost = attacker.ability==AbilityEnum.BEAST_BOOST
    is_grim_neigh = attacker.ability==AbilityEnum.GRIM_NEIGH
    beast_boost_index = (1+jnp.argmax(attacker.stats))
    index = (beast_boost_index*is_beast_boost +
             is_moxie * StatEnum.ATTACK +
             jnp.logical_or(is_soul_heart, is_grim_neigh) * StatEnum.SPECIAL_ATTACK)
    # yeah idk why this needs double index thingy here
    cond = jnp.logical_and(jnp.logical_or(is_moxie, is_beast_boost), 1-alive)[0]
    state = conditional_add_boosts(state, 1-defender_idx, cond, index, 1)

    # innards out
    is_innards_out = triple_and(1-alive, defender.ability==AbilityEnum.INNARDS_OUT, is_attack_damage)
    if allow_innards:
        state = jax.lax.cond(
            is_innards_out[0],
            lambda b, i, d: _take_damage_value(b, i, d, False, False),
            lambda b, i, d: b,
            state, 1-defender_idx, old_health
        )
    # TODO: consider making separate functions for take damage value and take damage from attack
    # this keeps active the same if current_hp!=0 and sets field as empty otherwise
    # there are some other conditions that should trigger emptying field like eject button
    # idk if that should be handled here or elsewhere
    return update_active(state, defender_idx, defender)

def boosted_stats_helper(stats, boosts, ignore_drops, ignore_boosts):
    # this probably works
    boosts = boosts.normal_boosts
    boosts = jnp.minimum(boosts, 13 - 7 * ignore_boosts)
    boosts = jnp.maximum(boosts, 6 * ignore_drops)
    return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + boosts])

def raw_boosted_stats(raw_stats, boosts, attacker_idx, attacker_unaware, defender_unaware, is_crit):
    attacker_stats = raw_stats[attacker_idx]
    defender_stats = raw_stats[1 - attacker_idx]
    attacker_boosts = boosts[attacker_idx]
    defender_boosts = boosts[attacker_idx]
    attacker_stats = boosted_stats_helper(
        attacker_stats,
        attacker_boosts,
        is_crit,
        defender_unaware)
    defender_stats = boosted_stats_helper(
        defender_stats,
        defender_boosts,
        attacker_unaware,
        jnp.logical_or(is_crit, attacker_unaware)
    )
    return attacker_stats, defender_stats

def take_damage_percent(state: BattleState, defender_idx, percent: chex.Array) -> BattleState:
    if jnp.ndim(defender_idx) > 0:
        for i in range(2):
            state = take_damage_percent(state, i, jnp.broadcast_to(percent, (2, 1))[i])
        return state
    damage = jnp.round(state.active.max_hp[defender_idx] * percent).astype(int)
    return take_damage_value(state, defender_idx, damage, False)





