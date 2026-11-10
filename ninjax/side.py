"""Immutable singles battle state and helpers for side and field updates.

The leading axis of a full battle is the two-player axis. Helpers return new
PyTrees, and random helpers take and return an explicit JAX key. The mechanics
implemented here are a subset of the intended battle rules.
"""

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

from ninjax.events import EventBuffer, EventKind, emit

Array = jax.Array



class VolatileStatus(struct.PyTreeNode):
    """Switch-reset effects for each active side, with shape (2, 1).

    Confusion and Transform fields reserve state; their mechanics are not yet implemented.
    """
    flash_fire: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))
    confusion_turns: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.int32))
    transformed: Array = field(default_factory=lambda: jnp.zeros((2, 1), dtype=jnp.bool_))

    def __getitem__(self, idx):
        """Select one side of every volatile-state leaf."""
        return jax.tree.map(lambda value: value[idx], self)

    def replace_row(self, idx: int, new_status: 'VolatileStatus'):
        """Return a copy with one side's volatile leaves replaced."""
        return jax.tree.map(lambda old, new: old.at[idx].set(new), self, new_status)

class BattleState(struct.PyTreeNode):
    """State for one singles battle, without an outer environment batch axis.

    ``team`` has player and team-slot axes ``(2, 6)``; most side conditions
    and ``active_index`` have shape ``(2, 1)``. Normal stat stages have shape
    ``(2, 6)``, accuracy/evasion stages ``(2, 2)``, and the legacy action mask
    ``(2, 15)``. ``roster`` selects participating team slots. Weather, terrain,
    and turn counters describe the shared field. Independent battles can be
    batched by applying ``jax.vmap`` outside these player axes.

    ``winner`` is -1 until set, otherwise a player index; ``perish_turns`` uses
    -1 for an inactive counter. This state stores mechanics, not observations.
    """
    team: Pokemon
    events: EventBuffer = field(default_factory=EventBuffer.empty)
    event_source: Array = field(default_factory=lambda: jnp.int32(-1))
    event_cause: Array = field(default_factory=lambda: jnp.int32(0))
    event_move: Array = field(default_factory=lambda: jnp.int32(0))
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
        """Select side-indexed fields while retaining shared field values.

        This is a convenience view for side helpers, not a complete independent
        battle: fields not explicitly indexed here retain their original shape.
        """
        return self.replace(
            team=self.team[idx], active_index=self.active_index[idx],
            stealth_rocks=self.stealth_rocks[idx], sticky_webs=self.sticky_webs[idx],
            spikes=self.spikes[idx], toxic_spikes=self.toxic_spikes[idx],
            reflect=self.reflect[idx], light_screen=self.light_screen[idx],
            aurora_veil=self.aurora_veil[idx], tailwind=self.tailwind[idx],
            toxic_counter=self.toxic_counter[idx], boosts=self.boosts[idx],
            legal_action_mask=self.legal_action_mask[idx], can_tera=self.can_tera[idx],
            volatile_status=self.volatile_status[idx])


    def replace_row(self, idx: int, new_side: 'BattleState'):
        """Replace the side fields listed below; shared field values are retained.

        Fields not listed in the replacement, including roster and pending switch
        information, remain unchanged.
        """
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
        """Gather the active Pokémon for both players from their team slots."""
        flattened_idx = jnp.squeeze(self.active_index)
        return self.team[[0,1], flattened_idx]

    @property
    def raw_boosted_stats(self):
        """Apply stage multipliers to active stats without ability modifiers."""
        stats = self.active.stats
        return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + self.boosts.normal_boosts])

    @property
    def boosted_stats(self):
        """Return active stats with the ability and status modifiers handled here.

        The result has player and stat axes ``(2, 6)``. This property is used
        for action ordering; the damage path separately applies its modifiers.
        Ability coverage is incomplete.
        """
        active = self.active
        ability = active.ability
        status = active.status
        stats = self.active.stats
        # squeeze removes dimensions with length 1 which makes this broadcast correctly
        # its probably going to be helpful to use this in other places

        # marvel scale
        temp = conditional_mult_round(stats[...,StatEnum.DEFENSE], 1.5,
                                      jnp.logical_and(ability==AbilityEnum.MARVEL_SCALE, active.has_status).squeeze())
        stats = stats.at[..., StatEnum.DEFENSE].set(temp)

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
        # Reduce over ability predicates, preserving the player axis so one
        # Pokémon's weather ability cannot affect its opponent's speed.
        arr = jnp.array([is_chlorophyll, is_slush_rush, is_sand_rush, is_swift_swim, is_surge_surfer])
        temp = conditional_mult_round(
            stats[..., StatEnum.SPEED],
            2,
            jnp.any(arr, axis=0))


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
        """Return accuracy and evasion stages for both players, shape ``(2, 2)``."""
        return self.boosts.acc_boosts

    @property
    def current_hp(self):
        """Return the active Pokémon's current HP for each player."""
        return self.active.current_hp

    def legal_switch_mask(self):
        """Build the legacy six-slot switch mask for a side view.

        This helper returns a Python list and does not apply roster selection;
        the environment's action mask is the authoritative decision mask.
        """
        # TODO: replace list comprehension with some kind of jax control flow
        return [self.team[j].is_alive * (j != self.active_index) for j in range(6)]


def set_boosts(state: BattleState, side_idx, new_boosts):
    """Replace one side's stat-stage record and return the updated state."""
    old = jnp.concatenate((state.boosts.normal_boosts[side_idx], state.boosts.acc_boosts[side_idx]))
    new_boosts = state.boosts.replace_row(side_idx, new_boosts)
    state = state.replace(boosts=new_boosts)
    new = jnp.concatenate((new_boosts.normal_boosts[side_idx], new_boosts.acc_boosts[side_idx]))
    def record(stat, state):
        a, b = old[stat], new[stat]
        return emit(state, EventKind.BOOST, side_idx, stat, a, b, enabled=a != b)
    return jax.lax.fori_loop(0, 8, record, state)


def clear_boosts(state: BattleState, side_idx) -> BattleState:
    """Reset one side's normal and accuracy/evasion stages."""
    return set_boosts(state, side_idx, StatBoosts())

def add_boosts(state: BattleState, side_idx, idx, val) -> BattleState:
    """Add normal stat stages, reversing their sign for Contrary.

    ``idx`` may be a scalar or an array of stat indices. ``StatBoosts`` clips
    the resulting stages to the supported range when constructed.
    """
    # contrary
    idx = jnp.asarray(idx)
    val = jnp.asarray(val) * (2 * (state.active.ability[side_idx]!=AbilityEnum.CONTRARY)[0]-1)
    if idx.ndim == 0:
        val = val.reshape(())
    new_boosts = state.boosts.normal_boosts[side_idx]
    new_boosts = new_boosts.at[idx].add(val)
    return set_boosts(state, side_idx, StatBoosts(normal_boosts=new_boosts, acc_boosts=state.boosts.acc_boosts[side_idx]))

def reduce_boosts(state: BattleState, side_idx, idx, val, bounceable=False) -> BattleState:
    """Apply the currently implemented stage-reduction and ability handling.

    Clear Body, Defiant, and Competitive handling is incomplete; in particular,
    the caller's source and ``bounceable`` are not yet used to distinguish all
    opposing and self-inflicted reductions.
    """
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
    """Set stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), set_boosts,
                        lambda state, side_idx, new_boosts: state,
                        state, side_idx, new_boosts)

def conditional_add_boosts(state: BattleState, side_idx, cond, idx, val) -> BattleState:
    """Add stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), add_boosts,
                        lambda state, side_idx, idx, val: state,
                        state, side_idx, idx, val)

def conditional_reduce_boosts(state: BattleState, side_idx, cond, idx, val, bounceable) -> BattleState:
    """Reduce stages when the scalar JAX predicate ``cond`` is true."""
    return jax.lax.cond(jnp.asarray(cond).reshape(()), reduce_boosts,
                        lambda state, side_idx, idx, val, bounceable: state,
                        state, side_idx, idx, val, bounceable)

def update_active(state: BattleState, side_idx, new_mon: Pokemon) -> BattleState:
    """Write a modified active Pokémon back into its team's current slot."""
    new_team = state.team.replace_row((side_idx, state[side_idx].active_index), new_mon)
    old = state.active.status[side_idx, 0]
    state = state.replace(team=new_team)
    return emit(state, EventKind.STATUS, side_idx, before=old, after=new_mon.status[0], enabled=old != new_mon.status[0])

def clear_volatile_status(state: BattleState, side_idx) -> BattleState:
    """Reset all represented switch-cleared effects for one active side."""
    volatile = jax.tree.map(lambda value: value.at[side_idx].set(jnp.zeros_like(value[side_idx])), state.volatile_status)
    return state.replace(volatile_status=volatile)


def set_terrain(
    state: BattleState, new_terrain: TerrainEnum, duration: int
    ):
    """Set a non-NONE terrain, retaining its duration if it already matches."""
    current_terrain = state.terrain.terrain
    requested = jnp.asarray(new_terrain, jnp.int32).reshape(())
    new_terrain = jnp.where(requested == TerrainEnum.NONE, current_terrain, requested)
    matching_terrain = current_terrain == new_terrain
    new_duration_terrain = duration * (1 - matching_terrain) + state.terrain.duration * matching_terrain
    state = state.replace(terrain=Terrain(new_terrain, new_duration_terrain))
    return state

def set_weather(
    state: BattleState, new_weather: WeatherEnum, duration: int):
    """Set non-NONE weather, retaining its duration if it already matches."""
    current_weather = state.weather.weather
    requested = jnp.asarray(new_weather, jnp.int32).reshape(())
    new_weather = jnp.where(requested == WeatherEnum.NONE, current_weather, requested)
    matching_weather = current_weather == new_weather
    # TODO: add item check for the weather stones
    new_duration_weather = duration * (1 - matching_weather) + state.weather.duration * matching_weather
    return state.replace(weather=Weather(new_weather, new_duration_weather))

def status_helper(key, active, status: Status):
    """Set an active Pokémon's status and initialize its sleep counter.

    The counter is sampled from 2 through 4 inclusive for sleep and set to zero
    for other statuses. The returned key has been advanced once.
    """
    key, sub_key = random.split(key, 2)
    # The move transition decrements this counter before checking sleep.
    turns = random.randint(sub_key, (1,), minval=2, maxval=5)[0]
    active = active.replace(status=status, sleep_counter=turns*(status==Status.SLEEP))
    return key, active



def set_status(key: chex.PRNGKey, state: BattleState, side_idx, status: Status) -> Tuple[chex.PRNGKey, BattleState]:
    """Attempt a major status, returning the key and updated battle state.

    Existing status, the represented type/ability immunities, and a NONE status
    prevent the update. A key is consumed only when ``status_helper`` runs.
    """
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
    """Apply signed HP damage to an active Pokémon and return updated state.

    Negative values heal. ``is_attack_damage`` controls the represented Sturdy,
    Multiscale, and Innards Out behavior; other damage-triggered effects remain
    incomplete. Fainted Pokémon are not marked alive again by healing.
    """
    return _take_damage_value(state, defender_idx, damage, is_attack_damage, True)

def _take_damage_value(state, defender_idx, damage, is_attack_damage, allow_innards):
    """Implement HP updates; optionally allow a single Innards Out response."""
    active = state.active
    defender = active[defender_idx]
    attacker = active[1-defender_idx]
    # TODO: there are some effects that trigger based on damage taken, like mirror coat
    health_full = defender.current_hp==defender.max_hp
    is_sturdy = defender.ability==AbilityEnum.STURDY
    is_multiscale = triple_and(health_full, defender.ability==AbilityEnum.MULTISCALE, is_attack_damage)
    damage = conditional_mult_round(damage, 0.5, is_multiscale)
    old_health = defender.current_hp
    over_half = jnp.greater_equal(old_health/defender.max_hp, 0.5)
    new_health = jax.lax.clamp(0, old_health-damage, jnp.asarray(defender.max_hp).reshape(()))
    sturdy_survives = health_full & is_sturdy & jnp.asarray(is_attack_damage) & (damage > 0) & (new_health == 0)
    new_health = jnp.maximum(new_health, sturdy_survives.astype(new_health.dtype))
    under_half = jnp.less(new_health/defender.max_hp, 0.5)

    # check to make sure we don't accidentally revive a pokemon
    alive = jnp.logical_and(new_health != 0, defender.is_alive)
    defender = defender.replace(current_hp=new_health, is_alive=alive)

    new_health = defender.current_hp
    delta = (new_health - old_health).reshape(())
    # Public HP endpoints use the same rounding as observations.
    public_delta = (jnp.ceil(new_health / defender.max_hp * 100) - jnp.ceil(old_health / defender.max_hp * 100)) / 100
    state = emit(state, EventKind.HP, defender_idx, detail=jnp.asarray(is_attack_damage, jnp.int32),
                 before=old_health, after=new_health, hp_change=delta,
                 hp_fraction_change=public_delta, enabled=delta != 0)
    state = emit(state, EventKind.FAINT, defender_idx, enabled=(old_health > 0) & (new_health == 0))

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
    # This flag is a static Python value: tracing the response with it disabled
    # prevents an unbounded recursive JAX trace.
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
    """Apply stages after optionally ignoring negative or positive stages."""
    boosts = boosts.normal_boosts
    boosts = jnp.minimum(boosts, jnp.where(ignore_boosts, 0, 6))
    boosts = jnp.maximum(boosts, jnp.where(ignore_drops, 0, -6))
    return jnp.floor(stats * STAT_MULTIPLIER_LOOKUP[6 + boosts])

def raw_boosted_stats(raw_stats, boosts, attacker_idx, attacker_unaware, defender_unaware, is_crit):
    """Return attacker and defender stats with the represented stage rules.

    Critical hits and Unaware alter which stages are applied. Ability coverage
    here is incomplete; callers apply additional modifiers separately.
    """
    attacker_stats = raw_stats[attacker_idx]
    defender_stats = raw_stats[1 - attacker_idx]
    attacker_boosts = boosts[attacker_idx]
    defender_boosts = boosts[1 - attacker_idx]
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

def take_damage_percent(state: BattleState, defender_idx, percent: chex.Array, cause=None) -> BattleState:
    """Apply a signed fraction of max HP as indirect damage or healing.

    Nonzero fractions use at least one HP after rounding the magnitude down.
    An array of indices invokes a fixed JAX loop over both sides, sharing the
    HP-update graph instead of expanding it twice during compilation.
    """
    if cause is not None:
        original_cause = state.event_cause
        state = take_damage_percent(state.replace(event_cause=jnp.int32(cause)), defender_idx, percent)
        return state.replace(event_cause=original_cause)
    if jnp.ndim(defender_idx) > 0:
        fractions = jnp.broadcast_to(percent, (2, 1))
        return jax.lax.fori_loop(0, 2,
            lambda i, s: take_damage_percent(s, i, fractions[i]), state)
    amount = state.active.max_hp[defender_idx] * percent
    damage = (jnp.sign(amount) * jnp.maximum(1, jnp.floor(jnp.abs(amount)))).astype(jnp.int32)
    return take_damage_value(state, defender_idx, damage, False)





