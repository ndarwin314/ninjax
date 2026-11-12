"""Battle transition helpers for moves, switching, and end-of-turn effects.

Functions return new battle state rather than modifying it in place. Random
transitions accept an explicit JAX key and generally return the advanced key.
The current implementation targets singles and covers only part of the rules.
"""

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


from ninjax.events import EventKind, Cause, emit

Array = chex.Array

# Scarlet/Violet Unseen Fist deals full damage through Protect. Change this
# modifier for rulesets that reduce the damage of a successful bypass.
UNSEEN_FIST_PROTECT_MULTIPLIER = 1.0



def do_damaging_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Resolve a damaging move and finalize it once.

    ``attacker_idx`` is a player index in a singles battle; ``stat_index`` and
    ``boost_value`` are unused operands shared with the other hit branches.
    Critical-hit and damage-roll keys are split independently for each hit.
    Per-hit accuracy and hit-count policy are carried by the move metadata.
    """
    return resolve_multi_hit(key, state, attacker_idx, move)


def sample_hit_count(key, rule, ability, item):
    """Sample curated hit-count policies; return an advanced key and count."""
    from ninjax.enum_types import ItemEnum
    key, count_key, extra_key = random.split(key, 3)
    roll = random.randint(count_key, (), 0, 20)
    ordinary = jnp.where(roll < 7, 2, jnp.where(roll < 14, 3, jnp.where(roll < 17, 4, 5)))
    dice = item == ItemEnum.LOADED_DICE
    skill = ability == AbilityEnum.SKILL_LINK
    standard = jnp.where(skill, 5, jnp.where(dice, random.randint(extra_key, (), 4, 6), ordinary))
    population = jnp.where(dice, random.randint(extra_key, (), 4, 11), 10)
    return key, jnp.where(rule == 1, 1, jnp.where(rule == 2, standard, jnp.where(rule == 3, 3, population)))


def resolve_multi_hit(key, state: BattleState, attacker_idx, move: Move):
    """Resolve up to ten hits in a fixed scan; initial accuracy was already checked."""
    from ninjax.enum_types import ItemEnum
    from ninjax.side import conditional_add_boosts, conditional_reduce_boosts
    attacker = state.active[attacker_idx]
    defender = state.active[1-attacker_idx]
    key, target_hits = jax.lax.cond(move.hit_rule[0] == 1,
        lambda k: (k, jnp.int32(1)),
        lambda k: sample_hit_count(k, move.hit_rule[0], attacker.ability[0], attacker.item[0]), key)
    # Loaded Dice and Skill Link remove multiaccuracy in the pinned generation.
    per_hit = move.per_hit_accuracy[0] & (attacker.ability[0] != AbilityEnum.SKILL_LINK) & (attacker.item[0] != ItemEnum.LOADED_DICE)
    no_recoil = move.replace(recoil_percent=jnp.where(move.hit_rule == 1, move.recoil_percent, 0))
    def body(carry, index):
        """Execute at most one hit, retaining state when the move has stopped."""
        key, state, alive, total, hits = carry
        # A contact effect can put the attacker to sleep between hits. Sleep
        # Talk's sleep-usable exception is deferred with Sleep Talk itself.
        awake = (index == 0) | (state.active.status[attacker_idx, 0] != Status.SLEEP)
        can_hit = (state.active.current_hp[attacker_idx, 0] > 0) & (state.active.current_hp[1-attacker_idx, 0] > 0) & awake
        valid = alive & (index < target_hits) & ((move.hit_rule[0] == 1) | can_hit)
        def attempt(args):
            """Check later-hit accuracy and apply its result."""
            key, state, total, hits = args
            def accuracy_check(k):
                """Advance the key for this independent accuracy check."""
                k, subkey = random.split(k)
                return k, random.uniform(subkey) < move_accuracy(state, attacker_idx, move)[0]
            key, hit = jax.lax.cond(per_hit & (index > 0), accuracy_check,
                                    lambda k: (k, jnp.bool_(True)), key)
            def apply(args):
                """Apply this hit at its indexed base power."""
                key, state, total, hits = args
                power = jnp.where(move.escalating_power, move.base_power * (index+1), move.base_power)
                one = no_recoil.replace(base_power=power)
                key, state, damage, _ = resolve_damaging_hit(key, state, attacker_idx, one)
                return key, state, total + damage, hits + 1
            key, state, total, hits = jax.lax.cond(hit, apply, lambda x: x, (key, state, total, hits))
            state = jax.lax.cond(~hit, lambda s: emit(s, EventKind.MISS, 1-attacker_idx), lambda s: s, state)
            return key, state, hit, total, hits
        return jax.lax.cond(valid, attempt, lambda x: (x[0], x[1], jnp.bool_(False), x[2], x[3]),
                            (key, state, total, hits)), None
    (key, state, _, damage, hits), _ = jax.lax.scan(body,
        (key, state, jnp.bool_(True), jnp.zeros((1,), jnp.int32), jnp.int32(0)), jnp.arange(10))
    # The currently supported multi-hit moves have no recoil. Calculated damage
    # is retained here for future recoil support; exact HP lost is not yet tracked.
    # Scale Shot (Showdown move ID 799) lowers Defense and raises Speed after hitting.
    state = jax.lax.cond((move.identifier[0] == 799) & (hits > 0) & (state.active.current_hp[attacker_idx, 0] > 0),
        lambda s: conditional_add_boosts(conditional_reduce_boosts(s, attacker_idx, True, StatEnum.DEFENSE, 1, False),
                                         attacker_idx, True, StatEnum.SPEED, 1), lambda s: s, state)
    return key, finish_damaging_move(state, attacker_idx, move, damage, defender)


def resolve_damaging_hit(key, state: BattleState, attacker_idx, move: Move):
    """Apply one hit; return key, state, calculated damage and defender snapshot.

    Damage has shape (1,) and may exceed the target's remaining HP. The defender
    snapshot is taken before this hit. A multi-hit caller must track actual HP
    lost separately; comparing final HP is insufficient when an item heals
    during a hit. Keys advance exactly as in the existing single-hit path.

    Recoil remains here for single-hit moves to preserve existing event order.
    The multi-hit caller suppresses it; multi-hit recoil requires an explicit
    actual-damage total and is not yet supported.
    """
    active = state.active
    attacker = active[attacker_idx]
    defender = active[1-attacker_idx]

    from ninjax.held_effects import effective_weather
    # base power modifications, technician, tera, terrain etc
    power = compute_base_power(
        attacker,
        defender,
        move,
        state.terrain, effective_weather(state), state.gravity_duration > 0
    )
    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_stage = (move.crit_stage+
                  (attacker.ability==AbilityEnum.SUPER_LUCK)[0] +
                  3*jnp.logical_and(attacker.ability==AbilityEnum.MERCILESS, defender.is_poisoned))
    crit_chance = CRIT_STAGES[crit_stage] * (defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = ((random.uniform(one) < crit_chance) | move.always_crit) & (defender.ability != AbilityEnum.BATTLE_ARMOR)
    state = emit(state, EventKind.CRITICAL, 1-attacker_idx, enabled=is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100

    # Select stage-adjusted stats before computing offensive and defensive
    # modifiers. Moves such as Body Press can select a nonstandard attack stat.
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


    from ninjax.held_effects import modify_stats, effective_weather, after_hit
    from ninjax.enum_types import ItemEnum
    attacker_stats = modify_stats(state, attacker_idx, attacker_stats)
    defender_stats = modify_stats(state, 1-attacker_idx, defender_stats)
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
        defender.status,
        state.volatile_status.flash_fire[attacker_idx]
    )

    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    damage = compute_damage_multipliers(
        key,
        attacker,
        defender,
        Weather(effective_weather(state), state.weather.duration),
        move,
        base_damage,
        is_crit,
        damage_roll,
        state.gravity_duration > 0
    )

    damage = damage_post_modifiers(damage, defender.ability, move)
    screen = jnp.where(move.move_type == MoveType.PHYSICAL, state.reflect[1-attacker_idx, 0] > 0,
                       state.light_screen[1-attacker_idx, 0] > 0) | (state.aurora_veil[1-attacker_idx, 0] > 0)
    damage = conditional_mult_round(damage, .5, screen & ~is_crit)
    damage = conditional_mult_round(damage, 5324/4096, attacker.item == ItemEnum.LIFE_ORB)
    bypass = state.volatile_status.protected[1-attacker_idx] & (attacker.ability == AbilityEnum.UNSEEN_FIST) & move.contact
    damage = conditional_mult_round(damage, UNSEEN_FIST_PROTECT_MULTIPLIER, bypass)

    # dealing damage
    state = take_damage_value(state, 1 - attacker_idx, damage, True)


    # lax.cond keeps both branches compatible with traced move and ability data.
    state = jax.lax.cond(
        jnp.logical_and(move.recoil, attacker.ability!=AbilityEnum.ROCK_HEAD)[0],
        do_recoil, lambda s, p, d: s,
        state, attacker_idx, jnp.trunc(damage*move.recoil_percent))

    key, state = after_every_hit(key, state, attacker_idx, move, is_crit)
    state = after_hit(state, attacker_idx, move, damage, calculate_effectiveness_multiplier(move.type, defender.type_list))
    return key, state, damage, defender


def finish_damaging_move(state: BattleState, attacker_idx, move: Move, damage, defender: Pokemon) -> BattleState:
    """Run existing once-per-move effects and resolve simultaneous knockouts.

    ``damage`` is the sum of calculated damage, not HP actually lost. Effects
    requiring actual HP lost need a separate total before they can be added.
    ``defender`` is the snapshot from before the first hit.
    """
    from ninjax.held_effects import after_move
    state = after_move(state, attacker_idx, move, damage[0] > 0)
    state = after_move_finished(state, defender, 1-attacker_idx, defender.current_hp)
    state = jax.lax.cond(move.self_ko[0],
        lambda s: take_damage_value(s, attacker_idx, s.active[attacker_idx].current_hp, False),
        lambda s: s, state)
    # Retain the cause-dependent winner for a simultaneous final knockout;
    # the environment uses this value when it constructs the terminal reward.
    both_out = ~jnp.any((state.team.current_hp[..., 0] > 0) & state.roster)
    winner = jnp.where(move.self_ko[0], 1-attacker_idx, attacker_idx)
    state = state.replace(winner=jnp.where(both_out, winner, state.winner))

    return state

def do_recoil(state: BattleState, attacker_idx: int, damage) -> BattleState:
    """Apply the supplied indirect recoil damage to the attacker."""
    original_cause = state.event_cause
    state = take_damage_value(state.replace(event_cause=jnp.int32(Cause.RECOIL)), attacker_idx, damage, False)
    state = state.replace(event_cause=original_cause)
    return state

def do_status_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply the generic status-move fields and return the key and state.

    The selected target can receive healing, status, stat stages, and hazards;
    weather, terrain, and Perish Song update field-wide state. Effect coverage
    is partial, and field durations are currently fixed at five turns.
    """
    # Consecutive Protect attempts succeed with probability 1 / 3**chain.
    key, protect_key = random.split(key)
    v = state.volatile_status
    protect = move.identifier[0] == 182
    success = protect & (random.uniform(protect_key) < jnp.power(1 / 3., v.protect_chain[attacker_idx, 0]))
    v = v.replace(protected=v.protected.at[attacker_idx, 0].set(success),
                  protect_chain=v.protect_chain.at[attacker_idx, 0].set(
                      jnp.where(protect & success, jnp.minimum(v.protect_chain[attacker_idx, 0] + 1, 20), 0)))
    state = state.replace(volatile_status=v)
    target = jnp.where(move.target_self[0], attacker_idx, 1 - attacker_idx)
    state = take_damage_percent(state, target, -move.healing_fraction)
    key, state = jax.lax.cond(move.status_effect[0] != Status.NONE,
        lambda k, s: set_status(k, s, target, move.status_effect),
        lambda k, s: (k, s), key, state)
    stat = move.boost_stat[0]
    stages = move.boost_stages[0]
    state = jax.lax.cond((stat >= 1) & (stat <= 5) & (stages != 0),
        lambda s: jax.lax.cond(stages > 0,
            lambda b: add_boosts(b, target, stat, stages),
            lambda b: reduce_boosts(b, target, stat, -stages), s),
        lambda s: s, state)
    state = set_weather(state, move.weather_effect[0], 5)
    state = set_terrain(state, move.terrain_effect[0], 5)
    state = state.replace(perish_turns=jnp.where(move.perish_effect[0] & (state.perish_turns < 0),
                                                4, state.perish_turns))
    # Hazard IDs encode none, Stealth Rock, Sticky Web, Spikes, and Toxic
    # Spikes. Layered hazards are capped at their represented maximum.
    hazard = move.hazard_effect[0]
    state = state.replace(
        stealth_rocks=state.stealth_rocks.at[target].set(state.stealth_rocks[target] | (hazard == 1)),
        sticky_webs=state.sticky_webs.at[target].set(state.sticky_webs[target] | (hazard == 2)),
        spikes=state.spikes.at[target].set(jnp.minimum(3, state.spikes[target] + (hazard == 3))),
        toxic_spikes=state.toxic_spikes.at[target].set(jnp.minimum(2, state.toxic_spikes[target] + (hazard == 4))))
    return key, state

# this is for when water absorb or volt absorb is triggered
def do_healing_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Heal the defending active by a quarter of max HP for an absorb ability."""
    state = take_damage_percent(state, 1-attacker_idx, -1/4)
    return key,state

def do_stat_boost_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply an absorption ability's stage change to the defending active."""
    state = add_boosts(state, 1-attacker_idx, stat_index, boost_value)
    return key, state


def do_flash_fire_from_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Activate the defending active's switch-cleared Flash Fire flag."""
    volatile = state.volatile_status.replace(
        flash_fire=state.volatile_status.flash_fire.at[1-attacker_idx].set(True))
    return key, state.replace(volatile_status=volatile)

def end_turn_damage(state: BattleState) -> BattleState:
    """Apply the represented weather, terrain, ability, and status HP effects.

    Both sides are updated through ``take_damage_percent``. The exact ordering,
    including speed-dependent ordering and simultaneous knockouts, is incomplete.
    """
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
    from ninjax.held_effects import effective_weather
    weather = effective_weather(state)
    active = state.active
    idx = jnp.array([0,1])
    is_floating = active.is_floating
    is_sand_immune = active.is_sand_immune

    # sand damage
    sand_damage = (1 - is_sand_immune) / 16 * (weather == WeatherEnum.SANDSTORM)

    # rain abilities
    rain_healing = ((active.ability==AbilityEnum.RAIN_DISH) + 2 * (active.ability==AbilityEnum.DRY_SKIN)) / 16 * (weather == WeatherEnum.RAIN)
    # why is this one an int but the others arent

    # sun abilities
    sun_damage = ((active.ability==AbilityEnum.DRY_SKIN) + (active.ability==AbilityEnum.SOLAR_POWER)) / 8 * (weather == WeatherEnum.SUN)

    # snow abilities
    snow_healing = (active.ability==AbilityEnum.ICE_BODY) / 16 * (weather == WeatherEnum.SNOW)


    # grassy terrain healing
    grass_healing = (is_floating - 1) / 16 * (state.terrain.terrain == TerrainEnum.GRASSY)

    # status damage
    # technically this should be factored out to multiple bits since it goes burn poison toxic in priority
    status_damage = (1 / 8 * (active.status==Status.POISON) +
                     1 / 16 * (active.status==Status.BURN) +
                     state.toxic_counter / 16 * (active.status==Status.TOXIC))
    # Preserve effect-major, then player-major ordering while compiling one
    # shared HP-update body. All fractions use the original active snapshot.
    fractions = jnp.stack((sand_damage, -rain_healing, sun_damage,
                           -snow_healing, grass_healing, status_damage))
    causes = jnp.array([Cause.WEATHER] * 4 + [Cause.TERRAIN, Cause.STATUS], jnp.int32)
    def apply_effect(s, effect):
        percent, cause = effect
        return take_damage_percent(s, idx, percent, cause=cause), None
    return jax.lax.scan(apply_effect, state, (fractions, causes))[0]

# TODO: at some point probably factor out part of this into like
# just swapping out to implement baton pass idk
def swap_out(
    key,
    state: BattleState,
    side_idx,
    new_active: int | jax.Array
) -> BattleState:
    """Switch one side to a team slot and apply represented entry effects.

    Stages, volatile state, toxic count, and Perish Song count are reset before
    hazards and switch-in abilities run. A Pokémon fainting to entry hazards
    does not run ``swap_is_alive``. Switch and hazard coverage is incomplete.

    The supplied key is used for status effects; this helper currently returns
    only state, so its caller must manage key consumption separately.
    """
    # swaps the active pokemon and does appropriate things like
    # 1. clearing volatile statuses
    # 2. resting boosts
    # 3. probably stuff im forgetting
    # 4. ahhh palafin, ahhh regenerator

    state = state.replace(event_source=jnp.asarray(side_idx, jnp.int32), event_move=jnp.int32(0), event_cause=jnp.int32(Cause.SWITCH))
    state = emit(state, EventKind.SWITCH, side_idx, before=state.active_index[side_idx, 0], after=new_active)
    outgoing = state.active[side_idx]
    state = update_active(state, side_idx, outgoing.replace(type_list=outgoing.base_types))
    state = take_damage_percent(state, side_idx, jnp.where(
        (outgoing.ability == AbilityEnum.REGENERATOR) & (outgoing.current_hp > 0), -1/3, 0))
    # clear boosts and volatile status from active pokemon, maybe add baton pass check here?
    state = clear_volatile_status(clear_boosts(state, side_idx), side_idx)

    # update active pokemon index
    # this update could be saved since we need an update later but idk if it will matter
    active = state.active_index.at[side_idx].set(new_active)
    toxic_counter = state.toxic_counter.at[side_idx].set(0)
    state = state.replace(active_index=active, toxic_counter=toxic_counter,
                          perish_turns=state.perish_turns.at[side_idx].set(-1))

    # hazards
    active = state[side_idx].team[new_active]
    from ninjax.enum_types import ItemEnum
    boots = active.item == ItemEnum.HEAVY_DUTY_BOOTS
    is_not_flying = (1 - active.is_floating) * ~boots
    is_not_hazard_immune = (1 - active.is_hazard_immune) * ~boots
    is_poison = active.is_type(Type.POISON)
    is_poison_immune = jnp.any(active.type_list == Type.STEEL)
    # stealth rocks
    state = take_damage_percent(
        state,
        side_idx,
        state[side_idx].stealth_rocks * calculate_effectiveness_multiplier(Type.ROCK, active.type_list) / 8 * is_not_hazard_immune
    )

    # spikes
    state = take_damage_percent(
        state, side_idx,
        (state[side_idx].spikes != 0) / (10 - 2 * state[side_idx].spikes) * is_not_flying * is_not_hazard_immune
    )

    # sticky webs
    # so apparently if a pokemon sets webs and a pokemon with mirror armor is effected by them
    # then it applies the stat reduction to that mon, but this only happens if they are on field
    state = conditional_reduce_boosts(state, side_idx,
        (is_not_flying * state[side_idx].sticky_webs).reshape(()) != 0, StatEnum.SPEED, 1, False)

    # toxic spikes
    # only remove is poison type and not floating
    toxic_spikes = state[side_idx].toxic_spikes * (1 - jnp.logical_and(is_poison, is_not_flying))
    toxic_spikes = state.toxic_spikes.at[side_idx].set(toxic_spikes)
    state = state.replace(toxic_spikes=toxic_spikes)
    # this returns 0, 5, 6 for 0, 1, 2
    status_ = (7 - state[side_idx].toxic_spikes) * (state[side_idx].toxic_spikes != 0)
    key, state = set_status(key, state, side_idx, status_ * (1-is_poison_immune) * is_not_flying)

    #check if switch in is still alive
    active_hp = state.active.current_hp[side_idx, 0]
    is_alive = active_hp != 0

    state = jax.lax.cond(is_alive, swap_is_alive, lambda a, b: a, state, side_idx)


    active = state.active[side_idx].replace(is_alive=is_alive)
    state = update_active(state, side_idx, active)
    return state

def swap_is_alive(
    state: BattleState,
    side_idx) -> BattleState:
    """Apply represented weather, terrain, Intimidate and Frisk entry abilities."""
    active = state.active
    attacker = active[side_idx]
    defender_ability = active[1-side_idx].ability

    # activate weather abilities if target is alive
    # we abuse the choice to WeatherEnum.NONE=0 to simplify the logic

    ability = attacker.ability[0]
    previous_weather = state.weather.weather
    previous_terrain = state.terrain.terrain
    new_weather = (jnp.logical_or(ability==AbilityEnum.DROUGHT, ability==AbilityEnum.ORICHALCUM_PULSE) * WeatherEnum.SUN +
                   (ability==AbilityEnum.DRIZZLE) * WeatherEnum.RAIN +
                   (ability==AbilityEnum.SAND_STREAM) * WeatherEnum.SANDSTORM +
                   (ability==AbilityEnum.SNOW_WARNING) * WeatherEnum.SNOW)
    state = set_weather(state, new_weather, 5)

    # terrain, same deal as weather
    new_terrain = (jnp.logical_or(ability==AbilityEnum.ELECTRIC_SURGE, ability==AbilityEnum.HADRON_ENGINE) * TerrainEnum.ELECTRIC +
                   (ability==AbilityEnum.PSYCHIC_SURGE) * TerrainEnum.PSYCHIC +
                   (ability==AbilityEnum.GRASSY_SURGE) * TerrainEnum.GRASSY +
                   (ability==AbilityEnum.MISTY_SURGE) * TerrainEnum.MISTY)
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
    state = conditional_reduce_boosts(state, 1-side_idx, intimidate_activated, StatEnum.ATTACK, 1, True)
    old_source = state.event_source
    state = state.replace(event_source=jnp.asarray(side_idx, jnp.int32))
    state = emit(state, EventKind.ABILITY, side_idx, detail=ability,
                 enabled=(state.weather.weather != previous_weather) | (state.terrain.terrain != previous_terrain))
    state = state.replace(event_source=old_source)
    from ninjax.ability_effects import frisk
    return frisk(state, side_idx)


def step_side_conditions(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    """Decrement side timers and update toxic counters; return the unchanged key."""
    toxic_counter = (state.toxic_counter + 1) * (state.active.status == Status.TOXIC)
    state = state.replace(
        reflect=jnp.maximum(state.reflect - 1, 0),
        light_screen=jnp.maximum(state.light_screen - 1, 0),
        aurora_veil=jnp.maximum(state.aurora_veil - 1, 0),
        tailwind=jnp.maximum(state.tailwind - 1, 0),
        toxic_counter=toxic_counter
    )
    state = state.replace(volatile_status=state.volatile_status.replace(protected=jnp.zeros_like(state.volatile_status.protected)))
    return key, state

def step_moody(
    key: chex.PRNGKey,
    state: BattleState,
) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply represented Moody stage changes, splitting one key per side.

    The fixed two-iteration loop is unrolled when tracing a singles battle.
    """
    for i in range(2):
        key, plus_key, minus_key = random.split(key, 3)
        stages = state.boosts.normal_boosts[i, 1:]
        can_raise = stages < 6
        plus = random.categorical(plus_key, jnp.where(can_raise, 0., -jnp.inf))
        can_lower = (stages > -6) & ((jnp.arange(5) != plus) | ~jnp.any(can_raise))
        minus = random.categorical(minus_key, jnp.where(can_lower, 0., -jnp.inf))
        active = (state.active.ability[i, 0] == AbilityEnum.MOODY) & (state.active.current_hp[i, 0] > 0)
        state = conditional_add_boosts(state, i, active & jnp.any(can_raise), 1+plus, 2)
        state = conditional_add_boosts(state, i, active & jnp.any(can_lower), 1+minus, -1)
    return key, state



def move_interrupted(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Return unchanged key and state for an interrupted move (placeholder)."""
    # TODO: eventually we will need to figure out we handle observations and include it here
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protect_chain=v.protect_chain.at[attacker_index, 0].set(0)))
    return key, emit(state, EventKind.INTERRUPTED, attacker_index)

def move_used(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Consume one PP and dispatch the move if its implementation is enabled.

    Unsupported moves consume PP but otherwise take the no-effect branch. The
    caller is responsible for ensuring the selected move has PP remaining.
    """
    move = state.active[attacker_index].moves[move_index]

    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protect_chain=v.protect_chain.at[attacker_index, 0].set(
        jnp.where(move.identifier[0] == 182, v.protect_chain[attacker_index, 0], 0))))

    state = state.replace(event_source=jnp.asarray(attacker_index, jnp.int32),
                          event_move=move.identifier[0], event_cause=jnp.int32(Cause.MOVE))
    state = emit(state, EventKind.MOVE, attacker_index, detail=move.identifier[0])
    from ninjax.enum_types import ItemEnum
    choice = jnp.any(state.active[attacker_index].item == jnp.array([ItemEnum.CHOICE_BAND, ItemEnum.CHOICE_SPECS]))
    lock = state.volatile_status.choice_lock
    state = state.replace(volatile_status=state.volatile_status.replace(choice_lock=lock.at[attacker_index, 0].set(
        jnp.where(choice & (lock[attacker_index, 0] < 0), move_index, lock[attacker_index, 0]))))
    from ninjax.ability_effects import prepare_type_change
    state = prepare_type_change(state, attacker_index, move)
    # decrement pp
    # theoretically the legal action mask should prevent us from using the move if its at 0 so we dont need to clip
    move = move.replace(current_pp=move.current_pp-1)
    active = state.active[attacker_index]
    new_pp = active.moves.current_pp.at[move_index].subtract(1)
    new_moves = active.moves.replace(current_pp=new_pp)
    active = active.replace(moves=new_moves)
    state = update_active(state, attacker_index, active)
    return jax.lax.cond(move.implemented[0], _apply_move_effects, move_misses,
                        key, state, attacker_index, move_index)


def effective_move(state, attacker_index, move):
    """Return a transient type/power override without mutating the move slot."""
    from ninjax.held_effects import conditional_mult_round
    ability = state.active.ability[attacker_index]
    # Showdown excludes Judgment (449), Multi-Attack (718), Natural Gift (363),
    # Revelation Dance (686), Techno Blast (546), Terrain Pulse (805), and
    # Weather Ball (311) from -ate type conversion.
    excluded = jnp.any(move.identifier[0] == jnp.array([449, 718, 363, 686, 546, 805, 311]))
    normal = (move.type == Type.NORMAL) & ~excluded
    changed = normal & ((ability == AbilityEnum.PIXILATE) | (ability == AbilityEnum.AERILATE) |
                        (ability == AbilityEnum.REFRIGERATE) | (ability == AbilityEnum.GALVANIZE))
    move_type = jnp.where(normal & (ability == AbilityEnum.PIXILATE), Type.FAIRY, move.type)
    move_type = jnp.where(normal & (ability == AbilityEnum.AERILATE), Type.FLYING, move_type)
    move_type = jnp.where(normal & (ability == AbilityEnum.REFRIGERATE), Type.ICE, move_type)
    move_type = jnp.where(normal & (ability == AbilityEnum.GALVANIZE), Type.ELECTRIC, move_type)
    move_type = jnp.where(ability == AbilityEnum.NORMALIZE, Type.NORMAL, move_type)
    move_type = jnp.where((ability == AbilityEnum.LIQUID_VOICE) & move.sound, Type.WATER, move_type)
    return move.replace(type=move_type, base_power=conditional_mult_round(move.base_power, 4915/4096,
                        changed | (ability == AbilityEnum.NORMALIZE)))


def _apply_move_effects(key, state, attacker_index, move_index):
    return move_not_drawn_in(key, state, attacker_index, move_index, StatEnum.SPECIAL_ATTACK)


def move_not_drawn_in(key: chex.PRNGKey, state: BattleState, attacker_index, move_index: int, unused: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Sample accuracy and dispatch a hit or miss, returning the advanced key.

    The calculation includes represented accuracy stages and abilities. The
    final ``unused`` operand is retained for compatibility with branch calls.
    """
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    move = effective_move(state, attacker_index, state.active[attacker_index].moves[move_index])
    accuracy = move_accuracy(state, attacker_index, move)

    # choose function based on if move hits
    return jax.lax.cond(
        jnp.less_equal(r, accuracy)[0],
        move_hits, move_misses,
        key, state, attacker_index, move_index)


def move_accuracy(state, attacker_index, move):
    """Calculate the currently represented accuracy for an individual attempt."""
    active = state.active
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
    flash_fire = (defender_ability == AbilityEnum.FLASH_FIRE) & (move.type == Type.FIRE) & ~move.target_self
    accuracy = jnp.where(flash_fire, 1., accuracy)
    # Storm Drain and Water Absorb intercept incoming Water moves in TryHit,
    # before Showdown's accuracy stage. Dispatch their branch even on a move
    # that would otherwise miss; self-targeted moves are handled separately.
    water_absorbed = (move.type == Type.WATER) & ~move.target_self & (
        (defender_ability == AbilityEnum.STORM_DRAIN) | (defender_ability == AbilityEnum.WATER_ABSORB))
    accuracy = jnp.where(water_absorbed, 1., accuracy)
    accuracy = jnp.where(move.target_self, 1., accuracy)

    return accuracy

def is_immune(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value):
    """Return unchanged state for the currently represented immunity branch."""
    return key, emit(state, EventKind.IMMUNE, 1-attacker_idx)

def move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Dispatch an implemented move after an accuracy check, or do nothing."""
    move = state.active[attacker_index].moves[move_index]
    return jax.lax.cond(move.implemented[0], _move_hits, move_misses,
                        key, state, attacker_index, move_index)


def _move_hits(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Select the represented hit outcome and record a pending pivot switch.

    Ability and type immunities, status moves, and damaging moves share one
    branch signature so ``lax.switch`` can trace each possible outcome.
    """

    # decide branch to execute based on ability immunities
    defender = state.active[1 - attacker_index]
    move = effective_move(state, attacker_index, state.active[attacker_index].moves[move_index])
    ability = defender.ability
    branches = [do_damaging_move,
                do_status_move,
                do_stat_boost_from_move,
                do_flash_fire_from_move,
                do_healing_from_move,
                is_immune]
    is_flash_fire = jnp.logical_and(ability==AbilityEnum.FLASH_FIRE, move.type==Type.FIRE)
    is_spa_boost = jnp.logical_or(
        jnp.logical_and(ability==AbilityEnum.STORM_DRAIN, move.type==Type.WATER),
        jnp.logical_and(ability==AbilityEnum.LIGHTNING_ROD, move.type==Type.ELECTRIC))
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
    immune = defender.is_immune_to_move(move, state.gravity_duration > 0)
    immune = jnp.logical_and(immune, ~not_bypass_move)
    is_status = jnp.logical_and(move.move_type == MoveType.STATUS, ~not_bypass_move & ~immune)
    # Encode the selected outcome for lax.switch. Self-targeted status moves
    # explicitly select the status branch instead of defender immunity branches.
    branch_index = (
            is_status +
            is_stat_boost * 2 +
            is_flash_fire * 3 +
            is_heal * 4 +
            immune * 5
    )
    branch_index = jnp.where(move.target_self & (move.move_type == MoveType.STATUS), 1, branch_index)
    # Psychic Terrain blocks opposing priority moves before absorption abilities
    # or other hit effects can activate, but only for grounded targets.
    grounded = ~defender.is_floating[0] | (state.gravity_duration > 0)
    priority_blocked = (state.terrain.terrain == TerrainEnum.PSYCHIC) & grounded & (move.priority[0] > 0) & ~move.target_self[0]
    branch_index = jnp.where(priority_blocked, 5, branch_index)
    # Protect blocks opposing targeted moves, but not field-wide hazards.
    bypass = (state.active.ability[attacker_index, 0] == AbilityEnum.UNSEEN_FIST) & move.contact[0]
    blocked = state.volatile_status.protected[1-attacker_index, 0] & ~move.target_self[0] & (move.hazard_effect[0] == 0) & ~bypass
    branch_index = jnp.where(blocked, 5, branch_index)
    boost_value = 1 + (ability==AbilityEnum.WELL_BAKED_BODY)

    # i think using a switch means we skip evaluating the branches we don't need
    # the stat_index only is used in the stat_boost branch so its value doesnt matter the rest of the time
    key, state = jax.lax.switch(
        branch_index[0],
        branches,
        key,
        state,
        attacker_index,
        move,
        is_attack_boost * StatEnum.ATTACK + is_def_boost * StatEnum.DEFENSE +
        is_spa_boost * StatEnum.SPECIAL_ATTACK + is_speed_boost * StatEnum.SPEED,
        boost_value
    )
    requested = move.switch_out[0] & (branch_index[0] <= 1)
    return key, state.replace(switch_requested=state.switch_requested.at[attacker_index].set(requested))

def move_misses(key: chex.PRNGKey, state: BattleState, attacker_index: int, move_index: int) -> Tuple[chex.PRNGKey, BattleState]:
    """Return unchanged key and state for a miss or unsupported move."""
    return key, emit(state, EventKind.MISS, 1-attacker_index, detail=state.active[attacker_index].moves[move_index].implemented[0])



