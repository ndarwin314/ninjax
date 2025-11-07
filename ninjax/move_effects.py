from typing import Tuple

import jax
import jax.random as random
import jax.numpy as jnp
import chex
from ninjax.utils import STAT_MULTIPLIER_LOOKUP, ACCURACY_MULTIPLIER_LOOKUP

from ninjax.enum_types import AbilityEnum, Status, Type, TerrainEnum, WeatherEnum, MoveType, Weather, Terrain, StatEnum
from ninjax.side import (BattleState, update_active, clear_volatile_status, clear_boosts, add_boosts,reduce_boosts,
                         conditional_add_boosts, conditional_reduce_boosts, set_weather, set_terrain, set_status,
                         take_damage_percent)
from ninjax.pokemon import Pokemon
from ninjax.move import Move
from ninjax.utils import (
    conditional_mult_round, TERRAIN_MULTIPLIER, TYPE_EFFECTIVENESS, CRIT_STAGES,calculate_effectiveness_multiplier,
    COMPOUND_EYES_MULTIPLIER, conditional_mult, WEATHER_VEIL_MODIFIER, triple_and, triple_or, quad_or, ROUGH_SKIN_DAMAGE,
    in_range, IRON_FIST, TOUGH_CLAWS, one_third, RECKLESS, VICTORY_STAR, four_thirds, quad_and,
    conditional_mult_prod_round, conditional_mult_prod, one_point_three)

jax.config.update("jax_disable_jit", True)

Array = chex.Array


def after_every_hit(
        key: chex.PRNGKey,
        state: BattleState,
        attacker_idx,
        move: Move,
        is_crit: bool
):
    defender_idx = 1 - attacker_idx
    defender = state.active[defender_idx]
    attacker = state.active[attacker_idx]
    #stamina
    is_stamina = jnp.logical_and(defender.ability == AbilityEnum.STAMINA, defender.is_alive)
    state = conditional_add_boosts(state, defender_idx, is_stamina, StatEnum.DEFENSE, 1)
    # cotton down
    is_cotton_down = (defender.ability == AbilityEnum.COTTON_DOWN)
    state = conditional_reduce_boosts(state, 1 - defender_idx, is_cotton_down, StatEnum.SPEED, 1, True)
    # seed sower
    state = set_weather(state, TerrainEnum.GRASSY * defender.ability==AbilityEnum.SEED_SOWER, 5)
    # weak armor
    state = conditional_add_boosts(
        state,
        defender_idx,
        jnp.logical_and(defender.ability==AbilityEnum.WEAK_ARMOR, move.move_type==MoveType.PHYSICAL),
        (StatEnum.DEFENSE, StatEnum.SPEED), (-1, 2)
    )
    # anger point
    is_angry = jnp.logical_and(defender.ability==AbilityEnum.ANGER_POINT, is_crit)[0]
    state = conditional_add_boosts(state, defender_idx, is_angry, StatEnum.ATTACK, 13)

    # thermal exchange
    state = conditional_add_boosts(
        state,
        defender_idx,
        jnp.logical_and(defender.ability==AbilityEnum.WEAK_ARMOR, move.type==Type.FIRE),
        StatEnum.ATTACK, 1
    )
    # do on contact effects
    makes_contact = jnp.logical_and(move.contact,  attacker.ability!=AbilityEnum.LONG_REACH)
    key, state = jax.lax.cond(makes_contact[0], do_contact, lambda k, s, a: (k, s), key, state, attacker_idx)
    return key, state

def after_move_finished(state: BattleState, defender: Pokemon, defender_idx, hp_start):
    # idk if stamina should actually be here or somewhere else, idk

    # berserk
    is_berserk = quad_and(
        defender.hp_greater_than(hp_start),
        defender.hp_less_than(0.5),
        defender.ability == AbilityEnum.BERSERK,
        defender.is_alive
    )
    state = conditional_add_boosts(state, defender_idx, is_berserk, StatEnum.SPECIAL_ATTACK, 1)
    return state

def effect_spore_status(r):
    return (jnp.less(r, 0.09) * Status.POISON +
            in_range(0.09, r, 0.19) * Status.PARALYZE +
            in_range(0.19, r, 0.3) * Status.SLEEP)

def do_contact(key: chex.PRNGKey, state: BattleState, attacker_idx) -> Tuple[chex.PRNGKey, BattleState]:
    active = state.active
    defender = active[1-attacker_idx]
    attacker = active[attacker_idx]
    key, def_key, attack_key = random.split(key, 3)

    # defender triggers
    is_rough_skin = (defender.ability == AbilityEnum.ROUGH_SKIN)[0]
    state = take_damage_percent(state, attacker_idx, ROUGH_SKIN_DAMAGE * is_rough_skin)
    is_static = defender.ability==AbilityEnum.STATIC
    is_flame = defender.ability==AbilityEnum.FLAME_BODY
    is_point = defender.ability==AbilityEnum.POISON_POINT
    is_effect_spore = jnp.logical_and(defender.ability==AbilityEnum.EFFECT_SPORE, 1-defender.is_powder_immune)
    r = random.uniform(def_key)
    triggered = jnp.less_equal(r, 0.3)
    status = (effect_spore_status(r) * is_effect_spore +
              is_flame * Status.BURN +
              is_static * Status.PARALYZE +
              is_point * Status.POISON) * triggered
    key, state = set_status(key, state, attacker_idx, status)

    # attacker triggers
    is_poison_touch = attacker.ability==AbilityEnum.POISON_TOUCH
    is_toxic_chain = attacker.ability==AbilityEnum.TOXIC_CHAIN
    r = random.uniform(attack_key)
    triggered = jnp.less_equal(r, 0.3)
    status = (is_poison_touch * Status.POISON +
              is_toxic_chain * Status.TOXIC) * triggered

    # gooey, idk why i take 0 index here but it happens in some places and i think was necessary
    is_gooey = (defender.ability==AbilityEnum.GOOEY)[0]
    state = conditional_reduce_boosts(state, attacker_idx, is_gooey, StatEnum.SPEED, 1, True)

    key, state = set_status(key, state, 1 - attacker_idx, status)
    return key, state
