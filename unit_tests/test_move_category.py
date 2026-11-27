"""Dynamic damage categories at move preparation and hit routing."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.moves import MoveType, ProtectionKind
from ninjax.enums.pokemon import StatEnum
from ninjax.moves.properties import resolve_move_category
from ninjax.moves.routing import move_not_drawn_in, prepare_hit_move
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('attack_stage,category,offense,defense', [
    (0, MoveType.SPECIAL, StatEnum.SPECIAL_ATTACK, StatEnum.SPECIAL_DEFENSE),
    (1, MoveType.PHYSICAL, StatEnum.ATTACK, StatEnum.DEFENSE),
])
def test_photon_geyser_preparation(neutral_type, attack_stage, category, offense, defense):
    """Photon Geyser uses stages, stays Special on ties, and never makes contact."""
    key, state = neutral_type
    state = state.replace(boosts=state.boosts.replace(
        normal_boosts=state.boosts.normal_boosts.at[0, StatEnum.ATTACK].set(attack_stage)))
    move = make_move('Photon Geyser')
    next_key, resolved = prepare_hit_move(key, state, 0, move)
    assert resolved.move_type[0] == category
    assert resolved.offensive_stat[0] == offense
    assert resolved.defensive_stat[0] == defense
    assert not resolved.contact[0]
    assert jnp.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))


def test_shell_side_arm_integer_tie():
    """Different stat ratios can tie after flooring the damage estimates."""
    resolve = jax.jit(resolve_move_category)
    move = make_move('Shell Side Arm')
    categories = set()
    for seed in range(12):
        key = jax.random.key(seed)
        next_key, resolved = resolve(key, move, 1, 100, 101, 100, 100)
        assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(jax.random.split(key)[0]))
        categories.add(int(resolved.move_type[0]))
        assert resolved.contact[0] == (resolved.move_type[0] == MoveType.PHYSICAL)
    assert categories == {int(MoveType.PHYSICAL), int(MoveType.SPECIAL)}


@pytest.fixture(scope='module')
def route_category_move():
    """Compile the normal accuracy and protection path once for both categories."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('defense_stage,contact', [(-2, True), (2, False)])
def test_shell_side_arm_protection_uses_resolved_contact(neutral_type, route_category_move, defense_stage, contact):
    """Defensive stages select the category before Spiky Shield checks contact."""
    key, state = neutral_type
    state = equip_move(state, 'Shell Side Arm')
    volatile = state.volatile_status
    state = state.replace(boosts=state.boosts.replace(
        normal_boosts=state.boosts.normal_boosts.at[1, StatEnum.DEFENSE].set(defense_stage)),
        volatile_status=volatile.replace(
            protection=volatile.protection.replace(
                protected=volatile.protection.protected.at[1, 0].set(True),
                protection_kind=volatile.protection.protection_kind.at[1, 0].set(ProtectionKind.SPIKY_SHIELD))))
    next_key, resolved = prepare_hit_move(key, state, 0, state.active_moves(0, 0))
    assert jnp.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))
    assert resolved.contact[0] == contact
    assert resolved.offensive_stat[0] == (StatEnum.ATTACK if contact else StatEnum.SPECIAL_ATTACK)
    assert resolved.defensive_stat[0] == (StatEnum.DEFENSE if contact else StatEnum.SPECIAL_DEFENSE)
    _, out = route_category_move(key, state, 0, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.PROTECTED
    assert (out.active.current_hp[0, 0] < state.active.current_hp[0, 0]) == contact


@pytest.mark.parametrize('defense_stage,physical', [(-2, True), (2, False)])
def test_shell_side_arm_uses_category_for_damage(neutral_type, route_category_move, defense_stage, physical):
    """Reflect reduces only the Physical result and Light Screen only the Special one."""
    key, state = neutral_type
    state = equip_move(state, 'Shell Side Arm')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([AbilityEnum.BATTLE_ARMOR])))
    state = state.replace(boosts=state.boosts.replace(
        normal_boosts=state.boosts.normal_boosts.at[1, StatEnum.DEFENSE].set(defense_stage)))
    _, normal = route_category_move(key, state, 0, 0, 0)
    _, reflected = route_category_move(key, state.replace(side_conditions=state.side_conditions.replace(reflect=state.side_conditions.reflect.at[1, 0].set(5))), 0, 0, 0)
    _, screened = route_category_move(key, state.replace(side_conditions=state.side_conditions.replace(light_screen=state.side_conditions.light_screen.at[1, 0].set(5))), 0, 0, 0)
    damage = normal.hit_results[0].damage
    assert normal.hit_results[0].outcome == HitOutcome.HIT and damage > 0
    assert reflected.hit_results[0].damage == (damage // 2 if physical else damage)
    assert screened.hit_results[0].damage == (damage if physical else damage // 2)
