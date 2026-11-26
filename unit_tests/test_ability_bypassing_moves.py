"""Ability bypass through the shared accuracy, damage and reaction paths."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import StatEnum, Type
from ninjax.moves.routing import move_not_drawn_in
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def route_ability_bypass():
    """Compile one ordinary hit path for the ability bypass cases."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('name', ['Sunsteel Strike', 'Moongeist Beam', 'Photon Geyser'])
def test_bypass_multiscale(neutral_type, route_ability_bypass, name):
    """All three moves bypass damage reduction unless Ability Shield is active."""
    key, state = neutral_type
    state = equip_move(state, name)
    assert make_move(name).implemented[0] and make_move(name).ignore_ability[0]
    mon = state.active[1]
    multiscale = mon.replace(ability=jnp.array([A.MULTISCALE]))
    _, plain = route_ability_bypass(key, state, 0, 0, 0)
    _, ignored = route_ability_bypass(key, update_active(state, 1, multiscale), 0, 0, 0)
    damage = plain.hit_results[0].damage
    assert damage > 0
    assert ignored.hit_results[0].damage == damage
    assert ignored.ability[1, 0] == A.MULTISCALE

    shielded = update_active(state, 1, multiscale.replace(item=jnp.array([I.ABILITY_SHIELD])))
    _, shield = route_ability_bypass(key, shielded, 0, 0, 0)
    _, suppressed = route_ability_bypass(key, shielded.replace(magic_room_duration=2), 0, 0, 0)
    assert shield.hit_results[0].damage < damage
    assert suppressed.hit_results[0].damage == damage


def test_moongeist_beam_preserves_type_immunity(neutral_type, route_ability_bypass):
    """Ignoring an ability does not let a Ghost move hit a Normal target."""
    key, state = neutral_type
    state = equip_move(state, 'Moongeist Beam')
    state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.NORMAL, Type.NONE])))
    _, out = route_ability_bypass(key, state, 0, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE


def test_bypass_also_applies_to_absorption(neutral_type, route_ability_bypass):
    """An ability intercept uses the same scoped rule as the damage path."""
    key, state = neutral_type
    state = equip_move(state, 'Moongeist Beam')
    moves = state.team.moves
    # Use Water typing to exercise an absorption branch with the bypass flag.
    moves = moves.replace(type=moves.type.at[0, 0, 0, 0].set(Type.WATER))
    state = state.replace(team=state.team.replace(moves=moves))
    absorber = state.active[1].replace(ability=jnp.array([A.WATER_ABSORB]))
    state = update_active(state, 1, absorber)
    _, ignored = route_ability_bypass(key, state, 0, 0, 0)
    assert ignored.hit_results[0].outcome == HitOutcome.HIT
    shielded = update_active(state, 1, absorber.replace(item=jnp.array([I.ABILITY_SHIELD])))
    _, absorbed = route_ability_bypass(key, shielded, 0, 0, 0)
    assert absorbed.hit_results[0].outcome == HitOutcome.IMMUNE


def test_sunsteel_strike_respects_rough_skin(neutral_type, route_ability_bypass):
    """An unbreakable contact reaction still costs the attacker HP."""
    key, state = neutral_type
    state = equip_move(state, 'Sunsteel Strike')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.ROUGH_SKIN])))
    _, out = route_ability_bypass(key, state, 0, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.HIT
    assert out.active.current_hp[0, 0] < state.active.current_hp[0, 0]


@pytest.mark.parametrize('item,expected_hp', [(I.NONE, 0), (I.FOCUS_SASH, 1), (I.ABILITY_SHIELD, 1)])
def test_sunsteel_strike_survival_checks(neutral_type, route_ability_bypass, item, expected_hp):
    """Sturdy is bypassed while Focus Sash and Ability Shield still work."""
    key, state = neutral_type
    state = equip_move(state, 'Sunsteel Strike')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.STURDY]), item=jnp.array([item])))
    stages = state.boosts.normal_boosts.at[0, StatEnum.ATTACK].set(6).at[1, StatEnum.DEFENSE].set(-6)
    state = state.replace(boosts=state.boosts.replace(normal_boosts=stages))
    _, out = route_ability_bypass(key, state, 0, 0, 0)
    assert out.active.current_hp[1, 0] == expected_hp
