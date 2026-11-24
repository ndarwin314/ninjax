"""Hit-time wiring and damage-stage exceptions for conditional power moves."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import compute_base_power, compute_damage_multipliers
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Weather, WeatherEnum as W
from ninjax.enums.moves import PowerRule
from ninjax.enums.pokemon import Status
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.fixture(scope='module')
def power():
    """Share compiled intrinsic and modifier resolution across move families."""
    return jax.jit(compute_base_power)


@pytest.mark.parametrize('name', ['Hex', 'Infernal Parade'])
def test_status_intrinsic_before_technician(neutral_type, power, name):
    """Intrinsic doubling prevents Technician from boosting a synthetic 40-power hit."""
    _, state = neutral_type
    user = state.active[0].replace(ability=jnp.array([A.TECHNICIAN]))
    target = state.active[1].replace(ability=jnp.array([A.COMATOSE]))
    move = make_move(name).replace(base_power=jnp.array([40]))
    assert power(user, target, move, state.terrain)[0] == 80


@pytest.mark.parametrize('name', ['Eruption', 'Water Spout', 'Dragon Energy'])
def test_live_user_hp(neutral_type, power, name):
    """Power reads the current HP snapshot rather than the stored move power."""
    _, state = neutral_type
    user = state.active[0].replace(current_hp=jnp.array([state.active.max_hp[0] // 2]))
    assert power(user, state.active[1], make_move(name), state.terrain)[0] == 75


@pytest.mark.parametrize('name,scale', [('Crush Grip', 120), ('Hard Press', 100)])
def test_live_target_hp(neutral_type, power, name, scale):
    """Target-HP callbacks read the target, with exact full and half-HP boundaries."""
    _, state = neutral_type
    target = state.active[1].replace(current_hp=jnp.array([state.active.max_hp[1] // 2]))
    assert power(state.active[0], target, make_move(name), state.terrain)[0] == scale // 2


@pytest.mark.parametrize('name,expected_power', [('Electro Ball', 80), ('Gyro Ball', 13)])
def test_damage_uses_effective_speeds(neutral_type, name, expected_power):
    """Tailwind enters hit-time power independently of the damage-stat path."""
    key, state = neutral_type
    state = state.replace(tailwind=state.tailwind.at[0, 0].set(4))
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.UNAWARE])))
    move = make_move(name)
    resolve = jax.jit(resolve_damaging_hit)
    _, actual, damage, _ = resolve(key, state, 0, move)
    fixed = move.replace(power_rule=jnp.array([PowerRule.FIXED]), base_power=jnp.array([expected_power]))
    _, expected, expected_damage, _ = resolve(key, state, 0, fixed)
    assert damage[0] == expected_damage[0]
    assert actual.active.current_hp[1, 0] == expected.active.current_hp[1, 0]


def test_facade_bypasses_only_its_burn_penalty(neutral_type):
    """Facade's damage exception is separate from its priority-zero power factor."""
    key, state = neutral_type
    user = state.active[0].replace(status=jnp.array([Status.BURN]))
    multiply = jax.jit(compute_damage_multipliers)
    facade = multiply(key, user, state.active[1], Weather(W.NONE, 0), make_move('Facade'), 100, False, 1.)
    tackle = multiply(key, user, state.active[1], Weather(W.NONE, 0), make_move('Tackle'), 100, False, 1.)
    assert facade[0] == 100 and tackle[0] == 50


@pytest.mark.parametrize('status,expected', [(Status.NONE, 70), (Status.BURN, 140),
    (Status.POISON, 140), (Status.PARALYZE, 140), (Status.SLEEP, 70)])
def test_facade_status_eligibility(neutral_type, power, status, expected):
    """Facade's priority-zero boost excludes sleep and a status-free user."""
    _, state = neutral_type
    user = state.active[0].replace(status=jnp.array([status]))
    assert power(user, state.active[1], make_move('Facade'), state.terrain)[0] == expected
