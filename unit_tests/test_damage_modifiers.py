"""Rounding and stage ownership checked against the pinned Showdown engine."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import compute_damage_multipliers
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Type, Status
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/showdown/damage-modifiers.json').read_text())


@pytest.fixture(scope='module')
def calculate():
    """Share one compiled damage calculation across dynamic reference cases."""
    return jax.jit(compute_damage_multipliers)


@pytest.mark.parametrize('probe', REFERENCE['probes'])
def test_showdown_damage_stages(neutral_type, calculate, probe):
    """Compare actual simulator output at rounding boundaries and stacked effects."""
    _, state = neutral_type
    user = state.active[0].replace(
        ability=jnp.array([A[probe['ability'].upper().replace(' ', '_')] if probe['ability'] != 'No Ability' else A.NONE]),
        item=jnp.array([I[probe['item'].upper().replace(' ', '_')] if probe['item'] else I.NONE]),
        type_list=jnp.array([Type[probe['userType'].upper()], Type.NONE]),
        tera_type=jnp.array([Type[probe['teraType'].upper()] if probe['teraType'] else Type.NONE]),
        is_terastallized=jnp.array([bool(probe['teraType'])]),
        status=jnp.array([Status.BURN if probe['status'] == 'brn' else Status.NONE]))
    target = state.active[1].replace(
        ability=jnp.array([A[probe['targetAbility'].upper().replace(' ', '_')] if probe['targetAbility'] != 'No Ability' else A.NONE]),
        type_list=jnp.array([Type[probe['targetType'].upper()], Type.NONE]),
        current_hp=state.active[1].max_hp - (not probe['fullHp']))
    weather = {'': W.NONE, 'sunnyday': W.SUN, 'raindance': W.RAIN}[probe['weather']]
    actual = calculate(user, target, jnp.int32(weather), make_move(probe['name']),
        jnp.array([probe['baseDamage']]), jnp.bool_(probe['crit']),
        probe['roll'] / 100, screen=jnp.bool_(probe['screen']),
        user_speed=jnp.int32(probe['userSpeed']), target_speed=jnp.int32(probe['targetSpeed']))
    assert actual[0] == probe['expected'], probe


@pytest.fixture(scope='module')
def hit():
    """Compile the shared path for Pokémon and Substitute recipients."""
    return jax.jit(resolve_damaging_hit)


@pytest.mark.parametrize('substitute', [False, True])
def test_multiscale_applies_once_to_both_recipients(neutral_type, hit, substitute):
    """Multiscale affects calculated damage identically before either HP commit."""
    key, state = neutral_type
    if substitute:
        state = state.replace(volatile_status=state.volatile_status.replace(
            substitute_hp=state.volatile_status.substitute_hp.at[1, 0].set(1000)))
    guarded = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.MULTISCALE])))
    move = make_move('Tackle')
    _, _, normal_damage, _ = hit(key, state, 0, move)
    _, reduced, reduced_damage, _ = hit(key, guarded, 0, move)
    assert reduced_damage[0] == normal_damage[0] // 2
    assert reduced.volatile_status.move_history.last_damage[1, 0] == reduced_damage[0]
    if substitute:
        assert reduced.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    else:
        assert reduced.active.current_hp[1, 0] == state.active.current_hp[1, 0] - reduced_damage[0]


def test_multiscale_turns_off_after_hp_loss(neutral_type, hit):
    """A damaged target no longer contributes the full-HP modifier."""
    key, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(
        ability=jnp.array([A.MULTISCALE]), current_hp=state.active[1].max_hp - 1))
    normal = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.NONE])))
    move = make_move('Tackle')
    assert hit(key, state, 0, move)[2][0] == hit(key, normal, 0, move)[2][0]
