"""Intrinsic formula boundaries and their interaction with power modifiers."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import compute_base_power, resolve_hit_base_power
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum as T, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import PowerRule
from ninjax.enums.pokemon import Type, Status
from ninjax.moves.properties import effective_move
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


POWER_PROBES = json.loads((Path(__file__).parent / 'fixtures/showdown/power-modifiers.json').read_text())['probes']


@pytest.fixture(scope='module')
def resolve_power():
    """Exercise runtime rule selection through one shared JIT executable."""
    return jax.jit(resolve_hit_base_power)


@pytest.mark.parametrize('rule,context,expected', [
    (PowerRule.FIXED, {}, 55),
    (PowerRule.DOUBLE_WITHOUT_ITEM, {'no_item': True}, 110),
    (PowerRule.DOUBLE_WITHOUT_ITEM, {'no_item': False}, 55),
    (PowerRule.DOUBLE_AFTER_FAILURE, {'previous_failed': True}, 110),
    (PowerRule.DOUBLE_AFTER_FAILURE, {'previous_failed': False}, 55),
    (PowerRule.POSITIVE_STAGES, {'positive_stages': 0}, 55),
    (PowerRule.POSITIVE_STAGES, {'positive_stages': 7}, 195),
    (PowerRule.DOUBLE_IN_WEATHER, {}, 55),
    (PowerRule.REMOVABLE_ITEM_BOOST, {}, 55),
    
])
def test_intrinsic_formulas(resolve_power, rule, context, expected):
    """Rules operate on supplied power and facts, independently of move IDs."""
    assert resolve_power(jnp.int32(55), jnp.int32(rule), **context) == expected


@pytest.mark.parametrize('weight,expected', [(99, 20), (100, 40), (249, 40),
    (250, 60), (499, 60), (500, 80), (999, 80), (1000, 100), (1999, 100), (2000, 120)])
def test_target_weight_boundaries(resolve_power, weight, expected):
    """Hectogram thresholds select the full Low Kick power table."""
    assert resolve_power(1, jnp.int32(PowerRule.TARGET_WEIGHT), target_weight=weight) == expected


@pytest.mark.parametrize('weight,expected', [(199, 40), (200, 60), (299, 60),
    (300, 80), (399, 80), (400, 100), (499, 100), (500, 120)])
def test_weight_ratio_boundaries(resolve_power, weight, expected):
    """Weight ratios use integer comparisons without division rounding."""
    assert resolve_power(1, jnp.int32(PowerRule.WEIGHT_RATIO),
                         user_weight=weight, target_weight=100) == expected


def test_rule_dispatch_is_independent_of_identifier(neutral_type):
    """A caller-provided move can choose a formula without impersonating a move ID."""
    _, state = neutral_type
    move = make_move('Tackle').replace(power_rule=jnp.array([PowerRule.DOUBLE_WITHOUT_ITEM]))
    resolved = effective_move(state, 0, move)
    assert resolved.base_power[0] == 40
    assert compute_base_power(state.active[0], state.active[1], resolved, state.terrain)[0] == 80
    assert move.base_power[0] == 40
    assert make_move('Low Kick').power_rule[0] == PowerRule.TARGET_WEIGHT
    assert make_move('Weather Ball').power_rule[0] == PowerRule.DOUBLE_IN_WEATHER


def test_intrinsic_power_precedes_technician(neutral_type):
    """Acrobatics' doubled power exceeds Technician's cutoff before modifiers."""
    _, state = neutral_type
    mon = state.active[0].replace(ability=jnp.array([A.TECHNICIAN]))
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0].set(mon.ability)))
    resolved = effective_move(state, 0, make_move('Acrobatics'))
    power = jax.jit(compute_base_power)(state.active[0], state.active[1], resolved, state.terrain)
    assert power[0] == 110


@pytest.fixture(scope='module')
def compiled_power():
    """Share compiled power evaluation across grounding and weather variants."""
    return jax.jit(compute_base_power)


@pytest.mark.parametrize('user_flying,target_flying,gravity,terrain,expected', [
    (False, False, False, T.ELECTRIC, 182),
    (True, False, False, T.ELECTRIC, 140),
    (False, True, False, T.ELECTRIC, 91),
    (True, True, False, T.ELECTRIC, 70),
    (True, True, True, T.ELECTRIC, 182),
    (False, False, False, T.NONE, 70),
])
def test_rising_voltage_grounding(neutral_type, compiled_power, user_flying, target_flying, gravity, terrain, expected):
    """Target grounding doubles power independently of the user's terrain bonus."""
    _, state = neutral_type
    user = state.active[0].replace(type_list=jnp.array([Type.FLYING if user_flying else Type.NORMAL, Type.NONE]))
    target = state.active[1].replace(type_list=jnp.array([Type.FLYING if target_flying else Type.NORMAL, Type.NONE]))
    power = compiled_power(user, target, make_move('Rising Voltage'), Terrain(terrain, 5), gravity_active=gravity)
    assert power[0] == expected


@pytest.mark.parametrize('ability,item', [(A.LEVITATE, I.NONE), (A.NONE, I.AIR_BALLOON)])
def test_rising_voltage_levitation(neutral_type, compiled_power, ability, item):
    """Levitate and Balloon prevent doubling until Gravity grounds the target."""
    _, state = neutral_type
    target = state.active[1].replace(ability=jnp.array([ability]), item=jnp.array([item]))
    move, terrain = make_move('Rising Voltage'), Terrain(T.ELECTRIC, 5)
    assert compiled_power(state.active[0], target, move, terrain)[0] == 91
    assert compiled_power(state.active[0], target, move, terrain, gravity_active=True)[0] == 182


def test_terrain_callback_order_with_technician(neutral_type, compiled_power):
    """Intrinsic doubling and Expanding Force's modifier straddle Technician."""
    _, state = neutral_type
    user = state.active[0].replace(ability=jnp.array([A.TECHNICIAN]))
    # Synthetic powers make the eligibility difference observable: Rising
    # Voltage becomes 80 before Technician; Expanding Force remains 40 for it.
    rising = make_move('Rising Voltage').replace(base_power=jnp.array([40]))
    expanding = make_move('Expanding Force').replace(base_power=jnp.array([40]))
    assert compiled_power(user, state.active[1], rising, Terrain(T.ELECTRIC, 5))[0] == 104
    assert compiled_power(user, state.active[1], expanding, Terrain(T.PSYCHIC, 5))[0] == 117


@pytest.mark.parametrize('weather', [W.NONE, W.SUN, W.RAIN, W.SANDSTORM, W.SNOW])
def test_dry_skin_fire_vulnerability_in_every_weather(neutral_type, compiled_power, weather):
    """Grouping with weather abilities must not make Fire vulnerability conditional."""
    _, state = neutral_type
    target = state.active[1].replace(ability=jnp.array([A.DRY_SKIN]))
    assert compiled_power(state.active[0], target, make_move('Flamethrower'),
                          state.terrain, weather)[0] == 112
    assert compiled_power(state.active[0], target, make_move('Tackle'),
                          state.terrain, weather)[0] == 40


@pytest.mark.parametrize('probe', POWER_PROBES, ids=lambda p: f"{p['name']}-{p['ability']}-{p['power']}")
def test_base_power_matches_showdown_event(neutral_type, compiled_power, probe):
    """Compare the complete chain to actual simulator event results, including odd powers."""
    _, state = neutral_type
    ability = A[probe['ability'].upper().replace(' ', '_')] if probe['ability'] else A.NONE
    defender_ability = A[probe['defenderAbility'].upper().replace(' ', '_')] if probe['defenderAbility'] else A.NONE
    item = I[probe['item'].upper().replace(' ', '_')] if probe['item'] else I.NONE
    defender_item = I[probe['defenderItem'].upper().replace(' ', '_')] if probe['defenderItem'] else I.NONE
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([ability]), item=jnp.array([item])))
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([defender_ability]), item=jnp.array([defender_item])))
    statuses = {'': Status.NONE, 'brn': Status.BURN, 'psn': Status.POISON, 'tox': Status.TOXIC}
    state = update_active(state, 0, state.active[0].replace(status=jnp.array([statuses[probe.get('userStatus', '')]])))
    state = update_active(state, 1, state.active[1].replace(status=jnp.array([statuses[probe.get('targetStatus', '')]])))
    if probe.get('halfHp'):
        state = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([state.active.max_hp[1] // 2])))
    if probe.get('targetType'):
        state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type[probe['targetType'].upper()], Type.NONE])))
    terrain = {'': T.NONE, 'Psychic Terrain': T.PSYCHIC, 'Electric Terrain': T.ELECTRIC}[probe['terrain']]
    move = effective_move(state, 0, make_move(probe['name']).replace(base_power=jnp.array([probe['power']])))
    assert compiled_power(state.active[0], state.active[1], move, Terrain(terrain, 5))[0] == probe['expected']
