"""Move-specific effectiveness across construction, routing and damage."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.type_effectiveness import hit_type_effectiveness
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import EffectivenessRule
from ninjax.enums.pokemon import Type
from ninjax.moves.routing import move_not_drawn_in
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def route_move():
    """Share a compiled normal accuracy, immunity and hit pipeline."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('name,rule', [
    ('Freeze-Dry', EffectivenessRule.FREEZE_DRY),
    ('Flying Press', EffectivenessRule.FLYING_PRESS),
])
def test_catalogue_projects_effectiveness_rule(name, rule):
    """Supported move metadata carries its rule and usable PP."""
    move = make_move(name)
    assert move.effectiveness_rule[0] == rule
    assert move.implemented[0] and move.current_pp[0] > 0


@pytest.mark.parametrize('name,types,outcome,rule', [
    ('Freeze-Dry', [Type.WATER, Type.GROUND], HitOutcome.HIT, EffectivenessRule.FREEZE_DRY),
    ('Flying Press', [Type.GRASS, Type.BUG], HitOutcome.HIT, EffectivenessRule.FLYING_PRESS),
    ('Flying Press', [Type.GHOST, Type.NONE], HitOutcome.IMMUNE, EffectivenessRule.FLYING_PRESS),
])
def test_move_rule_reaches_hit_pipeline(neutral_type, route_move, name, types, outcome, rule):
    """A move's projected rule survives early immunity and damage resolution."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 1, state.active[1].replace(type_list=jnp.array(types)))
    bank = state.team.moves
    # Keep the hit deterministic while exercising the ordinary support path.
    state = state.replace(team=state.team.replace(moves=bank.replace(
        accuracy=bank.accuracy.at[0, 0, 0, 0].set(1.))))
    _, actual = route_move(key, state, 0, 0, 0)
    assert actual.hit_results[0].outcome == outcome
    if outcome == HitOutcome.HIT:
        ordinary_bank = state.team.moves
        ordinary = state.replace(team=state.team.replace(moves=ordinary_bank.replace(
            effectiveness_rule=ordinary_bank.effectiveness_rule.at[0, 0, 0, 0].set(EffectivenessRule.DEFAULT))))
        _, baseline = route_move(key, ordinary, 0, 0, 0)
        assert actual.hit_results[0].damage > baseline.hit_results[0].damage
        assert actual.hit_results[0].damage == pytest.approx(
            baseline.hit_results[0].damage * 4, abs=2)


def test_ground_flying_press_gravity():
    """Gravity removes Ground immunity without losing Flying Press's second type."""
    raw, adjusted = hit_type_effectiveness(jnp.array([Type.GROUND]),
        jnp.array([Type.FLYING, Type.BUG]), jnp.array([AbilityEnum.NONE]), True,
        jnp.array([EffectivenessRule.FLYING_PRESS]))
    assert raw == 0
    assert adjusted == 1


def test_judgment_plate_hits_as_resolved_type(neutral_type, route_move):
    """Judgment uses its Plate for immunity checks, damage and power."""
    key, state = neutral_type
    state = equip_move(state, 'Judgment')
    mon = state.active[0].replace(item=jnp.array([ItemEnum.FLAME_PLATE]))
    state = update_active(state, 0, mon)
    state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.GRASS, Type.NONE])))
    _, fire = route_move(key, state, 0, 0, 0)
    normal_state = update_active(state, 0, mon.replace(item=jnp.array([ItemEnum.NONE])))
    _, normal = route_move(key, normal_state, 0, 0, 0)
    assert fire.hit_results[0].outcome == HitOutcome.HIT
    assert fire.hit_results[0].damage > normal.hit_results[0].damage * 2
    klutz_state = update_active(state, 0, mon.replace(ability=jnp.array([AbilityEnum.KLUTZ])))
    _, suppressed = route_move(key, klutz_state, 0, 0, 0)
    assert suppressed.hit_results[0].damage == normal.hit_results[0].damage
