"""Move-specific type chart factors independent of battle routing."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier, type_effectiveness_factors
from ninjax.enums.moves import EffectivenessRule as Rule
from ninjax.enums.pokemon import Type


@pytest.mark.parametrize('attack,defense,rule,expected', [
    (Type.ICE, [Type.WATER, Type.NONE], Rule.FREEZE_DRY, [2., 1.]),
    (Type.ICE, [Type.WATER, Type.GROUND], Rule.FREEZE_DRY, [2., 2.]),
    (Type.FIRE, [Type.WATER, Type.STEEL], Rule.FREEZE_DRY, [2., 2.]),
    (Type.ICE, [Type.WATER, Type.NONE], Rule.DEFAULT, [.5, 1.]),
    (Type.FIGHTING, [Type.GRASS, Type.BUG], Rule.FLYING_PRESS, [2., 1.]),
    (Type.FIGHTING, [Type.ROCK, Type.STEEL], Rule.FLYING_PRESS, [1., 1.]),
    (Type.FIGHTING, [Type.GHOST, Type.NONE], Rule.FLYING_PRESS, [0., 1.]),
    (Type.ELECTRIC, [Type.GROUND, Type.WATER], Rule.FLYING_PRESS, [0., 2.]),
    (Type.FIGHTING, [Type.GRASS, Type.BUG], Rule.DEFAULT, [1., .5]),
])
def test_move_type_factors(attack, defense, rule, expected):
    """Apply each move rule independently to every defending type."""
    defending_types = jnp.array(defense)
    factors = type_effectiveness_factors(jnp.array([attack]), defending_types, jnp.array([rule]))
    assert jnp.allclose(factors, jnp.array(expected))
    assert calculate_effectiveness_multiplier(jnp.array([attack]), defending_types, jnp.array([rule])) == pytest.approx(expected[0] * expected[1])


def test_move_type_factors_jit():
    """Select rules dynamically under JIT without changing the input type."""
    calculate = jax.jit(type_effectiveness_factors)
    types = jnp.array([Type.WATER, Type.GRASS])
    attack = jnp.array([Type.ICE])
    normal = calculate(attack, types, jnp.array([Rule.DEFAULT]))
    freeze_dry = calculate(attack, types, jnp.array([Rule.FREEZE_DRY]))
    assert jnp.allclose(normal, jnp.array([.5, 2.]))
    assert jnp.allclose(freeze_dry, jnp.array([2., 2.]))
