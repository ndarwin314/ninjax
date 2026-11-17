"""Shared numeric multipliers and rounding operations."""

import chex
import jax.numpy as jnp


Array = chex.Array


increase_mult = 1.1


one_point_three = 5325/4096


one_point_one = 4506/4096


four_thirds = 5448/4096


one_point_three_exact = 1.3


one_third = 1/3


TERRAIN_MULTIPLIER = one_point_three_exact


CRIT_STAGES = jnp.array([1/24, 1/8, 1/2, 1, 1, 1, 1, 1])


COMPOUND_EYES_MULTIPLIER = one_point_three


VICTORY_STAR = one_point_one


TOUGH_CLAWS = one_point_three


WEATHER_VEIL_MODIFIER = 3277/4096


FAIRY_AURA = four_thirds


DARK_AURA = four_thirds


ROUGH_SKIN_DAMAGE = 1/8


IRON_FIST = 1.2


RECKLESS = 1.2


def conditional_mult_round(damage, mult, cond) -> Array:
    # Truncate toward zero, matching numpy.fix.
    """Apply an enabled multiplier and truncate toward zero."""
    return jnp.trunc(conditional_mult(damage, mult, cond)).astype(int)


def conditional_mult(value, mult, cond):
    # TODO, there is a buggy thing happening where multiplying a (2,1) array by a (2,) makes a (2,2) instead of (2,1)
    # this doesnt seem correct to me but i guess its how jax broadcasts so figure out how to deal with that
    """Apply a multiplier where the supplied condition is enabled."""
    return value * jnp.power(mult, cond)


def conditional_mult_prod(value, mult, cond):
    """Multiply all enabled modifiers before applying them to the value."""
    return value * jnp.prod(jnp.power(mult, cond))


def conditional_mult_prod_round(damage, mult, cond):
    # Truncate toward zero, matching numpy.fix.
    """Apply the product of enabled modifiers and truncate toward zero."""
    return jnp.trunc(conditional_mult_prod(damage, mult, cond)).astype(int)
