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


def chain_modifier(modifier, numerator, enabled=True):
    """Chain a 4096-based factor, rounding intermediate modifier halves up."""
    factor = jnp.where(enabled, numerator, 4096).astype(jnp.int32)
    return (modifier * factor + 2048) >> 12


def apply_modifier(value, modifier):
    """Apply a 4096-based modifier with Showdown's half-down rounding."""
    return (jnp.asarray(value, jnp.int32) * modifier + 2047) >> 12
