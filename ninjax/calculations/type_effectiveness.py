"""Type matchup lookup and effectiveness multiplication."""

import jax.numpy as jnp
from chex import Array

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.moves import EffectivenessRule
from ninjax.enums.pokemon import Type


TYPE_EFFECTIVENESS = jnp.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 0, 1, 1, 1/2, 1],
        [1, 1, 1/2, 1/2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1/2, 1, 1/2, 1, 2, 1],
        [1, 1, 2, 1/2, 1, 1/2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1/2, 1, 1, 1],
        [1, 1, 1, 2, 1/2, 1/2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1/2, 1, 1, 1],
        [1, 1, 1/2, 2, 1, 1/2, 1, 1, 1/2, 2, 1/2, 1, 1/2, 2, 1, 1/2, 1, 1/2, 1],
        [1, 1, 1/2, 1/2, 1, 2, 1/2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1/2, 1],
        [1, 2, 1, 1, 1, 1, 2, 1, 1/2, 1, 1/2, 1/2, 1/2, 2, 0, 1, 2, 2, 1/2],
        [1, 1, 1, 1, 1, 2, 1, 1, 1/2, 1/2, 1, 1, 1, 1/2, 1/2, 1, 1, 0, 2],
        [1, 1, 2, 1, 2, 1/2, 1, 1, 2, 1, 0, 1, 1/2, 2, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1/2, 1],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1/2, 1, 1, 1, 1, 0, 1/2, 1],
        [1, 1, 1/2, 1, 1, 2, 1, 1/2, 1/2, 1, 1/2, 2, 1, 1, 1/2, 1, 2, 1/2, 1/2],
        [1, 1, 2, 1, 1, 1, 2, 1/2, 1, 1/2, 2, 1, 2, 1, 1, 1, 1, 1/2, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1/2, 0],
        [1, 1, 1, 1, 1, 1, 1, 1/2, 1, 1, 1, 2, 1, 1, 2, 1, 1/2, 1, 1/2],
        [1, 1, 1/2, 1/2, 1/2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1/2, 2],
        [1, 1, 1/2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1, 1, 1, 2, 2, 1/2, 1]])

def hit_type_effectiveness(
        move_type,
        defender_types,
        defender_ability,
        gravity_active,
        effectiveness_rule=0):
    """Return the raw item matchup and the damage matchup from one type lookup."""
    factors = type_effectiveness_factors(move_type, defender_types, effectiveness_rule)
    raw = jnp.prod(factors)
    # Gravity neutralizes Flying's type factor and suppresses Levitate.
    gravity = jnp.asarray(gravity_active)
    ground = move_type == Type.GROUND
    damage_factors = jnp.where(gravity & ground & (defender_types == Type.FLYING), 1, factors)
    levitate = (defender_ability == AbilityEnum.LEVITATE) & ground & ~gravity
    return raw, jnp.prod(damage_factors) * (1 - levitate)


def calculate_effectiveness_multiplier(attacking_type, defending_types, effectiveness_rule=0) -> Array:
    """Multiply type matchup factors across the defending Pokémon types."""
    return jnp.prod(type_effectiveness_factors(attacking_type, defending_types, effectiveness_rule))


def type_effectiveness_factors(attacking_type, defending_types, effectiveness_rule=0) -> Array:
    """Return per-type matchup factors for the resolved type and move rule.

    Freeze-Dry treats Water as weak regardless of its resolved attacking type.
    Flying Press multiplies each resolved-type factor by Flying's factor; its
    actual move type remains unchanged for STAB and other type-based effects.
    """
    factors = TYPE_EFFECTIVENESS[attacking_type, defending_types]
    # Apply move-specific chart callbacks separately for each defending type.
    factors = jnp.where((effectiveness_rule == EffectivenessRule.FREEZE_DRY) &
                        (defending_types == Type.WATER), 2., factors)
    flying_factors = TYPE_EFFECTIVENESS[Type.FLYING, defending_types]
    return jnp.where(effectiveness_rule == EffectivenessRule.FLYING_PRESS,
                     factors * flying_factors, factors)
