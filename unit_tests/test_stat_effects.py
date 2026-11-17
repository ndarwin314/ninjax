"""Opposing-drop reactions stay distinct from basic stage writes."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.pokemon import StatEnum as S
from ninjax.side.stat_effects import reduce_boosts, stat_changes
from ninjax.side.updates import add_boosts, update_active_fields
from unit_tests.test_battle import neutral_type


compiled_reduce = jax.jit(reduce_boosts)


@pytest.mark.parametrize('ability,defense,attack,special_attack', [
    (A.NONE, -1, 0, 0),
    (A.DEFIANT, -1, 2, 0),
    (A.COMPETITIVE, -1, 0, 2),
    (A.CONTRARY, 1, 0, 0),
    (A.CLEAR_BODY, 0, 0, 0),
    (A.BIG_PECKS, 0, 0, 0),
])
def test_opposing_drop_responses(neutral_type, ability, defense, attack, special_attack):
    """Only actual opposing reductions activate represented reactive abilities."""
    _, state = neutral_type
    state = update_active_fields(state, 1, ability=jnp.array([ability]))
    result = compiled_reduce(state, 1, S.DEFENSE, 1)
    stages = result.boosts.normal_boosts[1]
    assert stages[S.DEFENSE] == defense
    assert stages[S.ATTACK] == attack
    assert stages[S.SPECIAL_ATTACK] == special_attack


def test_capped_drop_does_not_activate_defiant(neutral_type):
    """A failed drop at the lower stage limit leaves offensive stages alone."""
    _, state = neutral_type
    state = update_active_fields(state, 1, ability=jnp.array([A.DEFIANT]))
    state = state.replace(boosts=state.boosts.replace(
        normal_boosts=state.boosts.normal_boosts.at[1, S.DEFENSE].set(-6)))
    result = compiled_reduce(state, 1, S.DEFENSE, 1)
    assert result.boosts.normal_boosts[1, S.DEFENSE] == -6
    assert result.boosts.normal_boosts[1, S.ATTACK] == 0


def test_self_drop_does_not_activate_defiant(neutral_type):
    """Self-inflicted reductions use the basic update without opposing hooks."""
    _, state = neutral_type
    state = update_active_fields(state, 1, ability=jnp.array([A.DEFIANT]))
    result = add_boosts(state, 1, S.DEFENSE, -1)
    assert result.boosts.normal_boosts[1, S.DEFENSE] == -1
    assert result.boosts.normal_boosts[1, S.ATTACK] == 0


def test_accuracy_drop_activates_defiant(neutral_type):
    """Accuracy/evasion reductions share the opposing-drop response path."""
    _, state = neutral_type
    state = update_active_fields(state, 1, ability=jnp.array([A.DEFIANT]))
    result = stat_changes(state, 1, jnp.array([6]), jnp.array([-1]), opposing=True)
    assert result.boosts.acc_boosts[1, 0] == -1
    assert result.boosts.normal_boosts[1, S.ATTACK] == 2
