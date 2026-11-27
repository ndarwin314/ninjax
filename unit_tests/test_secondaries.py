"""Catalogue projection and hit resolution for multi-effect secondaries."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.moves import SecondaryEffectType
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.moves.multi_hit import resolve_multi_hit
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.fixture(scope='module')
def resolve_hit():
    """Compile the shared damage and secondary pipeline once."""
    return jax.jit(resolve_multi_hit)


def test_independent_secondaries_can_both_apply(neutral_type, resolve_hit):
    """Fire Fang can both burn and flinch, while Shield Dust blocks both."""
    key, state = neutral_type
    move = make_move('Fire Fang')
    effects = move.secondaries
    assert effects.mode[0] == SecondaryEffectType.DEFAULT
    assert jnp.allclose(effects.probabilities, jnp.array([.1, .1, 0.]))
    move = move.replace(secondaries=effects.replace(probabilities=jnp.array([1., 1., 0.])))
    _, out = resolve_hit(key, state, 0, move)
    assert out.active.status[1, 0] == Status.BURN
    assert out.volatile_status.flinched[1, 0]
    shielded = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.SHIELD_DUST])))
    _, out = resolve_hit(key, shielded, 0, move)
    assert out.active.status[1, 0] == Status.NONE
    assert not out.volatile_status.flinched[1, 0]


def test_independent_stat_and_flinch_slots(neutral_type, resolve_hit):
    """Triple Arrows applies its stat and flinch effects from separate slots."""
    key, state = neutral_type
    move = make_move('Triple Arrows')
    assert jnp.allclose(move.secondaries.probabilities, jnp.array([.5, .3, 0.]))
    move = move.replace(secondaries=move.secondaries.replace(probabilities=jnp.array([1., 1., 0.])))
    _, out = resolve_hit(key, state, 0, move)
    assert out.boosts.normal_boosts[1, StatEnum.DEFENSE] == -1
    assert out.volatile_status.flinched[1, 0]


@pytest.mark.parametrize('chance', [0., 1.])
def test_user_secondary_uses_its_slot_probability_through_substitute(neutral_type, resolve_hit, chance):
    """A user boost uses its own slot and remains eligible through Substitute."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(50)))
    move = make_move('Charge Beam')
    effects = move.secondaries
    move = move.replace(secondaries=effects.replace(probabilities=effects.probabilities.at[0].set(chance)))
    _, out = resolve_hit(key, state, 0, move)
    assert out.boosts.normal_boosts[0, StatEnum.SPECIAL_ATTACK] == int(chance)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_exclusive_secondary_does_not_retry_an_immune_status(neutral_type, resolve_hit):
    """Dire Claw selects a single outcome, even if that status cannot land."""
    key, state = neutral_type
    move = make_move('Dire Claw')
    effects = move.secondaries
    assert effects.mode[0] == SecondaryEffectType.EXCLUSIVE
    assert jnp.allclose(effects.probabilities, jnp.full(3, .5 / 3))
    move = move.replace(secondaries=effects.replace(probabilities=jnp.array([1., 0., 0.])))
    _, out = resolve_hit(key, state, 0, move)
    assert out.active.status[1, 0] == Status.POISON
    immune = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.POISON, Type.NONE])))
    _, out = resolve_hit(key, immune, 0, move)
    assert out.active.status[1, 0] == Status.NONE


def test_exclusive_secondary_caps_the_total_probability(neutral_type, resolve_hit):
    """Only one exclusive slot fires even when the scaled chances sum above one."""
    key, state = neutral_type
    move = make_move('Tri Attack')
    effects = move.secondaries
    assert jnp.allclose(effects.probabilities, jnp.full(3, .2 / 3))
    effects = effects.replace(probabilities=jnp.ones(3),
        status=jnp.array([Status.BURN, Status.NONE, Status.NONE]),
        confusion=jnp.array([False, True, False]), flinch=jnp.array([False, False, True]))
    _, out = resolve_hit(key, state, 0, move.replace(secondaries=effects))
    outcomes = jnp.array([out.active.status[1, 0] == Status.BURN,
                          out.volatile_status.confusion_turns[1, 0] > 0,
                          out.volatile_status.flinched[1, 0]])
    assert jnp.sum(outcomes) == 1
