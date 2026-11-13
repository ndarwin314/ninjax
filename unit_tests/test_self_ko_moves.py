"""Explosion and Memento self-faint at their respective resolution points."""
import jax.numpy as jnp
import pytest
from ninjax.battle import step_move
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Type
from ninjax.hit import HitOutcome
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('mode', ['hit', 'protect', 'ghost', 'miss'])
def test_explosion_self_faints_even_when_target_unaffected(neutral_type, mode):
    """Explosion directly faints before the hit, independently of its outcome."""
    key, state = neutral_type
    state = equip_move(state, 'Explosion')
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.STURDY]), item=jnp.array([I.FOCUS_SASH])))
    if mode == 'protect':
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    if mode == 'ghost':
        state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE])))
    if mode == 'miss':
        mon = state.active[0]
        state = update_active(state, 0, mon.replace(moves=mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.current_hp[0, 0] == 0 and out.active.item[0, 0] == I.FOCUS_SASH
    assert out.volatile_status.explosion_hp[0, 0] == 0
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) == (mode == 'hit')


def test_damp_blocks_explosion(neutral_type):
    """A living effective Damp ability prevents the Explosion and its self-KO."""
    key, state = neutral_type
    state = equip_move(state, 'Explosion')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.DAMP])))
    _, out = step_move(key, state, 0, 0, False)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert out.hit_results[0].outcome == HitOutcome.BLOCKED
    assert out.known_abilities[1, 0] == A.DAMP
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1


@pytest.mark.parametrize('mode', ['hit', 'capped', 'clear_body', 'protect', 'substitute'])
def test_memento_requires_reaching_target(neutral_type, mode):
    """Memento self-faints on a hit even if the stat drops cannot apply."""
    key, state = neutral_type
    state = equip_move(state, 'Memento')
    if mode == 'hit':
        state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.MAGIC_GUARD]), item=jnp.array([I.FOCUS_SASH])))
    if mode == 'capped':
        state = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, jnp.array([1, 3])].set(-6)))
    if mode == 'clear_body':
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.CLEAR_BODY])))
    if mode in ('protect', 'substitute'):
        v = state.volatile_status
        if mode == 'protect':
            v = v.replace(protected=v.protected.at[1, 0].set(True))
        else:
            v = v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(20))
        state = state.replace(volatile_status=v)
    _, out = step_move(key, state, 0, 0, False)
    assert bool(out.active.current_hp[0, 0] == 0) == (mode in ('hit', 'capped', 'clear_body'))
    if mode == 'hit':
        assert out.boosts.normal_boosts[1, 1] == -2 and out.boosts.normal_boosts[1, 3] == -2
        assert out.active.item[0, 0] == I.FOCUS_SASH


def test_memento_miss_does_not_faint(neutral_type):
    """A missed Memento preserves the user and records the miss."""
    key, state = neutral_type
    state = equip_move(state, 'Memento')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert out.hit_results[0].outcome == HitOutcome.MISS


def test_explosion_simultaneous_final_ko(neutral_type):
    """The target wins if Explosion takes both final Pokémon out."""
    key, state = neutral_type
    state = equip_move(state, 'Explosion')
    state = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([1])))
    state = state.replace(roster=jnp.zeros((2, 6), bool).at[:, 0].set(True))
    _, out = step_move(key, state, 0, 0, False)
    assert jnp.all(out.active.current_hp == 0)
    assert out.winner == 1
