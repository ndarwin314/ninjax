"""Bounded Copycat execution through the ordinary move dispatcher."""
import jax
import jax.numpy as jnp

from ninjax.battle.actions import step_move
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def install_copycat(state):
    """Enable Copycat locally while catalogue support is being validated."""
    state = equip_move(state, 'Copycat')
    mon = state.active[0]
    moves = mon.moves.replace(
        implemented=mon.moves.implemented.at[0, 0].set(True),
        current_pp=mon.moves.current_pp.at[0, 0].set(10),
        max_pp=mon.moves.max_pp.at[0, 0].set(10))
    return update_active(state, 0, mon.replace(moves=moves))


def test_copycat_calls_opponent_move_without_mutating_own_slot(neutral_type):
    """Copycat repeats a previous opponent attack and only spends its own PP."""
    key, state = neutral_type
    state = equip_move(state, 'Tackle', player=1)
    state = install_copycat(state)
    step = jax.jit(step_move)
    key, state = step(key, state, 1, 0, False, False)
    before = state.active.current_hp[1, 0]
    key, out = step(key, state, 0, 0, False, False)
    assert out.active.current_hp[1, 0] < before
    assert out.active.moves.identifier[0, 0, 0] == 383
    assert out.active.moves.current_pp[0, 0, 0] == 9


def test_copycat_fails_without_history(neutral_type):
    """An empty last-move record fails without altering either Pokémon's HP."""
    key, state = neutral_type
    state = install_copycat(state)
    _, out = jax.jit(step_move)(key, state, 0, 0, False, False)
    assert out.hit_results[0].outcome == HitOutcome.FAILED
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
