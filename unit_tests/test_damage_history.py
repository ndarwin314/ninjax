"""Damage history behavior at the move and HP-commit boundaries."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_move
from ninjax.enums.moves import MoveType
from ninjax.records.battle_history import DamageHistory
from ninjax.side.faint_effects import commit_hp_change
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def step():
    """Share the compiled move entry point between history cases."""
    return jax.jit(step_move)


def test_history_turn_reset_preserves_battle_counters():
    """Turn cleanup retains accumulated hits and faint counts."""
    history = DamageHistory().replace(
        times_attacked=jnp.ones((2, 6), jnp.int32),
        total_fainted=jnp.array([2, 1]),
        hurt_this_turn=jnp.ones((2, 6), bool),
        counter_prepared=jnp.array([0, 1]))
    cleared = history.reset_turn()
    assert jnp.all(cleared.times_attacked == 1)
    assert jnp.array_equal(cleared.total_fainted, jnp.array([2, 1]))
    assert not jnp.any(cleared.hurt_this_turn)
    assert jnp.all(cleared.counter_prepared == -1)


def test_indirect_damage_marks_hurt_this_turn(neutral_type):
    """A Pokémon losing HP through residual damage enables Assurance."""
    _, state = neutral_type
    hp = state.active_field('current_hp', 1)
    result = commit_hp_change(state, 1, hp, hp - 1, state.active_stats(1)[0], -1, 0)
    assert result.damage_history.hurt_this_turn[1, 0]


def test_counter_records_only_prepared_category(neutral_type, step):
    """A prepared Counter remembers physical damage and ignores special damage."""
    key, state = neutral_type
    state = equip_move(state, 'Tackle')
    history = state.damage_history.replace(
        counter_prepared=state.damage_history.counter_prepared.at[1].set(MoveType.PHYSICAL))
    state = state.replace(damage_history=history)
    _, result = step(key, state, 0, 0, False, False)
    assert result.damage_history.times_attacked[1, 0] == 1
    assert result.damage_history.counter_damage[1, 0] == 2 * result.hit_results[0].damage
    assert result.damage_history.counter_source_slot[1, 0] == 0

    special = equip_move(state, 'Thunderbolt')
    _, ignored = step(key, special, 0, 0, False, False)
    assert ignored.damage_history.counter_damage[1, 0] == 0


def test_substitute_hit_counts_but_cannot_be_countered(neutral_type, step):
    """Rage Fist counts a hit to Substitute without recording retaliation."""
    key, state = neutral_type
    state = equip_move(state, 'Tackle')
    history = state.damage_history.replace(
        counter_prepared=state.damage_history.counter_prepared.at[1].set(MoveType.PHYSICAL))
    volatile = state.volatile_status
    state = state.replace(damage_history=history, volatile_status=volatile.replace(
        substitute_hp=volatile.substitute_hp.at[1, 0].set(20)))
    _, result = step(key, state, 0, 0, False, False)
    assert result.damage_history.times_attacked[1, 0] == 1
    assert result.damage_history.counter_damage[1, 0] == 0
