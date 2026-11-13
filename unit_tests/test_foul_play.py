"""Foul Play borrows the target's Attack while retaining user-owned modifiers."""
import jax
import jax.numpy as jnp
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Status
from ninjax.game_logic import resolve_damaging_hit
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type


def damage(key, state):
    """Return deterministic calculated damage for identical random keys."""
    return int(resolve_damaging_hit(key, state, 0, make_move('Foul Play'))[2][0])


def test_target_attack_and_stages_not_users(neutral_type):
    """Target Attack and its stages affect the attack; the user's do not."""
    key, state = neutral_type
    baseline = damage(key, state)
    boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 1].set(2)))
    assert damage(key, boosted) > baseline
    user_boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 1].set(2)))
    assert damage(key, user_boosted) == baseline
    mon = state.active[1]
    table = mon.stat_table.replace(base_stats=mon.stat_table.base_stats.at[1].set(150))
    stronger = update_active(state, 1, mon.replace(stat_table=table))
    assert damage(key, stronger) > baseline


def test_item_burn_and_ability_ownership(neutral_type):
    """Only the user's Choice Band, burn and Huge Power modify borrowed Attack."""
    key, state = neutral_type
    baseline = damage(key, state)
    target_band = update_active(state, 1, state.active[1].replace(item=jnp.array([I.CHOICE_BAND])))
    assert damage(key, target_band) == baseline
    user_band = update_active(state, 0, state.active[0].replace(item=jnp.array([I.CHOICE_BAND])))
    assert damage(key, user_band) > baseline
    target_burn = update_active(state, 1, state.active[1].replace(status=jnp.array([Status.BURN])))
    assert damage(key, target_burn) == baseline
    user_burn = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.BURN])))
    assert damage(key, user_burn) < baseline
    target_power = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.HUGE_POWER])))
    assert damage(key, target_power) == baseline
    user_power = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.HUGE_POWER])))
    assert damage(key, user_power) > baseline


def test_target_unaware_ignores_borrowed_stages(neutral_type):
    """The target's Unaware ignores its borrowed Attack stages for this attack."""
    key, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.UNAWARE])))
    boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 1].set(3)))
    assert damage(key, boosted) == damage(key, state)
    lowered = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 1].set(-3)))
    assert damage(key, lowered) == damage(key, state)


def test_foul_play_selection_traces(neutral_type):
    """The target-owner selector supports JAX batching without compiling a battle."""
    _, state = neutral_type
    from ninjax.enum_types import OffensiveStatOwner as Owner
    from ninjax.side import select_offensive_stats
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    result = jax.eval_shape(jax.vmap(lambda s: select_offensive_stats(s.active.stats, s.boosts, 0, Owner.TARGET, False, False)), batch)
    assert result.shape == (2, 6)


def test_foul_play_move_execution(neutral_type):
    """The catalogue move spends PP and uses the ordinary hit path."""
    from ninjax.battle import step_move
    from unit_tests.test_curated_moves import equip_move
    key, state = neutral_type
    state = equip_move(state, 'Foul Play')
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]


def test_critical_hit_ignores_negative_borrowed_stages(neutral_type):
    """A critical Foul Play ignores the target's negative Attack stages."""
    key, state = neutral_type
    move = make_move('Foul Play').replace(always_crit=jnp.array([True]))
    lowered = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 1].set(-3)))
    baseline = resolve_damaging_hit(key, state, 0, move)[2]
    actual = resolve_damaging_hit(key, lowered, 0, move)[2]
    assert jnp.array_equal(actual, baseline)
