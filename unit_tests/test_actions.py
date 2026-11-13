"""Indexed action PyTrees and compatibility with existing integer callers."""
import jax
import jax.numpy as jnp

from ninjax.battle import ActionType, Battle, BattleAction, BattleParams, Phase, step_action
from unit_tests.test_battle import neutral_type


def test_indexed_switches_and_pending_actions(neutral_type):
    """Both indexed switches execute and the scheduler retains structured actions."""
    key, battle = neutral_type
    env = Battle(battle.team)
    state, _ = env.reset(key)
    action = BattleAction(jnp.array([1, 2]), jnp.full(2, ActionType.SWITCH))
    env.action_spec.validate(action)
    out, ts = env.step(state, action)
    assert not ts.extras['invalid_action']
    assert jnp.array_equal(out.battle.active_index[:, 0], action.index)
    assert isinstance(out.pending_actions, BattleAction)
    assert out.turn == 1


def test_noop_and_action_validation(neutral_type):
    """No-op preserves the active Pokémon; invalid kinds and indices are rejected."""
    key, battle = neutral_type
    action = BattleAction.from_legacy(jnp.int32(14))
    _, out = step_action(key, battle, action, 0)
    assert jnp.array_equal(out.active_index, battle.active_index)
    assert jnp.array_equal(out.active.current_hp, battle.active.current_hp)
    invalid = BattleAction(jnp.array([6, 0]), jnp.array([ActionType.SWITCH, ActionType.ATTACK]))
    assert invalid.encode()[0] == -1
    assert BattleAction(jnp.array([0, 0]), jnp.array([99, ActionType.CONTINUED])).encode()[0] == -1
    env = Battle(battle.team)
    state, _ = env.reset(key)
    out, ts = env.step(state, invalid)
    assert ts.extras['invalid_action'] and out.turn == 0
    _, ts = env.step(state, BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.CONTINUED)))
    assert ts.extras['invalid_action']


def test_preview_with_indexed_action(neutral_type):
    """Team-preview selection remains available alongside indexed battle actions."""
    key, battle = neutral_type
    env = Battle(battle.team, BattleParams(team_preview=True))
    state, _ = env.reset(key)
    selection = jnp.zeros((2, 6), bool).at[0, 1].set(True).at[1, 2].set(True)
    action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.NO_OP), selection)
    out, ts = env.step(state, action)
    assert out.phase == Phase.TURN and not ts.extras['invalid_action']
    assert jnp.array_equal(out.selected, selection)


def test_action_pytree_transforms():
    """Action adapters can be batched and traced without Python value dispatch."""
    encoded = jnp.array([0, 3, 4, 5, 8, 13, 14])
    action = BattleAction.from_legacy(encoded)
    assert jnp.array_equal(jax.vmap(lambda a: a.encode())(action), encoded)
    assert jax.eval_shape(lambda a: a.encode(), action).shape == (7,)


def test_indexed_attack_dispatch(neutral_type):
    """A scalar attack action executes its selected move and spends one PP."""
    key, battle = neutral_type
    action = BattleAction(jnp.int32(0), jnp.int32(ActionType.ATTACK))
    _, out = step_action(key, battle, action, 0)
    assert out.active.moves.current_pp[0, 0, 0] == battle.active.moves.current_pp[0, 0, 0] - 1
    assert out.active.current_hp[1, 0] < battle.active.current_hp[1, 0]


def test_transformation_metadata_and_rejection(neutral_type):
    """Transformations retain identity under batching and cannot be ignored."""
    from ninjax.battle import Transformation
    key, battle = neutral_type
    for kind in Transformation:
        action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.ATTACK), transformation=kind)
        assert action.transformation.shape == (2,)
        assert jnp.all(jax.vmap(lambda a: a.transformation)(action) == kind)
        if kind != Transformation.NONE:
            assert jnp.all(action.encode() == -1)
    env = Battle(battle.team)
    state, _ = env.reset(key)
    for kind in (Transformation.TERA, Transformation.MEGA, Transformation.Z_MOVE, Transformation.DYNAMAX):
        action = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.ATTACK), transformation=kind)
        env.action_spec.validate(action)
        out, ts = env.step(state, action)
        assert ts.extras['invalid_action'] and out.turn == 0
        assert jnp.array_equal(out.battle.team.moves.current_pp, state.battle.team.moves.current_pp)
    legacy = BattleAction.from_legacy(jnp.array([0, 8]))
    assert jnp.all(legacy.transformation == Transformation.NONE)
