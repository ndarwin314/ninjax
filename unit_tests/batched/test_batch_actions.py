"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp

from ninjax.battle.environment import Battle
from ninjax.battle.types import ActionType, BattleAction, Transformation
from unit_tests.test_battle import neutral_type


def test_action_pytree_transforms():
    """Action adapters can be batched and traced without Python value dispatch."""
    encoded = jnp.array([0, 3, 4, 5, 8, 13, 14])
    action = BattleAction.from_legacy(encoded)
    assert jnp.array_equal(jax.vmap(lambda a: a.encode())(action), encoded)
    assert jax.eval_shape(lambda a: a.encode(), action).shape == (7,)


def test_transformation_metadata_and_rejection(neutral_type):
    """Transformations retain identity under batching and cannot be ignored."""

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
