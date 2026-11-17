"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enum_types import MoveType
from ninjax.residuals import step_side_conditions
from ninjax.restrictions import apply_restriction, resolve_move
from ninjax.team_builder import make_move
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_explicit_restriction_inputs_transform():
    """An exhausted Encore falls back; a usable but disabled Encore is blocked."""

    ids = jnp.array([105, 136, 347, 182])
    categories = jnp.array([MoveType.STATUS, MoveType.PHYSICAL, MoveType.STATUS, MoveType.STATUS])
    pp = jnp.array([[4, 1, 0, 4], [4, 1, 4, 4]])

    def resolve(values):
        """Keep the selected slot and every restriction a traced input."""
        current_pp, index, disabled_id, disable, taunt, encore_id, encore, gravity = values
        return resolve_move(ids, categories, current_pp, index,
            disabled_move_id=disabled_id, disable_turns=disable, taunt_turns=taunt,
            encore_move_id=encore_id, encore_turns=encore, gravity_duration=gravity)

    inputs = (pp, *[jnp.full(2, value, jnp.int32) for value in (3, 347, 2, 0, 347, 2, 0)])
    expected = (jnp.array([3, 2]), jnp.array([False, True]))
    batched = jax.jit(jax.vmap(resolve))(inputs)
    scanned = jax.jit(lambda xs: jax.lax.scan(lambda c, x: (c, resolve(x)), None, xs)[1])(inputs)
    for result in (batched, scanned):
        for actual, wanted in zip(result, expected):
            assert jnp.array_equal(actual, wanted)


@pytest.mark.jit
def test_restrictions_transform(neutral_type):
    """Restriction state is compatible with jit, vmap, and scan."""
    key, state = neutral_type
    move = make_move('Taunt')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out = jax.jit(jax.vmap(lambda s: apply_restriction(s, 0, move)))(batch)
    assert jnp.all(out.volatile_status.taunt_turns[:, 1, 0] == 3)
    def body(s, _):
        """Carry the restriction through two residual phases."""
        return step_side_conditions(key, s)[1], None
    final, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=2))(jax.tree.map(lambda x: x[0], out))
    assert final.volatile_status.taunt_turns[1, 0] == 1
