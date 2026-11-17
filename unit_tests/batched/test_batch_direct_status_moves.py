"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enum_types import Status
from ninjax.moves.status import direct_status_move, do_status_move
from ninjax.side.updates import update_active
from ninjax.team_builder import make_move
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_direct_hp_callbacks_transform(neutral_type):
    """Direct HP/status assignments support actual jit, vmap, and scan."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([20])))
    rest = make_move('Rest')
    split = make_move('Pain Split')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    keys = jax.random.split(key, 2)
    _, out = jax.jit(jax.vmap(lambda k, s: do_status_move(k, s, 0, rest, 0, 0)))(keys, batch)
    assert jnp.all(out.active.status[:, 0, 0] == Status.SLEEP)
    def step(carry, _):
        """Carry explicit state and key through repeated HP assignments."""
        return direct_status_move(carry[0], carry[1], 0, split), None
    (_, out), _ = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=2))((key, state))
    assert out.active.current_hp[0, 0] == out.active.current_hp[1, 0]
