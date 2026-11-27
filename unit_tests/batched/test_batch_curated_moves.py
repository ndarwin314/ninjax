"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.moves.properties import move_overrides
from ninjax.side.field_conditions import field_status
from ninjax.side.stat_effects import stat_changes
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_new_callbacks_jit_vmap_scan(neutral_type):
    """Shared callbacks preserve their PyTree structure under transformations."""
    _, state = neutral_type
    move = make_move('Light Screen')

    changes = make_move('Dragon Pulse').replace(self_stat=jnp.arange(1, 6), self_stages=jnp.ones(5, jnp.int32))
    def transition(s):
        """Apply one field effect and one conditional power override."""
        s = field_status(s, 0, move)
        s = stat_changes(s, 0, changes.self_stat, changes.self_stages)
        return s, move_overrides(s, 0, make_move('Stored Power')).base_power
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out, powers = jax.jit(jax.vmap(transition))(batch)
    assert jnp.all(out.side_conditions.light_screen[:, 0, 0] == 5)
    assert jnp.all(out.boosts.normal_boosts[:, 0, 1:] == 1)
    final, _ = jax.jit(lambda s: jax.lax.scan(lambda s, _: transition(s), s, None, length=2))(state)
    assert final.side_conditions.light_screen[0, 0] == 5
    assert jnp.all(final.boosts.normal_boosts[0, 1:] == 2)
