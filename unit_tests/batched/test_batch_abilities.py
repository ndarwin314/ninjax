"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.ability_effects import frisk
from ninjax.enum_types import ItemEnum as I
from unit_tests.test_abilities import frisk_state
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_frisk_jit_vmap_scan(neutral_type):
    """Public knowledge and event updates compile under all three transformations."""
    _, state = neutral_type
    state = frisk_state(state)
    out = jax.jit(frisk)(state, 0)
    assert out.known_items[1, 0] == I.LIFE_ORB
    batch = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    out = jax.jit(jax.vmap(frisk, in_axes=(0, None)))(batch, 0)
    assert jnp.all(out.known_items[:, 1, 0] == I.LIFE_ORB)
    def reveal(s, _):
        """Repeat an entry reveal in a scanned state transition."""
        s = frisk(s, 0)
        return s, s.known_items
    final, history = jax.jit(lambda s: jax.lax.scan(reveal, s, None, length=2))(state)
    assert jnp.all(history[:, 1, 0] == I.LIFE_ORB)
    assert not final.events.overflow
