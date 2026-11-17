"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.ability_effects import worry_seed
from ninjax.enum_types import AbilityEnum as A
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_worry_seed_jit_vmap_scan(neutral_type):
    """Ability changes and repeated failure remain compatible with transforms."""
    _, state = neutral_type
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out = jax.jit(jax.vmap(lambda s: worry_seed(s, 0)))(batch)
    assert jnp.all(jax.vmap(lambda s: s.ability)(out)[:, 1, 0] == A.INSOMNIA)
    final, _ = jax.jit(lambda s: jax.lax.scan(lambda s, _: (worry_seed(s, 0), None), s, None, length=2))(state)
    assert final.ability[1, 0] == A.INSOMNIA
