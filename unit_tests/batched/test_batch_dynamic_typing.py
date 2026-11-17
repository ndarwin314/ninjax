"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.pokemon import FormEnum as F, Type
from ninjax.moves.properties import effective_move
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.mark.jit
def test_dynamic_types_jit_vmap_scan(neutral_type):
    """Dynamic typing uses immutable arrays under JAX transformations."""
    _, state = neutral_type
    move = make_move('Ivy Cudgel')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    forms = batch.team.form.at[0, 0, 0, 0].set(F.OGERPON_WELLSPRING).at[1, 0, 0, 0].set(F.OGERPON_HEARTHFLAME)
    batch = batch.replace(team=batch.team.replace(form=forms))
    types = jax.jit(jax.vmap(lambda s: effective_move(s, 0, move).type[0]))(batch)
    assert jnp.array_equal(types, jnp.array([Type.WATER, Type.FIRE]))
    _, types = jax.jit(lambda s: jax.lax.scan(lambda s, _: (s, effective_move(s, 0, move).type[0]), s, None, length=2))(state)
    assert jnp.all(types == Type.GRASS)
