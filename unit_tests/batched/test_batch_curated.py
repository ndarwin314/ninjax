"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp

from ninjax.curated import load_sets, make_sandbox_pool, sample_teams


def test_sampled_teams_and_explicit_omissions():
    """Sampling is reproducible and projected mechanics are reported."""
    sets = load_sets()
    pool, omissions = make_sandbox_pool(sets)
    assert len(omissions) == len(sets)
    assert all(not report['moves'] for report in omissions)
    assert jnp.all(pool.ability == 0) and jnp.all(pool.item == 0)
    assert jnp.all(jnp.where(pool.moves.implemented, True, pool.moves.current_pp == 0))
    # Known defaults and stat indexing: Koraidon is Jolly with 4 HP/252 Atk/252 Spe.
    assert jnp.array_equal(pool.stat_table.evs[0], jnp.array([4, 252, 0, 0, 0, 252]))
    assert pool.stat_table.nature.increased[0, 0] == 5
    assert pool.stat_table.nature.decreased[0, 0] == 3
    keys = jax.random.split(jax.random.key(42), 8)
    samples = jax.jit(jax.vmap(sample_teams, in_axes=(0, None)))(keys, pool)
    teams, indices = samples
    assert indices.shape == (8, 2, 6)
    assert teams.current_hp.shape == (8, 2, 6, 1)
    assert jnp.all(jnp.diff(jnp.sort(indices, axis=-1), axis=-1) > 0)
    again, repeated = jax.jit(sample_teams)(keys[0], pool)
    assert jnp.array_equal(indices[0], repeated)
    assert jnp.array_equal(again.species, pool.species[repeated])
    assert not jnp.array_equal(indices[0], indices[1])
    for row in range(len(sets)):
        assert omissions[row]['species'] == sets[row].species
        assert omissions[row]['ability'] == sets[row].ability
        assert omissions[row]['item'] == sets[row].item
