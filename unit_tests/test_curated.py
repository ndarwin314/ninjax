"""Curated export parsing, reproducible teams, and a projected acceptance battle."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.curated import load_sets, parse_sets, make_sandbox_pool, make_item_pool, sample_teams
from ninjax.enum_types import ItemEnum, AbilityEnum
from ninjax.prototype import run_curated


def test_curated_file_and_defaults():
    """Load the evolving curated file and verify independent parser defaults."""
    sets = load_sets()
    assert len(sets) >= 6
    assert all(len(s.moves) == 4 for s in sets)
    koraidon = next(s for s in sets if s.species == 'Koraidon')
    assert koraidon.level == 100 and koraidon.ivs == (31,) * 6
    assert next(s for s in sets if s.species == 'Ninetales-Alola').nature
    assert next(s for s in sets if s.species == 'Meowscarada').item == ''
    assert next(s for s in sets if s.species == 'Ogerpon-Wellspring').gender == 'F'
    assert next(s for s in sets if s.species == 'Gastrodon').ivs[1] == 0
    custom = parse_sets('''Miraidon @ Choice Specs
Ability: Hadron Engine
Level: 72
IVs: 0 Atk / 30 Spe
EVs: 252 SpA / 252 Spe
Timid Nature
- Electro Drift
- Draco Meteor
- Volt Switch
- Parabolic Charge
''')[0]
    assert custom.level == 72
    assert custom.ivs == (31, 0, 31, 31, 31, 30)
    assert custom.evs == (0, 0, 0, 252, 0, 252)
    default = parse_sets('''Meowscarada
Ability: Protean
- Triple Axel
- Flower Trick
- Worry Seed
- Acrobatics
''')[0]
    assert default.nature == 'Serious' and default.evs == (0,) * 6
    with pytest.raises(ValueError, match='unrecognized field'):
        parse_sets('X @ Y\nAbility: Z\nUnexpected: ignored')


def test_sampled_teams_and_explicit_omissions():
    """Sampling is reproducible and projected mechanics are reported."""
    sets = load_sets()
    pool, omissions = make_sandbox_pool(sets)
    assert len(omissions) == len(sets)
    assert any(report['moves'] for report in omissions)
    assert 'Beat Up' in next(report for report in omissions if report['species'] == 'Maushold')['moves']
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


def test_curated_item_projection():
    """Supported items survive sampling; unimplemented items remain reported."""
    sets = load_sets()
    pool, reports = make_item_pool(sets, abilities=True)
    for species, item in [('Koraidon', ItemEnum.LIFE_ORB), ('Flutter Mane', ItemEnum.BOOSTER_ENERGY),
                          ('Rillaboom', ItemEnum.LOADED_DICE)]:
        i = next(i for i, s in enumerate(sets) if s.species == species)
        assert pool.item[i, 0] == item and reports[i]['item'] is None
    i = next(i for i, s in enumerate(sets) if s.species == 'Flutter Mane')
    assert pool.ability[i, 0] == AbilityEnum.PROTOSYNTHESIS
    i = next(i for i, s in enumerate(sets) if s.species == 'Altaria')
    assert pool.item[i, 0] == ItemEnum.MENTAL_HERB and reports[i]['item'] is None
    i = next(i for i, s in enumerate(sets) if s.species == 'Gastrodon')
    assert pool.item[i, 0] == ItemEnum.NONE and reports[i]['item'] == 'Absorb Bulb'
    teams, indices = jax.jit(sample_teams)(jax.random.key(3), pool)
    assert jnp.array_equal(teams.item, pool.item[indices])


@pytest.mark.parametrize('seed', [0, 17, 42])
@pytest.mark.jit
def test_complete_curated_sandbox(seed):
    """Complete a seeded curated battle using the shared compiled step."""
    result = run_curated(seed=seed)
    assert 0 < result['turns'] <= 100
    assert result['reward'] in ([1., -1.], [-1., 1.], [0., 0.])
    assert len(result['omissions']) == 12
    assert all(len(set(side)) == 6 for side in result['set_indices'])
    assert len(result['decisions']) < 1000
    assert all(min(d['hp']) >= 0 for d in result['decisions'])
