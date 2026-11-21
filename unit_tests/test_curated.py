"""Curated export parsing, reproducible teams, and a projected acceptance battle."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.prototype import run_curated
from ninjax.teams.curated import (
    load_sets,
    make_item_pool,
    make_sandbox_pool,
    parse_sets,
    sample_teams,
)


@pytest.fixture(scope='module')
def compiled_sample_teams():
    """Reuse the sampling executable across compatible team pools."""
    return jax.jit(sample_teams)


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


def test_sampled_teams_and_explicit_omissions(compiled_sample_teams):
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
    key = jax.random.key(42)
    teams, indices = compiled_sample_teams(key, pool)
    again, repeated = compiled_sample_teams(key, pool)
    assert indices.shape == (2, 6)
    assert jnp.all(jnp.diff(jnp.sort(indices, axis=-1), axis=-1) > 0)
    assert jnp.array_equal(indices, repeated)
    assert jnp.array_equal(teams.species, again.species)
    assert jnp.array_equal(again.species, pool.species[repeated])
    for row in range(len(sets)):
        assert omissions[row]['species'] == sets[row].species
        assert omissions[row]['ability'] == sets[row].ability
        assert omissions[row]['item'] == sets[row].item


def test_curated_item_projection(compiled_sample_teams):
    """Every current curated item survives construction and sampling."""
    sets = load_sets()
    pool, reports = make_item_pool(sets, abilities=True)
    for species, item in [('Koraidon', ItemEnum.LIFE_ORB), ('Flutter Mane', ItemEnum.BOOSTER_ENERGY),
                          ('Rillaboom', ItemEnum.LOADED_DICE), ('Maushold', ItemEnum.KINGS_ROCK),
                          ('Urshifu-Rapid-Strike', ItemEnum.PROTECTIVE_PADS),
                          ('Gastrodon', ItemEnum.ABSORB_BULB), ('Ogerpon-Wellspring', ItemEnum.WELLSPRING_MASK)]:
        i = next(i for i, s in enumerate(sets) if s.species == species)
        assert pool.item[i, 0] == item and reports[i]['item'] is None
    i = next(i for i, s in enumerate(sets) if s.species == 'Flutter Mane')
    assert pool.ability[i, 0] == AbilityEnum.PROTOSYNTHESIS
    i = next(i for i, s in enumerate(sets) if s.species == 'Altaria')
    assert pool.item[i, 0] == ItemEnum.MENTAL_HERB and reports[i]['item'] is None
    i = next(i for i, s in enumerate(sets) if s.species == 'Gastrodon')
    assert pool.item[i, 0] == ItemEnum.ABSORB_BULB and reports[i]['item'] is None
    assert all(report['item'] is None or not report['item'] for report in reports)
    teams, indices = compiled_sample_teams(jax.random.key(3), pool)
    assert jnp.array_equal(teams.item, pool.item[indices])


@pytest.mark.parametrize('seed', [0, 17, 42])
@pytest.mark.parametrize('enabled', [False, True], ids=['neutral', 'abilities-and-items'])
@pytest.mark.jit
def test_complete_curated_sandbox(seed, enabled, record_property):
    """Complete compiled battles with neutral or enabled curated abilities/items."""
    result = run_curated(seed=seed, abilities=enabled, items=enabled)
    assert 0 < result['turns'] <= 100
    assert result['reward'] in ([1., -1.], [-1., 1.], [0., 0.])
    assert len(result['omissions']) == 12
    assert all(len(set(side)) == 6 for side in result['set_indices'])
    assert len(result['decisions']) < 1000
    assert all(min(d['hp']) >= 0 for d in result['decisions'])
    assert all(not report['moves'] for report in result['omissions'])
    if enabled:
        assert all(report['ability'] is None and not report['item'] for report in result['omissions'])
    for name in ('seed', 'turns', 'reward'):
        record_property(name, result[name])
    record_property('decisions', len(result['decisions']))
