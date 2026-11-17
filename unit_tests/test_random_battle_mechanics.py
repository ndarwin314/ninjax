"""Pinned full-format inventory and controlled Showdown mechanics comparisons.

Unsupported moves are explicit skips, not successful mechanics comparisons.
The damage probes isolate ordinary neutral hits; interaction parity is separate.
Regenerate with tools/record_random_battle_mechanics.cjs against the local server's
simulator checkout. RNG streams differ, so compare the corresponding damage roll.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import Status, Type
from ninjax.moves.effects import secondary_effects
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.moves.status import do_status_move
from ninjax.side.updates import update_active
from ninjax.teams.construction import CATALOGUE, make_move
from unit_tests.test_battle import neutral_type


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/showdown/gen9-random-mechanics.json').read_text())


def test_inventory_is_pinned_and_complete():
    """Every species role retains its entire candidate movepool and abilities."""
    assert REFERENCE['revision'] == '23b41aa29e2ff5ff80e03f57b257b595a5852047'
    assert REFERENCE['format'] == 'gen9randombattle'
    assert len(REFERENCE['roles']) == 879
    assert len(REFERENCE['moves']) == 348
    union = {name for role in REFERENCE['roles'] for name in role['movepool']}
    assert union == set(REFERENCE['moves'])
    assert set(REFERENCE['abilities']) == {a for role in REFERENCE['roles'] for a in role['abilities']}


@pytest.mark.parametrize('role', REFERENCE['roles'], ids=lambda r: f"{r['species']}-{r['index']}")
def test_every_random_battle_role(role):
    """Inventory all selectable roles without pretending unsupported moves work."""
    # Ditto intentionally has only Transform; its other moves come from Imposter.
    assert len(set(role['movepool'])) >= (1 if role['species'] == 'ditto' else 4)
    assert role['abilities'] and role['teraTypes']
    assert 1 <= role['level'] <= 100
    assert all(REFERENCE['moves'][name]['num'] > 0 for name in role['movepool'])


@pytest.mark.parametrize('name', sorted(REFERENCE['moves']))
def test_supported_move_metadata_matches_showdown(name):
    """Check represented scalar metadata for every move in the format."""
    if name not in CATALOGUE:
        pytest.skip('Unimplemented random-battle move: ' + name)
    expected = REFERENCE['moves'][name]
    move = make_move(name)
    assert move.implemented[0]
    assert int(move.identifier[0]) == expected['num']
    assert int(move.type[0]) == Type[expected['type'].upper()]
    assert int(move.base_power[0]) == expected['power']
    assert int(move.priority[0]) == expected['priority']
    assert int(move.max_pp[0]) == expected['pp'] * 8 // 5
    assert bool(move.contact[0]) == expected['contact']
    np.testing.assert_allclose(move.accuracy[0], expected['accuracy'])


def neutral_reference_state(state):
    """Use Mew's reference stats with no STAB, criticals, items or abilities."""
    stats = state.team.stat_table.replace(evs=jnp.zeros((2, 6, 6), jnp.int32))
    team = state.team.replace(stat_table=stats, current_hp=jnp.full((2, 6, 1), 341),
        type_list=jnp.zeros((2, 6, 2), jnp.int32))
    state = state.replace(team=team)
    # Battle Armor suppresses even guaranteed criticals; no other damage modifier.
    return update_active(state, 1, state.active[1].replace(ability=jnp.array([AbilityEnum.BATTLE_ARMOR])))


@pytest.mark.parametrize('probe', REFERENCE['damage'], ids=lambda p: p['name'])
def test_neutral_hit_matches_showdown(neutral_type, probe):
    """Compare an engine hit to Showdown's exact integer damage at the same roll."""
    key, state = neutral_type
    state = neutral_reference_state(state)
    np.testing.assert_array_equal(state.active.stats[0, 1:], [probe['stats'][s] for s in ('atk', 'def', 'spa', 'spd', 'spe')])
    # resolve_damaging_hit splits its incoming key into key/critical/roll keys.
    roll_key = jax.random.split(key, 3)[2]
    roll = int(jax.random.randint(roll_key, (), 85, 101))
    _, out, damage, _ = resolve_damaging_hit(key, state, 0, make_move(probe['name']))
    expected = probe['rolls'][100-roll]
    assert int(damage[0]) == expected
    assert int(out.active.current_hp[1, 0]) == probe['hp'] - expected


@pytest.mark.parametrize('probe', REFERENCE['boosts'], ids=lambda p: p['name'])
def test_setup_boosts_match_showdown(neutral_type, probe):
    """Compare all seven stat stages after an actual setup move in each engine."""
    key, state = neutral_type
    _, out = do_status_move(key, state, 0, make_move(probe['name']), 0, 0)
    actual = list(np.asarray(out.boosts.normal_boosts[0, 1:])) + list(np.asarray(out.boosts.acc_boosts[0]))
    expected = [probe['expected'][s] for s in ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')]
    assert actual == expected


@pytest.mark.parametrize('probe', REFERENCE['secondaries'], ids=lambda p: p['name'])
def test_triggered_secondary_matches_showdown(neutral_type, probe):
    """Compare forced secondary outcomes from real Showdown moves, without RNG parity."""
    key, state = neutral_type
    move = make_move(probe['name']).replace(secondary_chance=jnp.array([1.]))
    _, out = secondary_effects(key, state, 0, move)
    statuses = {'': Status.NONE, 'par': Status.PARALYZE, 'brn': Status.BURN,
                'psn': Status.POISON, 'tox': Status.TOXIC, 'slp': Status.SLEEP,
                'frz': Status.FREEZE}
    assert int(out.active.status[1, 0]) == statuses[probe['status']]
    actual = list(np.asarray(out.boosts.normal_boosts[1, 1:])) + list(np.asarray(out.boosts.acc_boosts[1]))
    assert actual == [probe['boosts'][s] for s in ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')]
    assert bool(out.volatile_status.flinched[1, 0]) == probe['flinch']
    assert bool(out.volatile_status.confusion_turns[1, 0] > 0) == probe['confusion']
