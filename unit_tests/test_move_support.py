"""Regression checks for honest coverage and unavailable move projection."""
import json
from pathlib import Path

import pytest

from ninjax.moves.support import SUPPORTED_MOVES, is_move_supported
from ninjax.teams.construction import CATALOGUE, make_move, make_sandbox_teams
from tools.audit_random_battle_mechanics import coverage_report


@pytest.mark.parametrize('name', ['Doom Desire', 'Smack Down', 'Future Sight', 'Attract'])
def test_catalogue_membership_does_not_enable_missing_effects(name):
    """Known metadata must not advertise moves with missing execution rules."""
    assert name in CATALOGUE
    move = make_move(name)
    assert not move.implemented[0]
    assert move.current_pp[0] == 0
    assert move.identifier[0] == CATALOGUE[name]['num']


@pytest.mark.parametrize('name', ['Tackle', 'Protect', 'Sleep Talk', 'Flame Charge', 'Arm Thrust', 'Milk Drink', 'Freeze-Dry', 'Flying Press', 'Judgment'])
def test_reviewed_effects_remain_available(name):
    """Dedicated and generic mechanics share the same support decision."""
    assert is_move_supported(name)
    move = make_move(name)
    assert move.implemented[0]
    assert move.current_pp[0] > 0


def test_support_inventory_is_explicit_and_valid():
    """Review entries must be unique catalogue names; unknown moves fail closed."""
    names = json.loads((Path(__file__).resolve().parents[1] / 'ninjax/data/supported-moves.json').read_text())
    assert len(names) == len(SUPPORTED_MOVES)
    assert SUPPORTED_MOVES <= CATALOGUE.keys()
    assert not is_move_supported('Future Catalogue Addition')
    assert not make_move('Future Catalogue Addition').implemented[0]


def test_team_omissions_include_known_but_unimplemented_moves():
    """Team reports and move PP agree for catalogue entries lacking handlers."""
    record = dict(set=dict(species='Mew', level=100, moves=['doomdesire', 'tackle', 'futuresight', 'protect']),
                  nature=dict(plus='atk', minus='atk'),
                  baseStats=dict.fromkeys(('hp', 'atk', 'def', 'spa', 'spd', 'spe'), 100),
                  types=['Psychic'])
    teams, omissions = make_sandbox_teams(dict(initial=[[record] * 6] * 2))
    assert all(report['moves'] == ['Doom Desire', 'Future Sight'] for report in omissions)
    assert list(teams.moves.implemented[0, 0, :, 0]) == [False, True, False, True]
    assert list(teams.moves.current_pp[0, 0, [0, 2], 0]) == [0, 0]


def test_coverage_uses_support_and_retains_role_gaps():
    """Known unsupported and unknown moves count as gaps, including affected roles."""
    reference = dict(revision='test', format='singles', moves={'Tackle': {}, 'Doom Desire': {}, 'Unknown': {}},
                     abilities=[], sampledItems=[], itemSampleTeams=0, damage=[], boosts=[], secondaries=[],
                     roles=[dict(species='mew', index=0, movepool=['Doom Desire'], abilities=[])])
    report = coverage_report(reference)
    assert report['supported_moves'] == 1
    assert report['supported_move_names'] == ['Tackle']
    assert report['missing_moves'] == ['Doom Desire', 'Unknown']
    assert report['roles_with_move_or_ability_gaps'] == ['mew:0']
    assert report['catalogue_moves'] == len(CATALOGUE)
    assert report['supported_catalogue_moves'] == len(SUPPORTED_MOVES)
    assert 'Doom Desire' in report['unsupported_catalogue_moves']


def test_checked_in_coverage_matches_current_support():
    """Keep published counts and lists synchronized with runtime projection."""
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / 'unit_tests/fixtures/showdown/gen9-random-mechanics.json').read_text())
    published = json.loads((root / 'docs/mechanics/gen9-random-coverage.json').read_text())
    assert published == coverage_report(reference)
