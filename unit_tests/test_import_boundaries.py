"""Fresh-process checks for dependency direction and compatibility exports."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', [
    'ninjax.team_builder', 'ninjax.curated', 'ninjax.hit', 'ninjax.moves',
    'ninjax.moves.status', 'ninjax.moves.multi_hit',
    'ninjax.moves.execution', 'ninjax.moves.routing',
    'ninjax.switching', 'ninjax.residuals',
])
def test_shared_modules_do_not_import_battle_execution(module):
    """Construction and mechanics must load without the battle scheduler."""
    script = '''
import importlib
import sys

importlib.import_module(sys.argv[1])
unexpected = {'ninjax.prototype', 'ninjax.battle', 'ninjax.game_logic'} & sys.modules.keys()
assert not unexpected, sorted(unexpected)
if sys.argv[1] == 'ninjax.moves':
    forbidden = {'ninjax.side', 'ninjax.moves.status', 'ninjax.moves.hit_resolution'}
    assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
if sys.argv[1] == 'ninjax.moves.routing':
    assert 'ninjax.moves.execution' not in sys.modules
if sys.argv[1] == 'ninjax.residuals':
    forbidden = {'ninjax.switching', 'ninjax.ability_effects', 'ninjax.held_effects'}
    assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
'''
    result = subprocess.run(
        [sys.executable, '-c', script, module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_move_exports_retain_concrete_owners():
    """Legacy entry points retain identity without owning execution code."""
    script = '''
from ninjax import game_logic
from ninjax.moves import execution, routing

assert game_logic.move_used is execution.move_used
assert game_logic.dispatch_selected_move is execution.dispatch_selected_move
assert game_logic.execute_sleep_talk is execution.execute_sleep_talk
assert game_logic.move_hits is routing.move_hits
assert game_logic.move_misses is routing.move_misses
assert game_logic._prepare_normal_hit is routing._prepare_normal_hit
'''
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('module', [
    'ninjax.side', 'ninjax.side.state', 'ninjax.side.updates', 'ninjax.side.queries',
])
def test_side_foundation_does_not_import_effects(module):
    """Importing state or basic writes must not initialize reactive mechanics."""
    script = '''
import importlib
import sys

importlib.import_module(sys.argv[1])
forbidden = {
    'ninjax.held_effects', 'ninjax.ability_effects', 'ninjax.battle',
    'ninjax.side.hp_effects', 'ninjax.side.status_effects',
    'ninjax.side.item_effects', 'ninjax.side.stat_effects',
}
assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
'''
    result = subprocess.run(
        [sys.executable, '-c', script, module], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('reverse', [False, True])
def test_side_effects_import_in_either_order(reverse):
    """HP, status and item dependencies load independently of import order."""
    modules = [
        'ninjax.switching', 'ninjax.residuals',
        'ninjax.held_effects', 'ninjax.ability_effects',
        'ninjax.side.hp_effects', 'ninjax.side.status_effects',
        'ninjax.side.item_effects', 'ninjax.side.stat_effects',
    ]
    script = '''
import importlib
import sys

for module in sys.argv[1:]:
    importlib.import_module(module)

import ninjax.side as side
assert side.__all__ == ('BattleState', 'VolatileStatus')
assert side.BattleState.__module__ == 'ninjax.side.state'
assert side.VolatileStatus.__module__ == 'ninjax.side.state'
assert not hasattr(side, 'take_damage_value')
'''
    result = subprocess.run(
        [sys.executable, '-c', script, *(modules[::-1] if reverse else modules)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
