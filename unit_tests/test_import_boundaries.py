"""Fresh-process checks for concrete owners and dependency direction."""
import subprocess
import sys

import pytest


def test_profiling_cli_import_stays_lightweight():
    """Collecting benchmark targets must not load JAX before CLI limits apply."""
    script = '''
import sys
from tools.profile_compile import COMPONENT_TARGETS

assert 'move' in COMPONENT_TARGETS and 'dispatch' in COMPONENT_TARGETS
assert 'jax' not in sys.modules
assert 'ninjax.battle.environment' not in sys.modules
assert 'tools.compile_profile' not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('module', [
    'ninjax.teams.construction', 'ninjax.teams.curated', 'ninjax.records.hits', 'ninjax.moves',
    'ninjax.moves.status', 'ninjax.moves.multi_hit',
    'ninjax.moves.execution', 'ninjax.moves.routing',
    'ninjax.battle.switching', 'ninjax.battle.residuals',
    'ninjax.battle.types', 'ninjax.records.events',
    'ninjax.enums.abilities', 'ninjax.enums.items', 'ninjax.enums.pokemon',
    'ninjax.enums.moves', 'ninjax.enums.field',
    'ninjax.calculations.stats', 'ninjax.calculations.type_effectiveness',
])
def test_shared_modules_do_not_import_battle_execution(module):
    """Construction and mechanics must load without the battle scheduler."""
    script = '''
import importlib
import sys

importlib.import_module(sys.argv[1])
unexpected = {'ninjax.prototype', 'ninjax.battle.environment'} & sys.modules.keys()
assert not unexpected, sorted(unexpected)
if sys.argv[1] == 'ninjax.moves':
    forbidden = {'ninjax.side', 'ninjax.moves.status', 'ninjax.moves.hit_resolution'}
    assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
if sys.argv[1] == 'ninjax.moves.routing':
    assert 'ninjax.moves.execution' not in sys.modules
if sys.argv[1] == 'ninjax.battle.residuals':
    forbidden = {'ninjax.battle.switching', 'ninjax.ability_effects'}
    assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
'''
    result = subprocess.run(
        [sys.executable, '-c', script, module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_removed_compatibility_modules_are_unavailable():
    """Only concrete module paths remain after the migration cleanup."""
    script = '''
import importlib.util

for module in ('ninjax.game_logic', 'ninjax.held_effects', 'ninjax.move_effects',
               'ninjax.move', 'ninjax.curated_move_effects', 'ninjax.enum_types',
               'ninjax.hit', 'ninjax.events', 'ninjax.stats', 'ninjax.utils',
               'ninjax.damage', 'ninjax.team_builder', 'ninjax.curated',
               'ninjax.switching', 'ninjax.residuals'):
    assert importlib.util.find_spec(module) is None, module
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
    'ninjax.ability_effects', 'ninjax.battle.environment',
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
        'ninjax.battle.switching', 'ninjax.battle.residuals',
        'ninjax.ability_effects', 'ninjax.moves.execution', 'ninjax.moves.routing',
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


@pytest.mark.parametrize('package', [
    'ninjax.battle', 'ninjax.teams', 'ninjax.records',
    'ninjax.calculations', 'ninjax.enums',
])
def test_organizational_packages_do_not_initialize_the_engine(package):
    """Package parents remain safe for lower-level imports and JAX tracing."""
    script = """
import importlib
import sys
module = importlib.import_module(sys.argv[1])
assert module.__all__ == ()
assert 'jax' not in sys.modules
assert 'ninjax.battle.environment' not in sys.modules
assert 'ninjax.moves.execution' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', script, package], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
