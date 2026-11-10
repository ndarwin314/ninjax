"""Offline, RNG-independent assertions derived from ten pinned Showdown games."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ninjax.enum_types import Type
from ninjax.stats import Nature, StatTable
from ninjax.utils import calculate_effectiveness_multiplier
from ninjax.move import Move
from ninjax.pokemon import Pokemon
from ninjax.side import BattleState, take_damage_percent
from ninjax.tree import stack
from ninjax.damage import compute_base_damage

ROOT = Path(__file__).parent / 'fixtures' / 'showdown'
GAMES = [json.loads(p.read_text()) for p in sorted(ROOT.glob('battle-*.json'), key=lambda p: int(p.stem.split('-')[1]))]
STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']


def test_corpus_and_policy():
    assert len(GAMES) == 10
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert len(manifest['games']) == 10
    for game in GAMES:
        assert game['revision'] == manifest['revision']
        assert game['winner'] in ('Policy1', 'Policy2')
        assert any(line.startswith('|win|') for line in game['log'])
        assert len(game['initial']) == 2
        for step in game['steps']:
            for decision in step['decisions']:
                if decision['reason'] in ('wait', 'preview'):
                    continue
                category, index = decision['choice'].split()
                pool = decision['switches'] if category == 'switch' else decision['moves']
                assert int(index) in pool
                if decision['reason'] in ('move', 'switch') and decision['moves'] and decision['switches']:
                    assert (category == 'switch') == (decision['roll'] < .1)


@pytest.mark.parametrize('game', GAMES, ids=lambda g: str(g['seed']))
def test_generated_team_stats(game):
    mons = [p for side in game['initial'] for p in side]
    def values(mapping, default):
        return [mapping.get(s, default) for s in STATS]
    table = StatTable(
        level=jnp.array([[p['set']['level']] for p in mons]),
        base_stats=jnp.array([values(p['baseStats'], 0) for p in mons]),
        evs=jnp.array([values(p['set'].get('evs', {}), 0) for p in mons]),
        ivs=jnp.array([values(p['set'].get('ivs', {}), 31) for p in mons]),
        nature=Nature(
            increased=jnp.array([[STATS.index(p['nature']['plus']) if p['nature']['plus'] else 1] for p in mons]),
            decreased=jnp.array([[STATS.index(p['nature']['minus']) if p['nature']['minus'] else 1] for p in mons])))
    expected = [[p['maxhp']] + [p['stats'][s] for s in STATS[1:]] for p in mons]
    np.testing.assert_array_equal(jax.jit(lambda t: t.stats)(table), expected)


@pytest.mark.parametrize('game', GAMES, ids=lambda g: str(g['seed']))
def test_types_of_moves_seen_in_trace(game):
    probes = game['typeProbes']
    attacking = jnp.array([int(Type[p['attackType'].upper()]) for p in probes])
    defending = jnp.array([[int(Type[t.upper()]) for t in p['defenderTypes']] +
                           [0] * (2-len(p['defenderTypes'])) for p in probes])
    actual = jax.jit(jax.vmap(calculate_effectiveness_multiplier))(attacking, defending)
    np.testing.assert_array_equal(actual, [p['multiplier'] for p in probes])


@jax.jit
def generic_damage(level, power, category, attack, defense):
    one = jnp.ones(1, jnp.int32)
    move = Move(move_type=one*category, max_pp=one, current_pp=one, type=one*0,
                base_power=one*power, accuracy=one, priority=one*0,
                offensive_stat=one*jnp.where(category == 0, 1, 3),
                defensive_stat=one*jnp.where(category == 0, 2, 4), crit_stage=one*0)
    return compute_base_damage(jnp.array([0]), jnp.array([0]), jnp.array([1.]),
                               jnp.array([level]), attack, defense, move,
                               jnp.array([power]), jnp.array([0]), jnp.array([0]))[0]


@pytest.mark.parametrize('game', GAMES, ids=lambda g: str(g['seed']))
def test_controlled_damage_probes(game):
    for probe in game['damageProbes']:
        attack = jnp.array([0] + [probe['attackStats'][s] for s in STATS[1:]])
        defense = jnp.array([0] + [probe['defenseStats'][s] for s in STATS[1:]])
        actual = generic_damage(jnp.int32(probe['level']), jnp.int32(probe['power']),
                                jnp.int32(probe['category'] == 'Special'), attack, defense)
        assert int(actual) == probe['expected'], probe


def residual_cases():
    for game_no, game in enumerate(GAMES):
        for step_no, step in enumerate(game['steps']):
            health = {}
            status = {}
            for i, side in enumerate(step['before']):
                p = side['active'][0]
                health[f"p{i+1}a: {p['name']}"] = (p['hp'], p['maxhp'])
                status[f"p{i+1}a: {p['name']}"] = p['status']
            for event in step['events']:
                parts = event.split('|')
                if len(parts) < 4:
                    continue
                kind, name = parts[1:3]
                if kind == '-status':
                    status[name] = parts[3]
                if kind not in ('-damage', '-heal', 'switch', 'drag'):
                    continue
                raw = parts[3].split()[0]
                if '/' in raw:
                    hp, maxhp = map(int, raw.split('/'))
                elif raw == '0' and name in health:
                    hp, maxhp = 0, health[name][1]
                else:
                    continue
                previous = health.get(name)
                if kind == '-damage' and previous:
                    fraction = None
                    if '[from] brn' in parts:
                        fraction = 1/16
                    elif '[from] psn' in parts and status.get(name) == 'psn':
                        fraction = 1/8
                    elif '[from] Sandstorm' in parts:
                        fraction = 1/16
                    if fraction is not None:
                        yield (f'game{game_no+1}-decision{step_no}', previous[0], maxhp, fraction, hp)
                health[name] = (hp, maxhp)


RESIDUALS = list(residual_cases())


@jax.jit
def apply_fraction(hp, maxhp, fraction):
    # A neutral scaffold with the reference max HP; no ability/item modifiers.
    base = jnp.full(6, 100, jnp.int32).at[0].set((maxhp - 110) // 2)
    ivs = jnp.zeros(6, jnp.int32).at[0].set((maxhp - 110) % 2)
    stats = StatTable(base_stats=base, ivs=ivs, evs=jnp.zeros(6, jnp.int32))
    ints = jnp.ones((4, 1), jnp.int32)
    moves = Move(move_type=ints*0, max_pp=ints, current_pp=ints, type=ints,
                 base_power=ints, accuracy=ints, priority=ints*0,
                 offensive_stat=ints, defensive_stat=ints*2, crit_stage=ints*0)
    mon = Pokemon(type_list=jnp.array([0, 0]), moves=moves, stat_table=stats, current_hp=hp)
    side = stack([mon] * 6)
    state = BattleState(team=stack([side, side]))
    return take_damage_percent(state, 0, fraction).active.current_hp[0, 0]


def test_residual_reference_coverage():
    assert RESIDUALS, 'The capture must contain usable residual-damage events'


@pytest.mark.parametrize('case', RESIDUALS, ids=lambda c: c[0])
def test_residual_damage_rounding(case):
    _, hp, maxhp, fraction, expected = case
    assert int(apply_fraction(jnp.int32(hp), jnp.int32(maxhp), jnp.float32(fraction))) == expected
