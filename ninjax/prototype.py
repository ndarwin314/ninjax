"""Runnable, explicit neutral-ability/item sandbox based on captured Showdown teams.

python -m ninjax.prototype --reference unit_tests/fixtures/showdown/battle-1.json
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp

from ninjax.battle import Battle, BattleParams, Phase
from ninjax.enum_types import MoveType, Type, StatEnum, Status, WeatherEnum, TerrainEnum, MoveFlags
from ninjax.move import Move
from ninjax.pokemon import Pokemon
from ninjax.stats import Nature, StatTable
from ninjax.tree import stack

CATALOGUE = json.loads((Path(__file__).parent / 'data' / 'prototype-moves.json').read_text())
STAT_NAMES = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


def make_move(name):
    """Create a supported sandbox move or an unavailable placeholder."""
    data = CATALOGUE.get(name)
    supported = data is not None
    data = data or dict(type='Normal', category='Status', power=0, accuracy=1., priority=0, pp=0, num=0, contact=False)
    category = MoveType[data['category'].upper()]
    physical = category == MoveType.PHYSICAL
    effects = {}
    if name in ('Bullet Seed', 'Scale Shot', 'Tail Slap', 'Rock Blast'):
        effects['hit_rule'] = jnp.array([2])
    if name in ('Triple Axel', 'Surging Strikes'):
        effects['hit_rule'] = jnp.array([3])
    if name == 'Population Bomb':
        effects['hit_rule'] = jnp.array([4])
    if name in ('Triple Axel', 'Population Bomb'):
        effects['per_hit_accuracy'] = jnp.array([True])
    if name == 'Triple Axel':
        effects['escalating_power'] = jnp.array([True])
    if name == 'Surging Strikes':
        effects['always_crit'] = jnp.array([True])
    if name in ('U-turn', 'Volt Switch'):
        effects['switch_out'] = jnp.array([True])
    if name in ('Swords Dance', 'Nasty Plot'):
        effects.update(target_self=jnp.array([True]), boost_stat=jnp.array([StatEnum.ATTACK if name == 'Swords Dance' else StatEnum.SPECIAL_ATTACK]), boost_stages=jnp.array([2]))
    statuses = {'Will-O-Wisp': Status.BURN, 'Toxic': Status.TOXIC, 'Thunder Wave': Status.PARALYZE}
    if name in statuses:
        effects['status_effect'] = jnp.array([statuses[name]])
    hazards = {'Stealth Rock': 1, 'Sticky Web': 2, 'Spikes': 3, 'Toxic Spikes': 4}
    if name in hazards:
        effects['hazard_effect'] = jnp.array([hazards[name]])
    if name in ('Rain Dance', 'Sunny Day'):
        effects.update(target_self=jnp.array([True]), weather_effect=jnp.array([WeatherEnum.RAIN if name == 'Rain Dance' else WeatherEnum.SUN]))
    if name == 'Grassy Terrain':
        effects.update(target_self=jnp.array([True]), terrain_effect=jnp.array([TerrainEnum.GRASSY]))
    if name == 'Recover':
        effects.update(target_self=jnp.array([True]), healing_fraction=jnp.array([.5]))
    if name == 'Protect':
        effects['target_self'] = jnp.array([True])
    pp = data['pp'] * 8 // 5
    flags = (int(MoveFlags.CONTACT) if data['contact'] else 0) | (int(MoveFlags.BULLET) if name in ('Bullet Seed', 'Rock Blast') else 0) | (int(MoveFlags.PUNCHING) if name == 'Surging Strikes' else 0)
    return Move(move_type=jnp.array([int(category)]), max_pp=jnp.array([pp]), current_pp=jnp.array([pp]),
                type=jnp.array([int(Type[data['type'].upper()])]), base_power=jnp.array([data['power']]),
                accuracy=jnp.array([data['accuracy']]), priority=jnp.array([data['priority']]),
                offensive_stat=jnp.array([1 if physical else 3]), defensive_stat=jnp.array([2 if physical else 4]),
                crit_stage=jnp.array([0]), identifier=jnp.array([data['num']]),
                move_flags=jnp.array([flags]),
                implemented=jnp.array([supported]), **effects)


def make_sandbox_teams(reference):
    """Preserve reference stats/types. Explicitly omit unsupported moves/items/abilities."""
    sides = []
    omissions = []
    species_ids = {}
    for side in reference['initial']:
        mons = []
        for record in side:
            config = record['set']
            species = config['species']
            species_ids.setdefault(species, len(species_ids) + 1)
            plus = record['nature']['plus']
            minus = record['nature']['minus']
            stats = StatTable(level=jnp.array([config['level']]),
                base_stats=jnp.array([record['baseStats'][s] for s in STAT_NAMES]),
                ivs=jnp.array([config.get('ivs', {}).get(s, 31) for s in STAT_NAMES]),
                evs=jnp.array([config.get('evs', {}).get(s, 0) for s in STAT_NAMES]),
                nature=Nature(increased=jnp.array([STAT_NAMES.index(plus) if plus else 1]),
                              decreased=jnp.array([STAT_NAMES.index(minus) if minus else 1])))
            names = config['moves']
            # Captures use Showdown IDs; catalogue keys are display names.
            by_id = {data['id']: name for name, data in CATALOGUE.items()}
            display = [by_id.get(n, n) for n in names]
            missing = [n for n in display if n not in CATALOGUE]
            omissions.append({'species': species, 'moves': missing, 'ability': config.get('ability'), 'item': config.get('item')})
            types = [int(Type[t.upper()]) for t in record['types']]
            mons.append(Pokemon(type_list=jnp.array(types + [0]*(2-len(types))), moves=stack([make_move(n) for n in display]),
                                stat_table=stats, species=jnp.array([species_ids[species]])))
        sides.append(stack(mons))
    return stack(sides), omissions


@jax.jit
def random_actions(key, mask):
    """Uniform within move/switch classes, using the reference 90/10 policy."""
    roll_key, move_key, switch_key = jax.random.split(key, 3)
    moves = mask[:, :5]
    switches = mask[:, 8:14]
    move = jax.random.categorical(move_key, jnp.where(moves, 0., -jnp.inf), axis=-1)
    switch = 8 + jax.random.categorical(switch_key, jnp.where(switches, 0., -jnp.inf), axis=-1)
    choose_switch = jnp.any(switches, axis=1) & ((jax.random.uniform(roll_key, (2,)) < .1) | ~jnp.any(moves, axis=1))
    choice = jnp.where(choose_switch, switch, move).astype(jnp.int32)
    return jnp.where(mask[:, 14], 14, choice)


def run(reference_path, seed=0, max_decisions=1000):
    reference = json.loads(Path(reference_path).read_text())
    teams, omissions = make_sandbox_teams(reference)
    result = _run_teams(teams, omissions, seed, max_decisions)
    return {'reference_revision': reference['revision'], **result}


@lru_cache(maxsize=4)
def _sandbox_step(max_turns):
    # Teams live in the dynamic state, so different samples share compilation.
    env = Battle(params=BattleParams(max_steps_in_episode=max_turns))
    return jax.jit(env.step)


def _run_teams(teams, omissions, seed, max_decisions, max_turns=100):
    env = Battle(teams, BattleParams(max_steps_in_episode=max_turns))
    key, battle_key = jax.random.split(jax.random.key(seed))
    state, timestep = env.reset(battle_key)
    step = _sandbox_step(max_turns)
    decisions = []
    for _ in range(max_decisions):
        if bool(timestep.last()):
            break
        key, policy_key = jax.random.split(key)
        actions = random_actions(policy_key, timestep.extras['legal_action_mask'])
        state, timestep = step(state, actions)
        if bool(timestep.extras['invalid_action']):
            raise RuntimeError('Policy selected an invalid action')
        decisions.append({'turn': int(state.turn), 'phase': int(state.phase), 'actions': actions.tolist(),
                          'active': state.battle.active_index[:, 0].tolist(),
                          'hp': state.battle.active.current_hp[:, 0].tolist()})
    if not bool(timestep.last()):
        raise RuntimeError('Decision cap reached without termination')
    return {'seed': seed, 'sandbox': 'neutral abilities/items; unsupported moves unavailable',
            'omissions': omissions, 'turns': int(state.turn), 'reward': timestep.reward.tolist(), 'decisions': decisions}



def run_curated(path=None, seed=0, max_decisions=1000, max_turns=100, abilities=False, items=False):
    """Generate and run two teams from curated exports, reporting all omissions."""
    from ninjax.curated import DEFAULT_SETS, load_sets, make_sandbox_pool, make_ability_pool, make_item_pool, sample_teams
    path = Path(path) if path is not None else DEFAULT_SETS
    sets = load_sets(path)
    pool, reports = make_item_pool(sets, abilities=abilities) if items else (make_ability_pool if abilities else make_sandbox_pool)(sets)
    teams, indices = sample_teams(jax.random.fold_in(jax.random.key(seed), 0), pool)
    chosen = indices.tolist()
    omissions = [reports[i] for side in chosen for i in side]
    result = _run_teams(teams, omissions, seed, max_decisions, max_turns)
    if abilities:
        result['sandbox'] = 'verified curated abilities enabled; neutral items; unsupported moves/abilities omitted'
    if items:
        result['sandbox'] = ('represented curated items enabled; ' +
            ('represented abilities enabled' if abilities else 'neutral abilities') +
            '; unsupported effects omitted')
    return {**result, 'curated_source': str(path), 'set_indices': chosen,
            'teams': [[sets[i].species for i in side] for side in chosen]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--reference')
    source.add_argument('--curated', nargs='?', const=str(Path(__file__).parent / 'data' / 'curated-sets.txt'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--abilities', action='store_true', help='Enable verified abilities with --curated')
    parser.add_argument('--items', action='store_true', help='Enable represented items with --curated')
    args = parser.parse_args()
    if args.abilities and not args.curated:
        parser.error('--abilities requires --curated')
    if args.items and not args.curated:
        parser.error('--items requires --curated')
    result = run_curated(args.curated, args.seed, abilities=args.abilities, items=args.items) if args.curated else run(args.reference, args.seed)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('decisions', 'omissions')}))
    print(f"{len(result['decisions'])} decisions; omissions are included in --output")


if __name__ == '__main__':
    main()
