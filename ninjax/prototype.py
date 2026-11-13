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
from ninjax.enum_types import MoveType, Type, StatEnum, Status, WeatherEnum, TerrainEnum, MoveFlags, FormEnum
from ninjax.move import Move
from ninjax.pokemon import Pokemon
from ninjax.stats import Nature, StatTable
from ninjax.tree import stack
from ninjax.data.load_json import dex

CATALOGUE = json.loads((Path(__file__).parent / 'data' / 'prototype-moves.json').read_text())
STAT_NAMES = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')

# Showdown's Terastallized form names use the same Ivy Cudgel typing.
FORM_IDS = {name + suffix: value for name, value in {
    'ogerponwellspring': FormEnum.OGERPON_WELLSPRING,
    'ogerponhearthflame': FormEnum.OGERPON_HEARTHFLAME,
    'ogerponcornerstone': FormEnum.OGERPON_CORNERSTONE,
}.items() for suffix in ('', 'tera')}


def make_move(name):
    """Create a supported sandbox move or an unavailable placeholder."""
    data = CATALOGUE.get(name)
    supported = data is not None
    data = data or dict(type='Normal', category='Status', power=0, accuracy=1., priority=0, pp=0, num=0, contact=False)
    category = MoveType[data['category'].upper()]
    physical = int(category) == int(MoveType.PHYSICAL)
    effects = {}
    stat_ids = {'atk': 1, 'def': 2, 'spa': 3, 'spd': 4, 'spe': 5, 'accuracy': 6, 'evasion': 7}
    status_ids = {'brn': Status.BURN, 'psn': Status.POISON, 'slp': Status.SLEEP, 'frz': Status.FREEZE}
    # Move metadata is projected on the host; traced transitions only consume
    # arrays. Stat-index and stage vectors are padded by Move on construction.
    for prefix, boosts in [('boost', data.get('boosts', {})),
                           ('self', data.get('self', {}).get('boosts', {}))]:
        if boosts:
            effects[f'{prefix}_stat'] = jnp.array([stat_ids[stat] for stat in boosts])
            effects[f'{prefix}_stages'] = jnp.array(list(boosts.values()))
    if data.get('boosts'):
        effects['target_self'] = jnp.array([True])
    secondary = data.get('secondary') or {}
    effects['secondary_chance'] = jnp.array([secondary.get('chance', 0) / 100])
    effects['secondary_status'] = jnp.array([status_ids.get(secondary.get('status'), 0)])
    effects['secondary_flinch'] = jnp.array([secondary.get('volatileStatus') == 'flinch'])
    effects['secondary_confusion'] = jnp.array([secondary.get('volatileStatus') == 'confusion'])
    if secondary.get('boosts'):
        effects['secondary_stat'] = jnp.array([stat_ids[stat] for stat in secondary['boosts']])
        effects['secondary_stages'] = jnp.array(list(secondary['boosts'].values()))
    if data.get('drain'):
        effects['drain_fraction'] = jnp.array([data['drain'][0] / data['drain'][1]])
    if data.get('recoil'):
        effects['recoil_percent'] = jnp.array([data['recoil'][0] / data['recoil'][1]])
    effects['ignore_defensive'] = jnp.array([data.get('ignoreDefensive', False)])
    effects['always_crit'] = jnp.array([data.get('willCrit', False)])
    effects['always_hit'] = jnp.array([name in ('Flower Trick', 'Clear Smog', 'Pain Split', 'Whirlwind')])
    if name == 'Body Press':
        effects['offensive_stat'] = jnp.array([2])
    if name == 'Psyshock':
        effects['defensive_stat'] = jnp.array([2])
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
    statuses = {'Will-O-Wisp': Status.BURN, 'Toxic': Status.TOXIC, 'Thunder Wave': Status.PARALYZE, 'Spore': Status.SLEEP}
    if name in statuses:
        effects['status_effect'] = jnp.array([statuses[name]])
    hazards = {'Stealth Rock': 1, 'Sticky Web': 2, 'Spikes': 3, 'Toxic Spikes': 4}
    if name in hazards:
        effects['hazard_effect'] = jnp.array([hazards[name]])
    if name in ('Rain Dance', 'Sunny Day'):
        effects.update(target_self=jnp.array([True]), weather_effect=jnp.array([WeatherEnum.RAIN if name == 'Rain Dance' else WeatherEnum.SUN]))
    if name == 'Grassy Terrain':
        effects.update(target_self=jnp.array([True]), terrain_effect=jnp.array([TerrainEnum.GRASSY]))
    if name in ('Recover', 'Roost'):
        effects.update(target_self=jnp.array([True]), healing_fraction=jnp.array([.5]))
    if name in ('Protect', 'Spiky Shield', 'Rest', 'Sleep Talk', 'Substitute', 'Destiny Bond'):
        effects['target_self'] = jnp.array([True])
    if name in ('Light Screen', 'Aurora Veil', 'Tailwind', 'Trick Room', 'Haze', 'Court Change', 'Perish Song'):
        effects['target_self'] = jnp.array([True])
    if name == 'Perish Song':
        effects['perish_effect'] = jnp.array([True])
    if name == 'Defog':
        effects['always_hit'] = jnp.array([True])
    pp = data['pp'] * 8 // 5
    flags = (int(MoveFlags.CONTACT) if data['contact'] else 0) | (int(MoveFlags.BULLET) if name in ('Bullet Seed', 'Rock Blast') else 0) | (int(MoveFlags.PUNCHING) if name == 'Surging Strikes' else 0)
    for source, flag in [('sound', MoveFlags.SOUND), ('bullet', MoveFlags.BULLET),
                         ('pulse', MoveFlags.LAUNCHER), ('bite', MoveFlags.BITING),
                         ('heal', MoveFlags.HEALING), ('powder', MoveFlags.POWDER), ('defrost', MoveFlags.DEFROST),
                         ('bypasssub', MoveFlags.BYPASS_SUBSTITUTE)]:
        if data.get('flags', {}).get(source):
            flags |= int(flag)
    if 'flags' in data:
        effects['protectable'] = jnp.array([bool(data['flags'].get('protect'))])
    effects['encore_eligible'] = jnp.array([not bool(data.get('flags', {}).get('failencore'))])
    effects['sleep_talk_eligible'] = jnp.array([not bool(data.get('flags', {}).get('nosleeptalk') or data.get('flags', {}).get('charge'))])
    offensive = effects.pop('offensive_stat', jnp.array([1 if physical else 3]))
    defensive = effects.pop('defensive_stat', jnp.array([2 if physical else 4]))
    return Move(move_type=jnp.array([int(category)]), max_pp=jnp.array([pp]), current_pp=jnp.array([pp]),
                type=jnp.array([int(Type[data['type'].upper()])]), base_power=jnp.array([data['power']]),
                accuracy=jnp.array([data['accuracy']]), priority=jnp.array([data['priority']]),
                offensive_stat=offensive, defensive_stat=defensive,
                crit_stage=jnp.array([1 if name == 'Ivy Cudgel' else 0]), identifier=jnp.array([data['num']]),
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
            species_key = ''.join(c for c in species.lower() if c.isalnum())
            species_data = dex.get(species_key, {})
            weight_hg = record.get('weight_hg', round(species_data.get('weightkg', 10) * 10))
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
                                stat_table=stats, species=jnp.array([species_ids[species]]), form=jnp.array([int(FORM_IDS.get(species_key, FormEnum.DEFAULT))]),
                                weight_hg=jnp.array([weight_hg])))
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
