"""Host-side move and team construction for caller-provided battle teams.

Catalogue metadata is loaded as Python data. Constructors project reference
records into immutable Pokémon PyTrees and report omitted mechanics explicitly.
This module does not import the battle engine or prototype runner.
"""
import json
from pathlib import Path

import jax.numpy as jnp

from ninjax.calculations.stats import Nature, StatTable
from ninjax.data.load_json import dex
from ninjax.enums.field import TerrainEnum, WeatherEnum
from ninjax.enums.moves import MoveFlags, MoveType
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.moves.model import Move
from ninjax.pokemon import Pokemon
from ninjax.tree import stack


CATALOGUE = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'prototype-moves.json').read_text())
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
    status_ids = {'brn': Status.BURN, 'psn': Status.POISON, 'tox': Status.TOXIC,
                  'par': Status.PARALYZE, 'slp': Status.SLEEP, 'frz': Status.FREEZE}
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
    effects['self_chance'] = jnp.array([data.get('self', {}).get('chance', 100) / 100])
    if data.get('status'):
        effects['status_effect'] = jnp.array([status_ids[data['status']]])
    if data.get('heal'):
        effects.update(target_self=jnp.array([True]),
                       healing_fraction=jnp.array([data['heal'][0] / data['heal'][1]]))
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
    effects['always_hit'] = jnp.array([data.get('alwaysHit', False) or name in ('Flower Trick', 'Clear Smog', 'Pain Split', 'Whirlwind')])
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
                         ('bypasssub', MoveFlags.BYPASS_SUBSTITUTE), ('slicing', MoveFlags.SLICING),
                         ('punch', MoveFlags.PUNCHING), ('wind', MoveFlags.WIND)]:
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
                crit_stage=jnp.array([data.get('critRatio', 2 if name == 'Ivy Cudgel' else 1) - 1]), identifier=jnp.array([data['num']]),
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
