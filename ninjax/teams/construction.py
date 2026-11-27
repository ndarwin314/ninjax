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
from ninjax.enums.moves import EffectivenessRule, HitRule, MoveFlags, MoveType, PowerRule, SecondaryEffectType
from ninjax.enums.pokemon import FormEnum, StatEnum, Status, Type
from ninjax.moves.model import Move, MoveSecondaries, SECONDARY_SLOTS, STAT_CHANGE_SLOTS
from ninjax.moves.support import is_move_supported
from ninjax.pokemon import Pokemon
from ninjax.tree import stack


CATALOGUE = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'moves.json').read_text())
STAT_NAMES = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')

# Only formula selection belongs in metadata; live inputs come from battle state.
POWER_RULES = {
    'Acrobatics': PowerRule.DOUBLE_WITHOUT_ITEM,
    'Stomping Tantrum': PowerRule.DOUBLE_AFTER_FAILURE,
    'Stored Power': PowerRule.POSITIVE_STAGES, 'Power Trip': PowerRule.POSITIVE_STAGES,
    'Low Kick': PowerRule.TARGET_WEIGHT, 'Grass Knot': PowerRule.TARGET_WEIGHT,
    'Heavy Slam': PowerRule.WEIGHT_RATIO, 'Heat Crash': PowerRule.WEIGHT_RATIO,
    'Weather Ball': PowerRule.DOUBLE_IN_WEATHER,
    'Knock Off': PowerRule.REMOVABLE_ITEM_BOOST,
    'Rising Voltage': PowerRule.DOUBLE_ON_GROUNDED_ELECTRIC_TARGET,
    'Hex': PowerRule.TARGET_STATUS, 'Infernal Parade': PowerRule.TARGET_STATUS,
    'Eruption': PowerRule.USER_HP, 'Water Spout': PowerRule.USER_HP, 'Dragon Energy': PowerRule.USER_HP,
    'Crush Grip': PowerRule.TARGET_HP_120, 'Hard Press': PowerRule.TARGET_HP_100,
    'Electro Ball': PowerRule.ELECTRO_BALL, 'Gyro Ball': PowerRule.GYRO_BALL,
    'Facade': PowerRule.USER_STATUS_MODIFIER,
    'Venoshock': PowerRule.TARGET_POISON_MODIFIER, 'Barb Barrage': PowerRule.TARGET_POISON_MODIFIER,
    'Brine': PowerRule.TARGET_HALF_HP_MODIFIER,
    'Psyblade': PowerRule.ELECTRIC_TERRAIN_MODIFIER,
    'Collision Course': PowerRule.SUPER_EFFECTIVE_MODIFIER, 'Electro Drift': PowerRule.SUPER_EFFECTIVE_MODIFIER,
    'Terrain Pulse': PowerRule.DOUBLE_IN_TERRAIN
}

# Showdown's Terastallized form names use the same Ivy Cudgel typing.
FORM_IDS = {name + suffix: value for name, value in {
    'ogerponwellspring': FormEnum.OGERPON_WELLSPRING,
    'ogerponhearthflame': FormEnum.OGERPON_HEARTHFLAME,
    'ogerponcornerstone': FormEnum.OGERPON_CORNERSTONE,
}.items() for suffix in ('', 'tera')}

FORM_IDS.update({
    'taurospaldeacombat': FormEnum.TAUROS_PALDEA_COMBAT,
    'taurospaldeablaze': FormEnum.TAUROS_PALDEA_BLAZE,
    'taurospaldeaaqua': FormEnum.TAUROS_PALDEA_AQUA,
    'morpekohangry': FormEnum.MORPEKO_HANGRY,
})


def _pack_secondary_boosts(boosts, stat_ids):
    """Pad each slot's stat indices and deltas to the same fixed width."""
    stats = [[stat_ids[stat] for stat in row] + [1] * (STAT_CHANGE_SLOTS - len(row)) for row in boosts]
    stages = [list(row.values()) + [0] * (STAT_CHANGE_SLOTS - len(row)) for row in boosts]
    return jnp.array(stats, jnp.int32), jnp.array(stages, jnp.int32)


def _project_secondaries(name, entries, status_ids, stat_ids):
    """Pack independent effects or mutually exclusive outcomes into fixed slots."""
    outcomes = {
        'Dire Claw': ('psn', 'par', 'slp'),
        'Tri Attack': ('brn', 'par', 'frz'),
    }.get(name)
    mode = SecondaryEffectType.DEFAULT
    if outcomes:
        mode = SecondaryEffectType.EXCLUSIVE
        chance = entries[0]['chance'] / len(outcomes)
        entries = [{'status': status, 'chance': chance} for status in outcomes]
    if len(entries) > SECONDARY_SLOTS:
        raise ValueError(f'{name} has more than {SECONDARY_SLOTS} secondary effects')
    slots = entries + [{}] * (SECONDARY_SLOTS - len(entries))
    stats, stages = _pack_secondary_boosts([slot.get('boosts', {}) for slot in slots], stat_ids)
    user_stats, user_stages = _pack_secondary_boosts(
        [(slot.get('self') or {}).get('boosts', {}) for slot in slots], stat_ids)
    return MoveSecondaries(
        mode=jnp.array([mode], jnp.int32),
        probabilities=jnp.array([slot.get('chance', 0) / 100 for slot in slots], jnp.float32),
        status=jnp.array([status_ids.get(slot.get('status'), Status.NONE) for slot in slots], jnp.int32),
        stats=stats, stages=stages, user_stats=user_stats, user_stages=user_stages,
        flinch=jnp.array([slot.get('volatileStatus') == 'flinch' for slot in slots]),
        confusion=jnp.array([slot.get('volatileStatus') == 'confusion' for slot in slots]),
        restriction=jnp.array([i == 0 and name in ('Throat Chop', 'Psychic Noise')
                               for i in range(SECONDARY_SLOTS)]))


def make_move(name):
    """Project metadata, enabling PP only for reviewed singles implementations.

    Known unsupported moves retain their identity and metadata for observations
    and diagnostics, but have zero PP and ``implemented=False``.
    """
    data = CATALOGUE.get(name)
    supported = data is not None and is_move_supported(name)
    data = data or dict(type='Normal', category='Status', power=0, accuracy=1., priority=0, pp=0, num=0, contact=False)
    category = MoveType[data['category'].upper()]
    physical = int(category) == int(MoveType.PHYSICAL)
    effectiveness_rules = {'Freeze-Dry': EffectivenessRule.FREEZE_DRY,
                           'Flying Press': EffectivenessRule.FLYING_PRESS}
    effects = {'power_rule': jnp.array([POWER_RULES.get(name, PowerRule.FIXED)]),
               'effectiveness_rule': jnp.array([effectiveness_rules.get(name, EffectivenessRule.DEFAULT)])}
    # Normalize Showdown condition IDs on the host, never inside traced code.
    weather_ids = {'raindance': WeatherEnum.RAIN, 'sunnyday': WeatherEnum.SUN,
                   'sandstorm': WeatherEnum.SANDSTORM, 'snowscape': WeatherEnum.SNOW}
    terrain_ids = {'electricterrain': TerrainEnum.ELECTRIC, 'grassyterrain': TerrainEnum.GRASSY,
                   'psychicterrain': TerrainEnum.PSYCHIC, 'mistyterrain': TerrainEnum.MISTY}
    if data.get('weather'):
        effects['weather_effect'] = jnp.array([weather_ids[data['weather'].lower()]])
    if data.get('terrain'):
        effects['terrain_effect'] = jnp.array([terrain_ids[data['terrain'].lower()]])
    if data.get('target') in ('self', 'allySide', 'all') and category == MoveType.STATUS:
        effects['target_self'] = jnp.array([True])
    effects['primary_confusion'] = jnp.array([data.get('volatileStatus') == 'confusion'])
    stat_ids = {'atk': 1, 'def': 2, 'spa': 3, 'spd': 4, 'spe': 5, 'accuracy': 6, 'evasion': 7}
    status_ids = {'brn': Status.BURN, 'psn': Status.POISON, 'tox': Status.TOXIC,
                  'par': Status.PARALYZE, 'slp': Status.SLEEP, 'frz': Status.FREEZE}
    for source, destination in (('overrideOffensiveStat', 'offensive_stat'),
                                ('overrideDefensiveStat', 'defensive_stat')):
        if data.get(source):
            effects[destination] = jnp.array([stat_ids[data[source]]])
    # Move metadata is projected on the host; traced transitions only consume
    # arrays. Stat-index and stage vectors are padded by Move on construction.
    secondary = data.get('secondary') or {}
    secondaries = data.get('secondaries') or ([secondary] if secondary else [])
    self_effect = data.get('self') or {}
    for prefix, boosts in [('boost', data.get('boosts', {})),
                           ('self', self_effect.get('boosts', {}))]:
        if boosts:
            effects[f'{prefix}_stat'] = jnp.array([stat_ids[stat] for stat in boosts])
            effects[f'{prefix}_stages'] = jnp.array(list(boosts.values()))
    if data.get('boosts'):
        # Top-level boosts apply to the move's target, which may be the user.
        effects['target_self'] = jnp.array([data.get('target') == 'self'])
    effects['self_chance'] = jnp.array([self_effect.get('chance', 100) / 100])
    if data.get('status'):
        effects['status_effect'] = jnp.array([status_ids[data['status']]])
    if data.get('heal'):
        effects.update(target_self=jnp.array([True]),
                       healing_fraction=jnp.array([data['heal'][0] / data['heal'][1]]))
    effects['secondaries'] = _project_secondaries(name, secondaries, status_ids, stat_ids)
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
    # Catalogue multi-hit counts describe fixed hits or the standard 2–5 distribution.
    multihit = data.get('multihit')
    if multihit == [2, 5]:
        effects['hit_rule'] = jnp.array([HitRule.TWO_TO_FIVE])
    elif multihit == 2:
        effects['hit_rule'] = jnp.array([HitRule.FIXED_TWO])
    elif multihit == 3:
        effects['hit_rule'] = jnp.array([HitRule.FIXED_THREE])
    elif name == 'Population Bomb':
        effects['hit_rule'] = jnp.array([HitRule.POPULATION_BOMB])
    if name in ('Triple Axel', 'Triple Kick', 'Population Bomb'):
        effects['per_hit_accuracy'] = jnp.array([True])
    if name in ('Triple Axel', 'Triple Kick'):
        effects['escalating_power'] = jnp.array([True])
    if name == 'Surging Strikes':
        effects['always_crit'] = jnp.array([True])
    if data.get('selfSwitch') and name != 'Revival Blessing':
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
    pp = data['pp'] * 8 // 5 if supported else 0
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
    # Target thawing is independent of the defrost flag for the user.
    if data.get('thawsTarget'):
        flags |= int(MoveFlags.THAWS_TARGET)
    if data.get('ignoreAbility'):
        flags |= int(MoveFlags.IGNORE_ABILITY)
    if data.get('volatileStatus') == 'partiallytrapped':
        flags |= int(MoveFlags.BINDING)
    if self_effect.get('volatileStatus') == 'mustrecharge' or data.get('flags', {}).get('recharge'):
        flags |= int(MoveFlags.RECHARGE)
    if data.get('flags', {}).get('cantusetwice'):
        flags |= int(MoveFlags.CANNOT_REPEAT)
    if data.get('target') in ('allAdjacent', 'allAdjacentFoes'):
        flags |= int(MoveFlags.SPREAD)
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
    """Project reference teams with Pokédex species IDs and explicit omissions.

    Unknown species use ID zero; named battle mechanics rely on stable IDs.
    """
    sides = []
    omissions = []
    for side in reference['initial']:
        mons = []
        for record in side:
            config = record['set']
            species = config['species']
            species_key = ''.join(c for c in species.lower() if c.isalnum())
            species_data = dex.get(species_key, {})
            weight_hg = record.get('weight_hg', round(species_data.get('weightkg', 10) * 10))
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
            missing = [n for n in display if not is_move_supported(n)]
            omissions.append({'species': species, 'moves': missing, 'ability': config.get('ability'), 'item': config.get('item')})
            types = [int(Type[t.upper()]) for t in record['types']]
            mons.append(Pokemon(type_list=jnp.array(types + [0]*(2-len(types))), moves=stack([make_move(n) for n in display]),
                                stat_table=stats, species=jnp.array([species_data.get('num', 0)]), form=jnp.array([int(FORM_IDS.get(species_key, FormEnum.DEFAULT))]),
                                weight_hg=jnp.array([weight_hg])))
        sides.append(stack(mons))
    return stack(sides), omissions
