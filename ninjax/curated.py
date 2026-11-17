"""Parse curated Showdown exports and sample explicit sandbox teams.

Parsing/construction runs on the host. Sampling uses an explicit JAX key and
supports jit/vmap. No format legality checks are performed.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from ninjax.data.load_json import dex
from ninjax.enum_types import AbilityEnum, ItemEnum
from ninjax.team_builder import STAT_NAMES, make_sandbox_teams


DEFAULT_SETS = Path(__file__).parent / 'data' / 'curated-sets.txt'
STAT_LABELS = dict(zip(('HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'), STAT_NAMES))
_ORDER = ('atk', 'def', 'spe', 'spa', 'spd')
NATURES = {name: (_ORDER[i], _ORDER[j]) for i, row in enumerate((
    'Hardy Lonely Brave Adamant Naughty', 'Bold Docile Relaxed Impish Lax',
    'Timid Hasty Serious Jolly Naive', 'Modest Mild Quiet Bashful Rash',
    'Calm Gentle Sassy Careful Quirky')) for j, name in enumerate(row.split())}


@dataclass(frozen=True)
class CuratedSet:
    species: str
    item: str
    ability: str
    nature: str
    moves: tuple[str, ...]
    evs: tuple[int, ...]
    ivs: tuple[int, ...]
    level: int = 100
    tera_type: str | None = None
    gender: str | None = None


def parse_sets(text: str) -> tuple[CuratedSet, ...]:
    """Read exports; omitted EVs=0, IVs=31, level=100, nature=Serious.

    Unknown fields and malformed blocks raise errors rather than silently losing
    settings. Tera Type is retained as metadata for future mechanics support.
    """
    result = []
    for number, block in enumerate(re.split(r'\n\s*\n', text.strip()), 1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        try:
            species, separator, item = lines[0].partition(' @ ')
            if not species or (separator and not item):
                raise ValueError('expected a species and, if specified, an item')
            match = re.fullmatch(r'(.*?) \(([MF])\)', species)
            gender = match.group(2) if match else None
            species = match.group(1) if match else species
            fields = dict(species=species, item=item, level=100, tera_type=None, gender=gender, nature='Serious')
            stats = dict(EVs={s: 0 for s in STAT_NAMES}, IVs={s: 31 for s in STAT_NAMES})
            moves, seen = [], set()
            for line in lines[1:]:
                if line.startswith('- '):
                    moves.append(line[2:])
                    continue
                if line.endswith(' Nature'):
                    label, value = 'nature', line[:-7]
                    if value not in NATURES:
                        raise ValueError(f'unknown nature {value!r}')
                else:
                    label, sep, value = line.partition(': ')
                    if not sep or label not in ('Ability', 'Level', 'EVs', 'IVs', 'Tera Type'):
                        raise ValueError(f'unrecognized field {line!r}')
                if label in seen:
                    raise ValueError(f'duplicate field {label}')
                seen.add(label)
                if label in stats:
                    used = set()
                    for entry in value.split(' / '):
                        amount, name = entry.split()
                        stat = STAT_LABELS[name]
                        if stat in used:
                            raise ValueError(f'duplicate stat {name}')
                        stats[label][stat] = int(amount)
                        used.add(stat)
                else:
                    key = {'Ability': 'ability', 'Level': 'level', 'Tera Type': 'tera_type'}.get(label, label)
                    fields[key] = int(value) if label == 'Level' else value
            if len(moves) != 4 or 'ability' not in fields:
                raise ValueError('each set needs an ability and four moves')
            result.append(CuratedSet(**fields, moves=tuple(moves),
                evs=tuple(stats['EVs'][s] for s in STAT_NAMES),
                ivs=tuple(stats['IVs'][s] for s in STAT_NAMES)))
        except (ValueError, KeyError) as error:
            raise ValueError(f'Set {number} ({lines[0]}): {error}') from error
    if not result:
        raise ValueError('No curated sets found')
    return tuple(result)


def load_sets(path=DEFAULT_SETS):
    return parse_sets(Path(path).read_text())


def make_sandbox_pool(sets):
    """Construct a (sets, ...) Pokemon bank and a per-set omissions report.

    Preserve species, types, levels, EVs, IVs and natures. As in the existing
    prototype runner, abilities/items are neutral and unsupported moves have
    zero PP. Original set metadata remains available in `sets` and the report.
    """
    records = []
    for config in sets:
        species_id = re.sub('[^a-z0-9]', '', config.species.lower())
        if species_id not in dex:
            raise ValueError(f'Unknown species {config.species!r}')
        species = dex[species_id]
        plus, minus = NATURES[config.nature]
        records.append(dict(set=dict(species=config.species, level=config.level,
            ability=config.ability, item=config.item,
            moves=[re.sub('[^a-z0-9]', '', m.lower()) for m in config.moves],
            evs=dict(zip(STAT_NAMES, config.evs)), ivs=dict(zip(STAT_NAMES, config.ivs))),
            nature=dict(plus=plus, minus=minus), baseStats=species['baseStats'],
            types=species['types'], weight_hg=round(species['weightkg'] * 10)))
    if len(records) < 6:
        raise ValueError('At least six curated sets are required')
    # Reuse the prototype projection so omissions have identical semantics.
    teams, reports = make_sandbox_teams(dict(initial=[records, records]))
    bank = jax.tree.map(lambda x: x[0], teams)
    # Use stable dex numbers rather than selection-dependent local IDs.
    numbers = [dex[re.sub('[^a-z0-9]', '', c.species.lower())]['num'] for c in sets]
    bank = bank.replace(species=jnp.asarray(numbers, jnp.int32)[:, None])
    for config, report in zip(sets, reports[:len(sets)]):
        omitted = {re.sub('[^a-z0-9]', '', name.lower()) for name in report['moves']}
        report['moves'] = [name for name in config.moves
                           if re.sub('[^a-z0-9]', '', name.lower()) in omitted]
        report['tera_type'] = config.tera_type
        report['gender'] = config.gender
    return bank, reports[:len(sets)]


def sample_teams(key, pool):
    """Return (Pokemon teams, set indices), with shapes (2, 6, ...) and (2, 6).

    Each side samples six distinct set entries. Opponents may share entries.
    Selection does not enforce species clauses or other format restrictions.
    """
    size = pool.species.shape[0]
    if size < 6:
        raise ValueError('At least six curated sets are required')
    keys = jax.random.split(key, 2)
    indices = jax.vmap(lambda k: jax.random.permutation(k, size)[:6])(keys)
    return jax.tree.map(lambda x: x[indices], pool), indices


# Enable only the curated abilities covered by dedicated mechanics checks.
# Unsupported abilities remain neutral and visible in the omissions report.
CURATED_ABILITIES = {
    'Drizzle': 'DRIZZLE', 'Sand Stream': 'SAND_STREAM', 'Snow Warning': 'SNOW_WARNING',
    'Grassy Surge': 'GRASSY_SURGE', 'Psychic Surge': 'PSYCHIC_SURGE', 'Misty Surge': 'MISTY_SURGE',
    'Cloud Nine': 'CLOUD_NINE', 'Regenerator': 'REGENERATOR', 'Stamina': 'STAMINA',
    'Weak Armor': 'WEAK_ARMOR', 'Skill Link': 'SKILL_LINK', 'Technician': 'TECHNICIAN', 'Frisk': 'FRISK',
    'Orichalcum Pulse': 'ORICHALCUM_PULSE', 'Hadron Engine': 'HADRON_ENGINE',
    'Protosynthesis': 'PROTOSYNTHESIS', 'Quark Drive': 'QUARK_DRIVE',
    'Beads of Ruin': 'BEADS_OF_RUIN', 'Sword of Ruin': 'SWORD_OF_RUIN', 'Vessel of Ruin': 'VESSEL_OF_RUIN',
    'Libero': 'LIBERO', 'Protean': 'PROTEAN', 'Pixilate': 'PIXILATE',
    'Moody': 'MOODY', 'Sand Force': 'SAND_FORCE', 'Shadow Tag': 'SHADOW_TAG',
    'Storm Drain': 'STORM_DRAIN', 'Water Absorb': 'WATER_ABSORB',
    # Friend Guard only affects other active allies: it has no effect in singles.
    'Friend Guard': 'FRIEND_GUARD', 'Unseen Fist': 'UNSEEN_FIST',
    # Passive hooks audited against the pinned random-battle reference probes.
    'Adaptability': 'ADAPTABILITY', 'Blaze': 'BLAZE', 'Overgrow': 'OVERGROW',
    'Torrent': 'TORRENT', 'Swarm': 'SWARM', 'Huge Power': 'HUGE_POWER',
    'Pure Power': 'PURE_POWER', 'Guts': 'GUTS', 'Toxic Boost': 'TOXIC_BOOST',
    "Dragon's Maw": 'DRAGONS_MAW', 'Rocky Payload': 'ROCKY_PAYLOAD',
    'Transistor': 'TRANSISTOR', 'Strong Jaw': 'STRONG_JAW', 'Iron Fist': 'IRON_FIST',
    'Tough Claws': 'TOUGH_CLAWS', 'Mega Launcher': 'MEGA_LAUNCHER',
    'Punk Rock': 'PUNK_ROCK', 'Sharpness': 'SHARPNESS', 'Water Bubble': 'WATER_BUBBLE',
    'Tinted Lens': 'TINTED_LENS', 'Thick Fat': 'THICK_FAT', 'Heatproof': 'HEATPROOF',
    'Fur Coat': 'FUR_COAT', 'Ice Scales': 'ICE_SCALES', 'Purifying Salt': 'PURIFYING_SALT',
    'Filter': 'FILTER', 'Solid Rock': 'SOLID_ROCK', 'Prism Armor': 'PRISM_ARMOR',
    'Fluffy': 'FLUFFY', 'Multiscale': 'MULTISCALE', 'Shadow Shield': 'SHADOW_SHIELD',
    'Chlorophyll': 'CHLOROPHYLL', 'Swift Swim': 'SWIFT_SWIM', 'Slush Rush': 'SLUSH_RUSH',
    'Sand Rush': 'SAND_RUSH', 'Surge Surfer': 'SURGE_SURFER', 'Quick Feet': 'QUICK_FEET',
    'Limber': 'LIMBER', 'Water Veil': 'WATER_VEIL', 'Thermal Exchange': 'THERMAL_EXCHANGE',
    'Insomnia': 'INSOMNIA', 'Vital Spirit': 'VITAL_SPIRIT', 'Leaf Guard': 'LEAF_GUARD',
    'Poison Heal': 'POISON_HEAL', 'Natural Cure': 'NATURAL_CURE', 'Hydration': 'HYDRATION',
    'Shed Skin': 'SHED_SKIN', 'Speed Boost': 'SPEED_BOOST', 'Unnerve': 'UNNERVE',
    'Air Lock': 'AIR_LOCK', 'Drought': 'DROUGHT', 'Electric Surge': 'ELECTRIC_SURGE',
    'Volt Absorb': 'VOLT_ABSORB', 'Earth Eater': 'EARTH_EATER', 'Dry Skin': 'DRY_SKIN',
    'Lightning Rod': 'LIGHTNING_ROD', 'Motor Drive': 'MOTOR_DRIVE', 'Sap Sipper': 'SAP_SIPPER',
    'Well-Baked Body': 'WELL_BAKED_BODY', 'Queenly Majesty': 'QUEENLY_MAJESTY',
    'Levitate': 'LEVITATE', 'Soundproof': 'SOUNDPROOF', 'Bulletproof': 'BULLET_PROOF',
    'Overcoat': 'OVERCOAT', 'Rock Head': 'ROCK_HEAD', 'Shell Armor': 'SHELL_ARMOR',
    'Inner Focus': 'INNER_FOCUS', 'Shield Dust': 'SHIELD_DUST', 'Serene Grace': 'SERENE_GRACE',
    'No Guard': 'NO_GUARD', 'Compound Eyes': 'COMPOUND_EYES',
    'Clear Body': 'CLEAR_BODY', 'Full Metal Body': 'FULL_METAL_BODY', 'Contrary': 'CONTRARY',
    'Big Pecks': 'BIG_PECKS', 'Keen Eye': 'KEEN_EYE', 'Own Tempo': 'OWN_TEMPO',
    'Flash Fire': 'FLASH_FIRE', 'Damp': 'DAMP', 'Sticky Hold': 'STICKY_HOLD',
    'Heavy Metal': 'HEAVY_METAL', 'Light Metal': 'LIGHT_METAL', 'Ice Body': 'ICE_BODY',
    'Hustle': 'HUSTLE', 'Sniper': 'SNIPER', 'Reckless': 'RECKLESS',
    'Tablets of Ruin': 'TABLETS_OF_RUIN', 'Infiltrator': 'INFILTRATOR',
    'Good as Gold': 'GOOD_AS_GOLD', 'Galvanize': 'GALVANIZE', 'Liquid Voice': 'LIQUID_VOICE',
    'Prankster': 'PRANKSTER', 'Gale Wings': 'GALE_WINGS', 'Triage': 'TRIAGE',
    'Rough Skin': 'ROUGH_SKIN', 'Gooey': 'GOOEY',
    'Tangling Hair': 'TANGLING_HAIR', 'Static': 'STATIC', 'Flame Body': 'FLAME_BODY',
    'Effect Spore': 'EFFECT_SPORE', 'Poison Touch': 'POISON_TOUCH',
    'Early Bird': 'EARLY_BIRD', 'Download': 'DOWNLOAD', 'Intimidate': 'INTIMIDATE',
    'Water Compaction': 'WATER_COMPACTION', 'Justified': 'JUSTIFIED', 'Rattled': 'RATTLED',
    'Seed Sower': 'SEED_SOWER', 'Aroma Veil': 'AROMA_VEIL', 'Oblivious': 'OBLIVIOUS',
    'Liquid Ooze': 'LIQUID_OOZE',
    'Defiant': 'DEFIANT', 'Competitive': 'COMPETITIVE',
    # Like Friend Guard, Power Spot affects other allies and has no singles effect.
    'Power Spot': 'POWER_SPOT',
    'Sturdy': 'STURDY', 'Unaware': 'UNAWARE',
    'Cheek Pouch': 'CHEEK_POUCH', 'Aftermath': 'AFTERMATH', 'Bad Dreams': 'BAD_DREAMS',
    'Wind Rider': 'WIND_RIDER',
    'Arena Trap': 'ARENA_TRAP', 'Magnet Pull': 'MAGNET_PULL',
}


def make_ability_pool(sets):
    """Build the projected pool with verified abilities enabled and items neutral.

    A None ability in the omissions report means it was enabled. Ability IDs use
    the engine's enum. Unsupported moves and abilities are still reported.
    """

    pool, omissions = make_sandbox_pool(sets)
    abilities = []
    for config, report in zip(sets, omissions, strict=True):
        identifier = CURATED_ABILITIES.get(config.ability)
        abilities.append(int(AbilityEnum[identifier]) if identifier else 0)
        if identifier:
            report['ability'] = None
    return pool.replace(ability=jnp.asarray(abilities, jnp.int32)[:, None]), omissions


# Only items with existing engine hooks are projected into test battles.
CURATED_ITEMS = {
    '': 'NONE',
    'Life Orb': 'LIFE_ORB', 'Heavy-Duty Boots': 'HEAVY_DUTY_BOOTS',
    'Choice Specs': 'CHOICE_SPECS', 'Choice Band': 'CHOICE_BAND',
    'Assault Vest': 'ASSAULT_VEST', 'Weakness Policy': 'WEAKNESS_POLICY',
    'Loaded Dice': 'LOADED_DICE', 'Rocky Helmet': 'ROCKY_HELMET',
    'Leftovers': 'LEFTOVERS', 'Focus Sash': 'FOCUS_SASH',
    'Booster Energy': 'BOOSTER_ENERGY', 'Sitrus Berry': 'SITRUS_BERRY',
    'Throat Spray': 'THROAT_SPRAY',
    'Light Clay': 'LIGHT_CLAY',
    'Mental Herb': 'MENTAL_HERB',
    "King's Rock": 'KINGS_ROCK', 'Protective Pads': 'PROTECTIVE_PADS',
    'Wellspring Mask': 'WELLSPRING_MASK', 'Absorb Bulb': 'ABSORB_BULB',
    'Choice Scarf': 'CHOICE_SCARF', 'Wide Lens': 'WIDE_LENS', 'Expert Belt': 'EXPERT_BELT',
    'Light Ball': 'LIGHT_BALL', 'Magnet': 'MAGNET', 'Silk Scarf': 'SILK_SCARF',
    'Soul Dew': 'SOUL_DEW', 'Lustrous Orb': 'LUSTROUS_ORB',
    'Chesto Berry': 'CHESTO_BERRY', 'Lum Berry': 'LUM_BERRY',
    'Flame Orb': 'FLAME_ORB', 'Toxic Orb': 'TOXIC_ORB', 'White Herb': 'WHITE_HERB',
    'Air Balloon': 'AIR_BALLOON', 'Hearthflame Mask': 'HEARTHFLAME_MASK',
    'Cornerstone Mask': 'CORNERSTONE_MASK',
}


def make_item_pool(sets, abilities=False):
    """Enable represented held items, reporting unsupported items and moves.

    Ability projection is independent and can be enabled for interactions such
    as Booster Energy with Protosynthesis or Quark Drive.
    """

    pool, omissions = (make_ability_pool if abilities else make_sandbox_pool)(sets)
    items = []
    for config, report in zip(sets, omissions, strict=True):
        identifier = CURATED_ITEMS.get(config.item)
        items.append(int(ItemEnum[identifier]) if identifier else 0)
        if identifier:
            report['item'] = None
    return pool.replace(item=jnp.asarray(items, jnp.int32)[:, None]), omissions
