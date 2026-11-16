"""Report full-format coverage and gaps without conflating support with parity.

Run as ``python -m tools.audit_random_battle_mechanics`` from the project root
with the configured project interpreter. This audits
all role candidates, not every legal four-move combination or interaction.
"""
import json
from pathlib import Path

from ninjax.curated import CURATED_ABILITIES, CURATED_ITEMS
from ninjax.prototype import CATALOGUE


def coverage_report(reference):
    """Return exhaustive role/move/ability gaps and explicitly sampled item gaps."""
    missing_moves = sorted(set(reference['moves']) - CATALOGUE.keys())
    missing_abilities = sorted(set(reference['abilities']) - CURATED_ABILITIES.keys())
    missing_items = sorted(set(reference['sampledItems']) - CURATED_ITEMS.keys())
    affected = [f"{r['species']}:{r['index']}" for r in reference['roles']
                if set(r['movepool']) & set(missing_moves)
                or set(r['abilities']) & set(missing_abilities)]
    return {'revision': reference['revision'], 'format': reference['format'],
            'roles': len(reference['roles']), 'moves': len(reference['moves']),
            'supported_moves': len(reference['moves']) - len(missing_moves),
            'abilities': len(reference['abilities']),
            'projected_abilities': len(reference['abilities']) - len(missing_abilities),
            'projected_ability_names': len(CURATED_ABILITIES),
            'ability_projection_note': 'Unprojected does not imply no engine hook; further validation is required.',
            'missing_moves': missing_moves, 'unprojected_abilities': missing_abilities,
            'sampled_items': len(reference['sampledItems']),
            'projected_item_names': len(CURATED_ITEMS) - int('' in CURATED_ITEMS),
            'item_sample_teams': reference['itemSampleTeams'],
            'item_scope': 'Sampled procedural choices; not an exhaustive item inventory.',
            'unprojected_sampled_items': missing_items,
            'roles_with_move_or_ability_gaps': affected,
            'controlled_damage_probes': len(reference['damage']),
            'controlled_setup_probes': len(reference['boosts']),
            'controlled_secondary_probes': len(reference['secondaries'])}


if __name__ == '__main__':
    fixture = Path('unit_tests/fixtures/showdown/gen9-random-mechanics.json')
    print(json.dumps(coverage_report(json.loads(fixture.read_text())), indent=2))
