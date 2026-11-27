"""Checks for projecting move catalogue effects into battle metadata."""
import pytest

from ninjax.enums.moves import HitRule
from ninjax.enums.pokemon import StatEnum
from ninjax.teams.construction import CATALOGUE, make_move


def test_catalogue_moves_construct():
    """Every catalogue entry can be projected into a move PyTree."""
    for name in CATALOGUE:
        make_move(name)


def test_effect_targets_and_healing():
    """Stat effects retain their recipient and healing retains its fraction."""
    assert bool(make_move('Swords Dance').target_self[0])
    assert not bool(make_move('Growl').target_self[0])
    assert float(make_move('Milk Drink').healing_fraction[0]) == 0.5
    close_combat = make_move('Close Combat')
    assert list(close_combat.self_stat[:2]) == [StatEnum.DEFENSE, StatEnum.SPECIAL_DEFENSE]
    assert list(close_combat.self_stages[:2]) == [-1, -1]
    flame_charge = make_move('Flame Charge')
    assert int(flame_charge.secondaries.user_stats[0, 0]) == StatEnum.SPEED
    assert float(flame_charge.secondaries.probabilities[0]) == pytest.approx(1.0)
    assert float(make_move('Charge Beam').secondaries.probabilities[0]) == pytest.approx(0.7)


def test_catalogue_multihit_rules():
    """Fixed and variable hit counts are read from catalogue metadata."""
    assert int(make_move('Arm Thrust').hit_rule[0]) == HitRule.TWO_TO_FIVE
    assert int(make_move('Twin Beam').hit_rule[0]) == HitRule.FIXED_TWO
    assert int(make_move('Surging Strikes').hit_rule[0]) == HitRule.FIXED_THREE
    assert int(make_move('Population Bomb').hit_rule[0]) == HitRule.POPULATION_BOMB


@pytest.mark.parametrize('name', [
    'Aeroblast', 'Air Cutter', 'Aqua Cutter', 'Attack Order', 'Blaze Kick',
    'Crabhammer', 'Cross Chop', 'Cross Poison', 'Drill Run', 'Esper Wing',
    'Ivy Cudgel', 'Leaf Blade', 'Night Slash', 'Poison Tail', 'Psycho Cut',
    'Razor Leaf', 'Shadow Claw', 'Sky Attack', 'Slash', 'Snipe Shot',
    'Spacial Rend', 'Stone Edge', 'Triple Arrows',
])
def test_increased_critical_ratio_is_loaded(name):
    """Catalogue high-critical moves retain their increased critical stage."""
    assert int(make_move(name).crit_stage[0]) == 1
