"""Showdown parity and lifecycle tests for passive Gen 9 abilities and items."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.field import step_field
from ninjax.battle.interruptions import before_move
from ninjax.battle.residuals import residual_abilities
from ninjax.battle.switching import swap_is_alive, swap_out
from ninjax.battle.types import Phase
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum as T, Weather, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Status, Type
from ninjax.moves.effects import do_contact, secondary_effects
from ninjax.moves.execution import _prepare_move_use
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.moves.multi_hit import resolve_multi_hit
from ninjax.moves.properties import effective_move, effective_priority, move_accuracy
from ninjax.moves.routing import move_hits
from ninjax.moves.status import do_status_move
from ninjax.restrictions import apply_restriction
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import swap_items, update_items
from ninjax.side.stat_effects import stat_changes
from ninjax.side.status_effects import apply_confusion, set_status
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from ninjax.teams.curated import CURATED_ABILITIES, CURATED_ITEMS
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/showdown/passive-mechanics.json').read_text())
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
STATUSES = {'': Status.NONE, 'brn': Status.BURN, 'psn': Status.POISON, 'tox': Status.TOXIC,
            'par': Status.PARALYZE, 'slp': Status.SLEEP, 'frz': Status.FREEZE}
WEATHER = {'': W.NONE, 'sunnyday': W.SUN, 'raindance': W.RAIN, 'sandstorm': W.SANDSTORM, 'snowscape': W.SNOW}


def configure(state, config, source=None, target=None):
    """Project reference stats and passive identities into an immutable battle."""
    for player, record in enumerate((source, target)):
        mon = state.active[player]
        name = config.get('ability' if player == 0 else 'defender', '')
        ability = A[CURATED_ABILITIES[name]] if name else A.NONE
        item = I[CURATED_ITEMS[config['item']]] if player == 0 and config.get('item') else I.NONE
        if record:
            table = mon.stat_table.replace(base_stats=jnp.array([record['baseStats'][s] for s in STATS]), evs=jnp.full(6, 4))
            types = [Type[t.upper()] if t != '???' else Type.NONE for t in record['types']]
            types += [Type.NONE] * (2-len(types))
            mon = mon.replace(stat_table=table, species=jnp.array([record['species']]),
                current_hp=jnp.array([record['hp']]), type_list=jnp.array(types))
        mon = mon.replace(ability=jnp.array([ability]), item=jnp.array([item]),
                          status=jnp.array([STATUSES.get(config.get('status', '') if player == 0 else '', Status.NONE)]))
        state = update_active(state, player, mon)
    state = state.replace(weather=Weather(WEATHER[config.get('weather', '')], 5))
    if config.get('terrain'):
        state = state.replace(terrain=Terrain(T.ELECTRIC, 5))
    if config.get('stage'):
        state = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 5].set(config['stage'])))
    if config.get('tailwind'):
        state = state.replace(side_conditions=state.side_conditions.replace(tailwind=state.side_conditions.tailwind.at[0, 0].set(4)))
    if config.get('ability') == 'Speed Boost':
        # Match Battle.reset's initial lead, which enters before turn zero.
        state = state.replace(volatile_status=state.volatile_status.replace(
            entry_turn=state.volatile_status.entry_turn.at[0, 0].set(-1)))
    return state


@pytest.mark.parametrize('probe', REFERENCE['damage'], ids=lambda p: str(p['config']))
def test_damage_against_showdown(neutral_type, probe):
    """Compare real ability/item damage at the same roll, including odd stats."""
    key, state = neutral_type
    state = configure(state, probe['config'], probe['source'], probe['target'])
    # Disable random criticals by selecting a known noncritical key rather than
    # changing either participant's tested defensive ability.
    while float(jax.random.uniform(jax.random.split(key, 3)[1])) < .5:
        key = jax.random.fold_in(key, 1)
    roll = int(jax.random.randint(jax.random.split(key, 3)[2], (), 85, 101))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move(probe['config']['move']))
    assert int(out.active.current_hp[1, 0]) == probe['target']['hp'] - probe['rolls'][100-roll]


@pytest.mark.parametrize('probe', REFERENCE['speed'], ids=lambda p: str(p['config']))
def test_speed_against_showdown(neutral_type, probe):
    """Speed stages precede chained weather/item/field modifiers and paralysis."""
    _, state = neutral_type
    state = configure(state, probe['config'], probe['source'])
    assert int(state.boosted_stats[0, 5]) == probe['expected']


@pytest.mark.parametrize('probe', REFERENCE['statuses'], ids=lambda p: f"{p['ability']}-{p['status']}-{p['weather']}")
def test_status_immunity_against_showdown(neutral_type, probe):
    """Apply the same major status against each passive immunity ability."""
    key, state = neutral_type
    state = configure(state, {'defender': probe['ability'], 'weather': probe['weather']})
    _, out = set_status(key, state, 1, jnp.array([STATUSES[probe['status']]]))
    assert out.active.status[1, 0] == STATUSES[probe['expected']]


@pytest.mark.parametrize('probe', REFERENCE['berries'], ids=lambda p: str(p))
def test_status_berries_against_showdown(neutral_type, probe):
    """Lum clears both conditions; Chesto only reacts to sleep."""
    _, state = neutral_type
    state = configure(state, {'item': probe['item'], 'status': probe['status'], 'ability': probe['ability']}, probe['source'])
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([probe['initialHP']])))
    state = state.replace(volatile_status=state.volatile_status.replace(
        confusion_turns=state.volatile_status.confusion_turns.at[0, 0].set(3 if probe['confusion'] else 0)))
    out = update_items(state)
    assert out.active.status[0, 0] == STATUSES[probe['expectedStatus']]
    assert (out.active.item[0, 0] == I.NONE) == (probe['expectedItem'] == '')
    assert bool(out.volatile_status.confusion_turns[0, 0] > 0) == probe['expectedConfusion']
    assert out.active.current_hp[0, 0] == probe['expectedHP']


def test_rest_chesto_and_unnerve(neutral_type):
    """Rest fully heals before Chesto wakes its holder; Unnerve prevents eating."""
    key, state = neutral_type
    state = configure(state, {'item': 'Chesto Berry', 'status': 'brn'})
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([1])))
    _, out = do_status_move(key, state, 0, make_move('Rest'), 0, 0)
    assert out.active.current_hp[0, 0] == out.active.max_hp[0]
    assert out.active.status[0, 0] == Status.NONE and out.active.sleep_counter[0, 0] == 0
    assert out.active.item[0, 0] == I.NONE
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.UNNERVE])))
    _, out = do_status_move(key, state, 0, make_move('Rest'), 0, 0)
    assert out.active.status[0, 0] == Status.SLEEP and out.active.item[0, 0] == I.CHESTO_BERRY


def test_white_herb_after_shell_smash_and_received_item(neutral_type):
    """Clear both defense drops together, retain positive stages and consume once."""
    key, state = neutral_type
    state = configure(state, {'item': 'White Herb'})
    _, boosted = do_status_move(key, state, 0, make_move('Shell Smash'), 0, 0)
    out = update_items(boosted)
    np.testing.assert_array_equal(out.boosts.normal_boosts[0], [0, 2, 0, 2, 0, 2])
    assert out.active.item[0, 0] == I.NONE
    # Atomic exchange immediately activates a received herb on the recipient.
    recipient = boosted.replace(team=boosted.team.replace(item=boosted.team.item.at[0, 0, 0].set(I.NONE).at[1, 0, 0].set(I.WHITE_HERB)))
    out, success = swap_items(recipient, 0, 1)
    assert success and out.active.item[0, 0] == I.NONE
    np.testing.assert_array_equal(out.boosts.normal_boosts[0], [0, 2, 0, 2, 0, 2])


def test_choice_scarf_locks_clears_and_orders(neutral_type):
    """Scarf uses the shared Choice lifecycle for actions, switches and exchanges."""
    key, state = neutral_type
    state = configure(state, {'item': 'Choice Scarf'})
    env = Battle(state.team)
    episode, _ = env.reset(key)
    _, locked = _prepare_move_use(key, episode.battle, 0, 1)
    mask = env.legal_action_mask(episode.replace(battle=locked))
    assert mask[0, 1] and not mask[0, 0] and not mask[0, 2]
    switched = swap_out(key, locked, 0, 1)
    assert switched.volatile_status.restrictions.choice_lock[0, 0] == -1
    exchanged, success = swap_items(locked, 0, 1)
    assert success and exchanged.volatile_status.restrictions.choice_lock[0, 0] == -1


def test_wide_lens_accuracy(neutral_type):
    """Wide Lens boosts accuracy by 1.1 and never makes it exceed certainty."""
    _, state = neutral_type
    state = configure(state, {'item': 'Wide Lens'})
    np.testing.assert_allclose(move_accuracy(state, 0, make_move('Hydro Pump')), [.88], atol=1e-6)
    assert move_accuracy(state, 0, make_move('Tackle'))[0] == 1


def test_air_balloon_grounding_pop_and_substitute(neutral_type):
    """Balloon floats, ignores Ground hits, and only pops when the holder is hit."""
    key, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.AIR_BALLOON])))
    assert not state.is_grounded(1)[0]
    state = equip_move(state, 'Earthquake')
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.item[1, 0] == I.AIR_BALLOON
    state = equip_move(state, 'Tackle')
    _, out = move_hits(key, state, 0, 0)
    assert out.active.item[1, 0] == I.NONE and out.is_grounded(1)[0]
    sub = state.replace(volatile_status=state.volatile_status.replace(substitute_hp=state.volatile_status.substitute_hp.at[1, 0].set(100)))
    _, out = move_hits(key, sub, 0, 0)
    assert out.active.item[1, 0] == I.AIR_BALLOON


@pytest.mark.parametrize('item,status', [('Flame Orb', Status.BURN), ('Toxic Orb', Status.TOXIC)])
def test_orbs_delay_damage_and_do_not_consume(neutral_type, item, status):
    """New orb statuses do not inflict damage until the following turn."""
    key, state = neutral_type
    state = configure(state, {'item': item})
    key, out = step_field(key, state)
    assert out.active.status[0, 0] == status
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert out.active.item[0, 0] == state.active.item[0, 0]
    _, later = step_field(key, out)
    assert later.active.current_hp[0, 0] < out.active.current_hp[0, 0]


@pytest.mark.parametrize('status', ['psn', 'tox'])
def test_poison_heal_and_natural_cure(neutral_type, status):
    """Both poisons heal Poison Heal; Natural Cure restores the outgoing team slot."""
    key, state = neutral_type
    state = configure(state, {'ability': 'Poison Heal', 'status': status})
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([100])))
    _, out = step_field(key, state)
    assert out.active.current_hp[0, 0] == 100 + state.active.max_hp[0] // 8
    natural = configure(state, {'ability': 'Natural Cure', 'status': status})
    out = swap_out(key, natural, 0, 1)
    assert out.team.status[0, 0, 0] == Status.NONE


def test_hydration_speed_boost_and_switch_timing(neutral_type):
    """Hydration follows rain suppression; Speed Boost skips a fresh switch-in."""
    key, state = neutral_type
    rain = configure(state, {'ability': 'Hydration', 'status': 'brn', 'weather': 'raindance'})
    _, out = residual_abilities(key, rain)
    assert out.active.status[0, 0] == Status.NONE
    rain = update_active(rain, 1, rain.active[1].replace(ability=jnp.array([A.CLOUD_NINE])))
    _, out = residual_abilities(key, rain)
    assert out.active.status[0, 0] == Status.BURN
    speed = configure(state, {'ability': 'Speed Boost'})
    _, out = residual_abilities(key, speed)
    assert out.boosts.normal_boosts[0, 5] == 1
    speed = speed.replace(volatile_status=speed.volatile_status.replace(entry_turn=speed.volatile_status.entry_turn.at[0, 0].set(speed.turn_number)))
    _, out = residual_abilities(key, speed)
    assert out.boosts.normal_boosts[0, 5] == 0


@pytest.mark.parametrize('ability,move,stat,amount', [
    ('Lightning Rod', 'Thunderbolt', 3, 1), ('Motor Drive', 'Thunderbolt', 5, 1),
    ('Sap Sipper', 'Giga Drain', 1, 1), ('Well-Baked Body', 'Flamethrower', 2, 2),
    ('Volt Absorb', 'Thunderbolt', 0, 0), ('Earth Eater', 'Earthquake', 0, 0),
    ('Dry Skin', 'Waterfall', 0, 0)])
def test_absorption_reuses_hit_dispatch(neutral_type, ability, move, stat, amount):
    """Absorption works before accuracy and prevents ordinary target damage."""
    key, state = neutral_type
    state = configure(state, {'defender': ability})
    state = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([100])))
    state = equip_move(state, move)
    assert move_accuracy(state, 0, state.active[0].moves[0].replace(accuracy=jnp.array([0.])))[0] == 1
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] == (100 if stat else 100 + state.active.max_hp[1] // 4)
    if stat:
        assert out.boosts.normal_boosts[1, stat] == amount


def test_queenly_majesty_and_status_berry_integration(neutral_type):
    """Priority protection blocks attacks; status and confusion activate Lum immediately."""
    key, state = neutral_type
    protected = configure(state, {'defender': 'Queenly Majesty'})
    protected = equip_move(protected, 'Aqua Jet')
    _, out = move_hits(key, protected, 0, 0)
    assert out.active.current_hp[1, 0] == protected.active.current_hp[1, 0]
    berry = configure(state, {'item': 'Lum Berry'})
    _, out = set_status(key, berry, 0, jnp.array([Status.BURN]))
    assert out.active.status[0, 0] == Status.NONE and out.active.item[0, 0] == I.NONE
    _, out = apply_confusion(key, berry, 0)
    assert out.volatile_status.confusion_turns[0, 0] == 0 and out.active.item[0, 0] == I.NONE


def test_shed_skin_advances_only_for_eligible_status(neutral_type):
    """Shed Skin uses a 33% cure roll and does not draw RNG when healthy."""
    key, state = neutral_type
    poisoned = configure(state, {'ability': 'Shed Skin', 'status': 'tox'})
    expected_key, subkey = jax.random.split(key)
    cured = bool(jax.random.randint(subkey, (), 0, 100) < 33)
    actual_key, out = residual_abilities(key, poisoned)
    np.testing.assert_array_equal(jax.random.key_data(actual_key), jax.random.key_data(expected_key))
    assert (out.active.status[0, 0] == Status.NONE) == cured
    healthy = configure(state, {'ability': 'Shed Skin'})
    actual_key, _ = residual_abilities(key, healthy)
    np.testing.assert_array_equal(jax.random.key_data(actual_key), jax.random.key_data(key))


@pytest.mark.parametrize('ability,move', [('Levitate', 'Earthquake'), ('Soundproof', 'Boomburst'),
    ('Bulletproof', 'Shadow Ball'), ('Overcoat', 'Sleep Powder')])
def test_projected_move_immunities(neutral_type, ability, move):
    """Existing move-immunity hooks are reachable through enabled projections."""
    key, state = neutral_type
    state = equip_move(configure(state, {'defender': ability}), move)
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.status[1, 0] == Status.NONE


def test_passive_projection_enum_ids_are_valid():
    """Every enabled ability/item maps to a real enum, including aliases."""
    assert all(name in A.__members__ for name in CURATED_ABILITIES.values())
    assert all(name in I.__members__ for name in CURATED_ITEMS.values())


def test_compound_eyes_and_no_guard(neutral_type):
    """Accuracy modifiers use integer percentages; No Guard bypasses a zero base."""
    _, state = neutral_type
    eyes = configure(state, {'ability': 'Compound Eyes'})
    np.testing.assert_allclose(move_accuracy(eyes, 0, make_move('Focus Blast')), [.91], atol=1e-6)
    guard = configure(state, {'defender': 'No Guard'})
    assert move_accuracy(guard, 0, make_move('Focus Blast').replace(accuracy=jnp.array([0.])))[0] == 1


def test_projected_secondary_abilities(neutral_type):
    """Shield Dust, Inner Focus and Serene Grace apply to represented secondaries."""

    key, state = neutral_type
    dust = configure(state, {'defender': 'Shield Dust'})
    _, out = secondary_effects(key, dust, 0, make_move('Lumina Crash'))
    assert out.boosts.normal_boosts[1, 4] == 0
    focus = configure(state, {'defender': 'Inner Focus'})
    move = make_move('Iron Head')
    move = move.replace(secondaries=move.secondaries.replace(probabilities=move.secondaries.probabilities.at[0].set(1.)))
    _, out = secondary_effects(key, focus, 0, move)
    assert not out.volatile_status.flinched[1, 0]
    grace = configure(state, {'ability': 'Serene Grace'})
    _, roll_key = jax.random.split(key)
    _, out = secondary_effects(key, grace, 0, make_move('Iron Head'))
    assert bool(out.volatile_status.flinched[1, 0]) == bool(jax.random.uniform(roll_key) < .6)


def test_rock_head_and_shell_armor(neutral_type):
    """Recoil respects Rock Head and Magic Guard, and Shell Armor prevents crits."""
    key, state = neutral_type
    run = jax.jit(resolve_multi_hit)
    _, ordinary = run(key, state, 0, make_move('Double-Edge'))
    rock = configure(state, {'ability': 'Rock Head'})
    _, protected = run(key, rock, 0, make_move('Double-Edge'))
    assert ordinary.active.current_hp[0, 0] < protected.active.current_hp[0, 0]
    assert protected.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    magic_guard = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.MAGIC_GUARD])))
    _, guarded = run(key, magic_guard, 0, make_move('Double-Edge'))
    assert guarded.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    _, critical, _, _ = resolve_damaging_hit(key, state, 0, make_move('Wicked Blow'))
    armor = configure(state, {'defender': 'Shell Armor'})
    _, blocked, _, _ = resolve_damaging_hit(key, armor, 0, make_move('Wicked Blow'))
    assert blocked.active.current_hp[1, 0] > critical.active.current_hp[1, 0]


@pytest.mark.parametrize('probe', REFERENCE['residuals'], ids=lambda p: str(p['config']))
def test_complete_residual_turn_against_showdown(neutral_type, probe):
    """Compare status, HP, item and stages after a real Showdown residual turn."""
    key, state = neutral_type
    state = configure(state, probe['config'], probe['source'])
    if probe['config'].get('firstMove'):
        _, state = do_status_move(key, state, 0, make_move(probe['config']['firstMove']), 0, 0)
        state = update_items(state)
    _, out = step_field(key, state)
    assert out.active.current_hp[0, 0] == probe['expectedHP']
    assert out.active.status[0, 0] == STATUSES[probe['expectedStatus']]
    assert (out.active.item[0, 0] == I.NONE) == (probe['expectedItem'] == '')
    actual = list(np.asarray(out.boosts.normal_boosts[0, 1:])) + list(np.asarray(out.boosts.acc_boosts[0]))
    assert actual == [probe['expectedBoosts'][s] for s in ('atk','def','spa','spd','spe','accuracy','evasion')]


@pytest.mark.parametrize('ability,move,expected', [('Prankster', 'Calm Mind', 1),
    ('Gale Wings', 'Hurricane', 1), ('Triage', 'Drain Punch', 3)])
def test_priority_ability_projection(neutral_type, ability, move, expected):
    """Priority hooks modify effective moves without changing stored move metadata."""

    _, state = neutral_type
    state = configure(state, {'ability': ability})
    original = make_move(move)
    assert effective_move(state, 0, original).priority[0] == expected
    assert effective_priority(state, 0, original)[0] == expected
    assert original.priority[0] == 0
    if ability == 'Gale Wings':
        state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([1])))
        assert effective_move(state, 0, original).priority[0] == 0
        assert effective_priority(state, 0, original)[0] == 0


def test_prankster_dark_and_good_as_gold_field_targets(neutral_type):
    """Prankster cannot target Dark opponents; Good as Gold allows opposing hazards."""
    key, state = neutral_type
    prank = configure(state, {'ability': 'Prankster'})
    prank = update_active(prank, 1, prank.active[1].replace(type_list=jnp.array([Type.DARK, Type.NONE])))
    prank = equip_move(prank, 'Will-O-Wisp')
    _, out = move_hits(key, prank, 0, 0)
    assert out.active.status[1, 0] == Status.NONE
    gold = configure(state, {'defender': 'Good as Gold'})
    _, out = move_hits(key, equip_move(gold, 'Will-O-Wisp'), 0, 0)
    assert out.active.status[1, 0] == Status.NONE
    _, out = move_hits(key, equip_move(gold, 'Spikes'), 0, 0)
    assert out.hazards.spikes[1, 0] == 1


def test_stage_protections_and_contrary(neutral_type):
    """Passive stage prevention only blocks the corresponding opposing drops."""

    _, state = neutral_type
    for ability in ('Clear Body', 'Full Metal Body', 'Big Pecks'):
        protected = configure(state, {'defender': ability})
        out = stat_changes(protected, 1, jnp.array([2]), jnp.array([-1]), True)
        assert out.boosts.normal_boosts[1, 2] == 0
        own = stat_changes(protected, 1, jnp.array([2]), jnp.array([-1]), False)
        assert own.boosts.normal_boosts[1, 2] == -1
    contrary = configure(state, {'ability': 'Contrary'})
    out = stat_changes(contrary, 0, jnp.array([3]), jnp.array([-2]))
    assert out.boosts.normal_boosts[0, 3] == 2
    keen = configure(state, {'defender': 'Keen Eye'})
    out = stat_changes(keen, 1, jnp.array([6]), jnp.array([-1]), True)
    assert out.boosts.acc_boosts[1, 0] == 0


def test_type_conversion_and_weight_abilities(neutral_type):
    """Existing transient type and weight hooks support the newly enabled names."""

    _, state = neutral_type
    for ability, typ in [('Galvanize', Type.ELECTRIC), ('Liquid Voice', Type.WATER)]:
        s = configure(state, {'ability': ability})
        move = effective_move(s, 0, make_move('Boomburst'))
        assert move.type[0] == typ
        assert move.base_power[0] == (168 if ability == 'Galvanize' else 140)
    target = update_active(state, 1, state.active[1].replace(weight_hg=jnp.array([250])))
    heavy = configure(target, {'defender': 'Heavy Metal'})
    light = configure(target, {'defender': 'Light Metal'})
    assert effective_move(heavy, 0, make_move('Low Kick')).base_power[0] == 80
    assert effective_move(light, 0, make_move('Low Kick')).base_power[0] == 40


def test_unnerve_release_and_flash_fire(neutral_type):
    """A deferred Sitrus activates when Unnerve leaves; Flash Fire boosts later Fire hits."""
    key, state = neutral_type
    state = configure(state, {'defender': 'Unnerve'})
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.SITRUS_BERRY]), current_hp=jnp.array([100])))
    held = update_items(state)
    assert held.active.item[0, 0] == I.SITRUS_BERRY
    released = swap_out(key, held, 1, 1)
    assert released.active.item[0, 0] == I.NONE
    assert released.active.current_hp[0, 0] == 100 + state.active.max_hp[0] // 4
    fire = configure(state, {'defender': 'Flash Fire'})
    _, out = move_hits(key, equip_move(fire, 'Flamethrower'), 0, 0)
    assert out.volatile_status.ability_state.flash_fire[1, 0]
    assert out.active.current_hp[1, 0] == fire.active.current_hp[1, 0]


@pytest.mark.parametrize('ability,move,stat,amount', [('Water Compaction','Waterfall',2,2),
    ('Justified','Bite',1,1), ('Rattled','Shadow Ball',5,1)])
def test_surviving_hit_reactions(neutral_type, ability, move, stat, amount):
    """Represented damaging-hit reactions boost the correct surviving defender."""
    key, state = neutral_type
    state = configure(state, {'defender': ability})
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move(move))
    assert out.boosts.normal_boosts[1, stat] == amount
    assert out.active.current_hp[1, 0] > 0


def test_download_and_rattled_intimidate(neutral_type):
    """Download breaks defense ties toward SpA; Intimidate can also trigger Rattled."""

    _, state = neutral_type
    download = swap_is_alive(configure(state, {'ability': 'Download'}), 0)
    assert download.boosts.normal_boosts[0, 3] == 1
    physical = configure(state, {'ability': 'Download'})
    physical = physical.replace(boosts=physical.boosts.replace(normal_boosts=physical.boosts.normal_boosts.at[1, 4].set(1)))
    assert swap_is_alive(physical, 0).boosts.normal_boosts[0, 1] == 1
    rattled = swap_is_alive(configure(state, {'ability': 'Intimidate', 'defender': 'Rattled'}), 0)
    assert rattled.boosts.normal_boosts[1, 1] == -1 and rattled.boosts.normal_boosts[1, 5] == 1


def test_early_bird_and_restriction_immunities(neutral_type):
    """Early Bird wakes at zero; Aroma Veil and Oblivious prevent represented Taunt."""


    key, state = neutral_type
    early = configure(state, {'ability': 'Early Bird', 'status': 'slp'})
    early = update_active(early, 0, early.active[0].replace(sleep_counter=jnp.array([2])))
    _, out, blocked = before_move(key, early, 0)
    assert out.active.status[0, 0] == Status.NONE and not blocked
    for ability in ('Aroma Veil', 'Oblivious'):
        protected = configure(state, {'defender': ability})
        assert apply_restriction(protected, 0, make_move('Taunt')).volatile_status.restrictions.taunt_turns[1, 0] == 0
    own = configure(state, {'defender': 'Own Tempo'})
    _, out = apply_confusion(key, own, 1)
    assert out.volatile_status.confusion_turns[1, 0] == 0


@pytest.mark.parametrize('ability,status', [('Static', Status.PARALYZE), ('Flame Body', Status.BURN), ('Effect Spore', None)])
def test_contact_status_abilities(neutral_type, ability, status):
    """Existing contact status rolls activate and respect the major-status model."""

    key, state = neutral_type
    while float(jax.random.uniform(jax.random.split(key, 3)[1])) >= .3:
        key = jax.random.fold_in(key, 1)
    state = configure(state, {'defender': ability})
    _, out = do_contact(key, state, 0)
    assert out.active.status[0, 0] != Status.NONE
    if status is not None:
        assert out.active.status[0, 0] == status


def test_poison_touch_respects_shield_dust(neutral_type):
    """The contact-added secondary is prevented by Shield Dust."""

    key, state = neutral_type
    while float(jax.random.uniform(jax.random.split(key, 3)[2])) >= .3:
        key = jax.random.fold_in(key, 1)
    poison = configure(state, {'ability': 'Poison Touch'})
    _, out = do_contact(key, poison, 0)
    assert out.active.status[1, 0] == Status.POISON
    dust = configure(state, {'ability': 'Poison Touch', 'defender': 'Shield Dust'})
    _, out = do_contact(key, dust, 0)
    assert out.active.status[1, 0] == Status.NONE


@pytest.mark.parametrize('ability,stat', [('Defiant', 1), ('Competitive', 3)])
def test_actual_opposing_drops_trigger_responses(neutral_type, ability, stat):
    """Capped/prevented and self drops do not trigger responses; accuracy drops do."""

    _, state = neutral_type
    state = configure(state, {'defender': ability})
    out = stat_changes(state, 1, jnp.array([2]), jnp.array([-1]), True)
    assert out.boosts.normal_boosts[1, 2] == -1
    assert out.boosts.normal_boosts[1, stat] == 2
    own = stat_changes(state, 1, jnp.array([2]), jnp.array([-1]), False)
    assert own.boosts.normal_boosts[1, stat] == 0
    capped = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 2].set(-6)))
    out = stat_changes(capped, 1, jnp.array([2]), jnp.array([-1]), True)
    assert out.boosts.normal_boosts[1, stat] == 0
    accuracy = stat_changes(state, 1, jnp.array([6]), jnp.array([-1]), True)
    assert accuracy.boosts.acc_boosts[1, 0] == -1
    assert accuracy.boosts.normal_boosts[1, stat] == 2


def test_liquid_ooze_seed_sower_and_fainted_status(neutral_type):
    """Existing drain/terrain hooks work, and contact cannot status a fainted actor."""
    key, state = neutral_type
    ooze = configure(state, {'defender': 'Liquid Ooze'})
    ooze = update_active(ooze, 0, ooze.active[0].replace(current_hp=jnp.array([100])))
    _, out, _, _ = resolve_damaging_hit(key, ooze, 0, make_move('Giga Drain'))
    assert out.active.current_hp[0, 0] < 100
    seed = configure(state, {'defender': 'Seed Sower'})
    _, out, _, _ = resolve_damaging_hit(key, seed, 0, make_move('Tackle'))
    assert out.terrain.terrain == T.GRASSY
    fainted = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([0]), is_alive=jnp.array([False])))
    _, out = set_status(key, fainted, 0, jnp.array([Status.BURN]))
    assert out.active.status[0, 0] == Status.NONE


def test_sturdy_and_unaware_existing_hooks(neutral_type):
    """Sturdy saves a full-HP holder once; Unaware bypasses opposing stat stages."""
    key, state = neutral_type
    sturdy = configure(state, {'defender': 'Sturdy'})
    out = take_damage_value(sturdy, 1, 10000, True, source=0)
    assert out.active.current_hp[1, 0] == 1
    assert take_damage_value(out, 1, 10000, True, source=0).active.current_hp[1, 0] == 0
    unaware = configure(state, {'defender': 'Unaware'})
    raised = unaware.replace(boosts=unaware.boosts.replace(normal_boosts=unaware.boosts.normal_boosts.at[0, 1].set(6)))
    _, ordinary, _, _ = resolve_damaging_hit(key, unaware, 0, make_move('Tackle'))
    _, ignored, _, _ = resolve_damaging_hit(key, raised, 0, make_move('Tackle'))
    assert ordinary.active.current_hp[1, 0] == ignored.active.current_hp[1, 0]


@pytest.mark.parametrize('probe', REFERENCE['reactions'], ids=lambda p: str(p['config']))
def test_faint_and_wind_reactions_against_showdown(neutral_type, probe):
    """Compare real Showdown contact KOs and wind immunity/Tailwind reactions."""
    key, state = neutral_type
    state = configure(state, probe['config'], probe['source'], probe['target'])
    state = equip_move(state, probe['config']['move'])
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[0, 0] == probe['expectedSourceHP']
    assert out.active.current_hp[1, 0] == probe['expectedTargetHP']
    for player, field in [(0, 'expectedSourceBoosts'), (1, 'expectedTargetBoosts')]:
        actual = list(np.asarray(out.boosts.normal_boosts[player, 1:])) + list(np.asarray(out.boosts.acc_boosts[player]))
        assert actual == [probe[field][s] for s in ('atk','def','spa','spd','spe','accuracy','evasion')]


@pytest.mark.parametrize('ability', ['Arena Trap', 'Magnet Pull'])
def test_trapping_abilities_use_existing_action_mask(neutral_type, ability):
    """Traps respect grounding/type, Ghost immunity and replacement decisions."""
    key, state = neutral_type
    state = configure(state, {'defender': ability})
    state = update_active(state, 0, state.active[0].replace(type_list=jnp.array([Type.STEEL, Type.NONE])))
    env = Battle(state.team)
    episode, _ = env.reset(key)
    assert not jnp.any(env.legal_action_mask(episode)[0, 8:14])
    ghost = update_active(episode.battle, 0, episode.battle.active[0].replace(type_list=jnp.array([Type.STEEL, Type.GHOST])))
    assert jnp.any(env.legal_action_mask(episode.replace(battle=ghost))[0, 8:14])
    assert jnp.any(env.legal_action_mask(episode.replace(phase=jnp.int32(Phase.REPLACEMENT)))[0, 8:14])
    escaped = update_active(episode.battle, 0, episode.battle.active[0].replace(
        type_list=jnp.array([Type.NORMAL, Type.NONE]), item=jnp.array([I.AIR_BALLOON])))
    assert jnp.any(env.legal_action_mask(episode.replace(battle=escaped))[0, 8:14])
