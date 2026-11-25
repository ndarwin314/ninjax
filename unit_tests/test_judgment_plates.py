"""Judgment's existing Plate type resolution and priority preparation."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Type
from ninjax.item_rules import PLATE_TYPES
from ninjax.moves.properties import effective_move, effective_priority, judgment_move_type
from ninjax.calculations.damage import chain_item_power_modifiers, compute_base_power
from ninjax.side.item_effects import remove_item, swap_items
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move, make_sandbox_teams
from ninjax.teams.curated import CURATED_ITEMS
from unit_tests.test_battle import neutral_type


@pytest.mark.parametrize('item,expected', [
    (I.FIST_PLATE, Type.FIGHTING), (I.SKY_PLATE, Type.FLYING),
    (I.TOXIC_PLATE, Type.POISON), (I.EARTH_PLATE, Type.GROUND),
    (I.STONE_PLATE, Type.ROCK), (I.INSECT_PLATE, Type.BUG),
    (I.SPOOKY_PLATE, Type.GHOST), (I.IRON_PLATE, Type.STEEL),
    (I.FLAME_PLATE, Type.FIRE), (I.SPLASH_PLATE, Type.WATER),
    (I.MEADOW_PLATE, Type.GRASS), (I.ZAP_PLATE, Type.ELECTRIC),
    (I.MIND_PLATE, Type.PSYCHIC), (I.ICICLE_PLATE, Type.ICE),
    (I.DRACO_PLATE, Type.DRAGON), (I.DREAD_PLATE, Type.DARK),
    (I.PIXIE_PLATE, Type.FAIRY), (I.NONE, Type.NORMAL),
    (I.LEFTOVERS, Type.NORMAL),
])
def test_judgment_plate_types(item, expected):
    """Each represented Plate maps to its Gen 9 Judgment type."""
    assert jax.jit(judgment_move_type)(jnp.array([item]))[0] == expected
    modifier = chain_item_power_modifiers(jnp.int32(4096), expected, item, 0, 1)
    assert modifier == (4915 if item not in (I.NONE, I.LEFTOVERS) else 4096)
    assert chain_item_power_modifiers(jnp.int32(4096), Type.NONE, item, 0, 1) == 4096


@pytest.mark.parametrize('ability', [AbilityEnum.NONE, AbilityEnum.NORMALIZE, AbilityEnum.PIXILATE])
def test_judgment_type_and_priority(neutral_type, ability):
    """Plate typing is retained through type-conversion ability checks."""
    _, state = neutral_type
    mon = state.active[0].replace(item=jnp.array([I.FLAME_PLATE]), ability=jnp.array([ability]))
    state = update_active(state, 0, mon)
    move = make_move('Judgment')
    assert effective_move(state, 0, move).type[0] == Type.FIRE
    assert effective_priority(state, 0, move)[0] == 0


def test_judgment_sky_plate_gale_wings_priority(neutral_type):
    """Priority preparation uses Judgment's resolved Flying type."""
    _, state = neutral_type
    mon = state.active[0].replace(item=jnp.array([I.SKY_PLATE]), ability=jnp.array([AbilityEnum.GALE_WINGS]))
    state = update_active(state, 0, mon)
    assert effective_priority(state, 0, make_move('Judgment'))[0] == 1


@pytest.mark.parametrize('suppression', ['none', 'klutz', 'magic_room', 'embargo', 'suppressed_klutz'])
def test_plate_typing_and_power_suppression(neutral_type, suppression):
    """Suppression removes both Plate effects without removing the held item."""
    _, state = neutral_type
    ability = AbilityEnum.KLUTZ if suppression in ('klutz', 'suppressed_klutz') else AbilityEnum.NONE
    mon = state.active[0].replace(item=jnp.array([I.FLAME_PLATE]), ability=jnp.array([ability]))
    state = update_active(state, 0, mon)
    v = state.volatile_status
    state = state.replace(magic_room_duration=jnp.int32(2 if suppression == 'magic_room' else 0),
        volatile_status=v.replace(embargo_turns=v.embargo_turns.at[0, 0].set(2 if suppression == 'embargo' else 0),
                                  ability_suppressed=v.ability_suppressed.at[0, 0].set(suppression == 'suppressed_klutz')))
    suppressed = suppression in ('klutz', 'magic_room', 'embargo')
    judgment = effective_move(state, 0, make_move('Judgment'))
    assert judgment.type[0] == (Type.NORMAL if suppressed else Type.FIRE)
    fire = make_move('Flamethrower')
    attacker = state.effective_active[0]
    power = compute_base_power(attacker, state.active[1], fire, state.terrain,
        item_suppressed=(state.magic_room_duration > 0) | (state.volatile_status.embargo_turns[0, 0] > 0))
    assert power[0] == (90 if suppressed else 108)
    assert state.active.item[0, 0] == I.FLAME_PLATE


@pytest.mark.parametrize('plate', [I.FIST_PLATE, I.FLAME_PLATE, I.PIXIE_PLATE])
def test_arceus_plate_transfer_binding(neutral_type, plate):
    """Neither outgoing nor incoming Plates can be transferred to Arceus."""
    _, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(species=jnp.array([493]), item=jnp.array([plate])))
    assert not remove_item(state, 0, 1)[1]
    assert not swap_items(state, 0, 1)[1]
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.NONE])))
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([plate])))
    assert not swap_items(state, 0, 1)[1]
    state = update_active(state, 0, state.active[0].replace(species=jnp.array([1])))
    assert swap_items(state, 0, 1)[1]


def test_constructed_arceus_uses_stable_species_id():
    """Reference construction retains Arceus identity for item binding."""
    record = dict(set=dict(species='Arceus-Fire', level=100, moves=['tackle'] * 4),
                  nature=dict(plus=None, minus=None),
                  baseStats=dict.fromkeys(('hp', 'atk', 'def', 'spa', 'spd', 'spe'), 120),
                  types=['Fire'])
    team, _ = make_sandbox_teams(dict(initial=[[record] * 6] * 2))
    assert team.species[0, 0, 0] == 493


def test_all_plates_are_available_to_curated_teams():
    """Every represented Plate name projects to its stable item ID."""
    for item, _ in PLATE_TYPES:
        name = item.name.replace('_', ' ').title()
        assert CURATED_ITEMS[name] == item.name
