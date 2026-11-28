"""Temporary identity rules and restoration under JIT."""
import jax
import jax.numpy as jnp

from ninjax.ability_effects import ability_move
from ninjax.battle.environment import Battle
from ninjax.battle.switching import swap_out
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import FormEnum, StatEnum, Type
from ninjax.side.form_effects import held_crowned_form, held_origin_form, palafin_departure, tera_shift
from ninjax.side.property_effects import copy_move, restore_move_bank, stat_move, transform, type_move
from ninjax.side.queries import can_remove_item
from ninjax.side.updates import clear_volatile_status, update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def test_transform_copies_properties_and_restores_native_moves(neutral_type):
    """Retain HP and items, use copied raw stats and PP, and restore originals."""
    _, state = neutral_type
    state = equip_move(equip_move(state, 'Transform'), 'Hydro Pump', 1)
    state = update_active_fields(state, 0, item=jnp.array([I.LEFTOVERS]))
    state = update_active_fields(state, 1, species=jnp.array([25]), ability=jnp.array([A.LEVITATE]),
                                 type_list=jnp.array([Type.ELECTRIC, Type.NONE]))
    original_hp = state.active.current_hp[0, 0]
    out, success = jax.jit(transform)(state, 0)
    assert success and out.volatile_status.transformed[0, 0]
    assert out.active.current_hp[0, 0] == original_hp and out.active.item[0, 0] == I.LEFTOVERS
    assert out.active_identity('species', 0)[0] == 25 and out.ability[0, 0] == A.LEVITATE
    assert out.effective_types(0)[0] == Type.ELECTRIC
    assert out.active_moves(0, 0).current_pp[0] == 5
    out = clear_volatile_status(restore_move_bank(out, 0), 0)
    assert out.active_moves(0, 0).identifier[0] == make_move('Transform').identifier[0]
    assert out.ability[0, 0] == A.NONE


def test_mimic_and_sketch_use_distinct_lifetimes(neutral_type):
    """Mimic is restored, and Sketch retains a copied move at native PP."""
    _, state = neutral_type
    state = equip_move(equip_move(state, 'Mimic'), 'Hydro Pump', 1)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(move_history=v.move_history.replace(
        last_move_id=v.move_history.last_move_id.at[1, 0].set(make_move('Hydro Pump').identifier[0]))))
    copy = jax.jit(copy_move)
    mimicked, success = copy(state, 0, 102)
    assert success and mimicked.active_moves(0, 0).current_pp[0] == 5
    assert restore_move_bank(mimicked, 0).active_moves(0, 0).identifier[0] == 102
    sketched, success = copy(equip_move(state, 'Sketch'), 0, 166)
    assert success and sketched.active_moves(0, 0).current_pp[0] == 5
    assert restore_move_bank(sketched, 0).active_moves(0, 0).identifier[0] == make_move('Hydro Pump').identifier[0]


def test_forest_type_and_replacement(neutral_type):
    """A third type participates in matchups and replacement clears it."""
    key, state = neutral_type
    state = update_active_fields(state, 1, type_list=jnp.array([Type.GROUND, Type.FLYING]))
    move = jax.jit(type_move)
    key, forest, success = move(key, state, 0, 571)
    assert success
    types = forest.effective_types(1, include_added=True)
    assert jnp.array_equal(types, jnp.array([Type.GROUND, Type.FLYING, Type.GRASS]))
    assert calculate_effectiveness_multiplier(Type.ICE, types) == 8
    _, soaked, success = move(key, forest, 0, 487)
    assert success and jnp.array_equal(soaked.effective_types(1, include_added=True), jnp.array([Type.WATER, 0, 0]))


def test_raw_stat_override_and_baton_pass(neutral_type):
    """Power Trick transfers its rule, exchanging the replacement's own stats."""
    key, state = neutral_type
    tricked = jax.jit(stat_move)(state, 0, 379)
    raw = state.active_stats(0)
    assert tricked.active_stats(0)[StatEnum.ATTACK] == raw[StatEnum.DEFENSE]
    switched = jax.jit(swap_out)(key, tricked, 0, 1, 1)
    native = state.team.stat_table[0, 1].stats
    assert switched.active_stats(0)[StatEnum.ATTACK] == native[StatEnum.DEFENSE]


def test_ability_shield_and_neutralizing_gas(neutral_type):
    """Gas is dynamic and Ability Shield protects Klutz and blocks mutation."""
    _, state = neutral_type
    gas = update_active_fields(state, 0, ability=jnp.array([A.NEUTRALIZING_GAS]))
    target = update_active_fields(gas, 1, ability=jnp.array([A.KLUTZ]))
    assert target.ability[1, 0] == A.NONE
    shielded = update_active_fields(target, 1, item=jnp.array([I.ABILITY_SHIELD]))
    assert shielded.ability[1, 0] == A.KLUTZ
    change = jax.jit(ability_move)
    out, success = change(shielded, 0, 380)
    assert not success and out.known_items[1, 0] == I.ABILITY_SHIELD
    out, success = change(target, 1, 380)
    assert success and out.ability[1, 0] == A.KLUTZ and out.ability[0, 0] == A.NONE


def test_species_form_changes_use_holder_stats(neutral_type):
    """Palafin's departure and Terapagos's entry persist on their party slots."""
    _, state = neutral_type
    state = update_active_fields(state, 0, species=jnp.array([964]), ability=jnp.array([A.ZERO_TO_HERO]))
    became_hero = jax.jit(palafin_departure)(state, 0)
    assert became_hero.active.form[0, 0] == FormEnum.PALAFIN_HERO
    state = update_active_fields(state, 1, species=jnp.array([1024]), ability=jnp.array([A.TERA_SHIFT]))
    became_terastal = jax.jit(tera_shift)(state, 1)
    assert became_terastal.active.form[1, 0] == FormEnum.TERAPAGOS_TERASTAL
    assert became_terastal.ability[1, 0] == A.TERA_SHELL


def test_species_bound_items_apply_their_forms(neutral_type):
    """Griseous Core and Rusted Sword change identity and stay bound."""
    _, state = neutral_type
    state = update_active_fields(state, 0, species=jnp.array([487]), item=jnp.array([I.GRISEOUS_CORE]))
    state = held_origin_form(state, 0)
    assert state.active.form[0, 0] == FormEnum.GIRATINA_ORIGIN
    assert state.ability[0, 0] == A.LEVITATE and not can_remove_item(state, 0, 1)
    state = equip_move(state, 'Iron Head', 1)
    state = update_active_fields(state, 1, species=jnp.array([888]), item=jnp.array([I.RUSTED_SWORD]))
    state = held_crowned_form(state, 1)
    assert state.active.form[1, 0] == FormEnum.ZACIAN_CROWNED
    assert state.active_moves(1, 0).identifier[0] == make_move('Behemoth Blade').identifier[0]
    assert not can_remove_item(state, 1, 0)


def test_transform_through_public_turn_and_switch(neutral_type):
    """The scheduler spends the original PP and exposes only the active copy."""
    key, state = neutral_type
    state = equip_move(equip_move(state, 'Transform'), 'Growl', 1)
    env = Battle(state.team)
    initial, _ = env.reset(key)
    step = jax.jit(env.step)
    copied, result = step(initial, jnp.array([0, 0]))
    assert not result.extras['invalid_action']
    assert copied.battle.volatile_status.transformed[0, 0]
    assert copied.battle.active_moves(0, 0).identifier[0] == make_move('Growl').identifier[0]
    returned, _ = step(copied, jnp.array([9, 0]))
    bank = returned.battle.team.moves[0, 0, 0]
    assert bank.identifier[0] == 144 and bank.current_pp[0] == bank.max_pp[0] - 1
