"""Behavior checks for the seventeen moves wired from expanded metadata."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.residuals import step_side_conditions
from ninjax.battle.switching import prepare_forced_switch
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum as T, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import ProtectionKind
from ninjax.moves.execution import _prepare_move_use
from ninjax.moves.routing import move_hits
from ninjax.moves.status import _generic_status_move
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def hit_dispatch():
    """Reuse one compiled dispatcher across target and move variants."""
    return jax.jit(move_hits)


@pytest.fixture(scope='module')
def status_dispatch():
    """Compile the shared status primitives once for compound failure checks."""
    return jax.jit(_generic_status_move)


@pytest.mark.parametrize('name,weather,terrain', [
    ('Sandstorm', W.SANDSTORM, T.NONE), ('Snowscape', W.SNOW, T.NONE),
    ('Electric Terrain', W.NONE, T.ELECTRIC), ('Psychic Terrain', W.NONE, T.PSYCHIC),
    ('Misty Terrain', W.NONE, T.MISTY),
])
def test_field_moves_ignore_opponent_hit_gates(neutral_type, hit_dispatch, name, weather, terrain):
    """Field setters work through opposing protection, Substitute and Good as Gold."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.GOOD_AS_GOLD])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        protection=v.protection.replace(protected=v.protection.protected.at[1, 0].set(True)),
        substitute_hp=v.substitute_hp.at[1, 0].set(50)))
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert out.weather.weather == weather and out.terrain.terrain == terrain
    assert (out.weather.duration if weather != W.NONE else out.terrain.duration) == 5
    _, repeated = hit_dispatch(key, out, jnp.int32(0), jnp.int32(0))
    assert repeated.hit_results[0].outcome == HitOutcome.FAILED


def test_reflect_duration_expiry_and_damage(neutral_type, hit_dispatch):
    """Reflect uses Light Clay, expires normally and halves ordinary physical hits."""
    key, state = neutral_type
    state = equip_move(state, 'Reflect')
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.LIGHT_CLAY])))
    _, screen = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert screen.side_conditions.reflect[0, 0] == 8
    _, expired = step_side_conditions(key, screen.replace(side_conditions=screen.side_conditions.replace(reflect=screen.side_conditions.reflect.at[0, 0].set(1))))
    assert expired.side_conditions.reflect[0, 0] == 0
    # Battle Armor removes critical-hit variation; use identical random rolls.
    screen = update_active(screen, 0, screen.active[0].replace(ability=jnp.array([A.BATTLE_ARMOR])))
    screen = equip_move(screen, 'Tackle', 1)
    _, reduced = hit_dispatch(key, screen, jnp.int32(1), jnp.int32(0))
    _, full = hit_dispatch(key, screen.replace(side_conditions=screen.side_conditions.replace(reflect=jnp.zeros_like(screen.side_conditions.reflect))), jnp.int32(1), jnp.int32(0))
    assert 0 < reduced.hit_results[0].damage < full.hit_results[0].damage


@pytest.mark.parametrize('name', ['Psystrike', 'Secret Sword'])
def test_defensive_stat_overrides(neutral_type, hit_dispatch, name):
    """Special attacks using Defense ignore changes to the target's Special Defense."""
    key, state = neutral_type
    state = equip_move(state, name)
    _, first = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 4].set(6)))
    _, second = hit_dispatch(key, boosted, jnp.int32(0), jnp.int32(0))
    assert first.hit_results[0].damage == second.hit_results[0].damage
    assert first.hit_results[0].damage > 0


@pytest.mark.parametrize('name', ['Confuse Ray', 'Supersonic', 'Sweet Kiss', 'Teeter Dance', 'Flatter', 'Swagger'])
def test_primary_confusion_and_boosts(neutral_type, hit_dispatch, name):
    """All six primary confusion moves affect the opponent in singles."""
    key, state = neutral_type
    state = equip_move(state, name)
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert 2 <= out.volatile_status.confusion_turns[1, 0] <= 5
    assert out.volatile_status.confusion_turns[0, 0] == 0
    if name in ('Flatter', 'Swagger'):
        stat, amount = (3, 1) if name == 'Flatter' else (1, 2)
        assert out.boosts.normal_boosts[1, stat] == amount


@pytest.mark.parametrize('name', ['Flatter', 'Swagger'])
@pytest.mark.parametrize('cap,immune', [(False, True), (True, False), (True, True)])
def test_compound_confusion_success(neutral_type, status_dispatch, name, cap, immune):
    """One successful component is sufficient; failure requires both to fail."""
    key, state = neutral_type
    stat = 3 if name == 'Flatter' else 1
    if cap:
        state = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, stat].set(6)))
    if immune:
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.OWN_TEMPO])))
    _, out = status_dispatch(key, state, jnp.int32(0), make_move(name), 0, 0)
    assert (out.hit_results[0].outcome == HitOutcome.FAILED) == (cap and immune)


@pytest.mark.parametrize('case', ['own_tempo', 'misty', 'substitute', 'lum', 'protected'])
def test_confusion_prevention_and_cure(neutral_type, hit_dispatch, case):
    """Primary confusion respects target gates, immunities and immediate berries."""
    key, state = neutral_type
    state = equip_move(state, 'Confuse Ray')
    if case == 'own_tempo':
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.OWN_TEMPO])))
    if case == 'misty':
        state = state.replace(terrain=Terrain(T.MISTY, 5))
    if case == 'lum':
        state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.LUM_BERRY])))
    v = state.volatile_status
    if case == 'substitute':
        state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(50)))
    if case == 'protected':
        state = state.replace(volatile_status=v.replace(protection=v.protection.replace(protected=v.protection.protected.at[1, 0].set(True))))
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert out.volatile_status.confusion_turns[1, 0] == 0
    if case == 'lum':
        assert out.active.item[1, 0] == I.NONE
        assert out.hit_results[0].outcome == HitOutcome.HIT


def test_detect_blocks_and_shares_chain(neutral_type, hit_dispatch):
    """Detect protects normally and retains the consecutive protection counter."""
    key, state = neutral_type
    state = equip_move(equip_move(state, 'Detect'), 'Tackle', 1)
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert out.volatile_status.protection.protected[0, 0]
    assert out.volatile_status.protection.protection_kind[0, 0] == ProtectionKind.PROTECT
    assert out.volatile_status.protection.protect_chain[0, 0] == 1
    # Starting Detect must retain a chain built by Protect or Spiky Shield.
    chained = state.replace(volatile_status=state.volatile_status.replace(protection=state.volatile_status.protection.replace(protect_chain=state.volatile_status.protection.protect_chain.at[0, 0].set(4))))
    _, prepared = _prepare_move_use(key, chained, 0, 0)
    assert prepared.volatile_status.protection.protect_chain[0, 0] == 4
    _, blocked = hit_dispatch(key, out, jnp.int32(1), jnp.int32(0))
    assert blocked.hit_results[0].outcome == HitOutcome.PROTECTED
    assert blocked.active.current_hp[0, 0] == state.active.current_hp[0, 0]


@pytest.mark.parametrize('ability,expected', [(A.NONE, True), (A.WATER_ABSORB, False)])
def test_flip_turn_pivot_only_after_hit(neutral_type, hit_dispatch, ability, expected):
    """Flip Turn requests the existing pivot continuation unless Water is absorbed."""
    key, state = neutral_type
    state = equip_move(state, 'Flip Turn')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert bool(out.switch_requested[0]) == expected


def test_flip_turn_protection_blocks_pivot(neutral_type, hit_dispatch):
    """A protected target prevents both damage and the pivot request."""
    key, state = neutral_type
    state = equip_move(state, 'Flip Turn')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protection=v.protection.replace(protected=v.protection.protected.at[1, 0].set(True))))
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert not out.switch_requested[0]
    assert out.hit_results[0].outcome == HitOutcome.PROTECTED


@pytest.mark.parametrize('ability,switches', [(A.NONE, True), (A.WIND_RIDER, True),
    (A.SOUNDPROOF, False), (A.SUCTION_CUPS, False), (A.GUARD_DOG, False)])
def test_roar_uses_sound_instead_of_wind(neutral_type, hit_dispatch, ability, switches):
    """Roar drags through Wind Rider, while sound and drag immunities stop it."""
    key, state = neutral_type
    state = equip_move(state, 'Roar')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    state = state.replace(roster=state.roster.at[1].set(jnp.array([True, False, True, False, False, False])))
    _, out = hit_dispatch(key, state, jnp.int32(0), jnp.int32(0))
    assert out.active_index[1, 0] == (2 if switches else 0)
    assert out.boosts.normal_boosts[1, 1] == 0


def test_roar_direct_preparation_records_sound_immunity(neutral_type):
    """Direct switch preparation also respects Soundproof without drawing a key."""
    key, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.SOUNDPROOF])))
    next_key, out, request = prepare_forced_switch(key, state, 0, wind=False)
    assert not request.enabled
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE
    assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))
