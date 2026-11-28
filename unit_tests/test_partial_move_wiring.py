"""Check previously unreachable rules through real catalogue moves."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_move
from ninjax.calculations.damage import compute_base_power, resolve_hit_base_power
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum as T, Weather, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import HitRule, SemiInvulnerability
from ninjax.enums.pokemon import Status, Type
from ninjax.moves.finalization import finish_attempt
from ninjax.moves.properties import effective_move
from ninjax.moves.routing import move_not_drawn_in
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def hit():
    """Use one compiled ordinary accuracy and hit pipeline."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('name', ['Flail', 'Reversal'])
def test_low_hp_table_from_catalogue(name):
    """Both moves select all six breakpoints through the shared dispatcher."""
    move = make_move(name)
    assert move.implemented[0]
    hp = jnp.array([1, 19, 20, 49, 50, 99, 100, 169, 170, 329, 330, 480])
    powers = jax.jit(resolve_hit_base_power)(move.base_power[0], move.power_rule[0], user_hp=hp, user_max_hp=480)
    assert list(powers) == [200, 200, 150, 150, 100, 100, 80, 80, 40, 40, 20, 20]


@pytest.mark.parametrize('name', ['Flail', 'Reversal'])
def test_low_hp_power_reaches_damage(neutral_type, hit, name):
    """The hit pipeline reads current HP for the ordinary selected move."""
    key, state = neutral_type
    state = equip_move(state, name)
    _, full = hit(key, state, 0, 0, 0)
    injured = update_active_fields(state, 0, current_hp=jnp.array([1]))
    _, low = hit(key, injured, 0, 0, 0)
    assert low.hit_results[0].damage > full.hit_results[0].damage > 0


@pytest.mark.parametrize('name, facts, power', [
    ('Grass Knot', {'target_weight': 250}, 60),
    ('Heat Crash', {'user_weight': 400, 'target_weight': 100}, 100),
    ('Power Trip', {'positive_stages': 4}, 100),
    ('Temper Flare', {'previous_failed': True}, 150),
])
def test_shared_rules_are_available_at_the_hit_boundary(name, facts, power):
    """Catalogue entries carry live weight or stage rules with usable PP."""
    move = make_move(name)
    assert move.implemented[0] and move.current_pp[0] > 0
    assert resolve_hit_base_power(move.base_power[0], move.power_rule[0], **facts) == power


@pytest.mark.parametrize('terrain, move_type, power', [
    (T.NONE, Type.NORMAL, 50), (T.ELECTRIC, Type.ELECTRIC, 100),
    (T.GRASSY, Type.GRASS, 100), (T.MISTY, Type.FAIRY, 100),
    (T.PSYCHIC, Type.PSYCHIC, 100),
])
def test_terrain_pulse_prepares_type_and_power(neutral_type, terrain, move_type, power):
    """Resolve the terrain-dependent fields before the move reaches the hit path."""
    _, state = neutral_type
    move = make_move('Terrain Pulse')
    assert move.implemented[0]
    prepared = effective_move(state.replace(terrain=Terrain(terrain, 5)), 0, move)
    assert prepared.type[0] == move_type and prepared.base_power[0] == power


def test_terrain_pulse_balloon_and_suppression(neutral_type):
    """An active Balloon prevents conversion, and item suppression grounds the user."""
    _, state = neutral_type
    state = update_active_fields(state, 0, item=jnp.array([I.AIR_BALLOON]))
    state = state.replace(terrain=Terrain(T.ELECTRIC, 5))
    move = make_move('Terrain Pulse')
    airborne = effective_move(state, 0, move)
    assert airborne.type[0] == Type.NORMAL and airborne.base_power[0] == 50
    grounded = state.replace(magic_room_duration=2)
    prepared = effective_move(grounded, 0, move)
    assert prepared.type[0] == Type.ELECTRIC and prepared.base_power[0] == 100
    power = compute_base_power(grounded.active[0], grounded.active[1], prepared,
                               grounded.terrain, item_suppressed=True)
    assert power[0] == 130


def test_terrain_pulse_type_reaches_immunity_and_damage(neutral_type, hit):
    """A Ghost target is immune to Normal but takes the converted Electric hit."""
    key, state = neutral_type
    state = equip_move(state, 'Terrain Pulse')
    state = update_active_fields(state, 1, type_list=jnp.array([Type.GHOST, Type.NONE]))
    _, clear = hit(key, state, 0, 0, 0)
    _, converted = hit(key, state.replace(terrain=Terrain(T.ELECTRIC, 5)), 0, 0, 0)
    assert clear.hit_results[0].outcome == HitOutcome.IMMUNE
    assert converted.hit_results[0].damage > 0


def test_seismic_toss_uses_level_and_respects_immunity(neutral_type, hit):
    """The fixed hit uses level and still runs type immunity."""
    key, state = neutral_type
    state = equip_move(state, 'Seismic Toss')
    _, out = hit(key, state, 0, 0, 0)
    assert out.hit_results[0].damage == state.active[0].level[0]
    ghost = update_active_fields(state, 1, type_list=jnp.array([Type.GHOST, Type.NONE]))
    _, out = hit(key, ghost, 0, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE


@pytest.mark.parametrize('name', ['Gust', 'Twister'])
def test_airborne_hits_use_existing_damage_hooks(neutral_type, hit, name):
    """Fly's damage multiplier and Bounce's power multiplier use separate steps."""
    key, state = neutral_type
    state = equip_move(state, name)
    _, normal = hit(key, state, 0, 0, 0)
    volatile = state.volatile_status
    for charge in (19, 340):
        continuation = volatile.continuation.replace(
            charging_move_id=volatile.continuation.charging_move_id.at[1, 0].set(charge),
            semi_invulnerability=volatile.continuation.semi_invulnerability.at[1, 0].set(SemiInvulnerability.AIRBORNE))
        lifted = state.replace(volatile_status=volatile.replace(continuation=continuation))
        _, out = hit(key, lifted, 0, 0, 0)
        assert out.hit_results[0].damage > normal.hit_results[0].damage


def test_dream_eater_requires_sleep_or_comatose(neutral_type, hit):
    """The sleep gate precedes damage, and accepted targets share drain healing."""
    key, state = neutral_type
    state = equip_move(state, 'Dream Eater')
    state = update_active_fields(state, 0, current_hp=jnp.array([100]))
    _, awake = hit(key, state, 0, 0, 0)
    assert awake.hit_results[0].outcome == HitOutcome.FAILED
    for opponent in (
        update_active_fields(state, 1, status=jnp.array([Status.SLEEP])),
        update_active_fields(state, 1, ability=jnp.array([A.COMATOSE])),
    ):
        _, out = hit(key, opponent, 0, 0, 0)
        assert out.hit_results[0].damage > 0 and out.active.current_hp[0, 0] > 100


def test_life_dew_heals_only_the_singles_user(neutral_type, hit):
    """Life Dew restores a quarter of the user's HP and fails at full HP."""
    key, state = neutral_type
    state = equip_move(state, 'Life Dew')
    _, full = hit(key, state, 0, 0, 0)
    assert full.hit_results[0].outcome == HitOutcome.FAILED
    hurt = update_active_fields(state, 0, current_hp=jnp.array([100]))
    _, out = hit(key, hurt, 0, 0, 0)
    assert out.active.current_hp[0, 0] == 100 + state.active.max_hp[0] // 4
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


@pytest.mark.parametrize('name, defender_type, reckless_power', [
    ('Supercell Slam', Type.GROUND, 120), ('Axe Kick', Type.GHOST, 144),
])
def test_crash_moves_share_finalization(neutral_type, hit, name, defender_type, reckless_power):
    """Immunity records a crash and Reckless boosts these available moves."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active_fields(state, 1, type_list=jnp.array([defender_type, Type.NONE]))
    move = make_move(name)
    # Deterministically exercise the immunity branch for Axe Kick.
    moves = state.team.moves
    state = state.replace(team=state.team.replace(moves=moves.replace(
        accuracy=moves.accuracy.at[0, 0, 0, 0].set(1.))))
    _, failed = hit(key, state, 0, 0, 0)
    assert failed.hit_results[0].outcome == HitOutcome.IMMUNE
    out = jax.jit(finish_attempt)(failed, 0, move)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - state.active.max_hp[0] // 2
    reckless = state.active[0].replace(ability=jnp.array([A.RECKLESS]))
    assert compute_base_power(reckless, state.active[1], move, state.terrain)[0] == reckless_power


def test_growth_and_gravity_modifier(neutral_type, hit):
    """Sunny Growth uses real boosts and Gravity boosts Grav Apple's power."""
    key, state = neutral_type
    growth = equip_move(state.replace(weather=Weather(W.SUN, 5)), 'Growth')
    _, boosted = hit(key, growth, 0, 0, 0)
    assert list(boosted.boosts.normal_boosts[0, 1:4]) == [2, 0, 2]
    move = make_move('Grav Apple')
    assert move.implemented[0]
    assert compute_base_power(state.active[0], state.active[1], move, state.terrain, gravity_active=True)[0] == 120


def test_existing_screen_and_terrain_hooks(neutral_type, hit):
    """Psychic Fangs clears screens and Steel Roller needs terrain to remove it."""
    key, state = neutral_type
    conditions = state.side_conditions
    guarded = state.replace(side_conditions=conditions.replace(reflect=conditions.reflect.at[1, 0].set(3)))
    _, out = hit(key, equip_move(guarded, 'Psychic Fangs'), 0, 0, 0)
    assert out.hit_results[0].damage > 0 and out.side_conditions.reflect[1, 0] == 0
    roller = equip_move(state, 'Steel Roller')
    _, clear = hit(key, roller, 0, 0, 0)
    assert clear.hit_results[0].outcome == HitOutcome.FAILED
    _, field = hit(key, roller.replace(terrain=Terrain(T.ELECTRIC, 5)), 0, 0, 0)
    assert field.hit_results[0].damage > 0 and field.terrain.terrain == T.NONE


@pytest.mark.parametrize('name', ['Hyper Drill', 'Mighty Cleave'])
def test_protection_bypass_leaves_the_guard_active(neutral_type, hit, name):
    """Both attacks hit through Protect while its flag stays set."""
    key, state = neutral_type
    volatile = state.volatile_status
    protection = volatile.protection.replace(protected=volatile.protection.protected.at[1, 0].set(True))
    state = state.replace(volatile_status=volatile.replace(protection=protection))
    _, out = hit(key, equip_move(state, name), 0, 0, 0)
    assert out.hit_results[0].damage > 0 and out.volatile_status.protection.protected[1, 0]


def test_mortal_spin_clears_own_hazards_and_poison(neutral_type, hit):
    """Mortal Spin clears the user's hazards and attempts to poison the target."""
    key, state = neutral_type
    hazards = state.hazards.replace(spikes=state.hazards.spikes.at[0, 0].set(2))
    _, out = hit(key, equip_move(state.replace(hazards=hazards), 'Mortal Spin'), 0, 0, 0)
    assert out.hazards.spikes[0, 0] == 0
    assert out.active.status[1, 0] == Status.POISON


def test_snore_checks_the_users_sleep(neutral_type, hit):
    """Snore fails awake and can reach a target while its user sleeps."""
    key, state = neutral_type
    state = equip_move(state, 'Snore')
    _, awake = hit(key, state, 0, 0, 0)
    _, asleep = hit(key, update_active_fields(state, 0, status=jnp.array([Status.SLEEP])), 0, 0, 0)
    assert awake.hit_results[0].outcome == HitOutcome.FAILED
    assert asleep.hit_results[0].damage > 0


def test_projected_generic_move_effects():
    """Review previously disabled multihit, major status and sure-hit metadata."""
    moves = [make_move(name) for name in ('Bone Rush', 'Freezing Glare', 'Inferno', 'Smart Strike')]
    assert all(move.implemented[0] for move in moves)
    assert moves[0].hit_rule[0] == HitRule.TWO_TO_FIVE
    assert moves[1].secondaries.status[0] == Status.FREEZE
    assert moves[2].secondaries.status[0] == Status.BURN and moves[2].secondaries.probabilities[0] == 1
    assert moves[3].always_hit[0]


@pytest.mark.jit
def test_entry_and_self_faint_checks_through_step_move(neutral_type):
    """Exercise power, first-turn and Damp gates through the selected-move entry."""
    key, state = neutral_type
    step = jax.jit(step_move)
    for name in ('Flail', 'Reversal'):
        full = equip_move(state, name)
        injured = update_active_fields(full, 0, current_hp=jnp.array([1]))
        _, at_full = step(key, full, 0, 0, False)
        _, at_low = step(key, injured, 0, 0, False)
        assert at_low.hit_results[0].damage > at_full.hit_results[0].damage > 0
    first = equip_move(state, 'First Impression')
    _, allowed = step(key, first, 0, 0, False)
    assert allowed.hit_results[0].damage > 0
    _, too_late = step(key, allowed, 0, 0, False)
    assert too_late.hit_results[0].outcome == HitOutcome.FAILED
    for name in ('Self-Destruct', 'Misty Explosion'):
        armed = equip_move(state, name)
        _, detonated = step(key, armed, 0, 0, False)
        assert detonated.active.current_hp[0, 0] == 0
        assert detonated.hit_results[0].damage > 0
        damp = update_active_fields(armed, 1, ability=jnp.array([A.DAMP]))
        _, prevented = step(key, damp, 0, 0, False)
        assert prevented.hit_results[0].outcome == HitOutcome.BLOCKED
        assert jnp.array_equal(prevented.active.current_hp, state.active.current_hp)
