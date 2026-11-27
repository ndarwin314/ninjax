"""Behavioral validation for alternate protection routing and activation.

These tests intentionally assert working mechanics rather than accepting known
implementation errors. Expected retaliation follows the pinned Gen 9 Showdown
move conditions; no production functions are patched by the tests.
"""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.residuals import step_side_conditions
from ninjax.enums.items import ItemEnum
from ninjax.enums.field import Terrain, TerrainEnum
from ninjax.enums.moves import MoveResult, ProtectionKind
from ninjax.moves.routing import (
    _baneful_bunker_poison,
    _blocked_helper,
    _burning_bulwark_burn,
    _kings_shield_drop,
    _prepare_normal_hit,
    _prepare_accuracy,
    _resolve_and_hit,
    _silk_trap_drop,
    move_not_drawn_in,
    move_hits,
)
from ninjax.moves.status import _generic_status_move
from ninjax.moves.properties import effective_move
from ninjax.records.hits import HitOutcome
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.enums.abilities import AbilityEnum
from ninjax.side.updates import clear_volatile_status, update_active
from ninjax.side.hp_effects import take_damage_value
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def compiled_routing():
    """Trace the protection decision without compiling downstream damage."""
    return jax.jit(_prepare_normal_hit)


@pytest.fixture(scope='module')
def compiled_status():
    """Reuse status application across catalogue protection variants."""
    return jax.jit(_generic_status_move)


def test_unprotected_hit_routing(neutral_type, compiled_routing):
    """The default retaliation branch must accept the same operands as effects."""
    key, state = neutral_type
    result = compiled_routing(key, state, jnp.int32(0), make_move('Tackle'))
    assert result[3] == 0


def test_blocking_helper_traces(neutral_type):
    """Protection kind comparisons must support dynamic traced values."""
    _, state = neutral_type
    blocked = jax.jit(_blocked_helper)(state, make_move('Tackle'), jnp.int32(0))
    assert not blocked
    assert blocked.shape == ()


def test_unseen_fist_bypasses_damaging_contact(neutral_type):
    """Unseen Fist does not require the incoming move to be a status move."""
    _, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([AbilityEnum.UNSEEN_FIST])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        protection=v.protection.replace(
            protected=v.protection.protected.at[1, 0].set(True),
            protection_kind=v.protection.protection_kind.at[1, 0].set(ProtectionKind.PROTECT))))
    assert not _blocked_helper(state, make_move('Tackle'), 0)


@pytest.mark.parametrize('kind', [ProtectionKind.SILK_TRAP, ProtectionKind.BURNING_BULWARK,
                                ProtectionKind.KINGS_SHIELD])
def test_damage_only_guards_allow_status_moves(neutral_type, kind):
    """Status bypass is independent of the attacking Pokémon's ability."""
    _, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        protection=v.protection.replace(
            protected=v.protection.protected.at[1, 0].set(True),
            protection_kind=v.protection.protection_kind.at[1, 0].set(kind))))
    assert not _blocked_helper(state, make_move('Growl'), 0)


@pytest.mark.parametrize('kind,ready_expected', [
    (ProtectionKind.SILK_TRAP, False), (ProtectionKind.BURNING_BULWARK, False),
    (ProtectionKind.KINGS_SHIELD, False), (ProtectionKind.PROTECT, True),
    (ProtectionKind.SPIKY_SHIELD, True), (ProtectionKind.BANEFUL_BUNKER, True),
])
def test_damage_only_guard_does_not_force_status_accuracy_success(neutral_type, kind, ready_expected):
    """A guard that allows a status move cannot override that move's miss."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        protection=v.protection.replace(
            protected=v.protection.protected.at[1, 0].set(True),
            protection_kind=v.protection.protection_kind.at[1, 0].set(kind))))
    move = make_move('Growl').replace(accuracy=jnp.array([0.]))
    _, _, ready = jax.jit(_prepare_accuracy)(key, state, jnp.int32(0), move)
    assert bool(ready) == ready_expected


@pytest.mark.parametrize('kind,status,stat', [
    (ProtectionKind.BANEFUL_BUNKER, Status.POISON, 0),
    (ProtectionKind.BURNING_BULWARK, Status.BURN, 0),
    (ProtectionKind.SILK_TRAP, Status.NONE, StatEnum.SPEED),
    (ProtectionKind.KINGS_SHIELD, Status.NONE, StatEnum.ATTACK),
])
@pytest.mark.parametrize('ability', [AbilityEnum.NONE, AbilityEnum.LONG_REACH, AbilityEnum.UNSEEN_FIST])
def test_routed_retaliation(neutral_type, compiled_routing, kind, status, stat, ability):
    """Installed guards retaliate only on blocked contact and retain advanced keys."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([ability])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        protection=v.protection.replace(
            protected=v.protection.protected.at[1, 0].set(True),
            protection_kind=v.protection.protection_kind.at[1, 0].set(kind))))
    next_key, out, _, branch, _, _ = compiled_routing(key, state, jnp.int32(0), make_move('Tackle'))
    retaliates = ability == AbilityEnum.NONE
    assert out.active.status[0, 0] == (status if retaliates else Status.NONE)
    if stat:
        assert out.boosts.normal_boosts[0, stat] == (-1 if retaliates else 0)
    assert branch == (0 if ability == AbilityEnum.UNSEEN_FIST else 5)
    expected_key = jax.random.split(key)[0] if retaliates and status != Status.NONE else key
    assert jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(expected_key))


def test_selected_move_dispatch_traces(neutral_type):
    """The public dispatcher must compile even when the target is unprotected."""
    key, state = neutral_type
    state = equip_move(state, 'Tackle')
    _, out = jax.jit(move_hits)(key, state, jnp.int32(0), jnp.int32(0))
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]


@pytest.mark.parametrize('effect,stat', [(_silk_trap_drop, StatEnum.SPEED),
                                      (_kings_shield_drop, StatEnum.ATTACK)])
def test_retaliation_stat_drops_accept_inputs(neutral_type, effect, stat):
    """Stat retaliation applies exactly one opposing drop through the shared API."""
    key, state = neutral_type
    _, out = effect(key, state, attacker_index=0)
    assert out.boosts.normal_boosts[0, stat] == -1
    assert out.event_context.source == state.event_context.source


@pytest.mark.parametrize('effect,status', [(_baneful_bunker_poison, Status.POISON),
                                       (_burning_bulwark_burn, Status.BURN)])
def test_retaliation_status_and_key(neutral_type, effect, status):
    """Status retaliation affects the attacker and returns its advanced key."""
    key, state = neutral_type
    next_key, out = effect(key, state, attacker_index=0)
    assert out.active.status[0, 0] == status
    assert out.active.status[1, 0] == Status.NONE
    assert not jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))
    assert out.event_context.source == state.event_context.source


@pytest.mark.parametrize('name,kind', [('Baneful Bunker', ProtectionKind.BANEFUL_BUNKER),
    ('Burning Bulwark', ProtectionKind.BURNING_BULWARK), ('Silk Trap', ProtectionKind.SILK_TRAP),
    ('Endure', ProtectionKind.ENDURE)])
def test_alternate_protect_activation(neutral_type, compiled_status, name, kind):
    """A catalogue variant installs its kind, with Endure leaving hits open."""
    key, state = neutral_type
    move = make_move(name)
    assert move.implemented[0]
    _, out = compiled_status(key, state, jnp.int32(0), move, 0, 0)
    assert bool(out.volatile_status.protection.protected[0, 0]) == (kind != ProtectionKind.ENDURE)
    assert out.volatile_status.protection.protection_kind[0, 0] == kind
    assert out.volatile_status.protection.protect_chain[0, 0] == 1


def test_protection_fails_when_no_action_follows(neutral_type, compiled_status):
    """An otherwise guaranteed guard fails when used as the last action."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(
        queued_move=v.queued_move.replace(last_action=jnp.ones((2, 1), bool))))
    _, out = compiled_status(key, state, jnp.int32(0), make_move('Protect'), 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.FAILED
    assert not out.volatile_status.protection.protected[0, 0]


@pytest.mark.jit
def test_scheduler_rejects_the_later_protect(neutral_type):
    """Only the first Protect succeeds when both sides choose it in one turn."""
    key, state = neutral_type
    state = equip_move(equip_move(state, 'Protect'), 'Protect', 1)
    env = Battle(state.team)
    episode, _ = env.reset(key)
    episode, _ = env.step(episode, jnp.zeros(2, jnp.int32))
    history = episode.battle.volatile_status.move_history
    assert jnp.array_equal(jnp.sort(history.previous_move_result[:, 0]),
                           jnp.array([MoveResult.FAILED, MoveResult.SUCCEEDED]))


@pytest.mark.parametrize('name,blocked,allowed', [
    ('Quick Guard', 'Quick Attack', 'Earthquake'),
    ('Wide Guard', 'Earthquake', 'Quick Attack'),
    ('Wide Guard', 'Growl', 'Tackle'),
])
def test_side_guards_filter_by_priority_or_spread(neutral_type, compiled_status, name, blocked, allowed):
    """Side guards reliably refresh stall and filter the appropriate targets."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protection=v.protection.replace(
        protect_chain=v.protection.protect_chain.at[1, 0].set(6))))
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move(name), 0, 0)
    assert _blocked_helper(guarded, make_move(blocked), 0)
    assert not _blocked_helper(guarded, make_move(allowed), 0)
    assert _blocked_helper(clear_volatile_status(guarded, 1), make_move(blocked), 0)
    _, expired = step_side_conditions(key, guarded)
    assert not _blocked_helper(expired, make_move(blocked), 0)
    assert expired.volatile_status.protection.protect_chain[1, 0] == 6
    _, expired = step_side_conditions(key, expired)
    assert expired.volatile_status.protection.protect_chain[1, 0] == 0


def test_expanding_force_uses_hit_time_spread_target(neutral_type, compiled_status):
    """Wide Guard reads the spread target that Psychic Terrain can give a move."""
    key, state = neutral_type
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move('Wide Guard'), 0, 0)
    move = make_move('Expanding Force')
    assert not _blocked_helper(guarded, effective_move(guarded, 0, move), 0)
    terrain = guarded.replace(terrain=Terrain(TerrainEnum.PSYCHIC, 5))
    assert _blocked_helper(terrain, effective_move(terrain, 0, move), 0)
    flying = update_active(terrain, 0, terrain.active[0].replace(type_list=jnp.array([Type.FLYING, Type.NONE])))
    assert not _blocked_helper(flying, effective_move(flying, 0, move), 0)


def test_priority_block_precedes_contact_retaliation(neutral_type, compiled_status, compiled_routing):
    """Psychic Terrain stops Quick Attack before it can trigger Spiky Shield."""
    key, state = neutral_type
    state = state.replace(terrain=Terrain(TerrainEnum.PSYCHIC, 5))
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move('Spiky Shield'), 0, 0)
    _, out, _, _, _, _ = compiled_routing(key, guarded, jnp.int32(0), make_move('Quick Attack'))
    assert out.hit_results[0].outcome == HitOutcome.BLOCKED
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]


@pytest.mark.parametrize('ability,item', [(AbilityEnum.NONE, ItemEnum.FOCUS_SASH),
    (AbilityEnum.STURDY, ItemEnum.NONE)])
def test_endure_precedes_other_survival_checks(neutral_type, compiled_status, ability, item):
    """Endure keeps the holder at one through consecutive attacks this turn."""
    key, state = neutral_type
    mon = state.active[1].replace(ability=jnp.array([ability]), item=jnp.array([item]))
    state = update_active(state, 1, mon)
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move('Endure'), 0, 0)
    for _ in range(2):
        guarded = take_damage_value(guarded, 1, jnp.array([1000]), True, source=0)
        assert guarded.active.current_hp[1, 0] == 1
        assert guarded.active.item[1, 0] == item
    indirect = take_damage_value(guarded, 1, jnp.array([1]), False)
    assert indirect.active.current_hp[1, 0] == 0
    _, expired = step_side_conditions(key, guarded)
    assert take_damage_value(expired, 1, jnp.array([1]), True).active.current_hp[1, 0] == 0


def test_mat_block_first_action_without_stall(neutral_type, compiled_status):
    """Historical Mat Block protects the side from damage on the first action."""
    key, state = neutral_type
    mat_block = make_move('Protect').replace(identifier=jnp.array([561]))
    _, guarded = compiled_status(key, state, jnp.int32(1), mat_block, 0, 0)
    assert guarded.side_protection.mat_block[1, 0]
    assert _blocked_helper(guarded, make_move('Tackle'), 0)
    assert not _blocked_helper(guarded, make_move('Growl'), 0)
    assert guarded.volatile_status.protection.protect_chain[1, 0] == 0
    v = state.volatile_status
    late = state.replace(volatile_status=v.replace(move_history=v.move_history.replace(
        active_move_actions=v.move_history.active_move_actions.at[1, 0].set(2))))
    _, failed = compiled_status(key, late, jnp.int32(1), mat_block, 0, 0)
    assert failed.hit_results[0].outcome == HitOutcome.FAILED


@pytest.fixture(scope='module')
def compiled_hit():
    """Compile a resolved hit shared by the protection-breaking checks."""
    return jax.jit(_resolve_and_hit)


def test_feint_breaks_guard_but_preserves_endure(neutral_type, compiled_status, compiled_hit):
    """Feint cancels side and personal shields on a hit that reaches them."""
    key, state = neutral_type
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move('Protect'), 0, 0)
    _, guarded = compiled_status(key, guarded, jnp.int32(1), make_move('Wide Guard'), 0, 0)
    feint = make_move('Feint')
    _, broken = compiled_hit(key, guarded, jnp.int32(0), feint)
    assert broken.hit_results[0].outcome == HitOutcome.HIT
    assert not broken.volatile_status.protection.protected[1, 0]
    assert not broken.side_protection.wide_guard[1, 0]
    assert broken.volatile_status.protection.protect_chain[1, 0] == 0
    ghost = update_active(guarded, 1, guarded.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE])))
    _, immune = compiled_hit(key, ghost, jnp.int32(0), feint)
    assert immune.hit_results[0].outcome == HitOutcome.IMMUNE
    assert immune.volatile_status.protection.protected[1, 0]
    _, enduring = compiled_status(key, state, jnp.int32(1), make_move('Endure'), 0, 0)
    enduring = update_active(enduring, 1, enduring.active[1].replace(current_hp=jnp.ones(1, jnp.int32)))
    _, out = compiled_hit(key, enduring, jnp.int32(0), feint)
    assert out.active.current_hp[1, 0] == 1
    assert out.volatile_status.protection.protection_kind[1, 0] == ProtectionKind.ENDURE


def test_struggle_hits_the_protection_branch(neutral_type, compiled_status):
    """A synthetic Struggle correctly strikes and triggers Spiky Shield."""
    key, state = neutral_type
    _, guarded = compiled_status(key, state, jnp.int32(1), make_move('Spiky Shield'), 0, 0)
    guarded = guarded.replace(event_context=guarded.event_context.replace(move_id=jnp.int32(-1)))
    _, out = jax.jit(move_not_drawn_in)(key, guarded, 0, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.PROTECTED
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.current_hp[0, 0] < state.active.current_hp[0, 0]
