"""Behavioral validation for alternate protection routing and activation.

These tests intentionally assert working mechanics rather than accepting known
implementation errors. Expected retaliation follows the pinned Gen 9 Showdown
move conditions; no production functions are patched by the tests.
"""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.moves import ProtectionKind
from ninjax.moves.routing import (
    _baneful_bunker_poison,
    _blocked_helper,
    _burning_bulwark_burn,
    _kings_shield_drop,
    _prepare_normal_hit,
    _prepare_accuracy,
    _prepare_struggle_hit,
    _silk_trap_drop,
    move_hits,
)
from ninjax.moves.status import _generic_status_move
from ninjax.enums.pokemon import StatEnum, Status
from ninjax.enums.abilities import AbilityEnum
from ninjax.side.updates import update_active
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


def test_struggle_preparation_matches_normal_contract(neutral_type):
    """Both outer cond branches must return the advanced key plus five operands."""
    key, state = neutral_type
    assert len(_prepare_struggle_hit(key, state, 0, make_move('Tackle'))) == 6


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
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True),
        protection_kind=v.protection_kind.at[1, 0].set(ProtectionKind.PROTECT)))
    assert not _blocked_helper(state, make_move('Tackle'), 0)


@pytest.mark.parametrize('kind', [ProtectionKind.SILK_TRAP, ProtectionKind.BURNING_BULWARK,
                                ProtectionKind.KINGS_SHIELD])
def test_damage_only_guards_allow_status_moves(neutral_type, kind):
    """Status bypass is independent of the attacking Pokémon's ability."""
    _, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True),
        protection_kind=v.protection_kind.at[1, 0].set(kind)))
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
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True),
        protection_kind=v.protection_kind.at[1, 0].set(kind)))
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
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True),
        protection_kind=v.protection_kind.at[1, 0].set(kind)))
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
    assert out.event_source == state.event_source


@pytest.mark.parametrize('effect,status', [(_baneful_bunker_poison, Status.POISON),
                                       (_burning_bulwark_burn, Status.BURN)])
def test_retaliation_status_and_key(neutral_type, effect, status):
    """Status retaliation affects the attacker and returns its advanced key."""
    key, state = neutral_type
    next_key, out = effect(key, state, attacker_index=0)
    assert out.active.status[0, 0] == status
    assert out.active.status[1, 0] == Status.NONE
    assert not jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))
    assert out.event_source == state.event_source


@pytest.mark.parametrize('name,kind', [('Baneful Bunker', ProtectionKind.BANEFUL_BUNKER),
    ('Burning Bulwark', ProtectionKind.BURNING_BULWARK), ('Silk Trap', ProtectionKind.SILK_TRAP)])
def test_alternate_protect_activation(neutral_type, compiled_status, name, kind):
    """Using a catalogue variant must set its protected flag and correct kind."""
    key, state = neutral_type
    _, out = compiled_status(key, state, jnp.int32(0), make_move(name), 0, 0)
    assert out.volatile_status.protected[0, 0]
    assert out.volatile_status.protection_kind[0, 0] == kind
    assert out.volatile_status.protect_chain[0, 0] == 1
