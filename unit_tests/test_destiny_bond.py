"""Destiny Bond's lifetime and cause-aware faint resolution."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle import step_move
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Status
from ninjax.events import Cause, EventBuffer
from ninjax.game_logic import do_status_move
from ninjax.prototype import make_move
from ninjax.side import clear_volatile_status, resolve_faints, take_damage_value, update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def bonded(key, state, player=1):
    """Activate Destiny Bond on a selected Pokémon."""
    return do_status_move(key, state, player, make_move('Destiny Bond'), 0, 0)[1]


@pytest.mark.parametrize('ability,item', [(A.STURDY, I.FOCUS_SASH), (A.MAGIC_GUARD, I.NONE), (A.INNARDS_OUT, I.NONE)])
def test_direct_faint_bypasses_damage_protections(neutral_type, ability, item):
    """Bond faints the actual attacker without checking damage protections."""
    key, state = neutral_type
    state = bonded(key, state)
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([ability]), item=jnp.array([item])))
    state = state.replace(roster=jnp.zeros((2, 6), bool).at[:, 0].set(True))
    pending = take_damage_value(state, 1, jnp.array([9999]), True, source=0)
    assert pending.active.current_hp[0, 0] > 0
    # Mechanics must not depend on event-buffer room.
    pending = pending.replace(events=EventBuffer.empty().replace(count=jnp.int32(128)))
    out = resolve_faints(pending)
    assert jnp.all(out.active.current_hp == 0)
    assert not jnp.any(out.faint_pending)
    assert out.active.item[0, 0] == item
    assert out.winner == 0  # Last to faint wins the simultaneous final KO.


@pytest.mark.parametrize('source,attack,cause', [(-1, False, Cause.WEATHER), (0, False, Cause.RECOIL), (0, False, Cause.MOVE), (1, True, Cause.MOVE)])
def test_indirect_and_self_damage_do_not_trigger(neutral_type, source, attack, cause):
    """Indirect causes and self-inflicted move damage cannot activate the bond."""
    key, state = neutral_type
    state = bonded(key, state).replace(event_cause=jnp.int32(cause))
    out = resolve_faints(take_damage_value(state, 1, jnp.array([9999]), attack, source=source))
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert not out.volatile_status.destiny_bond[1, 0]


def test_consecutive_attempts_and_interruption(neutral_type):
    """Consecutive use fails, a third use succeeds, and flinch ends the bond."""
    key, state = neutral_type
    state = equip_move(state, 'Destiny Bond', player=0)
    key, first = step_move(key, state, 0, 0, False)
    assert first.volatile_status.destiny_bond[0, 0]
    key, second = step_move(key, first, 0, 0, False)
    assert not second.volatile_status.destiny_bond[0, 0]
    key, third = step_move(key, second, 0, 0, False)
    assert third.volatile_status.destiny_bond[0, 0]
    assert third.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 3
    v = third.volatile_status
    flinched = third.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True)))
    _, out = step_move(key, flinched, 0, 0, False)
    assert not out.volatile_status.destiny_bond[0, 0]
    assert out.active.moves.current_pp[0, 0, 0] == third.active.moves.current_pp[0, 0, 0]
    assert not clear_volatile_status(third, 0).volatile_status.destiny_bond[0, 0]


def test_real_attack_and_non_bond_action_expiry(neutral_type):
    """A normal attack resolves Bond at its boundary; using another move ends it."""
    key, state = neutral_type
    state = bonded(key, state)
    state = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([1])))
    state = equip_move(state, 'Dragon Pulse')
    _, out = step_move(key, state, 0, 0, False)
    assert jnp.all(out.active.current_hp == 0)
    state = equip_move(bonded(key, state, 0), 'Protect')
    _, out = step_move(key, state, 0, 0, False)
    assert not out.volatile_status.destiny_bond[0, 0]


def test_faint_resolution_trace(neutral_type):
    """Faint records and direct retaliation trace under batching without compilation."""
    key, state = neutral_type
    state = bonded(key, state)
    pending = take_damage_value(state, 1, jnp.array([9999]), True, source=0)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), pending)
    shape = jax.eval_shape(jax.vmap(lambda s: resolve_faints(s).active.current_hp), batch)
    assert shape.shape == (2, 2, 1)


def test_dead_or_replaced_source_is_not_retaliated_against(neutral_type):
    """Deferred resolution only affects the surviving original source Pokémon."""
    key, state = neutral_type
    state = bonded(key, state)
    pending = take_damage_value(state, 1, jnp.array([9999]), True, source=0)
    dead_source = take_damage_value(pending, 0, jnp.array([9999]), False)
    out = resolve_faints(dead_source)
    assert jnp.all(out.active.current_hp == 0)
    assert out.winner == dead_source.winner
    replaced = pending.replace(active_index=pending.active_index.at[0, 0].set(1))
    out = resolve_faints(replaced)
    assert out.active.current_hp[0, 0] == replaced.active.current_hp[0, 0]
