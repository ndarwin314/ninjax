"""Atomic removal and exchange of active held items."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enum_types import AbilityEnum as A, FormEnum as F, ItemEnum as I
from ninjax.held_effects import can_remove_item, remove_item, swap_items
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type


def items(state, first, second):
    """Assign two held items without revealing them for test setup."""
    for p, item in enumerate((first, second)):
        state = update_active(state, p, state.active[p].replace(item=jnp.array([item])))
    return state


def test_remove_updates_knowledge_and_choice_lock(neutral_type):
    """Removal clears an item, stale Choice lock, and previously revealed knowledge."""
    _, state = neutral_type
    state = items(state, I.NONE, I.CHOICE_BAND)
    v = state.volatile_status
    state = state.replace(known_items=state.known_items.at[1, 0].set(I.CHOICE_BAND),
        volatile_status=v.replace(choice_lock=v.choice_lock.at[1, 0].set(2)))
    out, success = remove_item(state, 1, 0)
    assert success and out.active.item[1, 0] == I.NONE
    assert out.known_items[1, 0] == I.NONE
    assert out.volatile_status.choice_lock[1, 0] == -1
    assert out.events.count > state.events.count
    failed, success = remove_item(out, 1, 0)
    assert not success and failed.events.count == out.events.count


@pytest.mark.parametrize('first,second', [(I.NONE, I.LEFTOVERS), (I.LIFE_ORB, I.NONE), (I.LIFE_ORB, I.LEFTOVERS), (I.LEFTOVERS, I.LEFTOVERS)])
def test_swaps_and_public_item_knowledge(neutral_type, first, second):
    """Exchange allows one empty slot and reveals both resulting item slots."""
    _, state = neutral_type
    state = items(state, first, second)
    out, success = swap_items(state, 0, 1)
    assert success
    assert out.active.item[0, 0] == second and out.active.item[1, 0] == first
    assert out.known_items[0, 0] == second and out.known_items[1, 0] == first


def test_failures_are_atomic(neutral_type):
    """Sticky Hold and unsupported item identities leave every leaf unchanged."""
    _, state = neutral_type
    state = items(state, I.LIFE_ORB, I.LEFTOVERS)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.STICKY_HOLD])))
    for mutate in (lambda s: swap_items(s, 0, 1), lambda s: remove_item(s, 1, 0)):
        out, success = mutate(state)
        assert not success
        assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(out)))
    assert can_remove_item(state, 1, 1)  # Its own Sticky Hold permits self-removal.
    dead = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([0]), is_alive=jnp.array([False])))
    assert can_remove_item(dead, 1, 0)
    empty = items(state, I.NONE, I.NONE)
    assert not swap_items(empty, 0, 1)[1]
    unknown = items(state, I.LIFE_ORB, 999999)
    assert not swap_items(unknown, 0, 1)[1]


def test_mask_bindings_apply_to_both_directions(neutral_type):
    """Ogerpon cannot give away its mask or receive one through item exchange."""
    _, state = neutral_type
    state = items(state, I.LEFTOVERS, I.WELLSPRING_MASK)
    ogerpon = update_active(state, 1, state.active[1].replace(form=jnp.array([F.OGERPON_WELLSPRING])))
    assert not remove_item(ogerpon, 1, 0)[1]
    assert not swap_items(ogerpon, 0, 1)[1]
    incoming = update_active(state, 0, state.active[0].replace(species=jnp.array([1017])))
    assert not swap_items(incoming, 0, 1)[1]
    # Masks carried by unrelated species are transferable.
    assert swap_items(state, 0, 1)[1]


def test_received_berry_activates_after_atomic_assignment(neutral_type):
    """A received Sitrus Berry can heal without consuming the other swapped item."""
    _, state = neutral_type
    state = items(state, I.CHOICE_SPECS, I.SITRUS_BERRY)
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([10])))
    out, success = swap_items(state, 0, 1)
    assert success
    assert out.active.item[0, 0] == I.NONE and out.active.item[1, 0] == I.CHOICE_SPECS
    assert out.active.current_hp[0, 0] == 10 + state.active.max_hp[0] // 4
    assert out.known_items[0, 0] == I.NONE


def test_effective_sticky_hold_and_received_mental_herb(neutral_type):
    """Ability overrides govern protection; acquiring Mental Herb cures Taunt."""
    _, state = neutral_type
    state = items(state, I.MENTAL_HERB, I.LEFTOVERS)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[1, 0].set(A.STICKY_HOLD)))
    assert not swap_items(state, 0, 1)[1]
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_suppressed=v.ability_suppressed.at[1, 0].set(True),
        taunt_turns=v.taunt_turns.at[1, 0].set(3)))
    out, success = swap_items(state, 0, 1)
    assert success and out.volatile_status.taunt_turns[1, 0] == 0
    assert out.active.item[1, 0] == I.NONE


def test_item_mutation_traces_under_jax(neutral_type):
    """Pure mutation helpers retain fixed shapes under vmap and scan tracing."""
    _, state = neutral_type
    state = items(state, I.LIFE_ORB, I.LEFTOVERS)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    shape = jax.eval_shape(jax.vmap(lambda s: swap_items(s, 0, 1)[0].active.item), batch)
    assert shape.shape == (2, 2, 1)
    def scan(s):
        """Carry immutable item state through repeated exchanges."""
        return jax.lax.scan(lambda s, _: (swap_items(s, 0, 1)[0], None), s, None, length=2)[0].active.item
    assert jax.eval_shape(scan, state).shape == (2, 1)


def test_choice_lock_preservation_and_booster_energy(neutral_type):
    """Choice-to-Choice exchanges retain locks; received Booster Energy activates."""
    _, state = neutral_type
    state = items(state, I.CHOICE_BAND, I.CHOICE_SPECS)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(choice_lock=v.choice_lock.at[:, 0].set(jnp.array([1, 2]))))
    out, success = swap_items(state, 0, 1)
    assert success and jnp.array_equal(out.volatile_status.choice_lock, state.volatile_status.choice_lock)
    state = items(state, I.BOOSTER_ENERGY, I.LEFTOVERS)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.QUARK_DRIVE])))
    out, success = swap_items(state, 0, 1)
    assert success and out.active.item[1, 0] == I.NONE
    assert out.volatile_status.paradox_booster[1, 0]
    assert out.volatile_status.paradox_stat[1, 0] > 0
    assert out.known_items[1, 0] == I.NONE
    assert out.volatile_status.choice_lock[1, 0] == -1
    assert out.event_source == state.event_source
