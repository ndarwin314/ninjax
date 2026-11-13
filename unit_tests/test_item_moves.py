"""Knock Off and item-exchange move integration."""
import jax.numpy as jnp
import pytest
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I
from ninjax.game_logic import effective_move, move_hits
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('name', ['Trick', 'Switcheroo'])
def test_exchange_and_substitute(neutral_type, name):
    """An exchange swaps items but cannot pass through Substitute."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.LEFTOVERS])))
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.LIFE_ORB])))
    v = state.volatile_status
    guarded = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(20)))
    _, blocked = move_hits(key, guarded, 0, 0)
    assert blocked.active.item[0, 0] == I.LEFTOVERS
    _, out = move_hits(key, state, 0, 0)
    assert out.active.item[0, 0] == I.LIFE_ORB and out.active.item[1, 0] == I.LEFTOVERS


def test_knock_off_power_and_removal(neutral_type):
    """Knock Off gains power and removes an eligible item, including after KO."""
    key, state = neutral_type
    state = equip_move(state, 'Knock Off')
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.LEFTOVERS]), current_hp=jnp.array([1])))
    assert effective_move(state, 0, make_move('Knock Off')).base_power[0] == 97
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] == 0 and out.active.item[1, 0] == I.NONE
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.STICKY_HOLD]), current_hp=jnp.array([200])))
    assert effective_move(state, 0, make_move('Knock Off')).base_power[0] == 97
    _, out = move_hits(key, state, 0, 0)
    assert out.active.item[1, 0] == I.LEFTOVERS
