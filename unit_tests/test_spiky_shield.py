"""Spiky Shield shares Protect's chain and retaliates on blocked contact."""
import jax
import jax.numpy as jnp
import pytest
from ninjax.enum_types import AbilityEnum as A, ProtectionKind
from ninjax.game_logic import do_status_move, move_hits, step_side_conditions
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def test_shared_counter_and_end_turn(neutral_type):
    """Alternating Protect variants uses one failure chain and clears the variant."""
    key, state = neutral_type
    key, state = do_status_move(key, state, 1, make_move('Protect'), 0, 0)
    assert state.volatile_status.protect_chain[1, 0] == 1
    key, state = step_side_conditions(key, state)
    assert state.volatile_status.protection_kind[1, 0] == ProtectionKind.NONE
    _, roll_key = jax.random.split(key)
    succeeds = bool(jax.random.uniform(roll_key) < 1/3)
    _, out = do_status_move(key, state, 1, make_move('Spiky Shield'), 0, 0)
    assert bool(out.volatile_status.protected[1, 0]) == succeeds
    assert out.volatile_status.protect_chain[1, 0] == (2 if succeeds else 0)
    assert out.volatile_status.protection_kind[1, 0] == (ProtectionKind.SPIKY_SHIELD if succeeds else ProtectionKind.NONE)


@pytest.mark.parametrize('ability,move,retaliates,bypasses', [
    (A.NONE, 'U-turn', True, False), (A.LONG_REACH, 'U-turn', False, False),
    (A.MAGIC_GUARD, 'U-turn', False, False), (A.UNSEEN_FIST, 'U-turn', False, True),
    (A.NONE, 'Dragon Pulse', False, False),
])
def test_block_contact_and_bypass(neutral_type, ability, move, retaliates, bypasses):
    """Only blocked contact causes damage; Unseen Fist bypasses the shield."""
    key, state = neutral_type
    state = equip_move(state, move)
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([ability])))
    key, state = do_status_move(key, state, 1, make_move('Spiky Shield'), 0, 0)
    _, out = move_hits(key, state, 0, 0)
    damage = max(1, int(state.active.max_hp[0]) // 8) if retaliates else 0
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - damage
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) == bypasses


def test_shield_blocks_before_accuracy(neutral_type):
    """A zero-accuracy contact move still hits the shield before accuracy."""
    from ninjax.battle import step_move
    key, state = neutral_type
    state = equip_move(state, 'U-turn')
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(0.))))
    key, state = do_status_move(key, state, 1, make_move('Spiky Shield'), 0, 0)
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - max(1, int(state.active.max_hp[0]) // 8)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
