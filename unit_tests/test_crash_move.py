"""High Jump Kick uses its recorded attempt outcome for crash damage."""
import jax.numpy as jnp
from ninjax.battle import step_move
from ninjax.enum_types import AbilityEnum as A, Type
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def test_crash_on_protect_and_immunity(neutral_type):
    """Protect and Ghost immunity cause half-maximum HP crash damage."""
    key, state = neutral_type
    state = equip_move(state, 'High Jump Kick')
    v = state.volatile_status
    protected = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    _, out = step_move(key, protected, 0, 0, False)
    assert out.active.current_hp[0, 0] == protected.active.current_hp[0, 0] - protected.active.max_hp[0] // 2
    ghost = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE])))
    _, out = step_move(key, ghost, 0, 0, False)
    assert out.active.current_hp[0, 0] == ghost.active.current_hp[0, 0] - ghost.active.max_hp[0] // 2


def test_magic_guard_and_gravity(neutral_type):
    """Magic Guard prevents a crash and Gravity stops use before PP is spent."""
    key, state = neutral_type
    state = equip_move(state, 'High Jump Kick')
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.MAGIC_GUARD])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    state = state.replace(gravity_duration=jnp.int32(3))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0]
