"""Roost healing and temporary effective-type changes."""
import jax
import jax.numpy as jnp
import pytest
from ninjax.battle import step_field
from ninjax.enum_types import AbilityEnum as A, Type
from ninjax.game_logic import do_status_move, effective_move
from ninjax.prototype import make_move
from ninjax.side import clear_volatile_status, update_active
from unit_tests.test_battle import neutral_type


@pytest.mark.parametrize('types,expected', [([Type.FLYING, Type.NONE], [Type.NORMAL, Type.NONE]), ([Type.FLYING, Type.WATER], [Type.WATER, Type.NONE]), ([Type.STEEL, Type.FLYING], [Type.STEEL, Type.NONE])])
def test_roost_effective_type_and_expiration(neutral_type, types, expected):
    """Healing removes Flying until the residual step while preserving stored types."""
    key, state = neutral_type
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(type_list=jnp.array(types), current_hp=jnp.array([20])))
    _, out = do_status_move(key, state, 0, make_move('Roost'), 0, 0)
    assert out.active.current_hp[0, 0] > 20
    assert jnp.array_equal(out.active.type_list[0], jnp.array(types))
    assert jnp.array_equal(out.effective_active.type_list[0], jnp.array(expected))
    assert out.is_grounded(0)[0]
    assert not out.is_immune_to_move(0, make_move('Earthquake'))
    assert effective_move(out, 0, make_move('Revelation Dance')).type[0] == expected[0]
    assert not clear_volatile_status(out, 0).volatile_status.roost[0, 0]
    _, expired = step_field(key, out)
    assert not expired.volatile_status.roost[0, 0]
    assert jnp.array_equal(expired.effective_active.type_list[0], jnp.array(types))


def test_roost_full_hp_tera_and_levitate(neutral_type):
    """Full HP fails; Tera retains Flying; Levitate still provides immunity."""
    key, state = neutral_type
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(type_list=jnp.array([Type.FLYING, Type.NONE])))
    _, full = do_status_move(key, state, 0, make_move('Roost'), 0, 0)
    assert not full.volatile_status.roost[0, 0]
    mon = state.active[0]
    tera = update_active(state, 0, mon.replace(current_hp=jnp.array([20]), is_terastallized=jnp.array([True])))
    _, out = do_status_move(key, tera, 0, make_move('Roost'), 0, 0)
    assert not out.volatile_status.roost[0, 0] and out.effective_active.type_list[0, 0] == Type.FLYING
    levitate = update_active(state, 0, mon.replace(current_hp=jnp.array([20]), ability=jnp.array([A.LEVITATE])))
    _, out = do_status_move(key, levitate, 0, make_move('Roost'), 0, 0)
    assert out.volatile_status.roost[0, 0] and not out.is_grounded(0)[0]


def test_effective_roost_types_trace(neutral_type):
    """Roost type and grounded checks trace under vmap and scan."""
    _, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(roost=v.roost.at[0, 0].set(True)))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.eval_shape(jax.vmap(lambda s: s.effective_active.type_list), batch).shape == (2, 2, 2)
    def scan(s):
        """Observe effective types across a fixed scan."""
        return jax.lax.scan(lambda s, _: (s, s.effective_active.type_list), s, None, length=2)[1]
    assert jax.eval_shape(scan, state).shape == (2, 2, 2)


def test_roost_preserves_existing_terastallization(neutral_type):
    """A previously Terastallized user's normal action keeps its Tera state."""
    from ninjax.battle import step_move
    from unit_tests.test_curated_moves import equip_move
    key, state = neutral_type
    state = equip_move(state, 'Roost')
    state = update_active(state, 0, state.active[0].replace(type_list=jnp.array([Type.FLYING, Type.NONE]), current_hp=jnp.array([20]), is_terastallized=jnp.array([True]), tera_type=jnp.array([Type.FLYING])))
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.is_terastallized[0, 0]
    assert out.effective_active.type_list[0, 0] == Type.FLYING
    assert not out.volatile_status.roost[0, 0]
