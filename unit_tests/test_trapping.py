"""Switch selection, bindings and their residual damage in singles."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.residuals import ResidualHPPhase, run_hp_residuals
from ninjax.battle.types import Phase
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Type
from ninjax.moves.multi_hit import resolve_multi_hit
from ninjax.side.trapping import tick_binding, voluntary_switch_blocked
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


@pytest.mark.parametrize('ability,types,item,gravity,blocked', [
    (A.SHADOW_TAG, [Type.NORMAL, Type.NONE], I.NONE, 0, True),
    (A.SHADOW_TAG, [Type.GHOST, Type.NONE], I.NONE, 0, False),
    (A.SHADOW_TAG, [Type.NORMAL, Type.NONE], I.SHED_SHELL, 0, False),
    (A.ARENA_TRAP, [Type.FLYING, Type.NONE], I.NONE, 0, False),
    (A.ARENA_TRAP, [Type.FLYING, Type.NONE], I.NONE, 2, True),
    (A.MAGNET_PULL, [Type.STEEL, Type.NONE], I.NONE, 0, True),
    (A.MAGNET_PULL, [Type.STEEL, Type.GHOST], I.NONE, 0, False),
])
def test_trapping_ability_masks_only_voluntary_switches(neutral_type, ability, types, item, gravity, blocked):
    """Effective types, items and grounding govern new switch choices."""
    key, state = neutral_type
    env = Battle(state.team)
    episode, _ = env.reset(key)
    state = update_active(episode.battle, 0, episode.battle.active[0].replace(
        type_list=jnp.array(types), item=jnp.array([item])))
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    episode = episode.replace(battle=state.replace(gravity_duration=jnp.int32(gravity)))
    assert bool(jnp.any(env.legal_action_mask(episode)[0, 8:14])) != blocked
    replacement = episode.replace(phase=jnp.int32(Phase.REPLACEMENT))
    assert jnp.any(env.legal_action_mask(replacement)[0, 8:14])


def _binding_residual(state):
    """Run the binding checkpoint without compiling the whole scheduler."""
    return run_hp_residuals(tick_binding(state), (ResidualHPPhase.BINDING,), jnp.arange(2))


def test_binding_hit_sets_source_and_expires(neutral_type):
    """A binding hit attributes residual HP loss until the source leaves."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.BINDING_BAND])))
    move = make_move('Wrap').replace(accuracy=jnp.ones(1))
    _, bound = jax.jit(resolve_multi_hit)(key, state, 0, move)
    trap = bound.volatile_status.trapping
    assert trap.binding_move_id[1, 0] == 35
    assert trap.binding_source_slot[1, 0] == state.active_index[0, 0]
    assert trap.binding_turns[1, 0] in (5, 6)
    assert trap.binding_divisor[1, 0] == 6
    assert voluntary_switch_blocked(bound)[1]
    hp = bound.active.current_hp[1, 0]
    damaged = jax.jit(_binding_residual)(bound)
    assert hp - damaged.active.current_hp[1, 0] == bound.active.max_hp[1] // 6
    expired = bound.replace(active_index=bound.active_index.at[0, 0].set(1))
    assert not voluntary_switch_blocked(expired)[1]
    expired = _binding_residual(expired)
    assert expired.active.current_hp[1, 0] == hp
    assert expired.volatile_status.trapping.binding_turns[1, 0] == 0
