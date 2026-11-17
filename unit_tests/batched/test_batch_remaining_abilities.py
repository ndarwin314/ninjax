"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.ability_effects import prepare_type_change
from ninjax.battle.residuals import step_moody
from ninjax.battle.switching import swap_is_alive
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.side.item_effects import refresh_paradox
from ninjax.side.queries import modify_stats
from unit_tests.test_battle import neutral_type
from unit_tests.test_remaining_abilities import ability_state


@pytest.mark.jit
def test_remaining_ability_hooks_jit_vmap_scan(neutral_type):
    """Exercise newly validated hooks with dynamic abilities and explicit keys."""
    key, state = neutral_type
    abilities = jnp.array([A.ORICHALCUM_PULSE,A.HADRON_ENGINE,A.PROTOSYNTHESIS,A.QUARK_DRIVE,
        A.BEADS_OF_RUIN,A.SWORD_OF_RUIN,A.VESSEL_OF_RUIN,A.LIBERO,A.PROTEAN,A.PIXILATE,
        A.MOODY,A.SAND_FORCE,A.SHADOW_TAG,A.STORM_DRAIN,A.WATER_ABSORB,A.FRIEND_GUARD])
    move = state.active[0].moves[0]
    def activate(key, state, ability):
        """Apply entry, type-change, residual and stat hooks for a selected ability."""
        state = ability_state(state, ability)
        state = refresh_paradox(swap_is_alive(state, 0))
        state = prepare_type_change(state, 0, move)
        key, state = step_moody(key, state)
        return key, state, modify_stats(state, 0, jnp.full(6, 101))
    keys = jax.random.split(key, len(abilities))
    _, batch, stats = jax.jit(jax.vmap(activate, in_axes=(0,None,0)))(keys, state, abilities)
    assert stats.shape == (16,6) and jnp.all(jnp.isfinite(stats))
    assert not jnp.any(batch.events.overflow)
    def transition(carry, ability):
        """Carry an immutable battle through a sequence of ability activations."""
        key, state, stats = activate(carry[0], carry[1], ability)
        return (key, state), stats
    (_, final), history = jax.jit(lambda k,s: jax.lax.scan(transition, (k,s), abilities))(key,state)
    assert history.shape == (16,6) and not final.events.overflow
