"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.moves.multi_hit import resolve_multi_hit, sample_hit_count
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def test_counts_distribution_and_modifiers():
    """Check Gen 9 hit distributions and hit-count modifiers."""
    keys = jax.random.split(jax.random.key(9), 20000)
    draw = jax.jit(jax.vmap(sample_hit_count, in_axes=(0, None, None, None)))
    _, single = draw(keys, 1, A.NONE, I.NONE)
    assert jnp.all(single == 1)
    _, values = draw(keys, 2, A.NONE, I.NONE)
    frequencies = jnp.bincount(values, length=6) / values.size
    assert jnp.all(jnp.abs(frequencies[2:6] - jnp.array([.35, .35, .15, .15])) < .015)
    _, dice = draw(keys, 2, A.NONE, I.LOADED_DICE)
    assert jnp.all((dice == 4) | (dice == 5))
    assert .48 < jnp.mean(dice == 4) < .52
    _, linked = draw(keys, 2, A.SKILL_LINK, I.NONE)
    assert jnp.all(linked == 5)
    _, population = draw(keys, 4, A.NONE, I.LOADED_DICE)
    assert jnp.min(population) == 4 and jnp.max(population) == 10
    _, fixed = draw(keys, 3, A.SKILL_LINK, I.LOADED_DICE)
    assert jnp.all(fixed == 3)


@pytest.mark.jit
def test_multihit_jit_vmap_scan(neutral_type):
    """The hit loop composes with batch and sequential JAX transformations."""
    key, state = neutral_type
    # Small hits leave both sides alive across the scan.
    move = make_move('Bullet Seed').replace(base_power=jnp.array([5]))
    keys = jax.random.split(key, 2)
    states = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    _, batch = jax.jit(jax.vmap(lambda k, s: resolve_multi_hit(k, s, 0, move)))(keys, states)
    assert jnp.all(batch.team.current_hp[:, 1, 0, 0] < states.team.current_hp[:, 1, 0, 0])
    def turn(carry, _):
        """Apply a move within a scanned sequence of decisions."""
        key, state = carry
        key, state = resolve_multi_hit(key, state, 0, move)
        return (key, state), state.active.current_hp[1, 0]
    (_, final), hp = jax.jit(lambda k, s: jax.lax.scan(turn, (k, s), None, length=2))(key, state)
    assert hp[1] < hp[0] < state.active.current_hp[1, 0]
    assert not final.events.overflow
