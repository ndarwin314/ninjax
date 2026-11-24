"""Checkpoint transformations for the shared final-damage pipeline."""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import compute_damage_multipliers
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def scan_damage(carry, damage, *, attacker, defender, move):
    """Calculate an independent hit without changing the scan carry."""
    return carry, compute_damage_multipliers(attacker, defender, W.NONE, move,
        damage, False, 1., screen=True)


@pytest.mark.jit
def test_damage_chain_under_vmap_and_scan(neutral_type):
    """Stacked screen, Multiscale and Life Orb retain reference integer results."""
    _, state = neutral_type
    attacker = state.active[0].replace(item=jnp.array([I.LIFE_ORB]))
    defender = state.active[1].replace(ability=jnp.array([A.MULTISCALE]))
    move = make_move('Tackle')
    damage = jnp.array([[3], [41], [101]], jnp.int32)
    # Pinned Showdown probes for these three inputs return 1, 13 and 33.
    expected = jnp.array([[1], [13], [33]], jnp.int32)
    calculate = partial(compute_damage_multipliers, is_crit=False,
                        damage_roll=1., screen=True)
    mapped = jax.jit(jax.vmap(calculate, in_axes=(None, None, None, None, 0)))(
        attacker, defender, W.NONE, move, damage)
    scan_step = partial(scan_damage, attacker=attacker, defender=defender, move=move)
    _, scanned = jax.jit(partial(jax.lax.scan, scan_step))(jnp.int32(0), damage)
    assert jnp.array_equal(mapped, expected)
    assert jnp.array_equal(scanned, expected)
