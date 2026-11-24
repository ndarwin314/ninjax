"""Checkpoint transformation checks for the unified intrinsic-power dispatcher."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import resolve_hit_base_power
from ninjax.enums.moves import PowerRule
from ninjax.enums.pokemon import Status


def resolve_example(base_power, rule):
    """Exercise old and newly introduced rules through identical numeric inputs."""
    return resolve_hit_base_power(base_power, rule, no_item=True,
        positive_stages=3, target_status=Status.BURN)


def scan_power(carry, entry):
    """Resolve one independent example without changing the scan carry."""
    return carry, resolve_example(*entry)


@pytest.mark.jit
def test_unified_power_under_vmap_and_scan():
    """Both transformations preserve dynamic rule selection and integer results."""
    powers = jnp.array([55, 65, 20], jnp.int32)
    rules = jnp.array([PowerRule.DOUBLE_WITHOUT_ITEM, PowerRule.TARGET_STATUS,
                       PowerRule.POSITIVE_STAGES], jnp.int32)
    expected = jnp.array([110, 130, 80], jnp.int32)
    mapped = jax.jit(jax.vmap(resolve_example))(powers, rules)
    _, scanned = jax.lax.scan(scan_power, jnp.int32(0), (powers, rules))
    assert jnp.array_equal(mapped, expected)
    assert jnp.array_equal(scanned, expected)
