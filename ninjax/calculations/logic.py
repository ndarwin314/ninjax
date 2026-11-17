"""Small array boolean predicates shared by mechanics."""

import jax.numpy as jnp


def triple_and(a, b, c):
    """Combine three array predicates with logical AND."""
    return jnp.logical_and(jnp.logical_and(a, b), c)


def triple_or(a, b, c):
    """Combine three array predicates with logical OR."""
    return jnp.logical_or(jnp.logical_or(a, b), c)


def quad_or(a, b, c, d):
    """Combine four array predicates with logical OR."""
    return jnp.logical_or(a, jnp.logical_or(jnp.logical_or(b, c), d))


def quad_and(a, b, c, d):
    """Combine four array predicates with logical AND."""
    return jnp.logical_and(a, jnp.logical_and(jnp.logical_and(b, c), d))


def in_range(lb, value, ub):
    """Check an exclusive lower bound and inclusive upper bound."""
    return jnp.logical_and(jnp.less(lb, value), jnp.less_equal(value, ub))
