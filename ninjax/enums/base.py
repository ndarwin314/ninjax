"""JAX-compatible enum comparison and stable Python hashing."""

from enum import IntEnum

import jax.numpy as jnp


class JaxEnum(IntEnum):
    """Integer identifiers that compare with traced arrays without Python branching."""
    def __eq__(self, other):
        """Compare an enum value with a scalar or traced array value."""
        return jnp.equal(self, other)
    def __hash__(self):
        """Retain the stable numeric value as the Python enum hash."""
        return self.value
