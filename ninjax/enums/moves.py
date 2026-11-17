"""Move classifications, outcomes, flags and offensive-stat ownership."""

from enum import IntFlag, auto

import jax.numpy as jnp

from ninjax.enums.base import JaxEnum


class ProtectionKind(JaxEnum):
    """Turn-scoped protection variants sharing the consecutive-use counter."""
    NONE = 0
    PROTECT = 1
    SPIKY_SHIELD = 2


class MoveResult(JaxEnum):
    """A Pokémon's whole-move outcome for the current or previous turn."""
    NOT_ATTEMPTED = 0
    FAILED = 1
    SUCCEEDED = 2
    SKIPPED = 3


class OffensiveStatOwner(JaxEnum):
    """Active Pokémon supplying the raw offensive stat and its stages."""
    USER = 0
    TARGET = 1


class MoveType(JaxEnum):
    """Physical, special and status move categories."""
    PHYSICAL = 0
    SPECIAL = 1
    STATUS = 2


class MoveFlags(IntFlag):
    """Independent move-property bits used by ability and item rules."""
    CONTACT = auto()
    PUNCHING = auto()
    SOUND = auto()
    BULLET = auto()
    LAUNCHER = auto()
    HEALING = auto()
    BITING = auto()
    POWDER = auto()
    DEFROST = auto()
    BYPASS_SUBSTITUTE = auto()
    SLICING = auto()
    WIND = auto()

    def __eq__(self, other):
        """Compare an enum value with a scalar or traced array value."""
        return jnp.equal(self, other)
    def __hash__(self):
        """Retain the stable numeric value as the Python enum hash."""
        return self.value
