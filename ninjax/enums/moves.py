"""Move classifications, outcomes, flags and offensive-stat ownership."""

from enum import IntFlag, auto

import jax.numpy as jnp

from ninjax.enums.base import JaxEnum


class PowerRule(JaxEnum):
    """Power computation rules, independent of move identity.

    Most select initial power before ordinary ability/item modifiers.
    REMOVABLE_ITEM_BOOST selects Knock Off's priority-0 event modifier instead.
    Per-hit plans (Beat Up and Triple Axel) supply their own initial power.
    """
    FIXED = 0
    DOUBLE_WITHOUT_ITEM = 1
    DOUBLE_AFTER_FAILURE = 2
    POSITIVE_STAGES = 3
    TARGET_WEIGHT = 4
    WEIGHT_RATIO = 5
    DOUBLE_IN_WEATHER = 6
    REMOVABLE_ITEM_BOOST = 7
    DOUBLE_ON_GROUNDED_ELECTRIC_TARGET = 8
    TARGET_STATUS = 9
    USER_HP = 10
    TARGET_HP_120 = 11
    TARGET_HP_100 = 12
    ELECTRO_BALL = 13
    GYRO_BALL = 14
    USER_STATUS_MODIFIER = 15
    TARGET_POISON_MODIFIER = 16
    TARGET_HALF_HP_MODIFIER = 17
    ELECTRIC_TERRAIN_MODIFIER = 18
    SUPER_EFFECTIVE_MODIFIER = 19
    DOUBLE_IN_TERRAIN = 20


class HitRule(JaxEnum):
    """Number of hits or distribution used by a damaging move."""
    SINGLE = 1
    TWO_TO_FIVE = 2
    FIXED_TWO = 3
    FIXED_THREE = 4
    POPULATION_BOMB = 5


class ProtectionKind(JaxEnum):
    """Turn-scoped protection variants sharing the consecutive-use counter."""
    NONE = 0
    PROTECT = 1
    SPIKY_SHIELD = 2
    BANEFUL_BUNKER = 3
    BURNING_BULWARK = 4
    SILK_TRAP = 5
    KINGS_SHIELD = 6
    MAT_BLOCK = 7
    ENDURE = 8


class SemiInvulnerability(JaxEnum):
    """Location of a Pokémon during a two-turn move's charge."""
    NONE = 0
    AIRBORNE = 1
    UNDERGROUND = 2
    UNDERWATER = 3
    VANISHED = 4


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
