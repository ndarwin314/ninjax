"""Fixed-size per-attempt hit records for singles and future doubles targeting."""
from enum import IntEnum

from flax import struct
import jax
import jax.numpy as jnp


class HitOutcome(IntEnum):
    """Outcome of a represented hit attempt; zero denotes an unused entry."""
    UNUSED = 0
    HIT = 1
    MISS = 2
    IMMUNE = 3
    BLOCKED = 4


class HitRecipient(IntEnum):
    """Entity that received a successful hit."""
    NONE = 0
    POKEMON = 1
    SUBSTITUTE = 2


@struct.dataclass
class HitResult:
    """One hit attempt; target coordinates are side and active-position indices.

    Singles currently uses active position zero. Damage is actual HP removed
    from the recipient, and is zero for misses, immunities and status moves.
    """
    intended_side: jax.Array
    intended_position: jax.Array
    actual_side: jax.Array
    actual_position: jax.Array
    outcome: jax.Array
    recipient: jax.Array
    damage: jax.Array

    @property
    def redirected(self):
        """Whether target selection changed the intended battle position."""
        return (self.intended_side != self.actual_side) | (self.intended_position != self.actual_position)

    def pack(self):
        """Pack the scalar result into one fixed-width integer row."""
        return jnp.asarray([self.intended_side, self.intended_position, self.actual_side,
                            self.actual_position, self.outcome, self.recipient, self.damage], jnp.int32).reshape((7,))


@struct.dataclass
class HitResults:
    """Up to ten attempted hits in execution order for the most recent move."""
    data: jax.Array

    @classmethod
    def empty(cls):
        """Create a history with no recorded attempts."""
        return cls(jnp.zeros((10, 7), jnp.int32))

    def __getitem__(self, index):
        """Unpack the result at a fixed or traced hit index."""
        row = self.data[index]
        return HitResult(*(row[..., i] for i in range(7)))

    def set(self, index, result):
        """Replace a single attempted hit without changing the history shape."""
        return self.replace(data=self.data.at[index].set(result.pack()))


def hit_result(side, outcome=HitOutcome.HIT, recipient=HitRecipient.NONE, damage=0):
    """Construct a singles attempt targeting side's only active position."""
    return HitResult(side, 0, side, 0, outcome, recipient, damage)
