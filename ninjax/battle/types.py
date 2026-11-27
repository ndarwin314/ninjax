"""Action and episode PyTrees and environment configuration."""
from enum import IntEnum
from functools import partial

import jax.lax
import jax.numpy as jnp
from flax import struct

from ninjax.side.state import BattleState


def _empty_selection():
    """Create the default (2, 6) preview selection mask."""
    return jnp.zeros((2, 6), bool)


def _select_action_leaf(x, *, player):
    """Read one player from an action PyTree leaf."""
    return x[player]


class ActionType(IntEnum):
    """Battle action kinds, including forced charge/recharge continuations."""
    ATTACK = 0
    SWITCH = 1
    CONTINUED = 2
    NO_OP = 3


class Transformation(IntEnum):
    """Optional transformation requested alongside an attack action."""
    NONE = 0
    MEGA = 1
    Z_MOVE = 2
    DYNAMAX = 3
    TERA = 4


class BattleAction(struct.PyTreeNode):
    """Indexed actions with player axis (2,); a selected player has scalar leaves.

    Attack indices 0..3 select moves; index 4 selects forced Struggle.
    During revival, a switch index selects a fainted party member to restore.
    Selection remains a separate team-preview payload, shape (2, 6).
    """
    index: jax.Array
    move_type: jax.Array
    selection: jax.Array = struct.field(default_factory=_empty_selection)
    transformation: jax.Array = Transformation.NONE

    def __post_init__(self):
        """Expand enum/default values to the action's scalar or player shape."""
        if isinstance(self.transformation, int):
            object.__setattr__(self, 'transformation',
                               jnp.full(jnp.shape(self.index), int(self.transformation), jnp.int32))

    def __getitem__(self, player):
        """Select one player's action while retaining the PyTree structure."""
        return jax.tree.map(partial(_select_action_leaf, player=player), self)

    def encode(self):
        """Adapt to legacy mask columns; reject invalid or unsupported requests."""
        attack = (self.move_type == ActionType.ATTACK) & (self.index >= 0) & (self.index <= 4)
        switch = (self.move_type == ActionType.SWITCH) & (self.index >= 0) & (self.index < 6)
        continued = (self.move_type == ActionType.CONTINUED) & (self.index == 0)
        noop = (self.move_type == ActionType.NO_OP) & (self.index == 0)
        encoded = jnp.where(attack, self.index, jnp.where(switch, self.index + 8,
            jnp.where(continued, 5, jnp.where(noop, 14, -1)))).astype(jnp.int32)
        # Transformation identities are retained in the PyTree, but their
        # eligibility and battle lifecycle are not implemented yet.
        return jnp.where(self.transformation == Transformation.NONE, encoded, -1)

    @classmethod
    def from_legacy(cls, action):
        """Convert legacy mask columns, including the reserved continuation column."""
        action = jnp.asarray(action, jnp.int32)
        attack = (action >= 0) & (action <= 4)
        switch = (action >= 8) & (action < 14)
        kind = jnp.where(attack, ActionType.ATTACK, jnp.where(switch, ActionType.SWITCH,
            jnp.where(action == 5, ActionType.CONTINUED, jnp.where(action == 14, ActionType.NO_OP, -1))))
        index = jnp.where(attack, action, jnp.where(switch, action - 8, 0))
        selection = jnp.zeros(action.shape + (6,), bool)
        return cls(index=index, move_type=kind, selection=selection)


class InformationMode(IntEnum):
    """Opponent information available before and during an episode."""
    HIDDEN = 0
    SPECIES = 1
    OPEN_SHEET = 2


class BattleParams(struct.PyTreeNode):
    """Turn limit, team-preview mode and opponent-information policy."""
    max_steps_in_episode: int = 100
    team_preview: bool = False
    information_mode: int = InformationMode.HIDDEN


class Phase(IntEnum):
    """Decision phase, including a paused turn awaiting a revival target."""
    TEAM_SELECTION = 3
    TURN = 0
    REPLACEMENT = 1
    TERMINAL = 2
    REVIVAL = 4


class EpisodeState(struct.PyTreeNode):
    """A decision point, including actions suspended by a switching move."""
    key: jax.Array
    battle: BattleState
    turn: jax.Array
    phase: jax.Array
    acting_player_mask: jax.Array
    pending_actions: BattleAction
    queued_actor_slot: jax.Array
    order: jax.Array
    cursor: jax.Array
    field_pending: jax.Array
    selected: jax.Array
    revealed: jax.Array
    revealed_moves: jax.Array
