from typing import Any
from enum import IntEnum

import chex
import jax.lax
from flax import struct
from jax import lax, random
from functools import cached_property
from jumanji.env import Environment
from jumanji import specs
from jumanji.types import TimeStep, restart, transition, truncation, termination
import jax.numpy as jnp


from ninjax.side import BattleState, update_active
from ninjax.pokemon import Pokemon
from ninjax.move import Move
from ninjax.game_logic import do_damaging_move
from ninjax.side import take_damage_value
from ninjax.utils import triple_and, triple_or
from ninjax.enum_types import StatEnum, Type, AbilityEnum, Weather, Terrain, Status, MoveType
from ninjax.game_logic import (step_side_conditions, swap_out, end_turn_damage, move_interrupted,
                               move_used, step_moody)


from ninjax.events import EventBuffer, EventKind, Cause, emit, capture_conditions
from ninjax.observations import Observation, observe, observation_spec

Binary = (0,1)

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
    Selection remains a separate team-preview payload, shape (2, 6).
    """
    index: jax.Array
    move_type: jax.Array
    selection: jax.Array = struct.field(default_factory=lambda: jnp.zeros((2, 6), bool))
    transformation: jax.Array = Transformation.NONE

    def __post_init__(self):
        """Expand enum/default values to the action's scalar or player shape."""
        if isinstance(self.transformation, int):
            object.__setattr__(self, 'transformation',
                               jnp.full(jnp.shape(self.index), int(self.transformation), jnp.int32))

    def __getitem__(self, player):
        """Select one player's action while retaining the PyTree structure."""
        return jax.tree.map(lambda x: x[player], self)

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
    HIDDEN = 0
    SPECIES = 1
    OPEN_SHEET = 2


class BattleParams(struct.PyTreeNode):
    max_steps_in_episode: int = 100
    team_preview: bool = False
    information_mode: int = InformationMode.HIDDEN

class Phase(IntEnum):
    TEAM_SELECTION = 3
    TURN = 0
    REPLACEMENT = 1
    TERMINAL = 2


class EpisodeState(struct.PyTreeNode):
    """A decision point, including actions suspended by a switching move."""
    key: jax.Array
    battle: BattleState
    turn: jax.Array
    phase: jax.Array
    acting_player_mask: jax.Array
    pending_actions: BattleAction
    order: jax.Array
    cursor: jax.Array
    field_pending: jax.Array
    selected: jax.Array
    revealed: jax.Array
    revealed_moves: jax.Array




class Battle(Environment):

    def __init__(self, teams: Pokemon | None = None, params: BattleParams | None = None):
        # Legacy adapter: moves 0–3, Struggle 4, reserved 5–7,
        # switches 8–13, inactive no-op 14. New callers can use BattleAction.
        self.action_set = jnp.array(range(15))
        self.teams = teams
        self.params = params or BattleParams()
        super().__init__()
        # idk
        #self.obs_shape = (1, 1)

    @property
    def name(self) -> str:
        return "Pokemon"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    @cached_property
    def action_spec(self):
        """Describe indexed action leaves and the team-preview selection payload."""
        return specs.Spec(BattleAction, "action",
            index=specs.BoundedArray((2,), jnp.dtype("int32"), 0, 5),
            move_type=specs.BoundedArray((2,), jnp.dtype("int32"), 0, 3),
            transformation=specs.BoundedArray((2,), jnp.dtype("int32"), 0, 4),
            selection=specs.BoundedArray((2, 6), jnp.dtype("bool"), False, True))

    @cached_property
    def observation_spec(self):
        return observation_spec()

    def observe(self, state):
        return observe(self, state)

    @cached_property
    def reward_spec(self):
        return specs.Array((2,), jnp.dtype("float32"), "reward")

    @cached_property
    def discount_spec(self):
        return specs.BoundedArray((2,), jnp.dtype("float32"), 0., 1., "discount")

    def _extras(self, state):
        return {"acting_player_mask": state.acting_player_mask,
                "legal_action_mask": self.legal_action_mask(state), "phase": state.phase,
                "invalid_action": jnp.bool_(False)}

    def legal_action_mask(self, state):
        """Mask moves and switches for the phase, including represented trapping."""
        battle = state.battle
        alive = (battle.team.current_hp[..., 0] > 0) & state.selected
        switches = alive & (jnp.arange(6)[None, :] != battle.active_index)
        active = battle.active
        moves = (active.moves.current_pp[..., 0] > 0) & ~active.moves.disabled[..., 0]
        from ninjax.restrictions import allowed_moves
        moves &= allowed_moves(battle)
        from ninjax.enum_types import ItemEnum
        items = active.item[:, 0]
        choice = (items == ItemEnum.CHOICE_BAND) | (items == ItemEnum.CHOICE_SPECS)
        lock = battle.volatile_status.choice_lock[:, 0]
        moves &= ~choice[:, None] | (lock[:, None] < 0) | (jnp.arange(4)[None, :] == lock[:, None])
        moves &= (items[:, None] != ItemEnum.ASSAULT_VEST) | (active.moves.move_type[..., 0] != MoveType.STATUS)
        trapped = (battle.ability[::-1, 0] == AbilityEnum.SHADOW_TAG) & (active.current_hp[::-1, 0] > 0) & (battle.ability[:, 0] != AbilityEnum.SHADOW_TAG) & ~jnp.any(active.type_list == Type.GHOST, axis=1)
        switches &= ~trapped[:, None] | (state.phase != Phase.TURN)
        forced = ~jnp.any(moves, axis=1) & (state.phase == Phase.TURN)
        moves &= state.phase == Phase.TURN
        mask = jnp.concatenate((moves, jnp.concatenate((forced[:, None], jnp.zeros((2, 3), bool)), axis=1), switches,
                                jnp.zeros((2, 1), bool)), axis=1)
        v = battle.volatile_status
        continuing = ((v.charging_move_id[:, 0] != 0) | v.recharging[:, 0]) & (state.phase == Phase.TURN) & (active.current_hp[:, 0] > 0)
        mask = jnp.where(continuing[:, None], jnp.arange(15)[None, :] == 5, mask)
        # Inactive players supply a no-op; their actual input is ignored.
        return jnp.where(state.acting_player_mask[:, None], mask,
                         jnp.arange(15)[None, :] == 14)

    def reset(self, key: jax.Array) -> tuple[EpisodeState, TimeStep[Observation]]:
        if self.teams is None:
            raise ValueError("Battle requires caller-provided teams for reset")
        state = EpisodeState(key=key, battle=BattleState(team=self.teams), turn=jnp.int32(0),
                             phase=jnp.int32(Phase.TURN), acting_player_mask=jnp.ones(2, bool),
                             pending_actions=BattleAction.from_legacy(jnp.full((2,), 14, jnp.int32)),
                             order=jnp.arange(2, dtype=jnp.int32), cursor=jnp.int32(2), field_pending=jnp.bool_(False), selected=jnp.ones((2, 6), bool),
                             revealed=jnp.zeros((2, 6), bool).at[:, 0].set(not self.params.team_preview),
                             revealed_moves=jnp.zeros((2, 6, 4), bool))
        state = state.replace(phase=jnp.int32(Phase.TEAM_SELECTION if self.params.team_preview else Phase.TURN))
        if not self.params.team_preview:
            state = self._activate_entry_abilities(state)
        return state, restart(self.observe(state), shape=(2,), extras=self._extras(state))

    def _activate_entry_abilities(self, state):
        """Resolve initial abilities in speed order, including team preview starts."""
        from ninjax.game_logic import swap_is_alive
        from ninjax.held_effects import refresh_paradox
        first, second = action_order(state.key, state.battle, jnp.zeros(2, jnp.int32))
        battle = swap_is_alive(state.battle, first)
        battle = swap_is_alive(battle, second)
        return state.replace(battle=refresh_paradox(battle))

    def step(self, state: EpisodeState, action: chex.Array | BattleAction) -> tuple[EpisodeState, TimeStep[Observation]]:
        """Validate indexed actions and advance the symmetric battle scheduler."""
        state = state.replace(battle=state.battle.replace(events=EventBuffer.empty()))
        if isinstance(action, BattleAction):
            selection = action.selection.astype(bool)
            selecting = state.phase == Phase.TEAM_SELECTION
            valid_selection = jnp.all(jnp.any(selection, axis=1)) & jnp.all(action.transformation == Transformation.NONE)
            encoded = action.encode()
            def choose(current):
                indices = jnp.argmax(selection, axis=1).astype(jnp.int32)
                battle = current.battle.replace(active_index=indices[:, None], roster=selection)
                revealed = jnp.zeros((2, 6), bool).at[jnp.arange(2), indices].set(True)
                current = current.replace(battle=battle, selected=selection, revealed=revealed,
                                          phase=jnp.int32(Phase.TURN))
                return self._activate_entry_abilities(current)
            if self.params.team_preview:
                return lax.cond(selecting,
                    lambda: (lax.cond(valid_selection, choose, lambda x: x, state),
                             transition(jnp.zeros(2), self.observe(lax.cond(valid_selection, choose, lambda x: x, state)),
                                        shape=(2,), extras={**self._extras(lax.cond(valid_selection, choose, lambda x: x, state)),
                                                          "invalid_action": ~valid_selection})),
                    lambda: self.step(state, encoded))
        else:
            action = BattleAction.from_legacy(action)
        encoded = action.encode()
        mask = self.legal_action_mask(state)
        valid = (state.phase != Phase.TEAM_SELECTION) & (encoded >= 0) & (encoded < 15)
        valid &= mask[jnp.arange(2), jnp.clip(encoded, 0, 14)]
        valid = jnp.all(valid | ~state.acting_player_mask)
        state = lax.cond(valid & (state.phase != Phase.TERMINAL), self._advance,
                         lambda current, _: current, state, action)
        alive = jnp.any((state.battle.team.current_hp[..., 0] > 0) & state.selected, axis=1)
        defeated = ~jnp.all(alive)
        timeout = state.turn >= self.params.max_steps_in_episode
        defeated = defeated & ~timeout  # The configured turn limit always gives a draw.
        done = defeated | timeout
        state = state.replace(phase=jnp.where(done, Phase.TERMINAL, state.phase),
                              acting_player_mask=state.acting_player_mask & ~done)
        reward = jnp.where(defeated, alive.astype(jnp.float32) - alive[::-1].astype(jnp.float32), 0.)
        reward = jnp.where(defeated & ~jnp.any(alive) & (state.battle.winner >= 0),
                           jnp.where(jnp.arange(2) == state.battle.winner, 1., -1.), reward)
        obs = self.observe(state)
        extras = self._extras(state)
        extras["invalid_action"] = ~valid
        timestep = lax.cond(defeated,
            lambda: termination(reward, obs, shape=(2,), extras=extras),
            lambda: lax.cond(timeout,
                lambda: truncation(reward, obs, shape=(2,), extras=extras),
                lambda: transition(reward, obs, shape=(2,), extras=extras)))
        return state, timestep

    def _advance(self, state, actions):
        """Execute queued simultaneous choices and pause for replacement decisions."""
        def select(current):
            key, order_key = random.split(current.key)
            v = current.battle.volatile_status
            continuing = (actions.move_type == ActionType.CONTINUED) & (v.charging_move_id[:, 0] != 0)
            charge_indices = jnp.argmax(current.battle.active.moves.identifier[..., 0] == v.charging_move_id, axis=1)
            indices = jnp.where(continuing, charge_indices, jnp.clip(actions.index, 0, 3))
            from ninjax.game_logic import effective_move
            priorities = jax.vmap(lambda p, i: effective_move(current.battle, p, current.battle.active[p].moves[i]).priority[0])(jnp.arange(2), indices)
            priorities = jnp.where(actions.move_type == ActionType.SWITCH, 7,
                jnp.where(((actions.move_type == ActionType.ATTACK) & (actions.index < 4)) | continuing, priorities, 0))
            first, second = action_order(order_key, current.battle, priorities)
            v = current.battle.volatile_status
            battle = current.battle.replace(volatile_status=v.replace(
                queued_priority=priorities[:, None], queued_priority_valid=(((actions.move_type == ActionType.ATTACK) & (actions.index < 4)) | continuing)[:, None]))
            return current.replace(key=key, battle=battle, pending_actions=actions,
                                   order=jnp.stack((first, second)), cursor=jnp.int32(0), field_pending=jnp.bool_(True),
                                   acting_player_mask=jnp.zeros(2, bool))

        def replace_active(current):
            for player in range(2):
                key, battle = lax.cond(current.acting_player_mask[player],
                    lambda k, b: step_switch(k, b, player, actions.index[player], False),
                    lambda k, b: (k, b), current.key, current.battle)
                from ninjax.held_effects import refresh_paradox
                battle = refresh_paradox(battle)
                battle = capture_conditions(current.battle, battle)
                current = current.replace(key=key, battle=battle)
            return current.replace(acting_player_mask=jnp.zeros(2, bool))

        was_turn = state.phase == Phase.TURN
        state = lax.cond(was_turn, select, replace_active, state)
        # A completed turn awaiting fainted replacements must not run field effects again.
        finished_replacement = ~state.field_pending
        state = state.replace(phase=jnp.int32(Phase.TURN))

        def resolve(_, current):
            def execute(current):
                player = current.order[current.cursor]
                chosen = current.pending_actions[player]
                other = 1-player
                other_action = current.pending_actions.move_type[other]
                pending = (current.cursor == 0) & ((other_action == ActionType.ATTACK) | ((other_action == ActionType.CONTINUED) &
                    (current.battle.volatile_status.charging_move_id[other, 0] != 0))) & (current.battle.active.current_hp[other, 0] > 0)
                volatile = current.battle.volatile_status
                battle_with_context = current.battle.replace(volatile_status=volatile.replace(
                    pending_move=volatile.pending_move.at[:, 0].set(False).at[other, 0].set(pending)))
                can_act = current.battle.active.current_hp[player, 0] > 0
                key, battle = lax.cond(can_act,
                    lambda k, b: step_action(k, b, chosen, player),
                    lambda k, b: (k, b), current.key, battle_with_context)
                from ninjax.held_effects import refresh_paradox
                battle = refresh_paradox(battle)
                battle = capture_conditions(current.battle, battle)
                slot = current.battle.active_index[player, 0]
                spent = battle.team.moves.current_pp[player, slot, :, 0] < current.battle.team.moves.current_pp[player, slot, :, 0]
                known = current.revealed_moves.at[player, slot].set(current.revealed_moves[player, slot] | spent)
                has_bench = jnp.any(current.selected[player] & (battle.team.current_hp[player, :, 0] > 0) &
                                   (jnp.arange(6) != battle.active_index[player, 0]))
                pivot = battle.switch_requested[player] & (battle.active.current_hp[player, 0] > 0) & has_bench
                battle = battle.replace(switch_requested=jnp.zeros(2, bool))
                requested = jnp.zeros(2, bool).at[player].set(pivot)
                return current.replace(key=key, battle=battle, revealed_moves=known, cursor=current.cursor + 1,
                    phase=jnp.where(pivot, Phase.REPLACEMENT, Phase.TURN), acting_player_mask=requested)
            return lax.cond((current.cursor < 2) & ~jnp.any(current.acting_player_mask), execute,
                            lambda c: c, current)

        state = lax.fori_loop(0, 2, resolve, state)
        def finish(current):
            before = current.battle.replace(event_source=jnp.int32(-1), event_move=jnp.int32(0), event_cause=jnp.int32(Cause.FIELD))
            key, battle = step_field(current.key, before)
            battle = capture_conditions(before, battle)
            return current.replace(key=key, battle=battle, turn=current.turn + 1, field_pending=jnp.bool_(False))
        state = lax.cond((state.cursor == 2) & ~jnp.any(state.acting_player_mask) & ~finished_replacement,
                         finish, lambda c: c, state)
        needs = (state.battle.active.current_hp[:, 0] <= 0) & jnp.any((state.battle.team.current_hp[..., 0] > 0) & state.selected, axis=1)
        waiting = jnp.any(state.acting_player_mask)
        needs = jnp.where(waiting, state.acting_player_mask, needs)
        state = state.replace(phase=jnp.where(jnp.any(needs), Phase.REPLACEMENT, Phase.TURN),
                              acting_player_mask=jnp.where(jnp.any(needs), needs, jnp.ones(2, bool)))
        revealed = state.revealed.at[jnp.arange(2), state.battle.active_index[:, 0]].set(True)
        return state.replace(revealed=revealed)


    def step_env(
        self,
        key: chex.PRNGKey,
        state: BattleState,
        actions: jax.Array,
        params: BattleParams,
    ) -> tuple[jax.Array, BattleState, jax.Array, jax.Array, dict[str, Any]]:
        """Run the legacy transition using effective active abilities."""
        act1, act2 = actions
        is_move, index, is_tera, is_no_op = decode_action(actions)
        active = state.effective_active
        moves = active.moves[(0, 1), index]
        is_gale_wings = triple_and(
            moves.type==Type.FLYING,
            active.ability==AbilityEnum.GALE_WINGS,
            active.hp_percent==1
        )
        is_prankster = jnp.logical_and(
            moves.move_type==MoveType.STATUS,
            active.ability==AbilityEnum.PRANKSTER
        )
        is_triage = jnp.logical_and(
            moves.healing,
            active.ability==AbilityEnum.TRIAGE
        )
        priorities = moves.priority + is_triage + is_prankster + is_gale_wings
        priorities = priorities * is_move + (1-is_move) * 7
        first, second = action_order(key, state, priorities)
        key, state = step_action(key, state, act1, first)
        key, state = step_action(key, state, act2, second)

        key, state = step_field(key, state)
        # should return obs, state, reward, done, info
        return jnp.array([0]), state, jnp.array([0]), jnp.array([0]), {}


    def reset_env(
        self, key: chex.PRNGKey, params: BattleParams
    ) -> tuple[jax.Array, BattleState]:
        raise NotImplementedError("Use the Jumanji reset method")

def equal_priority(r, speeds, priorities):
    first = jnp.argmax(speeds)
    tie_val = jnp.less_equal(r, 0.5)
    return jax.lax.select(speeds[0]!=speeds[1], jnp.array([first, 1-first]), jnp.array([tie_val, 1-tie_val]))

def action_order(
    key: chex.PRNGKey,
    state: BattleState,
    priorities: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Order priority brackets, reversing effective Speed under Trick Room."""
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)

    priorities = jnp.array(priorities)
    diff_priorities = priorities[0] != priorities[1]
    speeds = state.boosted_stats[:, StatEnum.SPEED]
    speeds = jnp.where(state.trick_room_duration > 0, -speeds, speeds)
    diff_speeds = speeds[0] != speeds[1]
    first_prio = jnp.argmax(priorities)
    first_speed = jnp.argmax(speeds)
    tiebreak = jnp.less_equal(r, 0.5)
    first = (first_prio * diff_priorities +
             first_speed * jnp.logical_and(diff_speeds, 1-diff_priorities) +
             tiebreak * jnp.logical_and(1-diff_priorities, 1-diff_speeds))
    return first, 1-first

def decode_action(action):
    # takes int in [0, 15) and returns a tuple of is_move, index, is_tera
    # index is a move index in [0,4) if action is a move, and in [0,6) if its a switch
    # if is_move == False then the third index should be ignored
    is_move_action = jnp.less(action, 8)
    move_index = action % 4
    is_tera = (action >= 4) & (action < 8)
    switch_index = action - 8
    index = move_index * is_move_action + switch_index * (1 - is_move_action)
    is_no_op = jnp.equal(action, 14)
    return is_move_action, index, is_tera, is_no_op


def before_move(key: chex.PRNGKey, state: BattleState, player_idx, inclusive_paralysis=True,
                restriction_blocked=False, defrost_move=False):
    """Resolve represented status and volatile interruptions before an action.

    Return the advanced key, immutable state and a scalar interruption flag.
    The explicit threshold parameter preserves the existing normal-move and
    Struggle comparisons until their rules are unified separately.
    """
    actor = state.active[player_idx]
    old_sleep = actor.sleep_counter
    sleep = jnp.maximum(0, old_sleep - 1)
    # Sleep decrements before acting; a counter reaching zero wakes the user.
    wakes = (sleep == 0) & (old_sleep == 1)
    actor = actor.replace(sleep_counter=sleep, status=jnp.where(wakes, Status.NONE, actor.status))
    state = update_active(state, player_idx, actor)
    # Freeze is checked before flinch and restrictions. Defrost moves bypass
    # the check but thaw only when they actually execute.
    def freeze_check(args):
        """Give a frozen user a 20% chance to thaw before acting."""
        key, state = args
        key, subkey = random.split(key)
        mon = state.active[player_idx]
        thaw = random.uniform(subkey) < .2
        mon = mon.replace(status=jnp.where(thaw, Status.NONE, mon.status))
        state = update_active(state, player_idx, mon)
        return key, state, ~thaw
    frozen = (actor.status[0] == Status.FREEZE) & ~jnp.asarray(defrost_move).reshape(())
    key, state, freeze_blocked = lax.cond(frozen, freeze_check,
        lambda args: (args[0], args[1], jnp.bool_(False)), (key, state))
    higher_blocked = ((state.active.status[player_idx, 0] == Status.SLEEP) | freeze_blocked |
                      state.volatile_status.flinched[player_idx, 0] | jnp.asarray(restriction_blocked).reshape(()))
    def confusion_check(args):
        """Count down confusion and possibly apply its special self-hit."""
        key, state = args
        v = state.volatile_status
        remaining = jnp.maximum(v.confusion_turns[player_idx, 0] - 1, 0)
        state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player_idx, 0].set(remaining)))
        state = emit(state, EventKind.CONDITION, player_idx, detail=12, before=1, after=0, enabled=remaining == 0)
        def roll_hit(args):
            """Sample the one-third self-hit chance and a separate damage roll."""
            key, state = args
            key, chance_key, damage_key = random.split(key, 3)
            hit = random.randint(chance_key, (), 0, 100) < 33
            stats = state.raw_boosted_stats[player_idx]
            level = state.active.level[player_idx, 0]
            base = (((2 * level // 5 + 2) * 40 * stats[1] // jnp.maximum(1, stats[2])) // 50 + 2).astype(jnp.int32)
            # Showdown truncates to 16 bits before the ordinary random roll.
            damage = jnp.maximum(1, ((base % 65536) * random.randint(damage_key, (), 85, 101)) // 100)
            def self_hit(s):
                """Damage only the user; do not invoke opposing on-hit abilities."""
                from ninjax.held_effects import after_hp_change
                mon = s.active[player_idx]
                hp = jnp.maximum(0, mon.current_hp - damage)
                old_source, old_cause = s.event_source, s.event_cause
                s = s.replace(event_source=jnp.asarray(player_idx, jnp.int32), event_cause=jnp.int32(Cause.CONFUSION))
                s = emit(s, EventKind.HP, player_idx, before=mon.current_hp, after=hp, hp_change=hp-mon.current_hp,
                    hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
                    enabled=hp[0] != mon.current_hp[0])
                s = emit(s, EventKind.FAINT, player_idx, enabled=(mon.current_hp[0] > 0) & (hp[0] == 0))
                s = update_active(s, player_idx, mon.replace(current_hp=hp, is_alive=mon.is_alive & (hp > 0)))
                s = after_hp_change(s, player_idx)
                return s.replace(event_source=old_source, event_cause=old_cause)
            return key, lax.cond(hit, self_hit, lambda s: s, state), hit
        return lax.cond(remaining > 0, roll_hit,
            lambda args: (args[0], args[1], jnp.bool_(False)), (key, state))
    confused = (state.volatile_status.confusion_turns[player_idx, 0] > 0) & ~higher_blocked
    key, state, self_hit = lax.cond(confused, confusion_check,
        lambda args: (args[0], args[1], jnp.bool_(False)), (key, state))
    key, subkey = random.split(key)
    roll = random.uniform(subkey)
    fully_paralyzed = (state.active.status[player_idx] == Status.PARALYZE) & jnp.where(inclusive_paralysis, roll <= .25, roll < .25)
    interrupted = higher_blocked | self_hit | fully_paralyzed
    return key, state, interrupted.reshape(())

def step_move(
    key: chex.PRNGKey,
    state: BattleState,
    player_idx: int | jax.Array,
    index: int | jax.Array,
    is_tera: bool | jax.Array) -> tuple[jax.Array, BattleState]:
    """Resolve status, volatile and move restrictions before the chosen move."""
    from ninjax.restrictions import resolve_move
    original_index = index
    index, restriction_blocked = resolve_move(state, player_idx, index)
    v = state.volatile_status
    # Encore replaces an already ordered action without changing its priority.
    priority = jnp.where(v.queued_priority_valid[player_idx, 0], v.queued_priority[player_idx, 0],
                         state.active[player_idx].moves.priority[original_index, 0])
    state = state.replace(volatile_status=v.replace(
        queued_priority=v.queued_priority.at[player_idx, 0].set(priority),
        encore_overridden=v.encore_overridden.at[player_idx, 0].set(index != original_index)))
    attacker = state.active[player_idx]
    state = update_active(state, player_idx, attacker.replace(is_terastallized=jnp.bool([is_tera])))
    key, state, is_interrupted = before_move(key, state, player_idx, restriction_blocked=restriction_blocked,
                                             defrost_move=state.active[player_idx].moves[index].defrost)
    key, state = jax.lax.cond(
        is_interrupted.reshape(()), move_interrupted, move_used,
        key, state, player_idx, index)
    return key, state


def step_switch(
    key: chex.PRNGKey,
    state: BattleState,
    player_index: int | jax.Array,
    index: int | jax.Array,
    is_tera: bool | jax.Array
) -> tuple[jax.Array, BattleState]:
    # switch needs to access the battle state because opponent switching triggers annoying things
    # TODO: add an opponent switched field somewhere for stakeout + analytic
    key, switch_key = random.split(key)
    return key, swap_out(switch_key, state, player_index, index)


def step_action(
    key: chex.PRNGKey,
    state: BattleState,
    action: int | jax.Array | BattleAction,
    player_index: int | jax.Array
) -> tuple[jax.Array, BattleState]:
    """Execute a selected move, switch, or forced Struggle for one player."""
    if not isinstance(action, BattleAction):
        action = BattleAction.from_legacy(action)
    state = state.replace(event_source=jnp.asarray(player_index, jnp.int32),
        event_cause=jnp.int32(Cause.MOVE), event_move=jnp.int32(0))
    # this will execute whatever move is selected
    def struggle(k, b):
        """Apply before-move interruptions before a forced Struggle."""
        k, b, interrupted = before_move(k, b, player_index, inclusive_paralysis=False)
        return lax.cond(interrupted,
            lambda k, s: move_interrupted(k, s, player_index, 0), execute_struggle, k, b)

    def execute_struggle(k, b):
        """Resolve Struggle and invalidate the previous move for Encore/Disable."""
        from ninjax.hit import HitResults
        v = b.volatile_status
        b = b.replace(hit_results=HitResults.empty(),
                      volatile_status=v.replace(last_move_id=v.last_move_id.at[player_index, 0].set(0)))
        b = b.replace(event_move=jnp.int32(-1))
        b = emit(b, EventKind.MOVE, player_index, detail=-1)
        move = Move(move_type=jnp.array([0]), max_pp=jnp.array([1]), current_pp=jnp.array([1]),
                    type=jnp.array([Type.NONE]), base_power=jnp.array([50]), accuracy=jnp.array([1.]),
                    priority=jnp.array([0]), offensive_stat=jnp.array([StatEnum.ATTACK]),
                    defensive_stat=jnp.array([StatEnum.DEFENSE]), crit_stage=jnp.array([0]))
        k, b = do_damaging_move(k, b, player_index, move, 0, 0)
        actor = b.active[player_index]
        hp = jnp.maximum(0, actor.current_hp - jnp.maximum(1, jnp.round(actor.max_hp / 4)))
        b = b.replace(event_cause=jnp.int32(Cause.RECOIL))
        b = emit(b, EventKind.HP, player_index, before=actor.current_hp, after=hp,
                 hp_change=hp-actor.current_hp,
                 hp_fraction_change=(jnp.ceil(hp / actor.max_hp * 100) -
                                     jnp.ceil(actor.current_hp / actor.max_hp * 100)) / 100,
                 enabled=hp != actor.current_hp)
        b = emit(b, EventKind.FAINT, player_index, enabled=(actor.current_hp > 0) & (hp == 0))
        b = update_active(b, player_index, actor.replace(current_hp=hp, is_alive=hp > 0))
        b = b.replace(winner=jnp.where(~jnp.any((b.team.current_hp[..., 0] > 0) & b.roster), player_index, b.winner))
        return k, b
    return lax.cond(action.transformation == Transformation.NONE,
        lambda k, b: lax.cond((action.move_type == ActionType.ATTACK) & (action.index == 4), struggle,
            lambda k, b: _ordinary_action(k, b, action, player_index), k, b),
        lambda k, b: (k, b), key, state)


def _ordinary_action(key, state, action, player_index):
    """Dispatch attacks, switches and state-dependent forced continuations."""
    return lax.cond(action.move_type == ActionType.CONTINUED,
        lambda k, s: step_continued(k, s, player_index),
        lambda k, s: lax.cond((action.move_type == ActionType.ATTACK) | (action.move_type == ActionType.SWITCH),
        lambda k, s: lax.cond(action.move_type == ActionType.ATTACK,
            step_move, step_switch, k, s, player_index, action.index, False),
        lambda k, s: (k, s), k, s), key, state)


def step_continued(key, state, player):
    """Spend recharge or attempt the stored charge move without another PP cost."""
    from ninjax.hit import HitResults
    from ninjax.game_logic import _apply_move_effects
    def recharge(k, s):
        """Recharge precedes sleep, freeze, flinch, confusion and paralysis checks."""
        v = s.volatile_status
        s = s.replace(hit_results=HitResults.empty(), volatile_status=v.replace(
            recharging=v.recharging.at[player, 0].set(False),
            protect_chain=v.protect_chain.at[player, 0].set(0)))
        return k, emit(s, EventKind.CONDITION, player, detail=15, before=1, after=0)
    def resume(k, s):
        """Release a charge once, including when an interruption prevents the hit."""
        v = s.volatile_status
        identifier = v.charging_move_id[player, 0]
        matches = (s.active[player].moves.identifier[:, 0] == identifier) & (identifier != 0)
        index = jnp.argmax(matches)
        s = s.replace(hit_results=HitResults.empty(), event_move=identifier,
            volatile_status=v.replace(charging_move_id=v.charging_move_id.at[player, 0].set(0),
                charge_target_side=v.charge_target_side.at[player, 0].set(0),
                charge_target_position=v.charge_target_position.at[player, 0].set(0),
                protect_chain=v.protect_chain.at[player, 0].set(0),
                encore_overridden=v.encore_overridden.at[player, 0].set(False)))
        s = emit(s, EventKind.CONDITION, player, detail=14, before=identifier, after=0)
        move = s.active[player].moves[index]
        # Forced continuations ignore Encore and PP, but Disable can stop them.
        blocked = ~jnp.any(matches) | ((v.disable_turns[player, 0] > 0) & (v.disabled_move_id[player, 0] == identifier))
        k, s, interrupted = before_move(k, s, player, restriction_blocked=blocked, defrost_move=move.defrost)
        def execute(k, s):
            """Record a released move and run its normal hit resolution."""
            s = emit(s, EventKind.MOVE, player, detail=identifier)
            return _apply_move_effects(k, s, player, index)
        return lax.cond(interrupted, lambda k, s: move_interrupted(k, s, player, index), execute, k, s)
    return lax.cond(state.volatile_status.recharging[player, 0], recharge,
        lambda k, s: lax.cond(s.volatile_status.charging_move_id[player, 0] != 0, resume, lambda k, s: (k, s), k, s), key, state)

def step_field(
    key: chex.PRNGKey,
    state: BattleState,
) -> tuple[jax.Array, BattleState]:
    # there is probably some reason we need rng or actions but idk rn
    weather_duration = jnp.maximum(state.weather.duration - 1, 0)
    new_weather = jnp.where(weather_duration > 0, state.weather.weather, 0)
    terrain_duration = jnp.maximum(state.terrain.duration - 1, 0)
    new_terrain = jnp.where(terrain_duration > 0, state.terrain.terrain, 0)

    key, state = step_side_conditions(key, state)

    # weather, terrain, status, items (leftovers etc)
    state = end_turn_damage(state)

    from ninjax.held_effects import leftovers, refresh_paradox
    state = leftovers(state)
    key, state = step_moody(key, state)
    # Perish Song resolves at the end of the turn, using effective speed order.
    speeds = state.boosted_stats[:, StatEnum.SPEED]
    effective = jnp.where(state.trick_room_duration > 0, -speeds, speeds)
    expiring = state.perish_turns[:, 0] == 1
    for player in range(2):
        state = lax.cond(expiring[player],
            lambda s: take_damage_value(s, player, s.active.current_hp[player], False), lambda s: s, state)
    key, tie_key = random.split(key)
    slowest = jnp.where(effective[0] == effective[1], random.randint(tie_key, (), 0, 2), jnp.argmin(effective))
    both_out = ~jnp.any((state.team.current_hp[..., 0] > 0) & state.roster)
    state = state.replace(perish_turns=jnp.where(state.perish_turns > 0, state.perish_turns-1, state.perish_turns),
                          winner=jnp.where(jnp.all(expiring) & both_out, slowest, state.winner))

    state = state.replace(
        turn_number=state.turn_number + 1,
        weather=Weather(new_weather, weather_duration),
        terrain=Terrain(new_terrain, terrain_duration),
        trick_room_duration=jnp.maximum(state.trick_room_duration - 1, 0),
        gravity_duration=jnp.maximum(state.gravity_duration - 1, 0),
    )

    return key, refresh_paradox(state)
