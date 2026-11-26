"""Jumanji battle environment, observations, legal masks and episode outcomes."""
from functools import cached_property, partial
from typing import Any

import chex
import jax.lax
import jax.numpy as jnp
from jax import lax
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from ninjax.battle.actions import decode_action, step_action
from ninjax.battle.field import step_field
from ninjax.battle.ordering import action_order
from ninjax.battle.scheduler import advance_turn
from ninjax.battle.switching import swap_is_alive
from ninjax.battle.types import BattleAction, BattleParams, EpisodeState, Phase, Transformation
from ninjax.calculations.logic import triple_and
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import Type
from ninjax.observations import Observation, observation_spec, observe
from ninjax.pokemon import Pokemon
from ninjax.records.events import EventBuffer
from ninjax.restrictions import allowed_moves
from ninjax.side.item_effects import refresh_paradox
from ninjax.side.queries import is_choice_item, is_floating_fields
from ninjax.side.state import BattleState


def _select_preview_leads(current, *, selection, self):
    """Start a selected battle with a fresh transition event buffer."""
    current = current.replace(battle=current.battle.replace(events=EventBuffer.empty()))
    indices = jnp.argmax(selection, axis=1).astype(jnp.int32)
    battle = current.battle.replace(active_index=indices[:, None], roster=selection)
    revealed = jnp.zeros((2, 6), bool).at[jnp.arange(2), indices].set(True)
    current = current.replace(battle=battle, selected=selection, revealed=revealed,
                              phase=jnp.int32(Phase.TURN))
    return self._activate_entry_abilities(current)


def _keep_episode(x):
    """Preserve the episode when a conditional scheduler branch is skipped."""
    return x


def _preview_timestep(*, valid_selection, choose, state, self):
    """Build the preview transition and report an invalid lead selection."""
    return (lax.cond(valid_selection, choose, _keep_episode, state),
                                 transition(jnp.zeros(2), self.observe(lax.cond(valid_selection, choose, _keep_episode, state)),
                                            shape=(2,), extras={**self._extras(lax.cond(valid_selection, choose, _keep_episode, state)),
                                                              "invalid_action": ~valid_selection}))


def _step_encoded_actions(*, self, state, encoded):
    """Dispatch ordinary encoded actions after team preview has finished."""
    return self.step(state, encoded)


def _advance_accepted_actions(current, chosen, *, self):
    """Clear previous transition events only when an action is accepted."""
    current = current.replace(battle=current.battle.replace(events=EventBuffer.empty()))
    return self._advance(current, chosen)


def _keep_rejected_episode(current, _):
    """Preserve the episode and its events when an action is rejected."""
    return current


def _terminal_timestep(*, reward, obs, extras):
    """Report the final observation and reward for a defeated team."""
    return termination(reward, obs, shape=(2,), extras=extras)


def _timeout_timestep(*, reward, obs, extras):
    """Report a truncated draw when the turn limit is reached."""
    return truncation(reward, obs, shape=(2,), extras=extras)


def _continuing_timestep(*, reward, obs, extras):
    """Report the observation and masks for the next decision."""
    return transition(reward, obs, shape=(2,), extras=extras)


def _nonterminal_timestep(*, timeout, reward, extras, obs):
    """Choose a timeout draw or ordinary nonterminal transition."""
    return lax.cond(timeout,
                    partial(_timeout_timestep, reward=reward, obs=obs, extras=extras),
                    partial(_continuing_timestep, reward=reward, obs=obs, extras=extras))


class Battle(Environment):

    """Battle."""
    def __init__(self, teams: Pokemon | None = None, params: BattleParams | None = None):
        # Legacy adapter: moves 0–3, Struggle 4, reserved 5–7,
        # switches 8–13, inactive no-op 14. New callers can use BattleAction.
        """Init."""
        self.action_set = jnp.array(range(15))
        self.teams = teams
        self.params = params or BattleParams()
        super().__init__()
        # idk
        #self.obs_shape = (1, 1)

    @property
    def name(self) -> str:
        """Name."""
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
        """Observation spec."""
        return observation_spec()

    def observe(self, state):
        """Observe."""
        return observe(self, state)

    @cached_property
    def reward_spec(self):
        """Reward spec."""
        return specs.Array((2,), jnp.dtype("float32"), "reward")

    @cached_property
    def discount_spec(self):
        """Discount spec."""
        return specs.BoundedArray((2,), jnp.dtype("float32"), 0., 1., "discount")

    def _extras(self, state):
        """Extras."""
        return {"acting_player_mask": state.acting_player_mask,
                "legal_action_mask": self.legal_action_mask(state), "phase": state.phase,
                "invalid_action": jnp.bool_(False)}

    def legal_action_mask(self, state):
        """Mask moves and switches for the phase, including represented trapping."""
        battle = state.battle
        alive = (battle.team.current_hp[..., 0] > 0) & state.selected
        switches = alive & (jnp.arange(6)[None, :] != battle.active_index)
        ability = battle.ability
        hp = battle.active_field('current_hp')
        base_types = battle.active_field('type_list')
        types = battle.effective_types()
        items = battle.active_field('item')[:, 0]
        categories = battle.active_move_field('move_type')[..., 0]
        moves = (battle.active_move_field('current_pp')[..., 0] > 0) & ~battle.active_move_field('disabled')[..., 0]

        moves &= allowed_moves(battle.active_move_field('identifier')[..., 0], categories,
            disabled_move_id=battle.volatile_status.disabled_move_id,
            disable_turns=battle.volatile_status.disable_turns,
            taunt_turns=battle.volatile_status.taunt_turns,
            encore_move_id=battle.volatile_status.encore_move_id,
            encore_turns=battle.volatile_status.encore_turns,
            gravity_duration=battle.gravity_duration,
            throat_chop_turns=battle.volatile_status.throat_chop_turns,
            psychic_noise_turns=battle.volatile_status.psychic_nouse_turns,
            is_sound=moves.sound,
            is_healing=moves.healing)


        choice = is_choice_item(items)
        lock = battle.volatile_status.choice_lock[:, 0]
        moves &= ~choice[:, None] | (lock[:, None] < 0) | (jnp.arange(4)[None, :] == lock[:, None])
        moves &= (items[:, None] != ItemEnum.ASSAULT_VEST) | (categories != MoveType.STATUS)
        trapped = (ability[::-1, 0] == AbilityEnum.SHADOW_TAG) & (hp[::-1, 0] > 0) & (ability[:, 0] != AbilityEnum.SHADOW_TAG) & ~jnp.any(base_types == Type.GHOST, axis=1)

        opposing = ability[::-1, 0]
        ghost = jnp.any(types == Type.GHOST, axis=1)
        grounded = (~is_floating_fields(types, ability, items[:, None]) | (battle.gravity_duration > 0))[:, 0]
        arena = (opposing == AbilityEnum.ARENA_TRAP) & grounded
        magnet = (opposing == AbilityEnum.MAGNET_PULL) & jnp.any(types == Type.STEEL, axis=1)
        trapped |= (arena | magnet) & ~ghost & (hp[::-1, 0] > 0)
        switches &= ~trapped[:, None] | (state.phase != Phase.TURN)
        forced = ~jnp.any(moves, axis=1) & (state.phase == Phase.TURN)
        moves &= state.phase == Phase.TURN
        mask = jnp.concatenate((moves, jnp.concatenate((forced[:, None], jnp.zeros((2, 3), bool)), axis=1), switches,
                                jnp.zeros((2, 1), bool)), axis=1)
        v = battle.volatile_status
        continuing = ((v.charging_move_id[:, 0] != 0) | v.recharging[:, 0]) & (state.phase == Phase.TURN) & (hp[:, 0] > 0)
        mask = jnp.where(continuing[:, None], jnp.arange(15)[None, :] == 5, mask)
        # Inactive players supply a no-op; their actual input is ignored.
        return jnp.where(state.acting_player_mask[:, None], mask,
                         jnp.arange(15)[None, :] == 14)

    def reset(self, key: jax.Array) -> tuple[EpisodeState, TimeStep[Observation]]:
        """Initialize caller-provided teams before the first turn's entry abilities."""
        if self.teams is None:
            raise ValueError("Battle requires caller-provided teams for reset")
        state = EpisodeState(key=key, battle=BattleState(team=self.teams), turn=jnp.int32(0),
                             phase=jnp.int32(Phase.TURN), acting_player_mask=jnp.ones(2, bool),
                             pending_actions=BattleAction.from_legacy(jnp.full((2,), 14, jnp.int32)),
                             queued_actor_slot=jnp.zeros((2,), jnp.int32),
                             order=jnp.arange(2, dtype=jnp.int32), cursor=jnp.int32(2), field_pending=jnp.bool_(False), selected=jnp.ones((2, 6), bool),
                             revealed=jnp.zeros((2, 6), bool).at[:, 0].set(not self.params.team_preview),
                             revealed_moves=jnp.zeros((2, 6, 4), bool))
        # Initial leads entered before turn zero. Ordinary switches record the
        # current turn instead, so Speed Boost can distinguish their timing.
        state = state.replace(battle=state.battle.replace(volatile_status=state.battle.volatile_status.replace(
            entry_turn=-jnp.ones((2, 1), jnp.int32))))
        state = state.replace(phase=jnp.int32(Phase.TEAM_SELECTION if self.params.team_preview else Phase.TURN))
        if not self.params.team_preview:
            state = self._activate_entry_abilities(state)
        return state, restart(self.observe(state), shape=(2,), extras=self._extras(state))

    def _activate_entry_abilities(self, state):
        """Resolve initial abilities in speed order, including team preview starts."""


        first, second = action_order(state.key, state.battle, jnp.zeros(2, jnp.int32))
        battle = swap_is_alive(state.battle, first)
        battle = swap_is_alive(battle, second)
        return state.replace(battle=refresh_paradox(battle))

    def step(self, state: EpisodeState, action: chex.Array | BattleAction) -> tuple[EpisodeState, TimeStep[Observation]]:
        """Validate indexed actions and advance the symmetric battle scheduler."""
        if isinstance(action, BattleAction):
            selection = action.selection.astype(bool)
            selecting = state.phase == Phase.TEAM_SELECTION
            valid_selection = jnp.all(jnp.any(selection, axis=1)) & jnp.all(action.transformation == Transformation.NONE)
            encoded = action.encode()
            choose = partial(_select_preview_leads, selection=selection, self=self)
            if self.params.team_preview:
                return lax.cond(selecting,
                    partial(_preview_timestep, valid_selection=valid_selection, choose=choose, state=state, self=self),
                    partial(_step_encoded_actions, self=self, state=state, encoded=encoded))
        else:
            action = BattleAction.from_legacy(action)
        encoded = action.encode()
        mask = self.legal_action_mask(state)
        valid = (state.phase != Phase.TEAM_SELECTION) & (encoded >= 0) & (encoded < 15)
        valid &= mask[jnp.arange(2), jnp.clip(encoded, 0, 14)]
        valid = jnp.all(valid | ~state.acting_player_mask)
        advance = partial(_advance_accepted_actions, self=self)
        state = lax.cond(valid & (state.phase != Phase.TERMINAL), advance,
                         _keep_rejected_episode, state, action)
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
            partial(_terminal_timestep, reward=reward, obs=obs, extras=extras),
            partial(_nonterminal_timestep, timeout=timeout, reward=reward, extras=extras, obs=obs))
        return state, timestep

    def _advance(self, state, actions):
        """Advance the pure simultaneous-action scheduler."""
        return advance_turn(state, actions)


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
        """Reset env."""
        raise NotImplementedError("Use the Jumanji reset method")
