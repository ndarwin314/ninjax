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

class BattleAction(struct.PyTreeNode):
    moves: jax.Array
    switches: jax.Array
    selection: jax.Array
    tera: jax.Array
    mega: jax.Array
    struggle: jax.Array = struct.field(default_factory=lambda: jnp.zeros(2, bool))


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
    pending_actions: jax.Array
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
        def boolean(shape):
            return specs.BoundedArray(shape, jnp.dtype("bool"), False, True)
        return specs.Spec(BattleAction, "action", moves=boolean((2, 4)), switches=boolean((2, 6)),
                          selection=boolean((2, 6)), tera=boolean((2,)), mega=boolean((2,)), struggle=boolean((2,)))

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
        battle = state.battle
        alive = (battle.team.current_hp[..., 0] > 0) & state.selected
        switches = alive & (jnp.arange(6)[None, :] != battle.active_index)
        active = battle.active
        moves = (active.moves.current_pp[..., 0] > 0) & ~active.moves.disabled[..., 0]
        moves &= (active.encore_move < 0) | (jnp.arange(4)[None, :] == active.encore_move)
        forced = ~jnp.any(moves, axis=1) & (state.phase == Phase.TURN)
        moves &= state.phase == Phase.TURN
        mask = jnp.concatenate((moves, jnp.concatenate((forced[:, None], jnp.zeros((2, 3), bool)), axis=1), switches,
                                jnp.zeros((2, 1), bool)), axis=1)
        # Inactive players supply a no-op; their actual input is ignored.
        return jnp.where(state.acting_player_mask[:, None], mask,
                         jnp.arange(15)[None, :] == 14)

    def reset(self, key: jax.Array) -> tuple[EpisodeState, TimeStep[Observation]]:
        if self.teams is None:
            raise ValueError("Battle requires caller-provided teams for reset")
        state = EpisodeState(key=key, battle=BattleState(team=self.teams), turn=jnp.int32(0),
                             phase=jnp.int32(Phase.TURN), acting_player_mask=jnp.ones(2, bool),
                             pending_actions=jnp.full((2,), 14, jnp.int32),
                             order=jnp.arange(2, dtype=jnp.int32), cursor=jnp.int32(2), field_pending=jnp.bool_(False), selected=jnp.ones((2, 6), bool),
                             revealed=jnp.zeros((2, 6), bool).at[:, 0].set(not self.params.team_preview),
                             revealed_moves=jnp.zeros((2, 6, 4), bool))
        state = state.replace(phase=jnp.int32(Phase.TEAM_SELECTION if self.params.team_preview else Phase.TURN))
        return state, restart(self.observe(state), shape=(2,), extras=self._extras(state))

    def step(self, state: EpisodeState, action: chex.Array | BattleAction) -> tuple[EpisodeState, TimeStep[Observation]]:
        state = state.replace(battle=state.battle.replace(events=EventBuffer.empty()))
        if isinstance(action, BattleAction):
            selection = action.selection.astype(bool)
            selecting = state.phase == Phase.TEAM_SELECTION
            valid_selection = jnp.all(jnp.any(selection, axis=1))
            choices = jnp.sum(action.moves, axis=1) + jnp.sum(action.switches, axis=1) + action.struggle
            valid_encoding = jnp.all((choices == 1) | ~state.acting_player_mask)
            valid_encoding &= ~jnp.any((action.tera | action.mega) & state.acting_player_mask)
            encoded = jnp.where(jnp.any(action.switches, axis=1),
                                8 + jnp.argmax(action.switches, axis=1), jnp.argmax(action.moves, axis=1))
            encoded = jnp.where(action.struggle, 4, encoded)
            def choose(current):
                indices = jnp.argmax(selection, axis=1).astype(jnp.int32)
                battle = current.battle.replace(active_index=indices[:, None], roster=selection)
                revealed = jnp.zeros((2, 6), bool).at[jnp.arange(2), indices].set(True)
                return current.replace(battle=battle, selected=selection, revealed=revealed,
                                       phase=jnp.int32(Phase.TURN))
            if self.params.team_preview:
                return lax.cond(selecting,
                    lambda: (lax.cond(valid_selection, choose, lambda x: x, state),
                             transition(jnp.zeros(2), self.observe(lax.cond(valid_selection, choose, lambda x: x, state)),
                                        shape=(2,), extras={**self._extras(lax.cond(valid_selection, choose, lambda x: x, state)),
                                                          "invalid_action": ~valid_selection})),
                    lambda: self.step(state, jnp.where(valid_encoding, encoded, -1)))
            action = jnp.where(valid_encoding, encoded, -1)
        action = jnp.asarray(action, jnp.int32)
        mask = self.legal_action_mask(state)
        valid = (state.phase != Phase.TEAM_SELECTION) & (action >= 0) & (action < 15)
        valid &= mask[jnp.arange(2), jnp.clip(action, 0, 14)]
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
        def select(current):
            key, order_key = random.split(current.key)
            indices = jnp.clip(actions % 4, 0, 3)
            priorities = current.battle.active.moves.priority[jnp.arange(2), indices, 0]
            priorities = jnp.where(actions == 4, 0, jnp.where(actions >= 8, 7, priorities))
            first, second = action_order(order_key, current.battle, priorities)
            return current.replace(key=key, pending_actions=actions,
                                   order=jnp.stack((first, second)), cursor=jnp.int32(0), field_pending=jnp.bool_(True),
                                   acting_player_mask=jnp.zeros(2, bool))

        def replace_active(current):
            for player in range(2):
                key, battle = lax.cond(current.acting_player_mask[player],
                    lambda k, b: step_switch(k, b, player, actions[player] - 8, False),
                    lambda k, b: (k, b), current.key, current.battle)
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
                can_act = current.battle.active.current_hp[player, 0] > 0
                key, battle = lax.cond(can_act,
                    lambda k, b: step_action(k, b, chosen, player),
                    lambda k, b: (k, b), current.key, current.battle)
                battle = capture_conditions(current.battle, battle)
                slot = current.battle.active_index[player, 0]
                known = current.revealed_moves.at[player, slot, jnp.clip(chosen, 0, 3)].set(
                    current.revealed_moves[player, slot, jnp.clip(chosen, 0, 3)] | ((chosen < 4) & can_act &
                    (battle.team.moves.current_pp[player, slot, jnp.clip(chosen, 0, 3), 0] <
                     current.battle.team.moves.current_pp[player, slot, jnp.clip(chosen, 0, 3), 0])))
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
        act1, act2 = actions
        is_move, index, is_tera, is_no_op = decode_action(actions)
        active = state.active
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
    # for now im just going to go off speed stats of active pokemon
    # obviously this needs to account for priority brackets later
    # key and actions arent used now but they will be needed for
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)

    priorities = jnp.array(priorities)
    diff_priorities = priorities[0] != priorities[1]
    speeds = state.boosted_stats[:, StatEnum.SPEED]
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

def step_move(
    key: chex.PRNGKey,
    state: BattleState,
    player_idx: int | jax.Array,
    index: int | jax.Array,
    is_tera: bool | jax.Array) -> tuple[jax.Array, BattleState]:
    # this, or something this calls is probably going to be the most complex function
    # for now im just going to implement a simplistic version
    # TODO: add the tera part of move
    # TODO: if condition for status moves before we do damage stuff

    # TODO: stuff to add
    # 1. glaive rush mult
    # 3. unaware for both
    # 4. guts/facade
    # 7. various crit damage and rate multipliers
    attacker = state.active[player_idx]

    # do tera stuff
    can_tera = state.can_tera.at[player_idx].set(~jnp.asarray(is_tera, bool))

    # do sleep stuff
    # more hacks to avoid branching
    old_sleep_counter = attacker.sleep_counter
    sleep_counter = jnp.maximum(0, old_sleep_counter - 1)
    # if we aren't asleep, sleep counter is already 0, so the diff is 0
    woken_up = jnp.logical_and(sleep_counter==0, old_sleep_counter-sleep_counter==1)
    new_status = attacker.status * (1-woken_up)

    attacker = attacker.replace(
        status=new_status,
        sleep_counter=sleep_counter,
        is_terastallized=jnp.bool([is_tera]))

    state = update_active(state, player_idx, attacker)
    # add check for if move happens because of flinch, sleep, paralysis, etc here
    key, sub_key = random.split(key, 2)
    r = random.uniform(sub_key)
    is_paralyzed = attacker.status==Status.PARALYZE
    is_fully_paralyzed = jnp.logical_and(jnp.less_equal(r, 0.25), is_paralyzed)
    is_sleeping = attacker.status==Status.SLEEP
    is_flinched = False
    is_interrupted = triple_or(is_flinched, is_sleeping, is_fully_paralyzed)
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
    action: int | jax.Array,
    player_index: int | jax.Array
) -> tuple[jax.Array, BattleState]:
    state = state.replace(event_source=jnp.asarray(player_index, jnp.int32),
        event_cause=jnp.int32(Cause.MOVE), event_move=jnp.int32(0))
    # this will execute whatever move is selected
    def struggle(k, b):
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
    return lax.cond(action == 4, struggle,
                    lambda k, b: _ordinary_action(k, b, action, player_index), key, state)


def _ordinary_action(key, state, action, player_index):
    is_move_action, index, is_tera, is_no_op = decode_action(action)
    # i think this is the best way to implement this conditional in jax
    return lax.cond(is_move_action, step_move, step_switch, key, state, player_index, index, is_tera)

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

    return key, state
