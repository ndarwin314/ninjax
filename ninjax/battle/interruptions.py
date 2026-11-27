"""Sleep, freeze, confusion, paralysis and pre-action interruptions."""
from functools import partial

import chex
import jax.numpy as jnp
from jax import lax, random

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import Status
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.item_effects import after_hp_change
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


def _before_move_freeze_check(args, *, player_idx):
    """Give a frozen user a 20% chance to thaw before acting."""
    key, state = args
    key, subkey = random.split(key)
    mon = state.active[player_idx]
    thaw = random.uniform(subkey) < .2
    mon = mon.replace(status=jnp.where(thaw, Status.NONE, mon.status))
    state = update_active_fields(state, player_idx, status=mon.status)
    return key, state, ~thaw


def _before_move_result(args):
    """
    Skip the freeze roll and preserve state with no freeze interruption.
    Continue without a self-hit after confusion expires.
    Skip confusion when a higher-priority condition already interrupted.
    """
    return (args[0], args[1], jnp.bool_(False))


def _before_move_self_hit(s, *, player_idx, damage):
    """Damage only the user; do not invoke opposing on-hit abilities."""

    mon = s.active[player_idx]
    hp = jnp.maximum(0, mon.current_hp - damage)
    old_source, old_cause = s.event_context.source, s.event_context.cause
    s = s.replace(
        event_context=s.event_context.replace(
            source=jnp.asarray(player_idx, jnp.int32),
            cause=jnp.int32(Cause.CONFUSION)))
    s = emit(s, EventKind.HP, player_idx, before=mon.current_hp, after=hp, hp_change=hp-mon.current_hp,
        hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
        enabled=hp[0] != mon.current_hp[0])
    s = emit(s, EventKind.FAINT, player_idx, enabled=(mon.current_hp[0] > 0) & (hp[0] == 0))
    s = update_active_fields(s, player_idx, current_hp=hp, is_alive=mon.is_alive & (hp > 0))
    s = after_hp_change(s, player_idx)
    return s.replace(
        event_context=s.event_context.replace(
            source=old_source,
            cause=old_cause))


def _before_move_unchanged(s):
    """Preserve HP when the confusion self-hit roll fails."""
    return s


def _before_move_roll_hit(args, *, player_idx):
    """Sample the one-third self-hit chance and a separate damage roll."""
    key, state = args
    key, chance_key, damage_key = random.split(key, 3)
    hit = random.randint(chance_key, (), 0, 100) < 33
    stats = state.raw_boosted_stats[player_idx]
    level = state.active.level[player_idx, 0]
    base = (((2 * level // 5 + 2) * 40 * stats[1] // jnp.maximum(1, stats[2])) // 50 + 2).astype(jnp.int32)
    # Showdown truncates to 16 bits before the ordinary random roll.
    damage = jnp.maximum(1, ((base % 65536) * random.randint(damage_key, (), 85, 101)) // 100)
    self_hit = partial(_before_move_self_hit, player_idx=player_idx, damage=damage)
    return key, lax.cond(hit, self_hit, _before_move_unchanged, state), hit


def _before_move_confusion_check(args, *, player_idx):
    """Count down confusion and possibly apply its special self-hit."""
    key, state = args
    v = state.volatile_status
    remaining = jnp.maximum(v.confusion_turns[player_idx, 0] - 1, 0)
    state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player_idx, 0].set(remaining)))
    state = emit(state, EventKind.CONDITION, player_idx, detail=12, before=1, after=0, enabled=remaining == 0)
    roll_hit = partial(_before_move_roll_hit, player_idx=player_idx)
    return lax.cond(remaining > 0, roll_hit,
        _before_move_result, (key, state))


def before_move(key: chex.PRNGKey, state: BattleState, player_idx,
                restriction_blocked=False, defrost_move=False, sleep_usable=False):
    """Resolve represented status and volatile interruptions before an action.

    Return the advanced key, immutable state and a scalar interruption flag.
    """
    actor = state.active[player_idx]
    old_sleep = actor.sleep_counter
    sleep = jnp.maximum(0, old_sleep - jnp.where(state.ability[player_idx, 0] == AbilityEnum.EARLY_BIRD, 2, 1))
    # Sleep decrements before acting; a counter reaching zero wakes the user.
    wakes = (sleep == 0) & (old_sleep > 0)
    actor = actor.replace(sleep_counter=sleep, status=jnp.where(wakes, Status.NONE, actor.status))
    state = update_active_fields(state, player_idx, status=actor.status, sleep_counter=actor.sleep_counter)
    # Freeze is checked before flinch and restrictions. Defrost moves bypass
    # the check but thaw only when they actually execute.
    freeze_check = partial(_before_move_freeze_check, player_idx=player_idx)
    frozen = (actor.status[0] == Status.FREEZE) & ~jnp.asarray(defrost_move).reshape(())
    key, state, freeze_blocked = lax.cond(frozen, freeze_check,
        _before_move_result, (key, state))
    higher_blocked = (((state.active.status[player_idx, 0] == Status.SLEEP) & ~jnp.asarray(sleep_usable).reshape(())) | freeze_blocked |
                      state.volatile_status.flinched[player_idx, 0] | jnp.asarray(restriction_blocked).reshape(()))
    confusion_check = partial(_before_move_confusion_check, player_idx=player_idx)
    confused = (state.volatile_status.confusion_turns[player_idx, 0] > 0) & ~higher_blocked
    key, state, self_hit = lax.cond(confused, confusion_check,
        _before_move_result, (key, state))
    key, subkey = random.split(key)
    roll = random.uniform(subkey)
    fully_paralyzed = (state.active.status[player_idx] == Status.PARALYZE) & (roll < .25)
    interrupted = higher_blocked | self_hit | fully_paralyzed
    return key, state, interrupted.reshape(())
