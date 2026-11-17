"""Multi hit helpers for immutable JAX battle state."""
import chex
import jax.numpy as jnp
import jax.random as random
import jax
from ninjax.side.state import BattleState
from ninjax.moves.model import Move
from typing import Tuple
from ninjax.enum_types import AbilityEnum, ItemEnum, StatEnum, Status
from ninjax.moves.properties import move_accuracy
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.events import EventKind, emit
from ninjax.hit import HitOutcome, record_hit
from functools import partial
from ninjax.side.updates import conditional_add_boosts
from ninjax.side.stat_effects import conditional_reduce_boosts
from ninjax.moves.finalization import finish_damaging_move

def do_damaging_move(key: chex.PRNGKey, state: BattleState, attacker_idx, move: Move, stat_index, boost_value) -> Tuple[chex.PRNGKey, BattleState]:
    """Resolve a damaging move and finalize it once.

    ``attacker_idx`` is a player index in a singles battle; ``stat_index`` and
    ``boost_value`` are unused operands shared with the other hit branches.
    Critical-hit and damage-roll keys are split independently for each hit.
    Per-hit accuracy and hit-count policy are carried by the move metadata.
    """
    return resolve_multi_hit(key, state, attacker_idx, move)


def sample_hit_count(key, rule, ability, item):
    """Sample curated hit-count policies; return an advanced key and count."""
    key, count_key, extra_key = random.split(key, 3)
    roll = random.randint(count_key, (), 0, 20)
    ordinary = jnp.where(roll < 7, 2, jnp.where(roll < 14, 3, jnp.where(roll < 17, 4, 5)))
    dice = item == ItemEnum.LOADED_DICE
    skill = ability == AbilityEnum.SKILL_LINK
    standard = jnp.where(skill, 5, jnp.where(dice, random.randint(extra_key, (), 4, 6), ordinary))
    population = jnp.where(dice, random.randint(extra_key, (), 4, 11), 10)
    return key, jnp.where(rule == 1, 1, jnp.where(rule == 2, standard, jnp.where(rule == 3, 3, population)))


def _single_hit_count(k):
    """Select one hit without consuming a hit-count key."""
    return (k, jnp.int32(1))


def _sample_move_hit_count(k, *, move, attacker):
    """Sample hits using the move rule and effective attacker modifiers."""
    return sample_hit_count(k, move.hit_rule[0], attacker.ability[0], attacker.item[0])


def _check_multi_hit_accuracy(k, *, state, attacker_idx, move):
    """Advance the key for this independent accuracy check."""
    k, subkey = random.split(k)
    return k, random.uniform(subkey) < move_accuracy(state, attacker_idx, move)[0]


def _apply_multi_hit(args, *, move, index, contributors, attacker_idx, beat_up, no_recoil):
    """Apply this hit at its indexed base power."""
    key, state, total, hits = args
    power = jnp.where(move.escalating_power, move.base_power * (index+1), move.base_power)
    contributor = contributors[jnp.minimum(index, 5)]
    base_attack = state.team.stat_table.base_stats[attacker_idx, contributor, StatEnum.ATTACK]
    power = jnp.where(beat_up, 5 + base_attack // 10, power)
    one = no_recoil.replace(base_power=power)
    key, state, damage, _ = resolve_damaging_hit(key, state, attacker_idx, one, index)
    return key, state, total + damage, hits + 1


def _emit_multi_hit_miss(s, *, attacker_idx):
    """Announce a failed accuracy check on a later hit."""
    return emit(s, EventKind.MISS, 1-attacker_idx)


def _record_multi_hit_miss(s, *, index, attacker_idx):
    """Record a later-hit miss at its attempt index."""
    return record_hit(s, index, 1-attacker_idx, HitOutcome.MISS)


def _attempt_multi_hit(args, *, per_hit, index, move, beat_up, attacker_idx, no_recoil, contributors):
    """Check later-hit accuracy and apply its result."""
    key, state, total, hits = args
    key, hit = jax.lax.cond(
        per_hit & (index > 0),
        partial(_check_multi_hit_accuracy, state=state, attacker_idx=attacker_idx, move=move),
        _successful_check,
        key,
    )
    key, state, total, hits = jax.lax.cond(
        hit,
        partial(
            _apply_multi_hit,
            move=move,
            index=index,
            contributors=contributors,
            attacker_idx=attacker_idx,
            beat_up=beat_up,
            no_recoil=no_recoil,
        ),
        _identity,
        (key, state, total, hits),
    )
    state = jax.lax.cond(~hit, partial(_emit_multi_hit_miss, attacker_idx=attacker_idx), _identity, state)
    state = jax.lax.cond(
        ~hit,
        partial(_record_multi_hit_miss, index=index, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    return key, state, hit, total, hits


def _stop_multi_hit(x):
    """Stop subsequent hits while retaining accumulated damage and hit count."""
    return (x[0], x[1], jnp.bool_(False), x[2], x[3])


def _multi_hit_scan(
    carry,
    index,
    *,
    attacker_idx,
    target_hits,
    move,
    beat_up,
    no_recoil,
    per_hit,
    contributors,
):
    """Execute at most one hit, retaining state when the move has stopped."""
    key, state, alive, total, hits = carry
    # Contact can induce sleep between hits; a move called by Sleep Talk
    # is allowed to continue while its user is asleep.
    awake = (index == 0) | (state.active.status[attacker_idx, 0] != Status.SLEEP) | state.volatile_status.sleep_talk_active[attacker_idx, 0]
    can_hit = (state.active.current_hp[attacker_idx, 0] > 0) & (state.active.current_hp[1-attacker_idx, 0] > 0) & awake
    valid = alive & (index < target_hits) & (((move.hit_rule[0] == 1) & ~beat_up) | can_hit)
    return jax.lax.cond(
        valid,
        partial(
            _attempt_multi_hit,
            per_hit=per_hit,
            index=index,
            move=move,
            beat_up=beat_up,
            attacker_idx=attacker_idx,
            no_recoil=no_recoil,
            contributors=contributors,
        ),
        _stop_multi_hit,
        (key, state, total, hits),
    ), None


def _scale_shot_boosts(s, *, attacker_idx):
    """Lower Defense and raise Speed once after a successful Scale Shot."""
    return conditional_add_boosts(conditional_reduce_boosts(s, attacker_idx, True, StatEnum.DEFENSE, 1, False),
                                             attacker_idx, True, StatEnum.SPEED, 1)


def _roll_self_effect(k, *, move):
    """Sample a once-per-move self effect with the explicit battle key."""
    k, subkey = random.split(k)
    return k, random.uniform(subkey) < move.self_chance[0]


def resolve_multi_hit(key, state: BattleState, attacker_idx, move: Move):
    """Resolve up to ten hits in a fixed scan; initial accuracy was already checked."""
    attacker = state.effective_active[attacker_idx]
    defender = state.effective_active[1-attacker_idx]
    beat_up = move.identifier[0] == 251
    # Beat Up includes the user even when statused; eligible teammates must
    # be selected, alive and status-free. Preserve party order for its hits.
    slots = jnp.arange(6)
    eligible = state.roster[attacker_idx] & ((slots == state.active_index[attacker_idx, 0]) |
        ((state.team.current_hp[attacker_idx, :, 0] > 0) & (state.team.status[attacker_idx, :, 0] == Status.NONE)))
    contributors = jnp.nonzero(eligible, size=6, fill_value=0)[0]
    key, target_hits = jax.lax.cond(
        move.hit_rule[0] == 1,
        _single_hit_count,
        partial(_sample_move_hit_count, move=move, attacker=attacker),
        key,
    )
    target_hits = jnp.where(beat_up, jnp.sum(eligible), target_hits)
    # Loaded Dice and Skill Link remove multiaccuracy in the pinned generation.
    per_hit = move.per_hit_accuracy[0] & (attacker.ability[0] != AbilityEnum.SKILL_LINK) & (attacker.item[0] != ItemEnum.LOADED_DICE)
    no_recoil = move.replace(recoil_percent=jnp.where(move.hit_rule == 1, move.recoil_percent, 0))
    (key, state, _, damage, hits), _ = jax.lax.scan(
        partial(
            _multi_hit_scan,
            attacker_idx=attacker_idx,
            target_hits=target_hits,
            move=move,
            beat_up=beat_up,
            no_recoil=no_recoil,
            per_hit=per_hit,
            contributors=contributors,
        ),
        (key, state, jnp.bool_(True), jnp.zeros((1,), jnp.int32), jnp.int32(0)),
        jnp.arange(10),
    )
    # The currently supported multi-hit moves have no recoil. Calculated damage
    # is retained here for future recoil support; actual HP loss is not yet
    # aggregated across hits.
    # Scale Shot (Showdown move ID 799) lowers Defense and raises Speed after hitting.
    state = jax.lax.cond(
        (move.identifier[0] == 799) & (hits > 0) & (state.active.current_hp[attacker_idx, 0] > 0),
        partial(_scale_shot_boosts, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    # One roll gates the whole self-boost vector, allowing future omniboost
    # moves. Guaranteed changes consume no additional random key.
    key, self_effect = jax.lax.cond(
        (move.self_chance[0] < 1) & jnp.any(move.self_stages != 0),
        partial(_roll_self_effect, move=move),
        _successful_check,
        key,
    )
    return key, finish_damaging_move(state, attacker_idx, move, damage, defender, self_effect)


def _successful_check(k):
    """Return success without consuming a random key."""
    return (k, jnp.bool_(True))


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x
