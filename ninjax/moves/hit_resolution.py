"""Hit resolution helpers for immutable JAX battle state."""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.calculations.damage import (
    compute_base_damage,
    compute_base_power,
    compute_damage_multipliers,
    hit_type_effectiveness,
)
from ninjax.calculations.modifiers import CRIT_STAGES
from ninjax.calculations.stats import raw_boosted_stats
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveType, OffensiveStatOwner, MoveResult
from ninjax.enums.pokemon import Status, Type
from ninjax.item_rules import item_is_suppressed
from ninjax.moves.effects import after_damage, after_every_hit, secondary_effects
from ninjax.moves.model import Move
from ninjax.moves.properties import substitute_intercepts
from ninjax.records.events import EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import after_hit, consume, remove_item
from ninjax.side.queries import (
    effective_weather,
    effective_speed,
    stat_modifier,
)
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


def _damage_substitute(s, *, target_idx, damage):
    """Remove capped Substitute HP without triggering Pokémon damage hooks."""
    v = s.volatile_status
    old = v.substitute_hp[target_idx, 0]
    loss = jnp.minimum(old, jnp.maximum(0, damage[0]))
    remaining = old - loss
    s = s.replace(volatile_status=v.replace(
        substitute_hp=v.substitute_hp.at[target_idx, 0].set(remaining),
        last_damage=v.last_damage.at[target_idx, 0].set(loss)))
    # Public events expose presence, never the private remaining HP.
    s = emit(s, EventKind.CONDITION, target_idx, detail=13, before=1, after=remaining > 0)
    return s


def _damage_hit_target(s, *, target_idx, damage, attacker_idx):
    """Commit calculated hit damage with its attacker's source attribution."""
    return take_damage_value(s, target_idx, damage, True, source=attacker_idx)


def _apply_drain(s, *, attacker_idx, defender_ability, healing):
    """Heal the attacker, or deal indirect damage for Liquid Ooze."""
    return take_damage_value(s, attacker_idx, jnp.where(defender_ability == AbilityEnum.LIQUID_OOZE, healing, -healing), False)


def _run_hit_effects(k, s, *, attacker_idx, move, is_crit, hit_index):
    """Apply per-hit move effects after damage reaches the Pokémon."""
    history = s.hit_results[:]
    total = jnp.sum(jnp.where(jnp.arange(history.damage.size) <= hit_index, history.damage, 0))
    return after_every_hit(k, s, attacker_idx, move, is_crit, history.damage[hit_index], total)


def _run_held_hit_effects(s, *, attacker_idx, move, damage, raw_effectiveness):
    """Apply held-item reactions using this hit and its effectiveness."""
    return after_hit(s, attacker_idx, move, damage, raw_effectiveness)


def _knock_off_item(s, *, target_idx, attacker_idx):
    """Remove the target item after a surviving Knock Off user hits."""
    return remove_item(s, target_idx, attacker_idx)[0]


def _run_secondary_effects(k, s, *, attacker_idx, move):
    """Roll secondary effects only for hits that reach the Pokémon."""
    return secondary_effects(k, s, attacker_idx, move)


def _run_after_damage(s, *, attacker_idx, move):
    """Apply move callbacks that run after damage or Substitute interception."""
    return after_damage(s, attacker_idx, move)


def resolve_damaging_hit(key, state: BattleState, attacker_idx, move: Move, hit_index=0):
    """Apply one hit; return key, state, calculated damage and defender snapshot.

    Calculated damage has shape (1,) and can exceed the recipient's HP.
    The hit record stores actual loss before healing. Move finalization uses
    that loss for recoil. The defender snapshot is from before the hit.
    """
    active = state.effective_active
    attacker = active[attacker_idx]
    defender = active[1-attacker_idx]
    # Damage calculation reads the same weather before any HP changes.
    weather = effective_weather(state.weather.weather, active.ability, active.current_hp)
    # Explosion's user has already fainted, but damage uses its pre-faint HP.
    pre_faint_hp = state.volatile_status.explosion_hp[attacker_idx]
    attacker = attacker.replace(current_hp=jnp.where((move.identifier == 153) & (pre_faint_hp > 0), pre_faint_hp, attacker.current_hp))


    gravity = state.gravity_duration > 0
    raw_effectiveness, damage_effectiveness = hit_type_effectiveness(
        move.type, defender.type_list, defender.ability, gravity, move.effectiveness_rule
    )
    # Base power and held-item reactions receive the raw type matchup.
    user_speed = effective_speed(state, attacker_idx, weather)
    target_speed = effective_speed(state, 1-attacker_idx, weather)
    power = compute_base_power(
        attacker,
        defender,
        move,
        state.terrain, weather, gravity, raw_effectiveness=raw_effectiveness,
        user_speed=user_speed,
        target_speed=target_speed,
        positive_stages=jnp.sum(jnp.maximum(state.boosts.normal_boosts[attacker_idx, 1:], 0)) +
                        jnp.sum(jnp.maximum(state.boosts.acc_boosts[attacker_idx], 0)),
        previous_failed=state.volatile_status.previous_move_result[attacker_idx] == MoveResult.FAILED,
        target_charging_move=state.volatile_status.charging_move_id[1-attacker_idx, 0],
        item_suppressed=item_is_suppressed(state.ability[attacker_idx], state.magic_room_duration,
                                            state.volatile_status.embargo_turns[attacker_idx]),
    )
    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_stage = (move.crit_stage+
                  (attacker.ability==AbilityEnum.SUPER_LUCK)[0] +
                  3*jnp.logical_and(attacker.ability==AbilityEnum.MERCILESS, defender.is_poisoned))
    crit_chance = CRIT_STAGES[crit_stage] * (defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = ((random.uniform(one) < crit_chance) | move.always_crit) & (defender.ability != AbilityEnum.BATTLE_ARMOR)
    state = emit(state, EventKind.CRITICAL, 1-attacker_idx, enabled=is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100

    # Select stage-adjusted stats before computing offensive and defensive
    # modifiers. Moves such as Body Press can select a nonstandard attack stat.
    raw_stats = active.stats
    boosts = state.boosts
    attacker_stats, defender_stats = raw_boosted_stats(
        raw_stats,
        boosts,
        attacker_idx,
        (attacker.ability==AbilityEnum.UNAWARE) | move.ignore_defensive,
        defender.ability==AbilityEnum.UNAWARE,
        is_crit,
        offensive_owner=jnp.where(move.identifier[0] == 492, OffensiveStatOwner.TARGET, OffensiveStatOwner.USER)
    )


    offensive_modifier = stat_modifier(state, attacker_idx, move.offensive_stat, weather)
    defensive_modifier = stat_modifier(state, 1 - attacker_idx, move.defensive_stat, weather)
    base_damage = compute_base_damage(
        attacker.ability,
        defender.ability,
        attacker.hp_percent,
        attacker.level,
        attacker_stats,
        defender_stats,
        move,
        power,
        attacker.status,
        defender.status,
        state.volatile_status.flash_fire[attacker_idx],
        offensive_modifier=offensive_modifier, defensive_modifier=defensive_modifier,
    )

    screen = jnp.where(move.move_type == MoveType.PHYSICAL, state.reflect[1-attacker_idx, 0] > 0,
                       state.light_screen[1-attacker_idx, 0] > 0) | (state.aurora_veil[1-attacker_idx, 0] > 0)
    charging = state.volatile_status.charging_move_id[1-attacker_idx, 0]
    ident = move.identifier[0]
    # Exceptional moves strike these charge states at double damage; Bounce
    # uses a separate BasePower callback instead of Fly's damage callback.
    semi_bonus = ((charging == 19) & ((ident == 16) | (ident == 239))) | (
        (charging == 91) & ((ident == 89) | (ident == 222))) | (
        (charging == 291) & ((ident == 57) | (ident == 250)))
    damage = compute_damage_multipliers(attacker, defender, weather, move,
        base_damage, is_crit, damage_roll, gravity,
        effectiveness=damage_effectiveness, screen=screen,
        user_speed=user_speed, target_speed=target_speed, semi_damage_boost=semi_bonus)
    # Gen 9 Unseen Fist bypasses Protect without reducing damage.

    # Fixed damage ignores damage modifiers (including screens), but reaches
    # this branch only after the normal type/ability/protection checks.
    ident = move.identifier[0]
    damage = jnp.where(ident == 101, attacker.level, damage)
    damage = jnp.where((ident == 162) | (ident == 877), jnp.maximum(1, defender.current_hp // 2), damage)
    damage = jnp.where(ident == 283, jnp.maximum(0, defender.current_hp-attacker.current_hp), damage)

    # Substitute intercepts this hit before Pokémon HP and on-hit callbacks.
    target_idx = 1-attacker_idx
    intercepted = substitute_intercepts(state.volatile_status.substitute_hp[1-(attacker_idx), 0], state.ability[attacker_idx, 0],
        move.target_self[0], move.sound[0], move.bypass_substitute[0])
    state = jax.lax.cond(
        intercepted,
        partial(_damage_substitute, target_idx=target_idx, damage=damage),
        partial(
            _damage_hit_target,
            target_idx=target_idx,
            damage=damage,
            attacker_idx=attacker_idx,
        ),
        state,
    )
    actual_loss = state.volatile_status.last_damage[target_idx]
    state = record_hit(state, hit_index, target_idx, HitOutcome.HIT,
        jnp.where(intercepted, HitRecipient.SUBSTITUTE, HitRecipient.POKEMON), actual_loss[0])
    result = state.hit_results[hit_index]
    # Subsequent callbacks consume the same recipient and capped damage that
    # the per-attempt record reports, including when the doll just broke.
    intercepted = result.recipient == HitRecipient.SUBSTITUTE
    actual_loss = jnp.reshape(result.damage, (1,))
    # A damaging Fire move thaws its target after the hit, even if it faints.
    target_status = state.active_field('status', target_idx)
    thaw = (move.type == Type.FIRE) & (actual_loss > 0) & ~intercepted & (target_status == Status.FREEZE)
    state = update_active_fields(state, 1-attacker_idx, status=jnp.where(thaw, Status.NONE, target_status))
    # Drain uses actual HP removed before on-hit healing, rounded half up.
    healing = jnp.maximum(1, jnp.floor(actual_loss * move.drain_fraction + .5)).astype(jnp.int32)
    state = jax.lax.cond(
        (move.drain_fraction[0] > 0) & (actual_loss[0] > 0),
        partial(_apply_drain, attacker_idx=attacker_idx, defender_ability=defender.ability, healing=healing),
        _identity,
        state,
    )

    key, state = jax.lax.cond(
        intercepted,
        _unchanged_key_state,
        partial(_run_hit_effects, attacker_idx=attacker_idx, move=move, is_crit=is_crit, hit_index=hit_index),
        key,
        state,
    )
    # Air Balloon breaks on a hit to the Pokémon, including zero-HP KOs, but
    # Substitute absorbs the hit without popping the held balloon.

    state = consume(state, target_idx, ~intercepted & (defender.item[0] == ItemEnum.AIR_BALLOON))
    state = jax.lax.cond(
        intercepted,
        _identity,
        partial(_run_held_hit_effects, attacker_idx=attacker_idx, move=move, damage=damage, raw_effectiveness=raw_effectiveness),
        state,
    )
    # Knock Off removes the item after a real hit, only if the attacker survives.
    state = jax.lax.cond(
        (ident == 282) & ~intercepted & (state.active_field('current_hp', attacker_idx)[0] > 0),
        partial(_knock_off_item, target_idx=target_idx, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    key, state = jax.lax.cond(
        intercepted,
        _unchanged_key_state,
        partial(_run_secondary_effects, attacker_idx=attacker_idx, move=move),
        key,
        state,
    )
    # These moves have explicit after-Substitute callbacks in Scarlet/Violet.
    after_sub = ((ident == 229) | (ident == 830) | (ident == 845)) & (state.active_field('current_hp', attacker_idx)[0] > 0)
    state = jax.lax.cond(
        ~intercepted | after_sub,
        partial(_run_after_damage, attacker_idx=attacker_idx, move=move),
        _identity,
        state,
    )
    return key, state, damage, defender


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)
