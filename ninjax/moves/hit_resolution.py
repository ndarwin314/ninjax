"""Hit resolution helpers for immutable JAX battle state."""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.calculations.damage import (
    compute_base_damage,
    compute_base_power,
    compute_damage_multipliers,
    damage_post_modifiers,
)
from ninjax.calculations.modifiers import CRIT_STAGES, conditional_mult_round
from ninjax.calculations.stats import raw_boosted_stats
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.field import Weather
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveType, OffensiveStatOwner
from ninjax.enums.pokemon import Status, Type
from ninjax.moves.effects import after_damage, after_every_hit, secondary_effects
from ninjax.moves.model import Move
from ninjax.moves.properties import substitute_intercepts
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.item_effects import after_hit, consume, remove_item
from ninjax.side.queries import (
    conditional_mult_round as fixed_modifier,
    effective_weather,
    modify_stats,
)
from ninjax.side.state import BattleState
from ninjax.side.updates import update_active_fields


# Generation 9 Unseen Fist deals full damage when bypassing Protect.
UNSEEN_FIST_PROTECT_MULTIPLIER = 1.0

def _damage_substitute(s, *, target_idx, damage, defender, ident):
    """Remove capped Substitute HP without triggering Pokémon damage hooks."""
    v = s.volatile_status
    old = v.substitute_hp[target_idx, 0]
    # Multiscale modifies ordinary damage calculation even against a doll;
    # fixed damage bypasses it just as it does for Pokémon HP.
    scaled = conditional_mult_round(damage, .5,
        (defender.ability == AbilityEnum.MULTISCALE) & (defender.current_hp == defender.max_hp) &
        ~((ident == 101) | (ident == 162) | (ident == 877) | (ident == 283)))
    loss = jnp.minimum(old, jnp.maximum(0, scaled[0]))
    remaining = old - loss
    s = s.replace(volatile_status=v.replace(
        substitute_hp=v.substitute_hp.at[target_idx, 0].set(remaining),
        last_damage=v.last_damage.at[target_idx, 0].set(loss)))
    # Public events expose presence, never the private remaining HP.
    s = emit(s, EventKind.CONDITION, target_idx, detail=13, before=1, after=remaining > 0)
    return s


def _damage_hit_target(s, *, target_idx, damage, ident, attacker_idx):
    """Apply direct hit damage, preserving fixed-damage and source attribution."""
    return take_damage_value(s, target_idx, damage, True,
                fixed_damage=(ident == 101) | (ident == 162) | (ident == 877) | (ident == 283), source=attacker_idx)


def _apply_drain(s, *, attacker_idx, defender, healing):
    """Heal the attacker, or deal indirect damage for Liquid Ooze."""
    return take_damage_value(s, attacker_idx, jnp.where(defender.ability == AbilityEnum.LIQUID_OOZE, healing, -healing), False)


def _unchanged_recoil_state(s, p, d):
    """Skip recoil while accepting the damage branch operands."""
    return s


def _run_hit_effects(k, s, *, attacker_idx, move, is_crit):
    """Apply per-hit move effects after damage reaches the Pokémon."""
    return after_every_hit(k, s, attacker_idx, move, is_crit)


def _run_held_hit_effects(s, *, attacker_idx, move, damage, defender):
    """Apply held-item reactions using this hit and its effectiveness."""
    return after_hit(s, attacker_idx, move, damage, calculate_effectiveness_multiplier(move.type, defender.type_list))


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

    Damage has shape (1,) and may exceed the target's remaining HP. The defender
    snapshot is taken before this hit. A multi-hit caller must track actual HP
    lost separately; comparing final HP is insufficient when an item heals
    during a hit. Keys explicitly advance for damage and secondary-effect rolls.

    Recoil remains here for single-hit moves to preserve existing event order.
    The multi-hit caller suppresses it; multi-hit recoil requires an explicit
    actual-damage total and is not yet supported.
    """
    active = state.effective_active
    attacker = active[attacker_idx]
    defender = active[1-attacker_idx]
    # Explosion's user has already fainted, but damage uses its pre-faint HP.
    pre_faint_hp = state.volatile_status.explosion_hp[attacker_idx]
    attacker = attacker.replace(current_hp=jnp.where((move.identifier == 153) & (pre_faint_hp > 0), pre_faint_hp, attacker.current_hp))


    # base power modifications, technician, tera, terrain etc
    power = compute_base_power(
        attacker,
        defender,
        move,
        state.terrain, effective_weather(state.weather.weather, state.ability, state.active.current_hp), state.gravity_duration > 0
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
    raw_stats = state.active.stats
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


    attacker_stats = modify_stats(state, attacker_idx, attacker_stats)
    defender_stats = modify_stats(state, 1-attacker_idx, defender_stats)
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
        state.volatile_status.flash_fire[attacker_idx]
    )

    # there is a specific order to the multipliers that i will preserve since rounding is done
    # between every multiplication by a modifier
    # at some point we can see if it makes any difference for speed to not do it this way
    damage = compute_damage_multipliers(
        key,
        attacker,
        defender,
        Weather(effective_weather(state.weather.weather, state.ability, state.active.current_hp), state.weather.duration),
        move,
        base_damage,
        is_crit,
        damage_roll,
        state.gravity_duration > 0
    )

    damage = damage_post_modifiers(damage, defender.ability, move)
    for name in ('reflect', 'light_screen', 'aurora_veil'):
        current = getattr(state, name)
        state = state.replace(**{name: current.at[1-attacker_idx].set(jnp.where(move.identifier[0] == 280, 0, current[1-attacker_idx]))})
    screen = jnp.where(move.move_type == MoveType.PHYSICAL, state.reflect[1-attacker_idx, 0] > 0,
                       state.light_screen[1-attacker_idx, 0] > 0) | (state.aurora_veil[1-attacker_idx, 0] > 0)
    damage = conditional_mult_round(damage, .5, screen & ~is_crit & (attacker.ability != AbilityEnum.INFILTRATOR))
    damage = conditional_mult_round(damage, 5324/4096, attacker.item == ItemEnum.LIFE_ORB)

    damage = fixed_modifier(damage, 4915/4096,
        (attacker.item == ItemEnum.EXPERT_BELT) & (calculate_effectiveness_multiplier(move.type, defender.type_list) > 1))
    bypass = state.volatile_status.protected[1-attacker_idx] & (attacker.ability == AbilityEnum.UNSEEN_FIST) & move.contact
    damage = conditional_mult_round(damage, UNSEEN_FIST_PROTECT_MULTIPLIER, bypass)

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
        partial(_damage_substitute, target_idx=target_idx, damage=damage, defender=defender, ident=ident),
        partial(
            _damage_hit_target,
            target_idx=target_idx,
            damage=damage,
            ident=ident,
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
    target = state.active[1-attacker_idx]
    thaw = (move.type == Type.FIRE) & (actual_loss > 0) & ~intercepted & (target.status == Status.FREEZE)
    state = update_active_fields(state, 1-attacker_idx, status=jnp.where(thaw, Status.NONE, target.status))
    # Drain and recoil use HP removed before a berry heals, with half-up
    # rounding, and never the uncapped calculated damage.
    healing = jnp.maximum(1, jnp.floor(actual_loss * move.drain_fraction + .5)).astype(jnp.int32)
    state = jax.lax.cond(
        (move.drain_fraction[0] > 0) & (actual_loss[0] > 0),
        partial(_apply_drain, attacker_idx=attacker_idx, defender=defender, healing=healing),
        _identity,
        state,
    )

    # lax.cond keeps both branches compatible with traced move and ability data.
    state = jax.lax.cond(
        jnp.logical_and(move.recoil, attacker.ability!=AbilityEnum.ROCK_HEAD)[0],
        do_recoil, _unchanged_recoil_state,
        state, attacker_idx, jnp.maximum(1, jnp.floor(actual_loss*move.recoil_percent + .5)).astype(jnp.int32))

    key, state = jax.lax.cond(
        intercepted,
        _unchanged_key_state,
        partial(_run_hit_effects, attacker_idx=attacker_idx, move=move, is_crit=is_crit),
        key,
        state,
    )
    # Air Balloon breaks on a hit to the Pokémon, including zero-HP KOs, but
    # Substitute absorbs the hit without popping the held balloon.

    state = consume(state, target_idx, ~intercepted & (defender.item[0] == ItemEnum.AIR_BALLOON))
    state = jax.lax.cond(
        intercepted,
        _identity,
        partial(_run_held_hit_effects, attacker_idx=attacker_idx, move=move, damage=damage, defender=defender),
        state,
    )
    # Knock Off removes the item after a real hit, only if the attacker survives.
    state = jax.lax.cond(
        (ident == 282) & ~intercepted & (state.active.current_hp[attacker_idx, 0] > 0),
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
    after_sub = ((ident == 229) | (ident == 830) | (ident == 845)) & (state.active.current_hp[attacker_idx, 0] > 0)
    state = jax.lax.cond(
        ~intercepted | after_sub,
        partial(_run_after_damage, attacker_idx=attacker_idx, move=move),
        _identity,
        state,
    )
    return key, state, damage, defender


def do_recoil(state: BattleState, attacker_idx: int, damage) -> BattleState:
    """Apply the supplied indirect recoil damage to the attacker."""
    original_cause = state.event_cause
    state = take_damage_value(state.replace(event_cause=jnp.int32(Cause.RECOIL)), attacker_idx, damage, False)
    state = state.replace(event_cause=original_cause)
    return state


def _identity(x):
    """Preserve the operand when a conditional effect is skipped."""
    return x


def _unchanged_key_state(k, s):
    """Skip an effect without advancing the key or changing state."""
    return (k, s)
