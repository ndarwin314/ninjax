"""Hit resolution helpers for immutable JAX battle state."""
from functools import partial

from ninjax.ability_effects import ability_contact_reaction

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
from ninjax.enums.moves import HitRule, MoveFlags, MoveType, OffensiveStatOwner, MoveResult, PowerRule
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.item_rules import item_is_suppressed
from ninjax.moves.effects import after_damage, after_every_hit, secondary_effects
from ninjax.moves.model import Move
from ninjax.moves.properties import substitute_intercepts
from ninjax.records.events import Cause, EventKind, emit
from ninjax.records.hits import HitOutcome, HitRecipient, record_hit
from ninjax.side.residual_conditions import root_healing
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.faint_effects import direct_faint
from ninjax.side.trapping import apply_binding
from ninjax.side.item_effects import (
    after_hit, after_hp_change, apply_flung_item, consume, destroy_opposing_berry,
    eat_opposing_berry, held_item_active, release_fling_item, remove_item, steal_item,
)
from ninjax.side.queries import (
    ability_against_move,
    effective_weather,
    effective_speed,
    stat_modifier,
)
from ninjax.side.state import BattleState
from ninjax.side.form_effects import break_form_shield, gulp_missile_hit, shield_form
from ninjax.side.status_effects import set_status
from ninjax.side.queries import contact_reactions_allowed
from ninjax.side.updates import update_active_fields


def _damage_substitute(s, *, target_idx, damage):
    """Remove capped Substitute HP without triggering Pokémon damage hooks."""
    v = s.volatile_status
    old = v.substitute_hp[target_idx, 0]
    loss = jnp.minimum(old, jnp.maximum(0, damage[0]))
    remaining = old - loss
    s = s.replace(volatile_status=v.replace(
        substitute_hp=v.substitute_hp.at[target_idx, 0].set(remaining),
        move_history=v.move_history.replace(last_damage=v.move_history.last_damage.at[target_idx, 0].set(loss))))
    # Public events expose presence, never the private remaining HP.
    s = emit(s, EventKind.CONDITION, target_idx, detail=13, before=1, after=remaining > 0)
    return s


def _damage_hit_target(s, *, target_idx, damage, attacker_idx, defender_ability, activate_items):
    """Commit calculated hit damage with its attacker's source attribution."""
    return take_damage_value(s, target_idx, damage, True, source=attacker_idx,
                             attack_defender_ability=defender_ability, activate_items=activate_items)


def _final_gambit_faint(state, *, attacker_idx):
    """Faint the user after snapshotting the damage for Final Gambit."""
    return direct_faint(state, attacker_idx, attacker_idx, Cause.SELF_KO)


def _beak_blast_burn(key, state, *, attacker_idx, defender_idx):
    """Attribute the pre-move contact burn to the preparing defender."""
    previous_source = state.event_context.source
    state = state.replace(event_context=state.event_context.replace(source=jnp.asarray(defender_idx, jnp.int32)))
    key, state = set_status(key, state, attacker_idx, Status.BURN)
    return key, state.replace(event_context=state.event_context.replace(source=previous_source))


def _apply_drain(s, *, attacker_idx, defender_ability, healing):
    """Heal the attacker, or deal indirect damage for Liquid Ooze."""
    ooze_damage = jnp.where(s.ability[attacker_idx, 0] == AbilityEnum.MAGIC_GUARD, 0, healing)
    return take_damage_value(s, attacker_idx,
        jnp.where(defender_ability == AbilityEnum.LIQUID_OOZE, ooze_damage, -healing), False)


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


def _early_item_effects(state, *, player, target, move):
    """Eat, destroy or deliver an item as a primary hit effect."""
    ident = move.identifier[0]
    branch = jnp.select(((ident == 450) | (ident == 365), ident == 510,
                        move.flung_item[0] != ItemEnum.NONE), (1, 2, 3), default=0)
    return jax.lax.switch(branch, (
        _identity,
        partial(eat_opposing_berry, player=player, target=target),
        partial(destroy_opposing_berry, player=player, target=target),
        partial(apply_flung_item, target=target, item=move.flung_item[0])), state)


def _run_secondary_effects(k, s, *, attacker_idx, move, reached_target):
    """Roll target effects on a direct hit, and user effects on either recipient."""
    return secondary_effects(k, s, attacker_idx, move, reached_target=reached_target)


def _run_after_damage(s, *, attacker_idx, move):
    """Apply move callbacks that run after damage or Substitute interception."""
    return after_damage(s, attacker_idx, move)


def resolve_damaging_hit(key, state: BattleState, attacker_idx, move: Move, hit_index=0):
    """Apply one hit; return key, state, calculated damage and defender snapshot.

    Calculated damage has shape (1,) and can exceed the recipient's HP.
    The hit record stores actual loss before healing. Move finalization uses
    that loss for recoil. The defender snapshot is from before the hit.
    """
    active = state.combat_active
    attacker = active[attacker_idx]
    defender = active[1-attacker_idx]
    attacker = attacker.replace(item=state.effective_item(attacker_idx))
    defender = defender.replace(item=state.effective_item(1-attacker_idx))
    # Damage calculation reads the same weather before any HP changes.
    weather = effective_weather(state.weather.weather, active.ability, active.current_hp)
    # Resolve a hit-local ability without affecting field weather or stored state.
    hit_defender = defender.replace(ability=ability_against_move(state, 1-attacker_idx, move))
    # Explosion's user has already fainted, but damage uses its pre-faint HP.
    pre_faint_hp = state.volatile_status.explosion_hp[attacker_idx]
    attacker = attacker.replace(current_hp=jnp.where(move.detonates & (pre_faint_hp > 0), pre_faint_hp, attacker.current_hp))


    gravity = state.gravity_duration > 0
    raw_effectiveness, damage_effectiveness = hit_type_effectiveness(
        move.type, defender.type_list, hit_defender.ability, state.grounding_override(1-attacker_idx), move.effectiveness_rule
    )
    fixed_damage = move.source_hp_damage | move.ohko | (move.identifier == 68) | (move.identifier == 243)
    # Tera Shell applies to every hit of a multihit attack that began at full HP.
    # Final Gambit and OHKO moves bypass this effectiveness check.
    activation = ~fixed_damage[0] & (hit_defender.ability[0] == AbilityEnum.TERA_SHELL) & (move.identifier[0] > 0) & (
        ((defender.current_hp[0] == defender.max_hp) & (damage_effectiveness >= 1)).reshape(()) |
        ((hit_index > 0) & state.volatile_status.ability_state.tera_shell[1-attacker_idx, 0]))
    v = state.volatile_status
    a = v.ability_state
    state = state.replace(volatile_status=v.replace(ability_state=a.replace(
        tera_shell=a.tera_shell.at[1-attacker_idx, 0].set(activation))))
    state = emit(state, EventKind.ABILITY, 1-attacker_idx, detail=AbilityEnum.TERA_SHELL,
                 enabled=activation & (hit_index == 0))
    damage_effectiveness = jnp.where(activation, .5, damage_effectiveness)
    raw_effectiveness = jnp.where(activation, .5, raw_effectiveness)
    # Base power and held-item reactions receive the raw type matchup.
    user_speed = effective_speed(state, attacker_idx, weather)
    target_speed = effective_speed(state, 1-attacker_idx, weather)
    bank = state.active_moves(attacker_idx)
    matching_slot = jnp.argmax(bank.identifier[:, 0] == move.identifier[0])
    native_priority = bank.priority[matching_slot, 0]
    # Callback moves whose listed power is zero or 150 and multihit moves
    # retain their computed power rather than receiving the Tera minimum.
    callback = ((move.power_rule[0] != PowerRule.FIXED) |
                (move.identifier[0] == 255) | (move.identifier[0] == 374))
    native_power = bank.base_power[matching_slot, 0]
    excluded_callback = callback & ((native_power == 0) | (native_power == 150))
    tera_floor = (attacker.is_terastallized[0] & (move.type[0] == attacker.tera_type[0]) &
                  (move.type[0] != Type.NONE) & (native_priority <= 0) &
                  (move.hit_rule[0] == HitRule.SINGLE) & (move.identifier[0] != 251) &
                  ~excluded_callback & ~fixed_damage[0])
    power = compute_base_power(
        attacker,
        hit_defender,
        move,
        state.terrain, weather, gravity, raw_effectiveness=raw_effectiveness,
        user_speed=user_speed,
        target_speed=target_speed,
        positive_stages=jnp.sum(jnp.maximum(state.boosts.normal_boosts[attacker_idx, 1:], 0)) +
                        jnp.sum(jnp.maximum(state.boosts.acc_boosts[attacker_idx], 0)),
        previous_failed=state.volatile_status.move_history.previous_move_result[attacker_idx] == MoveResult.FAILED,
        target_charging_move=state.volatile_status.continuation.charging_move_id[1-attacker_idx, 0],
        rooted_user=state.volatile_status.residuals.ingrain[attacker_idx, 0] != 0,
        rooted_target=state.volatile_status.residuals.ingrain[1-attacker_idx, 0] != 0,
        times_attacked=state.damage_history.times_attacked[attacker_idx, state.active_index[attacker_idx, 0]],
        total_fainted=state.damage_history.total_fainted[attacker_idx],
        target_hurt=state.damage_history.hurt_this_turn[1-attacker_idx, state.active_index[1-attacker_idx, 0]],
        user_stats_lowered=state.damage_history.stats_lowered_this_turn[attacker_idx, state.active_index[attacker_idx, 0]],
        last_successful_move=state.damage_history.last_successful_move_this_turn,
        stockpile_count=state.volatile_status.stockpile_count[attacker_idx, 0],
        tera_floor=tera_floor,
        item_suppressed=item_is_suppressed(state.ability[attacker_idx], state.magic_room_duration,
                                            state.volatile_status.embargo_turns[attacker_idx]),
    )
    key, one, two = random.split(key, num=3)
    # crit multiplier
    # battle armor prevents crits
    crit_stage = (move.crit_stage+
                  (attacker.ability==AbilityEnum.SUPER_LUCK)[0] +
                  2 * state.volatile_status.focus_energy[attacker_idx, 0] +
                  3*jnp.logical_and(attacker.ability==AbilityEnum.MERCILESS, defender.is_poisoned))
    crit_chance = CRIT_STAGES[jnp.clip(crit_stage, 0, len(CRIT_STAGES) - 1)] * (hit_defender.ability != AbilityEnum.BATTLE_ARMOR)
    is_crit = (((random.uniform(one) < crit_chance) | move.always_crit) &
               (hit_defender.ability != AbilityEnum.BATTLE_ARMOR) & ~fixed_damage)
    state = emit(state, EventKind.CRITICAL, 1-attacker_idx, enabled=is_crit)
    damage_roll = random.randint(two, (), minval=85, maxval=101) / 100

    # Select stage-adjusted stats before computing offensive and defensive
    # modifiers. Moves such as Body Press can select a nonstandard attack stat.
    raw_stats = state.active_stats()
    # Wonder Room exchanges the base defenses before applying each stat's
    # original stage. Body Press also switches its selected offensive stat.
    wonder = state.wonder_room_duration > 0
    defense, special_defense = raw_stats[:, StatEnum.DEFENSE], raw_stats[:, StatEnum.SPECIAL_DEFENSE]
    raw_stats = raw_stats.at[:, StatEnum.DEFENSE].set(jnp.where(wonder, special_defense, defense))
    raw_stats = raw_stats.at[:, StatEnum.SPECIAL_DEFENSE].set(jnp.where(wonder, defense, special_defense))
    offensive_stat = jnp.where(wonder & (move.identifier == 776), StatEnum.SPECIAL_DEFENSE, move.offensive_stat)
    move = move.replace(offensive_stat=offensive_stat)
    boosts = state.boosts
    attacker_stats, defender_stats = raw_boosted_stats(
        raw_stats,
        boosts,
        attacker_idx,
        (attacker.ability==AbilityEnum.UNAWARE) | move.ignore_defensive,
        hit_defender.ability==AbilityEnum.UNAWARE,
        is_crit,
        offensive_owner=jnp.where(move.identifier[0] == 492, OffensiveStatOwner.TARGET, OffensiveStatOwner.USER)
    )


    offensive_modifier = stat_modifier(state, attacker_idx, move.offensive_stat, weather)
    defensive_modifier = stat_modifier(state, 1 - attacker_idx, move.defensive_stat, weather,
                                       ability=hit_defender.ability[0])
    base_damage = compute_base_damage(
        attacker.ability,
        hit_defender.ability,
        attacker.hp_percent,
        attacker.level,
        attacker_stats,
        defender_stats,
        move,
        power,
        attacker.status,
        defender.status,
        state.volatile_status.ability_state.flash_fire[attacker_idx],
        offensive_modifier=offensive_modifier, defensive_modifier=defensive_modifier,
    )

    screen = jnp.where(move.move_type == MoveType.PHYSICAL, state.side_conditions.reflect[1-attacker_idx, 0] > 0,
                       state.side_conditions.light_screen[1-attacker_idx, 0] > 0) | (state.side_conditions.aurora_veil[1-attacker_idx, 0] > 0)
    charging = state.volatile_status.continuation.charging_move_id[1-attacker_idx, 0]
    ident = move.identifier[0]
    early_item = (ident == 450) | (ident == 365) | (ident == 510) | (move.flung_item[0] != ItemEnum.NONE)
    theft = (ident == 168) | (ident == 343)
    defer_hp_items = early_item | theft | (ident == 282) | (ident == 917)
    # Exceptional moves strike these charge states at double damage; Bounce
    # uses a separate BasePower callback instead of Fly's damage callback.
    semi_bonus = ((charging == 19) & ((ident == 16) | (ident == 239))) | (
        (charging == 91) & ((ident == 89) | (ident == 222))) | (
        (charging == 291) & ((ident == 57) | (ident == 250)))
    damage = compute_damage_multipliers(attacker, hit_defender, weather, move,
        base_damage, is_crit, damage_roll, gravity,
        effectiveness=damage_effectiveness, screen=screen,
        user_speed=user_speed, target_speed=target_speed, semi_damage_boost=semi_bonus,
        stab_types=state.effective_types(attacker_idx, include_added=True, ignore_tera=True))
    # Gen 9 Unseen Fist bypasses Protect without reducing damage.

    # Fixed damage ignores damage modifiers (including screens), but reaches
    # this branch only after the normal type/ability/protection checks.
    ident = move.identifier[0]
    damage = jnp.where((ident == 69) | (ident == 101), attacker.level, damage)
    damage = jnp.where((ident == 162) | (ident == 877), jnp.maximum(1, defender.current_hp // 2), damage)
    damage = jnp.where(ident == 283, jnp.maximum(0, defender.current_hp-attacker.current_hp), damage)
    # Counter and Mirror Coat return twice the recorded incoming hit.
    retaliation = state.damage_history.counter_damage[attacker_idx, state.active_index[attacker_idx, 0]]
    damage = jnp.where((ident == 68) | (ident == 243), jnp.maximum(1, retaliation), damage)
    # Fixed damage bypasses ordinary damage modifiers. The Final Gambit
    # callback faints its user only after hit gates allow damage calculation.
    damage = jnp.where(move.source_hp_damage, attacker.current_hp, damage)
    damage = jnp.where(move.ohko, defender.max_hp, damage)
    state = jax.lax.cond(move.source_hp_damage[0],
        partial(_final_gambit_faint, attacker_idx=attacker_idx), _identity, state)

    # Substitute intercepts this hit before Pokémon HP and on-hit callbacks.
    target_idx = 1-attacker_idx
    intercepted = substitute_intercepts(state.volatile_status.substitute_hp[1-(attacker_idx), 0], state.ability[attacker_idx, 0],
        move.target_self[0], move.sound[0], move.bypass_substitute[0])
    # False Swipe leaves the actual Pokémon at one HP, while a Substitute
    # receives the full hit and may be broken.
    damage = jnp.where(move.nonlethal & ~intercepted,
                       jnp.minimum(damage, jnp.maximum(0, defender.current_hp - 1)), damage)
    disguise, ice_face = shield_form(state, target_idx, hit_defender.ability[0], move.move_type[0], ~intercepted)
    damage = jnp.where(disguise | ice_face, 0, damage)
    state = jax.lax.cond(
        intercepted,
        partial(_damage_substitute, target_idx=target_idx, damage=damage),
        partial(
            _damage_hit_target,
            target_idx=target_idx,
            damage=damage,
            attacker_idx=attacker_idx,
            defender_ability=hit_defender.ability,
            activate_items=~defer_hp_items,
        ),
        state,
    )
    actual_loss = state.volatile_status.move_history.last_damage[target_idx]
    # Preparation reactions are OnHit effects and do not cross a Substitute.
    queued = state.volatile_status.queued_move
    focused = ~intercepted & (queued.preparation[target_idx, 0] == 264)
    queued = queued.replace(focus_lost=queued.focus_lost.at[target_idx, 0].set(
        queued.focus_lost[target_idx, 0] | focused))
    state = state.replace(volatile_status=state.volatile_status.replace(queued_move=queued))
    held = jnp.where(item_is_suppressed(state.ability[attacker_idx, 0], state.magic_room_duration,
        state.volatile_status.embargo_turns[attacker_idx, 0]), ItemEnum.NONE, state.active_field('item', attacker_idx)[0])
    beak_contact = (~intercepted & (queued.preparation[target_idx, 0] == 690) & move.contact[0] &
                    contact_reactions_allowed(state.ability[attacker_idx, 0], held))
    key, state = jax.lax.cond(beak_contact,
        partial(_beak_blast_burn, attacker_idx=attacker_idx, defender_idx=target_idx),
        _unchanged_key_state, key, state)
    # Rage Fist counts each hit, including hits absorbed by a Substitute.
    history = state.damage_history
    slot = state.active_index[target_idx, 0]
    records_counter = ~intercepted & (history.counter_prepared[target_idx] == move.move_type[0])
    history = history.replace(
        times_attacked=history.times_attacked.at[target_idx, slot].add(1),
        counter_damage=history.counter_damage.at[target_idx, slot].set(
            jnp.where(records_counter, actual_loss[0] * 2, history.counter_damage[target_idx, slot])),
        counter_category=history.counter_category.at[target_idx, slot].set(
            jnp.where(records_counter, move.move_type[0], history.counter_category[target_idx, slot])),
        counter_source_slot=history.counter_source_slot.at[target_idx, slot].set(
            jnp.where(records_counter, state.active_index[attacker_idx, 0], history.counter_source_slot[target_idx, slot])))
    state = state.replace(damage_history=history)
    state = record_hit(state, hit_index, target_idx, HitOutcome.HIT,
        jnp.where(intercepted, HitRecipient.SUBSTITUTE, HitRecipient.POKEMON), actual_loss[0])
    state = break_form_shield(state, target_idx, disguise, ice_face)
    result = state.hit_results[hit_index]
    # Subsequent callbacks consume the same recipient and capped damage that
    # the per-attempt record reports, including when the doll just broke.
    intercepted = result.recipient == HitRecipient.SUBSTITUTE
    actual_loss = jnp.reshape(result.damage, (1,))
    state = release_fling_item(state, attacker_idx, move.flung_item[0])
    state = jax.lax.cond(early_item & ~intercepted,
        partial(_early_item_effects, player=attacker_idx, target=target_idx, move=move), _identity, state)
    # A damaging Fire move thaws its target after the hit, even if it faints.
    target_status = state.active_field('status', target_idx)
    thaw = (move.type == Type.FIRE) & (actual_loss > 0) & ~intercepted & (target_status == Status.FREEZE)
    state = update_active_fields(state, 1-attacker_idx, status=jnp.where(thaw, Status.NONE, target_status))
    # Drain uses actual HP removed before on-hit healing, rounded half up.
    healing = jnp.maximum(1, jnp.floor(actual_loss * move.drain_fraction + .5)).astype(jnp.int32)
    healing = root_healing(state, attacker_idx, healing)
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
    # Damage to the Pokémon or its Substitute pops the balloon permanently.
    state = consume(state, target_idx, (actual_loss[0] > 0) & held_item_active(state, target_idx) &
                    (state.active_field('item', target_idx)[0] == ItemEnum.AIR_BALLOON), recoverable=False)
    state = jax.lax.cond(
        intercepted,
        _identity,
        # Fixed damage does not supply a type modifier to Weakness Policy.
        partial(_run_held_hit_effects, attacker_idx=attacker_idx, move=move, damage=damage,
                raw_effectiveness=jnp.where(fixed_damage, 1, raw_effectiveness)),
        state,
    )
    key, state = gulp_missile_hit(key, state, target_idx, ~intercepted & (actual_loss[0] > 0))
    # The one-HP floor can leave zero damage while still making contact.
    zero_damage_contact = move.nonlethal[0] & (defender.current_hp[0] == 1) & ~disguise & ~ice_face
    state = ability_contact_reaction(state, attacker_idx, hit_defender.ability[0],
                                     ~intercepted & move.contact[0] & ((actual_loss[0] > 0) | zero_damage_contact))
    # Knock Off removes the item after a real hit, only if the attacker survives.
    state = jax.lax.cond(
        (ident == 282) & ~intercepted & (state.active_field('current_hp', attacker_idx)[0] > 0),
        partial(_knock_off_item, target_idx=target_idx, attacker_idx=attacker_idx),
        _identity,
        state,
    )
    state = jax.lax.cond(theft & ~intercepted,
        partial(steal_item, recipient=attacker_idx, holder=target_idx), _identity, state)
    key, state = jax.lax.cond(
        ~intercepted | jnp.any(move.secondaries.user_stages != 0),
        partial(_run_secondary_effects, attacker_idx=attacker_idx, move=move, reached_target=~intercepted),
        _unchanged_key_state,
        key,
        state,
    )
    # Item mutation and Psychic Noise's Heal Block precede the berry HP check.
    state = jax.lax.cond(defer_hp_items & ~intercepted,
        partial(after_hp_change, player=target_idx), _identity, state)
    # Explicit target thawing runs after secondaries, so Scald cannot burn a
    # frozen target on the same hit that thaws it. Substitute blocks the thaw.
    target_status = state.active_field('status', target_idx)
    thaw = (move.thaws_target & ~intercepted & (actual_loss > 0) &
            (state.active_field('current_hp', target_idx) > 0) & (target_status == Status.FREEZE))
    state = update_active_fields(state, target_idx, status=jnp.where(thaw, Status.NONE, target_status))
    key, state = jax.lax.cond(
        ~intercepted & ((move.move_flags[0] & MoveFlags.BINDING) != 0),
        partial(apply_binding, source=attacker_idx, move_id=ident),
        _unchanged_key_state, key, state)
    # These moves have explicit after-Substitute callbacks in Scarlet/Violet.
    after_sub = (ident == 798) | (((ident == 229) | (ident == 866) | (ident == 830) | (ident == 845)) &
        (state.active_field('current_hp', attacker_idx)[0] > 0))
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
