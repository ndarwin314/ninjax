"""Effects helpers for immutable JAX battle state."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as random

from ninjax.calculations.logic import in_range
from ninjax.calculations.modifiers import ROUGH_SKIN_DAMAGE
from ninjax.enums.abilities import AbilityEnum, AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum, TerrainEnum as T
from ninjax.enums.items import ItemEnum
from ninjax.enums.moves import MoveType, SecondaryEffectType
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.knowledge import remember_item
from ninjax.moves.model import Move, SECONDARY_SLOTS
from ninjax.records.events import Cause, EventKind, emit
from ninjax.restrictions import apply_hit_restriction
from ninjax.side.field_conditions import set_terrain
from ninjax.side.hp_effects import apply_indirect_hp_change, take_damage_percent
from ninjax.side.item_effects import kings_rock_flinch
from ninjax.side.queries import ability_against_move, contact_reactions_allowed, is_powder_immune_fields
from ninjax.side.stat_effects import conditional_reduce_boosts, stat_changes
from ninjax.side.state import BattleState
from ninjax.side.status_effects import apply_confusion, set_status
from ninjax.side.residual_conditions import apply_residual_secondary, clear_leech_seed
from ninjax.side.trapping import apply_linked_trap, clear_binding
from ninjax.side.updates import add_boosts, clear_boosts, conditional_add_boosts


def _after_every_hit_update_field(k, s, a):
    """After every hit update field."""
    return (k, s)


def _after_damage_clear_boosts(s, *, target):
    """After damage clear boosts."""
    return clear_boosts(s, target)


def _preserve_operand(s):
    """Preserve operand."""
    return s


def _after_damage_update_field(s):
    """After damage update field."""
    return s.replace(terrain=Terrain(T.NONE, 0))


def _after_damage_add_boosts(s, *, player):
    """After damage add boosts."""
    return add_boosts(s, player, 5, 1)


def _clear_hazard_row(values, *, player):
    """Clear one side's layers or presence flag for Rapid Spin."""
    return values.at[player].set(jnp.zeros_like(values[player]))


def _after_damage_spin(s, *, player, rapid):
    """Remove entry hazards and bindings, adding Speed for Rapid Spin."""
    s = s.replace(hazards=jax.tree.map(partial(_clear_hazard_row, player=player), s.hazards))
    s = clear_leech_seed(clear_binding(s, player), player)
    return jax.lax.cond(rapid & (s.active_field('current_hp', player)[0] > 0),
        partial(_after_damage_add_boosts, player=player), _preserve_operand, s)


def _innards_out_damage(state, *, attacker_idx, defender_idx, damage):
    """Attribute Innards Out damage to the fainted holder."""
    previous_source, previous_cause = state.event_context.source, state.event_context.cause
    state = state.replace(
        event_context=state.event_context.replace(
            source=jnp.asarray(defender_idx, jnp.int32),
            cause=jnp.int32(Cause.INDIRECT)))
    state = apply_indirect_hp_change(state, attacker_idx, damage, source=defender_idx)
    return state.replace(
        event_context=state.event_context.replace(
            source=previous_source,
            cause=previous_cause))


def _skip_innards_out(state):
    """Keep state when the hit does not trigger Innards Out."""
    return state


def after_every_hit(
        key: chex.PRNGKey,
        state: BattleState,
        attacker_idx,
        move: Move,
        is_crit: bool,
        hit_damage=0,
        move_damage=0,
):
    """Apply the represented ability and contact effects after one hit."""
    defender_idx = 1 - attacker_idx
    # Keep the pre-contact ability and item snapshots for Aftermath.
    abilities = state.ability
    defender_ability = ability_against_move(state, defender_idx, move)
    attacker_ability = abilities[attacker_idx]
    defender_alive = state.active_field('is_alive', defender_idx)
    attacker_item = state.active_field('item', attacker_idx)[0]
    #stamina
    is_stamina = (defender_ability == AbilityEnum.STAMINA) & defender_alive
    state = conditional_add_boosts(state, defender_idx, is_stamina, StatEnum.DEFENSE, 1)
    # cotton down
    is_cotton_down = (defender_ability == AbilityEnum.COTTON_DOWN)
    state = conditional_reduce_boosts(state, 1 - defender_idx, is_cotton_down, StatEnum.SPEED, 1, True)
    # seed sower
    state = set_terrain(state, TerrainEnum.GRASSY * (defender_ability==AbilityEnum.SEED_SOWER), 5)
    # weak armor
    state = conditional_add_boosts(
        state,
        defender_idx,
        (defender_ability==AbilityEnum.WEAK_ARMOR) & defender_alive & (move.move_type==MoveType.PHYSICAL),
        (StatEnum.DEFENSE, StatEnum.SPEED), (-1, 2)
    )
    # anger point
    is_angry = jnp.logical_and(defender_ability==AbilityEnum.ANGER_POINT, is_crit)[0]
    state = conditional_add_boosts(state, defender_idx, is_angry, StatEnum.ATTACK, 13)

    # thermal exchange
    state = conditional_add_boosts(
        state,
        defender_idx,
        (defender_ability==AbilityEnum.THERMAL_EXCHANGE) & defender_alive & (move.type==Type.FIRE),
        StatEnum.ATTACK, 1
    )
    # Simple damaging-hit reactions only boost a surviving holder.
    alive = state.active_field('current_hp', defender_idx)[0] > 0
    state = conditional_add_boosts(state, defender_idx, alive & (defender_ability[0] == AbilityEnum.WATER_COMPACTION) &
        (move.type[0] == Type.WATER), StatEnum.DEFENSE, 2)
    state = conditional_add_boosts(state, defender_idx, alive & (defender_ability[0] == AbilityEnum.JUSTIFIED) &
        (move.type[0] == Type.DARK), StatEnum.ATTACK, 1)
    rattled = (move.type[0] == Type.DARK) | (move.type[0] == Type.BUG) | (move.type[0] == Type.GHOST)
    state = conditional_add_boosts(state, defender_idx, alive & (defender_ability[0] == AbilityEnum.RATTLED) & rattled, StatEnum.SPEED, 1)
    # do on contact effects
    makes_contact = jnp.logical_and(move.contact,  attacker_ability!=AbilityEnum.LONG_REACH)
    key, state = jax.lax.cond(makes_contact[0],
        partial(do_contact, target_ability=defender_ability), _after_every_hit_update_field,
        key, state, attacker_idx)
    # Aftermath is contact retaliation after the holder faints. Damp prevents
    # it, and Pads/Long Reach/Magic Guard protect the surviving attacker.

    damp = ((attacker_ability[0] == AbilityEnum.DAMP) & (state.active_field('current_hp', attacker_idx)[0] > 0) |
            (defender_ability[0] == AbilityEnum.DAMP) & (state.active_field('current_hp', defender_idx)[0] > 0))
    aftermath = ((defender_ability[0] == AbilityEnum.AFTERMATH) & ~defender_alive[0] & (hit_damage > 0) &
        move.contact[0] & contact_reactions_allowed(attacker_ability[0], attacker_item) & ~damp &
        (attacker_ability[0] != AbilityEnum.MAGIC_GUARD) & (state.active_field('current_hp', attacker_idx)[0] > 0))
    source = state.event_context.source
    state = state.replace(event_context=state.event_context.replace(source=jnp.asarray(defender_idx, jnp.int32)))
    state = take_damage_percent(state, attacker_idx, jnp.where(aftermath, 1/4, 0), cause=Cause.CONTACT)
    state = state.replace(event_context=state.event_context.replace(source=source))
    innards = ((defender_ability[0] == AbilityEnum.INNARDS_OUT) & ~defender_alive[0] &
               (hit_damage > 0) & (attacker_ability[0] != AbilityEnum.MAGIC_GUARD) &
               (state.active_field('current_hp', attacker_idx)[0] > 0))
    state = jax.lax.cond(innards,
        partial(_innards_out_damage, attacker_idx=attacker_idx, defender_idx=defender_idx, damage=move_damage),
        _skip_innards_out, state)
    return key, state


def effect_spore_status(r):
    """Return the status selected by an Effect Spore probability roll."""
    return (jnp.less(r, 0.09) * Status.POISON +
            in_range(0.09, r, 0.19) * Status.PARALYZE +
            in_range(0.19, r, 0.3) * Status.SLEEP)


def do_contact(key: chex.PRNGKey, state: BattleState, attacker_idx,
               *, target_ability=None) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply contact effects, preserving Pads' protection and public reveals."""
    abilities = state.ability
    defender_ability = abilities[1-attacker_idx]
    attacker_ability = abilities[attacker_idx]
    defender_item = state.active_field('item', 1-attacker_idx)
    attacker_item = state.active_field('item', attacker_idx)
    # Snapshot powder immunity before contact damage and status reactions.
    powder_immune = is_powder_immune_fields(state.effective_types(attacker_idx, include_added=True), attacker_ability)
    reacts = contact_reactions_allowed(attacker_ability[0], attacker_item[0])
    key, def_key, attack_key = random.split(key, 3)

    # defender triggers


    # Showdown announces Pads against deterministic Rough Skin/Gooey checks,
    # but not against Helmet or a prevented random status ability.
    reveal = ((attacker_item[0] == ItemEnum.PROTECTIVE_PADS) &
        (attacker_ability[0] != AbilityEnum.LONG_REACH) &
        ((defender_ability[0] == AbilityEnum.ROUGH_SKIN) | (defender_ability[0] == AbilityEnum.GOOEY)))
    state = remember_item(state, attacker_idx, ItemEnum.PROTECTIVE_PADS, reveal)
    owner = 1-attacker_idx
    slot = state.active_index[owner, 0]
    known = state.known_abilities[owner, slot]
    state = state.replace(known_abilities=state.known_abilities.at[owner, slot].set(
        jnp.where(reveal, defender_ability[0], known)))
    state = emit(state, EventKind.ABILITY, owner, detail=defender_ability[0], enabled=reveal)
    state = emit(state, EventKind.ITEM_REVEALED, attacker_idx, detail=ItemEnum.PROTECTIVE_PADS, enabled=reveal)
    helmet = (defender_item == ItemEnum.ROCKY_HELMET)[0] & reacts
    state = take_damage_percent(state, attacker_idx, helmet / 6, cause=Cause.CONTACT)
    is_rough_skin = (defender_ability == AbilityEnum.ROUGH_SKIN)[0] & reacts
    state = take_damage_percent(state, attacker_idx, ROUGH_SKIN_DAMAGE * is_rough_skin, cause=Cause.CONTACT)
    is_static = defender_ability==AbilityEnum.STATIC
    is_flame = defender_ability==AbilityEnum.FLAME_BODY
    is_point = defender_ability==AbilityEnum.POISON_POINT
    is_effect_spore = jnp.logical_and(defender_ability==AbilityEnum.EFFECT_SPORE, 1-powder_immune)
    r = random.uniform(def_key)
    triggered = jnp.less_equal(r, 0.3) & reacts
    status = (effect_spore_status(r) * is_effect_spore +
              is_flame * Status.BURN +
              is_static * Status.PARALYZE +
              is_point * Status.POISON) * triggered
    key, state = set_status(key, state, attacker_idx, status)

    # attacker triggers
    is_poison_touch = attacker_ability==AbilityEnum.POISON_TOUCH
    is_toxic_chain = attacker_ability==AbilityEnum.TOXIC_CHAIN
    r = random.uniform(attack_key)
    triggered = jnp.less_equal(r, 0.3)
    status = (is_poison_touch * (defender_ability != AbilityEnum.SHIELD_DUST) * Status.POISON +
              is_toxic_chain * Status.TOXIC) * triggered

    # gooey, idk why i take 0 index here but it happens in some places and i think was necessary
    is_gooey = (defender_ability==AbilityEnum.GOOEY)[0] & reacts
    state = conditional_reduce_boosts(state, attacker_idx, is_gooey, StatEnum.SPEED, 1, True)

    key, state = set_status(key, state, 1 - attacker_idx, status, target_ability=target_ability)
    return key, state


def _unchanged_secondary(key, state, *unused):
    """Keep both explicit key and state when a secondary is skipped."""
    return key, state


def _secondary_status(key, state, target, status):
    """Apply the move's represented major-status secondary."""
    return set_status(key, state, target, jnp.reshape(status, (1,)))


def _secondary_confusion(key, state, target):
    """Apply the move's represented confusion secondary."""

    return apply_confusion(key, state, target)


def _apply_secondary(key, state, *, target, move, ability, index):
    """Apply one successful slot in status, confusion, stat and flinch order."""
    effects = move.secondaries
    status = effects.status[index]
    key, state = jax.lax.cond(status != Status.NONE,
        _secondary_status, _unchanged_secondary, key, state, target, status)
    key, state = jax.lax.cond(effects.confusion[index],
        _secondary_confusion, _unchanged_secondary, key, state, target)
    state = stat_changes(state, target, effects.stats[index], effects.stages[index], True)
    state = jax.lax.cond(effects.restriction[index],
        _apply_secondary_condition, _skip_hit_restriction, state, target, move, ability)
    v = state.volatile_status
    flinch = effects.flinch[index] & (ability != A.INNER_FOCUS)
    return key, state.replace(volatile_status=v.replace(flinched=v.flinched.at[target, 0].set(v.flinched[target, 0] | flinch)))


def _apply_secondary_condition(state, target, move, ability):
    """Apply a restriction or an ongoing condition from a landed secondary."""
    ident = move.identifier[0]
    restriction = (ident == 675) | (ident == 917)
    return jax.lax.cond(restriction, apply_hit_restriction, _apply_ongoing_secondary,
                        state, target, move, ability)


def _apply_ongoing_secondary(state, target, move, ability):
    """Apply a non-restriction secondary to the active target."""
    return apply_residual_secondary(state, target, move)


def _jaw_lock(state, player):
    """Link both participants when Jaw Lock lands on the real target."""
    state, _ = apply_linked_trap(state, player, 1-player)
    state, _ = apply_linked_trap(state, 1-player, player)
    return state


def _skip_jaw_lock(state, player):
    """Preserve existing traps on other hits."""
    return state


def _skip_hit_restriction(state, target, move, ability):
    """Keep the turn counters when a secondary applies no move restriction."""
    return state


def _roll_independent_secondary(roll_key, chances):
    """Roll all slots independently, retaining the first slot's scalar draw."""
    first = random.uniform(roll_key)
    rest = random.uniform(random.fold_in(roll_key, 1), (SECONDARY_SLOTS - 1,))
    return jnp.concatenate((first[None], rest)) < chances


def _roll_exclusive_secondary(roll_key, chances):
    """Select at most one slot, preserving relative odds if the total caps."""
    chances = chances / jnp.maximum(1., jnp.sum(chances))
    ends = jnp.cumsum(chances)
    starts = jnp.concatenate((jnp.zeros_like(ends[:1]), ends[:-1]))
    roll = random.uniform(roll_key)
    return (roll >= starts) & (roll < ends)


def _apply_user_secondary(state, *, player, stats, stages):
    """Apply a secondary's stat changes to the move user."""
    return stat_changes(state, player, stats, stages)


def _apply_secondary_slot(index, carry, *, triggered, player, target, target_allowed, move, ability):
    """Apply both recipients' effects from the same secondary roll."""
    effects = move.secondaries
    apply = partial(_apply_secondary, target=target, move=move, ability=ability, index=index)
    key, state = jax.lax.cond(triggered[index] & target_allowed, apply, _unchanged_secondary, *carry)
    user_stages = effects.user_stages[index]
    user_allowed = jnp.any(user_stages != 0) & (state.active_field('current_hp', player)[0] > 0)
    state = jax.lax.cond(triggered[index] & user_allowed,
        partial(_apply_user_secondary, player=player, stats=effects.user_stats[index], stages=user_stages),
        _preserve_operand, state)
    return key, state


def secondary_effects(key, state, player, move, reached_target=True):
    """Roll each secondary once and gate the user and target effects separately."""
    key, roll_key = jax.random.split(key)
    target = 1-player
    ability = ability_against_move(state, target, move)[0]
    chances = move.secondaries.probabilities * jnp.where(state.ability[player, 0] == A.SERENE_GRACE, 2., 1.)
    triggered = jax.lax.cond(move.secondaries.mode[0] == SecondaryEffectType.EXCLUSIVE,
        _roll_exclusive_secondary, _roll_independent_secondary, roll_key, chances)
    target_allowed = reached_target & (ability != A.SHIELD_DUST) & (state.active_field('current_hp', target)[0] > 0)
    apply = partial(_apply_secondary_slot, triggered=triggered, player=player, target=target,
                    target_allowed=target_allowed, move=move, ability=ability)
    key, state = jax.lax.fori_loop(0, SECONDARY_SLOTS, apply, (key, state))

    return kings_rock_flinch(key, state, player, move, reached_target=reached_target)


def after_damage(state, player, move):
    """Apply guaranteed hit callbacks for terrain, hazards, and stage clearing."""
    ident = move.identifier[0]
    target = 1-player
    state = state.replace(
        hazards=state.hazards.replace(
            stealth_rocks=state.hazards.stealth_rocks.at[target].set(state.hazards.stealth_rocks[target] | (ident == 830)),
            spikes=state.hazards.spikes.at[target].set(jnp.minimum(3, state.hazards.spikes[target] + (ident == 845)))))
    state = jax.lax.cond(ident == 746, _jaw_lock, _skip_jaw_lock, state, player)
    terrain_clear = (ident == 861) | (ident == 798)
    rapid = ident == 229
    mortal = (ident == 866) & (state.active_field('current_hp', player)[0] > 0)
    branch = jnp.select((ident == 499, terrain_clear, rapid | mortal), (0, 1, 2), default=3)
    return jax.lax.switch(branch,
        (partial(_after_damage_clear_boosts, target=target), _after_damage_update_field,
         partial(_after_damage_spin, player=player, rapid=rapid), _preserve_operand), state)
