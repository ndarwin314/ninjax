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
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import StatEnum, Status, Type
from ninjax.knowledge import remember_item
from ninjax.moves.model import Move
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.field_conditions import set_terrain
from ninjax.side.hp_effects import apply_indirect_hp_change, take_damage_percent
from ninjax.side.item_effects import kings_rock_flinch
from ninjax.side.queries import contact_reactions_allowed, is_powder_immune_fields
from ninjax.side.stat_effects import conditional_reduce_boosts, stat_changes
from ninjax.side.state import BattleState
from ninjax.side.status_effects import apply_confusion, set_status
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


def _after_damage_spin(s, *, player):
    """Remove represented entry hazards and raise the surviving user's Speed."""
    s = s.replace(**{name: getattr(s, name).at[player].set(jnp.zeros_like(getattr(s, name)[player]))
                     for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs')})
    return jax.lax.cond(s.active_field('current_hp', player)[0] > 0,
        partial(_after_damage_add_boosts, player=player), _preserve_operand, s)


def _innards_out_damage(state, *, attacker_idx, defender_idx, damage):
    """Attribute Innards Out damage to the fainted holder."""
    previous_source, previous_cause = state.event_source, state.event_cause
    state = state.replace(event_source=jnp.asarray(defender_idx, jnp.int32),
                          event_cause=jnp.int32(Cause.INDIRECT))
    state = apply_indirect_hp_change(state, attacker_idx, damage, source=defender_idx)
    return state.replace(event_source=previous_source, event_cause=previous_cause)


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
    defender_ability = abilities[defender_idx]
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
    key, state = jax.lax.cond(makes_contact[0], do_contact, _after_every_hit_update_field, key, state, attacker_idx)
    # Aftermath is contact retaliation after the holder faints. Damp prevents
    # it, and Pads/Long Reach/Magic Guard protect the surviving attacker.

    damp = jnp.any((state.ability[:, 0] == AbilityEnum.DAMP) & (state.active_field('current_hp')[:, 0] > 0))
    aftermath = ((defender_ability[0] == AbilityEnum.AFTERMATH) & ~defender_alive[0] & (hit_damage > 0) &
        move.contact[0] & contact_reactions_allowed(attacker_ability[0], attacker_item) & ~damp &
        (attacker_ability[0] != AbilityEnum.MAGIC_GUARD) & (state.active_field('current_hp', attacker_idx)[0] > 0))
    source = state.event_source
    state = state.replace(event_source=jnp.asarray(defender_idx, jnp.int32))
    state = take_damage_percent(state, attacker_idx, jnp.where(aftermath, 1/4, 0), cause=Cause.CONTACT)
    state = state.replace(event_source=source)
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


def do_contact(key: chex.PRNGKey, state: BattleState, attacker_idx) -> Tuple[chex.PRNGKey, BattleState]:
    """Apply contact effects, preserving Pads' protection and public reveals."""
    abilities = state.ability
    defender_ability = abilities[1-attacker_idx]
    attacker_ability = abilities[attacker_idx]
    defender_item = state.active_field('item', 1-attacker_idx)
    attacker_item = state.active_field('item', attacker_idx)
    # Snapshot powder immunity before contact damage and status reactions.
    powder_immune = is_powder_immune_fields(state.effective_types(attacker_idx), attacker_ability)
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

    key, state = set_status(key, state, 1 - attacker_idx, status)
    return key, state


def _unchanged_secondary(key, state, *unused):
    """Keep both explicit key and state when a secondary is skipped."""
    return key, state


def _secondary_status(key, state, target, move):
    """Apply the move's represented major-status secondary."""
    return set_status(key, state, target, move.secondary_status)


def _secondary_confusion(key, state, target, move):
    """Apply the move's represented confusion secondary."""

    return apply_confusion(key, state, target)


def _apply_secondary(key, state, player, move, ability):
    """Apply a successful secondary in status, confusion, stat and flinch order."""
    target = 1-player
    key, state = jax.lax.cond(move.secondary_status[0] != 0,
        _secondary_status, _unchanged_secondary, key, state, target, move)
    key, state = jax.lax.cond(move.secondary_confusion[0],
        _secondary_confusion, _unchanged_secondary, key, state, target, move)
    state = stat_changes(state, target, move.secondary_stat, move.secondary_stages, True)
    v = state.volatile_status
    flinch = move.secondary_flinch[0] & (ability != A.INNER_FOCUS)
    return key, state.replace(volatile_status=v.replace(flinched=v.flinched.at[target, 0].set(v.flinched[target, 0] | flinch)))


def secondary_effects(key, state, player, move):
    """Roll one supported secondary effect, respecting Shield Dust and survival."""
    key, roll_key = jax.random.split(key)
    target = 1-player
    ability = state.ability[target, 0]
    chance = move.secondary_chance[0] * jnp.where(state.ability[player, 0] == A.SERENE_GRACE, 2., 1.)
    triggered = (jax.random.uniform(roll_key) < chance) & (ability != A.SHIELD_DUST) & (state.active_field('current_hp', target)[0] > 0)
    key, state = jax.lax.cond(triggered, _apply_secondary, _unchanged_secondary,
                              key, state, player, move, ability)

    return kings_rock_flinch(key, state, player, move)


def after_damage(state, player, move):
    """Apply guaranteed hit callbacks for terrain, hazards, and stage clearing."""
    ident = move.identifier[0]
    target = 1-player
    state = state.replace(
        stealth_rocks=state.stealth_rocks.at[target].set(state.stealth_rocks[target] | (ident == 830)),
        spikes=state.spikes.at[target].set(jnp.minimum(3, state.spikes[target] + (ident == 845))))
    # Clear Smog, Ice Spinner and Rapid Spin each have one distinct callback.
    branch = jnp.select((ident == 499, ident == 861, ident == 229), (0, 1, 2), default=3)
    return jax.lax.switch(branch,
        (partial(_after_damage_clear_boosts, target=target), _after_damage_update_field,
         partial(_after_damage_spin, player=player), _preserve_operand), state)
