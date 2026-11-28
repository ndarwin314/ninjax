"""Switch-cleared Pokémon typing, raw stats, forms and copied move banks."""
import jax
import jax.numpy as jnp
from jax import random

from ninjax.ability_effects import _start_new_ability, install_ability
from ninjax.calculations.stats import StatBoosts
from ninjax.calculations.type_effectiveness import TYPE_EFFECTIVENESS
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.moves import MoveFlags
from ninjax.enums.pokemon import FormEnum, StatEnum as S, Type
from ninjax.records.events import EventKind, emit
from ninjax.side.updates import set_boosts, update_active_fields


TYPE_MOVES = (160, 176, 487, 513, 571, 750)
STAT_MOVES = (244, 379, 384, 385, 391, 470, 471, 683)
COPY_MOVES = (102, 144, 166)


def can_change_type(state, player):
    """Reject mutable typing on Terastallized Pokémon and locked species."""
    species = state.active_identity('species', player)[0]
    return ~state.active_field('is_terastallized', player)[0] & (species != 493) & (species != 773)


def set_types(state, player, types, extra, enabled):
    """Change primary and added type layers together and record the change."""
    old = state.active_field('type_list', player)
    state = update_active_fields(state, player, type_list=jnp.where(enabled, types, old))
    v = state.volatile_status
    added = v.properties.added_type
    previous = added[player, 0]
    state = state.replace(volatile_status=v.replace(properties=v.properties.replace(
        added_type=added.at[player, 0].set(jnp.where(enabled, extra, previous)))))
    state = emit(state, EventKind.TYPE, player, before=old[0], after=types[0], enabled=enabled & (old[0] != types[0]))
    state = emit(state, EventKind.TYPE, player, detail=1, before=old[1], after=types[1], enabled=enabled & (old[1] != types[1]))
    return emit(state, EventKind.TYPE, player, detail=2, before=previous, after=extra, enabled=enabled & (previous != extra))


def _conversion_type(key, possibilities):
    """Choose one eligible Conversion 2 type uniformly."""
    key, draw = random.split(key)
    return key, random.categorical(draw, jnp.where(possibilities, 0., -jnp.inf)).astype(jnp.int32)


def _no_conversion(key, possibilities):
    """Keep the original key when there is no random type choice."""
    return key, jnp.int32(Type.NONE)


@jax.jit
def type_move(key, state, player, ident):
    """Apply the selected type replacement, copying or addition."""
    target = 1-player
    reflect = ident == 513
    conversion = ident == 160
    conversion_two = ident == 176
    add = ident == 571
    own_types = state.effective_types(player, include_added=True)
    other_types = state.effective_types(target, include_added=True)
    first = state.active_move_field('type', player)[0, 0]
    last = state.volatile_status.move_history.last_move_type[target, 0]
    candidates = jnp.arange(TYPE_EFFECTIVENESS.shape[0])
    possibilities = (last != Type.NONE) & (candidates != Type.NONE) & (
        TYPE_EFFECTIVENESS[last] < 1) & ~jnp.any(candidates[:, None] == own_types, axis=1)
    eligible = conversion_two & jnp.any(possibilities) & can_change_type(state, player)
    key, random_type = jax.lax.cond(eligible, _conversion_type, _no_conversion, key, possibilities)
    mono = jnp.select((conversion, conversion_two, ident == 750), (first, random_type, Type.PSYCHIC), default=Type.WATER)
    recipient = jnp.where(reflect | conversion | conversion_two, player, target)
    copied = other_types[:2]
    copied = copied.at[0].set(jnp.where(jnp.all(copied == Type.NONE) & (other_types[2] != Type.NONE), Type.NORMAL, copied[0]))
    types = jnp.where(reflect, copied, jnp.stack((mono, jnp.int32(Type.NONE))))
    types = jnp.where(add, state.active_field('type_list', target), types)
    extra = jnp.where(add, Type.GRASS, jnp.where(reflect, other_types[2], Type.NONE))
    new = jnp.where(reflect, jnp.any(copied != Type.NONE),
        jnp.where(add, ~jnp.any(other_types == Type.GRASS),
        jnp.where(conversion, (first != Type.NONE) & ~jnp.any(own_types == first),
        jnp.where(conversion_two, eligible, ~jnp.all(other_types == jnp.array([mono, 0, 0]))))))
    success = new & can_change_type(state, recipient)
    return key, set_types(state, recipient, types, extra, success), success


def lose_type(state, player, removed):
    """Remove an elemental type after a successful Double Shock or Burn Up."""
    types = state.active_field('type_list', player)
    extra = state.volatile_status.properties.added_type[player, 0]
    return set_types(state, player, jnp.where(types == removed, Type.NONE, types),
                     jnp.where(extra == removed, Type.NONE, extra), can_change_type(state, player))


def color_change(state, player, hit_type, enabled):
    """Change a surviving target to the type of a damaging hit."""
    types = state.effective_types(player, include_added=True)
    change = enabled & (hit_type != Type.NONE) & ~jnp.any(types == hit_type) & can_change_type(state, player)
    state = set_types(state, player, jnp.stack((hit_type, jnp.int32(Type.NONE))), Type.NONE, change)
    return emit(state, EventKind.ABILITY, player, detail=A.COLOR_CHANGE, enabled=change)


def _raw_stat_move(state, player, ident):
    """Apply Split, Speed Swap and Power Trick to unmodified stats."""
    target = 1-player
    stats = state.active_stats().astype(jnp.int32)
    v = state.volatile_status
    overrides = v.properties.stats
    columns = jnp.arange(6)
    guard = (columns == S.DEFENSE) | (columns == S.SPECIAL_DEFENSE)
    power = (columns == S.ATTACK) | (columns == S.SPECIAL_ATTACK)
    speed = columns == S.SPEED
    split = (stats[player] + stats[target]) // 2
    trick = stats[player].at[S.ATTACK].set(stats[player, S.DEFENSE]).at[S.DEFENSE].set(stats[player, S.ATTACK])
    selected = jnp.where(ident == 470, guard, jnp.where(ident == 471, power, speed))
    own = jnp.where(selected, jnp.where(ident == 683, stats[target], split), overrides[player])
    own = jnp.where(ident == 379, jnp.where((columns == S.DEFENSE) | (columns == S.ATTACK),
                                          trick, overrides[player]), own)
    other = jnp.where(selected & (ident != 379), jnp.where(ident == 683, stats[player], split), overrides[target])
    changed = overrides.at[player].set(own).at[target].set(other)
    trick_state = v.properties.power_trick
    active = trick_state.at[player, 0].set(trick_state[player, 0] ^ (ident == 379))
    return state.replace(volatile_status=v.replace(properties=v.properties.replace(stats=changed, power_trick=active)))


def pass_power_trick(state, player):
    """Apply a passed Power Trick to the recipient’s own raw stats."""
    return _raw_stat_move(state, player, jnp.int32(379))


def _stage_move(state, player, ident):
    """Copy or exchange stages without normal stage increase modifiers."""
    target = 1-player
    own, other = state.boosts[player], state.boosts[target]
    columns = jnp.arange(6)
    attack = (columns == S.ATTACK) | (columns == S.SPECIAL_ATTACK)
    defense = (columns == S.DEFENSE) | (columns == S.SPECIAL_DEFENSE)
    selected = jnp.where(ident == 384, attack, jnp.where(ident == 385, defense, True))
    all_stages = (ident == 244) | (ident == 391)
    state = set_boosts(state, player, StatBoosts(
        normal_boosts=jnp.where(selected, other.normal_boosts, own.normal_boosts),
        acc_boosts=jnp.where(all_stages, other.acc_boosts, own.acc_boosts)))
    return jax.lax.cond(ident != 244, set_boosts, _skip_boosts, state, target, StatBoosts(
        normal_boosts=jnp.where(selected, own.normal_boosts, other.normal_boosts),
        acc_boosts=jnp.where(all_stages, own.acc_boosts, other.acc_boosts)))


def _skip_boosts(state, player, boosts):
    """Preserve the target's stages during Psych Up."""
    return state


@jax.jit
def stat_move(state, player, ident):
    """Dispatch direct stage transfers or raw stat overrides."""
    return jax.lax.cond(jnp.isin(ident, jnp.array([244, 384, 385, 391])),
                        _stage_move, _raw_stat_move, state, player, ident)


def _choose_slots(mask, new, old):
    """Choose compatible move leaves with a mask over their move slots."""
    return jnp.where(mask.reshape((mask.size,) + (1,) * (new.ndim - 1)), new, old)


def _write_move_bank(state, player, bank):
    """Write one active move bank to its party slot."""
    moves = state.team.moves
    slot = state.active_index[player, 0]
    updated = jax.tree.map(lambda table, values: table.at[player, slot].set(values), moves, bank)
    return state.replace(team=state.team.replace(moves=updated))


def _pack_bank(bank):
    """Store move leaves in one buffer while preserving float bit patterns."""
    leaves = jax.tree.leaves(bank)
    return jnp.concatenate(tuple(
        jax.lax.bitcast_convert_type(leaf, jnp.int32).reshape(-1)
        if leaf.dtype == jnp.float32 else leaf.astype(jnp.int32).reshape(-1) for leaf in leaves))


def _unpack_bank(buffer, reference):
    """Recover the leaf shapes and dtypes from the live move schema."""
    leaves, tree = jax.tree.flatten(reference)
    offset = 0
    unpacked = []
    for leaf in leaves:
        values = buffer[offset:offset + leaf.size].reshape(leaf.shape)
        decoded = jax.lax.bitcast_convert_type(values, jnp.float32) if leaf.dtype == jnp.float32 else values.astype(leaf.dtype)
        unpacked.append(decoded)
        offset += leaf.size
    return jax.tree.unflatten(tree, unpacked)


def _save_and_write_bank(state, player, bank, slots):
    """Save newly replaced slots and retain originals for already copied slots."""
    current = state.active_moves(player)
    v = state.volatile_status
    copied = v.properties.copied_slots[player]
    original = _unpack_bank(state.original_moves[player], current)
    saved = jax.tree.map(lambda previous, real: _choose_slots(copied, previous, real), original, current)
    stored = state.original_moves.at[player].set(_pack_bank(saved))
    slot = state.active_index[player, 0]
    live_uses = state.used_move_ids[player, slot]
    saved_uses = jnp.where(copied, state.original_used_move_ids[player], live_uses)
    state = state.replace(original_moves=stored,
        original_used_move_ids=state.original_used_move_ids.at[player].set(saved_uses),
        used_move_ids=state.used_move_ids.at[player, slot].set(jnp.where(slots, 0, live_uses)),
        volatile_status=v.replace(
        properties=v.properties.replace(copied_slots=v.properties.copied_slots.at[player].set(copied | slots))))
    return _write_move_bank(state, player, bank)


@jax.jit
def restore_move_bank(state, player):
    """Restore only temporary slots, preserving other moves' spent PP."""
    copied = state.volatile_status.properties.copied_slots[player]
    return jax.lax.cond(jnp.any(copied), _restore_copied, _skip_player, state, player, copied)


def _restore_copied(state, player, copied):
    """Restore selected original move leaves on departure."""
    current = state.active_moves(player)
    original = _unpack_bank(state.original_moves[player], current)
    bank = jax.tree.map(lambda old, live: _choose_slots(copied, old, live), original, current)
    slot = state.active_index[player, 0]
    uses = jnp.where(copied, state.original_used_move_ids[player], state.used_move_ids[player, slot])
    state = state.replace(used_move_ids=state.used_move_ids.at[player, slot].set(uses))
    return _write_move_bank(state, player, bank)


def _skip_player(state, *unused):
    """Accept an inactive effect's operands without mutating state."""
    return state


def _base_pp(move):
    """Recover catalogue base PP from the max-PP-boosted team projection."""
    return jnp.where(move.native_pp > 0, move.native_pp, (move.max_pp * 5 + 7) // 8)


def _copy_single_move(state, player, move, index, sketch):
    """Replace one slot using Mimic's temporary or Sketch's persistent policy."""
    new = move.replace(current_pp=_base_pp(move), max_pp=_base_pp(move), disabled=jnp.zeros_like(move.disabled))
    original = state.active_moves(player)
    bank = jax.tree.map(lambda existing, replacement: existing.at[index].set(replacement), original, new)
    slots = jnp.arange(original.current_pp.shape[0]) == index
    state = jax.lax.cond(sketch, _write_move_bank_sketch, _save_and_write_bank, state, player, bank, slots)
    return emit(state, EventKind.MOVE_COPY, player, detail=move.identifier[0], before=index, after=sketch)


def _write_move_bank_sketch(state, player, bank, slots):
    """Install a persistent battle move without assigning temporary PP."""
    slot = state.active_index[player, 0]
    used = state.used_move_ids[player, slot]
    state = state.replace(used_move_ids=state.used_move_ids.at[player, slot].set(jnp.where(slots, 0, used)))
    return _write_move_bank(state, player, bank)


@jax.jit
def copy_move(state, player, ident):
    """Copy the opponent's most recently used eligible move."""
    target = 1-player
    own = state.active_move_field('identifier', player)[:, 0]
    opponent = state.active_moves(target)
    last = state.volatile_status.move_history.last_move_id[target, 0]
    matching = opponent.identifier[:, 0] == last
    candidate = opponent[jnp.argmax(matching)]
    sketch = ident == 166
    prohibited = jnp.where(sketch, int(MoveFlags.CANNOT_SKETCH), int(MoveFlags.CANNOT_MIMIC))
    allowed = (last != 0) & jnp.any(matching) & ~jnp.any(own == last) & jnp.any(own == ident) & (
        candidate.implemented[0] & ((candidate.move_flags[0] & prohibited) == 0)) & ~state.volatile_status.transformed[player, 0]
    state = jax.lax.cond(allowed, _copy_single_move, _skip_player, state, player, candidate,
                         jnp.argmax(own == ident), sketch)
    return state, allowed


def _apply_transform(state, player):
    """Copy combat identity and stages while retaining the user's HP and item."""
    target = 1-player
    bank = state.active_moves(target)
    pp = jnp.minimum(5, _base_pp(bank))
    bank = bank.replace(current_pp=pp, max_pp=pp, disabled=jnp.zeros_like(bank.disabled))
    state = _save_and_write_bank(state, player, bank, jnp.ones(bank.current_pp.shape[0], bool))
    types = state.effective_types(target, apply_roost=False)
    extra = state.volatile_status.properties.added_type[target, 0]
    state = set_types(state, player, types, extra, True)
    v = state.volatile_status
    p = v.properties
    p = p.replace(
        stats=p.stats.at[player, 1:].set(state.active_stats(target)[1:]),
        form=p.form.at[player].set(state.active_identity('form', target)),
        species=p.species.at[player].set(state.active_identity('species', target)),
        weight_hg=p.weight_hg.at[player].set(state.active_identity('weight_hg', target)))
    state = state.replace(volatile_status=v.replace(properties=p, transformed=v.transformed.at[player, 0].set(True)))
    state = set_boosts(state, player, state.boosts[target])
    identity = state.ability_identity[target, 0]
    changed = state.ability_identity[player, 0] != identity
    state = install_ability(state, player, identity)
    state = jax.lax.cond(changed, _start_new_ability, _skip_player, state, player)
    return emit(state, EventKind.FORM, player, detail=p.species[player, 0], after=p.form[player, 0])


@jax.jit
def transform(state, player, enabled=True):
    """Apply Transform or Imposter to a living, unsubstituted eligible foe."""
    target = 1-player
    v = state.volatile_status
    eligible = enabled & ~jnp.any(v.transformed) & (v.substitute_hp[target, 0] == 0) & (
        jnp.all(state.active_field('current_hp') > 0))
    return jax.lax.cond(eligible, _apply_transform, _skip_player, state, player), eligible


def toggle_hunger_form(state, player):
    """Toggle living Morpeko after its residual turn."""
    v = state.volatile_status
    form = state.active_identity('form', player)[0]
    living = state.active_field('current_hp', player)[0] > 0
    eligible = living & (state.ability[player, 0] == A.HUNGER_SWITCH) & (
        state.active_identity('species', player)[0] == 877) & ~state.active_field('is_terastallized', player)[0]
    after = jnp.where(form == FormEnum.MORPEKO_HANGRY, FormEnum.DEFAULT, FormEnum.MORPEKO_HANGRY)
    p = v.properties
    state = state.replace(volatile_status=v.replace(properties=p.replace(
        form=p.form.at[player, 0].set(jnp.where(eligible, after, p.form[player, 0])))))
    return emit(state, EventKind.FORM, player, detail=877, before=form, after=after, enabled=eligible)
