"""Item mutations and update hooks built on basic state writes."""
from functools import partial

import jax
import jax.numpy as jnp

from ninjax.calculations.stats import StatBoosts
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import TerrainEnum as T, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import StatEnum as S, Status, Type
from ninjax.item_rules import item_is_suppressed
from ninjax.knowledge import remember_item
from ninjax.records.events import EventKind, emit
from ninjax.side.queries import (
    ability_against_move,
    berry_allowed,
    can_remove_item,
    effective_weather,
    is_choice_item,
    item_transfer_allowed,
)
from ninjax.side.updates import add_boosts, set_boosts, update_active_fields


def _status_berry_cure(s, *, player, lum, confused):
    """Consume before clearing conditions so item knowledge remains accurate."""
    return _cure_with_berry(consume(s, player), player=player, lum=lum, confused=confused)


def _cure_with_berry(s, *, player, lum, confused):
    """Apply the status effect of a berry that has already been eaten."""
    s = cure_status(s, player)
    v = s.volatile_status
    s = s.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player, 0].set(
        jnp.where(lum, 0, v.confusion_turns[player, 0]))))
    s = emit(s, EventKind.CONDITION, player, detail=12, before=1, after=0, enabled=lum & confused)
    return cheek_pouch(s, player)


def _identity(s):
    """Preserve the operand when a conditional effect is skipped."""
    return s


def _white_herb_restore(s, *, player, normal, accuracy):
    """Keep positive stages, including Shell Smash's offensive increases."""
    return _restore_negative_stages(consume(s, player), player=player, normal=normal, accuracy=accuracy)


def _restore_negative_stages(s, *, player, normal, accuracy):
    """Restore negative stage counters without interacting with held items."""
    return set_boosts(s, player, StatBoosts(
        normal_boosts=jnp.maximum(normal, 0), acc_boosts=jnp.maximum(accuracy, 0)))


def apply_flung_item(state, target, item):
    """Apply a thrown berry or herb to a living direct-hit recipient."""
    state = apply_eaten_berry(state, target, item, effect_active=held_item_active(state, target))
    alive = state.active_field('current_hp', target)[0] > 0
    state = clear_mental_restrictions(state, target, alive & (item == I.MENTAL_HERB))
    return jax.lax.cond(alive & (item == I.WHITE_HERB), partial(
        _restore_negative_stages, player=target, normal=state.boosts.normal_boosts[target],
        accuracy=state.boosts.acc_boosts[target]), _identity, state)


def _update_items_update(player, s):
    """Check status berries before the post-action White Herb check."""
    return white_herb(status_berry(after_hp_change(s, player), player), player)


def _kings_rock_flinch_roll(args, *, player, target):
    """Serene Grace also doubles the item-added secondary chance."""
    key, s = args
    key, roll_key = jax.random.split(key)
    chance = jnp.where(s.ability[player, 0] == A.SERENE_GRACE, .2, .1)
    v = s.volatile_status
    flinch = v.flinched[target, 0] | (jax.random.uniform(roll_key) < chance)
    return key, s.replace(volatile_status=v.replace(flinched=v.flinched.at[target, 0].set(flinch)))


def _remove_held_item(s, *, holder, source):
    """Remove held item."""
    return _assign_item(s, holder, I.NONE, source)


def _swap_items_exchange(s, *, source, other, target, own):
    """Publish both sides of the swap before reacting to received items."""
    previous_source = s.event_context.source
    s = s.replace(event_context=s.event_context.replace(source=jnp.asarray(source, jnp.int32)))
    s = _assign_item(s, source, other, source)
    s = _assign_item(s, target, own, source)
    s = _activate_received_item(s, source)
    s = _activate_received_item(s, target)
    s = refresh_paradox(s)
    return s.replace(event_context=s.event_context.replace(source=previous_source))


def _activate_weakness_policy(s, *, target):
    """Activate weakness policy."""
    return add_boosts(s, target,
            jnp.array([S.ATTACK, S.SPECIAL_ATTACK]), jnp.array([2, 2]))


def _activate_absorb_bulb(s, *, target):
    """Activate absorb bulb."""
    return add_boosts(s, target, S.SPECIAL_ATTACK, 1)


def cure_status(state, player):
    """Clear a major status, its sleep counter and its toxic counter together."""

    old = state.active_field('status', player)[0]
    state = emit(state, EventKind.STATUS, player, before=old, after=Status.NONE, enabled=old != Status.NONE)
    state = update_active_fields(state, player, status=jnp.array([Status.NONE]), sleep_counter=jnp.zeros(1, jnp.int32))
    return state.replace(toxic_counter=state.toxic_counter.at[player, 0].set(0))


def cheek_pouch(state, player, eaten=True):
    """Heal after a represented berry's own effect, without recursive HP hooks."""
    current_hp = state.active_field('current_hp', player)
    max_hp = state.active_stats(player)[0]
    can_heal = state.volatile_status.restrictions.heal_block_turns[player, 0] == 0
    hp = _cheek_pouch_hp(current_hp, max_hp, state.ability[player, 0], eaten & can_heal)
    state = _emit_hp_change(state, player, current_hp, hp, max_hp)
    return update_active_fields(state, player, current_hp=hp)


def _cheek_pouch_hp(hp, max_hp, ability, eaten):
    """Calculate the additional Cheek Pouch healing after a berry."""
    eligible = eaten & (ability == A.CHEEK_POUCH) & (hp[0] > 0)
    return jnp.minimum(max_hp, hp + jnp.where(eligible, jnp.maximum(1, max_hp // 3), 0))


def _emit_hp_change(state, player, before, after, max_hp):
    """Emit the actual HP change and its public rounded fraction."""
    return emit(state, EventKind.HP, player, before=before, after=after,
        hp_change=after-before,
        hp_fraction_change=(jnp.ceil(after/max_hp*100)-jnp.ceil(before/max_hp*100))/100,
        enabled=after[0] != before[0])


@jax.jit
def status_berry(state, player):
    """Eat Chesto for sleep or Lum for major status/confusion, respecting Unnerve."""

    item = state.active_field('item', player)
    status = state.active_field('status', player)
    current_hp = state.active_field('current_hp', player)
    confused = state.volatile_status.confusion_turns[player, 0] > 0
    lum = item[0] == I.LUM_BERRY
    eligible = ((lum & ((status[0] != Status.NONE) | confused)) |
                ((item[0] == I.CHESTO_BERRY) & (status[0] == Status.SLEEP)))
    eligible &= (current_hp[0] > 0) & berry_allowed(state.ability[1-player, 0], state.active_field('current_hp', 1-player)[0])
    eligible &= held_item_active(state, player)
    cure = partial(_status_berry_cure, player=player, lum=lum, confused=confused)
    return jax.lax.cond(eligible, cure, _identity, state)


@jax.jit
def white_herb(state, player):
    """At an update boundary, consume once to clear all negative stat stages."""

    normal, accuracy = state.boosts.normal_boosts[player], state.boosts.acc_boosts[player]
    eligible = ((state.active_field('item', player)[0] == I.WHITE_HERB) & (state.active_field('current_hp', player)[0] > 0) &
                (jnp.any(normal < 0) | jnp.any(accuracy < 0)) & held_item_active(state, player))
    restore = partial(_white_herb_restore, player=player, normal=normal, accuracy=accuracy)
    return jax.lax.cond(eligible, restore, _identity, state)


@jax.jit
def update_items(state):
    """Apply represented item updates after a whole move or switch resolves."""
    update = _update_items_update
    return jax.lax.fori_loop(0, 2, update, state)


def kings_rock_flinch(key, state, player, move, *, reached_target=True):
    """Roll an independent item secondary once per hit reaching the Pokémon."""

    target = 1-player
    ability = ability_against_move(state, target, move)[0]
    eligible = (reached_target & (state.active_field('item', player)[0] == I.KINGS_ROCK) &
        (move.move_type[0] != MoveType.STATUS) & ~jnp.any(move.secondaries.flinch) &
        # Sheer Force runs before the item adds its secondary. Its later
        # suppression applies only when the original move earned its boost.
        ((state.ability[player, 0] != A.SHEER_FORCE) | ~jnp.any(move.secondaries.probabilities > 0)) &
        (ability != A.SHIELD_DUST) & (ability != A.INNER_FOCUS) &
        (state.active_field('current_hp', target)[0] > 0))
    roll = partial(_kings_rock_flinch_roll, player=player, target=target)
    return jax.lax.cond(eligible, roll, _identity, (key, state))


def held_item_active(state, player):
    """Check the holder's represented item suppression rules."""
    return ~item_is_suppressed(state.ability[player, 0], state.magic_room_duration,
                               state.volatile_status.embargo_turns[player, 0])


def is_berry(item):
    """Recognize every berry in the represented item inventory."""
    return (item == I.SITRUS_BERRY) | (item == I.CHESTO_BERRY) | (item == I.LUM_BERRY)


def consume(state, player, enabled=True, *, recoverable=True, eaten=True):
    """Consume a held item and remember its publicly visible disappearance."""
    return _consume_held_item(state, player, state.active_field('item', player), enabled, recoverable, eaten)


def _consume_held_item(state, player, item, enabled, recoverable=True, eaten=True):
    """Consume the supplied current item and publish its disappearance."""
    used = jnp.asarray(enabled).reshape(()) & (item[0] != I.NONE)
    state = emit(state, EventKind.ITEM, player, detail=item[0], enabled=used)
    state = remember_item(state, player, I.NONE, used)
    slot = state.active_index[player, 0]
    old = state.consumed_items[player, slot]
    v = state.volatile_status
    lock = v.restrictions.choice_lock[player, 0]
    state = state.replace(
        consumed_items=state.consumed_items.at[player, slot].set(jnp.where(used & recoverable, item[0], old)),
        ate_berry=state.ate_berry.at[player, slot].set(
            state.ate_berry[player, slot] | (used & eaten & is_berry(item[0]))),
        volatile_status=v.replace(restrictions=v.restrictions.replace(
            choice_lock=v.restrictions.choice_lock.at[player, 0].set(jnp.where(used & is_choice_item(item[0]), -1, lock)))))
    return update_active_fields(state, player, item=jnp.where(used, I.NONE, item))


def _assign_item(state, holder, item, source):
    """Commit a validated public item mutation and clean up Choice lock state."""
    old = state.active_field('item', holder)[0]
    old_choice = is_choice_item(old)
    new_choice = is_choice_item(item)
    state = update_active_fields(state, holder, item=jnp.asarray(item, jnp.int32).reshape((1,)))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(restrictions=v.restrictions.replace(choice_lock=v.restrictions.choice_lock.at[holder, 0].set(
        jnp.where(old_choice & new_choice, v.restrictions.choice_lock[holder, 0], -1)))))
    state = remember_item(state, holder, item)
    previous = state.event_context.source
    state = state.replace(event_context=state.event_context.replace(source=jnp.asarray(source, jnp.int32)))
    state = emit(state, EventKind.ITEM, holder, detail=old, before=old, after=item)
    state = emit(state, EventKind.ITEM_REVEALED, holder, detail=item, enabled=item != I.NONE)
    return state.replace(event_context=state.event_context.replace(source=previous))


def _activate_received_item(state, holder):
    """Run represented Sitrus and Mental Herb effects after an atomic transfer."""
    state = white_herb(status_berry(after_hp_change(state, holder), holder), holder)
    return mental_herb(state, holder)


def mental_herb(state, holder, enabled=True):
    """Consume an active Mental Herb and clear its supported restrictions."""
    v = state.volatile_status
    herb = (enabled & (state.active_field('item', holder)[0] == I.MENTAL_HERB) & (state.active_field('current_hp', holder)[0] > 0) &
            ((v.restrictions.disable_turns[holder, 0] > 0) | (v.restrictions.encore_turns[holder, 0] > 0) |
             (v.restrictions.taunt_turns[holder, 0] > 0) | (v.restrictions.heal_block_turns[holder, 0] > 0) |
             v.restrictions.tormented[holder, 0]) &
            ~item_is_suppressed(state.ability[holder, 0], state.magic_room_duration, v.embargo_turns[holder, 0]))
    state = consume(state, holder, herb)
    return clear_mental_restrictions(state, holder, herb)


def clear_mental_restrictions(state, holder, enabled=True):
    """Clear represented conditions cured by an applied Mental Herb."""
    herb = jnp.asarray(enabled)
    v = state.volatile_status
    restrictions = v.restrictions
    restrictions = restrictions.replace(tormented=restrictions.tormented.at[holder, 0].set(
        restrictions.tormented[holder, 0] & ~herb))
    for name in ('disabled_move_id', 'disable_turns', 'encore_move_id', 'encore_turns', 'taunt_turns', 'heal_block_turns'):
        values = getattr(restrictions, name)
        restrictions = restrictions.replace(**{
            name: values.at[holder, 0].set(jnp.where(herb, 0, values[holder, 0]))})
    return state.replace(volatile_status=v.replace(restrictions=restrictions))


def remove_item(state, holder, source, enabled=True):
    """Return ``(state, success)`` after validated removal of an active item.

    Failure changes no items, knowledge or events. Move-specific failure and
    ability-reveal messages belong to the caller. Removal is not consumption.
    """
    success = jnp.asarray(enabled).reshape(()) & can_remove_item(state, holder, source)
    state = jax.lax.cond(success, partial(_remove_held_item, holder=holder, source=source), _identity, state)
    return state, success


def swap_items(state, source, target, enabled=True):
    """Return ``(state, success)`` for an atomic Trick/Switcheroo-style exchange.

    Empty-handed exchanges and identical nonempty items are permitted. The
    opposing target's Sticky Hold blocks even an empty-handed target; the
    source's own Sticky Hold does not prevent giving its item. Validate both
    outgoing and incoming bindings before committing. Item activation follows
    both assignments, so a received berry cannot change the transferred item.
    """
    own, other = state.active_field('item', source)[0], state.active_field('item', target)[0]
    species, forms = state.active_field('species')[:, 0], state.active_field('form')[:, 0]
    sticky = state.ability[target, 0] == A.STICKY_HOLD
    success = (jnp.asarray(enabled).reshape(()) & (source != target) &
        (state.active_field('current_hp', source)[0] > 0) & (state.active_field('current_hp', target)[0] > 0) & ~sticky &
        ((own != I.NONE) | (other != I.NONE)) &
        item_transfer_allowed(species[source], forms[source], own) &
        item_transfer_allowed(species[target], forms[target], other) &
        item_transfer_allowed(species[source], forms[source], other) &
        item_transfer_allowed(species[target], forms[target], own))
    exchange = partial(_swap_items_exchange, source=source, other=other, target=target, own=own)
    return jax.lax.cond(success, exchange, _identity, state), success


def _steal_item(state, *, recipient, holder, item):
    """Transfer both slots before checking the recipient's item reaction."""
    state = _assign_item(state, holder, I.NONE, recipient)
    return _activate_received_item(_assign_item(state, recipient, item, recipient), recipient)


def steal_item(state, recipient, holder):
    """Take a removable item into an empty living recipient's slot."""
    item = state.active_field('item', holder)[0]
    allowed = ((state.active_field('item', recipient)[0] == I.NONE) &
               (state.active_field('current_hp', recipient)[0] > 0) & can_remove_item(state, holder, recipient) &
               item_transfer_allowed(state.active_field('species', recipient)[0],
                                     state.active_field('form', recipient)[0], item))
    return jax.lax.cond(allowed, partial(_steal_item, recipient=recipient, holder=holder, item=item),
                       _identity, state)


def _apply_berry(state, *, player, item):
    """Apply an already eaten berry to its eater, which may hold another item."""
    hp = state.active_field('current_hp', player)
    maximum = state.active_stats(player)[0]
    heal = (item == I.SITRUS_BERRY) & (state.volatile_status.restrictions.heal_block_turns[player, 0] == 0)
    healed = jnp.minimum(maximum, hp + jnp.where(heal, jnp.maximum(1, maximum // 4), 0))
    state = _emit_hp_change(state, player, hp, healed, maximum)
    state = update_active_fields(state, player, current_hp=healed)
    lum = item == I.LUM_BERRY
    confused = state.volatile_status.confusion_turns[player, 0] > 0
    status = state.active_field('status', player)[0]
    cure = (lum & ((status != Status.NONE) | confused)) | ((item == I.CHESTO_BERRY) & (status == Status.SLEEP))
    state = jax.lax.cond(cure, partial(_cure_with_berry, player=player, lum=lum, confused=confused),
                         partial(cheek_pouch, player=player), state)
    return state


def apply_eaten_berry(state, player, item, enabled=True, *, effect_active=True):
    """Apply a berry's effects and the eater's ability without removing an item."""
    eligible = enabled & is_berry(item) & (state.active_field('current_hp', player)[0] > 0)
    payload = jnp.where(effect_active, item, I.NONE)
    return jax.lax.cond(eligible, partial(_apply_berry, player=player, item=payload), _identity, state)


def force_eat_berry(state, player, enabled=True):
    """Consume a held berry on demand, applying its effect only when active."""
    item = state.active_field('item', player)[0]
    eaten = enabled & is_berry(item) & (state.active_field('current_hp', player)[0] > 0)
    effect_active = held_item_active(state, player)
    state = consume(state, player, eaten)
    return apply_eaten_berry(state, player, item, eaten, effect_active=effect_active)


def release_fling_item(state, player, item):
    """Spend the prepared Fling item once, even after a failed hit."""
    return consume(state, player, (item != I.NONE) & (state.active_field('item', player)[0] == item), eaten=False)


def destroy_opposing_berry(state, player, target):
    """Remove a represented Incinerate target without recording consumption."""
    item = state.active_field('item', target)[0]
    return remove_item(state, target, player, is_berry(item))[0]


def eat_opposing_berry(state, player, target):
    """Destroy the target's removable berry and apply it to a living attacker."""
    item = state.active_field('item', target)[0]
    state, removed = remove_item(state, target, player, is_berry(item) & (state.active_field('current_hp', player)[0] > 0))
    return apply_eaten_berry(state, player, item, removed, effect_active=held_item_active(state, player))


def _recover_item(state, *, player, slot, item):
    """Clear prior consumption before assigning and activating the restored item."""
    state = state.replace(consumed_items=state.consumed_items.at[player, slot].set(I.NONE))
    return _activate_received_item(_assign_item(state, player, item, player), player)


def recycle_item(state, player):
    """Recover this party member's last consumed item into an empty slot."""
    slot = state.active_index[player, 0]
    item = state.consumed_items[player, slot]
    allowed = (item != I.NONE) & (state.active_field('item', player)[0] == I.NONE)
    return jax.lax.cond(allowed, partial(_recover_item, player=player, slot=slot, item=item),
                       _identity, state), allowed


def refresh_paradox(state):
    """Activate once per field effect, retaining the chosen stat until it ends."""
    abilities = state.ability
    ability = abilities[:, 0]
    hp = state.active_field('current_hp')
    items = state.active_field('item')
    weather = effective_weather(state.weather.weather, abilities, hp)
    field = ((ability == A.PROTOSYNTHESIS) & (weather == W.SUN)) | (
             (ability == A.QUARK_DRIVE) & (state.terrain.terrain == T.ELECTRIC))
    eligible = (ability == A.PROTOSYNTHESIS) | (ability == A.QUARK_DRIVE)
    # Booster Energy consumption changes neither base stats nor stages.
    best_stats = 1 + jnp.argmax(state.raw_boosted_stats[:, 1:], axis=1)
    v = state.volatile_status
    for player in range(2):
        live = hp[player, 0] > 0
        item = items[player]
        use = live & eligible[player] & ~field[player] & ~v.ability_state.paradox_booster[player, 0] & (item[0] == I.BOOSTER_ENERGY)
        state = _consume_held_item(state, player, item, use)
        boosted = v.ability_state.paradox_booster[player, 0] | use
        active = live & eligible[player] & (field[player] | boosted)
        # getBestStat uses stages, with Atk/Def/SpA/SpD/Spe tie order.
        best = best_stats[player]
        old = v.ability_state.paradox_stat[player, 0]
        stat = jnp.where(active, jnp.where((old >= 1) & ~use, old, best), -1)
        state = emit(state, EventKind.ABILITY, player, detail=ability[player], after=stat,
                     enabled=(stat != old))
        v = v.replace(
            ability_state=v.ability_state.replace(
                paradox_stat=v.ability_state.paradox_stat.at[player, 0].set(stat),
                paradox_booster=v.ability_state.paradox_booster.at[player, 0].set(boosted)))
    return state.replace(volatile_status=v)


def after_hp_change(state, player):
    """Consume a reactive Sitrus Berry and update only the holder's HP leaf."""
    item = state.active_field('item', player)
    current_hp = state.active_field('current_hp', player)
    max_hp = state.active_stats(player)[0]
    heal_block = state.volatile_status.restrictions.heal_block_turns[player, 0] > 0
    berry = ((item[0] == I.SITRUS_BERRY) & (current_hp[0] > 0) &
             ((current_hp[0] / max_hp) <= .5) &
             berry_allowed(state.ability[1-player, 0], state.active_field('current_hp', 1-player)[0]))
    berry &= ~heal_block & held_item_active(state, player)
    state = _consume_held_item(state, player, item, berry)
    berry_hp = jnp.minimum(max_hp, current_hp + jnp.where(berry, jnp.maximum(1, max_hp // 4), 0))
    final_hp = _cheek_pouch_hp(berry_hp, max_hp, state.ability[player, 0], berry)
    # Each effect retains its own event and intermediate HP endpoint.
    state = _emit_hp_change(state, player, current_hp, berry_hp, max_hp)
    state = _emit_hp_change(state, player, berry_hp, final_hp, max_hp)
    return update_active_fields(state, player, current_hp=final_hp)


def after_hit(state, attacker, move, damage, effectiveness):
    """Activate represented held items after a damaging hit reaches its holder."""

    target = 1-attacker
    item = state.active_field('item', target)
    current_hp = state.active_field('current_hp', target)
    policy = (item[0] == I.WEAKNESS_POLICY) & (current_hp[0] > 0) & (damage > 0) & (effectiveness > 1)
    state = consume(state, target, policy)
    state = jax.lax.cond(jnp.asarray(policy).reshape(()), partial(_activate_weakness_policy, target=target), _identity, state)
    bulb = ((item[0] == I.ABSORB_BULB) & (current_hp[0] > 0) &
            (damage > 0) & (move.type[0] == Type.WATER))
    state = consume(state, target, bulb)
    return jax.lax.cond(jnp.asarray(bulb).reshape(()),
        partial(_activate_absorb_bulb, target=target), _identity, state)
