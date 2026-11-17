"""Item mutations and update hooks built on basic state writes."""
from functools import partial
import jax
import jax.numpy as jnp
from ninjax.enum_types import (
    AbilityEnum as A,
    ItemEnum as I,
    MoveType,
    StatEnum as S,
    Status,
    TerrainEnum as T,
    Type,
    WeatherEnum as W,
)
from ninjax.events import EventKind, emit
from ninjax.stats import StatBoosts
from ninjax.side.updates import add_boosts, set_boosts, update_active_fields
from ninjax.side.queries import (
    berry_allowed,
    can_remove_item,
    effective_weather,
    is_choice_item,
    item_transfer_allowed,
)
from ninjax.knowledge import remember_item

def _status_berry_cure(s, *, player, lum, confused):
    """Consume before clearing conditions so item knowledge remains accurate."""
    s = cure_status(consume(s, player), player)
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
    return set_boosts(consume(s, player), player, StatBoosts(
        normal_boosts=jnp.maximum(normal, 0), acc_boosts=jnp.maximum(accuracy, 0)))


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
    previous_source = s.event_source
    s = s.replace(event_source=jnp.asarray(source, jnp.int32))
    s = _assign_item(s, source, other, source)
    s = _assign_item(s, target, own, source)
    s = _activate_received_item(s, source)
    s = _activate_received_item(s, target)
    return refresh_paradox(s).replace(event_source=previous_source)


def _activate_weakness_policy(s, *, target):
    """Activate weakness policy."""
    return add_boosts(s, target,
            jnp.array([S.ATTACK, S.SPECIAL_ATTACK]), jnp.array([2, 2]))


def _activate_absorb_bulb(s, *, target):
    """Activate absorb bulb."""
    return add_boosts(s, target, S.SPECIAL_ATTACK, 1)


def cure_status(state, player):
    """Clear a major status, its sleep counter and its toxic counter together."""

    old = state.active.status[player, 0]
    state = emit(state, EventKind.STATUS, player, before=old, after=Status.NONE, enabled=old != Status.NONE)
    state = update_active_fields(state, player, status=jnp.array([Status.NONE]), sleep_counter=jnp.zeros(1, jnp.int32))
    return state.replace(toxic_counter=state.toxic_counter.at[player, 0].set(0))


def cheek_pouch(state, player, eaten=True):
    """Heal after a represented berry's own effect, without recursive HP hooks."""
    mon = state.active[player]
    eligible = eaten & (state.ability[player, 0] == A.CHEEK_POUCH) & (mon.current_hp[0] > 0)
    hp = jnp.minimum(mon.max_hp, mon.current_hp + jnp.where(eligible, jnp.maximum(1, mon.max_hp // 3), 0))
    state = emit(state, EventKind.HP, player, before=mon.current_hp, after=hp,
        hp_change=hp-mon.current_hp,
        hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
        enabled=hp[0] != mon.current_hp[0])
    return update_active_fields(state, player, current_hp=hp)


@jax.jit
def status_berry(state, player):
    """Eat Chesto for sleep or Lum for major status/confusion, respecting Unnerve."""

    mon = state.active[player]
    confused = state.volatile_status.confusion_turns[player, 0] > 0
    lum = mon.item[0] == I.LUM_BERRY
    eligible = ((lum & ((mon.status[0] != Status.NONE) | confused)) |
                ((mon.item[0] == I.CHESTO_BERRY) & (mon.status[0] == Status.SLEEP)))
    eligible &= (mon.current_hp[0] > 0) & berry_allowed(state, player)
    cure = partial(_status_berry_cure, player=player, lum=lum, confused=confused)
    return jax.lax.cond(eligible, cure, _identity, state)


@jax.jit
def white_herb(state, player):
    """At an update boundary, consume once to clear all negative stat stages."""

    normal, accuracy = state.boosts.normal_boosts[player], state.boosts.acc_boosts[player]
    eligible = ((state.active.item[player, 0] == I.WHITE_HERB) & (state.active.current_hp[player, 0] > 0) &
                (jnp.any(normal < 0) | jnp.any(accuracy < 0)))
    restore = partial(_white_herb_restore, player=player, normal=normal, accuracy=accuracy)
    return jax.lax.cond(eligible, restore, _identity, state)


@jax.jit
def update_items(state):
    """Apply represented item updates after a whole move or switch resolves."""
    update = _update_items_update
    return jax.lax.fori_loop(0, 2, update, state)


def kings_rock_flinch(key, state, player, move):
    """Roll an independent item secondary once per hit reaching the Pokémon."""

    target = 1-player
    ability = state.ability[target, 0]
    eligible = ((state.active.item[player, 0] == I.KINGS_ROCK) &
        (move.move_type[0] != MoveType.STATUS) & ~move.secondary_flinch[0] &
        # Sheer Force runs before the item adds its secondary. Its later
        # suppression applies only when the original move earned its boost.
        ((state.ability[player, 0] != A.SHEER_FORCE) | (move.secondary_chance[0] == 0)) &
        (ability != A.SHIELD_DUST) & (ability != A.INNER_FOCUS) &
        (state.active.current_hp[target, 0] > 0))
    roll = partial(_kings_rock_flinch_roll, player=player, target=target)
    return jax.lax.cond(eligible, roll, _identity, (key, state))


def consume(state, player, enabled=True):
    """Consume a held item and remember its publicly visible disappearance."""
    mon = state.active[player]
    used = jnp.asarray(enabled).reshape(()) & (mon.item[0] != I.NONE)
    state = emit(state, EventKind.ITEM, player, detail=mon.item[0], enabled=used)
    state = remember_item(state, player, I.NONE, used)
    return update_active_fields(state, player, item=jnp.where(used, I.NONE, mon.item))


def _assign_item(state, holder, item, source):
    """Commit a validated public item mutation and clean up Choice lock state."""
    mon = state.active[holder]
    old = mon.item[0]
    old_choice = is_choice_item(old)
    new_choice = is_choice_item(item)
    state = update_active_fields(state, holder, item=jnp.asarray(item, jnp.int32).reshape((1,)))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(choice_lock=v.choice_lock.at[holder, 0].set(
        jnp.where(old_choice & new_choice, v.choice_lock[holder, 0], -1))))
    state = remember_item(state, holder, item)
    previous = state.event_source
    state = state.replace(event_source=jnp.asarray(source, jnp.int32))
    state = emit(state, EventKind.ITEM, holder, detail=old, before=old, after=item)
    state = emit(state, EventKind.ITEM_REVEALED, holder, detail=item, enabled=item != I.NONE)
    return state.replace(event_source=previous)


def _activate_received_item(state, holder):
    """Run represented Sitrus and Mental Herb effects after an atomic transfer."""
    state = white_herb(status_berry(after_hp_change(state, holder), holder), holder)
    v = state.volatile_status
    herb = ((state.active.item[holder, 0] == I.MENTAL_HERB) & (state.active.current_hp[holder, 0] > 0) &
            ((v.disable_turns[holder, 0] > 0) | (v.encore_turns[holder, 0] > 0) | (v.taunt_turns[holder, 0] > 0)))
    state = consume(state, holder, herb)
    v = state.volatile_status
    for name in ('disabled_move_id', 'disable_turns', 'encore_move_id', 'encore_turns', 'taunt_turns'):
        values = getattr(v, name)
        v = v.replace(**{name: values.at[holder, 0].set(jnp.where(herb, 0, values[holder, 0]))})
    return state.replace(volatile_status=v)


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
    own, other = state.active.item[source, 0], state.active.item[target, 0]
    species, forms = state.active.species[:, 0], state.active.form[:, 0]
    sticky = state.ability[target, 0] == A.STICKY_HOLD
    success = (jnp.asarray(enabled).reshape(()) & (source != target) &
        (state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & ~sticky &
        ((own != I.NONE) | (other != I.NONE)) &
        item_transfer_allowed(species[source], forms[source], own) &
        item_transfer_allowed(species[target], forms[target], other) &
        item_transfer_allowed(species[source], forms[source], other) &
        item_transfer_allowed(species[target], forms[target], own))
    exchange = partial(_swap_items_exchange, source=source, other=other, target=target, own=own)
    return jax.lax.cond(success, exchange, _identity, state), success


def refresh_paradox(state):
    """Activate once per field effect, retaining the chosen stat until it ends."""
    ability = state.ability[:, 0]
    field = ((ability == A.PROTOSYNTHESIS) & (effective_weather(state.weather.weather, state.ability, state.active.current_hp) == W.SUN)) | (
             (ability == A.QUARK_DRIVE) & (state.terrain.terrain == T.ELECTRIC))
    eligible = (ability == A.PROTOSYNTHESIS) | (ability == A.QUARK_DRIVE)
    v = state.volatile_status
    for player in range(2):
        live = state.active.current_hp[player, 0] > 0
        item = state.active.item[player, 0]
        use = live & eligible[player] & ~field[player] & ~v.paradox_booster[player, 0] & (item == I.BOOSTER_ENERGY)
        state = consume(state, player, use)
        boosted = v.paradox_booster[player, 0] | use
        active = live & eligible[player] & (field[player] | boosted)
        # getBestStat uses stages, with Atk/Def/SpA/SpD/Spe tie order.
        stats = state.raw_boosted_stats[player, 1:]
        best = 1 + jnp.argmax(stats)
        old = v.paradox_stat[player, 0]
        stat = jnp.where(active, jnp.where((old >= 1) & ~use, old, best), -1)
        state = emit(state, EventKind.ABILITY, player, detail=ability[player], after=stat,
                     enabled=(stat != old))
        v = v.replace(paradox_stat=v.paradox_stat.at[player, 0].set(stat),
                      paradox_booster=v.paradox_booster.at[player, 0].set(boosted))
    return state.replace(volatile_status=v)


def after_hp_change(state, player):
    """Consume a reactive Sitrus Berry and update only the holder's HP leaf."""
    mon = state.active[player]
    berry = ((mon.item[0] == I.SITRUS_BERRY) & (mon.current_hp[0] > 0) &
             (mon.hp_percent <= .5) & berry_allowed(state, player))
    state = consume(state, player, berry)
    hp = jnp.minimum(mon.max_hp, mon.current_hp + jnp.where(berry, jnp.maximum(1, mon.max_hp // 4), 0))
    state = emit(state, EventKind.HP, player, before=mon.current_hp, after=hp,
        hp_change=hp-mon.current_hp,
        hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
        enabled=hp[0] != mon.current_hp[0])
    return cheek_pouch(update_active_fields(state, player, current_hp=hp), player, berry)


def after_hit(state, attacker, move, damage, effectiveness):
    """Activate represented held items after a damaging hit reaches its holder."""

    target = 1-attacker
    mon = state.active[target]
    policy = (mon.item[0] == I.WEAKNESS_POLICY) & (mon.current_hp[0] > 0) & (damage > 0) & (effectiveness > 1)
    state = consume(state, target, policy)
    state = jax.lax.cond(jnp.asarray(policy).reshape(()), partial(_activate_weakness_policy, target=target), _identity, state)
    bulb = ((mon.item[0] == I.ABSORB_BULB) & (mon.current_hp[0] > 0) &
            (damage > 0) & (move.type[0] == Type.WATER))
    state = consume(state, target, bulb)
    return jax.lax.cond(jnp.asarray(bulb).reshape(()),
        partial(_activate_absorb_bulb, target=target), _identity, state)
