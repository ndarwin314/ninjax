"""Held-item and field-ability hooks for immutable singles state.

Item identifiers are internal enums, not Showdown numeric identifiers.
"""
import jax
import jax.numpy as jnp
from ninjax.enum_types import ItemEnum as I, AbilityEnum as A, WeatherEnum as W, TerrainEnum as T, StatEnum as S
from ninjax.events import EventKind, emit

def conditional_mult_round(value, multiplier, enabled):
    """Showdown's 4096-based modifier, rounding exact halves down."""
    modifier = jnp.floor(jnp.asarray(multiplier) * 4096).astype(jnp.int32)
    rounded = jnp.floor((jnp.floor(value * modifier) + 2047) / 4096).astype(jnp.int32)
    return jnp.where(enabled, rounded, value)


def effective_weather(state):
    """Return weather after active effective ability suppression is applied."""
    suppressed = jnp.any(((state.ability == A.CLOUD_NINE) |
                          (state.ability == A.AIR_LOCK)) & (state.active.current_hp > 0))
    return jnp.where(suppressed, W.NONE, state.weather.weather)


def consume(state, player, enabled=True):
    """Consume a held item and remember its publicly visible disappearance."""
    from ninjax.side import update_active
    mon = state.active[player]
    used = jnp.asarray(enabled).reshape(()) & (mon.item[0] != I.NONE)
    state = emit(state, EventKind.ITEM, player, detail=mon.item[0], enabled=used)
    from ninjax.ability_effects import remember_item
    state = remember_item(state, player, I.NONE, used)
    return update_active(state, player, mon.replace(item=jnp.where(used, I.NONE, mon.item)))


def item_transfer_allowed(state, holder, item):
    """Check represented item identity and species binding for a given holder.

    Also use this predicate for the prospective recipient of a swap. Unknown
    IDs fail closed until their transfer restrictions are represented. This
    checks item rules only; it does not check Sticky Hold or item presence.
    """
    from ninjax.enum_types import FormEnum
    item = jnp.asarray(item).reshape(())
    known = jnp.any(item == jnp.array([int(value) for value in I], jnp.int32))
    mon = state.active[holder]
    ogerpon = (mon.species[0] == 1017) | jnp.any(mon.form[0] == jnp.array([
        FormEnum.OGERPON_WELLSPRING, FormEnum.OGERPON_HEARTHFLAME, FormEnum.OGERPON_CORNERSTONE]))
    mask = (item == I.WELLSPRING_MASK) | (item == I.HEARTHFLAME_MASK) | (item == I.CORNERSTONE_MASK)
    return known & ~(ogerpon & mask)


def can_remove_item(state, holder, source):
    """Check external removal, honoring effective Sticky Hold on a living holder.

    Self-removal bypasses Sticky Hold. Consumption uses its own helper and is
    not governed by these transfer rules. A fainted holder loses Sticky Hold's
    protection, which matters for Knock Off's after-hit removal.
    """
    item = state.active.item[holder, 0]
    sticky = (source != holder) & (state.active.current_hp[holder, 0] > 0) & (state.ability[holder, 0] == A.STICKY_HOLD)
    return (item != I.NONE) & item_transfer_allowed(state, holder, item) & ~sticky


def _assign_item(state, holder, item, source):
    """Commit a validated public item mutation and clean up Choice lock state."""
    from ninjax.side import update_active
    from ninjax.ability_effects import remember_item
    mon = state.active[holder]
    old = mon.item[0]
    old_choice = (old == I.CHOICE_BAND) | (old == I.CHOICE_SPECS)
    new_choice = (item == I.CHOICE_BAND) | (item == I.CHOICE_SPECS)
    state = update_active(state, holder, mon.replace(item=jnp.asarray(item, jnp.int32).reshape((1,))))
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
    state = after_hp_change(state, holder)
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
    state = jax.lax.cond(success, lambda s: _assign_item(s, holder, I.NONE, source), lambda s: s, state)
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
    sticky = state.ability[target, 0] == A.STICKY_HOLD
    success = (jnp.asarray(enabled).reshape(()) & (source != target) &
        (state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & ~sticky &
        ((own != I.NONE) | (other != I.NONE)) &
        item_transfer_allowed(state, source, own) & item_transfer_allowed(state, target, other) &
        item_transfer_allowed(state, source, other) & item_transfer_allowed(state, target, own))
    def exchange(s):
        """Publish both sides of the swap before reacting to received items."""
        previous_source = s.event_source
        s = s.replace(event_source=jnp.asarray(source, jnp.int32))
        s = _assign_item(s, source, other, source)
        s = _assign_item(s, target, own, source)
        s = _activate_received_item(s, source)
        s = _activate_received_item(s, target)
        return refresh_paradox(s).replace(event_source=previous_source)
    return jax.lax.cond(success, exchange, lambda s: s, state), success


def modify_stats(state, player, stats):
    """Chain represented stat modifiers at 4096 precision, then round once.

    For example, Choice Specs plus Hadron Engine combines into a 2x modifier;
    rounding the two stat changes separately loses a point on odd base stats.
    """
    from ninjax.enum_types import Type
    mon = state.active[player]
    item, ability = mon.item[0], state.ability[player, 0]
    indices = jnp.arange(6)
    weather = effective_weather(state)
    choice = ((indices == S.ATTACK) & (item == I.CHOICE_BAND)) | ((indices == S.SPECIAL_ATTACK) & (item == I.CHOICE_SPECS))
    vest = (indices == S.SPECIAL_DEFENSE) & (item == I.ASSAULT_VEST)
    field = ((indices == S.ATTACK) & (ability == A.ORICHALCUM_PULSE) & (weather == W.SUN)) | (
        (indices == S.SPECIAL_ATTACK) & (ability == A.HADRON_ENGINE) & (state.terrain.terrain == T.ELECTRIC))
    weather_defense = ((indices == S.DEFENSE) & (weather == W.SNOW) & jnp.any(mon.type_list == Type.ICE)) | (
        (indices == S.SPECIAL_DEFENSE) & (weather == W.SANDSTORM) & jnp.any(mon.type_list == Type.ROCK))
    paradox = indices == state.volatile_status.paradox_stat[player, 0]
    modifiers = jnp.full(6, 4096, jnp.int32)
    for factor in (jnp.where(choice | vest, 6144, 4096),
                   jnp.where(field, 5461, 4096),
                   jnp.where(weather_defense, 6144, 4096),
                   jnp.where(paradox, jnp.where(indices == S.SPEED, 6144, 5325), 4096)):
        modifiers = (modifiers * factor + 2048) // 4096
    return conditional_mult_round(stats, modifiers / 4096, True)


def refresh_paradox(state):
    """Activate once per field effect, retaining the chosen stat until it ends."""
    ability = state.ability[:, 0]
    field = ((ability == A.PROTOSYNTHESIS) & (effective_weather(state) == W.SUN)) | (
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
    from ninjax.side import update_active
    mon = state.active[player]
    berry = (mon.item[0] == I.SITRUS_BERRY) & (mon.current_hp[0] > 0) & (mon.hp_percent <= .5)
    state = consume(state, player, berry)
    hp = jnp.minimum(mon.max_hp, mon.current_hp + jnp.where(berry, jnp.maximum(1, mon.max_hp // 4), 0))
    state = emit(state, EventKind.HP, player, before=mon.current_hp, after=hp,
        hp_change=hp-mon.current_hp,
        hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
        enabled=hp[0] != mon.current_hp[0])
    return update_active(state, player, state.active[player].replace(current_hp=hp))


def after_hit(state, attacker, move, damage, effectiveness):
    from ninjax.side import add_boosts
    target = 1-attacker
    mon = state.active[target]
    policy = (mon.item[0] == I.WEAKNESS_POLICY) & (mon.current_hp[0] > 0) & (damage > 0) & (effectiveness > 1)
    state = consume(state, target, policy)
    return jax.lax.cond(jnp.asarray(policy).reshape(()), lambda s: add_boosts(s, target,
        jnp.array([S.ATTACK, S.SPECIAL_ATTACK]), jnp.array([2, 2])), lambda s: s, state)


def after_move(state, player, move, dealt_damage):
    from ninjax.side import take_damage_percent, add_boosts
    mon = state.active[player]
    alive = mon.current_hp[0] > 0
    state = take_damage_percent(state, player, jnp.where(alive & dealt_damage & (mon.item[0] == I.LIFE_ORB), .1, 0))
    spray = alive & move.sound[0] & (state.active.item[player, 0] == I.THROAT_SPRAY)
    state = consume(state, player, spray)
    return jax.lax.cond(spray, lambda s: add_boosts(s, player, S.SPECIAL_ATTACK, 1), lambda s: s, state)


def leftovers(state):
    from ninjax.side import take_damage_percent
    def heal(i, s):
        mon = s.active[i]
        return take_damage_percent(s, i, jnp.where((mon.item[0] == I.LEFTOVERS) & (mon.current_hp[0] > 0), -1/16, 0))
    return jax.lax.fori_loop(0, 2, heal, state)
