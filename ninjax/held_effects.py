"""Held-item and field-ability hooks for immutable singles state.

Item identifiers are internal enums, not Showdown numeric identifiers.
"""
import jax
import jax.numpy as jnp
from ninjax.enum_types import ItemEnum as I, AbilityEnum as A, WeatherEnum as W, TerrainEnum as T, StatEnum as S
from ninjax.events import EventKind, emit


def is_choice_item(item):
    """Identify all three Choice items for locking and item-transfer cleanup."""
    return (item == I.CHOICE_BAND) | (item == I.CHOICE_SPECS) | (item == I.CHOICE_SCARF)


def berry_allowed(state, player):
    """Living opposing Unnerve/As One users prevent eating held berries."""
    other = 1-player
    ability = state.ability[other, 0]
    unnerve = (ability == A.UNNERVE) | (ability == A.AS_ONE_CHILLING) | (ability == A.AS_ONE_GRIM)
    return ~unnerve | (state.active.current_hp[other, 0] <= 0)


def cure_status(state, player):
    """Clear a major status, its sleep counter and its toxic counter together."""
    from ninjax.side import update_active_fields
    from ninjax.enum_types import Status
    old = state.active.status[player, 0]
    state = emit(state, EventKind.STATUS, player, before=old, after=Status.NONE, enabled=old != Status.NONE)
    state = update_active_fields(state, player, status=jnp.array([Status.NONE]), sleep_counter=jnp.zeros(1, jnp.int32))
    return state.replace(toxic_counter=state.toxic_counter.at[player, 0].set(0))


def cheek_pouch(state, player, eaten=True):
    """Heal after a represented berry's own effect, without recursive HP hooks."""
    from ninjax.side import update_active_fields
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
    from ninjax.enum_types import Status
    mon = state.active[player]
    confused = state.volatile_status.confusion_turns[player, 0] > 0
    lum = mon.item[0] == I.LUM_BERRY
    eligible = ((lum & ((mon.status[0] != Status.NONE) | confused)) |
                ((mon.item[0] == I.CHESTO_BERRY) & (mon.status[0] == Status.SLEEP)))
    eligible &= (mon.current_hp[0] > 0) & berry_allowed(state, player)
    def cure(s):
        """Consume before clearing conditions so item knowledge remains accurate."""
        s = cure_status(consume(s, player), player)
        v = s.volatile_status
        s = s.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[player, 0].set(
            jnp.where(lum, 0, v.confusion_turns[player, 0]))))
        s = emit(s, EventKind.CONDITION, player, detail=12, before=1, after=0, enabled=lum & confused)
        return cheek_pouch(s, player)
    return jax.lax.cond(eligible, cure, lambda s: s, state)


@jax.jit
def white_herb(state, player):
    """At an update boundary, consume once to clear all negative stat stages."""
    from ninjax.side import set_boosts
    from ninjax.stats import StatBoosts
    normal, accuracy = state.boosts.normal_boosts[player], state.boosts.acc_boosts[player]
    eligible = ((state.active.item[player, 0] == I.WHITE_HERB) & (state.active.current_hp[player, 0] > 0) &
                (jnp.any(normal < 0) | jnp.any(accuracy < 0)))
    def restore(s):
        """Keep positive stages, including Shell Smash's offensive increases."""
        return set_boosts(consume(s, player), player, StatBoosts(
            normal_boosts=jnp.maximum(normal, 0), acc_boosts=jnp.maximum(accuracy, 0)))
    return jax.lax.cond(eligible, restore, lambda s: s, state)


@jax.jit
def update_items(state):
    """Apply represented item updates after a whole move or switch resolves."""
    def update(player, s):
        """Check status berries before the post-action White Herb check."""
        return white_herb(status_berry(after_hp_change(s, player), player), player)
    return jax.lax.fori_loop(0, 2, update, state)


def residual_orbs(key, state):
    """Inflict orb statuses after this turn's status damage; orbs are not consumed."""
    from ninjax.side import set_status
    from ninjax.enum_types import Status
    def apply(player, carry):
        """Use existing status immunity and explicit-key helpers for each holder."""
        key, s = carry
        item = s.active.item[player, 0]
        status = jnp.where(item == I.FLAME_ORB, Status.BURN, jnp.where(item == I.TOXIC_ORB, Status.TOXIC, Status.NONE))
        return jax.lax.cond((status != Status.NONE) & (s.active.current_hp[player, 0] > 0),
            lambda k, s: set_status(k, s, player, jnp.asarray(status).reshape((1,))),
            lambda k, s: (k, s), key, s)
    return jax.lax.fori_loop(0, 2, apply, (key, state))


def contact_reactions_allowed(ability, item):
    """Check the user's defenses against contact reactions, not move contact.

    Inputs are the attacker's scalar effective ability and held item.
    Callers also check the move's contact flag. Pads protect against the
    defender's reactions without removing contact bonuses or Poison Touch.
    """
    return (ability != A.LONG_REACH) & (item != I.PROTECTIVE_PADS)


def kings_rock_flinch(key, state, player, move):
    """Roll an independent item secondary once per hit reaching the Pokémon."""
    from ninjax.enum_types import MoveType
    target = 1-player
    ability = state.ability[target, 0]
    eligible = ((state.active.item[player, 0] == I.KINGS_ROCK) &
        (move.move_type[0] != MoveType.STATUS) & ~move.secondary_flinch[0] &
        # Sheer Force runs before the item adds its secondary. Its later
        # suppression applies only when the original move earned its boost.
        ((state.ability[player, 0] != A.SHEER_FORCE) | (move.secondary_chance[0] == 0)) &
        (ability != A.SHIELD_DUST) & (ability != A.INNER_FOCUS) &
        (state.active.current_hp[target, 0] > 0))
    def roll(args):
        """Serene Grace also doubles the item-added secondary chance."""
        key, s = args
        key, roll_key = jax.random.split(key)
        chance = jnp.where(s.ability[player, 0] == A.SERENE_GRACE, .2, .1)
        v = s.volatile_status
        flinch = v.flinched[target, 0] | (jax.random.uniform(roll_key) < chance)
        return key, s.replace(volatile_status=v.replace(flinched=v.flinched.at[target, 0].set(flinch)))
    return jax.lax.cond(eligible, roll, lambda args: args, (key, state))

def conditional_mult_round(value, multiplier, enabled):
    """Showdown's 4096-based modifier, rounding exact halves down."""
    modifier = jnp.floor(jnp.asarray(multiplier) * 4096).astype(jnp.int32)
    rounded = jnp.floor((jnp.floor(value * modifier) + 2047) / 4096).astype(jnp.int32)
    return jnp.where(enabled, rounded, value)


def effective_weather(weather, abilities, current_hp):
    """Resolve scalar weather using active abilities/HP of shape (2, 1).

    Callers supply effective abilities, including overrides and suppression.
    Fainted Cloud Nine/Air Lock users do not suppress field weather.
    """
    suppressed = jnp.any(((abilities == A.CLOUD_NINE) |
                          (abilities == A.AIR_LOCK)) & (current_hp > 0))
    return jnp.where(suppressed, W.NONE, weather)


def consume(state, player, enabled=True):
    """Consume a held item and remember its publicly visible disappearance."""
    from ninjax.side import update_active_fields
    mon = state.active[player]
    used = jnp.asarray(enabled).reshape(()) & (mon.item[0] != I.NONE)
    state = emit(state, EventKind.ITEM, player, detail=mon.item[0], enabled=used)
    from ninjax.ability_effects import remember_item
    state = remember_item(state, player, I.NONE, used)
    return update_active_fields(state, player, item=jnp.where(used, I.NONE, mon.item))


def item_transfer_allowed(species, form, item):
    """Check represented item identity and species binding for a given holder.

    Inputs are scalar species, form and item IDs. Also use this predicate
    for the prospective recipient of a swap. Unknown IDs fail closed until
    their transfer restrictions are represented. This
    checks item rules only; it does not check Sticky Hold or item presence.
    """
    from ninjax.enum_types import FormEnum
    item = jnp.asarray(item).reshape(())
    known = jnp.any(item == jnp.array([int(value) for value in I], jnp.int32))
    ogerpon = (species == 1017) | jnp.any(form == jnp.array([
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
    return (item != I.NONE) & item_transfer_allowed(state.active.species[holder, 0], state.active.form[holder, 0], item) & ~sticky


def _assign_item(state, holder, item, source):
    """Commit a validated public item mutation and clean up Choice lock state."""
    from ninjax.side import update_active_fields
    from ninjax.ability_effects import remember_item
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
    species, forms = state.active.species[:, 0], state.active.form[:, 0]
    sticky = state.ability[target, 0] == A.STICKY_HOLD
    success = (jnp.asarray(enabled).reshape(()) & (source != target) &
        (state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & ~sticky &
        ((own != I.NONE) | (other != I.NONE)) &
        item_transfer_allowed(species[source], forms[source], own) &
        item_transfer_allowed(species[target], forms[target], other) &
        item_transfer_allowed(species[source], forms[source], other) &
        item_transfer_allowed(species[target], forms[target], own))
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
    from ninjax.enum_types import Type, Status
    mon = state.active[player]
    item, ability = mon.item[0], state.ability[player, 0]
    indices = jnp.arange(6)
    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    choice = (((indices == S.ATTACK) & (item == I.CHOICE_BAND)) |
              ((indices == S.SPECIAL_ATTACK) & (item == I.CHOICE_SPECS)) |
              ((indices == S.SPEED) & (item == I.CHOICE_SCARF)))
    light_ball = (item == I.LIGHT_BALL) & (mon.species[0] == 25) & ((indices == S.ATTACK) | (indices == S.SPECIAL_ATTACK))
    vest = (indices == S.SPECIAL_DEFENSE) & (item == I.ASSAULT_VEST)
    field = ((indices == S.ATTACK) & (ability == A.ORICHALCUM_PULSE) & (weather == W.SUN)) | (
        (indices == S.SPECIAL_ATTACK) & (ability == A.HADRON_ENGINE) & (state.terrain.terrain == T.ELECTRIC))
    weather_defense = ((indices == S.DEFENSE) & (weather == W.SNOW) & jnp.any(mon.type_list == Type.ICE)) | (
        (indices == S.SPECIAL_DEFENSE) & (weather == W.SANDSTORM) & jnp.any(mon.type_list == Type.ROCK))
    paradox = indices == state.volatile_status.paradox_stat[player, 0]
    speed_ability = (((ability == A.CHLOROPHYLL) & (weather == W.SUN)) |
                     ((ability == A.SWIFT_SWIM) & (weather == W.RAIN)) |
                     ((ability == A.SLUSH_RUSH) & (weather == W.SNOW)) |
                     ((ability == A.SAND_RUSH) & (weather == W.SANDSTORM)) |
                     ((ability == A.SURGE_SURFER) & (state.terrain.terrain == T.ELECTRIC)))
    quick_feet = (ability == A.QUICK_FEET) & (mon.status[0] != Status.NONE)
    modifiers = jnp.full(6, 4096, jnp.int32)
    for factor in (jnp.where(choice | vest, 6144, 4096),
                   jnp.where(light_ball, 8192, 4096),
                   jnp.where(field, 5461, 4096),
                   jnp.where(weather_defense, 6144, 4096),
                   jnp.where((indices == S.SPEED) & speed_ability, 8192, 4096),
                   jnp.where((indices == S.SPEED) & quick_feet, 6144, 4096),
                   jnp.where((indices == S.SPEED) & (state.tailwind[player, 0] > 0), 8192, 4096),
                   jnp.where(paradox, jnp.where(indices == S.SPEED, 6144, 5325), 4096)):
        modifiers = (modifiers * factor + 2048) // 4096
    result = conditional_mult_round(stats, modifiers / 4096, True)
    # Paralysis halves the final Speed, unless Quick Feet replaces its effect.
    return jnp.where((indices == S.SPEED) & (mon.status[0] == Status.PARALYZE) & (ability != A.QUICK_FEET), result // 2, result)


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
    from ninjax.side import update_active_fields
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
    from ninjax.side import add_boosts
    from ninjax.enum_types import Type
    target = 1-attacker
    mon = state.active[target]
    policy = (mon.item[0] == I.WEAKNESS_POLICY) & (mon.current_hp[0] > 0) & (damage > 0) & (effectiveness > 1)
    state = consume(state, target, policy)
    state = jax.lax.cond(jnp.asarray(policy).reshape(()), lambda s: add_boosts(s, target,
        jnp.array([S.ATTACK, S.SPECIAL_ATTACK]), jnp.array([2, 2])), lambda s: s, state)
    bulb = ((mon.item[0] == I.ABSORB_BULB) & (mon.current_hp[0] > 0) &
            (damage > 0) & (move.type[0] == Type.WATER))
    state = consume(state, target, bulb)
    return jax.lax.cond(jnp.asarray(bulb).reshape(()),
        lambda s: add_boosts(s, target, S.SPECIAL_ATTACK, 1), lambda s: s, state)


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
