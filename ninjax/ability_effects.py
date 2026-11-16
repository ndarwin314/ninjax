"""Public item knowledge and switch-in ability effects for singles battles."""
import jax
import jax.numpy as jnp
from ninjax.enum_types import AbilityEnum, ItemEnum, Status
from ninjax.events import EventKind, emit

# Scarlet/Violet Showdown cantsuppress flags; these abilities cannot be
# replaced by Worry Seed. Keep IDs explicit rather than assuming enum order.
UNCHANGEABLE_ABILITIES = jnp.array([AbilityEnum.AS_ONE_CHILLING, AbilityEnum.AS_ONE_GRIM,
    AbilityEnum.BATTLE_BOND, AbilityEnum.COMATOSE, AbilityEnum.DISGUISE, AbilityEnum.GULP_MISSILE,
    AbilityEnum.ICE_FACE, AbilityEnum.MULTITYPE, AbilityEnum.POWER_CONSTRUCT, AbilityEnum.RKS_SYSTEM,
    AbilityEnum.SCHOOLING, AbilityEnum.SHIELDS_DOWN, AbilityEnum.STANCE_CHANGE, AbilityEnum.TERA_SHIFT,
    AbilityEnum.ZEN_MODE, AbilityEnum.ZERO_TO_HERO], jnp.int32)


def underlying_ability(base_ability, override):
    """Resolve scalar ability identity, ignoring effect suppression.

    A negative override means the active Pokémon's stored ability is used.
    """
    return jnp.where(override >= 0, override, base_ability)


def opposing_drop_response(state, player, lowered_count):
    """React to actual opposing stage reductions, including accuracy/evasion."""
    from ninjax.side import add_boosts
    from ninjax.enum_types import StatEnum
    ability = state.ability[player, 0]
    stat = jnp.where(ability == AbilityEnum.DEFIANT, StatEnum.ATTACK, StatEnum.SPECIAL_ATTACK)
    eligible = ((ability == AbilityEnum.DEFIANT) | (ability == AbilityEnum.COMPETITIVE)) & (lowered_count > 0)
    return jax.lax.cond(eligible, lambda s: add_boosts(s, player, stat, 2*lowered_count), lambda s: s, state)


def bad_dreams(state):
    """Damage sleeping opposing singles Pokémon at residual order eight."""
    from ninjax.side import take_damage_percent
    from ninjax.events import Cause
    def damage(player, s):
        """Skip fainted sources, and attribute indirect damage to the opponent."""
        other = 1-player
        sleeping = (s.active.status[player, 0] == Status.SLEEP) | (s.ability[player, 0] == AbilityEnum.COMATOSE)
        eligible = sleeping & (s.ability[other, 0] == AbilityEnum.BAD_DREAMS) & (s.active.current_hp[other, 0] > 0)
        eligible &= s.ability[player, 0] != AbilityEnum.MAGIC_GUARD
        source = s.event_source
        s = s.replace(event_source=jnp.asarray(other, jnp.int32))
        s = take_damage_percent(s, player, jnp.where(eligible, 1/8, 0), cause=Cause.INDIRECT)
        return s.replace(event_source=source)
    return jax.lax.fori_loop(0, 2, damage, state)


def residual_abilities(key, state, status_cures=True, speed_boost=True):
    """Resolve represented status cures and Speed Boost with explicit randomness.

    The scheduler runs status cures before poison/burn damage, then Speed Boost
    late in the turn. It does not activate on a normal switch-in's first turn.
    """
    from ninjax.held_effects import effective_weather, cure_status
    from ninjax.enum_types import WeatherEnum, StatEnum
    from ninjax.side import add_boosts
    weather = effective_weather(state.weather.weather, state.ability, state.active.current_hp)
    def apply(player, carry):
        """Only roll Shed Skin for a living, statused holder."""
        key, s = carry
        mon = s.active[player]
        ability = s.ability[player, 0]
        living = mon.current_hp[0] > 0
        def roll(k):
            """Use the pinned Scarlet/Violet 33-in-100 Shed Skin probability."""
            k, subkey = jax.random.split(k)
            return k, jax.random.randint(subkey, (), 0, 100) < 33
        key, shed = jax.lax.cond(jnp.asarray(status_cures) & living & (mon.status[0] != Status.NONE) & (ability == AbilityEnum.SHED_SKIN),
            roll, lambda k: (k, jnp.bool_(False)), key)
        cure = jnp.asarray(status_cures) & living & (mon.status[0] != Status.NONE) & (shed | ((ability == AbilityEnum.HYDRATION) & (weather == WeatherEnum.RAIN)))
        s = jax.lax.cond(cure, lambda s: cure_status(s, player), lambda s: s, s)
        speed = jnp.asarray(speed_boost) & living & (ability == AbilityEnum.SPEED_BOOST) & (s.volatile_status.entry_turn[player, 0] < s.turn_number)
        return key, jax.lax.cond(speed, lambda s: add_boosts(s, player, StatEnum.SPEED, 1), lambda s: s, s)
    return jax.lax.fori_loop(0, 2, apply, (key, state))


def worry_seed(state, attacker):
    """Replace an eligible target ability with Insomnia and wake the target.

    Preserve the team ability, clear represented old-ability state, refresh
    field-dependent Paradox effects, and record the publicly revealed ability.
    """
    from ninjax.side import update_active_fields
    from ninjax.held_effects import refresh_paradox
    target = 1-attacker
    old = underlying_ability(state.active.ability[target, 0], state.volatile_status.ability_override[target, 0])
    eligible = ((old != AbilityEnum.INSOMNIA) & (old != AbilityEnum.TRUANT) &
        ~jnp.any(UNCHANGEABLE_ABILITIES == old))
    shield = eligible & (state.active.item[target, 0] == ItemEnum.ABILITY_SHIELD)
    state = remember_item(state, target, ItemEnum.ABILITY_SHIELD, shield)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=ItemEnum.ABILITY_SHIELD, enabled=shield)
    allowed = eligible & ~shield
    def change(s):
        """End represented old-ability state before applying the new identity."""
        v = s.volatile_status
        paradox = (old == AbilityEnum.PROTOSYNTHESIS) | (old == AbilityEnum.QUARK_DRIVE)
        v = v.replace(ability_override=v.ability_override.at[target, 0].set(AbilityEnum.INSOMNIA),
            paradox_stat=v.paradox_stat.at[target, 0].set(jnp.where(paradox, -1, v.paradox_stat[target, 0])),
            paradox_booster=v.paradox_booster.at[target, 0].set(jnp.where(paradox, False, v.paradox_booster[target, 0])),
            flash_fire=v.flash_fire.at[target, 0].set(jnp.where(old == AbilityEnum.FLASH_FIRE, False, v.flash_fire[target, 0])),
            libero_used=v.libero_used.at[target, 0].set(jnp.where(old == AbilityEnum.PROTEAN, False, v.libero_used[target, 0])))
        s = s.replace(volatile_status=v)
        slot = s.active_index[target, 0]
        s = s.replace(known_abilities=s.known_abilities.at[target, slot].set(AbilityEnum.INSOMNIA))
        # The old ability may never have been public in hidden-team formats.
        s = emit(s, EventKind.ABILITY, target, detail=AbilityEnum.INSOMNIA, after=AbilityEnum.INSOMNIA)
        mon = s.active[target]
        asleep = mon.status == Status.SLEEP
        s = update_active_fields(s, target, status=jnp.where(asleep, Status.NONE, mon.status),
                                 sleep_counter=jnp.where(asleep, 0, mon.sleep_counter))
        return refresh_paradox(s)
    return jax.lax.cond(allowed, change, lambda s: s, state)


def remember_item(state, owner, item, enabled=True):
    """Retain a publicly revealed item for the owner's active team slot.

    `known_items` has shape (2 owners, 6 slots); -1 means unknown. Store zero
    after a publicly observed consumption so a previous Frisk reveal stays
    accurate without exposing unrelated private fields.
    """
    slot = state.active_index[owner, 0]
    known = state.known_items
    value = jnp.where(jnp.asarray(enabled).reshape(()), jnp.asarray(item).reshape(()), known[owner, slot])
    return state.replace(known_items=known.at[owner, slot].set(value))


def frisk(state, source):
    """Reveal a living opponent's held item; an empty item slot reveals nothing."""
    target = 1-source
    item = state.active.item[target, 0]
    enabled = (state.ability[source, 0] == AbilityEnum.FRISK) & (
        state.active.current_hp[source, 0] > 0) & (state.active.current_hp[target, 0] > 0) & (item != ItemEnum.NONE)
    state = remember_item(state, target, item, enabled)
    previous_source = state.event_source
    state = state.replace(event_source=jnp.asarray(source, jnp.int32))
    state = emit(state, EventKind.ABILITY, source, detail=AbilityEnum.FRISK, enabled=enabled)
    state = emit(state, EventKind.ITEM_REVEALED, target, detail=item, enabled=enabled)
    return state.replace(event_source=previous_source)


def prepare_type_change(state, player, move):
    """Apply Gen 9 Protean/Libero once per switch, only on an actual type change."""
    from ninjax.side import update_active_fields
    from ninjax.enum_types import Type
    mon = state.active[player]
    used = state.volatile_status.libero_used[player, 0]
    # A monotype matching the move does not consume the activation; a dual
    # type containing it still changes to a monotype and does consume it.
    matching = (mon.type_list[0] == move.type[0]) & (mon.type_list[1] == Type.NONE)
    trigger = (state.ability[player, 0] == AbilityEnum.PROTEAN) & ~used & ~matching & (
        move.type[0] != Type.NONE) & ~mon.is_terastallized[0] & move.implemented[0]
    types = jnp.where(trigger, jnp.array([move.type[0], Type.NONE]), mon.type_list)
    state = update_active_fields(state, player, type_list=types)
    state = emit(state, EventKind.ABILITY, player, detail=state.ability[player, 0], enabled=trigger)
    return state.replace(volatile_status=state.volatile_status.replace(
        libero_used=state.volatile_status.libero_used.at[player, 0].set(used | trigger)))
