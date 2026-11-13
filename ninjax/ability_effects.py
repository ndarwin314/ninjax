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


def underlying_ability(state, player):
    """Return the active ability identity even while its effects are suppressed."""
    override = state.volatile_status.ability_override[player, 0]
    return jnp.where(override >= 0, override, state.active.ability[player, 0])


def worry_seed(state, attacker):
    """Replace an eligible target ability with Insomnia and wake the target.

    Preserve the team ability, clear represented old-ability state, refresh
    field-dependent Paradox effects, and record the publicly revealed ability.
    """
    from ninjax.side import update_active_fields
    from ninjax.held_effects import refresh_paradox
    target = 1-attacker
    old = underlying_ability(state, target)
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
