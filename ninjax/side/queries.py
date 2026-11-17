"""Read-only field, immunity, item and stat queries; no state writes."""
import jax.numpy as jnp

from ninjax.calculations.logic import triple_or
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum as A, AbilityEnum
from ninjax.enums.field import TerrainEnum as T, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I, ItemEnum
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import FormEnum, StatEnum as S, Status, Type


BURN_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.WATER_VEIL, AbilityEnum.WATER_BUBBLE, AbilityEnum.THERMAL_EXCHANGE)
PARALYSIS_IMMUNE_ABILITIES = (AbilityEnum.LIMBER, AbilityEnum.PURIFYING_SALT)
POISON_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.IMMUNITY)
FREEZE_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.MAGMA_ARMOR)
SLEEP_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.INSOMNIA, AbilityEnum.VITAL_SPIRIT)


def is_floating(mon):
    """Check Flying type, effective Levitate and Air Balloon before grounding."""

    return mon.is_type(Type.FLYING) | (mon.ability == AbilityEnum.LEVITATE) | (mon.item == ItemEnum.AIR_BALLOON)


def is_sand_immune(mon):
    """Check represented type and effective ability immunities to sandstorm."""
    return (triple_or(mon.is_type(Type.STEEL), mon.is_type(Type.GROUND), mon.is_type(Type.ROCK)) |
            (mon.ability == AbilityEnum.SAND_FORCE) | (mon.ability == AbilityEnum.SAND_RUSH) |
            (mon.ability == AbilityEnum.OVERCOAT))


def is_powder_immune(mon):
    """Check represented Grass and Overcoat immunities to powder effects."""
    return mon.is_type(Type.GRASS) | (mon.ability == AbilityEnum.OVERCOAT)


def is_immune_to_move(mon, move, gravity_active=False):
    """Check represented move immunities using a resolved active snapshot."""
    types = jnp.where(jnp.asarray(gravity_active) & (move.type == Type.GROUND) &
                      (mon.type_list == Type.FLYING), Type.NONE, mon.type_list)
    checks_type = (move.move_type != MoveType.STATUS) | (move.identifier == 86)
    type_immunity = (calculate_effectiveness_multiplier(move.type, types) == 0) & checks_type

    levitate = ((mon.ability == AbilityEnum.LEVITATE) | (mon.item == ItemEnum.AIR_BALLOON)) & (move.type == Type.GROUND) & ~jnp.asarray(gravity_active) & checks_type
    return jnp.any(jnp.stack(jnp.broadcast_arrays(
        is_powder_immune(mon) & move.powder,
        (mon.ability == AbilityEnum.SOUNDPROOF) & move.sound,
        (mon.ability == AbilityEnum.BULLET_PROOF) & move.bullet,
        (mon.ability == AbilityEnum.GOOD_AS_GOLD) & (move.move_type == MoveType.STATUS) & (move.hazard_effect == 0) & ~move.target_self,
        type_immunity, levitate)))


def is_choice_item(item):
    """Identify all three Choice items for locking and item-transfer cleanup."""
    return (item == I.CHOICE_BAND) | (item == I.CHOICE_SPECS) | (item == I.CHOICE_SCARF)


def berry_allowed(state, player):
    """Living opposing Unnerve/As One users prevent eating held berries."""
    other = 1-player
    ability = state.ability[other, 0]
    unnerve = (ability == A.UNNERVE) | (ability == A.AS_ONE_CHILLING) | (ability == A.AS_ONE_GRIM)
    return ~unnerve | (state.active.current_hp[other, 0] <= 0)


def contact_reactions_allowed(ability, item):
    """Check the user's defenses against contact reactions, not move contact.

    Inputs are the attacker's scalar effective ability and held item.
    Callers also check the move's contact flag. Pads protect against the
    defender's reactions without removing contact bonuses or Poison Touch.
    """
    return (ability != A.LONG_REACH) & (item != I.PROTECTIVE_PADS)


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


def item_transfer_allowed(species, form, item):
    """Check represented item identity and species binding for a given holder.

    Inputs are scalar species, form and item IDs. Also use this predicate
    for the prospective recipient of a swap. Unknown IDs fail closed until
    their transfer restrictions are represented. This
    checks item rules only; it does not check Sticky Hold or item presence.
    """

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


def modify_stats(state, player, stats):
    """Chain represented stat modifiers at 4096 precision, then round once.

    For example, Choice Specs plus Hadron Engine combines into a 2x modifier;
    rounding the two stat changes separately loses a point on odd base stats.
    """

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
