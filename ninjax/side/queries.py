"""Read-only field, immunity, item and stat queries; no state writes."""
import jax.numpy as jnp

from ninjax.calculations.modifiers import apply_modifier, chain_modifier
from ninjax.calculations.stat_formulas import STAT_MULTIPLIER_LOOKUP

from ninjax.calculations.logic import triple_or
from ninjax.calculations.type_effectiveness import calculate_effectiveness_multiplier
from ninjax.enums.abilities import AbilityEnum as A, AbilityEnum
from ninjax.enums.field import TerrainEnum as T, WeatherEnum as W
from ninjax.enums.items import ItemEnum as I, ItemEnum
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import FormEnum, StatEnum as S, Status, Type
from ninjax.item_rules import ability_shield_active, item_is_suppressed, plate_type


BURN_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.WATER_VEIL, AbilityEnum.WATER_BUBBLE, AbilityEnum.THERMAL_EXCHANGE)
PARALYSIS_IMMUNE_ABILITIES = (AbilityEnum.LIMBER, AbilityEnum.PURIFYING_SALT)
POISON_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.IMMUNITY)
FREEZE_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.MAGMA_ARMOR)
SLEEP_IMMUNE_ABILITIES = (AbilityEnum.PURIFYING_SALT, AbilityEnum.INSOMNIA, AbilityEnum.VITAL_SPIRIT)


# Breakable abilities read by the represented hit, damage and reaction paths.
# Ability IDs are shared by aliases such as Battle Armor and Shell Armor.
BREAKABLE_HIT_ABILITIES = (
    A.AURA_BREAK, A.BATTLE_ARMOR, A.BULLET_PROOF, A.DAMP, A.DRY_SKIN,
    A.DISGUISE, A.ICE_FACE, A.SIMPLE, A.EARTH_EATER, A.FILTER, A.FLASH_FIRE, A.FLUFFY, A.FUR_COAT,
    A.GOOD_AS_GOLD, A.GRASS_PELT, A.HEATPROOF, A.HEAVY_METAL, A.ICE_SCALES,
    A.IMMUNITY, A.INNER_FOCUS, A.INSOMNIA, A.LEAF_GUARD,
    A.LEVITATE, A.LIGHT_METAL, A.LIGHTNING_ROD, A.LIMBER, A.MAGMA_ARMOR, A.MARVEL_SCALE, A.MOTOR_DRIVE,
    A.MULTISCALE, A.OVERCOAT, A.PUNK_ROCK, A.PURIFYING_SALT, A.QUEENLY_MAJESTY,
    A.SAND_VEIL, A.SAP_SIPPER, A.SHIELD_DUST, A.SNOW_CLOAK, A.SOUNDPROOF,
    A.STORM_DRAIN, A.STURDY, A.TANGLED_FEET, A.THERMAL_EXCHANGE, A.TERA_SHELL, A.THICK_FAT,
    A.UNAWARE, A.VITAL_SPIRIT, A.VOLT_ABSORB, A.WATER_ABSORB, A.WATER_BUBBLE,
    A.WATER_VEIL, A.WELL_BAKED_BODY,
    A.WIND_RIDER,
)


def ability_against_move(state, defender, move):
    """Read the defender's ability for this hit, respecting active Ability Shield."""
    ability = state.ability[defender]
    item = state.active_field('item', defender)
    shield = ability_shield_active(item, state.magic_room_duration, state.volatile_status.embargo_turns[defender])
    breakable = jnp.any(ability == jnp.array(BREAKABLE_HIT_ABILITIES, dtype=jnp.int32))
    return jnp.where(move.ignore_ability & breakable & ~shield, A.NONE, ability)


def is_floating(mon):
    """Check Flying type, effective Levitate and Air Balloon before grounding."""

    return is_floating_fields(mon.type_list, mon.ability, mon.item)


def is_floating_fields(types, ability, item):
    """Check airborne type, effective ability and held item."""
    flying = jnp.any(types == Type.FLYING, axis=-1, keepdims=True)
    return flying | (ability == AbilityEnum.LEVITATE) | (item == ItemEnum.AIR_BALLOON)


def is_sand_immune(mon):
    """Check represented type and effective ability immunities to sandstorm."""
    return (triple_or(mon.is_type(Type.STEEL), mon.is_type(Type.GROUND), mon.is_type(Type.ROCK)) |
            (mon.ability == AbilityEnum.SAND_FORCE) | (mon.ability == AbilityEnum.SAND_RUSH) |
            (mon.ability == AbilityEnum.OVERCOAT))


def is_powder_immune(mon):
    """Check represented Grass and Overcoat immunities to powder effects."""
    return is_powder_immune_fields(mon.type_list, mon.ability)


def is_powder_immune_fields(types, ability):
    """Check powder immunity from effective types and ability."""
    return jnp.any(types == Type.GRASS, axis=-1, keepdims=True) | (ability == AbilityEnum.OVERCOAT)


def is_immune_to_move(mon, move, gravity_active=False):
    """Check represented move immunities using a resolved active snapshot."""
    return move_immune_from_fields(mon.type_list, mon.ability, mon.item, move, gravity_active)


def move_immune_from_fields(defender_types, ability, item, move, gravity_active):
    """Check move immunities without gathering unrelated Pokémon fields."""
    types = jnp.where(jnp.asarray(gravity_active) & (move.type == Type.GROUND) &
                      (defender_types == Type.FLYING), Type.NONE, defender_types)
    checks_type = (move.move_type != MoveType.STATUS) | (move.identifier == 86)
    type_immunity = (calculate_effectiveness_multiplier(move.type, types, move.effectiveness_rule) == 0) & checks_type

    levitate = ((ability == AbilityEnum.LEVITATE) | (item == ItemEnum.AIR_BALLOON)) & (move.type == Type.GROUND) & ~jnp.asarray(gravity_active) & checks_type
    return jnp.any(jnp.stack(jnp.broadcast_arrays(
        is_powder_immune_fields(defender_types, ability) & move.powder,
        (ability == AbilityEnum.SOUNDPROOF) & move.sound,
        (ability == AbilityEnum.BULLET_PROOF) & move.bullet,
        (move.identifier == 73) & jnp.any(defender_types == Type.GRASS),
        (ability == AbilityEnum.GOOD_AS_GOLD) & (move.move_type == MoveType.STATUS) & (move.hazard_effect == 0) & ~move.target_self,
        type_immunity, levitate)))


def is_choice_item(item):
    """Identify all three Choice items for locking and item-transfer cleanup."""
    return (item == I.CHOICE_BAND) | (item == I.CHOICE_SPECS) | (item == I.CHOICE_SCARF)


def berry_allowed(ability, opposing_hp):
    """Living opposing Unnerve/As One users prevent eating held berries."""
    unnerve = (ability == A.UNNERVE) | (ability == A.AS_ONE_CHILLING) | (ability == A.AS_ONE_GRIM)
    return ~unnerve | (opposing_hp <= 0)


def contact_reactions_allowed(ability, item):
    """Check the user's defenses against contact reactions, not move contact.

    Inputs are the attacker's scalar effective ability and held item.
    Callers also check the move's contact flag. Pads protect against the
    defender's reactions without removing contact bonuses or Poison Touch.
    """
    return (ability != A.LONG_REACH) & (item != I.PROTECTIVE_PADS)


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
    # Plates cannot move from or onto an Arceus, including inactive forms.
    arceus_plate = (species == 493) & (plate_type(item) != Type.NONE)
    bound_origin = ((species == 487) & (item == I.GRISEOUS_CORE)) | (
        (species == 483) & (item == I.ADAMANT_CRYSTAL)) | ((species == 484) & (item == I.LUSTROUS_GLOBE))
    bound_regalia = ((species == 888) & (item == I.RUSTED_SWORD)) | ((species == 889) & (item == I.RUSTED_SHIELD))
    return known & ~(ogerpon & mask) & ~arceus_plate & ~bound_origin & ~bound_regalia


def can_remove_item(state, holder, source):
    """Check external removal, honoring effective Sticky Hold on a living holder.

    Self-removal bypasses Sticky Hold. Consumption uses its own helper and is
    not governed by these transfer rules. A fainted holder loses Sticky Hold's
    protection, which matters for Knock Off's after-hit removal.
    """
    item = state.active_field('item', holder)[0]
    sticky = (source != holder) & (state.active_field('current_hp', holder)[0] > 0) & (state.ability[holder, 0] == A.STICKY_HOLD)
    return (item != I.NONE) & item_transfer_allowed(state.active_field('species', holder)[0], state.active_field('form', holder)[0], item) & ~sticky


def effective_speed(state, player, weather):
    """Read battle Speed for power formulas, independently of crits and Unaware."""
    stage = state.boosts.normal_boosts[player, S.SPEED]
    speed = jnp.floor(state.active_stats(player)[S.SPEED] * STAT_MULTIPLIER_LOOKUP[6 + stage]).astype(jnp.int32)
    return modify_speed(speed, state.effective_item(player)[0], state.ability[player, 0],
        state.active_field('status', player)[0], weather, state.terrain.terrain,
        state.volatile_status.ability_state.paradox_stat[player, 0], state.side_conditions.tailwind[player, 0])


def stat_modifier(state, player, indices, weather, *, ability=None):
    """Read the shared item, field, and Speed factors for selected stats."""
    if ability is None:
        ability = state.ability[player, 0]
    return stat_modifier_from_fields(
        indices, state.effective_item(player)[0], ability,
        state.active_field('species', player)[0], state.effective_types(player, include_added=True),
        state.active_field('status', player)[0], weather, state.terrain.terrain,
        state.volatile_status.ability_state.paradox_stat[player, 0], state.side_conditions.tailwind[player, 0])


def modify_stats(state, player, stats, weather=None):
    """Apply the stat-view factors after stage rounding."""
    ability = state.ability[player, 0]
    status = state.active_field('status', player)[0]
    if weather is None:
        weather = effective_weather(state.weather.weather, state.ability, state.active_field('current_hp'))
    indices = jnp.arange(6)
    modifier = stat_modifier(state, player, indices, weather)
    modifier = chain_defense_ability(modifier, ability, status, indices)
    result = apply_modifier(stats, modifier)
    return jnp.where(indices == S.SPEED, _paralyzed_speed(result, ability, status), result)


def chain_defense_ability(modifier, ability, status, stat):
    """Add intrinsic Defense ability factors for views and hit calculations."""
    modifier = chain_modifier(modifier, 6144,
        (stat == S.DEFENSE) & (ability == A.MARVEL_SCALE) & (status != Status.NONE))
    return chain_modifier(modifier, 8192, (stat == S.DEFENSE) & (ability == A.FUR_COAT))


def stat_modifier_from_fields(indices, item, ability, species, types, status, weather,
                              terrain, paradox_stat, tailwind):
    """Build the shared modifier for one stat or a vector of stat indices."""
    choice = (((indices == S.ATTACK) & (item == I.CHOICE_BAND)) |
              ((indices == S.SPECIAL_ATTACK) & (item == I.CHOICE_SPECS)))
    light_ball = (item == I.LIGHT_BALL) & (species == 25) & ((indices == S.ATTACK) | (indices == S.SPECIAL_ATTACK))
    vest = (indices == S.SPECIAL_DEFENSE) & (item == I.ASSAULT_VEST)
    field = ((indices == S.ATTACK) & (ability == A.ORICHALCUM_PULSE) & (weather == W.SUN)) | (
        (indices == S.SPECIAL_ATTACK) & (ability == A.HADRON_ENGINE) & (terrain == T.ELECTRIC))
    weather_defense = ((indices == S.DEFENSE) & (weather == W.SNOW) & jnp.any(types == Type.ICE)) | (
        (indices == S.SPECIAL_DEFENSE) & (weather == W.SANDSTORM) & jnp.any(types == Type.ROCK))
    modifier = chain_modifier(4096, 6144, choice | vest)
    modifier = chain_modifier(modifier, 8192, light_ball)
    modifier = chain_modifier(modifier, 5461, field)
    modifier = chain_modifier(modifier, 6144, weather_defense)
    modifier = chain_modifier(modifier, 5325, indices == paradox_stat)
    # The scalar Speed query uses this exact same chain without other stat reads.
    return jnp.where(indices == S.SPEED,
        _speed_modifier(item, ability, status, weather, terrain, paradox_stat, tailwind), modifier)


def _speed_factors(ability, status, weather, terrain, tailwind):
    """Return the three intermediate Speed modifiers in calculation order."""
    doubled = (((ability == A.CHLOROPHYLL) & (weather == W.SUN)) |
               ((ability == A.SWIFT_SWIM) & (weather == W.RAIN)) |
               ((ability == A.SLUSH_RUSH) & (weather == W.SNOW)) |
               ((ability == A.SAND_RUSH) & (weather == W.SANDSTORM)) |
               ((ability == A.SURGE_SURFER) & (terrain == T.ELECTRIC)))
    quick_feet = (ability == A.QUICK_FEET) & (status != Status.NONE)
    return (jnp.where(doubled, 8192, 4096), jnp.where(quick_feet, 6144, 4096),
            jnp.where(tailwind > 0, 8192, 4096))


def _speed_modifier(item, ability, status, weather, terrain, paradox_stat, tailwind):
    """Build the shared Speed factor for scalar or vector callers."""
    modifier = chain_modifier(4096, 6144, item == I.CHOICE_SCARF)
    for factor in _speed_factors(ability, status, weather, terrain, tailwind):
        modifier = chain_modifier(modifier, factor)
    return chain_modifier(modifier, 6144, paradox_stat == S.SPEED)


def _paralyzed_speed(speed, ability, status):
    """Apply paralysis after the Speed modifier event."""
    return jnp.where((status == Status.PARALYZE) & (ability != A.QUICK_FEET), speed // 2, speed)


def modify_speed(speed, item, ability, status, weather, terrain, paradox_stat, tailwind):
    """Apply the shared Speed factors, followed by paralysis."""
    modifier = _speed_modifier(item, ability, status, weather, terrain, paradox_stat, tailwind)
    return _paralyzed_speed(apply_modifier(speed, modifier), ability, status)
