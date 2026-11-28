"""Shared held-item identity and Plate effect rules."""
import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Type


PLATE_TYPES = (
    (I.FIST_PLATE, Type.FIGHTING), (I.SKY_PLATE, Type.FLYING),
    (I.TOXIC_PLATE, Type.POISON), (I.EARTH_PLATE, Type.GROUND),
    (I.STONE_PLATE, Type.ROCK), (I.INSECT_PLATE, Type.BUG),
    (I.SPOOKY_PLATE, Type.GHOST), (I.IRON_PLATE, Type.STEEL),
    (I.FLAME_PLATE, Type.FIRE), (I.SPLASH_PLATE, Type.WATER),
    (I.MEADOW_PLATE, Type.GRASS), (I.ZAP_PLATE, Type.ELECTRIC),
    (I.MIND_PLATE, Type.PSYCHIC), (I.ICICLE_PLATE, Type.ICE),
    (I.DRACO_PLATE, Type.DRAGON), (I.DREAD_PLATE, Type.DARK),
    (I.PIXIE_PLATE, Type.FAIRY),
)


@jax.jit
def plate_type(item):
    """Return a Plate's type, or NONE for a non-Plate item."""
    return jnp.select([item == plate for plate, _ in PLATE_TYPES],
                      [move_type for _, move_type in PLATE_TYPES], default=Type.NONE)


def item_is_suppressed(ability, magic_room_duration=0, embargo_turns=0):
    """Check the represented ways an active holder ignores its held item."""
    return (ability == AbilityEnum.KLUTZ) | (magic_room_duration > 0) | (embargo_turns > 0)


# Powers for the represented item IDs, grouped by Gen 9 Fling power.
_FLING_POWERS = (
    (10, (I.CHOICE_SPECS, I.LEFTOVERS, I.MENTAL_HERB, I.FOCUS_SASH, I.CHOICE_BAND,
          I.SITRUS_BERRY, I.CHOICE_SCARF, I.WIDE_LENS, I.EXPERT_BELT, I.SILK_SCARF,
          I.CHESTO_BERRY, I.LUM_BERRY, I.WHITE_HERB, I.AIR_BALLOON, I.POWER_HERB, I.SHED_SHELL, I.BIG_ROOT)),
    (30, (I.LIFE_ORB, I.LIGHT_CLAY, I.LOADED_DICE, I.BOOSTER_ENERGY, I.THROAT_SPRAY,
          I.ABILITY_SHIELD, I.KINGS_ROCK, I.PROTECTIVE_PADS, I.ABSORB_BULB, I.LIGHT_BALL,
          I.MAGNET, I.SOUL_DEW, I.FLAME_ORB, I.TOXIC_ORB, I.BINDING_BAND)),
    (60, (I.ROCKY_HELMET, I.WELLSPRING_MASK, I.HEARTHFLAME_MASK, I.CORNERSTONE_MASK, I.LUSTROUS_ORB)),
    (80, (I.HEAVY_DUTY_BOOTS, I.ASSAULT_VEST, I.WEAKNESS_POLICY)),
    (90, (I.GRIP_CLAW,) + tuple(plate for plate, _ in PLATE_TYPES)),
)


def fling_power(item):
    """Read the Gen 9 Fling power, with zero for absent or unknown items."""
    return jnp.select([jnp.any(item == jnp.array(ids, jnp.int32)) for _, ids in _FLING_POWERS],
                      [power for power, _ in _FLING_POWERS], default=0)


def ability_shield_active(item, magic_room_duration, embargo_turns):
    """Ability Shield continues to work for Klutz holders."""
    return (item == I.ABILITY_SHIELD) & (magic_room_duration == 0) & (embargo_turns == 0)
