"""Shared held-item identity and Plate effect rules."""
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


def plate_type(item):
    """Return a Plate's type, or NONE for a non-Plate item."""
    return jnp.select([item == plate for plate, _ in PLATE_TYPES],
                      [move_type for _, move_type in PLATE_TYPES], default=Type.NONE)


def item_is_suppressed(ability, magic_room_duration=0, embargo_turns=0):
    """Check the represented ways an active holder ignores its held item."""
    return (ability == AbilityEnum.KLUTZ) | (magic_room_duration > 0) | (embargo_turns > 0)
