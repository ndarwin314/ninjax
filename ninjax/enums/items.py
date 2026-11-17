"""Stable held-item identifiers."""

from enum import auto

from ninjax.enums.base import JaxEnum


class ItemEnum(JaxEnum):
    """Itemenum."""
    NONE = 0
    LIFE_ORB = auto()
    HEAVY_DUTY_BOOTS = auto()
    CHOICE_SPECS = auto()
    ASSAULT_VEST = auto()
    WEAKNESS_POLICY = auto()
    LIGHT_CLAY = auto()
    LOADED_DICE = auto()
    ROCKY_HELMET = auto()
    LEFTOVERS = auto()
    MENTAL_HERB = auto()
    FOCUS_SASH = auto()
    BOOSTER_ENERGY = auto()
    CHOICE_BAND = auto()
    SITRUS_BERRY = auto()
    THROAT_SPRAY = auto()
    ABILITY_SHIELD = auto()
    # Matching Ogerpon forms receive a power boost and cannot transfer masks.
    WELLSPRING_MASK = auto()
    HEARTHFLAME_MASK = auto()
    CORNERSTONE_MASK = auto()
    KINGS_ROCK = auto()
    PROTECTIVE_PADS = auto()
    ABSORB_BULB = auto()
    CHOICE_SCARF = auto()
    WIDE_LENS = auto()
    EXPERT_BELT = auto()
    LIGHT_BALL = auto()
    MAGNET = auto()
    SILK_SCARF = auto()
    SOUL_DEW = auto()
    LUSTROUS_ORB = auto()
    CHESTO_BERRY = auto()
    LUM_BERRY = auto()
    FLAME_ORB = auto()
    TOXIC_ORB = auto()
    WHITE_HERB = auto()
    AIR_BALLOON = auto()
