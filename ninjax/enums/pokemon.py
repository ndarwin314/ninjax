"""Pokémon attributes and stat-axis identifiers."""

from enum import auto

from ninjax.enums.base import JaxEnum


class FormEnum(JaxEnum):
    """Stable battle form IDs; zero covers forms without special move rules."""
    DEFAULT = 0
    OGERPON_WELLSPRING = 1
    OGERPON_HEARTHFLAME = 2
    OGERPON_CORNERSTONE = 3
    TAUROS_PALDEA_COMBAT = 4
    TAUROS_PALDEA_BLAZE = 5
    TAUROS_PALDEA_AQUA = 6
    MORPEKO_HANGRY = 7
    MIMIKYU_BUSTED = 8
    EISCUE_NOICE = 9
    MINIOR_METEOR = 10
    PALAFIN_HERO = 11
    MELOETTA_PIROUETTE = 12
    CRAMORANT_GULPING = 13
    CRAMORANT_GORGING = 14
    TERAPAGOS_TERASTAL = 15
    GIRATINA_ORIGIN = 16
    DIALGA_ORIGIN = 17
    PALKIA_ORIGIN = 18
    ZACIAN_CROWNED = 19
    ZAMAZENTA_CROWNED = 20


class Type(JaxEnum):
    """Elemental type IDs, including the typeless NONE sentinel."""
    NONE = 0
    NORMAL = auto()
    FIRE = auto()
    WATER = auto()
    ELECTRIC = auto()
    GRASS = auto()
    ICE = auto()
    FIGHTING = auto()
    POISON = auto()
    GROUND = auto()
    FLYING = auto()
    PSYCHIC = auto()
    BUG = auto()
    ROCK = auto()
    GHOST = auto()
    DRAGON = auto()
    DARK = auto()
    STEEL = auto()
    FAIRY = auto()


class StatEnum(JaxEnum):
    """Positions of HP, Attack, Defense, Special Attack, Special Defense and Speed."""
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    SPECIAL_ATTACK = 3
    SPECIAL_DEFENSE = 4
    SPEED = 5


class AccuracyEnum(JaxEnum):
    """Positions of accuracy and evasion in their separate stage vector."""
    ACCURACY = 0
    EVASION = 1


class Status(JaxEnum):
    """Persistent major status-condition IDs."""
    NONE = 0
    BURN = 1
    PARALYZE = 2
    SLEEP = 3
    FREEZE = 4
    POISON = 5
    TOXIC = 6
