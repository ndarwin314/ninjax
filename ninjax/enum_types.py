from collections import namedtuple
from enum import IntEnum, auto

import jax.numpy as jnp

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])

class Type(IntEnum):
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


class MoveType(IntEnum):
    PHYSICAL = 0
    SPECIAL = 1
    STATUS = 2

class StatEnum(IntEnum):
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    SPECIAL_ATTACK = 3
    SPECIAL_DEFENSE = 4
    SPEED = 5

class AccuracyEnum(IntEnum):
    ACCURACY = 0
    EVASION = 1

class WeatherEnum(IntEnum):
    NONE = 0
    RAIN = 1
    SUN = 2
    SANDSTORM = 3
    SNOW = 4

class TerrainEnum(IntEnum):
    NONE = 0
    ELECTRIC = 1
    GRASSY = 2
    PSYCHIC = 3
    MISTY = 4

class Status(IntEnum):
    NONE = 0
    BURN = 1
    PARALYZE = 2
    SLEEP = 3
    FREEZE = 4
    POISON = 5
    TOXIC = 6

class TurnType(IntEnum):
    STANDARD = 0
    SWITCH_MOVE = 1
    END_SWITCH = 2

class AbilityEnum(IntEnum):
    STENCH = auto() # need to implement flinching first
    DRIZZLE = auto() # done
    DROUGHT = auto() # done
    SAND_STREAM = auto() # done
    SNOW_WARNING = auto() # done
    SPEED_BOOST = auto()
    BATTLE_ARMOR = auto() # done
    STURDY = auto() # done
    DAMP = auto()
    ADAPTABILITY = auto() # done
    LIMBER = auto()
    SAND_VEIL = auto()
    STATIC = auto()
    VOLT_ABSORB = auto()
    WATER_ABSORB = auto()
    FLASH_FIRE = auto()
    STORM_DRAIN = auto()
    EARTH_EATER = auto()
    LIGHTNING_ROD = auto()
    SAP_SIPPER = auto()

