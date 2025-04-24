from collections import namedtuple
from enum import IntEnum, auto

import jax.numpy as jnp

Weather = namedtuple("Weather", ["weather", "duration"])
Terrain = namedtuple("Terrain", ["terrain", "duration"])

class JaxEnum(IntEnum):
    def __eq__(self, other):
        return jnp.equal(self, other)
    def __hash__(self):
        return self.value

class Type(JaxEnum):
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


class MoveType(JaxEnum):
    PHYSICAL = 0
    SPECIAL = 1
    STATUS = 2

class StatEnum(JaxEnum):
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    SPECIAL_ATTACK = 3
    SPECIAL_DEFENSE = 4
    SPEED = 5

class AccuracyEnum(JaxEnum):
    ACCURACY = 0
    EVASION = 1

class WeatherEnum(JaxEnum):
    NONE = 0
    RAIN = 1
    SUN = 2
    SANDSTORM = 3
    SNOW = 4

class TerrainEnum(JaxEnum):
    NONE = 0
    ELECTRIC = 1
    GRASSY = 2
    PSYCHIC = 3
    MISTY = 4

class Status(JaxEnum):
    NONE = 0
    BURN = 1
    PARALYZE = 2
    SLEEP = 3
    FREEZE = 4
    POISON = 5
    TOXIC = 6

class TurnType(JaxEnum):
    STANDARD = 0
    SWITCH_MOVE = 1
    END_SWITCH = 2

class AbilityEnum(JaxEnum):
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
    SAND_VEIL = auto() # done
    SNOW_CLOAK = auto() # done
    STATIC = auto()
    VOLT_ABSORB = auto() # done
    WATER_ABSORB = auto() # done
    FLASH_FIRE = auto() # done
    STORM_DRAIN = auto() # done
    EARTH_EATER = auto() # done
    LIGHTNING_ROD = auto() # done
    SAP_SIPPER = auto() # done
    OBLIVIOUS = auto()
    CLOUD_NINE = auto()
    COMPOUND_EYES = auto() # done
    NO_GUARD = auto() # done

