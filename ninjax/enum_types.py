from collections import namedtuple
from enum import IntEnum, auto, IntFlag

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

class MoveFlags(IntFlag):
    CONTACT = auto()
    PUNCHING = auto()
    SOUND = auto()

    def __eq__(self, other):
        return jnp.equal(self, other)
    def __hash__(self):
        return self.value

class AbilityEnum(JaxEnum):
    NONE = 0
    STENCH = auto() # need to implement flinching first
    DRIZZLE = auto() # done
    DROUGHT = auto() # done
    SAND_STREAM = auto() # done
    SNOW_WARNING = auto() # done
    SPEED_BOOST = auto()
    BATTLE_ARMOR = auto() # done
    SHELL_ARMOR = BATTLE_ARMOR
    STURDY = auto() # done
    DAMP = auto()
    ADAPTABILITY = auto() # done
    LIMBER = auto() # done
    MAGMA_ARMOR = auto() # done
    WATER_VEIL = auto()
    SAND_VEIL = auto() # done
    SNOW_CLOAK = auto() # done
    STATIC = auto() # done
    FLAME_BODY = auto() # done
    POISON_POINT = auto() # done
    EFFECT_SPORE = auto() # done
    POISON_TOUCH = auto()
    TOXIC_CHAIN = auto()
    VOLT_ABSORB = auto() # done
    WATER_ABSORB = auto() # done
    FLASH_FIRE = auto() # done
    STORM_DRAIN = auto() # done
    EARTH_EATER = auto() # done
    LIGHTNING_ROD = auto() # done
    SAP_SIPPER = auto() # done
    OBLIVIOUS = auto()
    CLOUD_NINE = auto()
    AIR_LOCK = auto()
    COMPOUND_EYES = auto() # done
    NO_GUARD = auto() # done
    INSOMNIA = auto() # done
    VITAL_SPIRIT = INSOMNIA
    COLOR_CHANGE = auto()
    IMMUNITY = auto() # done
    SHIELD_DUST = auto()
    OWN_TEMPO = auto()
    SUCTION_CUPS = auto()
    INTIMIDATE = auto() # added not tested
    SHADOW_TAG = auto()
    ROUGH_SKIN = auto()
    GUTS = auto() # done
    LEVITATE = auto() # done
    MAGIC_GUARD = auto() # implemented hazard immunity
    OVERCOAT = auto() # done
    CLEAR_BODY = auto() # i think
    WHITE_SMOKE = CLEAR_BODY
    FULL_METAL_BODY = CLEAR_BODY
    NATURAL_CURE = auto()
    SERENE_GRACE = auto()
    SWIFT_SWIM = auto() # done
    CHLOROPHYLL = auto() # done
    SLUSH_RUSH = auto() # done
    SAND_RUSH = auto() # done
    TRACE = auto()
    HUGE_POWER = auto() # done
    PURE_POWER = HUGE_POWER
    INNER_FOCUS = auto()
    MAGNET_PULL = auto()
    SOUNDPROOF = auto() # done
    RAIN_DISH = auto() # done
    DRY_SKIN = auto() # need increased damage from being hit by fire move
    SOLAR_POWER = auto() # need to do power increase
    ICE_BODY = auto() # done
    PRESSURE = auto()
    THICK_FAT = auto()
    EARLY_BIRD = auto()
    RUN_AWAY = auto()
    KEEN_EYE = auto()
    HYPER_CUTTER = auto()
    PICKUP = auto()
    TRUANT = auto()
    HUSTLE = auto()
    CUTE_CHARM = auto()
    PLUS = auto()
    MINUS = PLUS
    FORECAST = auto()
    STICKY_HOLD = auto()
    SHED_SKIN = auto()
    MARVEL_SCALE = auto() # done
    LIQUID_OOZE = auto()
    OVERGROW = auto() # done
    BLAZE = auto() # done
    TORRENT = auto() # done
    SWARM = auto() # done
    ROCK_HEAD = auto() # done
    ARENA_TRAP = auto()
    TANGLED_FEET = auto()
    MOTOR_DRIVE = auto() # done
    RIVALRY = auto()
    STEADFAST = auto()
    GLUTTONY = auto()
    ANGER_POINT = auto() # done
    DOWNLOAD = auto()
    IRON_FIST = auto() # done
    TOUGH_CLAWS = auto() # done
    POISON_HEAL = auto()
    SKILL_LINK = auto()
    HYDRATION = auto()
    QUICK_FEET = auto() # done
    NORMALIZE = auto()
    SNIPER = auto() # done
    TECHNICIAN = auto() # done
    STALL = auto()
    LEAF_GUARD = auto()
    KLUTZ = auto()
    MOLD_BREAKER = auto()
    SUPER_LUCK = auto() # done
    AFTERMATH = auto()
    ANTICIPATION= auto()
    FOREWARN = auto()
    UNAWARE = auto()
    TINTED_LENS = auto() # done
    FILTER = auto() # done
    SOLID_ROCK = FILTER
    SLOW_START = auto()
    SCRAPPY = auto()
    HONEY_GATHER = NONE
    FRISK = auto()
    RECKLESS = auto() # done
    MULTITYPE = auto()
    FLOWER_GIFT = auto()
    BAD_DREAMS = auto()
    PICKPOCKET = auto()
    SHEER_FORCE = auto()
    CONTRARY = auto() # done
    UNNERVE = auto()
    DEFIANT = auto() # partial, issue noted in side.reduce_boosts
    DEFEATIST = auto() # done
    CURSED_BODY = auto()
    HEALER = auto()
    FRIEND_GUARD = auto()
    WEAK_ARMOR = auto() # done
    HEAVY_METAL = auto()
    LIGHT_METAL = auto()
    MULTISCALE = auto() # done
    TOXIC_BOOST = auto()
    FLARE_BOOST = auto()


