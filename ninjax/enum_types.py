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
    BULLET = auto()
    LAUNCHER = auto()
    HEALING = auto()
    BITING = auto()

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
    TOXIC_BOOST = auto() # done
    FLARE_BOOST = auto() # done
    HARVEST = auto()
    TELEPATHY = auto()
    MOODY = auto() # probably fine, to be correct we need to exclude stats that are already +6 or -6 but that seems hard
    REGENERATOR = auto()
    BIG_PECKS = auto()
    WONDER_SKIN = auto()
    ANALYTIC = auto()
    ILLUSION = auto()
    IMPOSTER = auto()
    MOXIE = auto() # done
    BEAST_BOOST = auto() # done
    JUSTIFIED = auto()
    RATTLED = auto()
    MAGIC_BOUNCE = auto()
    PRANKSTER = auto() # done
    SAND_FORCE = auto()
    IRON_BARBS = ROUGH_SKIN
    ZEN_MODE = auto()
    VICTORY_STAR = auto()
    TURBOBLAZE = MOLD_BREAKER
    TERAVOLT = TURBOBLAZE
    AROMA_VEIL = auto()
    FLOWER_VEIL = auto()
    CHEEK_POUCH = auto()
    PROTEAN = auto()
    LIBERO = PROTEAN
    FUR_COAT = auto()
    MAGICIAN = auto()
    BULLET_PROOF = auto()
    COMPETITIVE = auto() # done
    STRONG_JAW = auto() # done
    REFRIGERATE = auto() # done
    PIXILATE = auto() # done
    AERILATE = auto() # done
    GALVANIZE = auto() # done
    LIQUID_VOICE = auto() # done
    SWEET_VEIL = auto()
    STANCE_CHANGE = auto()
    GALE_WINGS = auto() # done
    TRIAGE = auto() # done
    GOOEY = auto() # done
    TANGLING_HAIR = GOOEY
    MEGA_LAUNCHER = auto() # done
    GRASS_PELT = auto()
    SYMBIOSIS = auto()
    PARENTAL_BOND = auto()
    DARK_AURA = auto() # done
    FAIRY_AURA = auto() # done
    AURA_BREAK = auto() # done
    PRIMORDIAL_SEA = auto()
    DESOLATE_LAND = auto()
    DELTA_STREAM = auto()
    STAMINA = auto() # done
    WIMP_OUT = auto()
    EMERGENCY_EXIT = WIMP_OUT
    WATER_COMPACTION = auto()
    MERCILESS = auto() # done
    SHIELDS_DOWN = auto()
    STAKEOUT = auto() # add some kind of switched out flag
    WATER_BUBBLE = auto() # done
    STEELWORKER = auto() # done
    BERSERK = auto() # hopefully done
    LONG_REACH = auto() # done
    SURGE_SURFER = auto() # done
    SCHOOLING = auto()
    DISGUISE = auto()
    BATTLE_BOND = auto()
    POWER_CONSTRUCT = auto()
    COMATOSE = auto() # i think done
    QUEENLY_MAJESTY = auto()
    DAZZLING = QUEENLY_MAJESTY
    ARMOR_TAIL = QUEENLY_MAJESTY
    INNARDS_OUT = auto() # done
    DANCER = auto()
    BATTERY = auto()
    FLUFFY = auto() # done
    SOUL_HEART = auto() # done, fine for singles but not doubles
    RKS_SYSTEM = auto()
    ELECTRIC_SURGE = auto()
    PSYCHIC_SURGE = auto()
    GRASSY_SURGE = auto()
    MISTY_SURGE = auto()
    SHADOW_SHIELD = MULTISCALE # TODO: THESE ARENT IDENTICAL BECAUSE THEY CANT BE SUPPRESSED
    PRISM_ARMOR = SOLID_ROCK # IDK HOW WE WILL HANDLE ABILITY SUPPRESSION STUFF
    NEUROFORCE = auto()
    INTREPID_SWORD = auto()
    DAUNTLESS_SHIELD = auto()
    BALL_FETCH = NONE
    COTTON_DOWN = auto() # done-ish
    PROPELLER_TAIL = auto()
    STALWART = PROPELLER_TAIL
    MIRROR_ARMOR = auto() # maybe
    GULP_MISSILE = auto()
    STEAM_ENGINE = auto()
    PUNK_ROCK = auto() # done
    SAND_SPIT = auto()
    ICE_SCALES = auto()
    RIPEN = auto()
    ICE_FACE = auto()
    POWER_SPOT = auto()
    MIMICRY = auto()
    SCREEN_CLEANER = auto()
    STEELY_SPIRIT = STEELWORKER # not in doubles
    PERISH_BODY = auto()
    WANDERING_SPIRIT = auto()
    GORILLA_TACTICS = auto()
    NEUTRALIZING_GAS = auto()
    PASTEL_VEIL = IMMUNITY # not in doubles
    HUNGER_SWITCH = auto()
    QUICK_DRAW = auto()
    UNSEEN_FIST = auto()
    CURIOUS_MEDICINE = auto()
    TRANSISTOR = auto() # done
    DRAGONS_MAW = auto() # done
    CHILLING_NEIGH = MOXIE
    GRIM_NEIGH = auto() # in singles this is functionally identical to soul heart but eventually i want to implement doubles
    AS_ONE_CHILLING = auto()
    AS_ONE_GRIM = auto() #so if these were flag enums i could just or them but there are over 300 abilities so idk if i want to make this a flag enum
    LINGERING_AROMA = WANDERING_SPIRIT
    SEED_SOWER = auto() # done
    THERMAL_EXCHANGE = auto() # done
    ANGER_SHELL = auto()
    PURIFYING_SALT = auto()
    WELL_BAKED_BODY = auto() # done


