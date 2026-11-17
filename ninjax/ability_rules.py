"""Python ability identity rules safe to import during a JAX trace."""
from ninjax.enums.abilities import AbilityEnum


UNCHANGEABLE_ABILITIES = (AbilityEnum.AS_ONE_CHILLING, AbilityEnum.AS_ONE_GRIM,
    AbilityEnum.BATTLE_BOND, AbilityEnum.COMATOSE, AbilityEnum.DISGUISE, AbilityEnum.GULP_MISSILE,
    AbilityEnum.ICE_FACE, AbilityEnum.MULTITYPE, AbilityEnum.POWER_CONSTRUCT, AbilityEnum.RKS_SYSTEM,
    AbilityEnum.SCHOOLING, AbilityEnum.SHIELDS_DOWN, AbilityEnum.STANCE_CHANGE, AbilityEnum.TERA_SHIFT,
    AbilityEnum.ZEN_MODE, AbilityEnum.ZERO_TO_HERO)
