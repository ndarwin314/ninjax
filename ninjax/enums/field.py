"""Field identities and immutable weather/terrain timer records."""

from collections import namedtuple

from ninjax.enums.base import JaxEnum


Weather = namedtuple("Weather", ["weather", "duration"])


Terrain = namedtuple("Terrain", ["terrain", "duration"])


class WeatherEnum(JaxEnum):
    """Weather identities, with NONE representing a clear field."""
    NONE = 0
    RAIN = 1
    SUN = 2
    SANDSTORM = 3
    SNOW = 4


class TerrainEnum(JaxEnum):
    """Terrain identities, with NONE representing an ordinary field."""
    NONE = 0
    ELECTRIC = 1
    GRASSY = 2
    PSYCHIC = 3
    MISTY = 4


class TurnType(JaxEnum):
    """Stored turn-continuation categories."""
    STANDARD = 0
    SWITCH_MOVE = 1
    END_SWITCH = 2
