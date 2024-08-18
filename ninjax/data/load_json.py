import os
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import orjson

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")

with open(
        os.path.join(root, "pokedex.json")
) as f:
    dex = orjson.loads(f.read())
# remove CAMOmons
dex = {k: v for k, v in dex.items() if v["num"] > 0}
# remove megas and gmax, currently only targeting gen 9
dex = {k: v for k, v in dex.items() if ("forme" not in v) or ("forme" in v and v["forme"] not in ["Mega", "Gmax"])}
