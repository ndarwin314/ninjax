import os
from functools import lru_cache
from typing import Any, Dict, Optional, Union
import orjson

def load_json(file: str) -> Dict[str, Any]:
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
    if(file == "gen9moves.json"):
        with open(
            os.path.join(root, file)
        ) as f:
            return orjson.loads(f.read())
    else:
        root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
        with open(os.path.join(root, "pokedex.json")) as f:
            dex = orjson.loads(f.read())
        dex = {k: v for k, v in dex.items() if v["num"] > 0}
    # remove megas and gmax, currently only targeting gen 9
        dex = {k: v for k, v in dex.items() if ("forme" not in v) or ("forme" in v and v["forme"] not in ["Mega", "Gmax"])}
        return dex
        
   





