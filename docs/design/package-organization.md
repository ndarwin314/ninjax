# Package ownership

Battle transitions retain their explicit PRNG keys and immutable PyTree state.
The directories organize ownership; they do not introduce a new mechanics layer.

| Package | Concrete owners |
| --- | --- |
| `battle` | `environment` for the Jumanji API; `types` for action/episode models; `scheduler` for queued turns and replacement decisions; `ordering` for priority/Speed; `actions` and `interruptions` for one actor; `switching`, `residuals` and `field` for their respective phases |
| `teams` | `construction` for caller-provided move/team builders; `curated` for set parsing, pools and sampling |
| `records` | `hits` for per-attempt outcomes and recipients; `events` for fixed-capacity public transition records |
| `calculations` | `damage`, `stats`, `stat_formulas`, `type_effectiveness`, `modifiers` and `logic` |
| `enums` | `base` for traced comparisons; `pokemon` for types/status/stats/forms; `moves` for categories/flags/outcomes; `field` for weather/terrain records and IDs; `abilities` and `items` for their stable IDs |

The five package initializers export nothing and do not load JAX or the engine.
Callers import concrete owners:

```python
from ninjax.battle.environment import Battle
from ninjax.battle.types import BattleAction, BattleParams
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import Status, Type
from ninjax.records.hits import HitOutcome, HitResults
from ninjax.teams.construction import make_move
```

`battle.environment` depends on the scheduler; the scheduler depends on actions
and field phases. Actions may invoke move execution and switching. Lower-level
state, records, enums and calculations never import the environment or scheduler.
`moves.status` can import `battle.switching` without initializing the environment.

The former flat files have been removed rather than retained as compatibility
wrappers. All repository callers, embedded subprocess test scripts and profiling
tools use the concrete owners. Team data paths still resolve to `ninjax/data`.
Existing `ninjax.side` and `ninjax.moves` public model exports are unchanged.

All 403 members and aliases from the former `enum_types.py` retain their integer
values. The weather/terrain namedtuple field names, model fields, array shapes,
event ordering and random-key operations are preserved. Nested battle callbacks
are module-level helpers; `partial` binds captured inputs without making battle
arrays static JIT arguments. Historical benchmark reports keep their original
source paths and timings.
