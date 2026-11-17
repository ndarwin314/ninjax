# Move mechanics

`ninjax.moves` exports only the `Move` PyTree. Construction and Pokémon storage
can import that model without initializing execution or side effects. Mechanics
callers import their concrete owners:

| Module | Responsibility |
| --- | --- |
| `moves.model` | Move fields, defaults and metadata normalization |
| `moves.execution` | PP and selection bookkeeping, called moves, charge setup and attempt dispatch |
| `moves.routing` | Accuracy, immunity, protection and damage/status branch selection |
| `moves.properties` | Effective typing, power, priority, accuracy and interception queries |
| `moves.hit_resolution` | One damaging hit, Substitute damage, drain, recoil and hit callback ordering |
| `moves.multi_hit` | Hit-count sampling, per-hit accuracy and repeated-hit execution |
| `moves.status` | Status move execution, including Rest, Pain Split, Substitute, Destiny Bond and Memento |
| `moves.effects` | Contact reactions, secondary effects and guaranteed after-damage effects |
| `moves.finalization` | Once-per-move completion, item hooks, recharge and outcome-dependent effects |

Generic stage-vector application belongs to `side.stat_effects`; field-wide
status moves belong to `side.field_conditions`. These are shared by damaging
and status moves and do not import the dispatcher. `curated_move_effects.py`
has been removed; its handlers now belong to these concrete modules.

Multi-hit execution depends on single-hit resolution, properties and
finalization. Single-hit resolution invokes per-hit effects and basic side
operations. Finalization does not import multi-hit execution. Status execution
can invoke switching but does not import damage routing or move selection.

`moves.execution` calls `moves.routing`, which calls damage/status mechanics
without importing selection or the battle scheduler. The scheduler in
`battle.py` imports the execution entry points directly. `game_logic.py` now
contains only explicit compatibility exports for the extracted mechanics.
`move.py`, `move_effects.py` and `held_effects.py` retain legacy
exports; repository callers import the concrete owners. Profiling setup keeps
its imports after cache-policy configuration.

Existing PRNG advancement, event ordering, per-hit versus once-per-move timing,
and JIT boundaries are preserved. Pain Split still assigns HP directly before
reactive berries; Rest still defers berry activation until healing completes.
Substitute handling still reports the actual recipient and capped HP loss.
This migration does not change mechanics or claim full generation 9 parity.
