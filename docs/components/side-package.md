# Side state and mechanics

`ninjax.side` exports `BattleState` and `VolatileStatus`. Import mechanics from
their owning submodules; the package initializer does not load effect handlers.

| Module | Responsibility |
| --- | --- |
| `side.state` | Immutable state types, factories, active gathers and read-only views |
| `side.queries` | Grounding, immunities, effective weather, item eligibility and stat modifiers |
| `side.updates` | Leaf writes, stage writes and volatile resets without reactive callbacks |
| `side.stat_effects` | Opposing stage drops and Defiant/Competitive reactions |
| `side.status_effects` | Status/confusion attempts, random counters and status-item reactions |
| `side.hp_effects` | Damage/healing, protections, HP reactions and faint recording |
| `side.faint_effects` | Faint records, direct faints and deferred Destiny Bond resolution |
| `side.field_conditions` | Weather and terrain setters |
| `side.item_effects` | Item mutation and update hooks built on basic state writes |
| `side.property_effects` | Effective ability, type and moveset changes |
| `side.form_effects` | Form changes and restoration |
| `side.residual_conditions` | Persistent and delayed effects processed at turn boundaries |
| `side.trapping` | Voluntary switch eligibility and trap application |

The foundation (`state`, `queries`, `updates`) does not import reactive modules
or high-level ability handlers. Item update hooks use basic writes rather
than calling HP/status resolution recursively. HP and status resolution can then
invoke those hooks without an import cycle. High-level residual effects live in
`ninjax.battle.residuals`, while switching lives in `ninjax.battle.switching`.
Move-completion item hooks live in `ninjax.moves.finalization`; the old
`held_effects.py` compatibility file has been removed. Callers import item and
residual helpers from their concrete owners. See [battle phases](battle-phases.md)
for those boundaries.

Pure stage arithmetic lives in `ninjax.calculations.stats`. Public item knowledge lives in
`ninjax.knowledge`, independent of ability handlers. Python ability identity
constants live in `ninjax.ability_rules`; importing them during a JAX trace does
not initialize the state or effect modules.

State factories retain the existing field order, shapes, dtypes and sentinel
values. Factories bind Python shapes and dtypes with `partial`; they create JAX
arrays when state is constructed. Read-only views can call query helpers but
must not invoke mutation or reactive hooks. Existing JIT boundaries, explicit
PRNG advancement and event/callback ordering are preserved by the extraction.

For example:

```python
from ninjax.side import BattleState, VolatileStatus
from ninjax.side.updates import update_active_fields
from ninjax.side.hp_effects import take_damage_value
```

The previous `from ninjax.side import take_damage_value` import is intentionally
replaced by the concrete submodule import. Tests and profiling tools follow the
same ownership boundaries. Historical compilation logs retain their original
`side.py` source locations.
