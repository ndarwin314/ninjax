# Switching and residual phases

`ninjax.switching` owns switch requests and switch execution.
`ninjax.residuals` owns end-of-turn effects. Both depend on concrete `side`
submodules rather than the move dispatcher or the environment.

| Owner | Public entry points |
| --- | --- |
| `switching` | `ForcedSwitchRequest`, `prepare_forced_switch`, `execute_forced_switch`, `force_switch_target`, `swap_out`, `swap_is_alive` |
| `residuals` | `end_turn_damage`, `step_side_conditions`, `step_moody`, `bad_dreams`, `residual_abilities`, `leftovers`, `residual_orbs` |

Switch preparation resolves represented prevention and samples a legal slot.
Only successful preparation advances the key for selection and entry effects.
Commitment performs departure, resets, entry hazards and surviving entry hooks.
The request is an action-local PyTree; it is not stored in battle state. The
move dispatcher still commits a forced switch before the next actor proceeds.

`battle.step_field` remains the scheduler. It retains the existing sequence of
side-condition updates, status cures, weather/terrain/status HP effects,
Leftovers, Moody, Speed Boost, status orbs, item updates, Perish Song, timer
updates and deferred faint resolution. This extraction preserves the current
event and PRNG ordering, including incomplete residual-order coverage; it does
not change mechanics or establish full generation 9 parity.

The residual module does not import switching, ability/item facades, move
execution or the environment. Switching uses the shared Frisk ability handler;
ability handlers may re-export residual functions, so dependencies must continue
to point toward the residual module rather than back from it.

Repository callers use direct imports:

```python
from ninjax.switching import prepare_forced_switch, execute_forced_switch
from ninjax.residuals import end_turn_damage, residual_abilities
```

The obsolete `game_logic.py` and `held_effects.py` compatibility files have been
removed, along with residual re-exports from `ability_effects`.
`moves.finalization.after_move` owns the Life Orb/Throat Spray move-completion hook. Profiling
components import switching from its owner after configuring benchmark cache
policy; historical benchmark logs keep their original source locations.
