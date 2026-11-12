# Remaining curated moves: implementation choices

Scarlet/Violet is the default ruleset. After implementing Disable, Encore and
Taunt, Worry Seed, Ivy Cudgel, Blizzard, Hurricane, Substitute, Electro Shot and
Hyper Beam, 16 distinct curated moves remain unavailable. The restriction design
discussion below records the chosen dedicated-field approach.

Actions now use `BattleAction(index, move_type)` with `ActionType.ATTACK`,
`SWITCH`, `CONTINUED`, and `NO_OP`. Queued actions retain this PyTree. The
continued kind forces an Electro Shot release or a Hyper Beam recharge turn;
inactive no-op remains separate. Attack index 4 represents
forced Struggle. Legacy integer inputs and the existing flat legal-action mask
are retained as compatibility adapters.
An independent `transformation` field uses `Transformation.NONE`, `MEGA`,
`Z_MOVE`, `DYNAMAX`, or `TERA`. Non-`NONE` requests remain rejected until
the corresponding battle mechanics and eligibility checks are implemented.

Substitute uses switch-cleared `VolatileStatus.substitute_hp`. A move's most
recent attempts are retained in a fixed ten-entry `HitResults` buffer, with
intended and actual side/active-position coordinates, outcome, recipient and
actual HP removed. Singles always uses active position zero and has no
redirection yet. Doubles target selection should resolve Follow Me and Rage
Powder before accuracy and carry both target coordinates into each attempt;
spread moves will need capacity beyond the present ten-hit singles buffer.
The buffer is internal and is replaced by the next executed move, rather than
being an observation or a complete turn history.

Not every remaining move needs an architectural change. The distinctions below
are about the current engine, not uncertainty about the game rules.

## Shared execution/state changes

| Moves | Missing shared support | Option A | Option B | Suggested approach |
| --- | --- | --- | --- | --- |
| Sleep Talk | Execute another move without spending its PP, while allowing its sleeping caller to act and enforcing the callable-move exclusions. | Extract a common move executor that takes a selected `Move` plus a small execution context (called move, PP policy, sleep exception). | Add a separate Sleep Talk dispatcher into the current hit path and duplicate or bypass selected checks. | A: avoid diverging normal and called-move behavior. A full execution queue is unnecessary. |
| Whirlwind | Pick a random legal replacement, run entry effects, and cancel the displaced Pokémon's pending action. | Switch immediately with an explicit PRNG key and attach actor identity to queued actions so stale actions are discarded. | Add an internal automatic forced-switch request resolved by the existing environment scheduler before continuing. | A is smaller; B is useful if several forced-switch families are planned. The replacement must remain random, not a player choice. |
| Destiny Bond | Record its lifetime and consecutive-use restriction; distinguish a direct opposing-move KO from recoil, weather, status, and other indirect causes. | Add a faint-resolution hook consuming explicit damage cause/source and the active Destiny Bond state. | Inspect accumulated faint/HP events at the end of a move and infer whether retaliation applies. | A: core mechanics should not depend on observation-event capacity or reconstruct causality after the fact. |

## Bounded extensions with reusable helpers

These require some state or interface work, but can fit the current architecture.

| Moves | Option A | Option B | Suggested approach |
| --- | --- | --- | --- |
| Fake Out, Sucker Punch, Stomping Tantrum | Add active tenure, action-taken/selected-action context, and a previous-move result; pass the relevant pending action from `EpisodeState` into move resolution. | Put a complete action queue and move history in `BattleState`. | A. Sucker Punch needs the opponent's still-pending damaging intent; Stomping Tantrum needs an explicit success/failure result, not an HP comparison. |
| Disable, Encore, Taunt | Dedicated switch-cleared move-ID/timer fields, checked in both legal-action masks and execution. | A fixed-size indexed volatile-condition table and generic timer/mask handling. | Implemented using dedicated fields, including Encore's queued-action override. |
| Knock Off, Trick, Switcheroo | Shared item mutation helpers that check removability/transfer legality, update public item knowledge, and emit events atomically. | A general item lifecycle callback system with before/after mutation hooks. | Shared helpers first; include species-bound items such as masks before enabling swaps universally. |
| Worry Seed | Save and restore the original ability when mutating the active Pokémon. | Keep the team ability as the base, with a switch-cleared override and suppression state. | Implemented using the override, move-specific eligibility, represented lifecycle cleanup and public reveal. A general ability-change lifecycle remains future work. |
| Beat Up | Extend the existing fixed hit scan with party eligibility and per-member base power. | Introduce a reusable fixed-size hit-plan PyTree with masks and per-hit parameters. | A is enough for Beat Up; B is useful if later moves need different sources or targets per hit. |

## Moves needing local mechanics rather than a redesign

| Moves | Implementation options |
| --- | --- |
| Foul Play | Select the target as the offensive-stat source in the existing damage calculation, or introduce an explicit offensive-stat-owner field/callback. Preserve the separate ownership rules for stages, offensive modifiers, and the user's burn. |
| High Jump Kick, Explosion, Memento | Add move-specific failure/self-KO hooks at the required resolution points, or introduce a small `MoveResult` and shared attempt/hit/failure/finalization phases. Their self-damage/self-KO timing differs; one unconditional end-of-move flag is insufficient. |
| Roost | Temporarily mutate and restore types, or add a turn-scoped Roost flag to a shared effective-types helper. The effective-types approach composes better with Libero/Protean and other type changes. |
| Ivy Cudgel | Runtime type resolution and form metadata are implemented. Mask damage modifiers and transfer restrictions remain separate item work. |
| Spiky Shield | Add a protection-variant enum alongside existing Protect state, or add a dedicated flag and contact retaliation hook. A shared protection variant is preferable if more protection moves are expected. It should share Protect's consecutive-use counter and existing Unseen Fist rules. |

A practical sequence is: explicit move results and action context; shared called-
move execution; multi-turn continuations and action restrictions; hit routing
for Substitute; then faint/forced-switch lifecycle hooks. Item/form and status
work can proceed independently of that sequence.

## Disable, Encore and Taunt: expanded options

The choice is how to store active conditions. Both options need the same
selection-time and execution-time behavior. Neither requires a new scheduler.

### Option A: dedicated typed fields

Put the following in switch-cleared `VolatileStatus`, using `(2, 1)` leaves:

```text
last_move_id          # 0 means no applicable move used since entry
entry_turn            # set when this Pokémon becomes active
disabled_move_id
disable_turns
encore_move_id
encore_turns
taunt_turns
```

All identifiers are engine move IDs; counters use zero for inactive. IDs are preferable to slots for future
move replacement/Transform support; resolve an ID to a current move slot when
needed. Recording `last_move_id` must occur when a move is actually used, not
when an action is merely selected or stopped before execution. A used move that
misses can still become the relevant last move.

Each move has a small application helper; shared pure helpers construct masks,
resolve Encore overrides, check restrictions before PP consumption, and advance
or clear timers. Existing switch-reset handling clears these fields. The current
`Pokemon.encore_move` placeholder should be migrated so there is one source of
truth, rather than keeping a second lock on the Pokémon.

Advantages: named, documented fields; straightforward tracing and debugging;
small implementation; easy focused tests. Cost: more named fields and explicit
lifecycle code as condition coverage grows. These fields remain JAX-compatible;
a small number of extra PyTree leaves does not require a new representation.

### Option B: an indexed condition table

Use fixed-shape arrays such as:

```text
condition_turns       (2 players, N condition types)
condition_move_id     (2 players, N condition types)
```

An enum assigns columns for Disable, Encore and Taunt. Zero turns means
inactive. Taunt's move-ID column is unused. Shared functions set, clear, expire,
and inspect a condition; move-specific rules still interpret its parameters.
Arrays stay fixed-size while tracing; adding a condition type changes their
static shape and requires recompilation, not dynamic insertion.

Advantages: consistent lifecycle operations, convenient bulk clearing, and a
natural representation for future condition observations. Cost: unused payload
cells, weaker semantic names, and more generic infrastructure. A table does not
remove the need for special Encore action overrides or different duration rules.
Do not use dynamic Python dictionaries of active conditions inside the engine.
Last-move history and entry timing are still separate fields/context under
this option; they are not active conditions themselves.

### Integration required by either option

1. **Application and history.** Disable and Encore refer to an eligible last
   move, verify its continued availability/PP, and apply their failure rules.
   Taunt applies a category restriction. Move eligibility belongs in metadata
   or shared rule helpers, not inferred solely from damage dealt.
2. **Selection-time mask.** Disable removes the matching move; Encore removes
   the other moves; Taunt removes Status moves. Intersect these with PP, Choice
   locks, Assault Vest, and other restrictions. If no moves remain, advertise
   Struggle while keeping any otherwise legal switches available.
3. **Execution-time check.** After an Encore override, check the resolved move
   against current Disable/Taunt state before consuming PP. A move blocked by
   a newly applied restriction fails; do not silently pick another move. Keep
   the turn's existing action order when Encore changes the move.
4. **Queue context and expiry.** Derive a small context from `EpisodeState`:
   whether the target still has a pending move and whether it entered this
   turn. Use that context to initialize durations with Scarlet/Violet timing.
   Blindly starting identical counters and decrementing them each turn would
   produce off-by-one errors. Clear conditions on switching, expire them in
   the appropriate residual order, and end Encore early when its move runs
   out of PP or becomes unavailable.
5. **Items, abilities and observations.** Support Mental Herb's immediate cure
   of these represented conditions, and applicable immunity checks such as
   Oblivious/Aroma Veil for Taunt. Expose public active-condition information;
   keep the opponent's queued choice private. Preserve any caller-specified
   static `Move.disabled` data rather than overwriting it with temporary flags.

### Recommendation

Use Option A behind a shared restriction interface. It matches the current
`VolatileStatus` design and implements these three moves without introducing a
condition framework. The pure helper interface keeps a later table migration
local: masks and action execution would not need to know how effects are stored.

Implement and test Taunt first, then Disable (last-move history), then Encore
(queued-action override and combinations). Check fast and slow application,
switching, failed/interrupted moves, PP exhaustion, Mental Herb, existing item
locks, overlapping Encore+Disable/Taunt, and actual `jit`/`vmap`/`scan` execution.
