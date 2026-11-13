# Curated move architecture and remaining interactions

Scarlet/Violet is the default ruleset. After implementing Disable, Encore and
Taunt, Worry Seed, Ivy Cudgel, Blizzard, Hurricane, Substitute, Electro Shot,
Hyper Beam, Destiny Bond, Knock Off, Trick, Switcheroo, Roost, High Jump Kick,
Explosion, Memento, Spiky Shield, Fake Out, Sucker Punch, Stomping Tantrum and
Foul Play, Beat Up, Sleep Talk and Whirlwind, no curated moves remain
unavailable in the singles sandbox projection. The restriction design
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

Roost uses a turn-scoped, switch-cleared volatile flag. The effective active
snapshot removes Flying without mutating stored types; pure Flying becomes
Normal, while Terastallized users retain their type. The flag expires after
end-of-turn effects and does not override abilities such as Levitate.

Explosion and Memento use a shared direct-faint helper. Explosion saves the
user's pre-faint HP for damage calculation and commits its faint before target
resolution; Damp prevents it. Memento uses `HitResult` to require reaching the
target, then attempts its stat drops and directly faints even if they fail.

Spiky Shield shares Protect's consecutive-use counter and tracks its variant
for blocked-contact damage. Unseen Fist bypasses it; Long Reach and Magic Guard
prevent its retaliation. Both protection moves still share the existing
limitation around use when no other action remains.

Fake Out counts move-action attempts in switch-cleared state. The scheduler
captures the queued actor's team slot and the pending move category for Sucker
Punch, including Struggle and charging continuations. Stomping Tantrum reads
the previous turn's `MoveResult`; at the turn boundary the current result is
copied and reset. `HitOutcome.PROTECTED` is distinct from a failed or immune
attempt. Status callbacks report represented failure conditions, though not all
status move failures are classified yet.

Destiny Bond is the first consumer of `record_faint`/`resolve_faints` in
`side.py`. Damage records capture the actual source side and team slot, target
slot, cause, and bond state. Resolution runs at action boundaries after recoil
and healing, independently of the public event buffer. Retaliation directly
faints a surviving opposing attacker; indirect damage and changed source slots
do not qualify. Its volatile expires on the next action attempt or switch,
and consecutive execution removes the bond and fails. Existing ability faint
hooks remain in the damage path; consolidating those timing rules is future work.

Substitute uses switch-cleared `VolatileStatus.substitute_hp`. A move's most
recent attempts are retained in a fixed ten-entry `HitResults` buffer, with
intended and actual side/active-position coordinates, outcome, recipient and
actual HP removed. Singles always uses active position zero and has no
redirection yet. Doubles target selection should resolve Follow Me and Rage
Powder before accuracy and carry both target coordinates into each attempt;
spread moves will need capacity beyond the present ten-hit singles buffer.
The buffer is internal and is replaced by the next executed move, rather than
being an observation or a complete turn history.

## Latest shared execution support

Beat Up constructs a fixed-shape hit plan from participating party members in
team order. The user contributes even when statused; other contributors must
be alive and status-free. Each hit's power is 5 plus its contributor's species
base Attack divided by ten, rounded down. The actual user supplies offensive
stats and modifiers.

Sleep Talk uses the shared selected-move executor without spending the called
move's PP or repeating the user's before-action checks. Eligible selection
ignores called-move PP and Disable, but excludes Sleep Talk, charge moves and
metadata marked `nosleeptalk`. Sleep or effective Comatose permits execution;
waking before using it makes it fail. Called moves are publicly revealed and
multi-hit calls can continue while asleep.

Whirlwind selects a living participating bench member with an explicit PRNG
key and runs the normal switch lifecycle. Queued actions retain their original
actor slot, so the displaced Pokémon's action is cancelled. The represented
Wind Rider, Suction Cups and Guard Dog checks prevent the forced switch.

No curated move remains unavailable, but additional forced-switch interactions,
protection variants, status-failure classification, ability/item interactions
and new charge/semi-invulnerable variants still need work. The focused tests do
not establish full parity or a completed compiled acceptance pass. See
[mechanics progress](curated-mechanics-progress.md).

## Disable, Encore and Taunt: expanded options

The implemented choice is Option A in `VolatileStatus`, with shared helpers in
`restrictions.py`. The following options preserve the original design rationale.
Both options need the same selection-time and execution-time behavior. Neither
requires a new scheduler.

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

Shared pure helpers apply restrictions, construct masks, resolve Encore
overrides, check restrictions before PP consumption, and advance
or clear timers. Existing switch-reset handling clears these fields. The current
Encore lock is held in volatile state rather than a second lock on the Pokémon.

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

Focused restriction tests cover application timing, switching, PP exhaustion,
Mental Herb, locks and queued overrides. Full acceptance of the latest engine
under `jit`/`vmap`/`scan` remains outstanding.

## Item mutation helpers

`held_effects.remove_item(state, holder, source, enabled=True)` and
`swap_items(state, source, target, enabled=True)` return `(state, success)`.
Failures leave all state unchanged; callers handle move failure messages and
revealing an ability that blocked the effect. `can_remove_item` checks actual
removal including effective Sticky Hold. `item_transfer_allowed` checks only
item identity and holder binding: a damage-power rule can use it independently
of Sticky Hold. Unknown item IDs fail closed.

Swaps validate both outgoing and incoming bindings before either assignment,
then update public item knowledge and events, clean up Choice locks, and run
represented received-item effects (Sitrus Berry, Mental Herb and Booster Energy).
Empty-handed exchanges are valid when at least one item exists. Same-item swaps
are valid and reveal both items. Mask IDs enforce Ogerpon transfer restrictions
and matching-form power modifiers; Wellspring Mask is enabled in the curated
item projection. Absorb Bulb joins Sitrus Berry, Mental Herb and Booster Energy
among the represented received-item effects, activating after a subsequent
damaging Water hit rather than on receipt. Consumption remains a separate
operation.
