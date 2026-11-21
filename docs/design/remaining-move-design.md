# Move implementation coverage and remaining architecture

Scarlet/Violet is the default ruleset. After implementing Disable, Encore and
Taunt, Worry Seed, Ivy Cudgel, Blizzard, Hurricane, Substitute, Electro Shot,
Hyper Beam, Destiny Bond, Knock Off, Trick, Switcheroo, Roost, High Jump Kick,
Explosion, Memento, Spiky Shield, Fake Out, Sucker Punch, Stomping Tantrum and
Foul Play, Beat Up, Sleep Talk and Whirlwind, all move slots in the original
curated sets have represented singles implementations. This does not cover the
expanded `moves.json` catalogue. The restriction design discussion below records
the chosen dedicated-field approach.

## Current coverage contract

The catalogue contains **685 moves**, of which **409** have reviewed represented
singles effects and **276** remain unavailable or unreviewed. In the pinned Gen 9
random-battle inventory, **250 of 348** moves are supported and **98** remain
gaps. These are implementation counts, not exhaustive interaction-parity counts.

The explicit support inventory is
[`supported-moves.json`](../../ninjax/data/supported-moves.json), read through
`ninjax.moves.support`. It includes the previous 242-move projection plus reviewed
generic damage, status, stat-change, drain, recoil and multi-hit additions.
Catalogue membership, successful construction, and neutral damage probes alone
do not establish implementation support. New names fail closed until reviewed.

`make_move` retains known metadata for unsupported entries but sets
`implemented=False` and PP to zero. Team construction reports these entries as
omissions even when they occur in `moves.json`. Random-battle metadata checks
skip unsupported entries explicitly. Controlled probes remain limited comparisons
of the particular effects they exercise, not certifications of whole moves.

Regenerate the machine-readable report with the configured project interpreter:

```sh
.venv/bin/python -m tools.audit_random_battle_mechanics > docs/design/gen9-random-coverage.json
```

The [coverage report](gen9-random-coverage.json) lists all unavailable catalogue
moves, supported random-battle names, random-battle gaps and affected roles.
The report uses the same support inventory as construction. Its ability/item
sections retain their separate projection and sampling scopes. Existing
historical batch counts in other design documents describe those earlier runs.

To enable a move, review its complete primary effect and metadata against the
execution path, add any required mechanics and focused regressions, then add its
display name to the support inventory and regenerate the report. Do not infer
support from base power, move flags or the existence of a catalogue record.

## Implementation readiness after the catalogue field expansion

The expanded catalogue now supplies `volatileStatus`, `sideCondition`,
`selfSwitch`, `forceSwitch`, `weather`, `terrain`, `overrideOffensiveStat`,
`overrideDefensiveStat`, `ignoreOffensive`, `ignoreNegativeOffensive` and
`ignorePositiveDefensive`, alongside the existing `ignoreDefensive` field.
These are host-side JSON descriptors. Construction now projects weather, terrain,
primary confusion and stat overrides into move arrays. Reflect, Detect, Flip Turn
and Roar reuse existing execution paths; some dispatch still selects by move
name or ID. Other condition strings remain descriptors without lifecycle support.

### Implemented batch using existing singles state

These **17 moves are now enabled** using existing singles state and shared
mechanics. The table records the wiring for this batch. Primary confusion adds
one boolean move-metadata leaf, not a new battle-state condition. Focused behavior
checks are in `unit_tests/test_wired_moves.py`; support remains scoped to the
represented singles effects and the existing interaction limitations.

| Catalogue information | Enabled moves | Wiring and scope |
| --- | --- | --- |
| `weather` | Sandstorm, Snowscape | Normalize weather names into `WeatherEnum`, project `weather_effect`, and reuse `set_weather` and existing weather residuals. Ensure field targeting avoids opponent-only immunity and Substitute checks. |
| `terrain` | Electric Terrain, Psychic Terrain, Misty Terrain | Map into `TerrainEnum`, project `terrain_effect`, and reuse `set_terrain` with field targeting. Grassy Terrain is already enabled. |
| `sideCondition: reflect` | Reflect | Extend screen application to the existing `BattleState.reflect` timer, including Light Clay, expiry and existing physical-damage reduction. Side targeting must bypass opponent-only hit gates. |
| `overrideDefensiveStat: def` | Psystrike, Secret Sword | Project `defensive_stat` from metadata instead of limiting the override to Psyshock. Body Press's `overrideOffensiveStat: def` likewise describes an already-enabled effect. |
| `volatileStatus: confusion` | Confuse Ray, Supersonic, Sweet Kiss | Dispatch primary confusion through `side.status_effects.apply_confusion` after normal hit checks; retain its existing counter, immunity and switch lifecycle. |
| `volatileStatus: confusion` plus `boosts` | Flatter, Swagger | Compose the existing target stat-change and confusion helpers in move order. Determine success for the whole move: a blocked boost must not automatically discard successful confusion, or vice versa. |
| `volatileStatus: confusion`, `target: allAdjacent` | Teeter Dance | Reuse primary confusion for the opposing active in singles. Full doubles execution still needs target enumeration and an attempt for each eligible adjacent Pokémon. |
| `volatileStatus: protect` | Detect | Share Protect's execution, consecutive-use counter and protection kind. Inherit the documented outstanding protection-timing limitations. |
| `selfSwitch: true` on a damaging move | Flip Turn | Project `switch_out` and reuse the U-turn/Volt Switch pivot continuation. Preserve no-switch behavior on a miss, protection or absorption and when the user cannot switch. |
| `forceSwitch: true` on a status move | Roar | Generalize Whirlwind's ID-specific routing to a forced-switch descriptor. Reuse replacement selection, switch lifecycle and displaced-actor cancellation, while applying Roar's sound flag and keeping Whirlwind-specific wind handling separate. |

Weather strings are not consistently cased: the current values include
`RainDance`, `Sandstorm`, `sunnyday` and `snowscape`. Normalize them through an
explicit mapping; `snowscape` selects `WeatherEnum.SNOW`. Terrain values are
Showdown condition IDs such as `electricterrain`. Do this conversion on the
host and pass numeric, fixed-shape metadata into traced execution.

### Described by the new fields, with additional execution work

| Moves | What the fields provide | Work still required |
| --- | --- | --- |
| Teleport, Chilly Reception | `selfSwitch: true`; Chilly Reception also has `weather: snowscape` | Reuse the existing pivot request with status/field targeting and correct success rules. Chilly Reception must set snow before switching. Verify replacement availability and queued-action timing; a status pivot cannot simply use damage-success rules. |
| Parting Shot | `selfSwitch: true` and sound/bypass flags | Its current record has no `boosts` vector for the offensive drops. Add those semantics and compose drops, reflection/immunity handling and switching with move-specific success rules. |
| Dragon Tail, Circle Throw | `forceSwitch: true` | Add a damaging-hit switch continuation, gated by the actual `HitResult` recipient and survival. Substitute interception must not drag the target; do not reuse Whirlwind's status-only gate unchanged. |
| Baton Pass, Shed Tail | `selfSwitch: copyvolatile` or `shedtail`; Shed Tail also names `substitute` | Typed transfer policies and departure/entry lifecycle. Shed Tail additionally needs its own HP cost and transfer timing; naming Substitute does not implement those rules. |
| Revival Blessing | `selfSwitch: true` | This marker is insufficient to describe revival. Implement fainted-party-member selection and HP restoration; do not treat it as an ordinary active-Pokémon pivot. |
| Mist, Safeguard, Quick Guard, Wide Guard | Named `sideCondition` values | Side-condition state/timers or turn-scoped guards and their consumers in stat, status and move routing. Reflect's existing timer does not provide these effects. |
| Baneful Bunker, Burning Bulwark, Silk Trap, Endure | Named protection/survival volatiles | Extend protection kinds and retaliation or damage-survival rules. Detect can share ordinary Protect; these variants cannot. |
| Bind, Fire Spin, Infestation, Magma Storm, Sand Tomb, Thunder Cage, Whirlpool, Wrap | `volatileStatus: partiallytrapped` | Source-aware trapping state, duration, residual damage, switch restrictions and cleanup. |
| Aqua Ring, Attract, Charge, Curse, Defense Curl, Dragon Cheer, Focus Energy, Gastro Acid, Imprison, Ingrain, Leech Seed, Magnet Rise, Minimize, No Retreat, Power Trick, Smack Down, Stockpile, Tar Shot, Torment, Yawn | Their condition identity | Move-specific condition payloads, application rules, consumers and expiry. The condition string identifies the required handler; it does not supply its lifecycle. |

`selfSwitch` must retain its distinction between boolean `true`, `copyvolatile`
and `shedtail`; reducing all three to a boolean loses transfer semantics. The
Revival Blessing record also demonstrates why even boolean `true` requires
move-specific interpretation. Likewise, compound moves must execute every
required effect before claiming support.

### Gaps the field expansion does not resolve

The added stat-ignore fields need separate offensive/defensive stage policies
if used; they must not all map to the existing defensive-ignore boolean.
Currently `ignoreOffensive`, `ignoreNegativeOffensive` and
`ignorePositiveDefensive` are false for every catalogue entry. The true
`ignoreDefensive` entries are Darkest Lariat and Sacred Sword, both already
enabled, so these additions alone unlock no additional move names.

The existing `secondaries` list describes Fire Fang, Ice Fang, Thunder Fang and
Triple Arrows, but construction still reads only singular `secondary`. These
moves still need multiple-effect representation and independent rolls. Tri
Attack and Dire Claw require custom status-selection rules beyond those lists.
Similarly, `thawsTarget` is present in the catalogue but is not projected into
the current Fire-type-only target-thaw path.

Facade, Hex, Eruption, Electro Ball and Freeze-Dry still need contextual damage
rules; Wish and Future Sight still need pending-effect state; Solar Beam still
needs its charge/weather rules; Seismic Toss still needs fixed-damage dispatch.
None becomes implemented merely through the new condition/switch/stat fields.
Transformations, richer battle history and doubles target resolution remain
architectural work as described below.

The metadata-wiring batch is implemented. Next candidates are status pivots and
damaging forced switches, with the composition requirements listed above. The
support inventory and generated coverage report include the seventeen additions.

### Remaining architecture families

The examples below are unavailable or partial; the report is the full inventory.
Some moves need several of these capabilities.

| Missing support | Example moves | Required extension |
| --- | --- | --- |
| Wiring existing primitives | Grass Knot, Heat Crash, Power Trip | Reuse existing power helpers with move-specific wiring. Weather/terrain setters, Reflect, primary confusion, Psystrike/Secret Sword, Detect, Flip Turn and Roar are implemented above. |
| Conditional damage/type rules | Facade, Hex, Eruption, Electro Ball, Freeze-Dry, Flying Press, Hydro Steam | Resolve contextual power, category, typing and effectiveness before damage. |
| Multiple/custom secondaries | Fire Fang, Ice Fang, Thunder Fang, Triple Arrows, Tri Attack, Dire Claw | Fixed-shape effect lists, independent rolls and custom status selection. |
| Residual conditions and trapping | Leech Seed, Salt Cure, Aqua Ring, Ingrain, Bind, Magma Storm, Syrup Bomb | Source-aware conditions, timers, switch cleanup and ordered residuals. |
| Delayed effects | Wish, Future Sight, Doom Desire, Yawn, Healing Wish | Pending records with side/slot semantics that survive the appropriate switches. |
| Restrictions and prerequisites | Torment, Imprison, Throat Chop, Psychic Noise, Gigaton Hammer, Belch, Last Resort | Shared selection/execution checks, timers and relevant history. |
| Charge/recharge/locked sequences | Solar Beam, Meteor Beam, Giga Impact, Outrage, Rollout | Generalize Electro Shot/Hyper Beam continuations and interruption rules. |
| Preparation and semi-invulnerability | Fly, Dig, Dive, Phantom Force, Focus Punch, Beak Blast | Preparation timing, targetability and pre-execution reactions. |
| Damage/action/team history | Counter, Mirror Coat, Rage Fist, Last Respects, Assurance, Lash Out, Fusion Bolt | Typed records and counters beyond the replaceable current-move hit buffer. |
| Protection and survival | Baneful Bunker, Burning Bulwark, Silk Trap, Endure, Feint | Additional protection variants, retaliation, bypass and HP survival rules. Detect now shares ordinary Protect. |
| Switching composition and state transfer | Dragon Tail, Circle Throw, Parting Shot, Chilly Reception, Teleport, Baton Pass, Shed Tail | The switch descriptors are available; add trigger-specific continuations, compound-effect ordering and transfer policies. Flip Turn and Roar are implemented above. |
| Item lifecycle | Thief, Bug Bite, Incinerate, Fling, Recycle, Stuff Cheeks, Teatime | Theft/consumption operations, provenance and recoverable-item history. |
| Ability/type/form/moveset mutation | Skill Swap, Gastro Acid, Soak, Conversion, Transform, Mimic, Sketch | Eligibility, lifecycle hooks, restoration, copied PP and public knowledge. |
| Custom stats, healing and costs | Belly Drum, Guard Split, Power Trick, Stockpile, Strength Sap, Synthesis, Revival Blessing | Direct assignments, HP costs, counters, contextual healing and party selection. |
| Special damage outcomes | Seismic Toss, Final Gambit, Fissure, False Swipe, Present, Steel Beam, Supercell Slam | Fixed/OHKO damage, HP floors, direct costs, randomized outcomes and crash rules. |
| Field/accuracy conditions | Gravity, Magic Room, Wonder Room, Safeguard, Mist, Magnet Rise, Lock-On, Focus Energy | Application plus lifecycle and consumers for each condition. |
| Calls and queue manipulation | Metronome, Copycat, Instruct, After You, Quash, Upper Hand, Thunderclap | Call eligibility/history, queued prerequisites and action reordering. |
| Doubles targeting/combinations | Follow Me, Rage Powder, Ally Switch, Helping Hand, Wide Guard, Dragon Darts, pledges | Target coordinates, redirection, spread plans, per-active queues and combinations. |
| Transformations | Tera Blast, Tera Starstorm | Eligibility and transformation lifecycle; requests are currently rejected. |

Supported means the represented singles effect exists, not that every exception
is complete. Known cross-cutting gaps include status-failure classification,
protection failure timing, residual ordering, non-Fire target-thaw metadata,
ability/item interactions and full-environment batching acceptance. In particular,
the existing Matcha Gotcha implementation does not establish target-thaw parity.

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

Detect and Spiky Shield share Protect's consecutive-use counter; Spiky Shield tracks its variant
for blocked-contact damage. Unseen Fist bypasses it; Long Reach and Magic Guard
prevent its retaliation. All three protection moves still share the existing
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

No original curated move remains unavailable, but additional forced-switch interactions,
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

`side.item_effects.remove_item(state, holder, source, enabled=True)` and
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
