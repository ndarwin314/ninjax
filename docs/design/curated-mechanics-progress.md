# Curated mechanics implementation progress

## Expanded passive ability and item coverage

The [passive mechanics report](random-battle-passives.md) records the current
expansion from 30 to **152 projected ability names** (149/203 in the pinned
random-battle inventory) and from 19 to **35 held-item names**. It documents
the new Showdown probes, correctness fixes, JAX helper validation and the
remaining form/history/event-lifecycle work. The generated
[coverage report](gen9-random-coverage.json) lists current omissions. Historical
batch descriptions below retain their original validation scope.

## Full Gen 9 random-battle suite: first batch

The test inventory now covers all **879 species/role entries**, **348 distinct
moves**, and **203 listed abilities** from the pinned Showdown
`data/random-battles/gen9/sets.json`. These are complete candidate movepools per
role, not an enumeration of every legal four-move combination or interaction.
The engine catalogue now represents **232 of those moves** after adding 127
metadata-driven moves. The remaining **116 moves are explicit test skips**.

The additions reuse existing damage, priority, critical-rate, stat-change,
secondary status/flinch/confusion, recoil, drain and half-HP healing hooks.
Move projection now reads critical ratios, never-miss accuracy, healing and
primary status metadata, toxic/paralysis secondaries and self-effect chances.
Examples include Shell Smash, Quiver Dance, Nuzzle, Wicked Blow, Leaf Storm,
Drain Punch and Diamond Storm. No state fields or dependencies were added.

`tools/record_random_battle_mechanics.cjs` records exact integer damage for all
16 random rolls on 109 new damaging moves, real Showdown outcomes for 12 setup
moves, and forced secondary-effect outcomes. Tests compare engine hits at the
corresponding random roll and compare setup/secondary stat stages and statuses.
Neutral damage probes remove STAB, resistances, criticals, items and other
abilities; passing them does not establish arbitrary interaction parity.
The secondary probes exercise successful procs, not their probability distribution.

The configured HTTP server was unavailable during capture. Reference probes use
the already-installed simulator behind that server at revision
`23b41aa29e2ff5ff80e03f57b257b595a5852047`, through its actual Battle/action APIs.
The server was subsequently launched with `node pokemon-showdown start
--skip-build 8000` from `/home/mathy/pokemon-showdown`, and an HTTP GET to
`http://127.0.0.1:8000/` returned **200**. Server startup and the HTTP check
required execution outside the sandbox: ordinary sandboxed requests still failed
to connect. Existing oracle fixtures were generated through the simulator API;
this HTTP availability check does not change their validation scope.
Fixtures run offline; regeneration requires the local Showdown checkout:

```sh
SHOWDOWN_REVISION=23b41aa29e2ff5ff80e03f57b257b595a5852047 \
  node tools/record_random_battle_mechanics.cjs /home/mathy/pokemon-showdown
.venv/bin/python -m tools.audit_random_battle_mechanics
.venv/bin/python -m pytest -q unit_tests/test_random_battle_mechanics.py
```

[The coverage report](gen9-random-coverage.json) lists missing moves and abilities
not yet enabled by the curated projection. An unprojected ability may already
have an engine hook; it still needs validation before enabling it here. Item
selection is procedural, so the report explicitly labels its 100 generated-team
item sample as non-exhaustive. That move-only batch did not enable new abilities
or items; the subsequent passive expansion is described above.
Further work includes additional projections and parity probes, conditional powers,
self secondaries, move-locking/recharge variants, trapping, delayed effects and
transformations. All existing curated limitations below still apply.

Validation for this batch: the inventory/metadata/damage/setup run passed
**1,233 tests with 116 explicit unsupported-move skips** (273.44 seconds).
The separately added secondary comparisons passed **54 tests** (23.64 seconds),
and four focused curated projection/self-effect regressions passed (19.20 seconds).
These were eager mechanics runs with the project virtual environment and the
persistent cache enabled at `.profiling/jax-cache`; no compilation benchmark or
new full-environment transformation acceptance is claimed.

The initial singles environment is implemented, and all 120 move slots in the
30 curated sets now have supported implementations. Full Scarlet/Violet
interaction parity is **not yet established**. The default curated runner keeps
abilities/items neutral; opt-in modes enable all 30 curated ability names and
all 19 curated item names. Unsupported items in custom collections remain
explicit omissions. Moves outside the supported catalogue are still reported
and made unavailable.

Focused mechanics and environment tests passed during implementation, including
the Beat Up, Sleep Talk, Whirlwind and final curated item additions. Complete
compiled singles battles now pass for seeds 0, 17 and 42 with abilities/items
enabled. Full-environment `vmap` and `scan` acceptance remains outstanding.
Earlier compiled checks do not establish coverage of subsequent changes.
The initial milestone requires a
usable singles Jumanji environment; doubles, transformations and self-play
training infrastructure are outside that milestone.

The 2026-11-13 compiled full-battle retry used curated seed 0 with abilities and
items enabled, two CPU cores and a 6 GiB process virtual-memory limit. It failed
with `MemoryError: std::bad_alloc` while preparing the first compiled step
(164.72 seconds in pytest). No compiled battle completed in that attempt.
That attempt left whole-step compilation as a validation blocker; the later
action-dispatch consolidation described below clears this gate for three seeds.

The corresponding full eager run (`jax.disable_jit()`, same seed and enabled
abilities/items, same CPU and memory limits) **passed**: 40 turns, 51 decisions,
reward `[1.0, -1.0]`, and 2113 seconds elapsed. It reached a team-defeat terminal
outcome before the 100-turn limit, with no invalid actions, nonnegative recorded
HP, six distinct selected sets per side and no omitted moves, abilities or held
items. The result validates this single eager rollout; it does not establish
compiled, batched or scanned full-battle acceptance or exhaustive mechanics
parity.

Subsequent [compilation-memory work](compilation-performance.md) reduced the
full-step trace from 802,117 to 70,666 equations and peak tracing RSS from
1,801 to 619 MiB. At that stage, whole-step backend compilation still exceeded
the tested memory budgets. Focused regression and shared-helper transformation checks
cover the optimizations; the earlier eager rollout does not validate them.
The initial isolated compilation checks passed 17 of 18 components; full move
execution exceeded the 4 GiB RSS watchdog. After refactoring dispatch to share
the hit pipeline, both selected-move dispatch and full move execution compile
and match eager output within that budget. Dispatcher backend time fell from
104.6 to 33.5 seconds, with peak RSS falling from 3,530 to 2,239 MiB. Full move
execution now completes at 2,163 MiB. The compilation-performance document
contains before/after flow diagrams, raw measurements and validation scope;
component results do not clear full-battle acceptance.
The first post-refactor full-step profiler still hit the 4 GiB RSS watchdog
during backend compilation (4,107 MiB observed).

The subsequent action-dispatch consolidation shares ordinary/charged move
execution and uses one sequential replacement-switch loop. Full-step HLO after
call-graph flattening falls from 587,633 to 381,214 instructions. Compilation
and all-leaf eager comparison now fit within the 4 GiB RSS guard. Three full
compiled battles with abilities/items enabled pass in one process at 3,173 MiB
peak RSS: seed 0 takes 40 turns/51 decisions, seed 17 takes 25/32 and seed 42
takes 26/35. Compile logs show one shared step compilation. The runner's policy
also now selects the forced continue action for charge/recharge turns. These
are three singles rollouts, not exhaustive interaction or batching acceptance.

The first implementation batch adds:

- Internal held-item IDs, consumption events, Focus Sash and Sitrus Berry.
- Choice Band/Specs modifiers and switch-cleared move locks; Assault Vest's
  special-defense modifier and status-move restriction.
- Life Orb damage/recoil hooks, Rocky Helmet contact damage, Leftovers healing,
  Weakness Policy, Throat Spray, and hazard bypass for Heavy-Duty Boots.
- Effective weather suppression by Cloud Nine/Air Lock without deleting weather.
- Orichalcum Pulse/Hadron Engine stat modifiers; state for Paradox activation,
  best-stat selection, Booster Energy consumption, and switching resets.
- Initial entry ability execution, Shadow Tag's switch mask, Regenerator,
  once-per-switch Libero and original-type restoration, and corrected Moody caps.
- Effective move-type handling for Pixilate and related abilities, Sand Force,
  and corrections to Ruin defensive-stat selection and matching-ability immunity.
- Frisk with persistent public item knowledge and an independent observation
  mask; public consumption updates the revealed item to none.
- Opt-in `--curated --abilities` mode for supported singles abilities,
  including entry activation after team preview and grounded terrain rules.
- Opt-in `--curated --items` initially projected 15 items with engine hooks;
  the final batch below expands this to all 19 current curated item names.
  It can be combined with `--abilities`.

The [remaining-ability validation matrix](remaining-ability-validation.md)
records the earlier follow-up audit. Unseen Fist is now enabled with initial
protection support as described below. Friend Guard
has no effect in singles, and its doubles behavior remains unvalidated.

These are engine hooks, not a claim that every interaction is complete or that
the curated sets can be enabled without omissions. Focused tests are in
`unit_tests/test_held_effects.py`. Numeric modifiers now owned by `side.queries`
use the pinned Showdown source's 4096-based rounding convention; the older
engine still has modifier-rounding work outside these new hooks.

Active ability reads and battle immunity checks now resolve switch-cleared
ability overrides and suppression through `BattleState`. The stored team
ability remains unchanged. Worry Seed uses a dedicated transition for its
represented lifecycle effects and public reveal. A general ability-change
lifecycle for other moves is still unimplemented; setting an override alone
does not run those hooks.

The next move batch enables **58 additional distinct curated moves**. The
current 30-set projection has **0 unavailable moves in 0 of 120
slots**, down from 87 distinct unavailable moves in 93 slots. Enabled means
the represented singles effect runs; it does not establish complete parity.

New shared mechanics include:

- Per-hit secondary status, stat drops and flinch, with Shield Dust, Serene
  Grace and Inner Focus handling. Flinch interrupts an action and expires at
  the end of the turn.
- Close Combat, Draco Meteor and Overheat self drops; Calm Mind and Dragon
  Dance setup; Body Press/Psyshock stat selection; Sacred Sword stage bypass;
  Flower Trick's guaranteed critical hit and accuracy bypass.
- Night Shade, Super Fang, Ruination and Endeavor fixed damage. Drain and
  single-hit recoil now use actual HP lost before reactive berry healing.
- Weather Ball typing/power, Acrobatics, Stored Power, Low Kick/Heavy Slam,
  Electro Drift and grounded Expanding Force power; Grassy Glide priority.
  Curated teams retain species weight in integer hectograms.
- Light Screen/Aurora Veil (including Light Clay duration), Tailwind and Trick
  Room ordering; Haze, Clear Smog, Brick Break, Defog, Court Change, Rapid Spin,
  Ceaseless Edge, Stone Axe, Ice Spinner and Soundproof-aware Perish Song.
- Catalogue move flags for contact, sound, powder, pulses, bites, healing,
  projectiles and protection. Protect now checks the move's protection flag.

Focused coverage is in `unit_tests/test_curated_moves.py`, including opt-in compiled
`jit`/`vmap`/`scan` checks for the shared callbacks. Existing engine limitations
remain, including incomplete general ability interactions, modifier rounding,
and damage-triggered effects. Doubles effects are not validated by this batch.

Move stat changes use paired index/stage vectors padded to seven slots,
independently of scalar move metadata. `self_chance` rolls once for the entire
self-stat vector, allowing future chance-based boosts to all ordinary stats.

A follow-up batch enables Poltergeist (item requirement and public reveal),
Pain Split (direct HP assignment before reactive berries), and Rest (sleep
eligibility, status replacement, full healing, and its fixed sleep duration).
These reuse existing state without adding fields. Focused checks are in
`unit_tests/test_direct_status_moves.py`. Implementation choices for the other
moves are listed in [remaining move design options](remaining-move-design.md).

Disable, Encore and Taunt use switch-cleared `VolatileStatus` counters and
move IDs. Their action masks, queued actions, Mental Herb interaction, and
public observations are implemented for singles. Durations are based on the
pinned Showdown residual timing; uncommon interactions still need parity audit.
Focused tests are in `unit_tests/test_move_restrictions.py`.

Worry Seed now replaces eligible abilities with Insomnia, wakes sleeping targets,
clears represented old-ability state and publicly reveals the changed ability.
Ability Shield blocks it; immutable ability identities are checked using the
pinned Scarlet/Violet ability flags. The base ability returns on switching.

Ivy Cudgel resolves its type from the user's stored form; Revelation Dance uses
the user's current primary type or ordinary Tera type. Curated construction
preserves the Ogerpon form separately from its shared Pokédex number. Item
mutation helpers enforce represented Ogerpon mask transfer restrictions. Matching
Wellspring, Hearthflame and Cornerstone masks apply the pinned 4915/4096 power
modifier to every damaging type. Wellspring Mask is enabled in curated teams;
automatic form selection for arbitrary caller-provided teams remains outside
the item hook.

Blizzard and Hurricane use weather-sensitive accuracy and their freeze and
confusion secondary effects. Freeze and confusion are resolved before actions;
confusion duration is private while its presence is observable.

Substitute stores switch-cleared HP and routes eligible hits to the doll;
sound, explicit bypass flags and Infiltrator bypass it. Hit attempts are
recorded with singles target coordinates for future doubles target resolution.

Electro Shot charges with a Special Attack boost, or attacks immediately in
effective rain. Hyper Beam forces a recharge turn after hitting. Both use the
continued action and switch-cleared state; charge release spends no extra PP.

Destiny Bond uses source-aware pending faint records, resolved after a move's
HP effects. It directly faints a surviving opposing attacker without damage
protections. Its lifetime and consecutive-use restriction are implemented;
the observation event buffer does not participate in mechanic resolution.

Roost heals and temporarily removes Flying from effective types until the end
of the turn. Pure Flying becomes Normal; Terastallized users retain their type.

Explosion directly faints its user before target resolution unless Damp blocks
the move. Memento lowers the target's offensive stats and directly faints its
user when the recorded hit reaches the Pokémon, including when drops fail.

Fake Out counts action attempts since entry. Sucker Punch uses the queued
opponent action and its original active slot. Stomping Tantrum doubles power
after a recorded previous-turn failure; protection, recharge and no action are
distinguished from failure. Some status-move failure reporting remains partial.

Foul Play selects the target's raw Attack and stages, then applies the user's
offensive modifiers and burn rules.

Beat Up uses eligible party members' species base Attack for individual hit
power. Sleep Talk selects an eligible move without using its PP; Whirlwind
randomly forces a legal replacement and cancels the displaced queued action.

The final curated item batch enables King's Rock, Protective Pads, Wellspring
Mask and Absorb Bulb. King's Rock adds an independent per-hit flinch chance
without stacking with native flinch. Protective Pads blocks represented defender
contact reactions and Spiky Shield retaliation while preserving contact bonuses,
Unseen Fist and Poison Touch. Prevented Rough Skin/Gooey reactions publicly
reveal Pads and the defending ability. Absorb Bulb consumes on a surviving holder after a
damaging Water hit reaches it and raises Special Attack once; Substitute and
Water absorption prevent activation. Focused tests are in
`unit_tests/test_remaining_items.py`. This enables every current curated item,
not every interaction with those items.

Remaining integration and mechanics include:

- Auditing move-specific exceptions and interactions across the enabled catalogue;
  zero unavailable moves does not imply complete mechanics coverage.
- Other Mental Herb effects.
- Protect and Spiky Shield share the protection chain and support Unseen Fist
  bypass; failure when no actions remain and additional protection variants
  are not yet represented.
- General ability-change lifecycle hooks beyond the represented Worry Seed
  transition, and further audits of public ability/item knowledge.
- Additional charge/semi-invulnerability variants, protection-independent
  move failure conditions, and additional forced-switch interactions.
- Further item binding, received-item effects and ability interactions beyond
  the existing Knock Off, Trick and Switcheroo implementations. Those moves use
  atomic item mutation helpers that update public knowledge and Choice locks.
- Full curated team construction with original held items and abilities enabled,
  controlled Showdown parity probes, and complete transformed curated rollouts.

Reference source: local Showdown data and simulator files under
`/home/mathy/pokemon-showdown` (the corpus pins revision
`23b41aa29e2ff5ff80e03f57b257b595a5852047`). Implementation work used its
`data/items.ts`, `data/abilities.ts`, and fixed-point `sim/battle.ts` modifier.

The initial multi-hit implementation uses a ten-iteration `lax.scan` for Bullet
Seed, Scale Shot, Tail Slap, Rock Blast, Surging Strikes, Triple Axel, and
Population Bomb and Beat Up. It supports the represented Skill Link, Loaded
Dice, per-hit accuracy, escalating power, and guaranteed-critical rules.
Other unimplemented effects and ability/item interactions still limit parity.
