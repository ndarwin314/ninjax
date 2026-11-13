# Curated mechanics implementation progress

The initial singles environment is implemented, and all 120 move slots in the
30 curated sets now have supported implementations. Full Scarlet/Violet
interaction parity is **not yet established**. The default curated runner keeps
abilities/items neutral; opt-in modes enable all 30 curated ability names and
all 19 curated item names. Unsupported items in custom collections remain
explicit omissions. Moves outside the supported catalogue are still reported
and made unavailable.

Focused mechanics and environment tests passed during implementation, including
the Beat Up, Sleep Talk, Whirlwind and final curated item additions. Compiled acceptance
checks exist, but a complete acceptance pass for the latest expanded engine
under `jit`, `vmap` and `scan` remains outstanding. Earlier compiled checks do
not establish coverage of subsequent changes. The initial milestone requires a
usable singles Jumanji environment; doubles, transformations and self-play
training infrastructure are outside that milestone.

The 2026-11-13 compiled full-battle retry used curated seed 0 with abilities and
items enabled, two CPU cores and a 6 GiB process virtual-memory limit. It failed
with `MemoryError: std::bad_alloc` while preparing the first compiled step
(164.72 seconds in pytest). No compiled battle completed in that attempt.
Reducing the whole-step compilation memory requirement remains a validation
blocker; passing focused or eager tests does not clear this gate.

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
1,801 to 619 MiB. Whole-step backend compilation still exceeds the tested
memory budgets. Focused regression and shared-helper transformation checks
cover the optimizations; the earlier eager rollout does not validate them.
Isolated component compilation checks now pass for 17 of 18 components,
including selected-move dispatch. Full move execution exceeds the 4 GiB RSS
watchdog during backend compilation. The compilation-performance document
records phase timings, memory checkpoints and compiled/eager comparisons;
these checks do not clear full-battle acceptance.

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
`unit_tests/test_held_effects.py`. Numeric modifiers added in `held_effects.py`
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
- Other Mental Herb effects and multi-hit recoil's once-per-move aggregation
  of actual HP lost.
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
