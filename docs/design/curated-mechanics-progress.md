# Curated mechanics implementation progress

The requested full curated ruleset is **not yet implemented**. The default
curated runner still projects abilities/items away and reports unsupported moves.
Passing sandbox battle tests is not evidence of coverage for these effects.

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
- Opt-in `--curated --items` projects the 15 items with existing engine hooks;
  it can be combined with `--abilities`. Other items remain in omissions.

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
current 30-set projection has **16 distinct unavailable moves in 17 of 120
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

Focused coverage is in `unit_tests/test_curated_moves.py`, including compiled
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
preserves the Ogerpon form separately from its shared Pokédex number. Mask item
damage modifiers and transfer restrictions remain unimplemented, so the curated
Wellspring Mask is still reported as an omission.

Blizzard and Hurricane use weather-sensitive accuracy and their freeze and
confusion secondary effects. Freeze and confusion are resolved before actions;
confusion duration is private while its presence is observable.

Substitute stores switch-cleared HP and routes eligible hits to the doll;
sound, explicit bypass flags and Infiltrator bypass it. Hit attempts are
recorded with singles target coordinates for future doubles target resolution.

Electro Shot charges with a Special Attack boost, or attacks immediately in
effective rain. Hyper Beam forces a recharge turn after hitting. Both use the
continued action and switch-cleared state; charge release spends no extra PP.

The remaining unavailable moves are: Beat Up, Destiny Bond,
Explosion, Fake Out, Foul Play, High Jump Kick,
Knock Off, Memento, Roost,
Sleep Talk, Spiky Shield, Stomping Tantrum, Sucker Punch, Switcheroo,
Trick and Whirlwind.

Remaining integration and mechanics include:

- A full curated move catalogue with callbacks represented as JAX state changes.
- Beat Up's eligible-party and per-member power rules; other Mental Herb effects.
  Multi-hit recoil still needs once-per-move aggregation of actual HP lost.
- King's Rock's per-hit flinch and Protective Pads' contact protection. The
  sandbox neutralizes both items. Initial Protect and Unseen Fist support covers
  targeted moves and contact bypass, but failure when no actions remain and
  other protection moves are not yet represented.
- Durable observation knowledge for revealed abilities and item changes from
  future item-exchange/removal move implementations.
- Charge/recharge turns, Substitute, Sleep Talk, Destiny Bond,
  protection-independent move failure conditions, and forced switches.
- The Sleep Talk exception that allows a sleeping user to continue multi-hit moves.
- Item removal/exchange and the remaining variable-damage, failure-condition,
  stateful, and move-specific interactions listed above.
- Full curated team construction with original held items and abilities enabled,
  controlled Showdown parity probes, and complete transformed curated rollouts.

Reference source: local Showdown data and simulator files under
`/home/mathy/pokemon-showdown` (the corpus pins revision
`23b41aa29e2ff5ff80e03f57b257b595a5852047`). Implementation work used its
`data/items.ts`, `data/abilities.ts`, and fixed-point `sim/battle.ts` modifier.

The initial multi-hit implementation uses a ten-iteration `lax.scan` for Bullet
Seed, Scale Shot, Tail Slap, Rock Blast, Surging Strikes, Triple Axel, and
Population Bomb. It supports the represented Skill Link, Loaded Dice, per-hit
accuracy, escalating power, and guaranteed-critical rules. Beat Up remains
unavailable in the sandbox until party-specific attack power is represented.
Other unimplemented effects and ability/item interactions still limit parity.
