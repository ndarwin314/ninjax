# Random-battle passive abilities and held items

This batch expands the caller-provided singles engine using existing move,
HP, stat-stage, switch, item-consumption and action-mask hooks. It adds no battle
state leaves and no dependencies. All randomness uses explicit JAX keys.

The curated projection now enables 152 ability names (149 of the 203 names in
the pinned random-battle inventory) and 35 held-item names, plus an empty item
slot. This expands the prior 30 ability/19 item projection. Projection means the
represented singles hooks are enabled; it does not establish every interaction
or full Scarlet/Violet parity. The generated
[coverage report](gen9-random-coverage.json) lists the exact remaining names.
Items in that report are sampled from 100 generated teams, not an exhaustive
inventory of the procedural item-selection code.

## Implemented and enabled

- Choice Scarf reuses Choice locking, switch cleanup and atomic item exchange.
  Its Speed modifier chains after stages with weather, Tailwind and Quick Feet;
  paralysis applies afterward. Weather suppression is respected.
- Wide Lens, Expert Belt, Light Ball, Magnet, Silk Scarf, Soul Dew and Lustrous
  Orb use existing accuracy, damage and species-specific modifier hooks.
  Species-specific items expect National Pokédex species numbers, as supplied
  by `make_sandbox_pool`; arbitrary callers must supply these identities.
- Chesto/Lum berries, White Herb, Flame/Toxic Orbs and Air Balloon use existing
  status, confusion, stat, consumption and grounding state. Unnerve prevents
  berry consumption, and deferred berries update after Unnerve leaves. Rest
  heals before Chesto/Lum wakes its user. White Herb clears all negative stages
  together after a move, rather than consuming between Shell Smash's drops.
- Hearthflame/Cornerstone masks enable the existing form-dependent power and
  transfer rules. Caller-provided Ogerpon form identity is still required.
- Passive damage/defense abilities include Sharpness, Fur Coat and Heatproof,
  plus the existing ordinary type, stat and move-flag modifiers. Move projection
  now preserves punching and wind flags. Purifying Salt and Sniper run in the
  proper damage phases instead of using superficially equivalent multipliers.
- Status and residual hooks include Natural Cure, Hydration, Shed Skin, Poison
  Heal, Speed Boost, Early Bird, Bad Dreams and Cheek Pouch. Hydration/Shed Skin
  precede poison/burn; Speed Boost/orbs occur later. Initial leads enter before
  turn zero, while normal switches retain the current turn number. Shed Skin
  uses the pinned Gen 9 probability of 33/100.
- Added/enabled reactions include Download, Justified, Rattled, Water Compaction,
  Aftermath, Wind Rider and existing contact/status reactions. Defiant and
  Competitive react after actual opposing drops, including accuracy/evasion;
  capped, prevented and self-inflicted drops do not trigger them.
- Prankster, Gale Wings and Triage alter queued priority. Dark targets block
  represented Prankster status moves. Good as Gold blocks targeted status moves
  while allowing field hazards. Arena Trap and Magnet Pull share the existing
  switch mask, including grounding, Ghost immunity and replacement exceptions.
- Friend Guard and Power Spot have no effect in singles. Enabling their names
  does not implement or validate their doubles effects.

## Reference and validation

`tools/record_passive_mechanics.cjs` uses the installed, pinned Showdown simulator
at revision `23b41aa29e2ff5ff80e03f57b257b595a5852047`. It records real damage,
stat/status, item-update, full-turn residual and move-reaction results. It does
not require HTTP or a dependency install. The HTTP server is also available,
but these controlled tests use its simulator API directly to expose exact HP.

The fixture contains 57 damage cases (all 16 possible damage rolls each),
12 Speed cases, 14 status immunity cases, eight berry cases, nine full residual
turns and seven contact/wind reactions. Tests match corresponding damage rolls
rather than assuming the JAX and Showdown PRNG streams are interchangeable.
Additional integration tests cover action masks, item transfers, replacements,
Substitute and status/ability timing.

```sh
SHOWDOWN_REVISION=23b41aa29e2ff5ff80e03f57b257b595a5852047 \
  node tools/record_passive_mechanics.cjs /home/mathy/pokemon-showdown
.venv/bin/python -m pytest -q unit_tests/test_passive_mechanics.py
.venv/bin/python -m pytest --run-jit -q \
  unit_tests/test_passive_mechanics.py::test_passive_helpers_jit_vmap_scan
.venv/bin/python -m tools.audit_random_battle_mechanics
```

Ordinary tests enable the persistent cache at `.profiling/jax-cache`. The helper
transformation check is an actual compiled `jit`/`vmap`/`scan` test, not a
compilation benchmark. Full-environment transformed rollout acceptance remains
separate, and was not rerun for this batch.

Final validation (per-run counts overlap):

| Run | Result | Elapsed |
| --- | --- | --- |
| Passive suite plus existing move-restriction and ability tests | 167 passed, 5 compiled checks deselected | 579.56 s |
| Added/updated berry, complete-residual, faint/wind and Sturdy/Unaware probes | 25 passed | 166.70 s |
| Actual compiled helper transformations, trapping masks, Whirlwind regressions and hit/finalization equivalence | 10 passed | 73.77 s |

Earlier existing held-item, item-mutation, curated-move and remaining-ability
regression cases also passed. These timings are ordinary test durations with
the persistent cache enabled, not compilation benchmark measurements.

## Items requiring additional representation

| Item | Required work |
| --- | --- |
| Eviolite | A form-aware evolution-eligibility lookup. National Pokédex numbers alone do not distinguish all eligible forms, such as Basculin's forms. |
| Draco Plate / Mind Plate | Arceus/Multitype form selection, Judgment typing, and holder/recipient-specific transfer restrictions. A type boost alone would omit those mechanics. |
| Griseous Core | Giratina-Origin form selection, stats/ability lifecycle and species-bound transfer rules. |
| Rusted Sword / Rusted Shield | Crowned form/stat/type initialization, signature-move conversion and transfer restrictions, alongside the Gen 9 once-per-battle ability lifecycle. |

## Abilities requiring more complex work

These are separate from the hooks enabled above; no unsupported ability is
silently enabled just because an enum or partial helper exists.

| Area | Examples | Missing representation or integration |
| --- | --- | --- |
| Form and identity lifecycle | Multitype, Disguise, Ice Face, Shields Down, Hunger Switch, Zero to Hero, Tera Shift, Gulp Missile, Illusion | Persistent form/identity and one-use state, form-specific stats/types/abilities, or public/private observation changes. |
| Copying and ability acquisition | Imposter, Trace, Dancer | Copy/restore contracts for moves, PP, stats and abilities, plus acquisition callbacks and called-move execution rules. An ability override alone is insufficient. |
| Per-Pokémon history | Intrepid Sword, Dauntless Shield, Battle Bond, Slow Start, Truant, Electromorphosis, Supreme Overlord, Unburden | Once-per-battle flags, active-turn/action history, charge/consumption state or entry-locked faint counts that survive switches correctly. |
| Consumed-item memory | Harvest, Cud Chew | Last-consumed berry identity and delayed reuse timers. An empty held-item slot is insufficient. |
| Attack-local ability suppression | Mold Breaker, Teravolt, Turboblaze, Mycelium Might | Event-scoped suppression, Ability Shield and unbreakable exceptions. Existing enum aliases such as Filter/Prism Armor and Multiscale/Shadow Shield must be split before this can be correct. |
| Source-sensitive routing | Mirror Armor, Magic Bounce, Corrosion, Flower Veil, Synchronize, Poison Puppeteer, Scrappy, Mind's Eye | Reliable source/effect classification, reflection without recursion, immunity exceptions and effect propagation through damage/item checks. |
| Indirect damage and direct costs | Magic Guard | Audit every HP path so recoil, contact and residual damage are prevented while Substitute costs, self-fainting and direct HP assignment still work. Applying it to every non-attack HP loss would be wrong. |
| Action/target context | Analytic, Stakeout, Pressure, Cute Charm | Opponent action/switch history, per-target PP accounting or a complete infatuation/gender eligibility model. |
| Item stealing | Magician, Pickpocket | Exact post-hit transfer timing, Substitute/contact rules, and received-item reactions. Existing atomic transfers are reusable but are not sufficient by themselves. |

Some remaining abilities can reuse existing state but still need a focused
integration/parity audit rather than new array fields:

- Moxie, Grim Neigh, Chilling Neigh, As One and Soul-Heart need correct
  source-aware faint-resolution timing. Existing HP callbacks are partial and
  must not award boosts for unrelated residual or self-KO events.
- Berserk and Anger Shell need the correct damage crossing and berry/multi-hit
  ordering; endpoint HP comparisons alone can lose intermediate threshold events.
- Cursed Body, Toxic Chain, Toxic Debris and Sheer Force need their proper
  per-hit/per-move event scheduling, failure rules and interaction checks.
- Comatose needs consistent sleep-like behavior across restrictions, damage,
  Sleep Talk and observations, while remaining distinct from an actual status.

Doubles, transformation lifecycles, every public ability/item reveal, combinations
of multiple fixed-point damage modifiers and arbitrary move interactions remain
outside the validated scope. In particular, enabling a damage modifier is not a
claim that every rounding interaction with another modifier has been audited.
