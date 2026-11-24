# Remaining mechanics

This inventory assesses the current code and the pinned Gen 9 random-battle
coverage report, not all legal Scarlet/Violet combinations. Projection means a
move or ability is enabled by the team construction path; it does not prove
complete interaction parity. An unprojected ability can already have a partial
engine hook. See [coverage data](gen9-random-coverage.json) for exact names.

## Singles priorities

1. **Complete and validate existing interactions.** Audit residual ordering,
   damage modifier rounding, protection, ability suppression, item reactions,
   switch and faint timing, and public information against the pinned Showdown
   reference. Existing focused tests and three compiled curated rollouts do not
   establish exhaustive parity. Revalidate the current working tree under JIT;
   full-environment `vmap` and `scan` rollouts remain outstanding.
2. **Expand the move catalogue.** The inventory has 232 of 348 distinct random
   battle moves projected; 116 remain unavailable. Examples requiring additional
   behavior include Future Sight and Wish (delayed effects), Outrage (move lock),
   Leech Seed (persistent residual), and Transform. Other omissions may reuse
   existing hooks but still need projection and parity tests. See
   [move design](remaining-move-design.md) and the coverage data for the list.
3. **Complete abilities and items.** 149 of 203 random-battle ability names are
   projected; 54 are not. Important missing foundations include attack-scoped
   ability suppression, ability acquisition, per-Pokémon history, consumed-item
   memory, and form changes. Six items in the 100-team sample are unprojected:
   Eviolite, Draco Plate, Mind Plate, Griseous Core, Rusted Sword and Rusted
   Shield. The sample is not an exhaustive item inventory. See
   [passive mechanics](random-battle-passives.md) for the detailed dependencies.
4. **Implement Terastallization and form lifecycles.** Action eligibility,
   once-per-battle state, type and damage rules, form-dependent stats and
   abilities, and observations need an integrated lifecycle. Existing guards or
   fields alone do not provide the full mechanic.

## Beyond the singles milestone

Doubles needs paired actions, multiple active Pokémon, targeting and spread
effects, ally interactions and doubles-specific ability behavior. Training and
opponent selection remain separate from the battle engine. Future ruleset
variants should keep these mechanics and their lifecycle rules explicit rather
than assuming every generation shares Gen 9 behavior.

## Evidence and limits

The inventory is derived from the pinned Showdown random-battle role data and
the current projection via `python -m tools.audit_random_battle_mechanics`.
Candidate role movepools are not all legal team combinations, and sampled items
are procedural. Historical test results and their scope are recorded in
[curated progress](curated-mechanics-progress.md) and
[passive mechanics](random-battle-passives.md); they are not a fresh test run.
