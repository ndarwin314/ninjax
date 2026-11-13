# Remaining curated ability validation

Reference: local Pokémon Showdown revision
`23b41aa29e2ff5ff80e03f57b257b595a5852047`, principally
`data/abilities.ts`, `sim/battle.ts`, and the hit-step order in
`sim/battle-actions.ts`. Tests use controlled values and source-defined
expectations; this is not full battle replay or exhaustive interaction parity.

| Ability | Singles result | Validation scope |
| --- | --- | --- |
| Orichalcum Pulse | Supported | Entry sun, 5461/4096 Attack modifier, inactive/suppressed weather |
| Hadron Engine | Supported | Entry Electric Terrain, 5461/4096 SpA modifier, inactive terrain, Choice Specs modifier chaining |
| Protosynthesis | Supported | Staged best-stat selection, activation lock, Speed/non-Speed modifiers, field loss and Booster Energy |
| Quark Drive | Supported | Same lifecycle under Electric Terrain |
| Beads of Ruin | Supported | Selected defensive stat, fixed-point rounding, holder exemption |
| Sword of Ruin | Supported | Defense reduction, holder exemption, Psyshock and Body Press stat selection |
| Vessel of Ruin | Supported | Special offense reduction, category and holder exemption |
| Libero | Supported | Actual changes only, once per switch, original-type restoration, Tera guard |
| Protean | Supported | Same Gen 9 mechanics as Libero |
| Pixilate | Supported | Normal-to-Fairy conversion and power modifier, seven source-defined exclusions, move-slot immutability |
| Moody | Supported | Distinct eligible stats, caps, exclusion of HP/accuracy/evasion, fainted holder |
| Sand Force | Supported | Rock/Ground/Steel modifier, other types unaffected, sand chip immunity |
| Shadow Tag | Supported | Ordinary switch rejection, Ghost/holder exemption, pivots, replacements, fainted holder |
| Storm Drain | Supported | Water immunity and one boost per intercepted move, before accuracy; self-targeted Water moves unaffected |
| Water Absorb | Supported | Quarter-max-HP healing instead of Water damage, before accuracy; self-targeted Water moves unaffected |
| Friend Guard | Supported in singles | No effect without another active ally. Doubles reduction remains unvalidated because doubles is not implemented |
| Unseen Fist | Supported for represented protection | Contact bypass of Protect and Spiky Shield; other protection variants remain outside coverage |

`unit_tests/test_remaining_abilities.py` tracks the earlier ability audit;
`unit_tests/test_protect.py` and `unit_tests/test_spiky_shield.py` cover the
subsequent protection work. Separately marked compiled checks exercise dynamic
ability hooks under jit/vmap/scan and the curated ability pool through reset/step.
Their existence does not establish a completed compiled acceptance run for the
latest engine changes.

Validation exposed and corrected same-type Protean/Libero activation consumption,
Ruin and Sand Force rounding, defending Sword of Ruin against Body Press, Sand
Force weather immunity, Pixilate's type-determining move exclusions, fainted
Shadow Tag trapping, absorption occurring after accuracy, and rounding stat
modifiers separately instead of chaining them before final rounding.

The opt-in curated ability mode recognizes all 30 named abilities in the current
set file. All curated move slots and item names are enabled in their respective
opt-in modes, while unsupported custom entries remain projected and reported.
Friend Guard's doubles behavior, additional protection
variants, uncatalogued move callbacks and deferred transformation mechanics
remain outside the supported singles scope above.
