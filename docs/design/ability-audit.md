# Ability audit

Reference: the local Pokémon Showdown checkout at commit
`23b41aa29e2ff5ff80e03f57b257b595a5852047`, particularly
`data/abilities.ts`. These are focused checks, not complete conformance claims.

| Abilities | Checked behavior |
| --- | --- |
| Chlorophyll and other weather/terrain speed abilities | Apply the speed multiplier to the holder only. |
| Marvel Scale | Boost the statused holder's Defense, not Special Defense. |
| Levitate | Block single-target Ground attacks; Gravity suppresses that immunity. |
| Tinted Lens, Filter, Neuroforce | Apply only to the appropriate resistance or weakness; Neuroforce belongs to the attacker. |
| Storm Drain, Lightning Rod | The target absorbs the matching move and gains Special Attack; the attacker's ability does not redirect its own move. |

The regression tests in `unit_tests/test_battle.py` cover these conditions,
including non-triggering cases, JIT, and a vmapped speed check. Gravity's effect
on Flying-type Ground immunity is covered as well.

## Follow-up findings

- Flash Fire absorbs Fire moves but its accumulated Fire damage boost is not
  represented in state yet.
- Sturdy's current HP clamp does not ensure that a Pokémon at full HP survives
  an otherwise lethal attack with one HP.
- Clear Body's current reduction helper does not distinguish self-inflicted
  drops from opposing drops, and its Defiant/Competitive ordering needs review.
- Several move-type-changing abilities compute a modified move that is not passed
  through the rest of move execution. Their behavior needs end-to-end tests.
- Ability suppression/bypass, doubles redirection, and all ability interactions
  with hazards, Gravity, and status moves remain outside this initial audit.
