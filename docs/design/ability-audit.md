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
| Flash Fire | Absorbs opposing Fire moves and activates a non-stacking Fire offensive boost that clears on switch-out. |
| Sturdy | Survives a lethal attack from full HP with one HP; does not protect against indirect damage. |
| Multiscale | Halves attack damage only at full HP; indirect damage is unaffected. |

The regression tests in `unit_tests/test_battle.py` cover these conditions,
including non-triggering cases, JIT, a vmapped speed check, and Flash Fire under
`vmap` and `scan`. Gravity's effect on Flying-type Ground immunity is covered as well.

## Follow-up findings

- Flash Fire's switch reset is implemented, but ability-loss/suppression hooks
  are not yet represented. Confusion and Transform have reserved volatile fields;
  their mechanics and Transform's original-state restoration are not implemented.
- Sturdy's separate immunity to one-hit knockout moves is not implemented.
- Clear Body's current reduction helper does not distinguish self-inflicted
  drops from opposing drops, and its Defiant/Competitive ordering needs review.
- Several move-type-changing abilities compute a modified move that is not passed
  through the rest of move execution. Their behavior needs end-to-end tests.
- Ability suppression/bypass, doubles redirection, and all ability interactions
  with hazards, Gravity, and status moves remain outside this initial audit.
