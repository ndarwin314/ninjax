# Tasks

## Singles priorities

1. **Validate the current engine.** Re-run public singles rollouts with curated
   abilities and items enabled. At a major checkpoint, exercise `reset` and
   `step` under full-environment `vmap` and `scan`. Review residual ordering,
   damage rounding, protection, ability suppression, item reactions, switch and
   faint timing, and information visibility against the pinned reference.
   [Implementation and validation history](mechanics/curated-mechanics-progress.md)
2. **Expand move coverage through shared mechanics.** Prioritize delayed
   attacks such as Future Sight and complete the called-move lifecycle.
   Enable catalogue entries once their execution paths and focused behavior
   checks are in place.
   [Move coverage and design](mechanics/remaining-move-design.md)
3. **Complete ability and item interactions.** Validate existing hooks and
   close gaps in projection and behavior, including attack-scoped ability
   bypass, consumed-item history, and form-sensitive mechanics. The item audit
   covers a sample of procedural choices.
   [Passive mechanics](mechanics/random-battle-passives.md) ·
   [Ability validation](mechanics/remaining-ability-validation.md)
4. **Extend Terastallization.** Validate ordinary-type action eligibility,
   switching and Tera Blast interactions. Add Stellar damage rules,
   species-specific Tera forms and Tera Starstorm.
   [Transformation design notes](mechanics/remaining-move-design.md#transformations)

## Later milestones

- **Doubles.** Add two active positions per side, paired actions, target
  coordinates, spread effects, ally interactions, and doubles-specific abilities.
- **Additional rulesets.** Make format-dependent mechanics explicit where their
  lifecycle or behavior differs from Gen 9. Keep self-play training and
  opponent selection outside the battle engine.

## Coverage and performance evidence

Use the [Gen 9 coverage report](mechanics/gen9-random-coverage.json) for exact
projection counts and missing names. Projection measures availability through
team construction. Interaction parity requires separate behavior checks.
Random-battle candidate movepools and sampled held items cover the pinned
format within the limits of the [reference corpus](../unit_tests/fixtures/showdown/README.md).
See [compilation results](compilation/README.md) for historical benchmark
conditions and reproduction commands.
