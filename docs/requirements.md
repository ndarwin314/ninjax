# Requirements

## Battle contract

- Build a Jumanji environment for Generation 9 Scarlet/Violet battle mechanics.
  Use the [pinned Pokémon Showdown reference](mechanics/showdown-reference.md)
  to validate the targeted interactions.
- Accept two caller-provided teams of six Pokémon. Optional team selection lets
  each player bring a nonempty subset before the first turn.
- Advance the environment with one action per player per decision. In singles,
  players choose a move or switch during a turn and supply a continuation,
  replacement, revival choice, or inactive no-op when the phase calls for it.
  The target doubles format uses one move or switch choice per active position.
- Expose the public battle state, each player's own team, and player-relative
  legal-action masks. Support hidden opponents, species-only preview, and open
  team sheets. Open sheets expose species, moves, abilities, items, and natures.
  Keep the opponent's exact stats, EVs, IVs, and other private state hidden.
- End the episode when a participating team has no Pokémon left. The default
  100-turn failsafe ends the episode as a draw. See the
  [environment contract](components/environment.md) for rewards and tie rules.

## Execution and validation

- Keep one battle symmetric, with a player axis of size two. Use immutable JAX
  PyTrees with documented shapes and explicit PRNG keys. Batch independent
  battles with an outer `vmap` axis and support `jit` and `scan`.
- Keep policy training and opponent selection outside battle mechanics. Keep
  ruleset-specific behavior explicit so future formats can reuse the engine.
- Validate the public `reset` and `step` paths, including action legality,
  observation visibility, phase transitions, and terminal outcomes. Compare
  mechanics to controlled cases in the [Showdown fixture corpus](../unit_tests/fixtures/showdown/README.md).
- Use the project environment with JIT and the persistent compilation cache for
  ordinary tests. Run batched and scanned cases at major checkpoints. Use fresh
  processes with caching disabled for compilation benchmarks. Follow the
  resource limits and test policies in [AGENTS.md](../AGENTS.md).

## Scope

The first milestone is a usable singles environment with representative damage,
priority, status, boosts, weather, terrain, and entry hazards. The engine already
supports that environment and the curated moves, with further interaction
validation and catalogue coverage tracked in [Tasks](tasks.md). Eligible
singles Pokémon can Terastallize into an ordinary type. Doubles, Stellar
typing and species-specific transformation lifecycles are later milestones.
