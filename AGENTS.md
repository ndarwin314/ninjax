# Ninjax project instructions

## Source of truth

Start at the [documentation index](docs/README.md). The `docs/` directory
defines the project contract, architecture, and backlog. This file defines
how to work in the repository.

| Question | Source |
| --- | --- |
| What behavior and formats are in scope? | [Requirements](docs/requirements.md) |
| Where does a change belong? | [Architecture and component guides](docs/architecture.md#component-ownership) |
| What work and validation remain? | [Tasks](docs/tasks.md) |
| What mechanics can team construction enable? | [Gen 9 coverage report](docs/mechanics/gen9-random-coverage.json) |
| Which ruleset and fixtures guide parity checks? | [Pinned Showdown reference](docs/mechanics/showdown-reference.md) and [fixture corpus](unit_tests/fixtures/showdown/README.md) |
| How were compilation measurements collected? | [Compilation results](docs/compilation/README.md) |

Inspect the relevant implementation and tests before editing. If a change
updates a contract, component responsibility, or task status, update the
corresponding page in `docs/` as part of the change. Use the coverage report
for projection and report interaction parity based on behavior checks.

## Development

- Use the existing project virtual environment. Keep changes focused, preserve
  unrelated work, and get approval before adding dependencies.
- Preserve `jit`, `vmap`, and `scan` compatibility. Use pure functions,
  immutable PyTrees with documented shapes, and explicit PRNG keys. Use JAX
  control flow for decisions that depend on traced values.
- Keep player handling symmetric. Keep policy training and opponent selection
  outside the battle engine. Prioritize singles and keep ruleset-dependent
  behavior explicit.
- Reuse shared mechanics in their owning modules. Verify changes to action,
  effect, or event ordering against the expected mechanic and relevant tests.
- Keep imports at module scope. Prefer module-level helpers over nested
  functions or lambdas where practical. Pass the inputs each helper and its
  branches need.
- Add docstrings to new and modified functions. Comment mechanic-specific
  rules and control flow where the behavior needs explanation.

## Tests and compilation

- Run relevant tests using the project environment after changing code. Ordinary
  mechanic tests run with JIT enabled. Pytest enables the persistent cache at
  `.profiling/jax-cache` by default.
- Use `pytest --run-jit` when the change needs opt-in compiled integration tests.
  Put `vmap` and `scan` tests in `unit_tests/batched`. Run
  `pytest --run-batched unit_tests/batched` at major checkpoints or on explicit
  request. Ordinary runs and `--run-jit` exclude that directory.
- Keep total test memory usage within 8 GB, including concurrent test jobs.
- Run compilation benchmarks in fresh processes with the persistent cache
  explicitly disabled, including benchmarks launched through pytest. Use
  `JAX_ENABLE_COMPILATION_CACHE=false` for standalone benchmarks that do not set
  it themselves. Pytest accepts `--no-jax-compilation-cache`.
- Record the cache setting with timing and peak memory results. When diagnosing
  cache misses, use `--jax-explain-cache-misses` with pytest or
  `tools/profile_compile.py` and retain the log with the benchmark results.

## Multi-agent workflow

The main Codex session coordinates the work and owns the integrated result. Use
the project roles from `.codex/agents/` when a task benefits from independent
investigation, implementation, or verification. Handle small, tightly coupled
changes directly. The coordinator delegates to specialists, and specialists
report back to the coordinator.

| Role | Responsibility |
| --- | --- |
| `ninjax_explorer` | Trace the affected battle path, mechanics, and existing tests. Return evidence and candidate files without editing them. |
| `ninjax_implementer` | Make the scoped change in assigned source and data files. Report changed files, behavior decisions, and checks. |
| `ninjax_tester` | Verify behavior and add focused regression coverage when needed in assigned test files. Report exact commands, outcomes, and coverage gaps. |
| `ninjax_reviewer` | Inspect the integrated diff for mechanics, JAX compatibility, and regressions. Report findings with file locations without editing them. |

1. Give each specialist a goal, expected result, and explicit file ownership.
   Assign one editor per file at a time and preserve existing worktree changes.
2. Use exploration findings to settle the expected behavior and affected paths.
   Implementation and test design can proceed in parallel once the behavior
   contract is clear and file assignments are disjoint.
3. Hand off the implementation before running dependent tests. The coordinator
   schedules JAX test and profiling jobs within the shared memory budget and
   avoids duplicate compilations.
4. The coordinator reviews the combined diff, resolves findings with the
   appropriate owner, verifies the final state, and reports results and gaps.
