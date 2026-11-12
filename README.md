# Ninjax

A Pokémon battle environment using JAX, Flax PyTrees, and Jumanji.

## Minimal trace-derived sandbox

From the project root, using the existing environment:

```bash
.venv/bin/python -m ninjax.prototype \
  --reference unit_tests/fixtures/showdown/battle-1.json \
  --seed 0 --output /tmp/ninjax-battle.json
```

This runs a complete singles battle through the public Jumanji interface with
the recorded Pokémon's stats and types. The policy selects a uniformly random
legal move 90% of the time and a uniformly random legal switch 10%, falling back
when one action class is unavailable. Replacements and pivots are handled as
separate decisions. A decision cap catches nonterminating runs.

This is explicitly a sandbox projection, not a replay of the original Showdown
game: items and abilities are neutralized, and unsupported move slots are
unavailable (zero PP). The output lists every omission. If no usable move remains,
Struggle is available. Supported moves and their pinned reference parameters are
listed in [prototype-moves.json](ninjax/data/prototype-moves.json); this small
catalogue comes from Showdown commit `23b41aa29e2ff5ff80e03f57b257b595a5852047`.
Supported families include plain damage, priority, pivots, single-stat boosts,
status, healing, weather, terrain, and entry hazards. Generic move effects are
described in [the design guide](docs/design/implemented-move-effects.md).

The first JIT compilation can take substantial time; subsequent decisions reuse
the compiled function. No running Showdown server is needed for this example.

## Validation

```bash
.venv/bin/python -m pytest unit_tests/test_showdown_reference.py unit_tests/test_prototype.py
```

The [reference corpus](unit_tests/fixtures/showdown/README.md) includes ten completed
Showdown games, full traces, controlled probes, and a reproduction command.
Prototype checks cover an end-to-end battle, policy sampling, explicit omissions,
and boost-stage handling. Broad move/ability/item parity and full trace replay
remain future work; the design decisions live in [docs/design](docs/design/README.md).

## Curated random teams

The Showdown exports in [curated-sets.txt](ninjax/data/curated-sets.txt) can seed
reproducible sandbox battles:

```bash
.venv/bin/python -m ninjax.prototype --curated --seed 17 --output /tmp/ninjax-curated.json
```

Pass a path after `--curated` to use another export file. Missing levels default
to 100, missing IVs to 31, missing EV stats to zero, and a missing nature to
neutral Serious. Headers may omit an item and include a gender marker. Set legality is not
validated. Each player receives six distinct set entries sampled without
replacement; opponents can share entries. The output records seed, selected set
indices, species, decisions, and every omitted move/item/ability.

This uses the existing **neutral ability/item sandbox projection**. Unsupported
moves have zero PP, and Struggle is available when needed. These battles test
team construction and environment integration, not full mechanics coverage of
the curated sets. The original parsed item and ability names remain available
for implementing and testing their mechanics later.

For Python callers, `ninjax.curated.load_sets` reads the exports,
`make_sandbox_pool` builds a Pokémon bank with an omissions report, and
`sample_teams(key, pool)` returns `(teams, set_indices)`. Sampling supports `jit`
and `vmap` and leaves the key management to the caller.

```bash
.venv/bin/python -m pytest unit_tests/test_curated.py unit_tests/test_prototype.py
```

The curated acceptance cases run complete seeded battles for seeds 0, 17 and
42. The runner shares its compiled step across generated teams.

## Routine tests and compiled validation

Routine mechanics checks run with JIT disabled, retaining their assertions
without recompiling a battle engine for each case:

```bash
.venv/bin/python -m pytest -q
```

Compiled episode, integration, and explicit `jit`/`vmap`/`scan` checks carry the
`jit` marker and are opt-in. Run this gate after engine/control-flow changes or
in a less frequent CI job:

```bash
.venv/bin/python -m pytest --run-jit -m jit -q
```

Use `--run-jit` without a marker filter to run both groups. Ordinary tests still
run eagerly in that mode; the marked checks actually compile. Eager JAX may
still compile individual array primitives, but avoids whole-engine compilation
for routine mechanics cases. Existing test assertions have been retained.

Measure engine compilation independently from test-suite scheduling in a fresh
process:

```bash
.venv/bin/python tools/benchmark_compile.py
```

The benchmark reports tracing/lowering and backend compilation separately, plus
synchronized warmed latency for a neutral single-battle step. Warmed iterations
reuse the same input state; this is a microbenchmark, not full training throughput.

## Curated ability mode

Enable the supported singles ability group while keeping item and move omissions
explicit:

```bash
.venv/bin/python -m ninjax.prototype --curated --abilities --seed 42 --output /tmp/ninjax-abilities.json
```

This mode recognizes all 30 named abilities in the current curated file, with
initial, incomplete protection support for Unseen Fist.
Unseen Fist remains omitted pending protection mechanics. Friend Guard is enabled
as a no-effect ability in singles; its doubles reduction is not validated.
See the [remaining-ability validation matrix](docs/design/remaining-ability-validation.md)
for scope and limitations. Unsupported abilities are listed in `omissions`;
an enabled ability has `ability: null` in that report. Items are still neutral.
Python callers can build these teams with `ninjax.curated.make_ability_pool`.

Add `--items` to enable the curated items with existing engine hooks, alone or
alongside `--abilities`. Unsupported items remain listed in `omissions`; an
enabled item has `item: null`. Python callers can use
`ninjax.curated.make_item_pool(sets, abilities=True)`. These hooks do not yet
cover every interaction; see the curated mechanics progress document.

Frisk stores revealed items by owner/team slot. Observations expose this through
`team.item_known`, independently of `sheet_known`; knowing an opponent's item
does not reveal its ability, nature or stats. Public consumption updates the
remembered item to none. Initial abilities also activate after team selection
when team preview is enabled. These features cover the supported engine subset,
not every move/item interaction in the complete curated collection.
