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
