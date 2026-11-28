# Team construction

Pass a [`Pokemon` PyTree](../../ninjax/pokemon.py) representing both teams to
`Battle(teams, params)`. The leading axes are `(2, 6)` in player and slot
order. Singles keeps one active slot per player. Each Pokémon contains four
move slots. Supply numeric species, item, ability, and move IDs that match the
engine vocabulary when using catalogue mechanics.
[Environment actions and selection](environment.md#team-selection-and-actions) ·
[Move fields and shapes](move-construction.md)

## Loaders

- [`teams.construction`](../../ninjax/teams/construction.py) builds individual
  moves with `make_move(name)` and projects trace-derived teams with
  `make_sandbox_teams(reference)`. Named moves outside the
  [supported-move list](../../ninjax/data/supported-moves.json) project to
  unavailable slots with zero PP.
- [`teams.curated`](../../ninjax/teams/curated.py) parses Showdown exports with
  `load_sets` and builds banks with `make_sandbox_pool`, `make_ability_pool`, or
  `make_item_pool`. The pool builders retain an omissions report. The sandbox
  uses neutral abilities and items, and the other pools enable supported
  mechanics by name.
- `sample_teams(key, pool)` returns the team PyTree and selected set indices.
  Each side samples six distinct entries. The two sides may sample the same
  entry. Callers supply the key and manage independent samples.

Team projection and omissions measure available engine features. The loaders
do not validate team legality. See the [curated team example](../../README.md#curated-random-teams)
and [coverage report](../mechanics/gen9-random-coverage.json).
