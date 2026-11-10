# Showdown reference corpus

Ten completed `[Gen 9] Random Battle` games from checkout
`23b41aa29e2ff5ff80e03f57b257b595a5852047`, using its existing compiled simulator.
The games cover 388 turns and 490 decision points. See `manifest.json` for each
result. This corpus uses Showdown's rules, not Ninjax's 100-turn cutoff.

## Capture and policy

`tools/record_showdown.cjs` uses the same local simulator engine behind
`simulate-battle`; it does not connect to a public server or change the checkout.
It generates two random teams per game and controls both sides. Team, battle,
and policy RNG seeds are recorded separately. When both action classes are legal,
it selects a move with probability 0.9 or a switch with probability 0.1, then
samples uniformly from that class's legal request entries. When a class is
unavailable it uses the other. Forced replacements use uniform living targets;
Revival Blessing uses uniform fainted targets and is recorded separately. No
Terastallization or other optional transformation is requested.

There are 703 move selections, 73 voluntary switch selections, and 101 forced
switches. Actual move events are fewer than selections because fainting,
immobilization, and other battle events can prevent a selected move from executing.

Each JSON file contains generated teams, initial stats, every decision with legal
candidate lists and policy rolls, before/after snapshots, and omniscient events.
`.log` files provide readable protocol traces. Private exact HP messages are
retained; their alternative public percentage messages are removed.

Regenerate from the project root, with an already-built matching checkout:

```bash
SHOWDOWN_REVISION=$(git -C /home/mathy/pokemon-showdown rev-parse HEAD) \
  node tools/record_showdown.cjs /home/mathy/pokemon-showdown
```

The run fails on rejected choices or if a game exceeds 5,000 decision points.
No game in this corpus hit that guard. The source commit is recorded; rebuild
Showdown yourself when changing revisions so its compiled artifacts match.

## Starting mechanics suite

`unit_tests/test_showdown_reference.py` runs offline without Node or Showdown:

- Checks corpus completion and the recorded action-class policy.
- Compares Ninjax's computed stats with all 120 generated Pokémon.
- Compares 3,144 type-chart probes using move types and roster types seen in
  these games, including immunities. These are chart probes, not full hit replay.
- Checks 174 controlled damage probes parameterized by observed move powers and
  categories. The Showdown oracle disables abilities, items, boosts, field
  conditions, critical hits, STAB, type effectiveness, and damage variation;
  special move callbacks are replaced by a generic attack. This isolates base
  damage arithmetic without requiring matching RNG implementations.
- Reconstructs 16 observed burn/ordinary-poison/sand residual damage events and
  compares the fractional HP helper, including truncation and lethal clipping.

These tests exposed and now cover level-factor rounding, incorrect defensive
stat selection, and fractional-damage rounding. The full random battles are
reference material, not yet complete Ninjax replays. They include 139 distinct
moves and many unimplemented interactions: items, ability events, multi-hit
attacks, protection, recovery, volatile conditions, and complex move effects.
Passing the starting suite does not establish parity for those interactions.
