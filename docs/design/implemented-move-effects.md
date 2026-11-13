# Generic move construction

Caller-provided `Pokemon` objects contain `Move` structs. Each effect field uses
the same shape as `current_pp`: normally `(2, 6, 4, 1)` for a pair of teams.
Scalar/default field values are broadcast by the constructor.

For a generic status-category move (`move_type=MoveType.STATUS`), configure:

| Field | Meaning |
| --- | --- |
| `target_self` | Select the user instead of its opponent; false by default. |
| `status_effect` | A `Status` value to inflict; `Status.NONE` disables it. |
| `boost_stat` | Stat index 1–5; -1 disables the stage effect. |
| `boost_stages` | Signed stage change; positive boosts and negative reductions. |
| `weather_effect` | A `WeatherEnum` value; zero leaves weather unchanged. |
| `terrain_effect` | A `TerrainEnum` value; zero leaves terrain unchanged. |
| `hazard_effect` | 1 Stealth Rock, 2 Sticky Web, 3 Spikes, 4 Toxic Spikes; zero disables it. |
| `healing_fraction` | Fraction of the selected target's maximum HP to restore; zero disables it. |

Weather and terrain effects set a five-turn duration. Spikes and Toxic Spikes
stack to three and two layers respectively. Hazard effects apply to the selected
side, normally the opponent. These are generic primitives, not a complete move
database or complete coverage of move-specific exceptions. They currently apply
to status-category moves, not secondary effects of damaging moves.

For a runnable, explicitly limited trace-derived sandbox, see the
[prototype command](../../README.md#minimal-trace-derived-sandbox). It preserves
recorded stats/types while reporting unsupported moves and neutralized items and
abilities. This is separate from claiming parity for the full captured games.

For example, configure `boost_stat=StatEnum.ATTACK`, `boost_stages=2`,
`target_self=True`, and `move_type=MoveType.STATUS` for a generic self Attack
boost. A burn-inflicting move instead uses `status_effect=Status.BURN` and the
default opponent target. Set the remaining effect fields to their defaults.

Set `implemented=False` when a move's effect is unsupported. Such moves consume
PP when attempted but do not execute their hit effects. Caller metadata must
identify unsupported moves: there is no species/move database loader yet.

## Environment interfaces

Enable selection with `BattleParams(team_preview=True)`. Reset then returns
`Phase.TEAM_SELECTION` with both players acting. Pass a `BattleAction` with a
nonempty boolean `selection` row per player, shape `(2, 6)`. The first selected
slot leads; unselected slots remain in storage but cannot switch in or count
toward victory. Selection does not consume a battle turn. With preview disabled
(the compatibility default), all six participate and slot zero leads.

`BattleAction` is an immutable PyTree with integer `index` and `move_type`
leaves of shape `(2,)`, plus `transformation` `(2,)` and the preview
`selection` payload `(2, 6)`. `Transformation` contains `NONE` (default),
`MEGA`, `Z_MOVE`, `DYNAMAX`, and `TERA`; these choices are mutually exclusive.
`ActionType` contains `ATTACK`, `SWITCH`, `CONTINUED`, and `NO_OP`.
Attack indices 0..3 select move slots; attack index 4 selects forced Struggle.
Switch indices 0..5 select team slots. No-op uses index zero for inactive
players. Continued uses index zero (legacy column 5) and is the only legal
choice during charging or recharging. Electro Shot releases without another
PP cost or boost; Hyper Beam spends its recharge turn before status checks.
The action spec describes these typed leaves.
Legacy integer inputs and the 15-column legal-action mask remain supported;
`BattleAction.encode()` maps an indexed action to its mask column. PP and
represented move restrictions determine usability. Struggle consumes no move
PP and applies fixed HP recoil. Non-`NONE` transformations are represented but
rejected until their eligibility and battle lifecycle rules are implemented;
legacy integer inputs default to `NONE`.

Choose `InformationMode.HIDDEN`, `SPECIES`, or `OPEN_SHEET` using the
`information_mode` parameter. Observations have shape `(2 players, 2 perspectives,
6 slots, 22 features)`. Perspective zero is the observer's team, one its opponent.
Features are species ID, HP fraction, status, ability, item, nature increased and
decreased stat, six stats, four move IDs, and five visibility flags (species,
HP/status, ability, item, nature). Opponent stats are always masked. Used move IDs
and visible active slots accumulate during battle; open sheets expose move IDs,
abilities, items, and natures before selection. Supply `Pokemon.species`,
`Pokemon.item`, and `Move.identifier` numeric IDs; their vocabulary is caller-owned.
Ability/item activation reveals and rich move-feature observations remain future work.

## Results and remaining coverage

The turn-limit outcome is always a draw, including a knockout at that limit.
Recoil/contact simultaneous knockouts award the attacker; `Move.self_ko=True`
instead awards the defender. `Move.perish_effect=True` starts counters for active
Pokémon. Simultaneous final Perish Song KOs award the slower effective Pokémon,
including Trick Room reversal; tied effective speeds use a keyed random tie break.
Items have observation metadata but no general item-effect engine yet, so Life
Orb damage itself remains unimplemented. Soundproof and other move-specific
Perish Song exceptions also need coverage. These flags are generic primitives,
not a complete catalogue of moves or abilities.
