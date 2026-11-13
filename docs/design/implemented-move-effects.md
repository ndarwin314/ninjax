# Generic move construction

Caller-provided `Pokemon` objects contain `Move` structs. Scalar metadata uses
the same shape as `current_pp`: normally `(2, 6, 4, 1)` for a pair of teams.
Paired stat-index/stage fields (`boost_*`, `self_*`, `secondary_*`) accept vectors
of up to seven changes and are padded to `(2, 6, 4, 7)`. Zero stages disable
padded entries. Stat indices 1–5 name ordinary stats; 6 and 7 are accuracy and
evasion. Scalar/default field values are broadcast by the constructor.

For a generic status-category move (`move_type=MoveType.STATUS`), configure:

| Field | Meaning |
| --- | --- |
| `target_self` | Select the user instead of its opponent; false by default. |
| `status_effect` | A `Status` value to inflict; `Status.NONE` disables it. |
| `boost_stat` | Stat-index vector, paired with `boost_stages`. |
| `boost_stages` | Signed stage-change vector; zero disables an entry. |
| `weather_effect` | A `WeatherEnum` value; zero leaves weather unchanged. |
| `terrain_effect` | A `TerrainEnum` value; zero leaves terrain unchanged. |
| `hazard_effect` | 1 Stealth Rock, 2 Sticky Web, 3 Spikes, 4 Toxic Spikes; zero disables it. |
| `healing_fraction` | Fraction of the selected target's maximum HP to restore; zero disables it. |

Weather and terrain effects set a five-turn duration. Spikes and Toxic Spikes
stack to three and two layers respectively. Hazard effects apply to the selected
side, normally the opponent. These are generic primitives, not a complete move
database or complete coverage of move-specific exceptions. Secondary effects
of damaging moves use separate `secondary_*` metadata. `self_stat` and
`self_stages` describe post-hit changes to the user; `self_chance` rolls once
for that entire vector. `healing_fraction` restores a fraction of maximum HP,
whereas `drain_fraction` restores a fraction of actual attack damage dealt.

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
identify unsupported moves. `prototype.make_move` loads the limited supported
catalogue, and curated construction helpers parse Showdown exports; neither is
a complete move database or team-legality validator.

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
`information_mode` parameter. Observations are structured PyTrees with `team`,
`active`, `sides`, `field`, `decision`, and `transitions` blocks. Team leaves begin
with `(2 observers, 2 perspectives, 6 slots, ...)`; perspective zero is self.
Opponent exact HP, stats, PP and participation are masked. Known moves and active
slots accumulate during battle; open sheets additionally expose moves, abilities,
items and natures before selection. Represented ability reveals and public item
changes update persistent knowledge. Supply numeric `Pokemon.species`,
`Pokemon.item`, and `Move.identifier` IDs matching the engine vocabulary when
using built-in mechanics. See [observations and events](observations.md) for
field shapes and visibility rules.

## Results and remaining coverage

The turn-limit outcome is always a draw, including a knockout at that limit.
Recoil/contact simultaneous knockouts award the attacker; `Move.self_ko=True`
instead awards the defender. `Move.perish_effect=True` starts counters for active
Pokémon. Simultaneous final Perish Song KOs award the slower effective Pokémon,
including Trick Room reversal; tied effective speeds use a keyed random tie break.
Held-item hooks include Life Orb damage/recoil, and Perish Song checks Soundproof.
Item removal/exchange helpers support Knock Off, Trick and Switcheroo, including
public knowledge updates. All current curated move slots are enabled, but other
move/ability/item interactions remain incomplete. See
[mechanics progress](curated-mechanics-progress.md) for implementation and
validation gaps; the latest expanded engine still needs a complete compiled
acceptance pass.
