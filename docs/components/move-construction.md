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
identify unsupported moves. `teams.construction.make_move` loads the limited supported
catalogue, and curated construction helpers parse Showdown exports; neither is
a complete move database or team-legality validator.

See [team construction](teams.md) for named move projection and team loaders.
