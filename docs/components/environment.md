# Battle environment

[`Battle`](../../ninjax/battle/environment.py) is the Jumanji entry point.
Construct it with caller-provided teams and optional `BattleParams`. Call
`reset(key)` for the first decision, then `step(state, action)` for each
subsequent decision. Both return an episode state and a Jumanji time step.

## Team selection and actions

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
During `Phase.REPLACEMENT`, a switch action selects an eligible replacement.
During `Phase.REVIVAL`, its index instead selects a fainted participating party
member to revive. `timestep.extras["acting_player_mask"]` identifies who is
choosing. Invalid actions from an acting player leave the decision unchanged
and set `timestep.extras["invalid_action"]`. See [battle phases](battle-phases.md)
for effect ordering.

Legacy integer inputs and the 15-column legal-action mask remain supported;
`BattleAction.encode()` maps an indexed action to its mask column. PP and
represented move restrictions determine usability. Struggle consumes no move
PP and applies fixed HP recoil. Typed attacks may request `Transformation.TERA`
when `timestep.extras["tera_action_mask"]` permits the attack column. The same
mask appears in `observation.decision.tera_actions`. A side can use its
allowance once. The scheduler resolves declarations after switches and before
moves. Ordinary Tera types are supported for eligible singles Pokémon.
Legacy integer actions always use `NONE`.

Choose `InformationMode.HIDDEN`, `SPECIES`, or `OPEN_SHEET` using the
`information_mode` parameter. Observations are structured PyTrees with `team`,
`active`, `sides`, `field`, `decision`, and `transitions` blocks. Team leaves begin
with `(2 observers, 2 perspectives, 6 slots, ...)`; perspective zero is self.
Opponent exact HP, stats, PP and participation are masked. Known moves and active
slots accumulate during battle; open sheets additionally expose moves, abilities,
items and natures before selection. Represented ability reveals and public item
changes update persistent knowledge. See [observations and events](observations.md)
for field shapes and visibility rules.

## Results and remaining coverage

The default turn limit is 100. Reaching it truncates the episode as a draw,
including a knockout at that limit. A decisive result gives rewards of +1 and
−1 in player order for the winner and loser respectively. A draw gives both
players zero.
Recoil/contact simultaneous knockouts award the attacker; `Move.self_ko=True`
instead awards the defender. `Move.perish_effect=True` starts counters for active
Pokémon. Simultaneous final Perish Song KOs award the slower effective Pokémon,
including Trick Room reversal; tied effective speeds use a keyed random tie break.
Held-item hooks include Life Orb damage/recoil, and Perish Song checks Soundproof.
Item removal/exchange helpers support Knock Off, Trick and Switcheroo, including
public knowledge updates. All current curated move slots are enabled, but other
move/ability/item interactions remain incomplete. See
[mechanics progress](../mechanics/curated-mechanics-progress.md) for implementation and
validation gaps; the latest expanded engine still needs a complete compiled
acceptance pass.
