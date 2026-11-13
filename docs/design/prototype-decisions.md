# Working prototype: decisions and examples

These questions supplement the project instructions. The current implementation
is still a prototype with incomplete battle mechanics. The current observation
contract is documented in [observations and events](observations.md). The prompts
and answers below record the original milestone decisions; see
[mechanics progress](curated-mechanics-progress.md) for current implementation
and validation status.

## Decisions with the most impact

1. **Singles team selection:** Each player provides six Pokémon. Do all six
   participate, or does each player select a subset before the battle? If a
   subset is selected, when and how is the choice made?

   Answer: Allow a subset to be selected. First give both players any relevant information corresponding to below.
   Then, make a choice of which subset to bring, the subset must not be empty.

2. **Opponent information:** For the first milestone, should teams be hidden
   before battle or use open team sheets? In each mode, which information becomes
   visible during play (team members, moves, abilities, items, HP, and stats)?

   Answer: The end goal will be to have several different modes. 
   Prioritize fully hidden information, both players start with no information about the opponent. 
   This corresponds to random battles on Smogon.
   Second mode, this corresponds to more standard formats on Smogon like OU.
   In this mode you can see the species of pokemon your opponent has and no other information.
   Third mode, OTS (open team sheet) corresponding to VGC, in this mode you can see the species of pokemon, moves, abilities, items, and natures, but not the stats


3. **Mechanics in scope:** Which small set of moves, effects, abilities, items,
   and conditions must work before the prototype is useful? Are representative
   generic damaging moves and switches sufficient initially? How should the
   environment handle an unsupported move or effect?

   Answer: Initial prototype should have generic damaging moves, moves with modified priority, status inflicting moves, boosting moves.
   Important effects to have working will be weather, terrain, and entry hazards.
   Treat unimplemented moves as no-ops.

4. **Teams and data:** What format will callers use to provide teams? Please
   provide an example team or a schema, including species/stats, moves, abilities,
   items, levels, and any team-selection information. Is there a preferred data
   source or rules version?

   Answer: Callers should provide a `Pokemon` object and in the future, helpers can be made to streamline construction

5. **End results:** Is a 100-turn timeout always a draw? How should simultaneous
   knockouts be resolved? Should rewards be win/loss/draw only, or include any
   intermediate reward?

   Answer: 100-turn timeout should always be a draw.
   Simultaneous knockouts happen in three main ways, recoil from a move, or a move that KOs the user, or perish song.
   For recoil, either from the move itself, life orb, or contact damage, the attacker wins.
   For self KOing moves like explosion, the attacker loses.
   If all pokemon remaining faint due to perish song, the player with the slowest pokemon, counting all speed modifications including trick room, wins.

## Decision points and action rules

The current interface uses a `BattleAction` PyTree with `(2,)` `index`,
`move_type` and `transformation` leaves, plus `(2, 6)` preview selection.
`ActionType` distinguishes attacks, switches, continuations and inactive no-ops.
Attack index 4 is forced Struggle; continued actions release Electro Shot or
spend Hyper Beam's recharge turn without another PP cost. Transformations are
represented but non-`NONE` requests are rejected. Legacy mask columns 0–3 are
moves, 4 Struggle, 5 continuation, 8–13 switches and 14 no-op; 6–7 are reserved.
The environment exposes an acting-player mask and pauses for replacements.
Invalid active-player actions leave state unchanged and set an extras flag.
See [environment interfaces](implemented-move-effects.md#environment-interfaces).

The following questions and answers are the original design discussion:

- Confirm or change the action encoding and invalid-action behavior.
- What happens when no move has PP and no switch is legal?
- When a Pokémon faints or switches out during a turn, give examples of when a
  replacement is selected and what happens to each queued action. Include a
  replacement that faints on entry and simultaneous replacements if relevant.
- Should the first milestone support Terastallization or defer it?

Answers and examples: The action encoding should likely be changed to something as follows
Move encoding, switch encoding, and flags to Terastallization and Mega-Evolution.
I think one-hot encoding makes sense for the moves and actions.
If a player has no PP and cannot switch, they are forced to use struggle. 
Similarly, they can be forced to struggle if they are encored into a move with no PP or is disabled.
If a pokemon faints in the middle of a turn, the opponents queued action, and end of turn effects should be processed first, then a replacement can be sent out
If a pokemon uses a switching move, i.e u-turn or parting shot, the player immediately selects a replacement and the opponents action is deferred if it has not taken place.
If a pokemon faints on entry, it does not active any on switch-in effect such as weather setting abilities, no player selects a move, then end of turn effects take place.
Terastallization is not needed for initial prototype

## Prototype acceptance

Describe an end-to-end battle that should run successfully, including the teams,
actions, expected observations, result, and any required edge cases. The project
instructions already require `jit`, `vmap`, and `scan` compatibility. Say whether
the first milestone also needs a self-play runner or only a usable Jumanji
environment.

Answer: For initial prototype, a usable Jumanji environment is enough.

