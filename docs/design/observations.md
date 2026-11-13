# Structured observations and transition events

`TimeStep.observation` is an immutable `Observation` PyTree. Leaves retain integer,
boolean, or floating-point types. Each leaf starts with the observer axis `(2,)`;
`vmap` over battles adds an outer batch axis. Side axes are always self, opponent.

- `team`: arrays `(2, 2, 6, ...)` for species, form, types, HP, status, stats,
  nature, ability, item, move IDs, PP, disabled flags, and participation.
  `species_known`, `hp_known`, `ability_known`, `item_known`, `sheet_known`, and `moves_known` mark visibility.
  Own information is available in full. Hidden opponents reveal species, HP,
  form, status, types, and used moves through the existing reveal tracking. Form
  follows species visibility; zero means no represented special form. Species
  preview reveals all species; open sheets additionally reveal ability, item,
  nature, and moves. Opponent exact HP, stats, PP, and participation are zeroed.
- `active`: active team slots `(2, 2)`, normal stages `(2, 2, 6)`, accuracy/evasion
  stages `(2, 2, 2)`, Flash Fire flags, Perish Song counters and confusion
  flags and Substitute presence `(2, 2)`. Confusion duration and remaining
  Substitute HP are hidden from observations. Charging move IDs and recharge
  flags are public `(2, 2)` leaves; their lifecycle events use details 14 and 15.
  Destiny Bond is a public boolean leaf with condition detail 16. Its direct
  retaliation uses `Cause.DESTINY_BOND`. Pending faint records remain internal.
  The `roost` boolean reports the temporary Flying-type removal; stored team
  types are unchanged and the flag clears at the end of the turn.
  `active_move_actions` counts attempts since entry and `previous_move_result`
  uses the `MoveResult` enum. Queued move categories remain private.
- `sides`: public hazards and screen/tailwind counters, each `(2, 2)`.
- `field`: turn, weather, terrain, durations, Trick Room and gravity, each `(2,)`.
- `decision`: phase, acting flag, legacy legal-action mask `(2, 15)` and team
  selection eligibility `(2, 6)`. Selection must still be nonempty per player.
  The original `extras` entries remain available.

Opponent HP is rounded up to percentage points. Unrevealed values use zero plus
a false known mask. Team and move slots currently retain the engine's ordering;
there is no reveal-order remapping or inferred ability knowledge tracker. Frisk
retains public item knowledge by owner/team slot. Public consumption and removal
update that knowledge to no item; Trick and Switcheroo update knowledge of both
exchanged items. Sleep Talk reveals the called move independently of PP usage.

## Transition events

`observation.transitions` is a `TransitionEvents` PyTree. Event arrays have shape
`(2, 128)`; `count` and `overflow` have shape `(2,)`. Read entries with `valid=True`.
Reset produces an empty buffer. Every `step` starts a fresh buffer, including
invalid and terminal steps. A turn suspended for a pivot therefore produces one
segment before replacement and another segment when resolution resumes. The
field turn counter and decision phase distinguish these segments. It is not a
rolling history of earlier turns.

Events carry `kind`, initiating `source`, `target`, active team `slot`, `detail`,
`cause`, initiating `move`, `before`, `after`, signed `hp_change`, and signed
`hp_fraction_change`. Player indices are relative to the observer; -1 means no
player. Positive HP changes heal; negative changes damage. `HP` detail is 1 for
attack damage and 0 for indirect damage/healing. The move ID can be zero for
synthetic moves or data without an assigned ID. Source identifies the initiating
action's player, not necessarily the holder of a triggered ability.

HP, fainting, status and normal stat stages are captured in their mutation
helpers, preserving separate damage, recoil, contact, healing and status effects.
HP amounts reflect clipping, Sturdy and Multiscale. Exact HP values and deltas
are only visible to the target's owner. Public HP fraction changes are differences
between rounded public endpoints, matching the observation's HP display.
Weather/terrain and other represented public condition changes are recorded at
action/replacement/field boundaries, after that boundary's primitive events.
Their before/after values describe the net change at that boundary, not every
internal callback. Conditions include hazards, screens, tailwind, Perish Song,
Flash Fire, Trick Room and gravity; confusion start/end uses condition detail
12 without exposing its duration; Substitute uses detail 13 without exposing
remaining HP. Other detail IDs are documented in `events.py`.
Move use, misses, immunities, interrupted moves and explicit Fake Out/Sucker
Punch failures also generate events. An
unsupported move follows the existing no-effect path, with MISS detail 0;
an ordinary miss has detail 1. The log does not imply additional mechanics exist.

The buffer never wraps. If capacity is exceeded it preserves the first 128 events
and sets `overflow=True`. Callers should check overflow when consuming traces.

`env.observation_spec` describes the nested PyTree using Jumanji specs. The API
replaces the former 22-feature float array; use named blocks rather than offsets.

Example access (the first axis selects the observing player):

```python
state, timestep = env.reset(key)
state, timestep = env.step(state, actions)
obs = timestep.observation
own_move_pp = obs.team.move_pp[0, 0]       # (6, 4)
legal = obs.decision.legal_actions[0]       # (15,)
events = obs.transitions
valid = events.valid[0]                    # (128,)
hp_changes = events.hp_change[0]           # exact changes for player 0 only
public_changes = events.hp_fraction_change[0]
```

`BOOST` detail indices 0..5 match normal stat indices; 6 and 7 represent accuracy
and evasion. `CRITICAL` records critical hits. Synthetic Struggle move events use
move/detail ID -1. The internal `EventBuffer` packs integer columns to keep
compiled append operations small; observers receive the unpacked named fields.
