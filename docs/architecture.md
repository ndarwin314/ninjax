# Architecture

A caller builds teams, passes them to `Battle(teams, params)`, and calls
`reset(key)` followed by `step(state, actions)`. The environment returns
`EpisodeState` and a Jumanji time step. `EpisodeState` owns the PRNG key,
decision phase, and pending turn. Its `BattleState` owns the team arrays,
active slots, conditions, mechanic state, and transition buffer. The player
dimension is internal to a single battle. Batching adds a separate outer axis.

## Decision flow

1. The environment validates both players' choices against the current phase
   and legal-action masks.
2. The scheduler orders committed choices by priority and effective Speed,
   binding each action to its original active slot. Actors that faint or leave
   their slots lose their queued actions.
3. Actions run switch or move mechanics. A pivot or revival request pauses the
   queue for a separate decision. Resolution resumes from the saved cursor.
4. The field phase applies end-of-turn effects once the queued actions finish.
   The environment exposes any required replacements, then builds the next
   player-relative observation and determines the episode outcome.

## Component ownership

| Component | Owns | Details |
| --- | --- | --- |
| Battle environment | Jumanji API, phase-specific legal actions, outcomes | [Environment contract](components/environment.md) |
| Turn scheduler and field | Queued actions, switch requests, end-of-turn effects | [Battle phases](components/battle-phases.md) |
| Moves | Metadata, move selection, hit routing, execution, per-hit and per-move effects | [Move mechanics](components/moves-package.md), [dispatch flow](components/move-dispatch-flow.md) |
| Side state | State reads and writes, HP and status reactions, item and form changes | [Side state and mechanics](components/side-package.md) |
| Teams | Caller team construction, catalogue projection, curated set loading | [Team construction](components/teams.md), [move metadata](components/move-construction.md) |
| Observations and records | Relative views, reveal tracking, per-step public events, per-hit records | [Observations and events](components/observations.md) |
| Calculations and enums | Numerical rules and stable, traceable identifiers | [Package ownership](components/package-organization.md) |

Dependencies flow from the environment through the scheduler and concrete
mechanics into state helpers, records, calculations, and enums. Keep reusable
calculations independent of scheduling. Keep host-side catalogue parsing and
team setup outside the traced battle step. See
[compilation results](compilation/README.md) for measured effects of past
changes to those boundaries.
