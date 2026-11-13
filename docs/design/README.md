# Battle design notes

Use this directory for rules, examples, and decisions that guide implementation.
Start with [prototype decisions](prototype-decisions.md). Answers can be brief;
mark anything undecided or outside the first milestone. Link to a source when a
rule should match a particular Pokémon Champions format or version.

For current implementation status, start with
[curated mechanics progress](curated-mechanics-progress.md). The singles
environment and all current curated move slots are implemented; full mechanics
parity and compiled acceptance of the latest engine remain outstanding. The
prototype decisions file records the original milestone requirements.

See [implemented move effects](implemented-move-effects.md) for the current
generic construction API and remaining work. The project instructions currently
target generation 9 Scarlet/Violet mechanics.

`AGENTS.md` remains the project's instruction file. Put detailed examples and
implementation contracts here, and update the instructions if a decision changes
an existing project rule. Distinguish intended behavior from current behavior.

See [structured observations and events](observations.md) for the observation PyTree,
visibility rules, and per-step transition buffer.
