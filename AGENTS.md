# Project Instructions

## Stack
- Python
- JAX
- Jumanji
- pytest

## Development
- Use the existing project virtual environment.
- Preserve JAX compatibility with jit, vmap, and scan.
- Prefer pure functions and immutable state.
- Avoid Python-side control flow on traced JAX values.
- Use explicit PRNG keys for randomness.
- Aim for parity with gen 9 scarlet and violet mechanics
- Look towards future compatibility with slightly different versions of mechanics

## Workflow
- Inspect existing code before making changes.
- Prefer minimal, focused changes.
- Run relevant tests after modifying code.
- Do not add dependencies without approval.
- Do not modify unrelated files.

## Initial environment contract
- Battle format: Singles and doubles 6 pokemon per team.
- Team setup: Caller Provided.
- Actions: One action per player. In singles this is a move or a switch. In doubles this is a pair of move or switch
- Observations: Each player can see battle state and their own team. View of opponents team may either be none before battle, or OTS format with all info about the opponents team except EVS and IVS
- Episode ends when: When all of one players team is fainted or 100 turns pass as a failsafe.
- 
## Implementation priorities
- Prioritize implementing Single battles although aim to make it easy to add support for Doubles
- First milestone: Small working subset.
- Keep battle state as immutable JAX PyTrees, with documented array shapes.
- A single battle contains the player axis; use vmap for independent battles.
- Validate reset and step under jit, vmap, and scan.
- Treat Jumanji as the environment API and remove Gymnax after migration.
- 
## Self-play architecture
- Battle receives caller-provided teams.
- One step accepts one action for each player.
- Keep the battle engine symmetric between players.
- Use player-relative observations and legal-action masks.
- A single battle owns the player axis; vmap runs independent battles.
- Keep state and randomness explicit and compatible with jit, vmap, and scan.
- Keep policy training and opponent selection separate from battle mechanics.
