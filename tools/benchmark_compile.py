"""Measure cold preparation and warm latency of one neutral singles step.

Run from the repository root with the project interpreter. Each measurement
should use a fresh process. No persistent compilation cache is configured here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
from ninjax.battle import Battle
from ninjax.move import Move
from ninjax.pokemon import Pokemon
from ninjax.tree import stack


def benchmark(iterations=20):
    """Return trace/lowering, backend compilation, and synchronized warm timings."""
    if iterations < 1:
        raise ValueError('iterations must be positive')
    def four(value):
        """Create the four move slots with a single trailing feature axis."""
        return jnp.full((4, 1), value)
    moves = Move(move_type=four(0), max_pp=four(8), current_pp=four(8),
                 type=four(1), base_power=four(80), accuracy=four(1.),
                 priority=four(0), offensive_stat=four(1), defensive_stat=four(4), crit_stage=four(0))
    mon = Pokemon(type_list=jnp.array([0, 0]), moves=moves)
    team = stack([mon] * 6)
    env = Battle(stack([team, team]))
    state, _ = env.reset(jax.random.key(5))
    # Flatten arguments for lower(): the model's normalization constructors
    # expect arrays, whereas the lowering API also reconstructs ArgInfo leaves.
    flat, tree = jax.tree.flatten((state, jnp.array([0, 0], jnp.int32)))
    def step(*leaves):
        """Reconstruct dynamic inputs inside the traced function."""
        return env.step(*jax.tree.unflatten(tree, leaves))
    start = time.perf_counter()
    lowered = jax.jit(step).lower(*flat)
    traced = time.perf_counter()
    compiled = lowered.compile()
    prepared = time.perf_counter()
    jax.block_until_ready(compiled(*flat))
    warm = time.perf_counter()
    for _ in range(iterations):
        jax.block_until_ready(compiled(*flat))
    end = time.perf_counter()
    return dict(jax_version=jax.__version__, backend=jax.default_backend(),
                trace_lower_seconds=traced-start, backend_compile_seconds=prepared-traced,
                preparation_seconds=prepared-start, warm_step_ms=(end-warm)*1000/iterations,
                stablehlo_chars=len(lowered.as_text()), iterations=iterations)


def main():
    """Print one fresh-process benchmark as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.iterations), indent=2))


if __name__ == '__main__':
    main()
