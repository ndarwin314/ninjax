"""Profile tracing/lowering/compiler memory in a fresh, externally bounded process.

Example: taskset -c 0,1 .venv/bin/python tools/profile_compile.py --stage trace
Use a shell memory limit when profiling the full step on constrained machines.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPONENT_TARGETS = ('ability', 'active', 'update', 'stats', 'power', 'hp',
                     'status', 'secondary', 'contact', 'hit', 'multi_hit',
                     'status_move', 'before_move', 'switch', 'field', 'observe', 'dispatch', 'move')


def component_result(target, env, state, action):
    """Execute one component with dynamic state, player and move-slot inputs."""
    import jax.numpy as jnp
    from ninjax import game_logic as logic
    from ninjax.battle import before_move, step_field
    from ninjax.curated_move_effects import secondary_effects
    from ninjax.damage import compute_base_power
    from ninjax.enum_types import Status
    from ninjax.held_effects import modify_stats, effective_weather
    from ninjax.move_effects import after_every_hit
    from ninjax.side import update_active_fields, take_damage_value, set_status
    battle, key = state.battle, state.key
    player, index = action[0], action[1]
    if target == 'step':
        return env.step(state, action)
    if target == 'ability':
        return battle.ability
    if target == 'active':
        return battle.active
    if target == 'update':
        return update_active_fields(battle, player, current_hp=jnp.array([17]))
    if target == 'stats':
        return modify_stats(battle, player, battle.raw_boosted_stats[player])
    if target == 'hp':
        return take_damage_value(battle, player, jnp.array([10]), True, source=1-player)
    if target == 'status':
        return set_status(key, battle, player, jnp.array([Status.SLEEP]))
    if target == 'before_move':
        return before_move(key, battle, player)
    if target == 'switch':
        return logic.swap_out(key, battle, player, (index+1) % 6)
    if target == 'field':
        return step_field(key, battle)
    if target == 'observe':
        return env.observe(state)
    if target == 'move':
        return logic.move_used(key, battle, player, index)
    if target == 'dispatch':
        return logic.dispatch_selected_move(key, battle, player, index)
    move = battle.active[player].moves[index]
    if target == 'power':
        active = battle.effective_active
        return compute_base_power(active[player], active[1-player], move,
            battle.terrain, effective_weather(battle), battle.gravity_duration > 0)
    if target == 'secondary':
        return secondary_effects(key, battle, player, move)
    if target == 'contact':
        return after_every_hit(key, battle, player, move, False)
    if target == 'hit':
        return logic.resolve_damaging_hit(key, battle, player, move)
    if target == 'multi_hit':
        return logic.resolve_multi_hit(key, battle, player, move)
    if target == 'status_move':
        return logic.do_status_move(key, battle, player, move, 0, 0)
    raise ValueError(f'Unknown component: {target}')


def graph_counts(closed):
    """Count equations once per shared subgraph, retaining primitive totals."""
    seen = set()
    counts = Counter()

    def visit(value):
        """Traverse nested control-flow and jit parameters without printing IR."""
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, 'eqns'):
            if id(value) not in seen:
                seen.add(id(value))
                for equation in value.eqns:
                    counts[equation.primitive.name] += 1
                    visit(equation.params)
        elif hasattr(value, 'jaxpr'):
            visit(value.jaxpr)

    visit(closed)
    return dict(equations=sum(counts.values()), subgraphs=len(seen),
                primitives=dict(counts.most_common(12)))


def profile(stage='trace', target='step', rss_limit_mib=0, check=False):
    """Report cold preparation phases, peak RSS and a bounded graph summary."""
    if rss_limit_mib:
        def guard():
            """Stop native compilation if actual process RAM exceeds the budget.

            A virtual-address limit also includes reserved allocator arenas and
            thread stacks. RSS provides a separate bound on actual RAM use.
            """
            while True:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                if peak > rss_limit_mib:
                    print(json.dumps(dict(phase='rss_limit_exceeded', peak_rss_mib=round(peak, 1),
                        limit_mib=rss_limit_mib)), file=sys.stderr, flush=True)
                    os._exit(137)
                time.sleep(.05)
        threading.Thread(target=guard, daemon=True).start()
    # Import after the CLI has applied affinity and memory limits, before any
    # JAX backend threads or device constants can be created.
    import jax
    import jax.numpy as jnp
    from ninjax.battle import Battle
    from ninjax.curated import load_sets, make_item_pool, sample_teams
    from ninjax.prototype import make_move
    from ninjax.enum_types import AbilityEnum, ItemEnum, Status
    pool, _ = make_item_pool(load_sets(), abilities=True)
    teams, _ = sample_teams(jax.random.fold_in(jax.random.key(0), 0), pool)
    # Keep original step/ability fixtures for historical comparisons. Other
    # components receive meaningful effects with all metadata still dynamic.
    if target not in ('step', 'ability', 'active', 'observe'):
        name = {'secondary': 'Lumina Crash', 'status_move': 'Calm Mind',
                'multi_hit': 'Population Bomb'}.get(target, 'U-turn')
        move = make_move(name)
        moves = jax.tree.map(lambda bank, value: bank.at[0, 0, 0].set(value), teams.moves, move)
        teams = teams.replace(moves=moves)
        if target == 'contact':
            teams = teams.replace(ability=teams.ability.at[1, 0, 0].set(AbilityEnum.ROUGH_SKIN),
                                  item=teams.item.at[1, 0, 0].set(ItemEnum.ROCKY_HELMET))
        if target == 'before_move':
            teams = teams.replace(status=teams.status.at[0, 0, 0].set(Status.SLEEP),
                                  sleep_counter=teams.sleep_counter.at[0, 0, 0].set(3))
    env = Battle(teams)
    state, _ = env.reset(jax.random.key(7))
    inputs = (state, jnp.array([0, 0], jnp.int32))
    flat, tree = jax.tree.flatten(inputs)

    def function(*leaves):
        """Keep abstract metadata reconstruction outside the public models."""
        state, action = jax.tree.unflatten(tree, leaves)
        result = component_result(target, env, state, action)
        return jax.tree.leaves(result)

    start = time.perf_counter()

    def report(phase, **extra):
        """Flush progress before a potentially memory-intensive next phase."""
        print(json.dumps(dict(phase=phase, target=target,
            elapsed_seconds=round(time.perf_counter()-start, 3),
            peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1),
            **extra)), flush=True)

    report('start', jax_version=jax.__version__, backend=jax.default_backend(),
           rss_limit_mib=rss_limit_mib, python_version=sys.version.split()[0],
           cpu_affinity=sorted(os.sched_getaffinity(0)), xla_flags=os.environ.get('XLA_FLAGS', ''))
    # Trace once; lower the same traced program rather than retracing for IR.
    traced = jax.jit(function).trace(*flat)
    report('traced', **graph_counts(traced.jaxpr))
    if stage == 'trace':
        return
    lowered = traced.lower()
    report('lowered')
    if stage == 'lower':
        return
    try:
        compiled = lowered.compile()
    except MemoryError:
        report('compile_memory_error')
        raise
    report('compiled')
    actual = jax.block_until_ready(compiled(*flat))
    report('executed')
    if check:
        import numpy as np
        with jax.disable_jit():
            expected = function(*flat)
        assert len(actual) == len(expected)
        for index, (a, b) in enumerate(zip(actual, expected)):
            # Jit materializes scalar Python state defaults as array outputs.
            if not hasattr(b, 'dtype'):
                b = jnp.asarray(b)
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            assert a.shape == b.shape and a.dtype == b.dtype, f'Leaf {index} metadata differs'
            if jnp.issubdtype(a.dtype, jnp.inexact):
                assert np.all(np.isfinite(a)), f'Leaf {index} contains nonfinite values'
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5, err_msg=f'Leaf {index}')
            else:
                np.testing.assert_array_equal(a, b, err_msg=f'Leaf {index}')
        report('validated', comparison='compiled_vs_eager', leaves=len(actual))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['trace', 'lower', 'compile'], default='trace')
    parser.add_argument('--target', choices=['step', *COMPONENT_TARGETS], default='step')
    parser.add_argument('--rss-limit-mib', type=int, default=0,
                        help='Stop the process if peak RSS exceeds this RAM budget (Linux MiB)')
    parser.add_argument('--virtual-limit-mib', type=int, default=0)
    parser.add_argument('--cpu-count', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Compare every compiled result leaf against eager execution')
    args = parser.parse_args()
    if min(args.rss_limit_mib, args.virtual_limit_mib, args.cpu_count) < 0:
        parser.error('Resource limits must be nonnegative')
    if args.check and args.stage != 'compile':
        parser.error('--check requires --stage compile')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.virtual_limit_mib:
        requested = args.virtual_limit_mib * 1024 * 1024
        _, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        hard = requested if old_hard == resource.RLIM_INFINITY else min(requested, old_hard)
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    if args.cpu_count:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_count])
    profile(args.stage, args.target, args.rss_limit_mib, args.check)
