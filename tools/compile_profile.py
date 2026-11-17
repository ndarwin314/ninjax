"""Compilation profiling implementation loaded after CLI resource setup.

Use tools/profile_compile.py for fresh-process profiling. Importing this module
loads JAX and the engine; the CLI configures limits and cache policy first.
"""

import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np

from ninjax.battle.environment import Battle
from ninjax.battle.field import step_field
from ninjax.battle.interruptions import before_move
from ninjax.battle.switching import swap_out
from ninjax.calculations.damage import compute_base_power
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import Status
from ninjax.moves.effects import after_every_hit, secondary_effects
from ninjax.moves.execution import dispatch_selected_move, move_used
from ninjax.moves.hit_resolution import resolve_damaging_hit
from ninjax.moves.multi_hit import resolve_multi_hit
from ninjax.moves.status import do_status_move
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.queries import effective_weather, modify_stats
from ninjax.side.status_effects import set_status
from ninjax.side.updates import update_active_fields
from ninjax.teams.construction import make_move
from ninjax.teams.curated import load_sets, make_item_pool, sample_teams


def component_result(target, env, state, action):
    """Execute one component with dynamic state, player and move-slot inputs."""
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
        return swap_out(key, battle, player, (index+1) % 6)
    if target == 'field':
        return step_field(key, battle)
    if target == 'observe':
        return env.observe(state)
    if target == 'move':
        return move_used(key, battle, player, index)
    if target == 'dispatch':
        return dispatch_selected_move(key, battle, player, index)
    move = battle.active[player].moves[index]
    if target == 'power':
        active = battle.effective_active
        return compute_base_power(active[player], active[1-player], move,
            battle.terrain, effective_weather(battle.weather.weather, battle.ability, battle.active.current_hp), battle.gravity_duration > 0)
    if target == 'secondary':
        return secondary_effects(key, battle, player, move)
    if target == 'contact':
        return after_every_hit(key, battle, player, move, False)
    if target == 'hit':
        return resolve_damaging_hit(key, battle, player, move)
    if target == 'multi_hit':
        return resolve_multi_hit(key, battle, player, move)
    if target == 'status_move':
        return do_status_move(key, battle, player, move, 0, 0)
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


def profile(stage='trace', target='step', rss_limit_mib=0, check=False, profile_dir=None,
            explain_cache_misses=False):
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
    # Benchmarks must compile even when launched from pytest or a shell with
    # persistent caching enabled. Apply before any fixture imports/compilation.
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
    jax.config.update('jax_enable_compilation_cache', False)
    jax.config.update('jax_compilation_cache_dir', None)
    if explain_cache_misses:
        jax.config.update('jax_explain_cache_misses', True)
        jax.config.update('jax_log_compiles', True)
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
           persistent_cache_enabled=jax.config.jax_enable_compilation_cache,
           persistent_cache_dir=jax.config.jax_compilation_cache_dir,
           explain_cache_misses=jax.config.jax_explain_cache_misses,
           log_compiles=jax.config.jax_log_compiles,
           rss_limit_mib=rss_limit_mib, python_version=sys.version.split()[0],
           cpu_affinity=sorted(os.sched_getaffinity(0)), xla_flags=os.environ.get('XLA_FLAGS', ''))
    def prepare():
        """Capture compiler events without retaining fixture/eager call events."""
        # Trace once; lower the same program rather than retracing for IR.
        with jax.profiler.TraceAnnotation('ninjax.trace'):
            traced = jax.jit(function).trace(*flat)
        report('traced', **graph_counts(traced.jaxpr))
        if stage == 'trace':
            return None
        with jax.profiler.TraceAnnotation('ninjax.lower'):
            lowered = traced.lower()
        report('lowered')
        if stage == 'lower':
            return None
        try:
            with jax.profiler.TraceAnnotation('ninjax.compile'):
                compiled = lowered.compile()
        except MemoryError:
            report('compile_memory_error')
            raise
        report('compiled')
        actual = jax.block_until_ready(compiled(*flat))
        report('executed')
        return actual

    capture = nullcontext()
    if profile_dir:
        options = jax.profiler.ProfileOptions()
        # Python call tracing retains millions of events during JAX staging.
        # Host compiler events and our phase annotations suffice for this job.
        options.python_tracer_level = 0
        capture = jax.profiler.trace(profile_dir, create_perfetto_trace=True,
                                     profiler_options=options)
    with capture:
        actual = prepare()
    if profile_dir:
        report('profile_written', directory=str(profile_dir))
    if stage != 'compile':
        return
    if check:
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

