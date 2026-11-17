"""Launch fresh-process compilation profiling after applying resource limits.

Example: taskset -c 0,1 .venv/bin/python tools/profile_compile.py --stage trace
Use a shell memory limit when profiling the full step on constrained machines.
The implementation lives in tools.compile_profile; constants remain lightweight
so pytest can collect opt-in component checks without loading the engine here.
"""
import argparse
import os
import resource
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPONENT_TARGETS = ('ability', 'active', 'update', 'stats', 'power', 'hp',
                     'status', 'secondary', 'contact', 'hit', 'multi_hit',
                     'status_move', 'before_move', 'switch', 'field', 'observe', 'dispatch', 'move')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['trace', 'lower', 'compile'], default='trace')
    parser.add_argument('--target', choices=['step', *COMPONENT_TARGETS], default='step')
    parser.add_argument('--rss-limit-mib', type=int, default=0,
                        help='Stop the process if peak RSS exceeds this RAM budget (Linux MiB)')
    parser.add_argument('--virtual-limit-mib', type=int, default=0)
    parser.add_argument('--cpu-count', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Compare every compiled result leaf against eager execution')
    parser.add_argument('--profile-dir', help='Write an XProf/Perfetto compiler timeline, excluding fixture/eager work')
    parser.add_argument('--jax-explain-cache-misses', action='store_true',
                        help='Log JAX tracing cache-miss reasons and compilation timings')
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
    # Configure cold-cache policy and process limits before loading any JAX
    # backend or engine constants. The implementation uses module-scope imports.
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
    from tools.compile_profile import profile

    profile(args.stage, args.target, args.rss_limit_mib, args.check, args.profile_dir,
            args.jax_explain_cache_misses)
