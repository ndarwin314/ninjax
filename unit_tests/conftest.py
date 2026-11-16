"""Run mechanics eagerly by default; opt into compiled integration checks."""
import os
from pathlib import Path

import jax
import pytest


def pytest_addoption(parser):
    """Expose compiled validation, persistent-cache policy and diagnostics."""
    parser.addoption('--run-jit', action='store_true', default=False,
                     help='Include expensive JIT/transformation and compiled battle tests')
    parser.addoption('--no-jax-compilation-cache', action='store_true',
                     help='Disable the persistent JAX cache for this test process')
    parser.addoption('--jax-explain-cache-misses', action='store_true',
                     help='Log JAX tracing cache-miss reasons and compilation timings')


def pytest_configure(config):
    """Configure JAX before test collection creates arrays or compiles helpers."""
    enabled = not config.getoption('--no-jax-compilation-cache')
    cache_dir = str(Path(__file__).resolve().parents[1] / '.profiling' / 'jax-cache')
    # Export settings for ordinary test subprocesses. Compilation benchmarks
    # independently force caching off before constructing their fixtures.
    os.environ['JAX_ENABLE_COMPILATION_CACHE'] = str(enabled).lower()
    if enabled:
        os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir
    else:
        os.environ.pop('JAX_COMPILATION_CACHE_DIR', None)
    jax.config.update('jax_enable_compilation_cache', enabled)
    jax.config.update('jax_compilation_cache_dir', cache_dir if enabled else None)
    if config.getoption('--jax-explain-cache-misses'):
        os.environ['JAX_EXPLAIN_CACHE_MISSES'] = 'true'
        os.environ['JAX_LOG_COMPILES'] = 'true'
        jax.config.update('jax_explain_cache_misses', True)
        jax.config.update('jax_log_compiles', True)


def pytest_report_header(config):
    """Expose cache policy in test output so warm and cold runs are identifiable."""
    return (f'JAX persistent cache: {jax.config.jax_enable_compilation_cache}; '
            f'directory: {jax.config.jax_compilation_cache_dir}; '
            f'cache-miss explanations: {jax.config.jax_explain_cache_misses}')


def pytest_collection_modifyitems(config, items):
    """Deselect compiled checks unless requested, preserving normal -k/-m filters."""
    if config.getoption('--run-jit'):
        return
    selected, deferred = [], []
    for item in items:
        (deferred if item.get_closest_marker('jit') else selected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deferred)


@pytest.fixture(autouse=True)
def execution_mode(request):
    """Avoid repeatedly compiling mechanics cases; marked tests really compile."""
    compiled = request.node.get_closest_marker('jit') is not None
    with jax.disable_jit(not compiled):
        yield
