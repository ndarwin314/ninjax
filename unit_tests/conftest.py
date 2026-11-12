"""Run mechanics eagerly by default; opt into compiled integration checks."""
import jax
import pytest


def pytest_addoption(parser):
    """Expose the explicit compiled-validation gate."""
    parser.addoption('--run-jit', action='store_true', default=False,
                     help='Include expensive JIT/transformation and compiled battle tests')


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
