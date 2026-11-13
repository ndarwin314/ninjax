"""Opt-in, isolated component compilation checks with durable phase reports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from tools.profile_compile import COMPONENT_TARGETS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def component_reports():
    """Write all outcomes, including failures, when a report path is requested."""
    results = []
    yield results
    destination = os.environ.get('NINJAX_COMPONENT_REPORT')
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(cpu_count=2, rss_limit_mib=4096,
            virtual_limit_mib=8192, timeout_seconds=180, results=results), indent=2)+'\n')


@pytest.mark.jit
@pytest.mark.parametrize('target', COMPONENT_TARGETS)
def test_component_compiles_and_matches_eager(target, component_reports, record_property):
    """Compile, execute and compare one component in a fresh bounded process."""
    command = [sys.executable, str(ROOT/'tools/profile_compile.py'),
        '--target', target, '--stage', 'compile', '--check',
        '--rss-limit-mib', '4096', '--virtual-limit-mib', '8192', '--cpu-count', '2']
    environment = dict(os.environ, OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/ninjax-mpl')
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=180)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        code = None
        stdout, stderr = error.stdout or b'', error.stderr or b''
        stdout = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
    events = []
    for line in (stdout+'\n'+stderr).splitlines():
        if line.startswith('{'):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    phases = {event['phase']: event for event in events if 'phase' in event}
    status = ('timeout' if code is None else 'memory_limit' if 'rss_limit_exceeded' in phases
              else 'passed' if code == 0 and 'validated' in phases else 'failed')
    report = dict(target=target, status=status, returncode=code,
        wall_seconds=round(time.perf_counter()-start, 3), phases=events,
        stderr=stderr[-6000:])
    component_reports.append(report)
    record_property('component_report', json.dumps(report))
    assert status == 'passed', json.dumps(report, indent=2)
