"""Summarize textual XLA pass dumps without loading whole modules into memory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


INSTRUCTION = re.compile(r'^\s+(?:ROOT )?%?[\w.-]+ = .*?\b([a-z][a-z0-9-]*)\(')


def summarize(path):
    """Count static instructions and computations, including cloned regions.

    Counts describe the dumped graph, not runtime execution counts or memory.
    Text size includes debug metadata, so it is only a graph-growth indicator.
    """
    operations = Counter()
    computations = 0
    with path.open() as source:
        for line in source:
            match = INSTRUCTION.match(line)
            if match:
                operations[match[1]] += 1
            elif line.startswith(('%', 'ENTRY ')) and line.rstrip().endswith('{'):
                computations += 1
    return dict(file=path.name, bytes=path.stat().st_size,
                computations=computations, instructions=sum(operations.values()),
                control_flow={op: operations[op] for op in ('call', 'conditional', 'while')},
                operations=dict(operations.most_common(10)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    reports = [summarize(path) for path in sorted(args.directory.glob('*.txt'))]
    # XLA also writes buffer-assignment and memory reports with .txt suffixes.
    print(json.dumps([report for report in reports if report['instructions']], indent=2))
