"""Explicit host-side coverage of represented singles move effects.

Catalogue metadata alone cannot describe simulator callbacks. New catalogue
entries therefore remain unavailable until their effects have been reviewed.
Support does not certify exhaustive ability/item interactions or doubles parity.
"""
import json
from pathlib import Path


SUPPORTED_MOVES = frozenset(json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'supported-moves.json').read_text()))


def is_move_supported(name):
    """Return whether the named move has a represented singles implementation."""
    return name in SUPPORTED_MOVES
