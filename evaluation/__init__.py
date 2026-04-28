"""Evaluation helpers."""

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from fer_d5.evaluation.metrics import compute_metrics

__all__ = ["compute_metrics"]
