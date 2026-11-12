"""SPDC gain-regime benchmark (Task 14)."""
from __future__ import annotations

from verification.gain_regimes.definitions import DEFINITIONS_MANIFEST, R_SWEEP
from verification.gain_regimes.runner import run_gain_regime_suite

__all__ = ["DEFINITIONS_MANIFEST", "R_SWEEP", "run_gain_regime_suite"]
