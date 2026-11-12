"""QIUP verification/monte_carlo — U13 frame-count / realization convergence.

Separates optical stochasticity, detector stochasticity, and optional
physical drift. Seeds use ``SeedSequence.spawn`` for deterministic,
independent realizations.
"""
from __future__ import annotations

from verification.monte_carlo.definitions import (
    CHANNELS,
    DEFINITIONS_MANIFEST,
    FRAME_COUNTS,
    MASTER_SEED,
    PROFILES,
)
from verification.monte_carlo.runner import run_channel_sweep, run_monte_carlo_suite
from verification.monte_carlo.seeds import realization_seed_int, seed_manifest
from verification.monte_carlo.statistics import (
    confidence_interval_mean,
    summarize_scalar_series,
)

__all__ = [
    "CHANNELS",
    "DEFINITIONS_MANIFEST",
    "FRAME_COUNTS",
    "MASTER_SEED",
    "PROFILES",
    "confidence_interval_mean",
    "realization_seed_int",
    "run_channel_sweep",
    "run_monte_carlo_suite",
    "seed_manifest",
    "summarize_scalar_series",
]
