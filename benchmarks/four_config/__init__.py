"""QIUP benchmarks/four_config package (Task 15 matched A/B/C/D benchmark)."""
from __future__ import annotations

from benchmarks.four_config.definitions import (
    ARCHITECTURE_LABELS,
    BENCHMARK_TARGETS,
    DEFINITIONS_MANIFEST,
    PROFILES,
)
from benchmarks.four_config.runner import run_four_config_suite

__all__ = [
    "ARCHITECTURE_LABELS",
    "BENCHMARK_TARGETS",
    "DEFINITIONS_MANIFEST",
    "PROFILES",
    "run_four_config_suite",
]
