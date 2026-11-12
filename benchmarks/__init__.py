"""QIUP benchmarks package (Task 02 / Task 07 architecture)."""
from __future__ import annotations

from benchmarks.targets import (
    DEFAULT_N,
    DEFAULT_PIXEL_UM,
    PHYSICAL_TRUTH,
    TARGET_IDS,
    BenchmarkTarget,
    build_all_targets,
    build_target,
)

__all__ = [
    "DEFAULT_N",
    "DEFAULT_PIXEL_UM",
    "PHYSICAL_TRUTH",
    "TARGET_IDS",
    "BenchmarkTarget",
    "build_all_targets",
    "build_target",
]
