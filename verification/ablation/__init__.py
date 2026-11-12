"""Noise / physical-distortion ablation study (Task 13)."""
from __future__ import annotations

from verification.ablation.definitions import (
    COMBINATIONS,
    DEFINITIONS_MANIFEST,
    FULL_MODEL_MECHANISMS,
    MECHANISMS,
)
from verification.ablation.runner import run_ablation_suite

__all__ = [
    "COMBINATIONS",
    "DEFINITIONS_MANIFEST",
    "FULL_MODEL_MECHANISMS",
    "MECHANISMS",
    "run_ablation_suite",
]
