"""Suite runner for Task 09 FoV / spatial-mode analysis."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from verification.spatial.analysis import evaluate_profile
from verification.spatial.definitions import DEFINITIONS_MANIFEST, PROFILES
from verification.spatial.sweeps import run_all_sweeps


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_fov_spatial_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    array_n: int = 64,
) -> dict[str, Any]:
    results = {
        "utc": utc_now(),
        "definitions": DEFINITIONS_MANIFEST,
        "array_n_for_sampling_diagnostic_only": int(array_n),
        "profiles": {},
        "sweeps": run_all_sweeps(profiles),
    }
    for p in profiles:
        results["profiles"][p] = evaluate_profile(p, array_n=array_n)
    return results
