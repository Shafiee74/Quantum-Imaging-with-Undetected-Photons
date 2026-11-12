"""QIUP verification/spatial — U11 FoV and resolved spatial modes.

Physical usable FoV is the pump 1/e² intensity contour diameter, **not**
``N × pixel_pitch``. Resolved mode counts use correlation σ from
``resolved_spatial_mapping``.
"""
from __future__ import annotations

from verification.spatial.analysis import evaluate_profile, pump_waist_from_setup
from verification.spatial.definitions import DEFINITIONS_MANIFEST, PROFILES
from verification.spatial.fov import (
    analytical_gaussian_fov_diameter_um,
    usable_fov_object_plane,
)
from verification.spatial.modes import mode_area_um2, spatial_mode_metrics
from verification.spatial.runner import run_fov_spatial_suite
from verification.spatial.sweeps import run_all_sweeps

__all__ = [
    "DEFINITIONS_MANIFEST",
    "PROFILES",
    "analytical_gaussian_fov_diameter_um",
    "evaluate_profile",
    "mode_area_um2",
    "pump_waist_from_setup",
    "run_all_sweeps",
    "run_fov_spatial_suite",
    "spatial_mode_metrics",
    "usable_fov_object_plane",
]
