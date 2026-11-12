"""QIUP verification/resolution — publication-grade PSF/ESF/LSF analysis.

Intensity and visibility are separate quantity families and are never silently
treated as the same metric.
"""
from __future__ import annotations

from verification.resolution.definitions import (
    DEFAULT_GRID_N,
    DEFAULT_NBAR,
    DEFAULT_SEED,
    DEFINITIONS_MANIFEST,
    PROFILES,
    QUANTITIES,
    QUANTITY_INTENSITY,
    QUANTITY_VISIBILITY,
    TARGETS,
)
from verification.resolution.profiles import (
    ProfileFit,
    build_esf_profile,
    build_lsf_from_esf,
    build_psf_profile,
    fwhm_linear,
)
from verification.resolution.quantities import (
    intensity_map_from_result,
    quantity_maps,
    visibility_map_from_result,
)
from verification.resolution.runner import run_all_profiles, run_profile_suite

__all__ = [
    "DEFAULT_GRID_N",
    "DEFAULT_NBAR",
    "DEFAULT_SEED",
    "DEFINITIONS_MANIFEST",
    "PROFILES",
    "QUANTITIES",
    "QUANTITY_INTENSITY",
    "QUANTITY_VISIBILITY",
    "TARGETS",
    "ProfileFit",
    "build_esf_profile",
    "build_lsf_from_esf",
    "build_psf_profile",
    "fwhm_linear",
    "intensity_map_from_result",
    "quantity_maps",
    "run_all_profiles",
    "run_profile_suite",
    "visibility_map_from_result",
]
