"""Predeclared U14 grid-convergence definitions (Task 10).

Physical parameters (wavelengths, waist, L, magnifications, nbar, …) stay fixed.
Only numerical sampling ``(N, object_map_pixel_um)`` changes, with
``N * object_map_pixel_um = L_domain`` held constant so the physical domain is
preserved.
"""
from __future__ import annotations

from typing import Any

PROFILES: tuple[str, ...] = ("A", "B", "C", "D")

# Relative grid sizes. Absolute N and Δx are set per profile from stock pitch
# so that L_domain = N_mid * stock_object_map_pixel_um.
GRID_LEVEL_FACTORS: tuple[tuple[str, float], ...] = (
    ("grid_level_1", 0.5),   # N = 0.5 * N_mid
    ("grid_level_2", 1.0),   # N = N_mid (baseline sampling of the domain)
    ("grid_level_3", 2.0),   # N = 2 * N_mid  → numerical reference
)
ADDITIONAL_LEVEL_FACTORS: tuple[tuple[str, float], ...] = (
    ("additional_levels/grid_level_1p5", 0.75),
)

N_MID_DEFAULT: int = 48
SEED: int = 20261010
NBAR: float = 0.01

# Metrics evaluated at each level
METRIC_KEYS: tuple[str, ...] = (
    "intensity_shape",
    "visibility",
    "phase",
    "spatial_mapping",
    "psf_esf_width",
    "reconstruction_error",
)

REFERENCE_POLICY: dict[str, Any] = {
    "spatial_mapping_sigma_object_um": {
        "kind": "analytical_reference",
        "source": "resolved_spatial_mapping (continuous physical formula)",
    },
    "spatial_mapping_signed_magnification": {
        "kind": "analytical_reference",
        "source": "resolved_spatial_mapping magnification identity",
    },
    "visibility_V_50_50": {
        "kind": "analytical_reference",
        "source": "production moment identity V=2|C12|/(N1+N2) at fixed physics",
        "note": "value may be grid-insensitive; still recorded per level",
    },
    "intensity_normalized_map": {
        "kind": "numerical_reference",
        "source": "finest grid_level_3 resampled to common coordinates",
    },
    "visibility_map": {
        "kind": "numerical_reference",
        "source": "finest grid_level_3 visibility map",
    },
    "psf_fwhm_um": {
        "kind": "numerical_reference",
        "source": "finest grid_level_3 PSF FWHM",
    },
    "esf_10_90_um": {
        "kind": "numerical_reference",
        "source": "finest grid_level_3 ESF 10-90% width",
    },
    "phase_reconstruction_rmse_rad": {
        "kind": "analytical_reference",
        "source": "known synthetic object phase (constant φ truth)",
    },
}

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "u_id": "U14",
    "domain_policy": (
        "Fix L_domain_um = N_mid * stock_object_map_pixel_um; "
        "set object_map_pixel_um = L_domain_um / N for each level. "
        "Do not change waist, L, wavelengths, magnification, or nbar."
    ),
    "grid_level_factors": {k: v for k, v in GRID_LEVEL_FACTORS},
    "additional_level_factors": {k: v for k, v in ADDITIONAL_LEVEL_FACTORS},
    "n_mid_default": N_MID_DEFAULT,
    "metrics": list(METRIC_KEYS),
    "reference_policy": REFERENCE_POLICY,
    "profiles": list(PROFILES),
    "seed": SEED,
    "nbar": NBAR,
}
