"""Predeclared U11 FoV and spatial-mode definitions (Task 09).

All conventions below are fixed **before** interpreting numerical results.
Physical FoV is **not** ``N × pixel_pitch`` (array window ≠ usable field).
"""
from __future__ import annotations

from typing import Any

PROFILES: tuple[str, ...] = ("A", "B", "C", "D")

# ---------------------------------------------------------------------------
# Usable field of view
# ---------------------------------------------------------------------------
# Plane: declared object / crystal plane (same as pump sampling plane).
# Units: micrometres (µm) for lengths; µm² for areas.
#
# Threshold/convention:
#   Pump intensity I_p(r) is usable where
#       I_p(r) / I_p,max  ≥  τ_FoV
#   with τ_FoV = e^{-2}  (1/e² intensity contour).
#   For a Gaussian pump I∝exp(−2 r²/w_p²) this contour is exactly r = w_p,
#   so the usable FoV diameter is
#       D_FoV = 2 w_p
#   independently of the discrete array size.
#
# Contour measurement (non-Gaussian / discrete check):
#   Sample I_p on an analysis mesh spanning ±FOV_MESH_RADIUS_OVER_WAIST·w_p
#   with pitch ≪ w_p (not the detector pitch), then measure the equivalent
#   diameter of the τ_FoV superlevel set. If the mesh truncates the contour,
#   report truncated=True rather than substituting N·Δx.
#
FOV_INTENSITY_THRESHOLD: float = float(__import__("math").e ** (-2))
FOV_THRESHOLD_NAME: str = "one_over_e_squared_intensity"
FOV_PLANE: str = "declared_object_crystal_plane"
FOV_LENGTH_UNIT: str = "um"
FOV_AREA_UNIT: str = "um^2"
FOV_MESH_RADIUS_OVER_WAIST: float = 3.0
FOV_MESH_SAMPLES_PER_WAIST: float = 40.0  # analysis pitch = w_p / 40

# ---------------------------------------------------------------------------
# Resolved spatial mode
# ---------------------------------------------------------------------------
# Definition (object plane):
#   Correlation width σ_o = resolved_spatial_mapping(...).correlation_sigma_object_um
#   One resolved spatial mode is modelled as a disk of radius σ_o:
#       A_mode = π σ_o²
#   Number of resolved modes in the usable FoV:
#       N_modes = A_FoV / A_mode
#   with A_FoV = π (D_FoV/2)².
#   Linear mode count along one diameter:
#       N_modes_1d = D_FoV / (2 σ_o)
#
# Camera-plane companions use σ_c and D_FoV,cam = |M| · D_FoV,obj.
#
MODE_RADIUS_IN_SIGMA: float = 1.0
MODE_AREA_NAME: str = "disk_radius_one_correlation_sigma"
MODE_COUNT_NAME: str = "usable_FoV_area_over_mode_area"

# Detector sampling (not FoV):
#   Report array window L_window = N · object_map_pixel_um as a *sampling*
#   diagnostic only. Usable FoV remains the pump 1/e² diameter.
#
SAMPLING_WINDOW_LABEL: str = "array_sampling_window_not_FoV"

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "u_id": "U11",
    "usable_field_of_view": {
        "plane": FOV_PLANE,
        "units_length": FOV_LENGTH_UNIT,
        "units_area": FOV_AREA_UNIT,
        "threshold": FOV_INTENSITY_THRESHOLD,
        "threshold_name": FOV_THRESHOLD_NAME,
        "gaussian_diameter": "D_FoV = 2 * pump_waist_um",
        "not_array_extent": True,
        "mesh_radius_over_waist": FOV_MESH_RADIUS_OVER_WAIST,
        "mesh_samples_per_waist": FOV_MESH_SAMPLES_PER_WAIST,
    },
    "resolved_spatial_mode": {
        "sigma_source": "resolved_spatial_mapping.correlation_sigma_object_um",
        "mode_radius": "MODE_RADIUS_IN_SIGMA * sigma_object_um",
        "mode_radius_in_sigma": MODE_RADIUS_IN_SIGMA,
        "mode_area": "pi * (mode_radius)^2",
        "mode_area_name": MODE_AREA_NAME,
        "N_modes": "A_FoV / A_mode",
        "N_modes_1d": "D_FoV / (2 * sigma_object_um)",
        "mode_count_name": MODE_COUNT_NAME,
    },
    "sampling_window_policy": SAMPLING_WINDOW_LABEL,
    "profiles": list(PROFILES),
}
