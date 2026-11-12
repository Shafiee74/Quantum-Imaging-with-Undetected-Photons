"""Object-plane coordinate helpers for config-agnostic benchmark targets.

Uses the same centered-grid convention as production
``object_plane_coordinate_grids_um``:
``x[i] = (i - 0.5*(N-1)) * pixel_um``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Canonical sampling shared by all A/B/C/D consumers of these targets.
DEFAULT_N: int = 64
DEFAULT_PIXEL_UM: float = 10.0


def centered_axis_um(n: int, pixel_um: float) -> np.ndarray:
    """1-D centered coordinate axis [µm]."""
    n_i = int(n)
    pitch = float(pixel_um)
    if n_i < 1:
        raise ValueError("n must be >= 1")
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pixel_um must be finite and > 0")
    return (np.arange(n_i, dtype=np.float64) - 0.5 * (n_i - 1)) * pitch


def coordinate_grids_um(
    shape: tuple[int, int],
    pixel_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_um, y_um, X_um, Y_um)`` with meshgrids in µm."""
    h, w = int(shape[0]), int(shape[1])
    x_um = centered_axis_um(w, pixel_um)
    y_um = centered_axis_um(h, pixel_um)
    X_um, Y_um = np.meshgrid(x_um, y_um)
    return x_um, y_um, X_um, Y_um


def physical_extent_um(n: int, pixel_um: float) -> dict[str, float]:
    """Physical FOV extent for an ``n×n`` map sampled at ``pixel_um``."""
    axis = centered_axis_um(n, pixel_um)
    width = float(axis[-1] - axis[0] + pixel_um) if n > 0 else 0.0
    # Equivalent: n * pixel_um
    extent = float(n) * float(pixel_um)
    return {
        "width_um": extent,
        "height_um": extent,
        "x_min_um": float(axis[0]) if n else 0.0,
        "x_max_um": float(axis[-1]) if n else 0.0,
        "y_min_um": float(axis[0]) if n else 0.0,
        "y_max_um": float(axis[-1]) if n else 0.0,
        "span_axis_um": float(axis[-1] - axis[0]) if n > 1 else 0.0,
        "pixel_um": float(pixel_um),
        "n": int(n),
    }


def geometry_metadata(
    shape: tuple[int, int],
    pixel_um: float,
) -> dict[str, Any]:
    ext = physical_extent_um(shape[0], pixel_um)
    # Allow non-square; recompute width/height separately.
    x_um = centered_axis_um(shape[1], pixel_um)
    y_um = centered_axis_um(shape[0], pixel_um)
    return {
        "shape_hw": [int(shape[0]), int(shape[1])],
        "pixel_um": float(pixel_um),
        "pixel_spacing_um": float(pixel_um),
        "width_um": float(shape[1]) * float(pixel_um),
        "height_um": float(shape[0]) * float(pixel_um),
        "x_min_um": float(x_um[0]),
        "x_max_um": float(x_um[-1]),
        "y_min_um": float(y_um[0]),
        "y_max_um": float(y_um[-1]),
        "coordinate_convention": "centered_i_minus_half_Nm1_times_pitch",
        "units": {
            "coordinates": "um",
            "pixel_spacing": "um",
            "transmission_T": "intensity_0_to_1",
            "amplitude": "sqrt_T",
            "phase": "rad",
        },
        "config_agnostic": True,
        "square_extent_check_um": ext["width_um"],
    }
