"""Usable field-of-view extraction (physical; not array extent)."""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.spatial.definitions import (
    FOV_INTENSITY_THRESHOLD,
    FOV_MESH_RADIUS_OVER_WAIST,
    FOV_MESH_SAMPLES_PER_WAIST,
    FOV_THRESHOLD_NAME,
)


def gaussian_pump_intensity(
    X_um: np.ndarray,
    Y_um: np.ndarray,
    waist_um: float,
    *,
    center_x_um: float = 0.0,
    center_y_um: float = 0.0,
) -> np.ndarray:
    """I/I0 = exp(−2 r²/w_p²) peak-normalized."""
    w = float(waist_um)
    if w <= 0.0 or not np.isfinite(w):
        raise ValueError("waist_um must be finite and > 0")
    r2 = (X_um - float(center_x_um)) ** 2 + (Y_um - float(center_y_um)) ** 2
    return np.exp(-2.0 * r2 / (w * w))


def analytical_gaussian_fov_diameter_um(waist_um: float) -> float:
    """1/e² intensity diameter for a Gaussian pump: D = 2 w_p."""
    w = float(waist_um)
    if w <= 0.0 or not np.isfinite(w):
        raise ValueError("waist_um must be finite and > 0")
    return 2.0 * w


def analysis_mesh_for_waist(
    waist_um: float,
    *,
    radius_over_waist: float = FOV_MESH_RADIUS_OVER_WAIST,
    samples_per_waist: float = FOV_MESH_SAMPLES_PER_WAIST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fine physical mesh (±k w_p), independent of detector array pitch."""
    w = float(waist_um)
    half = float(radius_over_waist) * w
    pitch = w / float(samples_per_waist)
    if pitch <= 0.0:
        raise ValueError("analysis pitch must be > 0")
    axis = np.arange(-half, half + 0.5 * pitch, pitch, dtype=np.float64)
    X, Y = np.meshgrid(axis, axis)
    return axis, axis, X, Y, float(pitch)


def measure_fov_from_intensity(
    intensity: np.ndarray,
    X_um: np.ndarray,
    Y_um: np.ndarray,
    *,
    threshold: float = FOV_INTENSITY_THRESHOLD,
) -> dict[str, Any]:
    """Equivalent diameter of the I/Imax ≥ threshold superlevel set.

    Equivalent diameter uses area: D_eq = 2 sqrt(A/π).
    Also reports max radial extent of included samples.
    """
    I = np.asarray(intensity, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("intensity must be 2-D")
    peak = float(np.max(I))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("intensity peak must be positive and finite")
    unit = I / peak
    mask = unit >= float(threshold)
    # Pixel area from local spacing (assume uniform mesh).
    dx = float(np.mean(np.diff(X_um[0, :]))) if X_um.shape[1] > 1 else float("nan")
    dy = float(np.mean(np.diff(Y_um[:, 0]))) if Y_um.shape[0] > 1 else float("nan")
    area = float(np.count_nonzero(mask) * abs(dx) * abs(dy))
    d_eq = float(2.0 * np.sqrt(area / np.pi)) if area > 0.0 else 0.0
    if np.any(mask):
        r = np.sqrt(X_um[mask] ** 2 + Y_um[mask] ** 2)
        r_max = float(np.max(r))
        # Truncation: mask touches mesh border
        on_border = (
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        )
    else:
        r_max = 0.0
        on_border = False
    return {
        "threshold": float(threshold),
        "threshold_name": FOV_THRESHOLD_NAME,
        "peak_intensity": peak,
        "area_um2": area,
        "diameter_eq_um": d_eq,
        "radius_max_um": r_max,
        "diameter_max_um": 2.0 * r_max,
        "n_pixels_above": int(np.count_nonzero(mask)),
        "mesh_dx_um": dx,
        "mesh_dy_um": dy,
        "truncated_by_mesh": bool(on_border),
        "mask": mask,
        "unit_intensity": unit,
    }


def usable_fov_object_plane(
    waist_um: float,
    *,
    center_x_um: float = 0.0,
    center_y_um: float = 0.0,
) -> dict[str, Any]:
    """Usable object-plane FoV from pump 1/e² convention + mesh check."""
    d_analytical = analytical_gaussian_fov_diameter_um(waist_um)
    x, y, X, Y, pitch = analysis_mesh_for_waist(waist_um)
    I = gaussian_pump_intensity(
        X, Y, waist_um, center_x_um=center_x_um, center_y_um=center_y_um
    )
    measured = measure_fov_from_intensity(I, X, Y)
    return {
        "plane": "object_crystal",
        "pump_waist_um": float(waist_um),
        "threshold": FOV_INTENSITY_THRESHOLD,
        "threshold_name": FOV_THRESHOLD_NAME,
        "diameter_analytical_um": d_analytical,
        "radius_analytical_um": 0.5 * d_analytical,
        "area_analytical_um2": float(np.pi * (0.5 * d_analytical) ** 2),
        "diameter_measured_eq_um": measured["diameter_eq_um"],
        "area_measured_um2": measured["area_um2"],
        "mesh_pitch_um": pitch,
        "mesh_half_extent_um": float(x[-1]),
        "truncated_by_mesh": measured["truncated_by_mesh"],
        # Primary reported FoV uses the analytical Gaussian contour (physical).
        "D_FoV_um": d_analytical,
        "A_FoV_um2": float(np.pi * (0.5 * d_analytical) ** 2),
        "units": "um",
        "source": "pump_1e2_intensity_contour_not_array_extent",
        "measured": {k: v for k, v in measured.items() if k not in ("mask", "unit_intensity")},
        "map_X_um": X,
        "map_Y_um": Y,
        "map_I": I,
        "map_mask": measured["mask"],
        "map_radial_r_um": np.linspace(0.0, float(x[-1]), 200),
        "map_radial_I": np.exp(
            -2.0 * (np.linspace(0.0, float(x[-1]), 200) / float(waist_um)) ** 2
        ),
    }
