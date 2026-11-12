"""Independent reference maps and predicted fringe observables (Task 17).

No production imports.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from verification.config_c.transverse_lens.derivation import (
    config_c_pass_factor,
    equivalent_wavelength_nm,
    fringe_period_um,
    fringe_spatial_frequency_per_um,
    phase_slope_rad_per_um,
    thin_lens_linear_phase_rad,
    transverse_wavevector_rad_per_um,
)


def camera_grid_um(n: int, pixel_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered camera coordinates [µm] matching production mgrid convention."""
    size = int(n)
    if size <= 1:
        raise ValueError("grid size must be > 1")
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x_um = (xx - (size - 1) * 0.5) * float(pixel_um)
    y_um = (yy - (size - 1) * 0.5) * float(pixel_um)
    return x_um, y_um


def transverse_phase_map(
    n: int,
    *,
    pixel_um: float,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float | None = None,
) -> dict[str, Any]:
    """Independent far-field transverse phase map on the camera grid."""
    x_um, y_um = camera_grid_um(n, pixel_um)
    f_c_um = float(f_c_mm) * 1.0e3
    if f_c_um <= 0.0:
        raise ValueError("f_c must be > 0")
    phi = (
        4.0
        * math.pi
        * (float(delta_x_um) * x_um + float(delta_y_um) * y_um)
        / (float(lambda_s_nm) * 1.0e-3 * f_c_um)
    )
    return {
        "phase_rad": phi.astype(np.float64),
        "x_um": x_um,
        "y_um": y_um,
        "pass_factor": float(config_c_pass_factor() if pass_factor is None else pass_factor),
        "pass_factor_applied_to_transverse": False,
        "delta_x_um": float(delta_x_um),
        "delta_y_um": float(delta_y_um),
        "lambda_s_nm": float(lambda_s_nm),
        "lambda_eq_nm": equivalent_wavelength_nm(lambda_s_nm, lambda_i_nm),
        "f_c_mm": float(f_c_mm),
        "equation": "Φ_tr = 4π (Δx x + Δy y) / (λ_s f_c)",
    }


def predicted_observables(
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float | None = None,
) -> dict[str, Any]:
    """Closed-form fringe orientation / κ / period / slope / k⃗."""
    g = float(config_c_pass_factor() if pass_factor is None else pass_factor)
    sx, sy = phase_slope_rad_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    kx, ky = transverse_wavevector_rad_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    kappa = fringe_spatial_frequency_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    period = fringe_period_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    if abs(kx) < 1.0e-30 and abs(ky) < 1.0e-30:
        orientation_deg = None
        fringe_normal_deg = None
    else:
        orientation_deg = float(math.degrees(math.atan2(ky, kx)))
        fringe_normal_deg = orientation_deg
    return {
        "delta_x_um": float(delta_x_um),
        "delta_y_um": float(delta_y_um),
        "pass_factor": g,
        "phase_slope_x_rad_per_um": sx,
        "phase_slope_y_rad_per_um": sy,
        "kx_rad_per_um": kx,
        "ky_rad_per_um": ky,
        "kappa_cycles_per_um": kappa,
        "fringe_period_um": period,
        "wavevector_angle_deg": orientation_deg,
        "fringe_normal_angle_deg": fringe_normal_deg,
        "fringes_perpendicular_to_delta": True,
    }


def sample_point_phase(
    x_um: float,
    y_um: float,
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float | None = None,
) -> float:
    g = float(config_c_pass_factor() if pass_factor is None else pass_factor)
    return thin_lens_linear_phase_rad(
        x_um,
        y_um,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )


def estimate_phase_slope_from_map(
    phase_rad: np.ndarray,
    *,
    pixel_um: float,
) -> tuple[float, float]:
    """Linear least-squares fit of φ ≈ sx x + sy y + c on the full grid."""
    phi = np.asarray(phase_rad, dtype=np.float64)
    n = int(phi.shape[0])
    x_um, y_um = camera_grid_um(n, pixel_um)
    a = np.column_stack(
        [x_um.ravel(), y_um.ravel(), np.ones(x_um.size, dtype=np.float64)]
    )
    coef, _, _, _ = np.linalg.lstsq(a, phi.ravel(), rcond=None)
    return float(coef[0]), float(coef[1])


def estimate_kappa_from_fft(
    fringe: np.ndarray,
    *,
    pixel_um: float,
) -> dict[str, float]:
    """Peak spatial frequency of cos(φ) via 2-D FFT (DC suppressed)."""
    img = np.asarray(fringe, dtype=np.float64)
    n = int(img.shape[0])
    img = img - float(np.mean(img))
    spec = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(spec)
    cy = n // 2
    cx = n // 2
    mag[cy, cx] = 0.0
    mag[max(0, cy - 1) : cy + 2, max(0, cx - 1) : cx + 2] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    fx_cycles_per_px = (ix - cx) / float(n)
    fy_cycles_per_px = (iy - cy) / float(n)
    kappa_x = fx_cycles_per_px / float(pixel_um)
    kappa_y = fy_cycles_per_px / float(pixel_um)
    kappa = float(math.hypot(kappa_x, kappa_y))
    return {
        "kappa_cycles_per_um": kappa,
        "kappa_x_cycles_per_um": float(kappa_x),
        "kappa_y_cycles_per_um": float(kappa_y),
        "peak_ix": float(ix),
        "peak_iy": float(iy),
    }
