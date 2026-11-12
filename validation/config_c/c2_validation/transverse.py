"""C2 transverse experimental metrics: FFT, κ, period, orientation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from validation.config_c.c2_validation.longitudinal import visibility_metrics
from verification.config_c.transverse_lens.derivation import (
    config_c_pass_factor,
    fringe_period_um,
    fringe_spatial_frequency_per_um,
    phase_slope_rad_per_um,
)


def dominant_fourier_peak(
    img: np.ndarray, *, pixel_um: float
) -> dict[str, Any]:
    z = np.asarray(img, dtype=np.float64)
    z = z - float(np.mean(z))
    # Hann window to reduce edge leakage
    wy = np.hanning(z.shape[0])
    wx = np.hanning(z.shape[1])
    window = np.outer(wy, wx)
    spec = np.fft.fftshift(np.fft.fft2(z * window))
    mag = np.abs(spec)
    cy, cx = z.shape[0] // 2, z.shape[1] // 2
    # suppress DC neighbourhood
    mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    # cycles per pixel
    fy = (iy - cy) / float(z.shape[0])
    fx = (ix - cx) / float(z.shape[1])
    # cycles per µm
    kappa_x = fx / float(pixel_um)
    kappa_y = fy / float(pixel_um)
    kappa = float(math.hypot(kappa_x, kappa_y))
    period = float(1.0 / kappa) if kappa > 0 else None
    orient = float(math.degrees(math.atan2(kappa_y, kappa_x))) if kappa > 0 else None
    return {
        "kappa_x_cycles_per_um": float(kappa_x),
        "kappa_y_cycles_per_um": float(kappa_y),
        "kappa_cycles_per_um": kappa,
        "fringe_period_um": period,
        "orientation_deg": orient,
        "peak_iy": int(iy),
        "peak_ix": int(ix),
        "fft_magnitude": mag,
        "pixel_um_used": float(pixel_um),
        "pixel_um_provenance": "ASSUMED_STOCK_C_16UM",
    }


def theory_transverse(
    delta_x_um: float,
    *,
    f_c_mm: float = 150.0,
    lambda_s_nm: float = 810.0,
    lambda_i_nm: float = 1550.0,
) -> dict[str, Any]:
    g = config_c_pass_factor()
    sx, sy = phase_slope_rad_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=0.0,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    kappa = fringe_spatial_frequency_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=0.0,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    period = fringe_period_um(
        delta_x_um=delta_x_um,
        delta_y_um=0.0,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=g,
    )
    return {
        "phase_slope_x_rad_per_um": sx,
        "phase_slope_y_rad_per_um": sy,
        "kappa_cycles_per_um": kappa,
        "fringe_period_um": period,
        "orientation_deg": 0.0 if delta_x_um != 0 else None,
        "pass_factor": g,
        "equation": "Φ_tr = 4π Δx x / (λ_s f_c)",
    }


def analyze_transverse_mean(
    mean_img: np.ndarray,
    *,
    delta_x_um: float,
    pixel_um_assumed: float = 16.0,
) -> dict[str, Any]:
    h, w = mean_img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    fft = dominant_fourier_peak(mean_img, pixel_um=pixel_um_assumed)
    vis = visibility_metrics(mean_img, cy, cx)
    th = theory_transverse(delta_x_um)
    # central row profile for fringe display
    row = mean_img[int(cy), :]
    return {
        "fft": {k: v for k, v in fft.items() if k != "fft_magnitude"},
        "fft_magnitude": fft["fft_magnitude"],
        "visibility": vis,
        "theory": th,
        "row_profile": row,
        "kappa_exp": fft["kappa_cycles_per_um"],
        "kappa_theory": th["kappa_cycles_per_um"],
        "period_exp_um": fft["fringe_period_um"],
        "period_theory_um": th["fringe_period_um"],
        "orientation_exp_deg": fft["orientation_deg"],
    }
