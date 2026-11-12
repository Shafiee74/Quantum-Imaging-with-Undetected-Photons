"""Metric extraction for ablation cases (reuse shared phase recon helpers)."""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.monte_carlo.statistics import (
    covariance_pair,
    normalized_correlation,
)
from verification.phase_reconstruction import (
    phase_error_metrics,
    reconstruct_phase_steps,
)


def michelson_visibility(plus: np.ndarray, minus: np.ndarray, *, eps: float = 1e-15) -> float:
    """Frame-derived Michelson V = mean|I+−I−| / mean(I++I−)."""
    Ip = np.asarray(plus, dtype=np.float64)
    Im = np.asarray(minus, dtype=np.float64)
    num = float(np.mean(np.abs(Ip - Im)))
    den = float(np.mean(Ip + Im)) + float(eps)
    return num / den


def snr_mean_over_std(img: np.ndarray, *, eps: float = 1e-15) -> float:
    """Spatial SNR proxy: mean / std (finite, non-negative intensities)."""
    a = np.asarray(img, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("nan")
    std = float(np.std(a, ddof=1))
    return float(np.mean(a) / max(std, eps))


def extract_case_metrics(
    result: Any,
    *,
    n: int,
    phi_truth_rad: float,
    detector_noise: bool,
) -> dict[str, Any]:
    """Compute intensity / visibility / phase / SNR / cov metrics from SimResult."""
    plus = np.asarray(result.frames_plus, dtype=np.float64)
    minus = np.asarray(result.frames_minus, dtype=np.float64)
    D = np.asarray(result.frames_D, dtype=np.float64)

    # Prefer mid-step (frame 0 of 4-step) maps for spatial metrics.
    I_plus = plus[0]
    I_minus = minus[0]
    I_sig = I_plus  # selected port proxy

    vo = dict(result.meta.get("visibility_observables") or {})
    V_5050 = float(vo.get("V_50_50") or 0.0)
    V_frame = michelson_visibility(I_plus, I_minus)

    rec = reconstruct_phase_steps(
        {"I0": D[0], "I_pi_2": D[1], "I_pi": D[2], "I_3pi_2": D[3]}
    )
    truth = np.full((n, n), float(phi_truth_rad), dtype=np.float64)
    err = phase_error_metrics(
        rec.phase_wrapped_rad, truth, remove_piston=True
    )
    # No-piston RMSE exposes global phase-drift bias vs protocol truth.
    err_raw = phase_error_metrics(
        rec.phase_wrapped_rad, truth, remove_piston=False
    )

    flat_p = I_plus.ravel()
    flat_m = I_minus.ravel()
    cov_pm = covariance_pair(flat_p, flat_m)
    corr_pm = normalized_correlation(flat_p, flat_m)

    return {
        "detector_noise": bool(detector_noise),
        "intensity_mean": float(np.mean(I_sig)),
        "intensity_std": float(np.std(I_sig, ddof=1)),
        "intensity_map": I_sig,
        "intensity_plus_map": I_plus,
        "intensity_minus_map": I_minus,
        "visibility_V_50_50": V_5050,
        "visibility_frame_michelson": V_frame,
        "visibility_map": np.abs(I_plus - I_minus) / (I_plus + I_minus + 1e-15),
        "phase_reconstructed": rec.phase_wrapped_rad,
        "phase_mean_rad": float(np.mean(rec.phase_wrapped_rad)),
        "phase_rmse_rad": float(err["rmse_rad"]),
        "phase_rmse_no_piston_rad": float(err_raw["rmse_rad"]),
        "phase_mae_rad": float(err["mae_rad"]),
        "phase_max_abs_rad": float(err["max_abs_rad"]),
        "snr_spatial": snr_mean_over_std(I_sig),
        "covariance_plus_minus": float(cov_pm),
        "correlation_plus_minus": float(corr_pm),
        "C_12_abs": float(vo.get("C_12_abs") or 0.0),
        "N_1": float(vo.get("N_1") or 0.0),
        "N_2": float(vo.get("N_2") or 0.0),
    }


def degradation_row(
    case_id: str,
    stage: str,
    metrics: dict[str, Any],
    clean: dict[str, Any],
) -> dict[str, Any]:
    """Scalar degradation vs Stage-A clean baseline."""

    def d(key: str) -> float:
        return float(metrics[key]) - float(clean[key])

    def rel(key: str) -> float:
        c = float(clean[key])
        m = float(metrics[key])
        if abs(c) < 1e-15:
            return float("nan")
        return (m - c) / abs(c)

    return {
        "case_id": case_id,
        "stage": stage,
        "intensity_mean": float(metrics["intensity_mean"]),
        "intensity_delta": d("intensity_mean"),
        "intensity_rel": rel("intensity_mean"),
        "visibility_frame": float(metrics["visibility_frame_michelson"]),
        "visibility_frame_delta": d("visibility_frame_michelson"),
        "visibility_V_50_50": float(metrics["visibility_V_50_50"]),
        "visibility_V_50_50_delta": d("visibility_V_50_50"),
        "phase_rmse_rad": float(metrics["phase_rmse_rad"]),
        "phase_rmse_delta": d("phase_rmse_rad"),
        "phase_rmse_no_piston_rad": float(metrics["phase_rmse_no_piston_rad"]),
        "phase_rmse_no_piston_delta": d("phase_rmse_no_piston_rad"),
        "phase_mean_rad": float(metrics["phase_mean_rad"]),
        "phase_mean_delta": d("phase_mean_rad"),
        "snr_spatial": float(metrics["snr_spatial"]),
        "snr_spatial_delta": d("snr_spatial"),
        "snr_spatial_rel": rel("snr_spatial"),
        "covariance_plus_minus": float(metrics["covariance_plus_minus"]),
        "correlation_plus_minus": float(metrics["correlation_plus_minus"]),
        "detector_noise": bool(metrics["detector_noise"]),
    }
