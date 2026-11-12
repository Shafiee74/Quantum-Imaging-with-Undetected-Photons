"""Phase-error and visibility helpers for shared reconstruction."""
from __future__ import annotations

from typing import Any

import numpy as np


def wrap_phase(phase_rad: np.ndarray | float) -> np.ndarray:
    """Map phase to principal value in (-π, π]."""
    return np.angle(np.exp(1j * np.asarray(phase_rad, dtype=np.float64)))


def circular_phase_error(
    estimate_rad: np.ndarray | float,
    reference_rad: np.ndarray | float,
) -> np.ndarray:
    """Principal-value phase difference estimate − reference in (-π, π]."""
    return wrap_phase(
        np.asarray(estimate_rad, dtype=np.float64)
        - np.asarray(reference_rad, dtype=np.float64)
    )


def remove_global_piston(
    estimate_rad: np.ndarray,
    reference_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract circular-mean piston so that mean circular error is ~0."""
    err = circular_phase_error(estimate_rad, reference_rad)
    if mask is None:
        piston = float(np.angle(np.mean(np.exp(1j * err))))
    else:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return np.asarray(estimate_rad, dtype=np.float64)
        piston = float(np.angle(np.mean(np.exp(1j * err[m]))))
    return wrap_phase(np.asarray(estimate_rad, dtype=np.float64) - piston)


def phase_error_metrics(
    estimate_rad: np.ndarray,
    reference_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    remove_piston: bool = False,
) -> dict[str, Any]:
    """Quantitative circular error summary against synthetic truth."""
    est = np.asarray(estimate_rad, dtype=np.float64)
    ref = np.asarray(reference_rad, dtype=np.float64)
    if remove_piston:
        est = remove_global_piston(est, ref, mask=mask)
    err = circular_phase_error(est, ref)
    if mask is None:
        vals = err.ravel()
    else:
        vals = err[np.asarray(mask, dtype=bool)].ravel()
    if vals.size == 0:
        return {
            "n_pixels": 0,
            "rmse_rad": float("nan"),
            "mae_rad": float("nan"),
            "max_abs_rad": float("nan"),
            "piston_removed": bool(remove_piston),
        }
    return {
        "n_pixels": int(vals.size),
        "rmse_rad": float(np.sqrt(np.mean(vals * vals))),
        "mae_rad": float(np.mean(np.abs(vals))),
        "max_abs_rad": float(np.max(np.abs(vals))),
        "piston_removed": bool(remove_piston),
    }


def visibility_from_ab(
    amplitude_B: np.ndarray | float,
    dc_A: np.ndarray | float,
    *,
    eps: float = 1.0e-15,
) -> np.ndarray:
    """Michelson-style visibility ``V = B / A`` for ``I = A + B cos(·)``."""
    A = np.asarray(dc_A, dtype=np.float64)
    B = np.asarray(amplitude_B, dtype=np.float64)
    return B / np.maximum(np.abs(A), eps)
