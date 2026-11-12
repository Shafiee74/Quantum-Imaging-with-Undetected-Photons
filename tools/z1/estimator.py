"""Authoritative pixel-wise four-step visibility — ONE function for Exp and Sim."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_four_step_visibility(
    I0: np.ndarray,
    I1: np.ndarray,
    I2: np.ndarray,
    I3: np.ndarray,
    *,
    eps: float = 0.0,
) -> dict[str, Any]:
    """Pixel-wise four-step visibility for phases 0, pi/2, pi, 3pi/2.

    Frozen estimator (identical for experiment and simulation)::

        A = I0 - I2
        B = I1 - I3
        S = I0 + I1 + I2 + I3
        V = 2 * sqrt(A^2 + B^2) / S    (S > 0)
        phi = atan2(B, A)

    This is the same formula used in Y3/Y4 experimental maps.
    """
    I0 = np.asarray(I0, dtype=np.float64)
    I1 = np.asarray(I1, dtype=np.float64)
    I2 = np.asarray(I2, dtype=np.float64)
    I3 = np.asarray(I3, dtype=np.float64)
    A = I0 - I2
    B = I1 - I3
    S = I0 + I1 + I2 + I3
    with np.errstate(divide="ignore", invalid="ignore"):
        V = 2.0 * np.sqrt(A * A + B * B) / np.where(S > eps, S, np.nan)
        phi = np.arctan2(B, A)
    M = np.sqrt(A * A + B * B)
    valid = np.isfinite(V) & (S > eps)
    return {
        "V": V,
        "phase_rad": phi,
        "modulation": M,
        "DC": S,
        "valid_mask": valid,
        "equation": "V=2*sqrt((I0-I2)^2+(I1-I3)^2)/(I0+I1+I2+I3)",
        "estimator_id": "Z1_compute_four_step_visibility",
    }


def visibility_stats(V: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    v = np.asarray(V, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(v)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(v)
    vals = v[mask]
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n_valid": 0}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "n_valid": int(vals.size),
    }


def global_mean_adu_four_step(I0, I1, I2, I3) -> float:
    """Diagnostic only — NOT the primary Figure-5 comparison metric."""
    m0, m1, m2, m3 = float(np.mean(I0)), float(np.mean(I1)), float(np.mean(I2)), float(np.mean(I3))
    a, b, s = m0 - m2, m1 - m3, m0 + m1 + m2 + m3
    return float(2.0 * np.sqrt(a * a + b * b) / s) if s > 0 else float("nan")
