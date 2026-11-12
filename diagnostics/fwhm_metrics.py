"""Horizontal FWHM diagnostics through the intensity peak."""
from __future__ import annotations

from typing import Any

import numpy as np


def fwhm_horizontal_through_peak(
    image: np.ndarray,
    x_um: np.ndarray,
) -> dict[str, Any]:
    """Measure horizontal FWHM on the row that contains the global peak.

    Returns ``status="EDGE_CLIPPED"`` with ``fwhm_um=nan`` when the half-maximum
    contour reaches either horizontal edge of the sampled window (flat-top /
    window-limited profiles).
    """
    img = np.asarray(image, dtype=np.float64)
    x = np.asarray(x_um, dtype=np.float64).reshape(-1)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    if x.size != img.shape[1]:
        raise ValueError("x_um length must match image width")

    peak_flat = int(np.nanargmax(img))
    peak_row, peak_col = np.unravel_index(peak_flat, img.shape)
    profile = np.asarray(img[peak_row, :], dtype=np.float64)
    peak_val = float(profile[peak_col])
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return {
            "status": "NO_PEAK",
            "fwhm_um": float("nan"),
            "peak_row": int(peak_row),
            "peak_col": int(peak_col),
            "half_max": float("nan"),
        }

    half = 0.5 * peak_val
    above = profile >= half
    if bool(above[0]) or bool(above[-1]):
        return {
            "status": "EDGE_CLIPPED",
            "fwhm_um": float("nan"),
            "peak_row": int(peak_row),
            "peak_col": int(peak_col),
            "half_max": float(half),
        }

    left = peak_col
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_col
    while right < profile.size - 1 and above[right + 1]:
        right += 1

    def _crossing(i0: int, i1: int) -> float:
        y0 = float(profile[i0])
        y1 = float(profile[i1])
        x0 = float(x[i0])
        x1 = float(x[i1])
        if y1 == y0:
            return x0
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    left_i1 = left
    left_i0 = max(left - 1, 0)
    right_i0 = right
    right_i1 = min(right + 1, profile.size - 1)
    x_left = _crossing(left_i0, left_i1)
    x_right = _crossing(right_i0, right_i1)
    return {
        "status": "OK",
        "fwhm_um": float(abs(x_right - x_left)),
        "peak_row": int(peak_row),
        "peak_col": int(peak_col),
        "half_max": float(half),
        "x_left_um": float(x_left),
        "x_right_um": float(x_right),
    }
