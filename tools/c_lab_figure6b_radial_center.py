"""Ring-center estimation for Figure 6(b) radial analysis."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from tools.c_lab_figure6b_ring_extraction import (
    RingCenter,
    _sector_profiles,
    intensity_weighted_centroid,
    radial_profile,
)

Modality = Literal["theory", "experiment"]


def _sector_peak_radii_mm(
    img: np.ndarray,
    center: RingCenter,
    *,
    pixel_mm: float,
) -> list[float]:
    sectors = _sector_profiles(img, center, pixel_mm=pixel_mm)
    half = len(sectors[0]) // 2
    r_mm = np.arange(len(sectors[0]), dtype=np.float64) * pixel_mm
    peaks: list[float] = []
    for prof in sectors:
        if not np.isfinite(prof).any():
            continue
        y = prof - float(np.nanmin(prof))
        if float(np.max(y)) <= 0:
            continue
        i = int(np.argmax(y))
        if i > half // 4:
            peaks.append(float(r_mm[i]))
    return peaks


def estimate_ring_center(
    img: np.ndarray,
    *,
    modality: Modality,
    label: str,
    pixel_mm: float = 0.013,
    search_px: int = 7,
) -> RingCenter:
    """Intensity centroid seed + local search minimizing azimuthal peak-radius scatter."""
    cy0, cx0 = intensity_weighted_centroid(img)
    best_cy, best_cx = cy0, cx0
    best_score = float("inf")
    z = np.asarray(img, dtype=np.float64)
    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            cy, cx = cy0 + dy, cx0 + dx
            if cy < 8 or cx < 8 or cy >= z.shape[0] - 8 or cx >= z.shape[1] - 8:
                continue
            c = RingCenter(cy=cy, cx=cx, method="search", modality=modality)
            peaks = _sector_peak_radii_mm(z, c, pixel_mm=pixel_mm)
            if len(peaks) < 4:
                score = 1e6 + abs(dy) + abs(dx)
            else:
                score = float(np.std(peaks)) + 0.01 * (abs(dy) + abs(dx))
            if score < best_score:
                best_score = score
                best_cy, best_cx = cy, cx
    method = (
        f"5th-percentile intensity centroid + azimuthal sector peak-radius "
        f"symmetry search (±{search_px} px) on {label}"
    )
    return RingCenter(cy=best_cy, cx=best_cx, method=method, modality=modality)


def center_sensitivity(
    img: np.ndarray,
    center: RingCenter,
    *,
    pixel_mm: float,
    jitter_px: int = 3,
) -> dict[str, float]:
    """Peak-radius std vs small center jitter (artifact sensitivity)."""
    base = _sector_peak_radii_mm(img, center, pixel_mm=pixel_mm)
    base_std = float(np.std(base)) if len(base) >= 3 else float("nan")
    scores = []
    for dy in range(-jitter_px, jitter_px + 1):
        for dx in range(-jitter_px, jitter_px + 1):
            c = RingCenter(
                cy=center.cy + dy,
                cx=center.cx + dx,
                method=center.method,
                modality=center.modality,
            )
            pr = _sector_peak_radii_mm(img, c, pixel_mm=pixel_mm)
            if len(pr) >= 3:
                scores.append(float(np.std(pr)))
    return {
        "sector_peak_radius_std_mm": base_std,
        "center_jitter_std_mean": float(np.mean(scores)) if scores else float("nan"),
        "center_jitter_std_max": float(np.max(scores)) if scores else float("nan"),
    }
