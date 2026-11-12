"""Robust spatial-frequency extraction for Figure 6(a) transverse fringes."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

PIXEL_MM = 0.013


def signal_roi_mask(img: np.ndarray, *, floor_pct: float = 12.0) -> np.ndarray:
    """Soft mask of illuminated signal spot (modality-specific, from this frame)."""
    z = np.asarray(img, dtype=np.float64)
    z = z - float(np.min(z))
    peak = float(np.max(z)) + 1e-12
    m = z >= (floor_pct / 100.0) * peak
    if not np.any(m):
        m = z > 0.05 * peak
    return m.astype(np.float64)


def _bin_projection(img: np.ndarray, mask: np.ndarray, normal_deg: float) -> tuple[np.ndarray, float]:
    """Average intensity along direction normal to fringes → 1D profile vs position (mm)."""
    h, w = img.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    phi = math.radians(normal_deg)
    # Coordinate along fringe normal (spatial axis for frequency).
    u = xx * math.cos(phi) + yy * math.sin(phi)
    u_px = u[mask > 0.5]
    val = img[mask > 0.5]
    if u_px.size < 32:
        u_px = u.ravel()
        val = img.ravel()
    umin, umax = float(np.min(u_px)), float(np.max(u_px))
    nb = min(512, max(64, int((umax - umin) * 0.8)))
    edges = np.linspace(umin, umax, nb + 1)
    prof = np.full(nb, np.nan, dtype=np.float64)
    for i in range(nb):
        sel = (u_px >= edges[i]) & (u_px < edges[i + 1])
        if np.count_nonzero(sel) >= 2:
            prof[i] = float(np.mean(val[sel]))
    centers = 0.5 * (edges[:-1] + edges[1:]) * PIXEL_MM
    return centers, prof


def _peak_k_cycles_mm(pos_mm: np.ndarray, prof: np.ndarray, *, k_hint: float | None) -> dict[str, Any]:
    y = np.asarray(prof, dtype=np.float64)
    x = np.asarray(pos_mm, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 16:
        return {"k_cycles_per_mm": float("nan"), "peak_index": -1, "ambiguous": True}
    y = y - float(np.mean(y[m]))
    y = gaussian_filter(np.where(m, y, 0.0), 1.0)
    y[~m] = 0.0
    n = len(y)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=(x[1] - x[0]) if n > 1 else PIXEL_MM)
    spec[0] = 0.0
    if spec.size <= 2:
        return {"k_cycles_per_mm": float("nan"), "peak_index": -1, "ambiguous": True}
    prom = max(0.08 * float(np.max(spec)), 1e-9)
    idx, props = find_peaks(spec, prominence=prom, distance=2)
    if idx.size == 0:
        idx = np.array([int(np.argmax(spec[1:])) + 1])
    candidates = [(int(i), float(spec[i]), float(freqs[i])) for i in idx if freqs[i] > 0.05]
    if not candidates:
        return {"k_cycles_per_mm": float("nan"), "peak_index": -1, "ambiguous": True}
    if k_hint is not None and math.isfinite(k_hint) and k_hint > 0:
        i_sel = min(candidates, key=lambda t: abs(t[2] - k_hint))[0]
        k_sel = [c for c in candidates if c[0] == i_sel][0][2]
    else:
        i_sel = max(candidates, key=lambda t: t[1])[0]
        k_sel = [c for c in candidates if c[0] == i_sel][0][2]
    return {
        "k_cycles_per_mm": float(k_sel),
        "peak_index": int(i_sel),
        "ambiguous": len(candidates) > 3,
        "n_candidates": len(candidates),
        "candidate_frequencies_cycles_per_mm": [c[2] for c in candidates],
        "candidate_magnitudes": [c[1] for c in candidates],
        "spectrum_freqs": freqs,
        "spectrum_mag": spec,
    }


def estimate_fringe_normal_deg(img: np.ndarray, mask: np.ndarray) -> float:
    """Dominant fringe-normal angle from masked 2D FFT (degrees)."""
    h, w = img.shape
    z = np.asarray(img, dtype=np.float64)
    z = (z - float(np.mean(z[mask > 0.5]))) * mask
    wy, wx = np.hanning(h), np.hanning(w)
    spec = np.fft.fftshift(np.fft.fft2(z * np.outer(wy, wx)))
    mag = np.abs(spec)
    cy, cx = h // 2, w // 2
    mag[cy - 3 : cy + 4, cx - 3 : cx + 4] = 0
    yy, xx = np.indices(mag.shape)
    rr = np.hypot(yy - cy, xx - cx)
    band = (rr >= 5) & (rr <= min(h, w) * 0.32)
    mag[~band] = 0
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    return float(math.degrees(math.atan2(iy - cy, ix - cx)))


def extract_series(
    images: dict[float, np.ndarray],
    *,
    dx_list: list[float],
    k_hints: dict[float, float] | None = None,
    normal_ref_dx: float = 40.0,
) -> tuple[list[dict[str, Any]], float]:
    """Extract k(dx) with shared orientation from strong-fringe reference frame."""
    ref_img = images[normal_ref_dx]
    ref_mask = signal_roi_mask(ref_img)
    normal_deg = estimate_fringe_normal_deg(ref_img, ref_mask)

    rows: list[dict[str, Any]] = []
    k_prev = 0.0
    for dx in dx_list:
        img = images[dx]
        mask = signal_roi_mask(img)
        # suppress outside spot
        masked = (img - float(np.mean(img[mask > 0.5]))) * mask
        hint = (k_hints or {}).get(dx)
        pos, prof = _bin_projection(masked, mask, normal_deg)
        pk = _peak_k_cycles_mm(pos, prof, k_hint=hint)
        k = pk["k_cycles_per_mm"]
        # Continuity: if multiple close candidates, prefer k >= k_prev - small tolerance
        if math.isfinite(k) and dx > 0 and k + 0.15 < k_prev and pk.get("spectrum_mag") is not None:
            freqs = pk["spectrum_freqs"]
            spec = pk["spectrum_mag"]
            prom = max(0.08 * float(np.max(spec)), 1e-9)
            idx, _ = find_peaks(spec, prominence=prom, distance=2)
            alts = [float(freqs[i]) for i in idx if freqs[i] > 0.05 and freqs[i] >= k_prev - 0.1]
            if alts:
                k = min(alts, key=lambda v: abs(v - (hint or k_prev)))
                pk["k_cycles_per_mm"] = k
                pk["continuity_adjusted"] = True
        if math.isfinite(k):
            k_prev = k
        rows.append(
            {
                "dx_um": dx,
                "k_abs_cycles_per_mm": k,
                "fringe_normal_deg": normal_deg,
                "peak_index_1d": pk.get("peak_index"),
                "ambiguous": pk.get("ambiguous", False),
                "continuity_adjusted": pk.get("continuity_adjusted", False),
                "n_candidates": pk.get("n_candidates", 0),
                "candidate_frequencies_cycles_per_mm": pk.get(
                    "candidate_frequencies_cycles_per_mm", []
                ),
                "candidate_magnitudes": pk.get("candidate_magnitudes", []),
                "pos_mm": pos,
                "prof": prof,
                "spec_freqs": pk.get("spectrum_freqs"),
                "spec_mag": pk.get("spectrum_mag"),
            }
        )
    return rows, normal_deg
