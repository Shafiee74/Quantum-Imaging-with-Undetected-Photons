"""Reproducible radial ring extraction for C_lab Figure 6(b) (analysis only)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

PIXEL_MM = 0.013


@dataclass
class RingCenter:
    """Documented ring center in image pixel coordinates (row, col)."""

    cy: float
    cx: float
    method: str
    modality: Literal["theory", "experiment"]


@dataclass
class RadialExtremum:
    radius_mm: float
    radius_sq_mm2: float
    fringe_order: int
    prominence: float
    kind: Literal["bright", "dark"]
    index_px: int
    quality: str


@dataclass
class RadialAnalysis:
    r_mm: np.ndarray
    intensity_raw: np.ndarray
    intensity_smooth: np.ndarray
    n_pixels_per_bin: np.ndarray
    center: RingCenter
    bright: list[RadialExtremum] = field(default_factory=list)
    dark: list[RadialExtremum] = field(default_factory=list)
    sector_notes: str = ""


def intensity_weighted_centroid(img: np.ndarray, *, percentile_floor: float = 5.0) -> tuple[float, float]:
    z = np.asarray(img, dtype=np.float64)
    z = z - float(np.percentile(z, percentile_floor))
    w = np.clip(z, 0.0, None)
    s = float(w.sum()) + 1e-12
    yy, xx = np.indices(z.shape, dtype=np.float64)
    return float(np.sum(w * yy) / s), float(np.sum(w * xx) / s)


def reference_center(
    img_ref: np.ndarray,
    *,
    modality: Literal["theory", "experiment"],
) -> RingCenter:
    cy, cx = intensity_weighted_centroid(img_ref)
    return RingCenter(
        cy=cy,
        cx=cx,
        method=f"intensity-weighted centroid (5th-percentile floor) on Δz_eff=0 {modality} frame",
        modality=modality,
    )


def radial_profile(
    img: np.ndarray,
    center: RingCenter,
    *,
    pixel_mm: float = PIXEL_MM,
    min_pixels_per_bin: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annular mean intensity vs radius; bins with poor coverage are NaN."""
    z = np.asarray(img, dtype=np.float64)
    yy, xx = np.indices(z.shape, dtype=np.float64)
    rr = np.hypot(yy - center.cy, xx - center.cx)
    rmax = int(
        min(
            center.cy,
            center.cx,
            z.shape[0] - 1 - center.cy,
            z.shape[1] - 1 - center.cx,
        )
    )
    prof = np.full(rmax + 1, np.nan, dtype=np.float64)
    counts = np.zeros(rmax + 1, dtype=np.int32)
    for ri in range(rmax + 1):
        m = (rr >= ri - 0.5) & (rr < ri + 0.5)
        n = int(np.count_nonzero(m))
        counts[ri] = n
        if n >= min_pixels_per_bin:
            prof[ri] = float(np.mean(z[m]))
    r_mm = np.arange(prof.size, dtype=np.float64) * pixel_mm
    return r_mm, prof, counts


def _smooth_profile(y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    finite = np.isfinite(y)
    if finite.sum() < 9:
        return y.copy()
    fill = float(np.nanmedian(y[finite]))
    z = np.where(finite, y, fill)
    win = 7 if z.size >= 7 else max(3, z.size // 2 * 2 + 1)
    if win >= 5 and win % 2 == 1 and z.size >= win:
        try:
            sm = savgol_filter(z, window_length=win, polyorder=2, mode="interp")
        except ValueError:
            sm = uniform_filter1d(z, size=win, mode="nearest")
    else:
        sm = uniform_filter1d(z, size=max(3, win), mode="nearest")
    sm[~finite] = np.nan
    sm[counts < 12] = np.nan
    return sm


def _sector_profiles(
    img: np.ndarray,
    center: RingCenter,
    *,
    pixel_mm: float = PIXEL_MM,
    n_sectors: int = 8,
    half_len: int | None = None,
) -> list[np.ndarray]:
    """Line samples along eight azimuths; used when annular mean is ambiguous."""
    z = np.asarray(img, dtype=np.float64)
    h, w = z.shape
    if half_len is None:
        half_len = int(
            min(center.cy, center.cx, h - 1 - center.cy, w - 1 - center.cx)
        )
    profiles: list[np.ndarray] = []
    for k in range(n_sectors):
        ang = math.pi * k / n_sectors
        t = np.arange(-half_len, half_len + 1, dtype=np.float64)
        yy = center.cy + t * math.sin(ang)
        xx = center.cx + t * math.cos(ang)
        vals = []
        for y, x in zip(yy, xx):
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < h and 0 <= ix < w:
                vals.append(float(z[iy, ix]))
            else:
                vals.append(float("nan"))
        profiles.append(np.asarray(vals, dtype=np.float64))
    return profiles


def _merge_sector_peaks(
    sector_raw: list[np.ndarray],
    *,
    pixel_mm: float,
    kind: Literal["bright", "dark"],
) -> list[tuple[float, float]]:
    """Collect peak radii from sectors; cluster within tolerance."""
    half = len(sector_raw[0]) // 2
    r_mm = np.arange(len(sector_raw[0]), dtype=np.float64) * pixel_mm
    hits: list[tuple[float, float]] = []
    for prof in sector_raw:
        sm = _smooth_profile(prof, np.full_like(prof, 100, dtype=np.int32))
        finite = np.isfinite(sm) & (r_mm >= 0.12)
        if finite.sum() < 12:
            continue
        y = sm.copy()
        y[~finite] = np.nanmedian(sm[finite])
        span = float(np.nanmax(y[finite]) - np.nanmin(y[finite]))
        prom = max(0.03 * span, 1e-9)
        dist = max(3, int(round(0.025 / pixel_mm)))
        sig = y if kind == "bright" else -y
        idx, props = find_peaks(sig[finite], prominence=prom, distance=dist)
        valid_r = r_mm[finite][idx]
        valid_p = props["prominences"]
        for rho, p in zip(valid_r, valid_p):
            if rho >= 0.12:
                hits.append((float(rho), float(p)))
    if not hits:
        return []
    hits.sort(key=lambda t: t[0])
    merged: list[tuple[float, float]] = []
    tol = 0.018  # mm
    for rho, p in hits:
        if merged and abs(rho - merged[-1][0]) <= tol:
            merged[-1] = ((merged[-1][0] + rho) / 2.0, max(merged[-1][1], p))
        else:
            merged.append((rho, p))
    return merged


def _drop_inner_artifacts(extrema: list[RadialExtremum], *, inner_mm: float = 0.11) -> list[RadialExtremum]:
    """Remove spurious peaks just outside the masked core (often at ~4 px)."""
    kept = [e for e in extrema if e.radius_mm >= inner_mm]
    if not kept and extrema:
        # Keep the outermost candidate only if everything was inside the core.
        kept = [max(extrema, key=lambda e: e.radius_mm)]
    for k, e in enumerate(kept, start=1):
        e.fringe_order = k
    return kept


def detect_rings(
    img: np.ndarray,
    center: RingCenter,
    *,
    pixel_mm: float = PIXEL_MM,
    min_radius_mm: float = 0.12,
    use_sector_fallback: bool = True,
    dense_rings: bool = False,
) -> RadialAnalysis:
    """Detect bright/dark annular extrema.

    Parameters
    ----------
    dense_rings :
        If True, use a tighter peak distance and mild radial detrending so
        higher-order Haidinger rings are not merged (needed after decoupling
        idler-lens Δz from the imaging defocus PSF).
    """
    r_mm, raw, counts = radial_profile(img, center, pixel_mm=pixel_mm)
    smooth = _smooth_profile(raw, counts)
    finite = np.isfinite(smooth) & (r_mm >= min_radius_mm)
    analysis = RadialAnalysis(
        r_mm=r_mm,
        intensity_raw=raw,
        intensity_smooth=smooth,
        n_pixels_per_bin=counts,
        center=center,
    )
    if finite.sum() < 15:
        analysis.sector_notes = "Insufficient valid annular bins."
        return analysis

    y = smooth.copy()
    y[~finite] = np.nanmedian(y[finite])
    if dense_rings:
        # Remove slow radial envelope so fine fringes keep local prominence.
        win = max(15, int(round(0.35 / pixel_mm)) | 1)
        if win >= 5 and y.size >= win:
            try:
                from scipy.ndimage import uniform_filter1d

                trend = uniform_filter1d(y, size=win, mode="nearest")
                y = y - trend
                y[~finite] = 0.0
            except Exception:
                pass
    span = float(np.nanmax(y[finite]) - np.nanmin(y[finite]))
    if span <= 0:
        analysis.sector_notes = "Flat radial profile."
        return analysis

    # Prominence scales with contrast; distance ~ half expected fringe spacing at high dz.
    prom_bright = max(0.025 * span, 0.02 * float(np.nanstd(y[finite])))
    if dense_rings:
        prom_bright = max(0.015 * span, 0.01 * float(np.nanstd(y[finite])))
    prom_dark = prom_bright
    # Minimum spacing between ring extrema (~0.15–0.25 mm typical on this crop).
    # Dense mode: ~0.06 mm allows higher-order Haidinger rings after PSF fix.
    dist_mm = 0.06 if dense_rings else 0.18
    dist_px = max(4 if dense_rings else 8, int(round(dist_mm / pixel_mm)))

    b_idx, b_props = find_peaks(y, prominence=prom_bright, distance=dist_px)
    d_idx, d_props = find_peaks(-y, prominence=prom_dark, distance=dist_px)

    def _pack(
        idx: np.ndarray,
        props: dict[str, np.ndarray],
        kind: Literal["bright", "dark"],
    ) -> list[RadialExtremum]:
        out: list[RadialExtremum] = []
        order = 0
        for i, prom in zip(idx, props["prominences"]):
            if not finite[i] or r_mm[i] < min_radius_mm:
                continue
            order += 1
            rho = float(r_mm[i])
            out.append(
                RadialExtremum(
                    radius_mm=rho,
                    radius_sq_mm2=rho * rho,
                    fringe_order=order,
                    prominence=float(prom),
                    kind=kind,
                    index_px=int(i),
                    quality="annular",
                )
            )
        return out

    dark_all = _pack(d_idx, d_props, "dark")
    dark_all = _drop_inner_artifacts(dark_all, inner_mm=min_radius_mm * 0.85)
    analysis.dark = dark_all

    bright_inner = min_radius_mm
    if dark_all:
        bright_inner = max(min_radius_mm, dark_all[0].radius_mm + 0.04)
    bright_all = _pack(b_idx, b_props, "bright")
    analysis.bright = _drop_inner_artifacts(bright_all, inner_mm=bright_inner)
    # Drop weak shoulders relative to the strongest detected ring.
    if analysis.bright:
        pmax = max(e.prominence for e in analysis.bright)
        floor = max(0.06 * pmax, 0.015 * span)
        analysis.bright = _drop_inner_artifacts(
            [e for e in analysis.bright if e.prominence >= floor],
            inner_mm=bright_inner,
        )

    # Sector fallback only when annular mean finds no rings (e.g. strongly distorted patterns).
    if use_sector_fallback and len(analysis.bright) == 0 and len(analysis.dark) == 0 and finite.sum() > 40:
        sectors = _sector_profiles(img, center, pixel_mm=pixel_mm)
        for kind in ("bright", "dark"):
            merged = _merge_sector_peaks(sectors, pixel_mm=pixel_mm, kind=kind)
            if kind == "bright" and len(merged) > len(analysis.bright):
                sector_bright = [
                    RadialExtremum(
                        radius_mm=rho,
                        radius_sq_mm2=rho * rho,
                        fringe_order=k + 1,
                        prominence=p,
                        kind="bright",
                        index_px=int(round(rho / pixel_mm)),
                        quality="sector_median",
                    )
                    for k, (rho, p) in enumerate(merged)
                ]
                analysis.bright = _drop_inner_artifacts(sector_bright, inner_mm=min_radius_mm)
                analysis.sector_notes = "Annular mean yielded ≤1 bright ring; sector consensus used."
            elif kind == "dark" and len(merged) > len(analysis.dark):
                analysis.dark = [
                    RadialExtremum(
                        radius_mm=rho,
                        radius_sq_mm2=rho * rho,
                        fringe_order=k + 1,
                        prominence=p,
                        kind="dark",
                        index_px=int(round(rho / pixel_mm)),
                        quality="sector_median",
                    )
                    for k, (rho, p) in enumerate(merged)
                ]
                if not analysis.sector_notes:
                    analysis.sector_notes = "Sector consensus supplemented dark-ring detection."

    return analysis


def analysis_to_rows(
    analysis: RadialAnalysis,
    *,
    dz_stage_mm: float,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ext in analysis.bright + analysis.dark:
        rows.append(
            {
                "dz_stage_mm": dz_stage_mm,
                "source": source,
                "extremum": ext.kind,
                "fringe_order": ext.fringe_order,
                "radius_mm": round(ext.radius_mm, 4),
                "radius_sq_mm2": round(ext.radius_sq_mm2, 6),
                "prominence": round(ext.prominence, 6),
                "detection_quality": ext.quality,
                "center_cy_px": round(analysis.center.cy, 3),
                "center_cx_px": round(analysis.center.cx, 3),
                "center_method": analysis.center.method,
                "sector_notes": analysis.sector_notes,
            }
        )
    return rows
