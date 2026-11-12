"""Fringe-center spacing extraction for manuscript Figure 6(a)."""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

Modality = Literal["simulation", "experiment"]

PIXEL_UM = 13.0
PIXEL_MM = PIXEL_UM * 1e-3
ORIENT_SMOOTH_SIGMA_PX = 3.0
N_PROFILE_BINS = 720


def signal_mask(img: np.ndarray, *, floor_frac: float = 0.12) -> np.ndarray:
    z = np.asarray(img, float)
    z -= float(np.min(z))
    peak = float(np.max(z)) + 1e-12
    return (z >= floor_frac * peak).astype(bool)


def estimate_normal_deg(img: np.ndarray, mask: np.ndarray) -> float:
    z = gaussian_filter(np.where(mask, img - float(np.mean(img[mask])), 0.0), ORIENT_SMOOTH_SIGMA_PX)
    h, w = z.shape
    wy, wx = np.hanning(h), np.hanning(w)
    spec = np.fft.fftshift(np.fft.fft2(z * np.outer(wy, wx)))
    mag = np.abs(spec)
    cy, cx = h // 2, w // 2
    mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    yy, xx = np.indices(mag.shape)
    rr = np.hypot(yy - cy, xx - cx)
    band = (rr >= 2) & (rr <= min(h, w) * 0.14)
    mag[~band] = 0
    if float(mag.max()) <= 0:
        return 0.0
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
    return float(math.degrees(math.atan2(iy - cy, ix - cx)))


def build_profile(
    img: np.ndarray,
    mask: np.ndarray,
    normal_deg: float,
    *,
    smooth_sigma_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    h, w = img.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    phi = math.radians(normal_deg)
    u = xx * math.cos(phi) + yy * math.sin(phi)
    v = -xx * math.sin(phi) + yy * math.cos(phi)
    base = np.where(mask, img.astype(np.float64), np.nan)
    base -= float(np.nanmean(base))
    if smooth_sigma_px > 0:
        filled = np.where(mask, base, 0.0)
        sm = gaussian_filter(filled, float(smooth_sigma_px))
        cnt = gaussian_filter(mask.astype(float), float(smooth_sigma_px))
        base = np.where(mask, sm / np.maximum(cnt, 1e-9), np.nan)
    u_flat = u[mask]
    val = base[mask]
    umin, umax = float(u_flat.min()), float(u_flat.max())
    edges = np.linspace(umin, umax, N_PROFILE_BINS + 1)
    prof = np.full(N_PROFILE_BINS, np.nan)
    for i in range(N_PROFILE_BINS):
        sel = (u_flat >= edges[i]) & (u_flat < edges[i + 1])
        if np.count_nonzero(sel) >= 2:
            prof[i] = float(np.nanmean(val[sel]))
    u_centers = 0.5 * (edges[:-1] + edges[1:])
    du_px = float(u_centers[1] - u_centers[0]) if N_PROFILE_BINS > 1 else 1.0
    v_mean = float(np.mean(v[mask]))
    return u_centers, prof, du_px, v_mean


def estimate_period_bins(u_centers: np.ndarray, prof: np.ndarray, du_px: float) -> tuple[float, float]:
    m = np.isfinite(prof)
    y = np.asarray(prof[m], float)
    x_mm = u_centers[m] * PIXEL_MM
    if y.size < 40:
        return float("nan"), float("nan")
    y = y - float(np.mean(y))
    dx_mm = float(np.mean(np.diff(x_mm))) if x_mm.size > 1 else du_px * PIXEL_MM
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dx_mm)
    spec[0] = 0.0
    band = (freqs >= 0.25) & (freqs <= 6.0)
    if not np.any(band):
        return float("nan"), float("nan")
    ib = np.where(band)[0]
    i_peak = ib[int(np.argmax(spec[band]))]
    k_hint = float(freqs[i_peak])
    period_mm = 1.0 / k_hint if k_hint > 1e-6 else float("nan")
    period_bins = period_mm / (du_px * PIXEL_MM) if math.isfinite(period_mm) else float("nan")
    return period_bins, k_hint


def adaptive_smooth_sigma_px(period_bins: float) -> float:
    if not math.isfinite(period_bins) or period_bins <= 0:
        return 2.0
    return float(min(6.0, max(1.0, period_bins / 8.0)))


def fft_diagnostic_k(u_centers: np.ndarray, prof: np.ndarray, du_px: float) -> dict[str, Any]:
    """Internal diagnostic only — not plotted on manuscript figure."""
    m = np.isfinite(prof)
    y = np.asarray(prof[m], float)
    x_mm = u_centers[m] * PIXEL_MM
    if y.size < 40:
        return {"k_fft_cycles_per_mm": float("nan"), "fft_note": "short_profile"}
    env = gaussian_filter1d(y, max(5, y.size // 12))
    mod = (y - env) * np.hanning(y.size)
    mod -= float(np.mean(mod))
    dx_mm = float(np.mean(np.diff(x_mm)))
    freqs = np.fft.rfftfreq(mod.size, d=dx_mm)
    spec = np.abs(np.fft.rfft(mod))
    spec[0] = 0
    band = (freqs >= 0.25) & (freqs <= 7.0)
    if not np.any(band):
        return {"k_fft_cycles_per_mm": float("nan"), "fft_note": "no_peak"}
    ib = np.where(band)[0]
    j = ib[int(np.argmax(spec[band]))]
    return {"k_fft_cycles_per_mm": float(freqs[j]), "fft_note": "envelope_corrected"}


def _detect_peaks(
    u_centers: np.ndarray,
    prof: np.ndarray,
    du_px: float,
    *,
    modality: Modality,
    dx_um: float,
    period_bins: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    m = np.isfinite(prof)
    x = u_centers[m]
    y = np.asarray(prof[m], float)
    if y.size < 20:
        return np.array([]), {"reason": "short_profile"}
    y0 = y - float(np.min(y))
    rng = float(np.max(y0) - np.min(y0)) + 1e-12

    if modality == "simulation" and dx_um <= 30.0:
        y_use = y0
        sigma_px = 0.0
        min_dist = max(3, int(0.30 * period_bins)) if math.isfinite(period_bins) else 4
        rel = 0.03
        prom = max(rel * rng, 0.015 * float(np.std(y_use)))
    else:
        # Profile may already include adaptive 2D pre-smooth; do not low-pass again in 1D.
        y_use = y0
        sigma_px = 0.0
        if math.isfinite(period_bins) and period_bins > 2:
            min_dist = max(2, int(0.45 * period_bins))
        else:
            min_dist = 4
        prom = max(0.06 * rng, 0.02 * float(np.std(y_use)))

    idx, _ = find_peaks(y_use, prominence=prom, distance=min_dist)
    meta = {
        "prominence": prom,
        "min_distance_bins": min_dist,
        "sigma_px": sigma_px,
        "n_peaks": int(idx.size),
        "period_bins_est": period_bins,
    }
    return x[idx], meta


def extract_center_spacing(
    img: np.ndarray,
    *,
    dx_um: float,
    modality: Modality,
) -> dict[str, Any]:
    floor_frac = 0.08 if (modality == "simulation" and dx_um <= 30.0) else 0.12
    mask = signal_mask(img, floor_frac=floor_frac)
    normal_deg = estimate_normal_deg(img, mask)
    u0, p0, du, _vmean = build_profile(img, mask, normal_deg, smooth_sigma_px=0.0)
    period_bins, k_hint = estimate_period_bins(u0, p0, du)

    if modality == "simulation" and dx_um <= 30.0:
        u_use, p_use = u0, p0
        pre_sigma = 0.0
    else:
        pre_sigma = adaptive_smooth_sigma_px(period_bins)
        u_use, p_use, _, _ = build_profile(img, mask, normal_deg, smooth_sigma_px=pre_sigma)

    peaks_u, peak_meta = _detect_peaks(
        u_use, p_use, du, modality=modality, dx_um=dx_um, period_bins=period_bins
    )
    fft_diag = fft_diagnostic_k(u0, p0, du)

    n_centers = int(peaks_u.size)
    freq: dict[str, Any] = {
        "k_cycles_per_mm": float("nan"),
        "measurable": False,
        "mean_spacing_mm": float("nan"),
        "spacing_std_mm": float("nan"),
        "spacing_cv": float("nan"),
        "n_spacings": 0,
    }
    status = "not_measurable"

    if dx_um == 0.0:
        status = "reference_zero_no_fringe_modulation"
    elif n_centers >= 2:
        sp = np.diff(np.sort(peaks_u)) * PIXEL_MM
        mu = float(np.mean(sp))
        sd = float(np.std(sp)) if sp.size > 1 else 0.0
        cv = sd / mu if mu > 1e-9 else float("nan")
        k = 1.0 / mu if mu > 1e-9 else float("nan")
        freq = {
            "k_cycles_per_mm": k,
            "measurable": math.isfinite(k),
            "mean_spacing_mm": mu,
            "spacing_std_mm": sd,
            "spacing_cv": cv,
            "n_spacings": int(sp.size),
        }
        if n_centers == 2:
            status = "measured_single_spacing"
        elif cv > 0.25 if math.isfinite(cv) else True:
            status = "measured_high_spacing_variability"
        else:
            status = "measured"
    elif n_centers == 1:
        status = "single_center_only"
    else:
        status = "no_peaks"

    k_plot = 0.0 if dx_um == 0.0 else (freq["k_cycles_per_mm"] if freq["measurable"] else float("nan"))

    return {
        "dx_um": dx_um,
        "modality": modality,
        "n_centers": n_centers,
        "fringe_normal_deg": normal_deg,
        "k_center_spacing_cycles_per_mm": freq["k_cycles_per_mm"],
        "k_plotted_cycles_per_mm": k_plot,
        "mean_spacing_mm": freq.get("mean_spacing_mm", float("nan")),
        "spacing_std_mm": freq.get("spacing_std_mm", float("nan")),
        "spacing_cv": freq.get("spacing_cv", float("nan")),
        "n_spacings": freq.get("n_spacings", 0),
        "center_spacing_measurable": freq["measurable"],
        "measurement_status": status,
        "period_bins_est": period_bins,
        "profile_pre_smooth_sigma_px": pre_sigma,
        "k_hint_profile_fft": k_hint,
        "k_fft_diagnostic_cycles_per_mm": fft_diag.get("k_fft_cycles_per_mm", float("nan")),
        "fft_diagnostic_note": fft_diag.get("fft_note", ""),
        "peak_prominence": peak_meta.get("prominence", ""),
        "peak_min_distance_bins": peak_meta.get("min_distance_bins", ""),
    }


def extract_series(
    images: dict[float, np.ndarray],
    dx_list: list[float],
    modality: Modality,
) -> list[dict[str, Any]]:
    return [
        extract_center_spacing(images[dx], dx_um=float(dx), modality=modality)
        for dx in dx_list
        if dx in images
    ]
