"""C2 longitudinal experimental metrics: rings, a(δ), visibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.physics.independent_equations import (
    equivalent_wavelength_nm,
    idler_lens_equivalent_propagation_um,
    longitudinal_quadratic_coefficient_per_um2,
)


def read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def load_mean_std(files: list[dict[str, Any]]) -> dict[str, Any]:
    frames = []
    loaded = []
    failed = []
    for f in files:
        try:
            img = read_tiff(Path(f["absolute_path"]))
            frames.append(img)
            loaded.append(f["relative_path"])
        except Exception as exc:  # noqa: BLE001
            failed.append({"path": f["relative_path"], "error": repr(exc)})
    if not frames:
        return {"status": "LOAD_FAILED", "failed": failed}
    # enforce identical shapes — do not resize
    shapes = {im.shape for im in frames}
    if len(shapes) != 1:
        return {
            "status": "SHAPE_MISMATCH",
            "shapes": [list(s) for s in shapes],
            "note": "Refusing to resize; select a single primary shape upstream.",
        }
    stack = np.stack(frames, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    return {
        "status": "OK",
        "mean": mean,
        "std": std,
        "n_repeats": int(stack.shape[0]),
        "shape": list(mean.shape),
        "loaded": loaded,
        "failed": failed,
        "resize_applied": False,
        "crop_applied": False,
    }


def estimate_ring_center(img: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid after high-pass-ish demean."""
    z = img - float(np.mean(img))
    w = np.abs(z)
    w = w / (np.max(w) + 1.0e-12)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    s = float(np.sum(w)) + 1.0e-12
    cy = float(np.sum(w * yy) / s)
    cx = float(np.sum(w * xx) / s)
    return cy, cx


def radial_profile(
    img: np.ndarray, cy: float, cx: float, *, n_bins: int | None = None
) -> dict[str, np.ndarray]:
    h, w = img.shape
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.hypot(yy - cy, xx - cx)
    rmax = float(min(cy, cx, h - 1 - cy, w - 1 - cx))
    nb = int(n_bins or max(16, int(rmax)))
    bins = np.linspace(0.0, rmax, nb + 1)
    inds = np.digitize(rr.ravel(), bins) - 1
    prof = np.zeros(nb, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.float64)
    flat = img.ravel()
    for i, v in zip(inds, flat):
        if 0 <= i < nb:
            prof[i] += v
            counts[i] += 1.0
    mask = counts > 0
    prof[mask] /= counts[mask]
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return {
        "r_px": r_centers,
        "profile": prof,
        "counts": counts,
        "rmax_px": np.array([rmax]),
    }


def detect_ring_radii(r_px: np.ndarray, profile: np.ndarray) -> dict[str, Any]:
    """Peak/valley detection on radial profile → fringe orders."""
    if profile.size < 8:
        return {"status": "TOO_SHORT", "radii_px": [], "orders": []}
    # Smooth lightly
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    sm = np.convolve(profile, kernel, mode="same")
    # local maxima
    peaks = []
    for i in range(2, len(sm) - 2):
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > sm[i - 2] and sm[i] > sm[i + 2]:
            peaks.append(i)
    # keep significant peaks
    if not peaks:
        return {"status": "NO_PEAKS", "radii_px": [], "orders": [], "profile_smooth": sm}
    vals = sm[peaks]
    thr = float(np.median(sm) + 0.15 * (np.max(sm) - np.median(sm)))
    peaks = [p for p, v in zip(peaks, vals) if v >= thr]
    radii = [float(r_px[p]) for p in peaks if r_px[p] > 2.0]
    orders = list(range(1, len(radii) + 1))
    return {
        "status": "OK" if radii else "NO_PEAKS",
        "radii_px": radii,
        "orders": orders,
        "profile_smooth": sm,
        "peak_indices": peaks,
    }


def fit_rho2_vs_order(
    radii_px: list[float], orders: list[int], *, pixel_um: float
) -> dict[str, Any]:
    if len(radii_px) < 2:
        return {"status": "INSUFFICIENT_RINGS", "n": len(radii_px)}
    rho_um = np.asarray(radii_px, dtype=np.float64) * float(pixel_um)
    rho2 = rho_um * rho_um
    n = np.asarray(orders, dtype=np.float64)
    # rho2 = (1/a) * n + c  if a*rho2 = n + n0 with φ=2π a ρ²
    coef = np.polyfit(n, rho2, 1)
    slope = float(coef[0])  # µm² per order
    intercept = float(coef[1])
    pred = slope * n + intercept
    rmse = float(np.sqrt(np.mean((rho2 - pred) ** 2)))
    a_from_slope = float(1.0 / slope) if abs(slope) > 0 else float("nan")
    return {
        "status": "OK",
        "slope_um2_per_order": slope,
        "intercept_um2": intercept,
        "a_exp_per_um2": a_from_slope,
        "rmse_um2": rmse,
        "rho2_um2": rho2.tolist(),
        "orders": orders,
        "pixel_um_used": float(pixel_um),
        "pixel_um_provenance": "ASSUMED_STOCK_C_16UM",
    }


def visibility_metrics(img: np.ndarray, cy: float, cx: float) -> dict[str, float]:
    h, w = img.shape
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.hypot(yy - cy, xx - cx)
    rmax = float(min(cy, cx, h - 1 - cy, w - 1 - cx))
    roi = rr <= 0.6 * rmax
    vals = img[roi]
    if vals.size == 0:
        return {"V_roi": float("nan"), "V_local_proxy": float("nan")}
    vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    V_roi = (vmax - vmin) / (vmax + vmin + 1.0e-12)
    # local proxy: std/mean
    V_local = float(np.std(vals) / (np.mean(vals) + 1.0e-12))
    return {"V_roi": float(V_roi), "V_local_proxy": V_local}


def theory_a(
    delta_z_um: float,
    *,
    f_u_mm: float = 75.0,
    f_c_mm: float = 150.0,
    lambda_s_nm: float = 810.0,
    lambda_i_nm: float = 1550.0,
    pass_factor: float = 2.0,
) -> dict[str, float]:
    a = longitudinal_quadratic_coefficient_per_um2(
        delta_z_um=delta_z_um,
        f_u_mm=f_u_mm,
        f_c_mm=f_c_mm,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        pass_factor=pass_factor,
    )
    d = idler_lens_equivalent_propagation_um(delta_z_um, f_u_mm)
    return {
        "a_theory_per_um2": float(a),
        "d_um": float(d),
        "lambda_eq_nm": float(equivalent_wavelength_nm(lambda_s_nm, lambda_i_nm)),
        "f_u_mm": float(f_u_mm),
        "f_c_mm": float(f_c_mm),
        "pass_factor": float(pass_factor),
    }


def analyze_longitudinal_mean(
    mean_img: np.ndarray,
    *,
    delta_z_um: float,
    pixel_um_assumed: float = 16.0,
    f_u_mm: float = 75.0,
) -> dict[str, Any]:
    cy, cx = estimate_ring_center(mean_img)
    rad = radial_profile(mean_img, cy, cx)
    rings = detect_ring_radii(rad["r_px"], rad["profile"])
    fit = fit_rho2_vs_order(
        rings.get("radii_px") or [],
        rings.get("orders") or [],
        pixel_um=pixel_um_assumed,
    )
    vis = visibility_metrics(mean_img, cy, cx)
    th = theory_a(delta_z_um, f_u_mm=f_u_mm)
    return {
        "center_y_px": cy,
        "center_x_px": cx,
        "radial": {
            "r_px": rad["r_px"],
            "profile": rad["profile"],
        },
        "rings": {k: v for k, v in rings.items() if k != "profile_smooth"},
        "rho2_fit": fit,
        "visibility": vis,
        "theory": th,
        "a_exp_per_um2": fit.get("a_exp_per_um2"),
        "a_theory_per_um2": th["a_theory_per_um2"],
    }
