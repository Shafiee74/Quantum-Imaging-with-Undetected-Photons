#!/usr/bin/env python3
"""Figure 6 — C_lab idler-lens displacement validation (no-object, production forward)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec

from camera.c_lab_display_zoom import export_zoom_config_from_setup, resolve_fixed_crop_window
from setup_params import equivalent_wavelength_nm, idler_lens_equivalent_propagation_um, resolve_setup_selection
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.c_lab_rabbit_object_orientation_correction import DISPLAY_ROT90_K, N
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6_lens_displacement_validation"
EXP_ROOT = ROOT / "exp data" / "lens drift" / "witout object"
X0_MM = 4.340
Z0_MM = 6.910
TRANS_DX_UM = (0.0, 20.0, 30.0, 40.0, 60.0, 180.0)
AXIAL_DZ_MM = (0.0, 2.400, 3.580, 5.812)
AXIAL_DZ_EFF_MM = tuple(2.0 * value for value in AXIAL_DZ_MM)
PIXEL_UM = 13.0


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_abs_mm(name: str) -> float | None:
    m = re.search(r"(?:T=|L=)?(\d+)[_-](\d+)", name)
    if not m:
        return None
    return float(m.group(1)) + float(m.group(2)) / (10 ** len(m.group(2)))


def group_tiffs(folder: Path, *, ref_mm: float, rel_scale: float) -> dict[float, list[Path]]:
    """rel_scale: 1000 for transverse (um), 1 for axial (mm)."""
    out: dict[float, list[Path]] = {}
    for p in sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff")):
        abs_mm = parse_abs_mm(p.name)
        if abs_mm is None:
            continue
        rel = (abs_mm - ref_mm) * rel_scale
        out.setdefault(round(rel, 6), []).append(p)
    return out


def mean_stack(paths: list[Path]) -> dict[str, Any]:
    frames = [tifffile.imread(p).astype(np.float64) for p in paths]
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        from collections import Counter

        shape = Counter(f.shape for f in frames).most_common(1)[0][0]
        frames = [f for f in frames if f.shape == shape]
    stack = np.stack(frames, axis=0)
    return {
        "mean": np.mean(stack, axis=0),
        "std": np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0]),
        "n": int(stack.shape[0]),
        "shape": stack.shape[1:],
        "paths": [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths[: len(frames)]],
    }


def theory_display(i_plus: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    from camera.c_lab_display_zoom import apply_fixed_crop

    oriented = np.rot90(np.asarray(i_plus, dtype=np.float64), k=DISPLAY_ROT90_K)
    return apply_fixed_crop(oriented, bounds)


def run_theory(
    setup_base,
    dk: DeltaKCalibrationState,
    bounds: dict[str, int],
    *,
    dx_um: float = 0.0,
    dz_um: float = 0.0,
) -> np.ndarray:
    """Forward I_+ for Figure 6 lens-displacement panels.

    Idler-lens stage ``delta_z_um`` / ``lens_shift_*`` drive the L06 Haidinger
    phase on the idler seed. That mechanical idler move must **not** also set the
    C7 signal-imaging defocus PSF (``σ_defocus = K|Δz|``): doing so blurs away
    higher-order rings and under-counts rings relative to experiment. Imaging
    defocus is therefore forced to zero for these theory panels only.
    """
    setup = setup_base.overridden(
        enable_lens_rings=True,
        lens_shift_x_um=float(dx_um),
        lens_shift_y_um=0.0,
        delta_z_um=float(dz_um),
        # Decouple idler-lens Δz from signal-path optical defocus blur (C7).
        use_derived_K_defocus=False,
        defocus_span_px=0.0,
        K_defocus_px_per_um=0.0,
    )
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    case = run_forward_case(setup, ref, dk, grid_n=N)
    return theory_display(case["I_plus"], bounds)


def weighted_centroid(img: np.ndarray) -> tuple[float, float]:
    z = np.asarray(img, dtype=np.float64)
    z = z - float(np.percentile(z, 5))
    w = np.clip(z, 0, None)
    s = float(w.sum()) + 1e-12
    yy, xx = np.indices(z.shape, dtype=np.float64)
    return float(np.sum(w * yy) / s), float(np.sum(w * xx) / s)


def fft_fringe_metrics(img: np.ndarray) -> dict[str, Any]:
    z = np.asarray(img, dtype=np.float64)
    z = z - float(np.mean(z))
    wy, wx = np.hanning(z.shape[0]), np.hanning(z.shape[1])
    spec = np.fft.fftshift(np.fft.fft2(z * np.outer(wy, wx)))
    mag = np.abs(spec)
    cy, cx = z.shape[0] // 2, z.shape[1] // 2
    mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    yy, xx = np.indices(mag.shape)
    rr = np.hypot(yy - cy, xx - cx)
    band = (rr >= 4) & (rr <= min(z.shape) * 0.35)
    masked = np.where(band, mag, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    fy = (iy - cy) / float(z.shape[0])
    fx = (ix - cx) / float(z.shape[1])
    kappa_cyc_px = float(math.hypot(fx, fy))
    return {
        "kappa_cycles_per_px": kappa_cyc_px,
        "kappa_cycles_per_mm": kappa_cyc_px / (PIXEL_UM * 1e-3) if PIXEL_UM > 0 else float("nan"),
        "orientation_deg": float(math.degrees(math.atan2(fy, fx))) if kappa_cyc_px > 0 else float("nan"),
    }


def line_profile(img: np.ndarray, cy: float, cx: float, angle_deg: float, half_len: int) -> dict[str, np.ndarray]:
    ang = math.radians(angle_deg + 90.0)
    t = np.arange(-half_len, half_len + 1, dtype=np.float64)
    yy = cy + t * math.sin(ang)
    xx = cx + t * math.cos(ang)
    h, w = img.shape
    vals = []
    for y, x in zip(yy, xx):
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < h and 0 <= ix < w:
            vals.append(float(img[iy, ix]))
        else:
            vals.append(float("nan"))
    pos_mm = t * PIXEL_UM * 1e-3
    arr = np.asarray(vals, dtype=np.float64)
    rel = arr.copy()
    bg = float(np.nanmedian(rel))
    rel = rel - bg
    pk = float(np.nanmax(rel)) if np.isfinite(rel).any() else 1.0
    if pk > 0:
        rel = rel / pk
    return {"pos_mm": pos_mm, "raw": arr, "relative": rel}


def radial_azimuthal_mean(img: np.ndarray, cy: float, cx: float) -> dict[str, np.ndarray]:
    yy, xx = np.indices(img.shape, dtype=np.float64)
    rr = np.hypot(yy - cy, xx - cx)
    rmax = int(min(cy, cx, img.shape[0] - 1 - cy, img.shape[1] - 1 - cx))
    prof = np.full(rmax + 1, np.nan, dtype=np.float64)
    for ri in range(rmax + 1):
        m = (rr >= ri - 0.5) & (rr < ri + 0.5)
        if np.any(m):
            prof[ri] = float(np.mean(img[m]))
    r_mm = np.arange(prof.size, dtype=np.float64) * PIXEL_UM * 1e-3
    rel = prof - float(np.nanmedian(prof))
    pk = float(np.nanmax(rel)) if np.isfinite(rel).any() else 1.0
    if pk > 0:
        rel = rel / pk
    return {"r_mm": r_mm, "raw": prof, "relative": rel}


def theory_kappa_cycles_per_mm(dx_um: float, setup) -> float:
    lam_eq_um = equivalent_wavelength_nm(setup.lambda_s_nm, setup.lambda_i_nm) * 1e-3
    f_c_um = float(setup.F2_mm) * 1e3
    g = 2.0
    if abs(dx_um) < 1e-9:
        return float("nan")
    return g * abs(dx_um) / (lam_eq_um * f_c_um) * 1e3


def save_pair_figure(
    path: Path,
    theory: np.ndarray,
    exp: np.ndarray,
    title: str,
    t_vmax: float,
    e_vmax: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 5))
    axes[0].imshow(theory, cmap="gray", origin="upper", vmin=0, vmax=t_vmax)
    axes[0].set_title("Simulation\n" + title, fontsize=8)
    axes[1].imshow(exp, cmap="gray", origin="upper", vmin=0, vmax=e_vmax)
    axes[1].set_title("Experimental\n" + title, fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    supp_t = out / "supplement" / "transverse"
    supp_a = out / "supplement" / "axial"
    supp_t.mkdir(parents=True, exist_ok=True)
    supp_a.mkdir(parents=True, exist_ok=True)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref_clear = ref_from_T(np.ones((N, N), dtype=np.float32))
    I_ref = run_forward_case(
        setup.overridden(enable_lens_rings=True),
        ref_clear,
        dk,
        grid_n=N,
    )["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))

    trans_groups = group_tiffs(EXP_ROOT / "T drift", ref_mm=X0_MM, rel_scale=1000.0)
    axial_groups = group_tiffs(EXP_ROOT / "L drift", ref_mm=Z0_MM, rel_scale=1.0)

    inventory: list[dict[str, Any]] = []
    for rel, paths in sorted(trans_groups.items()):
        inventory.append(
            {
                "scan": "transverse",
                "stage_folder": "T drift",
                "relative_displacement": rel,
                "relative_units": "um",
                "abs_stage_mm": X0_MM + rel / 1000.0,
                "n_tiff": len(paths),
                "example_file": paths[0].name,
            }
        )
    for rel, paths in sorted(axial_groups.items()):
        inventory.append(
            {
                "scan": "axial",
                "stage_folder": "L drift",
                "relative_displacement": rel,
                "relative_units": "mm",
                "abs_stage_mm": Z0_MM + rel,
                "n_tiff": len(paths),
                "example_file": paths[0].name,
            }
        )

    # --- Transverse ---
    trans_rows: list[dict] = []
    trans_theory_imgs: dict[float, np.ndarray] = {}
    trans_exp_imgs: dict[float, np.ndarray] = {}
    trans_profiles: list[dict] = []
    ref_cy, ref_cx = weighted_centroid(mean_stack(trans_groups[0.0])["mean"]) if 0.0 in trans_groups else (0, 0)

    for dx in TRANS_DX_UM:
        key = round(dx, 6)
        if key not in trans_groups:
            continue
        exp_m = mean_stack(trans_groups[key])
        trans_exp_imgs[dx] = exp_m["mean"]
        th = run_theory(setup, dk, bounds, dx_um=dx, dz_um=0.0)
        trans_theory_imgs[dx] = th
        cy, cx = ref_cy, ref_cx
        fft_th = fft_fringe_metrics(th)
        fft_ex = fft_fringe_metrics(exp_m["mean"])
        ang = fft_ex["orientation_deg"] if np.isfinite(fft_ex["orientation_deg"]) else 0.0
        prof_th = line_profile(th, cy, cx, ang, half_len=min(th.shape) // 3)
        prof_ex = line_profile(exp_m["mean"], cy, cx, ang, half_len=min(exp_m["mean"].shape) // 3)
        k_th = theory_kappa_cycles_per_mm(dx, setup)
        trans_rows.append(
            {
                "dx_um": dx,
                "kappa_theory_cycles_per_mm": k_th,
                "kappa_exp_cycles_per_mm": fft_ex["kappa_cycles_per_mm"],
                "orientation_deg": ang,
                "n_exp_frames": exp_m["n"],
                "exp_shape_h_w": str(exp_m["shape"]),
            }
        )
        nprof = min(len(prof_th["pos_mm"]), len(prof_ex["pos_mm"]))
        for i in range(nprof):
            trans_profiles.append(
                {
                    "dx_um": dx,
                    "pos_mm": prof_th["pos_mm"][i],
                    "theory_raw_au": prof_th["raw"][i],
                    "theory_relative": prof_th["relative"][i],
                    "exp_raw_adu": prof_ex["raw"][i],
                    "exp_relative": prof_ex["relative"][i],
                }
            )
        save_pair_figure(
            supp_t / f"pair_dx_{int(dx):03d}um.png",
            th,
            exp_m["mean"],
            f"Δx = {dx:.0f} µm",
            float(np.percentile(th, 99.5)),
            float(np.percentile(exp_m["mean"], 99.5)),
        )

    # --- Axial ---
    axial_rows: list[dict] = []
    axial_theory: dict[float, np.ndarray] = {}
    axial_exp: dict[float, np.ndarray] = {}
    axial_profiles: list[dict] = []
    ref_cy_a, ref_cx_a = weighted_centroid(mean_stack(axial_groups[0.0])["mean"]) if 0.0 in axial_groups else (0, 0)

    for dz_mm in AXIAL_DZ_MM:
        key = round(dz_mm, 6)
        if key not in axial_groups:
            continue
        dz_um = dz_mm * 1000.0
        exp_m = mean_stack(axial_groups[key])
        axial_exp[dz_mm] = exp_m["mean"]
        th = run_theory(setup, dk, bounds, dx_um=0.0, dz_um=dz_um)
        axial_theory[dz_mm] = th
        rad_th = radial_azimuthal_mean(th, ref_cy_a, ref_cx_a)
        rad_ex = radial_azimuthal_mean(exp_m["mean"], ref_cy_a, ref_cx_a)
        d_eq = idler_lens_equivalent_propagation_um(dz_um, float(setup.far_field_focal_mm))
        axial_rows.append(
            {
                "z_stage_mm": dz_mm,
                "z_eff_mm": 2.0 * dz_mm,
                "delta_z_stage_um_forward": dz_um,
                "d_eq_um": d_eq,
                "pass_factor": 2.0,
                "equivalent_distance_pre_doubled": False,
                "n_exp_frames": exp_m["n"],
                "exp_shape_h_w": str(exp_m["shape"]),
            }
        )
        n = min(len(rad_th["r_mm"]), len(rad_ex["r_mm"]))
        for i in range(n):
            axial_profiles.append(
                {
                    "z_stage_mm": dz_mm,
                    "z_eff_mm": 2.0 * dz_mm,
                    "r_mm": rad_th["r_mm"][i],
                    "theory_raw_au": rad_th["raw"][i],
                    "theory_relative": rad_th["relative"][i],
                    "exp_raw_adu": rad_ex["raw"][i],
                    "exp_relative": rad_ex["relative"][i],
                }
            )
        save_pair_figure(
            supp_a / f"pair_dz_{dz_mm:.3f}mm.png",
            th,
            exp_m["mean"],
            f"Δz_eff = {2.0 * dz_mm:.3f} mm",
            float(np.percentile(th, 99.5)),
            float(np.percentile(exp_m["mean"], 99.5)),
        )

    # --- Figure 6A ---
    dx_list = [d for d in TRANS_DX_UM if d in trans_theory_imgs]
    ncols = len(dx_list)
    t_vmax = max(np.percentile(trans_theory_imgs[d], 99.5) for d in dx_list)
    e_vmax = max(np.percentile(trans_exp_imgs[d], 99.5) for d in dx_list)
    fig = plt.figure(figsize=(2.6 * ncols, 7))
    gs = GridSpec(3, ncols, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.15)
    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(trans_theory_imgs[dx], cmap="gray", origin="upper", vmin=0, vmax=t_vmax)
        ax.set_title(f"Simulation\nΔx = {dx:.0f} µm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(trans_exp_imgs[dx], cmap="gray", origin="upper", vmin=0, vmax=e_vmax)
        ax.set_title(f"Experimental\nΔx = {dx:.0f} µm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axs = fig.add_subplot(gs[2, :])
    xs = [r["dx_um"] for r in trans_rows]
    axs.plot(xs, [r["kappa_theory_cycles_per_mm"] for r in trans_rows], "o-", label="Simulation")
    axs.plot(xs, [r["kappa_exp_cycles_per_mm"] for r in trans_rows], "s--", label="Experimental")
    axs.set_xlabel("Transverse idler-lens displacement Δx (µm)")
    axs.set_ylabel("Fringe spatial frequency (cycles/mm)")
    axs.legend(fontsize=8)
    fig.suptitle("Figure 6(a). Transverse idler-lens displacement", fontsize=12, y=0.98)
    fig.savefig(out / "FIGURE6A_TRANSVERSE_LENS_DISPLACEMENT.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIGURE6A_TRANSVERSE_LENS_DISPLACEMENT.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 6B ---
    dz_list = [d for d in AXIAL_DZ_MM if d in axial_theory]
    ncols = len(dz_list)
    t_vmax = max(np.percentile(axial_theory[d], 99.5) for d in dz_list)
    e_vmax = max(np.percentile(axial_exp[d], 99.5) for d in dz_list)
    fig = plt.figure(figsize=(2.8 * ncols, 7))
    gs = GridSpec(3, ncols, height_ratios=[1, 1, 0.9], hspace=0.35, wspace=0.15)
    for j, dz in enumerate(dz_list):
        z_eff = 2.0 * dz
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(axial_theory[dz], cmap="gray", origin="upper", vmin=0, vmax=t_vmax)
        ax.set_title(f"Simulation\nΔz_eff = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(axial_exp[dz], cmap="gray", origin="upper", vmin=0, vmax=e_vmax)
        ax.set_title(f"Experimental\nΔz_eff = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axs = fig.add_subplot(gs[2, :])
    axs.plot([r["z_eff_mm"] for r in axial_rows], [r["d_eq_um"] for r in axial_rows], "o-", label="Equivalent d (simulation model)")
    axs.set_xlabel("Effective axial idler-lens displacement Δz_eff (mm)")
    axs.set_ylabel("Equivalent propagation d (µm)")
    axs.legend(fontsize=8)
    fig.suptitle("Figure 6(b). Axial idler-lens displacement", fontsize=12, y=0.98)
    fig.savefig(out / "FIGURE6B_AXIAL_LENS_DISPLACEMENT.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIGURE6B_AXIAL_LENS_DISPLACEMENT.pdf", bbox_inches="tight")
    plt.close(fig)

    # CSV outputs
    with (out / "LENS_DISPLACEMENT_INVENTORY.csv").open("w", newline="", encoding="utf-8") as fh:
        if inventory:
            w = csv.DictWriter(fh, fieldnames=list(inventory[0].keys()))
            w.writeheader()
            w.writerows(inventory)
    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({k for row in rows for k in row})
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "TRANSVERSE_SPATIAL_PROFILES.csv", trans_profiles)
    write_csv(out / "AXIAL_RADIAL_PROFILES.csv", axial_profiles)
    write_csv(out / "FRINGE_SPACING_COMPARISON.csv", trans_rows)
    write_csv(out / "RING_RADII_COMPARISON.csv", axial_rows)
    write_csv(
        out / "LENS_DISPLACEMENT_SUMMARY.csv",
        [{**r, "scan": "transverse"} for r in trans_rows]
        + [{**r, "scan": "axial"} for r in axial_rows],
    )

    compat = f"""# Physical model compatibility — Figure 6

## Production forward path
- **Model:** `optics._optical_intensity` → `idler_lens_misalignment_phase_map` applied to idler seed before C12 when `enable_lens_rings=True`.
- **Imaging PSF:** Figure 6 theory forces `defocus_span_px=0` / `K_defocus=0` so idler-lens stage Δz does not drive C7 signal-defocus blur (which otherwise washes out higher-order rings).
- **Transverse:** `lens_shift_x_um`, `lens_shift_y_um` (linear tilt term, λ_eq, f_c, pass factor 2 for single-crystal Michelson).
- **Axial:** mechanical `delta_z_um` enters `d = f²δ/(f²+δ²)` once; the production single-crystal `pass_factor = 2` supplies the round-trip phase. Display labels use `Δz_eff = 2 Δz_stage`.
- **C_lab default YAML:** `enable_lens_rings=False`; Figure 6 theory runs use **runtime overrides only** (no YAML/production edits).

## Applicability vs Hochrainer et al. (2017)
- Reference two-crystal / 4f idler lens formulas are **conceptual benchmarks** only.
- C_lab uses **single-crystal double-pass Michelson**; production implements pass factor 2 and C_lab focal lengths (f_u = f_c = 150 mm, Mi = 2 display path unchanged).

## Experimental axis verification
- **T drift** folders: transverse stage coordinate (absolute mm in filenames); reference **x₀ = 4.340 mm** → Δx in µm.
- **L drift** folders: mechanical axial stage coordinate; reference **z₀ = 6.910 mm** → `Δz_stage` in mm.
- Figure labels report **effective displacement** `Δz_eff = 2 Δz_stage`; the mechanical value is supplied to the forward model once, without pre-doubling.

## Experimental TIFF notes
- Transverse means: typically **411×453** px; axial means: **615×679** px (native EMCCD export, no resize).
- Simulation display: native 1024 → rot90(k={DISPLAY_ROT90_K}) → fixed crop **419×455**.

## Blocker status
**None** for phase-map-based lens perturbations when `enable_lens_rings=True` at forward time. No post-hoc image shifts or synthetic ring textures were used.
"""
    (out / "PHYSICAL_MODEL_COMPATIBILITY.md").write_text(compat, encoding="utf-8")

    caption = """# Figure 6 caption (draft)

**Figure 6. Spatial interference patterns under controlled idler-lens displacement (no object).**

**(a)** Transverse displacement Δx of the idler relay lens. Top row: simulated signal intensity (`I_+`); bottom row: EMCCD mean frames at matched stage positions (0–180 µm relative to x₀ = 4.340 mm). Lower panel: estimated fringe spatial frequency from FFT peak analysis (theory prediction from λ_eq, f_c, and Δx shown for comparison).

**(b)** Effective axial displacement Δz_eff for the double-pass Configuration C path. Top: simulation; bottom: experimental frames at Δz_eff = 0, 4.800, 7.160, and 11.624 mm, corresponding to mechanical stage displacements Δz_stage = 0, 2.400, 3.580, and 5.812 mm relative to z₀ = 6.910 mm. Radial profiles use the zero-displacement reference center. The forward model evaluates d(Δz_stage) once and the production pass factor supplies the round-trip phase.

Simulation intensities are arbitrary units; experimental data are ADU/pixel without dark subtraction. Pixel pitch 13 µm assumed for profile axes when converting to mm.
"""
    (out / "FIGURE6_CAPTION_DRAFT.md").write_text(caption, encoding="utf-8")

    report = {
        "transverse_dx_um_found": sorted(trans_groups.keys()),
        "axial_z_stage_mm_found": sorted(axial_groups.keys()),
        "transverse_positions_used": dx_list,
        "axial_stage_positions_used_mm": dz_list,
        "axial_effective_positions_used_mm": [2.0 * value for value in dz_list],
        "axial_mapping": "delta_z_eff = 2 * delta_z_stage; stage input supplied once",
        "model_support": "idler_lens_misalignment_phase_map with enable_lens_rings=True (runtime override)",
        "production_physics_modified": False,
        "theory_display": {"rot90_k": DISPLAY_ROT90_K, "crop": "419x455"},
        "experimental_shapes": {str(k): trans_exp_imgs[k].shape for k in dx_list},
    }
    (out / "FIGURE6_GENERATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "FIGURE6_GENERATION_REPORT.md").write_text(
        f"# Figure 6 generation report\n\nEvidence: `{out}`\n\nSee JSON and PHYSICAL_MODEL_COMPATIBILITY.md.\n",
        encoding="utf-8",
    )
    (out / "FINAL_DECISION.md").write_text(
        f"""# Figure 6 lens displacement validation

- **Evidence:** `{out}`
- **PRODUCTION_PHYSICS_MODIFIED:** NO (runtime `enable_lens_rings=True` only)
- **Transverse Δx (µm):** {dx_list}
- **Mechanical axial Δz_stage (mm):** {dz_list}
- **Displayed effective axial Δz_eff (mm):** {[2.0 * value for value in dz_list]}

## Main figures
- FIGURE6A_TRANSVERSE_LENS_DISPLACEMENT.png / .pdf
- FIGURE6B_AXIAL_LENS_DISPLACEMENT.png / .pdf

Supplementary pairs under `supplement/transverse/` and `supplement/axial/`.
""",
        encoding="utf-8",
    )
    print(f"Done: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
