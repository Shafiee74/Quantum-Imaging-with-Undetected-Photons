#!/usr/bin/env python3
"""Simulation ring radii and phase curvature vs effective axial defocus."""
from __future__ import annotations

import argparse
import csv
import json
import math
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

from camera.c_lab_display_zoom import export_zoom_config_from_setup, resolve_fixed_crop_window
from optics import idler_lens_misalignment_phase_map
from setup_params import equivalent_wavelength_nm, resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import N, PIXEL_UM, run_theory
from tools.c_lab_figure6b_corrected_effective_defocus import (
    Z_EFF_MM,
    Z_STAGE_MM,
    array_sha256,
)
from tools.c_lab_figure6b_ring_extraction import PIXEL_MM, detect_rings, reference_center
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6b_theoretical_ring_evolution"
REF_Z_EFF = list(Z_EFF_MM)
N_SAMPLES = 120
MIN_RING_RHO_MM = 0.35
MAX_RING_JUMP_MM = 0.45


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def peaks_by_defocus(ring_rows: list[dict], kind: str) -> dict[float, list[float]]:
    out: dict[float, list[float]] = {}
    for r in ring_rows:
        if r.get("ring_type") != kind or r.get("radius_mm") in ("", None):
            continue
        z = float(r["z_eff_mm"])
        rho = float(r["radius_mm"])
        if rho >= MIN_RING_RHO_MM:
            out.setdefault(z, []).append(rho)
    for z in out:
        out[z] = sorted(out[z])
    return out


def tracked_rank_series(
    z_sorted: list[float],
    by_z: dict[float, list[float]],
    rank: int,
) -> list[tuple[list[float], list[float]]]:
    """Return matplotlib segments for the rank-th innermost resolved ring (1-based)."""
    segments: list[tuple[list[float], list[float]]] = []
    xs: list[float] = []
    ys: list[float] = []
    prev: float | None = None
    for z in z_sorted:
        if z < 0.05:
            continue
        radii = by_z.get(z, [])
        if len(radii) < rank:
            if xs:
                segments.append((xs, ys))
                xs, ys = [], []
            prev = None
            continue
        rho = radii[rank - 1]
        if prev is not None and abs(rho - prev) > MAX_RING_JUMP_MM:
            segments.append((xs, ys))
            xs, ys = [], []
        xs.append(z)
        ys.append(rho)
        prev = rho
    if xs:
        segments.append((xs, ys))
    return segments


def max_radius_mm(img: np.ndarray, cy: float, cx: float) -> float:
    h, w = img.shape
    return float(min(cy, cx, h - 1 - cy, w - 1 - cx) * PIXEL_MM)


def phase_quadratic_coefficient_mm2(
    setup,
    delta_z_um: float,
) -> dict[str, float]:
    """a in mm^-2 from Φ ≈ 2π a ρ² on idler-lens phase map (P11 + pass_factor)."""
    st = idler_lens_misalignment_phase_map(
        N, setup.overridden(enable_lens_rings=True, delta_z_um=delta_z_um), delta_z_um=delta_z_um
    )
    d_um = float(st["equivalent_propagation_um"])
    f_c_um = float(setup.F2_mm) * 1e3
    lam_eq_um = equivalent_wavelength_nm(setup.lambda_s_nm, setup.lambda_i_nm) * 1e-3
    pf = float(st["pass_factor"])
    # phi = pf * pi * d_um * rho_um^2 / (f_c^2 * lam_eq); 2 pi a rho_mm^2 => a = pf*d_um*1e6/(2 f_c^2 lam_eq)
    a_model = pf * d_um * 1e6 / (2.0 * f_c_um * f_c_um * lam_eq_um)
    d_mm = d_um * 1e-3
    f_c_mm = float(setup.F2_mm)
    lam_eq_mm = lam_eq_um * 1e-3
    a_hoch = d_mm / (2.0 * f_c_mm * f_c_mm * lam_eq_mm) if d_mm > 0 else 0.0
    return {
        "d_um_phase": d_um,
        "d_mm_phase": d_mm,
        "a_model_mm-2": float(a_model),
        "a_hochrainer_paraxial_mm-2": float(a_hoch),
        "pass_factor": pf,
    }


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    I_ref = run_forward_case(setup.overridden(enable_lens_rings=True), ref, dk, grid_n=N)["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))

    z_eff_grid = np.linspace(0.0, 11.624, N_SAMPLES)
    center_ref: Any = None
    ring_rows: list[dict] = []
    phase_rows: list[dict] = []
    ref_hashes: dict[str, str] = {}
    ref_ring_summary: list[dict[str, Any]] = []

    for z_eff in z_eff_grid:
        z_stage = float(z_eff) / 2.0
        dz_um = z_stage * 1000.0
        img = run_theory(setup, dk, bounds, dz_um=dz_um)
        if center_ref is None:
            center_ref = reference_center(img, modality="simulation")
        center = center_ref
        r_lim = max_radius_mm(img, center.cy, center.cx)

        ph = phase_quadratic_coefficient_mm2(setup, dz_um)
        phase_rows.append(
            {
                "z_eff_mm": round(float(z_eff), 4),
                "z_stage_mm": round(z_stage, 4),
                "delta_z_stage_um_forward": dz_um,
                "d_um_in_phase_map": ph["d_um_phase"],
                "d_mm_in_phase_map": ph["d_mm_phase"],
                "a_model_mm-2": ph["a_model_mm-2"],
                "a_hochrainer_paraxial_mm-2": ph["a_hochrainer_paraxial_mm-2"],
                "pass_factor": ph["pass_factor"],
                "equivalent_distance_pre_doubled": False,
            }
        )

        if z_eff < 0.05:
            ring_rows.append(
                {
                    "z_eff_mm": float(z_eff),
                    "z_stage_mm": z_stage,
                    "ring_type": "bright",
                    "relative_order": "",
                    "radius_mm": "",
                    "note": "no distinct rings at zero defocus",
                }
            )
            continue

        ana = detect_rings(img, center, pixel_mm=PIXEL_MM, use_sector_fallback=False)
        for ext in ana.bright:
            if ext.radius_mm > r_lim:
                continue
            ring_rows.append(
                {
                    "z_eff_mm": round(float(z_eff), 4),
                    "z_stage_mm": round(z_stage, 4),
                    "ring_type": "bright",
                    "relative_order": ext.fringe_order,
                    "radius_mm": round(ext.radius_mm, 4),
                    "radius_sq_mm2": round(ext.radius_sq_mm2, 6),
                    "prominence": round(ext.prominence, 6),
                }
            )
        for ext in ana.dark:
            if ext.radius_mm > r_lim:
                continue
            ring_rows.append(
                {
                    "z_eff_mm": round(float(z_eff), 4),
                    "z_stage_mm": round(z_stage, 4),
                    "ring_type": "dark",
                    "relative_order": ext.fringe_order,
                    "radius_mm": round(ext.radius_mm, 4),
                    "radius_sq_mm2": round(ext.radius_sq_mm2, 6),
                    "prominence": round(ext.prominence, 6),
                }
            )

    for z_stage, z_eff_ref in zip(Z_STAGE_MM, REF_Z_EFF):
        dz_um = float(z_stage) * 1000.0
        img_ref = run_theory(setup, dk, bounds, dz_um=dz_um)
        ref_hashes[f"{z_eff_ref:.3f}"] = array_sha256(img_ref)
        if center_ref is None:
            center_ref = reference_center(img_ref if z_eff_ref == 0 else run_theory(setup, dk, bounds, dz_um=0), modality="simulation")
        center = center_ref
        ph_ref = phase_quadratic_coefficient_mm2(setup, dz_um)
        if z_eff_ref < 0.05:
            ref_ring_summary.append(
                {
                    "z_eff_mm": z_eff_ref,
                    "z_stage_mm": z_stage,
                    "d_mm_in_phase_map": ph_ref["d_mm_phase"],
                    "a_model_mm-2": ph_ref["a_model_mm-2"],
                    "n_bright_resolvable": 0,
                    "n_dark_resolvable": 0,
                    "bright_radii_mm": [],
                    "dark_radii_mm": [],
                    "note": "no distinct circular rings at zero defocus",
                }
            )
        else:
            ana_ref = detect_rings(img_ref, center, pixel_mm=PIXEL_MM, use_sector_fallback=False)
            r_lim = max_radius_mm(img_ref, center.cy, center.cx)
            br = [round(e.radius_mm, 4) for e in ana_ref.bright if e.radius_mm <= r_lim]
            dr = [round(e.radius_mm, 4) for e in ana_ref.dark if e.radius_mm <= r_lim]
            ref_ring_summary.append(
                {
                    "z_eff_mm": z_eff_ref,
                    "z_stage_mm": z_stage,
                    "d_mm_in_phase_map": ph_ref["d_mm_phase"],
                    "a_model_mm-2": ph_ref["a_model_mm-2"],
                    "n_bright_resolvable": len(br),
                    "n_dark_resolvable": len(dr),
                    "bright_radii_mm": br,
                    "dark_radii_mm": dr,
                }
            )

    # Plot A — ring radii vs z_eff (rank by radius; break tracks on large jumps)
    z_sorted = sorted({float(r["z_eff_mm"]) for r in ring_rows if r.get("radius_mm") not in ("", None)})
    bright_by_z = peaks_by_defocus(ring_rows, "bright")
    dark_by_z = peaks_by_defocus(ring_rows, "dark")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_b = ["C0", "C1", "C2", "C3", "C4"]
    for k in range(1, 6):
        labeled = False
        for xs, ys in tracked_rank_series(z_sorted, bright_by_z, k):
            ax.plot(
                xs,
                ys,
                "-o",
                color=colors_b[(k - 1) % len(colors_b)],
                label=f"Bright ring {k}" if not labeled else None,
                ms=3,
                lw=1.2,
            )
            labeled = True
    for k in range(1, 4):
        labeled = False
        for xs, ys in tracked_rank_series(z_sorted, dark_by_z, k):
            ax.plot(
                xs,
                ys,
                "--",
                color="gray",
                alpha=0.7,
                label=f"Dark ring {k}" if k == 1 and not labeled else None,
                lw=1.0,
            )
            labeled = True
    for zr in REF_Z_EFF:
        ax.axvline(zr, color="k", alpha=0.12, lw=0.8)
    ax.set_xlabel(r"Effective axial defocus $\Delta z_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel(r"Ring radius $\rho$ (mm)")
    ax.set_title("Simulated ring radii vs effective defocus (I+)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "THEORY_RING_RADIUS_VS_DEFOCUS.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "THEORY_RING_RADIUS_VS_DEFOCUS.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot B — phase curvature vs d_mm
    d_mm = np.array([p["d_mm_in_phase_map"] for p in phase_rows])
    a_mod = np.array([p["a_model_mm-2"] for p in phase_rows])
    a_h = np.array([p["a_hochrainer_paraxial_mm-2"] for p in phase_rows])
    z_eff = np.array([p["z_eff_mm"] for p in phase_rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d_mm, a_mod, "-", color="C0", label=r"Model $a(d)$ from $\Phi=2\pi a\rho^2$")
    ax.plot(d_mm, a_h, "--", color="C2", alpha=0.8, label="Hochrainer paraxial reference")
    for zr in REF_Z_EFF:
        idx = int(np.argmin(np.abs(z_eff - zr)))
        ax.plot(d_mm[idx], a_mod[idx], "ko", ms=6)
    ax.set_xlabel("Equivalent propagation distance $d$ (mm)")
    ax.set_ylabel(r"Quadratic phase coefficient $a$ (mm$^{-2}$)")
    ax.set_title("Simulated phase curvature vs propagation distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "THEORY_PHASE_CURVATURE_VS_PROPAGATION.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "THEORY_PHASE_CURVATURE_VS_PROPAGATION.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_eff, a_mod, "-", color="C0", label="Model $a$")
    ax.plot(z_eff, a_h, "--", color="C2", alpha=0.8, label="Hochrainer reference (same $d$)")
    for zr in REF_Z_EFF:
        idx = int(np.argmin(np.abs(z_eff - zr)))
        ax.plot(z_eff[idx], a_mod[idx], "ko", ms=6)
    ax.set_xlabel(r"Effective axial defocus $\Delta z_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel(r"Quadratic phase coefficient $a$ (mm$^{-2}$)")
    ax.set_title("Simulated phase curvature vs effective defocus")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "THEORY_PHASE_CURVATURE_VS_DEFOCUS.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "THEORY_PHASE_CURVATURE_VS_DEFOCUS.pdf", bbox_inches="tight")
    plt.close(fig)

    # Diagnostics at reference positions
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, z_stage, z_eff in zip(axes.ravel(), Z_STAGE_MM, Z_EFF_MM):
        dz_um = z_stage * 1000.0
        img = run_theory(setup, dk, bounds, dz_um=dz_um)
        ax.imshow(img, cmap="gray", origin="upper")
        ana = detect_rings(img, center_ref, pixel_mm=PIXEL_MM, use_sector_fallback=False)
        for ext in ana.bright:
            ang = 0.0
            yy = center_ref.cy + (ext.radius_mm / PIXEL_MM) * math.sin(ang)
            xx = center_ref.cx + (ext.radius_mm / PIXEL_MM) * math.cos(ang)
            ax.plot(xx, yy, "o", mfc="none", mec="lime", ms=6)
        ax.set_title(f"Δz_eff={z_eff:.3f} mm")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ring tracking on simulation crop", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "THEORY_RING_TRACKING_DIAGNOSTICS.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (out / "THEORY_RING_RADII.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in ring_rows for k in r.keys()})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(ring_rows)

    with (out / "THEORY_PHASE_CURVATURE.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(phase_rows[0].keys()))
        w.writeheader()
        w.writerows(phase_rows)

    hash_check = {
        k: {
            "computed_simulation_sha256": ref_hashes.get(k),
            "mechanical_input_pre_doubled": False,
            "production_pass_factor": 2.0,
        }
        for k in ref_hashes
    }

    linearity = np.corrcoef(d_mm[d_mm > 0], a_mod[d_mm > 0])[0, 1] if np.any(d_mm > 0) else float("nan")
    ratio = float(np.nanmedian(a_mod[d_mm > 0] / a_h[d_mm > 0])) if np.any(d_mm > 0) else float("nan")

    (out / "PHYSICAL_MODEL_REPORT.md").write_text(
        f"""# Physical model report — simulated ring evolution

## Model
- Mechanical `δ = Δz_stage` is supplied once and `idler_lens_equivalent_propagation_um` evaluates `d(δ)` once.
- Phase (P11): `φ = pass_factor × π d ρ_um² / (f_c² λ_eq)` on the idler seed; production `pass_factor = 2` supplies the single-crystal Michelson round trip.
- Display mapping: `Δz_eff = 2 Δz_stage`; the equivalent propagation distance is not pre-doubled.
- **a** from `Φ ≈ 2π a ρ_mm²`: `a = pass_factor × d_um × 10⁶ / (2 f_c_um² λ_eq_um)`.

## Hochrainer reference
`a_paraxial = d_mm / (2 f_c_mm² λ_eq_mm)` uses the **same d** from the phase map but **without pass_factor** in the paraxial shortcut. C_lab model includes **pass_factor = 2** → model slope ≈ 2× paraxial reference if d matches.

## Linearity a(d)
Pearson r(d, a_model) ≈ {linearity:.4f} over sampled range (a(d) linear in d by construction of φ).

## Simulation image hashes (exact mechanical positions)
{json.dumps(hash_check, indent=2)}

## Reference ring radii and phase (exact positions)
{json.dumps(ref_ring_summary, indent=2)}

## Ring-inferred a (secondary): bright order m → a ≈ m / ρ²
Relative fringe index m from annular peak order; not absolute optical order.

## Michelson vs Hochrainer
Median a_model / a_hochrainer ≈ {ratio:.4f} (expected ≈ pass_factor = 2 when d is shared).

**PRODUCTION_PHYSICS_MODIFIED:** NO (evidence tool only; double-counting removed from this evidence path).
""",
        encoding="utf-8",
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Simulation-only Hochrainer-style plots

- Output: `{out}`
- Samples: {N_SAMPLES} from Δz_eff=0 to 11.624 mm
- Reference positions: {REF_Z_EFF}
- PRODUCTION_PHYSICS_MODIFIED: NO
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
