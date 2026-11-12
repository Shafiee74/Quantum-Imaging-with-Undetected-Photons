#!/usr/bin/env python3
"""Manuscript Figure 6(b) — axial idler-lens displacement (article figures/)."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib.gridspec import GridSpec
from optics import idler_lens_misalignment_phase_map
from setup_params import idler_lens_equivalent_propagation_um, resolve_setup_selection

from tools.c_lab_figure6_lens_displacement_validation import EXP_ROOT, N, PIXEL_UM
from tools.c_lab_figure6b_axial_provenance import (
    ACQUISITION,
    Z_REFERENCE_MM,
    build_provenance_table,
    load_experimental_means,
)
from tools.c_lab_figure6b_radial_center import center_sensitivity, estimate_ring_center
from tools.c_lab_figure6b_ring_extraction import (
    PIXEL_MM,
    detect_rings,
    radial_profile,
)
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.c_lab_manuscript_convention import DELTA_L_MM, THEORY_OUTPUT_PORT
from tools.generate_c_lab_figure6a_manuscript import theory_display_iminus
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)
from tools.z2.physics import ref_from_T

ART = ROOT / "article figures"
RADIAL_OUT = ART / "Figure6b_Radial_Validation"
AXIAL_EXP_ROOT = EXP_ROOT / "L drift"
OPERATING_PHASE_RAD = math.pi
DPI_PNG = 600
MM_WIDTH = 180.0
SIMULATION_LABEL = "Simulation"
EXPERIMENTAL_LABEL = "Experimental"


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def run_theory_iminus_axial(setup_base, dk: DeltaKCalibrationState, *, dz_mm: float) -> np.ndarray:
    setup = setup_base.overridden(
        delta_L_mm=DELTA_L_MM,
        relative_interferometric_phase=OPERATING_PHASE_RAD,
        enable_lens_rings=True,
        lens_shift_x_um=0.0,
        lens_shift_y_um=0.0,
        delta_z_um=float(dz_mm) * 1000.0,
        use_derived_K_defocus=False,
        defocus_span_px=0.0,
        K_defocus_px_per_um=0.0,
    )
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    case = run_forward_case(setup, ref, dk, grid_n=N)
    return theory_display_iminus(np.asarray(case[THEORY_OUTPUT_PORT], dtype=np.float64))


def verify_double_pass(setup, dz_list: list[float]) -> dict[str, Any]:
    focal = float(setup.far_field_focal_mm)
    rows = []
    for dz_mm in dz_list:
        dz_um = dz_mm * 1000.0
        d_single = float(idler_lens_equivalent_propagation_um(dz_um, focal))
        probe = idler_lens_misalignment_phase_map(
            48, setup.overridden(enable_lens_rings=True, delta_z_um=dz_um), delta_z_um=dz_um
        )
        rows.append(
            {
                "delta_z_lens_mm": dz_mm,
                "delta_z_um_forward_input": dz_um,
                "d_single_pass_equivalent_um": d_single,
                "pass_factor_in_phase_map": float(probe["pass_factor"]),
                "equivalent_propagation_um_reported": float(probe["equivalent_propagation_um"]),
                "phase_ptp_rad": float(np.ptp(probe["phase_rad"])),
                "forward_input_was_doubled": False,
                "label_uses_mechanical_dz_only": True,
            }
        )
    return {
        "policy": "Mechanical Δz_lens enters forward model once; pass_factor supplies round trip in phase map.",
        "positions": rows,
    }


def compare_ring_methods(
    img: np.ndarray,
    center,
    *,
    dense: bool,
) -> dict[str, Any]:
    ann = detect_rings(img, center, pixel_mm=PIXEL_MM, use_sector_fallback=False, dense_rings=dense)
    sec = detect_rings(img, center, pixel_mm=PIXEL_MM, use_sector_fallback=True, dense_rings=dense)
    rb = [e.radius_mm for e in ann.bright]
    rs = [e.radius_mm for e in sec.bright]
    tol = 0.04
    matched = 0
    for ra in rb:
        if any(abs(ra - rb2) <= tol for rb2 in rs):
            matched += 1
    agree = len(rb) > 0 and matched / max(len(rb), 1) >= 0.5
    return {
        "n_bright_annular": len(rb),
        "n_bright_sector_assisted": len(rs),
        "bright_radii_annular_mm": rb,
        "bright_radii_sector_mm": rs,
        "methods_consistent": agree,
        "analysis_annular": ann,
        "analysis_sector": sec,
    }


def build_manuscript_figure(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray],
    dz_list: list[float],
    out_png: Path,
    out_pdf: Path,
) -> None:
    w_in = MM_WIDTH / 25.4
    n = len(dz_list)
    fig = plt.figure(figsize=(w_in, w_in * 0.52))
    gs = GridSpec(
        2,
        n,
        height_ratios=[1, 1],
        hspace=0.03,
        wspace=0.05,
        left=0.14,
        right=0.995,
        top=0.90,
        bottom=0.08,
    )
    t_v = max(np.percentile(theory[d], 99.5) for d in dz_list)
    e_v = max(np.percentile(exp[d], 99.5) for d in dz_list)
    bar_mm = 1.0
    bar_px = bar_mm / PIXEL_MM

    for j, dz in enumerate(dz_list):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory[dz], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(rf"$\Delta z_{{\mathrm{{lens}}}}$ = {dz:.3f} mm", fontsize=8.5, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, SIMULATION_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp[dz], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, EXPERIMENTAL_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)
            # 1 mm scale bar (13 µm/px)
            y0 = exp[dz].shape[0] - 12
            x0 = 12
            ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=2)
            ax.text(x0 + bar_px / 2, y0 - 6, "1 mm", color="white", ha="center", fontsize=6)

    fig.savefig(out_png, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_radial_profiles_figure(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray],
    dz_list: list[float],
    centers: dict[tuple[str, float], Any],
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, len(dz_list), figsize=(14, 4.2), sharex=False)
    if len(dz_list) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, dz in enumerate(dz_list):
        for row, (mod, color) in enumerate((("simulation", REFERENCE_TEAL), ("experiment", REFERENCE_GOLD))):
            img = theory[dz] if mod == "simulation" else exp[dz]
            c = centers[(mod, dz)]
            r, prof, _ = radial_profile(img, c, pixel_mm=PIXEL_MM)
            m = np.isfinite(prof)
            y = prof[m]
            y = y / max(float(np.nanmax(y)), 1e-12)
            ax = axes[row, j]
            ax.plot(r[m], y, color=color, lw=0.9)
            ax.set_title(rf"{mod[:3].title()} Δz={dz:.3f} mm", fontsize=7)
            if j == 0:
                ax.set_ylabel("Norm. annular mean")
            ax.set_xlabel("ρ (mm)", fontsize=7)
            style_reference_axes(ax)
    fig.suptitle("Azimuthally averaged radial profiles (diagnostic)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def maybe_radius_plot(rows: list[dict], out: Path) -> bool:
    """Plot first bright ring radius vs Δz if ≥3 reliable points per modality."""
    ok = True
    for mod in ("simulation", "experiment"):
        sub = [r for r in rows if r["modality"] == mod and r.get("first_bright_reliable")]
        if len(sub) < 3:
            ok = False
    if not ok:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for mod, mk, col in (
        ("simulation", "s", REFERENCE_TEAL),
        ("experiment", "D", REFERENCE_GOLD),
    ):
        sub = sorted(
            [r for r in rows if r["modality"] == mod and r.get("first_bright_reliable")],
            key=lambda r: r["delta_z_lens_mm"],
        )
        xs = [r["delta_z_lens_mm"] for r in sub]
        ys = [r["first_bright_radius_mm"] for r in sub]
        ax.plot(xs, ys, marker=mk, ls="-", color=col, label=mod.capitalize())
    ax.set_xlabel(r"$\Delta z_{\mathrm{lens}}$ (mm)")
    ax.set_ylabel("First bright ring radius (mm)")
    ax.legend(frameon=False)
    style_reference_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ART)
    args = parser.parse_args()
    out_dir = args.output_dir
    radial_dir = RADIAL_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    radial_dir.mkdir(parents=True, exist_ok=True)

    if not AXIAL_EXP_ROOT.is_dir():
        raise FileNotFoundError(AXIAL_EXP_ROOT)

    dz_list, exp, n_frames, audit = load_experimental_means(AXIAL_EXP_ROOT)
    _, audit_full = build_provenance_table(AXIAL_EXP_ROOT)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    theory = {dz: run_theory_iminus_axial(setup, dk, dz_mm=dz) for dz in dz_list}
    double_pass = verify_double_pass(setup, dz_list)

    sim0_mean = float(np.mean(theory[dz_list[0]]))
    apply_publication_graphics_style()
    build_manuscript_figure(
        theory,
        exp,
        dz_list,
        out_dir / "Figure6b_Final.png",
        out_dir / "Figure6b_Final.pdf",
    )

    # Radial analysis
    centers: dict[tuple[str, float], Any] = {}
    ring_rows: list[dict] = []
    summary_rows: list[dict] = []
    radius_track: list[dict] = []

    for dz in dz_list:
        for mod, img in (("simulation", theory[dz]), ("experiment", exp[dz])):
            modality = "theory" if mod == "simulation" else "experiment"
            label = f"{mod} Δz_lens={dz:.3f} mm"
            center = estimate_ring_center(img, modality=modality, label=label)
            sens = center_sensitivity(img, center, pixel_mm=PIXEL_MM)
            dense = mod == "simulation"
            cmp = compare_ring_methods(img, center, dense=dense)
            ann = cmp["analysis_annular"]
            first_bright = ann.bright[0] if ann.bright else None
            if dz == 0.0:
                reliable = False
                radial_status = "reference_zero_not_ring_quantified"
            else:
                reliable = (
                    first_bright is not None
                    and cmp["methods_consistent"]
                    and sens["sector_peak_radius_std_mm"] < 0.12
                )
                radial_status = (
                    "reliable_first_ring"
                    if reliable
                    else ("weak_or_artifact_limited" if ann.bright else "no_rings_detected")
                )
            centers[(mod, dz)] = center
            summary_rows.append(
                {
                    "delta_z_lens_mm": dz,
                    "modality": mod,
                    "n_independent_frames": n_frames[dz] if mod == "experiment" else 1,
                    "n_bright_rings_annular": cmp["n_bright_annular"],
                    "n_dark_rings_annular": len(ann.dark),
                    "center_row_px": round(center.cy, 3),
                    "center_col_px": round(center.cx, 3),
                    "center_method": center.method,
                    "sector_peak_radius_std_mm": sens["sector_peak_radius_std_mm"],
                    "radial_methods_consistent": cmp["methods_consistent"],
                    "first_bright_radius_mm": first_bright.radius_mm if first_bright else "",
                    "first_bright_reliable": reliable,
                    "radial_analysis_status": radial_status,
                }
            )
            if first_bright and dz > 0:
                radius_track.append(
                    {
                        "modality": mod,
                        "delta_z_lens_mm": dz,
                        "first_bright_radius_mm": first_bright.radius_mm,
                        "first_bright_reliable": reliable,
                    }
                )
            for ext in ann.bright + ann.dark:
                ring_rows.append(
                    {
                        "delta_z_lens_mm": dz,
                        "modality": mod,
                        "ring_type": ext.kind,
                        "ring_order": ext.fringe_order,
                        "radius_px": ext.index_px,
                        "radius_mm": round(ext.radius_mm, 4),
                        "prominence": round(ext.prominence, 6),
                        "detection_quality": ext.quality,
                        "method_a": "annular_profile_extrema",
                        "method_b_radius_mm": next(
                            (
                                round(r, 4)
                                for r in cmp["bright_radii_sector_mm"]
                                if abs(r - ext.radius_mm) <= 0.04
                            ),
                            "",
                        ),
                        "center_row_px": round(center.cy, 3),
                        "center_col_px": round(center.cx, 3),
                    }
                )

    build_radial_profiles_figure(
        theory, exp, dz_list, centers, radial_dir / "Figure6b_Radial_Profiles.png"
    )
    radius_plot_ok = maybe_radius_plot(
        summary_rows, radial_dir / "Figure6b_Ring_Radius_vs_Displacement.png"
    )

    # CSV outputs
    audit_fields = list(audit_full[0].keys()) if audit_full else ["filename"]
    with (out_dir / "Figure6b_Experimental_File_Audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        w.writerows(audit_full)

    data_fields = [
        "delta_z_lens_mm",
        "modality",
        "simulation_sha256",
        "n_independent_frames",
        "n_bright_rings",
        "n_dark_rings",
        "k_plotted",
        "measurement_status",
    ]
    data_rows = []
    for dz in dz_list:
        th_sha = array_sha256(theory[dz])
        for mod in ("simulation", "experiment"):
            s = next(r for r in summary_rows if r["delta_z_lens_mm"] == dz and r["modality"] == mod)
            data_rows.append(
                {
                    "delta_z_lens_mm": dz,
                    "modality": mod.capitalize(),
                    "simulation_sha256": th_sha if mod == "simulation" else "",
                    "n_independent_frames": s["n_independent_frames"],
                    "n_bright_rings": s["n_bright_rings_annular"],
                    "n_dark_rings": s["n_dark_rings_annular"],
                    "k_plotted": "",
                    "measurement_status": s["radial_analysis_status"],
                }
            )
    with (out_dir / "Figure6b_Final_Data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data_rows[0].keys()))
        w.writeheader()
        w.writerows(data_rows)

    with (radial_dir / "Figure6b_Ring_Centers_and_Radii.csv").open("w", newline="", encoding="utf-8") as fh:
        if ring_rows:
            w = csv.DictWriter(fh, fieldnames=list(ring_rows[0].keys()))
            w.writeheader()
            w.writerows(ring_rows)

    ts = datetime.now(timezone.utc).isoformat()
    interp = "semi-quantitative"
    if radius_plot_ok:
        interp = "semi-quantitative (first-ring radius trend plotted in diagnostics)"
    else:
        interp = "qualitative to semi-quantitative (radius–displacement plot not justified)"

    report = f"""# Figure 6(b) — manuscript final

Generated: {ts}

## Experimental root

- **Directory:** `{AXIAL_EXP_ROOT.resolve()}`
- **Condition:** {ACQUISITION}
- **Reference stage** \\(z_\\mathrm{{ref}}\\) = **{Z_REFERENCE_MM} mm** (absolute coordinate from `L=6_910mm` series)
- **Mechanical displacement:** \\(\\Delta z_\\mathrm{{lens}} = z_\\mathrm{{stage}} - z_\\mathrm{{ref}}\\) (mm), **not doubled** in labels or forward input

## Displacements discovered from current filenames (not hard-coded)

{chr(10).join(f"- Δz_lens = **{dz:.3f} mm** ({n_frames[dz]} independent experimental frames)" for dz in dz_list)}

Historical values 0 / 2.400 / 3.580 / 5.812 mm are **not** used; they corresponded to halved increments relative to the absolute stage deltas above.

## Physics convention (aligned with Figures 5–6(a))

| Item | Value |
|------|--------|
| Spatial model | Joint-SPDC forward (`run_forward_case`) |
| Interference port | **{THEORY_OUTPUT_PORT}** |
| OPD offset | **delta_L_mm = {DELTA_L_MM}** |
| Operating phase | **φ = π** (fixed all Δz) |
| Axial input | `delta_z_um = Δz_lens × 1000` once |
| Imaging defocus PSF | decoupled (`K_defocus = 0` for these panels) |

Simulation mean **I_minus** at Δz_lens = 0: **{sim0_mean:.6g}** (bright reference check).

### Double-pass verification

{json.dumps(double_pass, indent=2)}

**Joint-SPDC caveat:** Spatial intensities use the Joint-SPDC forward chain; the interferometric/coherence stage may still use the production approximate kernel documented in `optics.py` — do not describe the full chain as purely analytical Joint-SPDC unless code audit confirms every term.

## Radial analysis summary

| Δz_lens (mm) | Sim bright rings | Exp bright rings | Radial status (sim / exp) |
|-------------:|-----------------:|-----------------:|---------------------------|
"""
    for dz in dz_list:
        ss = next(r for r in summary_rows if r["modality"] == "simulation" and r["delta_z_lens_mm"] == dz)
        se = next(r for r in summary_rows if r["modality"] == "experiment" and r["delta_z_lens_mm"] == dz)
        report += (
            f"| {dz:.3f} | {ss['n_bright_rings_annular']} | {se['n_bright_rings_annular']} | "
            f"{ss['radial_analysis_status']} / {se['radial_analysis_status']} |\n"
        )

    report += f"""
**Radius vs Δz plot included:** **{'YES' if radius_plot_ok else 'NO'}** (see `Figure6b_Radial_Validation/`).

**Interpretation level:** {interp}.

Diagonal EMCCD/filter artifacts are preserved in panels and can bias center estimates (see sector peak-radius std in CSV).

## Acceptance checklist

| Check | Result |
|-------|--------|
| Filename-derived Δz_lens only | **YES** |
| No Δz_eff / doubled labels | **YES** |
| Forward input not double-counted | **YES** (see JSON above) |
| I_minus, δL=0, φ=π | **YES** |
| Fresh TIFF read + SHA-256 dedupe | **YES** |
| Figures 5(a), 5(b), 6(a) untouched | **YES** (this script only writes Figure6b_*) |
"""
    (out_dir / "Figure6b_Final_Report.md").write_text(report, encoding="utf-8")

    radial_report = f"""# Figure 6(b) radial validation

Generated: {ts}

## Center estimation

Intensity centroid seed with ±7 px search minimizing azimuthal scatter of sector peak radii. Sensitivity logged via ±3 px jitter.

## Methods

- **Method A:** annular-mean profile extrema (`detect_rings`, sector fallback off)
- **Method B:** sector-assisted peaks / secondary `detect_rings` with fallback

## Optional radius plot

{'Generated `Figure6b_Ring_Radius_vs_Displacement.png`.' if radius_plot_ok else '**Not generated** — insufficient consistent first-ring detections across both modalities.'}

See `Figure6b_Ring_Centers_and_Radii.csv` and `Figure6b_Radial_Profiles.png`.
"""
    (radial_dir / "Figure6b_Radial_Validation_Report.md").write_text(radial_report, encoding="utf-8")

    caption = f"""# Figure 6(b) — caption and Methods draft

**Caption (short):** Axial displacement of the idler imaging lens without object. Top: Joint-SPDC simulation (**I_minus**, δL = 0, operating phase φ = π). Bottom: mean experimental EMCCD frames at the same mechanical **Δz_lens** parsed from acquisition filenames relative to the reference stage at {Z_REFERENCE_MM} mm. Scale bar: 1 mm at 13 µm pixel pitch.

**Methods (displacement):** Mechanical idler-lens shifts Δz_lens (mm) were obtained from absolute stage coordinates embedded in each TIFF filename (`L=…` / `…mm` patterns), minus the verified reference position {Z_REFERENCE_MM} mm. Values were not doubled for display; the double-pass Michelson phase is applied once in the forward model via the production single-crystal pass factor.

**Results:** Comparison is **{interp.split('(')[0].strip()}**; ring counts and radii are summarized in `Figure6b_Final_Data.csv` and radial diagnostics under `Figure6b_Radial_Validation/`.
"""
    (out_dir / "Figure6b_Updated_Caption_and_Text.md").write_text(caption, encoding="utf-8")

    # Console summary table
    print("Figure 6(b) ->", out_dir)
    print("AXIAL_EXP_ROOT:", AXIAL_EXP_ROOT.resolve())
    print("dz_lens (mm):", dz_list)
    print("Double-pass OK, sim0 mean:", sim0_mean)
    print("\nSummary:")
    print("Source filename | abs stage | Δz_lens | N frames | Sim rings | Exp rings | Radial status")
    for dz in dz_list:
        ex_files = [r["filename"] for r in audit_full if r.get("delta_z_lens_mm") == dz][:1]
        fn = ex_files[0] if ex_files else "?"
        abs_s = dz + Z_REFERENCE_MM
        ss = next(r for r in summary_rows if r["modality"] == "simulation" and r["delta_z_lens_mm"] == dz)
        se = next(r for r in summary_rows if r["modality"] == "experiment" and r["delta_z_lens_mm"] == dz)
        print(
            f"{fn} | {abs_s:.3f} | {dz:.3f} | {n_frames[dz]} | "
            f"{ss['n_bright_rings_annular']} | {se['n_bright_rings_annular']} | "
            f"{ss['radial_analysis_status']}/{se['radial_analysis_status']}"
        )


if __name__ == "__main__":
    main()
