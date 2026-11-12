#!/usr/bin/env python3
"""Reproduce manuscript Figure 6(a) from production optics.py.

No monkey patch. The transverse phase is the production single-crystal
far-field law. Image frequencies use the uniform 20% prominence gate.
The simulated 20 µm point is the analytical gradient because that frame
has only one bright center. Δx = 0 is the aligned-reference convention.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optics
from setup_params import internal_camera_sampling_pixel_um, resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, X0_MM, group_tiffs
from tools.c_lab_isolated_pilot import DeltaKCalibrationState
from tools.c_lab_l2_ms1_mi1_calculated_dk0_rabbit_images import N
from tools.figure6a_simulated_fringe_frequency_correction import bright_crests
from tools.generate_c_lab_figure6a_manuscript import (
    EXP_ROOT,
    mean_stack_deduped,
    run_theory_iminus,
    write_audit_csv,
)
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)

ART = ROOT / "article figures"
ARCHIVE = ROOT / "archive" / "figure6a_pre_transverse_integration"
BASELINE = ARCHIVE / "baseline_phase_and_dx0.npz"
LAMBDA_S_UM = 810.0 * 1.0e-3
F_C_UM = 150.0 * 1.0e3
REL_PROM = 0.20
PUBLISHED = {
    0.0: (0.0, 0.0),
    20.0: (0.329218, 0.429938),
    30.0: (0.584966, 0.549472),
    40.0: (0.722282, 0.785464),
    60.0: (1.005530, 0.968635),
    180.0: (2.970003, 3.043022),
}
CAPTION = (
    "Figure 6(a). Simulated and experimental transverse interference patterns "
    "obtained at idler-lens displacements of 0, 20, 30, 40, 60, and 180 µm. "
    "The corresponding spatial-frequency curves are based on the mean separation "
    "between adjacent resolved bright-fringe centers. For the simulated images, "
    "a uniform relative prominence threshold of 20% was applied. At 20 µm, the "
    "simulated intensity distribution contains fewer than two resolvable "
    "bright-fringe centers; therefore, the plotted simulated frequency is obtained "
    "from the analytical phase gradient of the corrected transverse optical model "
    "rather than direct image-based fringe spacing. The zero-displacement point "
    "represents the aligned-reference convention. All remaining nonzero frequencies "
    "are extracted directly from simulated or experimental intensity distributions."
)
UNTOUCHED = [
    "Figure3_Final_Revision/Fig 3a_Final.png",
    "Figure3_Final_Revision/Fig 3a_Final.pdf",
    "Figure3_Final_Revision/Fig 3b_Final.png",
    "Figure3_Final_Revision/Fig 3b_Final.pdf",
    "Figure3_Final_Revision/Fig 3c_Final.png",
    "Figure3_Final_Revision/Fig 3c_Final.pdf",
    "Figure3_Final_Revision/Fig 3d_Final.png",
    "Figure3_Final_Revision/Fig 3d_Final.pdf",
    "Figure3_Final_Revision/Fig 3e_Final.png",
    "Figure3_Final_Revision/Fig 3e_Final.pdf",
    "Figure4_Final_Scientific_Audit.md",
    "Figure5a_Final.png",
    "Figure5a_Final.pdf",
    "Figure5b_Final.png",
    "Figure5b_Final.pdf",
    "Figure6b_Final.png",
    "Figure6b_Final.pdf",
    "Figure6b_Final_Data.csv",
    "Figure6b_Final_Report.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analytical_kappa_per_mm(dx_um: float, setup) -> float:
    if dx_um == 0.0:
        return 0.0
    lam_um = float(setup.lambda_s_nm) * 1.0e-3
    f_um = float(setup.F2_mm) * 1.0e3
    return 2.0 * abs(dx_um) / (lam_um * f_um) * 1.0e3


def central_slope(phase: np.ndarray, pixel_um: float) -> float:
    n = phase.shape[0]
    x = (np.arange(n) - (n - 1) * 0.5) * pixel_um
    return float(np.polyfit(x, phase[n // 2], 1)[0])


def require_production_source() -> None:
    source = (ROOT / "optics.py").read_text(encoding="utf-8")
    if "4.0 * np.pi * ramp / (lam_s_um * f_c_um)" not in source:
        raise RuntimeError("production optics.py does not contain the corrected transverse term")
    if optics.idler_lens_misalignment_phase_map is None:
        raise RuntimeError("production phase function is missing")


def regression_phase(setup_lab) -> dict[str, Any]:
    base = np.load(BASELINE)
    report: dict[str, Any] = {}
    same = {
        "C_lab_dx0_dz80": ("C_lab", 0.0, 80.0),
        "A_dx40_dz0": ("A", 40.0, 0.0),
        "B_dx40_dz0": ("B", 40.0, 0.0),
        "D_dx40_dz0": ("D", 40.0, 0.0),
    }
    for key, (name, dx, dz) in same.items():
        setup = resolve_setup_selection(name).overridden(
            enable_lens_rings=True, lens_shift_x_um=dx, lens_shift_y_um=0.0, delta_z_um=dz
        )
        phase = optics.idler_lens_misalignment_phase_map(N, setup, delta_z_um=dz)["phase_rad"]
        delta = float(np.max(np.abs(np.asarray(phase) - base[key])))
        report[key] = {"max_abs_difference_rad": delta, "unchanged": delta == 0.0}
        if delta != 0.0:
            raise RuntimeError(f"{key} changed by {delta} rad")

    changed = {}
    for key, name, dx, dz in (
        ("C_lab_dx40_dz0", "C_lab", 40.0, 0.0),
        ("C_dx40_dz0", "C", 40.0, 0.0),
    ):
        setup = resolve_setup_selection(name).overridden(
            enable_lens_rings=True, lens_shift_x_um=dx, lens_shift_y_um=0.0, delta_z_um=dz
        )
        state = optics.idler_lens_misalignment_phase_map(N, setup, delta_z_um=dz)
        phase = np.asarray(state["phase_rad"])
        pixel = float(internal_camera_sampling_pixel_um(setup))
        slope = central_slope(phase, pixel)
        expected = 4.0 * np.pi * dx / (float(setup.lambda_s_nm) * 1.0e-3 * float(setup.F2_mm) * 1.0e3)
        if abs(slope - expected) > 1.0e-12:
            raise RuntimeError(f"{name} slope {slope} != {expected}")
        if state["pass_factor_multiplies_transverse"]:
            raise RuntimeError(f"{name} still multiplies the transverse term by pass_factor")
        if np.allclose(phase, base[key]):
            raise RuntimeError(f"{name} transverse phase did not change")
        changed[key] = {
            "slope_rad_per_um": slope,
            "analytical_slope_rad_per_um": expected,
            "max_abs_difference_from_previous_rad": float(np.max(np.abs(phase - base[key]))),
        }
    report["corrected_single_crystal_far_field"] = changed
    report["lambda_s_um"] = LAMBDA_S_UM
    report["f_c_um"] = F_C_UM
    return report


def draw(theory, exp, sim_k, exp_k, png: Path, pdf: Path) -> None:
    apply_publication_graphics_style()
    dx_list = [float(d) for d in TRANS_DX_UM]
    w_in = 180.0 / 25.4
    fig = plt.figure(figsize=(w_in, w_in * 0.76))
    gs_img = GridSpec(
        2, len(dx_list), height_ratios=[1, 1], hspace=0.025, wspace=0.045,
        left=0.145, right=0.995, top=0.925, bottom=0.485,
    )
    gs_plot = GridSpec(1, 1, left=0.145, right=0.995, top=0.435, bottom=0.105)
    t_v = max(float(np.percentile(theory[d], 99.5)) for d in dx_list)
    e_v = max(float(np.percentile(exp[d], 99.5)) for d in dx_list)
    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs_img[0, j])
        ax.imshow(theory[dx], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δx = {dx:.0f} µm", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, "Simulation", transform=ax.transAxes, va="center", ha="right", fontsize=10)
        ax = fig.add_subplot(gs_img[1, j])
        ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, "Experimental", transform=ax.transAxes, va="center", ha="right", fontsize=10)
    axp = fig.add_subplot(gs_plot[0, 0])
    axp.plot(
        dx_list, sim_k, marker="s", linestyle="--", color=REFERENCE_TEAL, label="Simulation",
        markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35,
    )
    axp.plot(
        dx_list, exp_k, marker="D", linestyle=(0, (2.5, 2.0, 6.0, 2.0)), color=REFERENCE_GOLD,
        label="Experimental", markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35,
    )
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)", labelpad=4)
    axp.set_ylabel("Spatial frequency (cycles mm$^{-1}$)", labelpad=6)
    axp.set_xticks(dx_list)
    axp.legend(frameon=False, loc="upper left")
    style_reference_axes(axp)
    axp.set_ylim(0.0, max(sim_k + exp_k) * 1.12)
    axp.set_xlim(-8.0, max(dx_list) * 1.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    require_production_source()
    before_untouched = {rel: sha256(ART / rel) for rel in UNTOUCHED if (ART / rel).is_file()}
    phase_report = regression_phase(resolve_setup_selection("C_lab"))
    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    groups = group_tiffs(EXP_ROOT, ref_mm=X0_MM, rel_scale=1000.0)
    theory: dict[float, np.ndarray] = {}
    exp: dict[float, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for dx in TRANS_DX_UM:
        key = round(float(dx), 6)
        image, file_audit = mean_stack_deduped(groups[key])
        exp[float(dx)] = image
        theory[float(dx)] = run_theory_iminus(setup, dk, dx_um=float(dx))
        for item in file_audit:
            audit.append({**item, "dx_um": float(dx)})

    baseline_dx0 = np.load(BASELINE)["dx0_I_minus"]
    dx0_delta = float(np.max(np.abs(theory[0.0] - baseline_dx0)))
    if dx0_delta != 0.0:
        raise RuntimeError(f"Δx = 0 image changed by max abs {dx0_delta}")

    rows = []
    centers = []
    for dx in TRANS_DX_UM:
        dx = float(dx)
        det = bright_crests(theory[dx], dx, rel_prom_min=REL_PROM)
        image_k = float(det["k"]) if len(det["centers"]) >= 2 else float("nan")
        analytical = analytical_kappa_per_mm(dx, setup)
        if dx == 0.0:
            plotted_sim, source = 0.0, "aligned_reference_convention"
        elif len(det["centers"]) < 2:
            plotted_sim, source = analytical, "analytical_phase_gradient"
        else:
            plotted_sim, source = image_k, "image_center_spacing_20_percent"
        rows.append((dx, det, image_k, analytical, plotted_sim, source))
        for crest in det["all_crests"]:
            centers.append({"dx_um": dx, **crest})

    import csv
    with (ROOT / "Figure6a_Experimental_Fringe_Center_Correction" / "Experimental_Frequency_Comparison.csv").open(encoding="utf-8", newline="") as fh:
        exp_csv = {float(r["lens_displacement_um"]): r for r in csv.DictReader(fh)}
    table = []
    sim_plotted = []
    exp_plotted = []
    for dx, det, image_k, analytical, plotted_sim, source in rows:
        exp_row = exp_csv[dx]
        exp_text = exp_row["corrected_periodic_frequency_cycles_per_mm"]
        exp_measured = float(exp_text) if exp_text not in ("", "nan") else float("nan")
        plotted_exp = 0.0 if dx == 0.0 else exp_measured
        exp_source = "aligned_reference_convention" if dx == 0.0 else "image_center_spacing"
        published_sim, published_exp = PUBLISHED[dx]
        if round(plotted_sim, 6) != published_sim or round(plotted_exp, 6) != published_exp:
            raise RuntimeError(
                f"Δx={dx}: plotted {(plotted_sim, plotted_exp)} != published {(published_sim, published_exp)}"
            )
        if source == "image_center_spacing_20_percent" and not np.isfinite(image_k):
            raise RuntimeError(f"image-derived label without a measurement at {dx}")
        if source == "analytical_phase_gradient" and np.isfinite(image_k):
            raise RuntimeError(f"analytical label used although an image frequency exists at {dx}")
        sim_plotted.append(plotted_sim)
        exp_plotted.append(plotted_exp)
        table.append(
            {
                "dx_um": dx,
                "simulation_n_centers": len(det["centers"]),
                "simulation_image_derived_cycles_per_mm": image_k,
                "simulation_analytical_cycles_per_mm": analytical,
                "simulation_plotted_cycles_per_mm": plotted_sim,
                "simulation_frequency_source": source,
                "simulation_mean_spacing_mm": det["mean_spacing_mm"],
                "experiment_n_centers": int(exp_row["corrected_n_centers"]),
                "experiment_image_derived_cycles_per_mm": exp_measured,
                "experiment_plotted_cycles_per_mm": plotted_exp,
                "experiment_frequency_source": exp_source,
            }
        )

    ART.mkdir(parents=True, exist_ok=True)
    png = ART / "Figure6a_Final.png"
    pdf = ART / "Figure6a_Final.pdf"
    draw(theory, exp, sim_plotted, exp_plotted, png, pdf)
    if png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Figure6a_Final.png is not a PNG")
    if not pdf.read_bytes().startswith(b"%PDF"):
        raise RuntimeError("Figure6a_Final.pdf is not a PDF")
    shutil.copy2(png, ART / "Figure6a.png")
    shutil.copy2(pdf, ART / "Figure6a.pdf")
    write_table(ART / "Figure6a_Final_Data.csv", table)
    write_table(ART / "Figure6a_Fringe_Centers.csv", centers)
    write_audit_csv(ART / "Figure6a_Experimental_File_Audit.csv", audit)
    (ART / "Figure6a_Updated_Caption_and_Text.md").write_text(CAPTION + "\n", encoding="utf-8")
    (ART / "Figure6a_Final_Report.md").write_text(
        "\n".join(
            [
                "# Figure 6(a)",
                "",
                "Reproduced by `python tools/reproduce_figure6a.py`.",
                "",
                "Simulation frequencies at 30, 40, 60, and 180 µm are image-derived with a uniform 20% prominence gate.",
                "The simulated 20 µm point is the analytical phase gradient. Δx = 0 is the aligned-reference convention.",
                "Experimental nonzero frequencies are the corrected image measurements.",
                "",
                CAPTION,
                "",
            ]
        ),
        encoding="utf-8",
    )
    after_untouched = {rel: sha256(ART / rel) for rel in before_untouched}
    if after_untouched != before_untouched:
        raise RuntimeError(f"untouched figures changed: {[k for k in before_untouched if before_untouched[k] != after_untouched[k]]}")
    summary = {
        "phase_regression": phase_report,
        "dx0_max_abs_difference": dx0_delta,
        "dx0_mean": float(np.mean(theory[0.0])),
        "table": table,
        "untouched_figures_unchanged": True,
        "png": str(png),
        "pdf": str(pdf),
        "optics_sha256": sha256(ROOT / "optics.py"),
    }
    (ART / "Figure6a_Reproduction_Summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"dx0_mean": summary["dx0_mean"], "sources": [r["simulation_frequency_source"] for r in table], "plotted_sim": sim_plotted}, indent=2))


if __name__ == "__main__":
    main()
