#!/usr/bin/env python3
"""Figure 6(a) independent computation, plotting, and validation.

Scientific integrity rule
-------------------------
Simulation frequencies are always computed from the production model
(``optics.py`` + ``C_lab`` → simulated I_minus images → 20% prominence
bright-crest spacing, with the 20 µm point from the analytical phase
gradient). Reference CSV/JSON under ``reproducibility/figure6a/reference/``
are validation targets only and are **never** written into the plotted
simulation series.

Public experimental frequency markers use the shipped derived Table S3
record (not raw EMCCD TIFFs). Re-extraction from private TIFFs is a separate
laboratory mode.

Commands::

    python tools/reproduce_figure6a.py --compute          # model → computed CSV → figure
    python tools/reproduce_figure6a.py --verify            # compare computed vs reference
    python tools/reproduce_figure6a.py --public            # compute then verify
    python tools/reproduce_figure6a.py --replot-archived   # explicit CSV-only replot (NOT reproduction)
    python tools/reproduce_figure6a.py --laboratory        # private TIFF image strip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optics
from setup_params import resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, X0_MM, group_tiffs
from tools.c_lab_isolated_pilot import DeltaKCalibrationState
from tools.figure6a_simulated_fringe_frequency_correction import bright_crests
from tools.generate_c_lab_figure6a_manuscript import (
    EXP_ROOT,
    mean_stack_deduped,
    run_theory_iminus,
)
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)

PKG = ROOT / "reproducibility" / "figure6a"
REF_DIR = PKG / "reference"
COMPUTED_DIR = PKG / "computed"
OUT_DIR = PKG / "out"
REF_EXP_CSV = REF_DIR / "Experimental_Frequency_Comparison.csv"
REF_FINAL_CSV = REF_DIR / "Figure6a_Final_Data.csv"
COMPUTED_CSV = COMPUTED_DIR / "Figure6a_Computed_Data.csv"
COMPUTED_SUMMARY = COMPUTED_DIR / "Figure6a_Computed_Summary.json"
VALIDATION_JSON = OUT_DIR / "Figure6a_Validation_Report.json"
VALIDATION_MD = OUT_DIR / "Figure6a_Validation_Report.md"

REL_PROM = 0.20  # uniform 20% relative prominence for simulated image frequencies
# Predetermined verification tolerances (versioned).
TOL_ABS_CYCLES_PER_MM = 5.0e-6
TOL_CENTER_COUNT_EXACT = True
EXPECTED_MAE_IMG = 0.05215
EXPECTED_RMSE_IMG = 0.05465
TOL_MAE_RMSE = 5.0e-4

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analytical_kappa_per_mm(dx_um: float, setup) -> float:
    """κ_tr = 2|Δx_ℓ|/(λ_s f_c) in cycles/mm (manuscript transverse law)."""
    if dx_um == 0.0:
        return 0.0
    lam_um = float(setup.lambda_s_nm) * 1.0e-3
    f_um = float(setup.F2_mm) * 1.0e3
    return 2.0 * abs(dx_um) / (lam_um * f_um) * 1.0e3


def require_production_source() -> None:
    source = (ROOT / "optics.py").read_text(encoding="utf-8")
    if "4.0 * np.pi * ramp / (lam_s_um * f_c_um)" not in source:
        raise RuntimeError("production optics.py lacks the corrected transverse phase term")
    if optics.idler_lens_misalignment_phase_map is None:
        raise RuntimeError("production phase function is missing")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_exp_derived(path: Path = REF_EXP_CSV) -> dict[float, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Derived experimental frequency record not found: {path}\n"
            "This file is a public Table S3 analysis record, not a simulation substitute."
        )
    return {float(r["lens_displacement_um"]): r for r in load_csv(path)}


# ---------------------------------------------------------------------------
# Computation module — physical model only
# ---------------------------------------------------------------------------
def compute_simulation_rows(theory: dict[float, np.ndarray], setup) -> list[dict[str, Any]]:
    """Independently extract simulated frequencies from newly generated images."""
    rows: list[dict[str, Any]] = []
    for dx in TRANS_DX_UM:
        dx = float(dx)
        det = bright_crests(theory[dx], dx, rel_prom_min=REL_PROM)
        n_centers = len(det["centers"])
        image_k = float(det["k"]) if n_centers >= 2 else float("nan")
        analytical = analytical_kappa_per_mm(dx, setup)
        if dx == 0.0:
            plotted, source = 0.0, "aligned_reference_convention"
        elif n_centers < 2:
            # Manuscript: at 20 µm only one broad maximum → analytical κ_tr.
            plotted, source = analytical, "analytical_phase_gradient"
        else:
            plotted, source = image_k, "image_center_spacing_20_percent"
        rows.append(
            {
                "dx_um": dx,
                "simulation_n_centers": n_centers,
                "simulation_image_derived_cycles_per_mm": image_k,
                "simulation_analytical_cycles_per_mm": analytical,
                "simulation_plotted_cycles_per_mm": float(plotted),
                "simulation_frequency_source": source,
                "simulation_mean_spacing_mm": det["mean_spacing_mm"],
            }
        )
    return rows


def attach_experimental_derived(
    sim_rows: list[dict[str, Any]],
    exp_csv: dict[float, dict[str, str]],
) -> list[dict[str, Any]]:
    """Attach public derived experimental frequencies for plotting/MAE only."""
    out: list[dict[str, Any]] = []
    for row in sim_rows:
        dx = float(row["dx_um"])
        er = exp_csv[dx]
        exp_text = er.get("corrected_periodic_frequency_cycles_per_mm", "")
        exp_k = float(exp_text) if exp_text not in ("", "nan") else float("nan")
        plotted_exp = 0.0 if dx == 0.0 else exp_k
        out.append(
            {
                **row,
                "experiment_n_centers": int(float(er.get("corrected_n_centers", 0) or 0)),
                "experiment_plotted_cycles_per_mm": plotted_exp,
                "experiment_frequency_source": (
                    "aligned_reference_convention"
                    if dx == 0.0
                    else "derived_table_S3_public_record"
                ),
            }
        )
    return out


def run_compute(*, include_exp_images: bool = False) -> dict[str, Any]:
    """Execute model → write computed CSV → draw figure from computed rows only."""
    require_production_source()
    COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    theory = {float(dx): run_theory_iminus(setup, dk, dx_um=float(dx)) for dx in TRANS_DX_UM}
    sim_rows = compute_simulation_rows(theory, setup)
    exp_csv = load_exp_derived(REF_EXP_CSV)
    rows = attach_experimental_derived(sim_rows, exp_csv)

    exp_images: dict[float, np.ndarray] | None = None
    if include_exp_images:
        if not EXP_ROOT.is_dir():
            raise FileNotFoundError(
                f"Laboratory EMCCD root not found: {EXP_ROOT}\n"
                "Omit --laboratory / use --compute for the public numerical path."
            )
        groups = group_tiffs(EXP_ROOT, ref_mm=X0_MM, rel_scale=1000.0)
        exp_images = {}
        for dx in TRANS_DX_UM:
            key = round(float(dx), 6)
            exp_images[float(dx)], _ = mean_stack_deduped(groups[key])

    write_csv(COMPUTED_CSV, rows)
    sim_k = [float(r["simulation_plotted_cycles_per_mm"]) for r in rows]
    exp_k = [float(r["experiment_plotted_cycles_per_mm"]) for r in rows]
    sources = [str(r["simulation_frequency_source"]) for r in rows]
    png = OUT_DIR / "Figure6a_Final.png"
    pdf = OUT_DIR / "Figure6a_Final.pdf"
    draw(theory, exp_images, sim_k, exp_k, sources, png, pdf, public=exp_images is None)

    img_sim = [
        float(r["simulation_plotted_cycles_per_mm"])
        for r in rows
        if float(r["dx_um"]) > 0 and r["simulation_frequency_source"] != "analytical_phase_gradient"
    ]
    img_exp = [
        float(r["experiment_plotted_cycles_per_mm"])
        for r in rows
        if float(r["dx_um"]) > 0 and r["simulation_frequency_source"] != "analytical_phase_gradient"
    ]
    mae, rmse = mae_rmse(img_sim, img_exp)

    summary = {
        "stage": "compute",
        "model": {
            "transverse_phase": "Phi_tr = [4*pi/(lambda_s*f_c)]*(dx*x + dy*y)",
            "analytical_frequency": "kappa_tr = 2*|dx|/(lambda_s*f_c)  [cycles/mm]",
            "lambda_s_nm": float(setup.lambda_s_nm),
            "f_c_mm": float(setup.F2_mm),
            "simulation_prominence": REL_PROM,
            "setup_id": setup.setup_id,
            "optics_sha256": sha256(ROOT / "optics.py"),
        },
        "outputs": {
            "computed_csv": str(COMPUTED_CSV),
            "png": str(png),
            "pdf": str(pdf),
        },
        "computed_simulation": [
            {
                "dx_um": float(r["dx_um"]),
                "n_centers": int(r["simulation_n_centers"]),
                "plotted_cycles_per_mm": float(r["simulation_plotted_cycles_per_mm"]),
                "source": r["simulation_frequency_source"],
            }
            for r in rows
        ],
        "mae_vs_derived_experiment_image_points": mae,
        "rmse_vs_derived_experiment_image_points": rmse,
        "dx0_mean": float(np.mean(theory[0.0])),
        "note": (
            "Simulation values are independently computed. "
            "Experimental frequency markers are derived Table S3 public records. "
            "Reference Final_Data.csv was not used as a computational substitute."
        ),
    }
    COMPUTED_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "Figure6a_Updated_Caption_and_Text.md").write_text(CAPTION + "\n", encoding="utf-8")
    (OUT_DIR / "Figure6a_Final_Report.md").write_text(
        "\n".join(
            [
                "# Figure 6(a) — independently computed",
                "",
                "Command: `python tools/reproduce_figure6a.py --compute`",
                "",
                f"MAE vs derived experiment (30/40/60/180 µm) = {mae:.5f} cycles/mm",
                f"RMSE = {rmse:.5f} cycles/mm",
                "",
                "Simulation sources: " + ", ".join(sources),
                "",
                CAPTION,
                "",
                summary["note"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    # Convenience copy for readers expecting Final_Data next to the figure.
    write_csv(OUT_DIR / "Figure6a_Final_Data.csv", rows)
    return summary


# ---------------------------------------------------------------------------
# Visualization module
# ---------------------------------------------------------------------------
def draw(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray] | None,
    sim_k: list[float],
    exp_k: list[float],
    sources: list[str],
    png: Path,
    pdf: Path,
    *,
    public: bool,
) -> None:
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
    e_v = max(float(np.percentile(exp[d], 99.5)) for d in dx_list) if exp is not None else 1.0
    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs_img[0, j])
        ax.imshow(theory[dx], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δx = {dx:.0f} µm", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, "Simulation", transform=ax.transAxes, va="center", ha="right", fontsize=10)
        ax = fig.add_subplot(gs_img[1, j])
        if exp is not None:
            ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        else:
            ax.set_facecolor("0.92")
            ax.text(
                0.5, 0.5, "EMCCD\nnot public", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="0.35",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            label = "Experimental" if exp is not None else "Experimental\n(not public)"
            ax.text(-0.24, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=9)

    axp = fig.add_subplot(gs_plot[0, 0])
    dx_sq = [d for d, src in zip(dx_list, sources) if src != "analytical_phase_gradient"]
    sim_sq = [k for k, src in zip(sim_k, sources) if src != "analytical_phase_gradient"]
    axp.plot(dx_list, sim_k, linestyle="--", color=REFERENCE_TEAL, linewidth=1.35, zorder=2)
    axp.plot(
        dx_sq, sim_sq, linestyle="None", marker="s", color=REFERENCE_TEAL,
        markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, zorder=3,
    )
    dx_an = [d for d, src in zip(dx_list, sources) if src == "analytical_phase_gradient"]
    sim_an = [k for k, src in zip(sim_k, sources) if src == "analytical_phase_gradient"]
    if dx_an:
        axp.plot(
            dx_an, sim_an, linestyle="None", marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=REFERENCE_TEAL, markeredgewidth=1.4, zorder=4,
        )
    axp.plot(
        dx_list, exp_k, marker="D", linestyle=(0, (2.5, 2.0, 6.0, 2.0)), color=REFERENCE_GOLD,
        markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35, zorder=3,
    )
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)", labelpad=4)
    axp.set_ylabel("Spatial frequency (cycles mm$^{-1}$)", labelpad=6)
    axp.set_xticks(dx_list)
    axp.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_TEAL, lw=1.35, ls="--", marker="s",
                   markersize=6, markeredgecolor="0.25", label="Simulation (image-derived)"),
            Line2D([0], [0], color=REFERENCE_TEAL, lw=0, marker="o", markersize=8,
                   markerfacecolor="none", markeredgewidth=1.4, label="Simulation (analytical 20 µm)"),
            Line2D([0], [0], color=REFERENCE_GOLD, lw=1.35, ls=(0, (2.5, 2.0, 6.0, 2.0)),
                   marker="D", markersize=6, markeredgecolor="0.25", label="Experimental (derived)"),
        ],
        frameon=False,
        loc="upper left",
    )
    style_reference_axes(axp)
    finite = [v for v in sim_k + exp_k if math.isfinite(float(v))]
    axp.set_ylim(0.0, max(finite) * 1.12)
    axp.set_xlim(-8.0, max(dx_list) * 1.02)
    if public:
        fig.text(
            0.01, 0.01,
            "Simulation: independently computed. Experimental frequencies: derived Table S3 record. "
            "Original EMCCD TIFFs are not publicly available.",
            fontsize=6.5, color="0.35",
        )
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def mae_rmse(sim: list[float], exp: list[float]) -> tuple[float, float]:
    err = np.asarray(sim, float) - np.asarray(exp, float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def run_replot_archived() -> dict[str, Any]:
    """Explicit archived replot from reference CSV only.

    This mode is intentionally labelled and must never be treated as Comment A2
    independent reproduction. It does not execute the physical model.
    """
    if not REF_FINAL_CSV.is_file():
        raise FileNotFoundError(f"Missing reference CSV: {REF_FINAL_CSV}")
    if not REF_EXP_CSV.is_file():
        raise FileNotFoundError(f"Missing experimental reference CSV: {REF_EXP_CSV}")

    rows = load_csv_rows(REF_FINAL_CSV)
    dx_list = [float(r["dx_um"]) for r in rows]
    sim_k = [float(r["simulation_plotted_cycles_per_mm"]) for r in rows]
    exp_k = [float(r["experiment_plotted_cycles_per_mm"]) for r in rows]
    sources = [str(r["simulation_frequency_source"]) for r in rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_publication_graphics_style()
    fig, ax = plt.subplots(figsize=(180.0 / 25.4, 3.2))
    dx_sq = [d for d, src in zip(dx_list, sources) if src != "analytical_phase_gradient"]
    sim_sq = [k for k, src in zip(sim_k, sources) if src != "analytical_phase_gradient"]
    ax.plot(dx_list, sim_k, linestyle="--", color=REFERENCE_TEAL, linewidth=1.35, zorder=2)
    ax.plot(
        dx_sq, sim_sq, linestyle="None", marker="s", color=REFERENCE_TEAL,
        markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, zorder=3,
    )
    dx_an = [d for d, src in zip(dx_list, sources) if src == "analytical_phase_gradient"]
    sim_an = [k for k, src in zip(sim_k, sources) if src == "analytical_phase_gradient"]
    if dx_an:
        ax.plot(
            dx_an, sim_an, linestyle="None", marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=REFERENCE_TEAL, markeredgewidth=1.4, zorder=4,
        )
    ax.plot(
        dx_list, exp_k, marker="D", linestyle=(0, (2.5, 2.0, 6.0, 2.0)), color=REFERENCE_GOLD,
        markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35, zorder=3,
    )
    ax.set_xlabel("Transverse idler-lens displacement, Δx (µm)")
    ax.set_ylabel("Spatial frequency (cycles mm$^{-1}$)")
    ax.set_xticks(dx_list)
    style_reference_axes(ax)
    ax.set_ylim(0.0, max(sim_k + exp_k) * 1.12)
    ax.set_xlim(-8.0, max(dx_list) * 1.02)
    fig.suptitle("ARCHIVED REPLOT FROM REFERENCE CSV — NOT INDEPENDENT COMPUTE", fontsize=9, color="0.25")
    fig.text(
        0.01, 0.01,
        "This figure copies archived numerical targets. Comment A2 requires --compute / --verify.",
        fontsize=7, color="0.4",
    )
    png = OUT_DIR / "Figure6a_Archived_Replot.png"
    pdf = OUT_DIR / "Figure6a_Archived_Replot.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    note = {
        "stage": "replot-archived",
        "status": "ARCHIVED_REPLOT_ONLY",
        "independent_reproduction": False,
        "source_csv": str(REF_FINAL_CSV),
        "png": str(png),
        "pdf": str(pdf),
        "warning": (
            "Values were read from the reference CSV; the physical model was not executed. "
            "Do not cite this mode as Comment A2 closure."
        ),
    }
    (OUT_DIR / "Figure6a_Archived_Replot_NOTE.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8"
    )
    return note


# ---------------------------------------------------------------------------
# Validation module — reference files are read-only comparison evidence
# ---------------------------------------------------------------------------
def run_verify() -> dict[str, Any]:
    if not COMPUTED_CSV.is_file():
        raise FileNotFoundError(
            f"Missing computed results: {COMPUTED_CSV}\n"
            "Run `python tools/reproduce_figure6a.py --compute` first."
        )
    if not REF_FINAL_CSV.is_file():
        raise FileNotFoundError(f"Missing validation reference: {REF_FINAL_CSV}")

    computed = {float(r["dx_um"]): r for r in load_csv(COMPUTED_CSV)}
    reference = {float(r["dx_um"]): r for r in load_csv(REF_FINAL_CSV)}
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []

    for dx in TRANS_DX_UM:
        dx = float(dx)
        c = computed[dx]
        r = reference[dx]
        c_k = float(c["simulation_plotted_cycles_per_mm"])
        r_k = float(r["simulation_plotted_cycles_per_mm"])
        abs_err = abs(c_k - r_k)
        rel_err = abs_err / abs(r_k) if abs(r_k) > 1e-15 else (0.0 if abs_err == 0 else float("inf"))
        c_n = int(float(c["simulation_n_centers"]))
        r_n = int(float(r["simulation_n_centers"]))
        ok_k = abs_err <= TOL_ABS_CYCLES_PER_MM
        ok_n = (c_n == r_n) if TOL_CENTER_COUNT_EXACT else True
        if not ok_k:
            failures.append(
                f"Δx={dx}: computed sim k={c_k:.9f} vs reference {r_k:.9f} "
                f"(abs={abs_err:.3e} > tol {TOL_ABS_CYCLES_PER_MM})"
            )
        if not ok_n:
            failures.append(f"Δx={dx}: computed centers={c_n} vs reference {r_n}")
        comparisons.append(
            {
                "dx_um": dx,
                "computed_simulation_cycles_per_mm": c_k,
                "reference_simulation_cycles_per_mm": r_k,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "computed_n_centers": c_n,
                "reference_n_centers": r_n,
                "pass_frequency": ok_k,
                "pass_centers": ok_n,
                "computed_source": c.get("simulation_frequency_source", ""),
                "reference_source": r.get("simulation_frequency_source", ""),
            }
        )

    img_dx = [30.0, 40.0, 60.0, 180.0]
    mae, rmse = mae_rmse(
        [float(computed[d]["simulation_plotted_cycles_per_mm"]) for d in img_dx],
        [float(computed[d]["experiment_plotted_cycles_per_mm"]) for d in img_dx],
    )
    mae_ok = abs(mae - EXPECTED_MAE_IMG) <= TOL_MAE_RMSE
    rmse_ok = abs(rmse - EXPECTED_RMSE_IMG) <= TOL_MAE_RMSE
    if not mae_ok:
        failures.append(f"MAE={mae:.6f} vs expected≈{EXPECTED_MAE_IMG} (tol {TOL_MAE_RMSE})")
    if not rmse_ok:
        failures.append(f"RMSE={rmse:.6f} vs expected≈{EXPECTED_RMSE_IMG} (tol {TOL_MAE_RMSE})")

    status = "PASS" if not failures else "FAIL"
    report = {
        "stage": "verify",
        "status": status,
        "tolerances": {
            "abs_cycles_per_mm": TOL_ABS_CYCLES_PER_MM,
            "center_count_exact": TOL_CENTER_COUNT_EXACT,
            "mae_rmse_tol": TOL_MAE_RMSE,
            "expected_mae": EXPECTED_MAE_IMG,
            "expected_rmse": EXPECTED_RMSE_IMG,
        },
        "mae_image_derived": mae,
        "rmse_image_derived": rmse,
        "comparisons": comparisons,
        "failures": failures,
        "computed_csv": str(COMPUTED_CSV),
        "reference_csv": str(REF_FINAL_CSV),
        "max_abs_simulation_error": max(c["abs_error"] for c in comparisons),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# Figure 6(a) validation — **{status}**",
        "",
        f"Computed: `{COMPUTED_CSV}`",
        f"Reference (read-only): `{REF_FINAL_CSV}`",
        "",
        f"MAE (30/40/60/180) = {mae:.5f} (expected ≈ {EXPECTED_MAE_IMG})",
        f"RMSE = {rmse:.5f} (expected ≈ {EXPECTED_RMSE_IMG})",
        f"Max |Δk_sim| vs reference = {report['max_abs_simulation_error']:.3e}",
        "",
        "| Δx | computed k | reference k | abs err | centers c/r | pass |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['dx_um']:.0f} | {c['computed_simulation_cycles_per_mm']:.9f} | "
            f"{c['reference_simulation_cycles_per_mm']:.9f} | {c['abs_error']:.3e} | "
            f"{c['computed_n_centers']}/{c['reference_n_centers']} | "
            f"{'PASS' if c['pass_frequency'] and c['pass_centers'] else 'FAIL'} |"
        )
    if failures:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in failures]
    else:
        lines += ["", "All predetermined tolerances satisfied. Simulation independently reproduced."]
    VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", action="store_true", help="Independent model computation + figure")
    parser.add_argument("--verify", action="store_true", help="Compare computed vs reference (no overwrite)")
    parser.add_argument("--public", action="store_true", help="compute then verify (clean checkout)")
    parser.add_argument(
        "--replot-archived",
        action="store_true",
        help="Explicit CSV-only replot (NOT independent reproduction)",
    )
    parser.add_argument("--laboratory", action="store_true", help="Also load private EMCCD image strip")
    # Legacy aliases
    parser.add_argument(
        "--mode",
        choices=("public", "laboratory", "compute", "verify", "replot-archived"),
        default=None,
    )
    args = parser.parse_args(argv)

    mode = None
    if args.replot_archived or args.mode == "replot-archived":
        mode = "replot-archived"
    elif args.laboratory or args.mode == "laboratory":
        mode = "laboratory"
    elif args.verify or args.mode == "verify":
        mode = "verify"
    elif args.compute or args.mode == "compute":
        mode = "compute"
    elif args.public or args.mode == "public" or (
        not args.compute
        and not args.verify
        and not args.laboratory
        and not args.replot_archived
        and args.mode is None
    ):
        mode = "public"

    if mode == "replot-archived":
        note = run_replot_archived()
        print(json.dumps(note, indent=2))
        # Nonzero so automation cannot treat archived replot as successful reproduction.
        return 3

    if mode in ("compute", "public", "laboratory"):
        summary = run_compute(include_exp_images=(mode == "laboratory"))
        print(json.dumps(
            {
                "stage": "compute",
                "computed_csv": summary["outputs"]["computed_csv"],
                "png": summary["outputs"]["png"],
                "computed_simulation": summary["computed_simulation"],
                "mae_vs_derived_experiment_image_points": summary["mae_vs_derived_experiment_image_points"],
                "rmse_vs_derived_experiment_image_points": summary["rmse_vs_derived_experiment_image_points"],
            },
            indent=2,
        ))
        if mode == "compute":
            return 0

    if mode in ("verify", "public"):
        report = run_verify()
        print(json.dumps(
            {
                "stage": "verify",
                "status": report["status"],
                "mae_image_derived": report["mae_image_derived"],
                "rmse_image_derived": report["rmse_image_derived"],
                "max_abs_simulation_error": report["max_abs_simulation_error"],
                "failures": report["failures"],
                "validation_report": str(VALIDATION_MD),
            },
            indent=2,
        ))
        return 0 if report["status"] == "PASS" else 1

    parser.error("select --compute, --verify, --public, --replot-archived, or --laboratory")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
