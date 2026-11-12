#!/usr/bin/env python3
"""Figure 6(a) from the saved candidate images and the corrected frequency rule.

The simulated 20 µm point is the analytical phase gradient. Every other
nonzero frequency is the mean adjacent bright-center spacing. Simulation uses
the uniform 20% prominence gate. The approved article-figure files are not written.
"""
from __future__ import annotations

import csv
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

from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, X0_MM, group_tiffs
from tools.figure6a_simulated_fringe_frequency_correction import bright_crests
from tools.generate_c_lab_figure6a_manuscript import EXP_ROOT, mean_stack_deduped
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)

OUT = ROOT / "Figure6a_Manuscript"
NPZ = ROOT / "Figure6a_Transverse_Phase_Candidate" / "Candidate_I_minus_display.npz"
EXP_CSV = (
    ROOT
    / "Figure6a_Experimental_Fringe_Center_Correction"
    / "Experimental_Frequency_Comparison.csv"
)
LAMBDA_S_UM = 0.810
F_C_UM = 150.0e3
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


def analytical_kappa_per_mm(dx_um: float) -> float:
    if dx_um == 0.0:
        return 0.0
    return 2.0 * abs(dx_um) / (LAMBDA_S_UM * F_C_UM) * 1.0e3


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw(theory: dict[float, np.ndarray], exp: dict[float, np.ndarray], sim_k: list[float], exp_k: list[float]) -> None:
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
        dx_list, sim_k, marker="s", linestyle="--", color=REFERENCE_TEAL,
        label="Simulation", markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35,
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
    fig.savefig(OUT / "Figure6a.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / "Figure6a.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(NPZ)
    theory = {float(dx): archive["I_minus"][i] for i, dx in enumerate(archive["displacements_um"])}
    groups = group_tiffs(EXP_ROOT, ref_mm=X0_MM, rel_scale=1000.0)
    exp = {float(dx): mean_stack_deduped(groups[round(float(dx), 6)])[0] for dx in TRANS_DX_UM}
    exp_rows = {float(r["lens_displacement_um"]): r for r in load_csv(EXP_CSV)}

    rows: list[dict[str, Any]] = []
    sim_k: list[float] = []
    exp_k: list[float] = []
    for dx in TRANS_DX_UM:
        dx = float(dx)
        det = bright_crests(theory[dx], dx, rel_prom_min=0.20)
        image_k = float(det["k"]) if len(det["centers"]) >= 2 else float("nan")
        analytical = analytical_kappa_per_mm(dx)
        exp_text = exp_rows[dx]["corrected_periodic_frequency_cycles_per_mm"]
        exp_measured = float(exp_text) if exp_text not in ("", "nan") else float("nan")
        if dx == 0.0:
            plotted_sim = 0.0
            plotted_exp = 0.0
            sim_source = "aligned_reference_convention"
            exp_source = "aligned_reference_convention"
        elif dx == 20.0:
            plotted_sim = analytical
            plotted_exp = exp_measured
            sim_source = "analytical_phase_gradient"
            exp_source = "image_center_spacing"
        else:
            if not np.isfinite(image_k):
                raise RuntimeError(f"image frequency missing at Δx = {dx} µm")
            plotted_sim = image_k
            plotted_exp = exp_measured
            sim_source = "image_center_spacing_20_percent"
            exp_source = "image_center_spacing"
        sim_k.append(plotted_sim)
        exp_k.append(plotted_exp)
        rows.append(
            {
                "dx_um": dx,
                "stage_mm": X0_MM + dx * 1.0e-3,
                "simulation_n_centers": len(det["centers"]),
                "simulation_image_frequency_cycles_per_mm": image_k,
                "simulation_analytical_frequency_cycles_per_mm": analytical,
                "simulation_plotted_frequency_cycles_per_mm": plotted_sim,
                "simulation_frequency_source": sim_source,
                "simulation_mean_spacing_mm": det["mean_spacing_mm"],
                "experiment_n_centers": int(exp_rows[dx]["corrected_n_centers"]),
                "experiment_plotted_frequency_cycles_per_mm": plotted_exp,
                "experiment_frequency_source": exp_source,
                "experiment_mean_spacing_mm": exp_rows[dx]["mean_adjacent_spacing_mm"],
            }
        )
    nonzero = [r for r in rows if r["dx_um"] > 0]
    err = np.array([r["simulation_plotted_frequency_cycles_per_mm"] for r in nonzero]) - np.array(
        [r["experiment_plotted_frequency_cycles_per_mm"] for r in nonzero]
    )
    write_csv(OUT / "Figure6a_Data.csv", rows)
    (OUT / "Figure6a_Caption.md").write_text(CAPTION + "\n", encoding="utf-8")
    draw(theory, exp, sim_k, exp_k)
    print(
        {
            "sim_k": sim_k,
            "exp_k": exp_k,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "dx0_mean": float(np.mean(theory[0.0])),
        }
    )


if __name__ == "__main__":
    main()
