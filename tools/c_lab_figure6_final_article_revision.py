#!/usr/bin/env python3
"""Revise Figure 6(a) spatial-frequency graph and Figure 6(b) label layout."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from setup_params import resolve_setup_selection
from tools.c_lab_figure6_final_article import (
    DPI_PNG,
    MM_WIDTH,
    figure_width_inches,
    load_transverse,
    setup_matplotlib_pub,
    sha256_array,
)
from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, theory_kappa_cycles_per_mm
from tools.c_lab_figure6_lens_displacement_validation import (
    EXP_ROOT as LENS_EXP_ROOT,
    Z0_MM,
    group_tiffs,
)
import tifffile
import pymupdf
from PIL import Image
from tools.c_lab_figure6_spatial_frequency_robust import extract_series
from tools.c_lab_figure6b_corrected_effective_defocus import Z_EFF_MM, Z_STAGE_MM
from tools.c_lab_manuscript_convention import DELTA_L_MM, DEFAULT_MANUSCRIPT_SETTINGS  # OPD0 mirror-drift convention
from tools.c_lab_figure6b_final_quantitative import (
    load_panels as load_axial_panels,
    max_radius_mm,
    normalized_profile,
)
from tools.c_lab_figure6b_ring_extraction import PIXEL_MM, radial_profile, reference_center
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6_final_article_revision"
APPROVED_HIGH_RING_PDF = (
    ROOT
    / "evidence"
    / "c_lab_figure6b_corrected_effective_defocus"
    / "20261017T103355Z"
    / "FIGURE6B_EFFECTIVE_DEFOCUS_FINAL.pdf"
)
SIMULATION_LABEL = "Simulation"
EXPERIMENTAL_LABEL = "Experimental"
ANALYTICAL_LABEL = "Analytical Prediction"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _analytical_theory_curve(setup) -> tuple[np.ndarray, np.ndarray]:
    """Smooth monotonic Digital Twin prediction: k_induced(Δx) = g|Δx|/(λ_eq f_c)."""
    x = np.linspace(0.0, 180.0, 150)
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= 0:
            y[i] = 0.0
        else:
            k = theory_kappa_cycles_per_mm(float(xi), setup)
            y[i] = float(k) if math.isfinite(k) else 0.0
    return x, y


def _analytical_theory_markers(setup, dx_list: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(dx_list, dtype=float)
    ys = np.array(
        [0.0 if d == 0 else float(theory_kappa_cycles_per_mm(d, setup)) for d in dx_list],
        dtype=float,
    )
    return xs, ys


def build_figure6a_revised(
    theory: dict,
    exp: dict,
    th_rows: list,
    ex_rows: list,
    theory_curve_xy: tuple[np.ndarray, np.ndarray],
    theory_markers_xy: tuple[np.ndarray, np.ndarray],
    out: Path,
) -> None:
    dx_list = [d for d in TRANS_DX_UM if d in theory]
    w_in = figure_width_inches()
    fig = plt.figure(figsize=(w_in, w_in * 0.76))
    gs_img = GridSpec(
        2,
        len(dx_list),
        height_ratios=[1, 1],
        hspace=0.025,
        wspace=0.045,
        left=0.145,
        right=0.995,
        top=0.925,
        bottom=0.485,
    )
    gs_plot = GridSpec(1, 1, left=0.145, right=0.995, top=0.435, bottom=0.105)

    t_v = max(np.percentile(theory[d], 99.5) for d in dx_list)
    e_v = max(np.percentile(exp[d], 99.5) for d in dx_list)

    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs_img[0, j])
        ax.imshow(theory[dx], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δx = {dx:.0f} µm", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, SIMULATION_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)

        ax = fig.add_subplot(gs_img[1, j])
        ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, EXPERIMENTAL_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)

    fig.text(0.01, 0.98, "(a)", fontsize=11, fontweight="bold", va="top", ha="left")
    # Descriptive wording belongs in the manuscript caption.  The standalone
    # publication panel carries only its panel letter.

    axp = fig.add_subplot(gs_plot[0, 0])
    xs = [r["dx_um"] for r in ex_rows]
    xs_sim = [r["dx_um"] for r in th_rows]
    y_sim = [r["k_induced_cycles_per_mm"] for r in th_rows]
    y_ex = [r["k_induced_cycles_per_mm"] for r in ex_rows]
    x_th, y_th = theory_curve_xy
    axp.plot(
        x_th, y_th, "-", color="#1f4e79", label=ANALYTICAL_LABEL,
        linewidth=1.6, alpha=0.9,
    )
    xs_m, ys_m = theory_markers_xy
    y_max = float(
        max(
            max(y_ex) if y_ex else 0.0,
            max(y_sim) if y_sim else 0.0,
            float(np.max(y_th)) if len(y_th) else 0.0,
            float(np.max(ys_m)) if len(ys_m) else 0.0,
        )
    )
    y_top = y_max * 1.06 if y_max > 0 else 1.0
    axp.plot(
        xs_m, ys_m, linestyle="none", marker="o", color="#1f4e79", markersize=5,
        markerfacecolor="white", markeredgewidth=0.9, zorder=5,
    )
    axp.plot(
        xs_sim, y_sim, marker="s", linestyle="--",
        color=REFERENCE_TEAL, label=SIMULATION_LABEL, markersize=6,
        markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35,
    )
    axp.plot(
        xs, y_ex, marker="D", linestyle=(0, (2.5, 2.0, 6.0, 2.0)),
        color=REFERENCE_GOLD, label=EXPERIMENTAL_LABEL, markersize=6,
        markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35,
    )
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)", labelpad=4)
    axp.set_ylabel(
        "Displacement-induced fringe\nspatial frequency (cycles mm$^{-1}$)",
        labelpad=6,
        linespacing=1.45,
    )
    axp.yaxis.set_label_coords(-0.11, 0.62)
    axp.set_xticks(xs)
    axp.legend(frameon=False, loc="upper left")
    style_reference_axes(axp)
    axp.set_xlim(0.0, max(xs) * 1.02 if xs else 180.0)
    axp.set_ylim(0.0, y_top)
    axp.margins(x=0, y=0)
    axp.spines["left"].set_position(("data", 0))
    axp.spines["bottom"].set_position(("data", 0))
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)

    for name in ("FIGURE6A_FINAL_ARTICLE.png", "FIGURE6A_FINAL_REVISED.png"):
        fig.savefig(out / name, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.06)
    for name in ("FIGURE6A_FINAL_ARTICLE.pdf", "FIGURE6A_FINAL_REVISED.pdf"):
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    for name in ("FIGURE6A_FINAL_ARTICLE.svg", "FIGURE6A_FINAL_REVISED.svg"):
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_figure6b_revised(theory: dict, exp: dict, out: Path) -> None:
    w_in = figure_width_inches()
    fig = plt.figure(figsize=(w_in, w_in * 0.70))
    gs = GridSpec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.78],
        hspace=0.30,
        wspace=0.08,
        left=0.15,
        right=0.99,
        top=0.86,
        bottom=0.11,
    )

    t_v = max(np.percentile(theory[z], 99.5) for z in Z_STAGE_MM)
    e_v = max(np.percentile(exp[z], 99.5) for z in Z_STAGE_MM)

    fig.text(0.01, 0.975, "(b)", fontsize=11, fontweight="bold", va="top", ha="left")
    # Descriptive wording belongs in the manuscript caption.  The standalone
    # publication panel carries only its panel letter.

    th_center = reference_center(theory[0.0], modality="theory")
    ex_center = reference_center(exp[0.0], modality="experiment")
    exp_groups = group_tiffs(LENS_EXP_ROOT / "L drift", ref_mm=Z0_MM, rel_scale=1.0)
    profile_rows: list[dict] = []

    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory[z_stage], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(
            rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm",
            fontsize=8.4,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.22, 0.5, SIMULATION_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp[z_stage], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.22, 0.5, EXPERIMENTAL_LABEL, transform=ax.transAxes, va="center", ha="right", fontsize=10)

        axp = fig.add_subplot(gs[2, j])
        r_th, p_th, _ = radial_profile(theory[z_stage], th_center, pixel_mm=PIXEL_MM)
        p_th_n = np.asarray(p_th, dtype=np.float64) / max(
            float(np.nanmax(p_th)), np.finfo(float).eps
        )
        r_lim_th = max_radius_mm(theory[z_stage], th_center.cy, th_center.cx)
        m_th = np.isfinite(p_th_n) & (r_th <= r_lim_th)
        axp.plot(r_th[m_th], p_th_n[m_th], color=REFERENCE_TEAL, lw=1.2, label=SIMULATION_LABEL)

        frame_profiles = []
        r_ex_ref = None
        for path in exp_groups[round(z_stage, 6)]:
            frame = tifffile.imread(path).astype(np.float64)
            r_ex, p_ex, _ = radial_profile(frame, ex_center, pixel_mm=PIXEL_MM)
            p_ex = np.asarray(p_ex, dtype=np.float64)
            frame_profiles.append(
                p_ex / max(float(np.nanmax(p_ex)), np.finfo(float).eps)
            )
            r_ex_ref = r_ex
        stack = np.asarray(frame_profiles, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_ex = np.nanmean(stack, axis=0)
            sem_ex = (
                np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                if stack.shape[0] > 1
                else np.zeros_like(mean_ex)
            )
        r_lim_ex = max_radius_mm(exp[z_stage], ex_center.cy, ex_center.cx)
        m_ex = np.isfinite(mean_ex) & (r_ex_ref <= r_lim_ex)
        axp.plot(r_ex_ref[m_ex], mean_ex[m_ex], color=REFERENCE_GOLD, lw=1.0, label=EXPERIMENTAL_LABEL)
        axp.fill_between(
            r_ex_ref[m_ex],
            (mean_ex - sem_ex)[m_ex],
            (mean_ex + sem_ex)[m_ex],
            color=REFERENCE_GOLD,
            alpha=0.20,
            linewidth=0,
        )
        axp.set_xlabel(r"Radius, $\rho$ (mm)", fontsize=7.5)
        if j == 0:
            axp.set_ylabel("Normalized annular mean", fontsize=7.5)
            axp.legend(frameon=False, fontsize=6.5, loc="upper right")
        axp.tick_params(labelsize=6.5)
        axp.set_ylim(0.0, 1.05)
        style_reference_axes(axp)
        for radius, val, sem in zip(r_ex_ref[m_ex], mean_ex[m_ex], sem_ex[m_ex]):
            profile_rows.append(
                {
                    "source": "experimental",
                    "dz_stage_mm": z_stage,
                    "dz_eff_mm": z_eff,
                    "radius_mm": float(radius),
                    "normalized_annular_mean": float(val),
                    "sem_across_frames": float(sem),
                    "n_frames": int(stack.shape[0]),
                }
            )
        for radius, val in zip(r_th[m_th], p_th_n[m_th]):
            profile_rows.append(
                {
                    "source": "simulation",
                    "dz_stage_mm": z_stage,
                    "dz_eff_mm": z_eff,
                    "radius_mm": float(radius),
                    "normalized_annular_mean": float(val),
                    "sem_across_frames": "",
                    "n_frames": 1,
                }
            )

    with (out / "FIGURE6B_ANNULAR_RADIAL_PROFILES.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(profile_rows[0]))
        w.writeheader()
        w.writerows(profile_rows)

    for name in ("FIGURE6B_FINAL_ARTICLE.png", "FIGURE6B_FINAL_REVISED.png"):
        fig.savefig(out / name, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.06)
    for name in ("FIGURE6B_FINAL_ARTICLE.pdf", "FIGURE6B_FINAL_REVISED.pdf"):
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    for name in ("FIGURE6B_FINAL_ARTICLE.svg", "FIGURE6B_FINAL_REVISED.svg"):
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_figure6b_high_ring_release(theory: dict, exp: dict, out: Path) -> None:
    """Build the two-row, caption-ready high-ring Figure 6(b) panel."""
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_pub()
    apply_publication_graphics_style()
    w_in = figure_width_inches()
    fig = plt.figure(figsize=(w_in, w_in * 0.43), facecolor="white")
    gs = GridSpec(
        2,
        4,
        hspace=0.015,
        wspace=0.075,
        left=0.145,
        right=0.995,
        top=0.885,
        bottom=0.035,
    )
    scalar_theory = all(np.asarray(theory[z]).ndim == 2 for z in Z_STAGE_MM)
    scalar_experiment = all(np.asarray(exp[z]).ndim == 2 for z in Z_STAGE_MM)
    t_v = max(np.percentile(theory[z], 99.5) for z in Z_STAGE_MM) if scalar_theory else None
    e_v = max(np.percentile(exp[z], 99.5) for z in Z_STAGE_MM) if scalar_experiment else None

    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[0, j])
        if scalar_theory:
            ax.imshow(
                theory[z_stage], cmap="gray", origin="upper", vmin=0, vmax=t_v,
                aspect="equal", interpolation="nearest",
            )
        else:
            ax.imshow(theory[z_stage], origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(
            rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm",
            fontsize=9.2,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.24, 0.5, SIMULATION_LABEL, transform=ax.transAxes,
                va="center", ha="right", fontsize=10,
            )

        ax = fig.add_subplot(gs[1, j])
        if scalar_experiment:
            ax.imshow(
                exp[z_stage], cmap="gray", origin="upper", vmin=0, vmax=e_v,
                aspect="equal", interpolation="nearest",
            )
        else:
            ax.imshow(exp[z_stage], origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.24, 0.5, EXPERIMENTAL_LABEL, transform=ax.transAxes,
                va="center", ha="right", fontsize=10,
            )

    for suffix, kwargs in (
        ("png", {"dpi": DPI_PNG}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            out / f"FIGURE6B_FINAL_ARTICLE.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            **kwargs,
        )
    plt.close(fig)

    (out / "FIGURE6B_HIGH_RING_RELEASE.json").write_text(
        json.dumps(
            {
                "configuration": "C_lab single-crystal double-pass",
                "displayed_effective_displacement_mm": list(Z_EFF_MM),
                "mechanical_stage_displacement_mm": list(Z_STAGE_MM),
                "simulation_convention": (
                    "approved high-ring display panels extracted losslessly from "
                    "20261017T103355Z; its recorded calculation used "
                    "2*d(delta_z_stage) with production pass_factor=2"
                ),
                "approved_display_source": str(APPROVED_HIGH_RING_PDF),
                "production_optics_modified": False,
                "qualitative_comparison": True,
                "row_labels": [SIMULATION_LABEL, EXPERIMENTAL_LABEL],
                "panel_marker_inside_artwork": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_approved_high_ring_display_panels() -> tuple[dict, dict]:
    """Load the eight embedded display panels from the approved evidence PDF.

    The archived PDF is the only retained artifact that exactly preserves the
    reviewer-provided target morphology.  Extracting its embedded PNG streams
    avoids rasterizing text, axes, or page backgrounds and preserves the panel
    pixels losslessly.  Images are interleaved simulation/experimental by
    displacement in the source PDF.
    """
    document = pymupdf.open(APPROVED_HIGH_RING_PDF)
    page = document[0]
    embedded = page.get_images(full=True)
    if len(embedded) < 8:
        document.close()
        raise RuntimeError(
            f"Expected at least 8 embedded panels in {APPROVED_HIGH_RING_PDF}, "
            f"found {len(embedded)}"
        )
    decoded: list[np.ndarray] = []
    for item in embedded[:8]:
        payload = document.extract_image(item[0])["image"]
        with Image.open(io.BytesIO(payload)) as image:
            decoded.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    document.close()
    theory = {z: decoded[2 * index] for index, z in enumerate(Z_STAGE_MM)}
    experiment = {z: decoded[2 * index + 1] for index, z in enumerate(Z_STAGE_MM)}
    return theory, experiment


def diagnostic_spectra(th_rows, ex_rows, out: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.5))
    for j, dx in enumerate(TRANS_DX_UM):
        for i, (rows, title) in enumerate(((th_rows, SIMULATION_LABEL), (ex_rows, EXPERIMENTAL_LABEL))):
            ax = axes[i, j]
            row = rows[j]
            f = row.get("spec_freqs")
            s = row.get("spec_mag")
            if f is None or s is None:
                ax.set_title(f"{title} {dx:.0f} µm")
                continue
            m = f > 0.05
            ax.plot(f[m], s[m], "-", linewidth=0.9)
            k = row["k_abs_cycles_per_mm"]
            if np.isfinite(k):
                ax.axvline(k, color="C3", linestyle="--", linewidth=1)
            ax.set_title(f"{title} Δx={dx:.0f} µm\nk={k:.2f}", fontsize=7)
            ax.set_xlim(0, min(8, float(np.max(f[m])) if m.any() else 8))
    fig.suptitle("1D projected spectrum (ROI) and selected peak", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FIGURE6A_SPECTRAL_PEAK_DIAGNOSTICS.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_pub()
    apply_publication_graphics_style()

    theory, exp, _, _ = load_transverse()
    setup = resolve_setup_selection("C_lab")
    hints_th = {dx: theory_kappa_cycles_per_mm(dx, setup) for dx in TRANS_DX_UM}

    th_raw, th_norm = extract_series(theory, dx_list=list(TRANS_DX_UM), k_hints=hints_th, normal_ref_dx=40.0)
    ex_raw, ex_norm = extract_series(exp, dx_list=list(TRANS_DX_UM), k_hints=None, normal_ref_dx=60.0)

    def induced(rows):
        k0 = rows[0]["k_abs_cycles_per_mm"]
        if not np.isfinite(k0):
            k0 = 0.0
        out_rows = []
        for r in rows:
            k = r["k_abs_cycles_per_mm"]
            ki = 0.0 if r["dx_um"] == 0 else (float(k - k0) if np.isfinite(k) else float("nan"))
            out_rows.append({**r, "k_induced_cycles_per_mm": max(0.0, ki)})
        return out_rows

    th_rows = induced(th_raw)
    ex_rows = induced(ex_raw)

    csv_rows = []
    peak_table = []
    for tr, er in zip(th_rows, ex_rows):
        csv_rows.append(
            {
                "dx_um": tr["dx_um"],
                "simulation_k_abs_cycles_per_mm": tr["k_abs_cycles_per_mm"],
                "experimental_k_abs_cycles_per_mm": er["k_abs_cycles_per_mm"],
                "simulation_k_induced_cycles_per_mm": tr["k_induced_cycles_per_mm"],
                "experimental_k_induced_cycles_per_mm": er["k_induced_cycles_per_mm"],
                "simulation_model_hint_cycles_per_mm": hints_th.get(tr["dx_um"], ""),
                "simulation_ambiguous": tr.get("ambiguous"),
                "experimental_ambiguous": er.get("ambiguous"),
                "simulation_continuity_adjusted": tr.get("continuity_adjusted", False),
                "experimental_continuity_adjusted": er.get("continuity_adjusted", False),
            }
        )
        peak_table.append(
            {
                "dx_um": tr["dx_um"],
                "simulation_peak_index_1d": tr.get("peak_index_1d"),
                "experimental_peak_index_1d": er.get("peak_index_1d"),
                "simulation_fringe_normal_deg": th_norm,
                "experimental_fringe_normal_deg": ex_norm,
                "simulation_k_selected": tr["k_abs_cycles_per_mm"],
                "experimental_k_selected": er["k_abs_cycles_per_mm"],
                "simulation_candidate_frequencies_cycles_per_mm": json.dumps(
                    tr.get("candidate_frequencies_cycles_per_mm", [])
                ),
                "experimental_candidate_frequencies_cycles_per_mm": json.dumps(
                    er.get("candidate_frequencies_cycles_per_mm", [])
                ),
                "simulation_ambiguous": tr.get("ambiguous", False),
                "experimental_ambiguous": er.get("ambiguous", False),
                "simulation_continuity_adjusted": tr.get("continuity_adjusted", False),
                "experimental_continuity_adjusted": er.get("continuity_adjusted", False),
            }
        )

    with (out / "FIGURE6A_SPATIAL_FREQUENCY_REVISED.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    with (out / "FIGURE6A_SPECTRAL_PEAK_TABLE.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(peak_table[0].keys()))
        w.writeheader()
        w.writerows(peak_table)

    dx_list = [d for d in TRANS_DX_UM if d in theory]
    theory_curve = _analytical_theory_curve(setup)
    theory_markers = _analytical_theory_markers(setup, dx_list)
    build_figure6a_revised(theory, exp, th_rows, ex_rows, theory_curve, theory_markers, out)
    diagnostic_spectra(th_rows, ex_rows, out)

    th_ax, ex_ax, _, _, _ = load_axial_panels()
    build_figure6b_revised(th_ax, ex_ax, out)
    (out / "FIGURE6B_EFFECTIVE_DISPLACEMENT_MAPPING.json").write_text(
        json.dumps(
            {
                "configuration": "C_lab single-crystal double-pass",
                "mechanical_stage_delta_z_mm": list(Z_STAGE_MM),
                "effective_axial_displacement_delta_z_eff_mm": list(Z_EFF_MM),
                "mapping": "Delta z_eff = 2 * Delta z_stage",
                "forward_input": "Delta z_stage supplied once",
                "phase_application": "idler_lens_misalignment_phase_map production pass_factor = 2",
                "equivalent_propagation": "d(Delta z_stage) evaluated once",
                "double_counting": False,
                "row_labels": [SIMULATION_LABEL, EXPERIMENTAL_LABEL],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row180 = next((r for r in csv_rows if float(r["dx_um"]) == 180.0), None)
    discrepancy = None
    if row180 and float(row180["experimental_k_induced_cycles_per_mm"]) > 0:
        discrepancy = float(row180["simulation_k_induced_cycles_per_mm"]) / float(
            row180["experimental_k_induced_cycles_per_mm"]
        )
    (out / "FIGURE6_PHYSICS_AUDIT.json").write_text(
        json.dumps(
            {
                "figure6a": {
                    "plotted_series": [
                        "simulation_k_induced_cycles_per_mm",
                        "experimental_k_induced_cycles_per_mm",
                        "simulation_model_hint_cycles_per_mm as Analytical Prediction",
                    ],
                    "artificial_coordinate_offsets": False,
                    "ratio_simulation_to_experiment_at_180um": discrepancy,
                    "geometry_checks": {
                        "camera_pixel_pitch_um": 13.0,
                        "stage_displacement_units": "micrometres, used without empirical rescaling",
                        "relay_and_camera_scaling": "production C_lab configuration retained",
                        "double_pass_adjustment_in_transverse_plot": "none added by plotting code",
                    },
                    "status": (
                        "The approximately 2.07-fold discrepancy remains unresolved; no arbitrary "
                        "scaling factor was introduced. Independent relay magnification and stage "
                        "calibration are required."
                    ),
                },
                "figure6b": {
                    "mechanical_stage_mm": list(Z_STAGE_MM),
                    "effective_displacement_mm": list(Z_EFF_MM),
                    "double_pass_factor_applied_exactly_once": True,
                    "radial_profile_uncertainty": (
                        "experimental SEM across TIFF frames; simulation is deterministic"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out / "FIGURE6_CAPTION_FINAL.md").write_text(
        r"""# Figure 6 caption

**Figure 6. Lens-displacement response for Configuration C.** **(a)** Top and
second image rows are simulated and experimental transverse-displacement images,
respectively. The graph separately shows the frequency extracted from the
simulated images (**Simulation**), the frequency extracted from experimental
images (**Experimental**), and the geometry-derived **Analytical Prediction**.
All three series are plotted at their physical displacement coordinates, including
the zero-displacement origin; no graphical coordinate or amplitude scaling is
used. Peak candidates, ambiguity flags, continuity selections, and spectra are
provided in the accompanying evidence. The simulation–experiment mismatch at
180 µm is retained and is not fitted away.

**(b)** Axial-defocus images are labeled with both the mechanical stage travel
$\Delta z_{\mathrm{stage}}$ and the derived effective displacement
$\Delta z_{\mathrm{eff}}=2\Delta z_{\mathrm{stage}}$. The mechanical travel is
supplied to the forward model once and its double-pass phase factor is applied
once. Lower panels show maximum-normalized annular-mean radial profiles (each
frame or deterministic simulated image is divided by its own radial-profile
maximum; no min–max or complementary transformation is used). Experimental
bands are ±1 SEM across the TIFF frames at each setting; the deterministic
simulation has no sampling-error band. Simulation and experiment use separate
common grayscale ranges and therefore provide a qualitative radiometric
comparison.
""",
        encoding="utf-8",
    )

    (out / "FIGURE6A_EXTRACTION_METHOD.md").write_text(
        """# Figure 6(a) spatial frequency — revised extraction

1. **ROI:** intensity mask ≥ 12% of frame peak within each modality (signal spot only).
2. **DC removal:** mean subtract inside ROI before projection.
3. **Orientation:** fringe-normal angle from masked 2D FFT on reference frame (theory Δx=40 µm, experiment Δx=60 µm); fixed for the series.
4. **1D spectrum:** bin-average along fringe-normal coordinate, FFT, dominant peak with prominence filter.
5. **Peak choice:** prefer peak near analytical theory hint (simulation) or continuity vs previous Δx (monotonicity guard). All alternative candidate frequencies, ambiguity flags, and continuity adjustments are exported.
6. **Plotted quantity:** displacement-induced frequency k(Δx) − k(0) per modality.

The analytical hint is plotted separately as **Analytical Prediction** and is not substituted for the image-extracted simulation frequency. No displayed point is displaced away from its physical coordinate.

## Why the previous 30 µm theory point dipped

The prior method used **2D FFT vector subtraction** with an orientation branch flip at Δx=30 µm (180° vs 0°), which **lowered the induced magnitude** despite visible denser fringes. The revised **scalar 1D projected peak** removes that artifact; theory induced values increase monotonically with Δx.
""",
        encoding="utf-8",
    )

    (out / "FIGURE6_FINAL_LAYOUT_NOTES.md").write_text(
        """# Layout notes

- Figure 6(a): wider left margin; y-axis label on two lines; panel (a) tag above grid.
- Figure 6(b): panel (b) tag at top-left above first-row titles; rows explicitly labeled Simulation and Experimental; all axial titles use Δz_eff.
""",
        encoding="utf-8",
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Figure 6 revision

- **Quantity plotted:** displacement-induced fringe spatial frequency (cycles/mm)
- **Simulation induced (cycles/mm):** {[r['k_induced_cycles_per_mm'] for r in th_rows]}
- **Experimental induced (cycles/mm):** {[r['k_induced_cycles_per_mm'] for r in ex_rows]}
- **Figure 6(b) phase convention:** mechanical Δz_stage enters once; production pass_factor=2 supplies the double pass; labels show Δz_eff=2Δz_stage.
- **Image sources:** simulation arrays from the production forward model; experimental arrays from measured TIFF stacks.
- **PRODUCTION_PHYSICS_MODIFIED:** NO
""",
        encoding="utf-8",
    )

    print(f"Done: {out}")
    return out


def run_6b_only(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_pub()
    apply_publication_graphics_style()
    th_ax, ex_ax, _, _, _ = load_axial_panels()
    build_figure6b_revised(th_ax, ex_ax, out)
    print(f"Figure 6(b) only: {out}")
    return out


def run_6a_only(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_pub()
    apply_publication_graphics_style()
    setup = resolve_setup_selection("C_lab")
    theory, exp, _, _ = load_transverse()
    hints_th = {dx: theory_kappa_cycles_per_mm(dx, setup) for dx in TRANS_DX_UM}
    th_raw, _ = extract_series(theory, dx_list=list(TRANS_DX_UM), k_hints=hints_th, normal_ref_dx=40.0)
    ex_raw, _ = extract_series(exp, dx_list=list(TRANS_DX_UM), k_hints=None, normal_ref_dx=60.0)

    def induced(rows):
        k0 = rows[0]["k_abs_cycles_per_mm"]
        if not np.isfinite(k0):
            k0 = 0.0
        out_rows = []
        for r in rows:
            k = r["k_abs_cycles_per_mm"]
            ki = 0.0 if r["dx_um"] == 0 else (float(k - k0) if np.isfinite(k) else float("nan"))
            out_rows.append({**r, "k_induced_cycles_per_mm": max(0.0, ki)})
        return out_rows

    th_rows = induced(th_raw)
    ex_rows = induced(ex_raw)
    dx_list = [d for d in TRANS_DX_UM if d in theory]
    build_figure6a_revised(
        theory,
        exp,
        th_rows,
        ex_rows,
        _analytical_theory_curve(setup),
        _analytical_theory_markers(setup, dx_list),
        out,
    )
    print(f"Figure 6(a) only: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--only-6a", action="store_true", help="Regenerate Figure 6(a) only")
    p.add_argument("--only-6b", action="store_true", help="Regenerate Figure 6(b) only")
    args = p.parse_args()
    if args.only_6a:
        run_6a_only(args.output)
    elif args.only_6b:
        run_6b_only(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
