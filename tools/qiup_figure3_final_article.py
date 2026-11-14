#!/usr/bin/env python3
"""Final publication Figure 3 — synchronized verified panel generators.

Panels (a), (c), and (d) are regenerated from their independent numerical
verification paths. Panel (e) is regenerated from the hash-locked
fixed-detector raw-response evidence.

Panel (b) overlays:
  - equilibrium crystal temperature where Δk = 0 (x = 0 on ΔkL/(2π));
  - C/D single NL at 83.7 °C (NL1 benchmark); A/B NL1/NL2 at 83.7/84.7 °C (Lemos 2014).

Writes to ``evidence/qiup_figure3_final_article/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import tools.taskY5_figure3_final_corrections as y5
from tools import figure3a_manuscript_final as figure3a
from tools import figure3c_manuscript_candidate as figure3c
from tools import figure3d_manuscript_candidate as figure3d
from tools import figure3e_fixed_detector_evidence as fixed3e
from tools.publication_graphics_style import (
    CONFIGURATION_COLORS,
    CONFIGURATION_LINESTYLES,
    CONFIGURATION_MARKERS,
    apply_publication_graphics_style,
    style_reference_axes,
)

Y5 = y5.OUT
EVIDENCE_ROOT = ROOT / "evidence" / "qiup_figure3_final_article"
DPI_PNG = 600

CAPTION_PANEL_B_NOTE = (
    "Panel (b): the normalized analytical response "
    "$R=\\mathrm{sinc}^{2}(\\Delta kL/2)$ is shown over the symmetric range "
    "$\\Delta kL/(2\\pi)\\in[-1.5,1.5]$, including the complete central lobe, "
    "adjacent side-lobe portions, and zeros at $\\pm1$. The vertical dotted line at zero marks "
    "phase matching ($T\\approx75.59~^\\circ$C for the reference Sellmeier model). "
    "Operating markers: NL1 in A/B and NL in C/D at 83.7 °C; NL2 in A/B at 84.7 °C "
    "(Lemos et al., 2014 / adopted NL1 reference for single-crystal C/D)."
)

CAPTION_PANEL_CD_NOTE = (
    "Panel (c): production visibility $V=2|C_{12}|/(N_1+N_2)$ versus effective "
    "object intensity response $T$. For two-crystal A/B, $T=|t|^2$ is the "
    "single-pass transmissive-object intensity transmission. For single-crystal "
    "C/D, $T=|r_{\\mathrm{obj}}|^2$ is the reflective end-object intensity "
    "coefficient for one effective object interaction; the crystal is traversed "
    "twice. A/B and C/D coincide within their respective families and therefore "
    "use combined labels; no display offsets are applied. Panel (d): raw spatial "
    "mean of the simulated plus-port signal intensity versus the uniform object "
    "phase on $[0,2\\pi]$. The displayed curves are not peak-, mean-, or min-max "
    "normalized; extrema and visibility are retained in the evidence tables."
)

CAPTION_PANEL_E_NOTE = (
    "Panel (e): numerical grid-sensitivity assessment on a fixed physical 48 x 48 "
    "detector with 16 um pitch and 768 um field of view. Independent field/object "
    "grids are refined while raw amplitude-step response "
    "$\\Delta I=I_{\\mathrm{step}}-I_{\\mathrm{clear}}$ is area-averaged onto the same "
    "detector. The displayed metric is the physical-coordinate weighted relative "
    "$L_2$ error against the finest evaluated grid, $N_{\\mathrm{ref}}=240$, without "
    "peak normalization or coarse-grid interpolation. The reference is finite-grid, "
    "so no asymptotic-convergence claim is made."
)

# Publication styling only. Numerical curves are never shifted for display.
FIG3C_COLORS = CONFIGURATION_COLORS
FIG3C_DASH = CONFIGURATION_LINESTYLES
FIG3C_MARKER_PHASE = {"A": 0, "B": 1, "C": 2, "D": 3}


def legend_inside_configs(
    ax,
    *,
    corner: str = "upper right",
    ncol: int = 1,
) -> None:
    """Configuration legend inside the subplot (upper/lower corners)."""
    anchors: dict[str, tuple[tuple[float, float], str]] = {
        "lower right": ((0.98, 0.03), "lower right"),
        "lower left": ((0.03, 0.03), "lower left"),
        "upper right": ((0.98, 0.97), "upper right"),
        "upper left": ((0.03, 0.97), "upper left"),
        # Panel (e): legend sits under the two-line note (note at y≈0.97).
        "below_note_right": ((0.98, 0.68), "upper right"),
    }
    xy, loc = anchors.get(corner, anchors["upper right"])
    leg = ax.legend(
        fontsize=6.5,
        frameon=False,
        fancybox=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cccccc",
        loc=loc,
        bbox_to_anchor=xy,
        bbox_transform=ax.transAxes,
        ncol=ncol,
        borderaxespad=0.35,
        handlelength=2.2,
        columnspacing=0.9,
    )
    if leg is not None:
        leg.set_zorder(25)


def plot_article_profile(
    ax,
    x,
    y,
    profile: str,
    *,
    label: str | None = None,
    markevery_mod: int | None = 4,
    marker_phase: int | None = None,
    marker_size: float = 7.0,
    zorder_rank: int | None = None,
) -> None:
    """Article styling shared by panels (c), (d), and (e)."""
    color = FIG3C_COLORS[profile]
    z = y5.PROFILES.index(profile) if zorder_rank is None else zorder_rank
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    ax.plot(
        x_arr,
        y_arr,
        color=color,
        linestyle=FIG3C_DASH[profile],
        lw=1.8,
        alpha=0.92,
        zorder=2 + z,
    )
    if markevery_mod is None:
        mask = np.ones(len(x_arr), dtype=bool)
    else:
        ph = FIG3C_MARKER_PHASE[profile] if marker_phase is None else marker_phase
        idx = np.arange(len(x_arr))
        mask = (idx + ph) % markevery_mod == 0
    ax.plot(
        x_arr[mask],
        y_arr[mask],
        linestyle="none",
        marker=CONFIGURATION_MARKERS[profile],
        color=color,
        ms=marker_size,
        mfc="white",
        mec=color,
        mew=1.4,
        zorder=10 + z,
        label=label or f"Configuration {profile}",
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def draw_panel_a_verified(ax, table: list[dict[str, Any]]) -> dict[str, Any]:
    """Full-range gain verification and residuals on the composite axes."""
    r = np.asarray([row["r"] for row in table], dtype=np.float64)
    n_ref = np.asarray([row["nbar_reference_independent"] for row in table], dtype=np.float64)
    n_prod = np.asarray([row["nbar_production_NL1"] for row in table], dtype=np.float64)
    rel_pct = np.asarray([row["rel_error_implementation_percent"] for row in table], dtype=np.float64)
    marker_idx = figure3a.marker_display_indices(len(r))
    ax.plot(r, n_ref, color=FIG3C_COLORS["A"], lw=1.8, label="Analytical reference")
    ax.scatter(
        r[marker_idx], n_prod[marker_idx], s=24, color=FIG3C_COLORS["B"],
        edgecolors="0.2", linewidths=0.35, zorder=5, label="Production simulator",
    )
    ax.set_xlabel(r"Parametric gain, $r$")
    ax.set_ylabel(r"Mean photon number per mode, $\bar{n}$")
    ax.set_xlim(float(r[0]), float(r[-1]))
    ax.set_ylim(0.0, 1.04 * float(max(n_ref.max(), n_prod.max())))
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    inset = ax.inset_axes([0.12, 0.53, 0.42, 0.38])
    mask = (r > 0) & np.isfinite(rel_pct) & (rel_pct > 0)
    inset.plot(r[mask], rel_pct[mask], color=FIG3C_COLORS["D"], lw=0.8, marker="o", ms=2.3)
    inset.set_yscale("log")
    inset.set_xlabel(r"$r$", fontsize=7)
    inset.set_ylabel("Relative error (%)", fontsize=7)
    inset.tick_params(labelsize=6.5, pad=1)
    style_reference_axes(inset)
    style_reference_axes(ax)
    return {
        "r_domain": [float(r[0]), float(r[-1])],
        "verification_points": int(r.size),
        "display_markers": int(marker_idx.size),
        "maximum_relative_error_percent": float(np.nanmax(rel_pct)),
    }


def draw_panel_c_article(
    ax, prod_rows: list[dict[str, Any]], analytic_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Visibility vs effective T with combined labels for coincident families."""
    family_specs = (
        (figure3c.FAMILY_TWO, "A", "Two-crystal (A, B)", "o"),
        (figure3c.FAMILY_SINGLE, "C", "Single-crystal (C, D)", "D"),
    )
    handles: list[Line2D] = []
    for family, representative, label, marker in family_specs:
        ana = sorted(
            [row for row in analytic_rows if row["family"] == family],
            key=lambda row: float(row["intensity_T"]),
        )
        sim = sorted(
            [row for row in prod_rows if row["configuration"] == representative],
            key=lambda row: float(row["intensity_T"]),
        )
        color = FIG3C_COLORS[representative]
        ls = FIG3C_DASH[representative]
        ax.plot(
            [float(row["intensity_T"]) for row in ana],
            [float(row["V_low_gain_sqrt_endpoint_normalized"]) for row in ana],
            color=color, linestyle=ls, lw=1.8, zorder=2,
        )
        idx = figure3c.marker_indices(len(sim), figure3c.N_MARKERS)
        ax.plot(
            [float(sim[i]["intensity_T"]) for i in idx],
            [float(sim[i]["V_production"]) for i in idx],
            linestyle="none", marker=marker, ms=6.0, color=color,
            mfc="white", mec=color, mew=1.1, zorder=5,
        )
        handles.extend(
            [
                Line2D([0], [0], color=color, linestyle=ls, lw=1.8, label=f"{label} — Analytical"),
                Line2D([0], [0], color=color, marker=marker, linestyle="none", mfc="white", label=f"{label} — Simulation"),
            ]
        )
    ax.set_xlabel(r"Effective object intensity response, $T$")
    ax.set_ylabel(r"Visibility, $V$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.03)
    ax.legend(handles=handles, fontsize=6.0, frameon=False, loc="lower right")
    style_reference_axes(ax)
    return {
        "display_offsets_applied": False,
        "combined_labels": ["A/B", "C/D"],
        "overlap": figure3c.overlap_metrics(prod_rows),
        "two_crystal_convention": "T=|t|^2, single-pass transmissive object",
        "single_crystal_convention": "T=|r_obj|^2, reflective end object; one effective object interaction",
    }


def draw_panel_d_simulation_article(
    ax, rows: list[dict[str, Any]], metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    r"""Raw plus-port phase fringe on [0,2pi], without display transforms."""
    for profile in y5.PROFILES:
        series = sorted(
            [row for row in rows if row["configuration"] == profile],
            key=lambda row: float(row["phi_rad"]),
        )
        phi = np.asarray([float(row["phi_rad"]) for row in series])
        intensity = np.asarray([float(row["I_raw_mean"]) for row in series])
        plot_article_profile(ax, phi, intensity, profile, markevery_mod=20, marker_size=4.8)

    ax.set_xlabel(r"Relative object phase, $\varphi$ (rad)")
    ax.set_ylabel(r"Mean plus-port intensity, $\langle I_+\rangle$ (simulation units)")
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_ylim(bottom=0.0)
    legend_inside_configs(ax, corner="upper right", ncol=1)
    style_reference_axes(ax)
    return {
        "observable": "spatial mean of ideal_signal_image",
        "display_transform": "none",
        "normalization": "none",
        "simulation_port": "physical plus port",
        "swept_phase": "uniform object phase",
        "path_override": "delta_L_mm=0 for this verification panel",
        "phi_axis_rad": "[0, 2pi]",
        "extrema_and_visibility": metrics,
    }


def draw_panel_e_article(ax, rows: list[dict]) -> dict[str, Any]:
    """Validated fixed-detector data with the approved Figure 3 graphics style."""
    plot_ns = fixed3e.N_NUM_LEVELS
    # A/C and B/D are physically nearly coincident.  Nested hollow marker sizes
    # and z-order expose all series, including their common zero at N_ref=240,
    # without shifting or otherwise altering data.
    marker_sizes = {"A": 4.5, "B": 6.0, "C": 8.0, "D": 10.0}
    zorder_ranks = {"A": 4, "B": 3, "C": 2, "D": 1}
    ymax = 0.0
    for profile in y5.PROFILES:
        selected = sorted(
            [
                r
                for r in rows
                if r["configuration"] == profile and int(r["N_num"]) in plot_ns
            ],
            key=lambda r: int(r["N_num"]),
        )
        xs = [int(r["N_num"]) for r in selected]
        ys = [float(r["response_relative_L2_vs_N240"]) for r in selected]
        ymax = max(ymax, *ys)
        plot_article_profile(
            ax,
            xs,
            ys,
            profile,
            markevery_mod=None,
            marker_size=marker_sizes[profile],
            zorder_rank=zorder_ranks[profile],
        )
    ax.set_xlabel(r"Internal numerical grid size, $N_\mathrm{num}$")
    ax.set_ylabel(
        r"Relative spatial $L_2$ error vs $N_\mathrm{ref}=240$"
        "\n(fixed 48 \u00d7 48 detector)"
    )
    ax.set_xticks(list(plot_ns))
    ax.set_xlim(36, 252)
    ax.set_ylim(-1.0e-4, ymax * 1.14)
    note = (
        "48 \u00d7 48 detector; 16 \u00b5m pitch; 768 \u00b5m FOV.\n"
        r"Area-weighted raw $\Delta I$ error vs $N_\mathrm{ref}=240$; "
        "finite-grid sensitivity (not asymptotic)."
    )
    # Top edge of note box flush with top spine of axes (no gap above the panel).
    note_artist = ax.text(
        0.5,
        1.0,
        note,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=8.0,
        color="#222",
        linespacing=1.05,
        bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="#cccccc", alpha=0.96, pad=0.0),
        clip_on=False,
        zorder=30,
    )
    ax.figure.canvas.draw()
    note_bbox = note_artist.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    note_bbox_axes = note_bbox.transformed(ax.transAxes.inverted())
    leg = ax.legend(
        fontsize=6.5,
        frameon=False,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cccccc",
        loc="upper center",
        bbox_to_anchor=(0.5, float(note_bbox_axes.y0)),
        bbox_transform=ax.transAxes,
        ncol=4,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=0.85,
    )
    if leg is not None:
        leg.set_zorder(29)
    style_reference_axes(ax)
    return {
        "plot_ns": list(plot_ns),
        "reference_n": fixed3e.N_REFERENCE,
        "metric": "physical-coordinate weighted relative L2 of raw object response",
        "peak_normalization": False,
        "fixed_detector": {
            "n": fixed3e.N_DETECTOR,
            "pitch_um": fixed3e.DETECTOR_PITCH_UM,
            "fov_um": fixed3e.DETECTOR_FOV_UM,
        },
        "asymptotic_convergence_claim": False,
    }


def write_final_decision(out_dir: Path, stamp: str) -> None:
    text = f"""# FINAL DECISION — QIUP Figure 3 (article)

**STATUS:** **CORRECTED / NUMERICALLY VERIFIED** ({stamp[:8]})
**Manuscript composite:** `figure3/FIGURE3_FINAL_ARTICLE.pdf`
**Panel (d):** raw mean signal intensity at the simulated plus output port

**PRODUCTION_PHYSICS_MODIFIED:** NO

## Authoritative bundle (this folder)

- `figure3/FIGURE3_FINAL_ARTICLE.png` (600 dpi)
- `figure3/FIGURE3_FINAL_ARTICLE.pdf`
- `figure3/panels/FIGURE3a_FINAL_ARTICLE.*`, `FIGURE3b_FINAL_ARTICLE.*`, `FIGURE3c_FINAL_ARTICLE.*`, `FIGURE3d_FINAL_ARTICLE.*`, `FIGURE3e_FINAL_ARTICLE.*`
- `FIGURE3_CAPTION_FINAL.md`
- `data/panel_b_complete_sinc2_curve.csv`
- `data/panel_b_complete_sinc2_verification.json`
- `data/panel_a_gain_verification.csv`
- `data/panel_c_visibility_verification.csv`
- `data/panel_d_raw_phase_response.csv`
- `data/panel_e_convergence_audit.json`
- `data/figure3e_fixed_detector_grid_sensitivity_ABCD.csv` (validated raw-response data)
- `data/NUMERICAL_REFERENCE_COMPARISON.json` (value-by-value comparison to 20261020)

No artificial visibility offsets, complementary transformation, or per-curve
normalization is applied in panels (c) or (d).
"""
    (out_dir / "FINAL_DECISION.md").write_text(text, encoding="utf-8")
    (EVIDENCE_ROOT / "LATEST.txt").write_text(
        f"{stamp}\n{out_dir.resolve()}\n", encoding="utf-8"
    )
    sync_current_bundle(out_dir, stamp)
    write_evidence_readme(stamp)


def sync_current_bundle(out_dir: Path, stamp: str) -> None:
    """Stable manuscript path: always copy latest composite into ``CURRENT/``."""
    current = EVIDENCE_ROOT / "CURRENT"
    if current.is_dir():
        shutil.rmtree(current)
    for sub in ("figure3", "figure3/panels", "publication", "data", "provenance"):
        (current / sub).mkdir(parents=True, exist_ok=True)
    copy_globs = (
        "figure3/FIGURE3_FINAL_ARTICLE.png",
        "figure3/FIGURE3_FINAL_ARTICLE.pdf",
        "figure3/FIGURE3_FINAL_ARTICLE.svg",
        "figure3/FIGURE3_CAPTION_FINAL.md",
        "figure3/panels/FIGURE3a_FINAL_ARTICLE.png",
        "figure3/panels/FIGURE3a_FINAL_ARTICLE.pdf",
        "figure3/panels/FIGURE3a_FINAL_ARTICLE.svg",
        "figure3/panels/FIGURE3b_FINAL_ARTICLE.png",
        "figure3/panels/FIGURE3b_FINAL_ARTICLE.pdf",
        "figure3/panels/FIGURE3b_FINAL_ARTICLE.svg",
        "figure3/panels/FIGURE3c_FINAL_ARTICLE.png",
        "figure3/panels/FIGURE3c_FINAL_ARTICLE.pdf",
        "figure3/panels/FIGURE3c_FINAL_ARTICLE.svg",
        "figure3/panels/FIGURE3d_FINAL_ARTICLE.png",
        "figure3/panels/FIGURE3d_FINAL_ARTICLE.pdf",
        "figure3/panels/FIGURE3d_FINAL_ARTICLE.svg",
        "figure3/panels/FIGURE3e_FINAL_ARTICLE.png",
        "figure3/panels/FIGURE3e_FINAL_ARTICLE.pdf",
        "figure3/panels/FIGURE3e_FINAL_ARTICLE.svg",
        "publication/FIGURE3_FINAL_ARTICLE.pdf",
        "publication/FIGURE3_FINAL_ARTICLE.png",
        "publication/FIGURE3_FINAL_ARTICLE.svg",
        "publication/FIGURE3_CAPTION_FINAL.md",
        "FINAL_DECISION.md",
        "data/panel_b_temperature_annotations.json",
        "data/panel_b_complete_sinc2_curve.csv",
        "data/panel_b_complete_sinc2_verification.json",
        "data/panel_a_gain_verification.csv",
        "data/panel_a_gain_verification_summary.json",
        "data/panel_c_visibility_verification.csv",
        "data/panel_c_low_gain_reference.csv",
        "data/panel_c_overlap_verification.json",
        "data/panel_d_raw_phase_response.csv",
        "data/panel_d_extrema_visibility.csv",
        "data/panel_d_moment_verification.csv",
        "data/panel_d_fringe_display.json",
        "data/panel_e_convergence_audit.json",
        "data/panel_e_plot_metadata.json",
        "data/figure3e_fixed_detector_grid_sensitivity_ABCD.csv",
        "data/figure3e_fixed_detector_maps.npz",
        "data/figure3e_fixed_detector_summary.json",
        "data/NUMERICAL_REFERENCE_COMPARISON.json",
        "PHYSICS_TEAM_REPORT_RESPONSE.md",
        "REPRODUCE_FIGURE3e.md",
        "provenance/source_lineage.json",
    )
    for rel in copy_globs:
        src = out_dir / rel
        if src.is_file():
            shutil.copy2(src, current / rel)
    (current / "SOURCE_STAMP.txt").write_text(
        f"Copied from timestamp folder: {stamp}\n"
        f"Full path: {out_dir.resolve()}\n"
        f"Do not edit CURRENT manually; re-run tools/qiup_figure3_final_article.py\n",
        encoding="utf-8",
    )


def write_evidence_readme(stamp: str) -> None:
    readme = f"""# QIUP Figure 3 — final article evidence

Each run of `tools/qiup_figure3_final_article.py` writes a **new timestamp folder**
(`YYYYMMDDTHHMMSSZ/`). Older folders are **archived snapshots**, not updated in place.

## Use this for the manuscript

| Path | Purpose |
|------|---------|
| **`CURRENT/figure3/FIGURE3_FINAL_ARTICLE.pdf`** | Stable composite (a–e), always latest |
| **`CURRENT/publication/FIGURE3_FINAL_ARTICLE.pdf`** | Same, publication copy |
| `CURRENT/figure3/panels/FIGURE3c_FINAL_ARTICLE.*` | Standalone panel (c), generated from the same frozen source data |
| `LATEST.txt` | Timestamp + path of last run |
| `{stamp}/` | Full provenance bundle of last run |

## Why an old folder may look wrong

Example: `20261017T154612Z` only added panel **(b)** annotations. Panels **(c)** and **(d)**
article styling were added in later runs (`155906Z`, `160630Z`, `161158Z`, …).

Check `provenance/source_lineage.json` → field `change` in each folder.

**Latest stamp:** `{stamp}`
"""
    (EVIDENCE_ROOT / "README.md").write_text(readme, encoding="utf-8")


def load_inputs() -> dict:
    """Load unchanged panel (c) inputs; panel (e) is generated independently."""
    rows_c = y5.read_csv(Y5 / "data" / "Figure3c_source.csv")
    return {"rows": rows_c}


def write_panel_e_response(out_dir: Path) -> None:
    response = """# Physics-team Figure 3(e) report response

## Regression identified

The composite generator had reverted to the legacy Y5 panel: successive-grid
L2 on independently peak-normalized maps after coarse-to-fine interpolation.
That path did not implement the report's fixed-detector raw-response method.

## Restored method

- Physical detector: 48 x 48 pixels, 16 um pitch, 768 um field of view.
- Independent numerical field grids: N_num = 48, 96, 144, 192, 240.
- High-resolution fields are equal-area averaged onto the same detector.
- Responsive observable: raw Delta I = I_step - I_clear.
- Primary metric: physical-coordinate weighted relative L2 against the finest
  evaluated N_num=240 grid.
- No independent peak normalization and no coarse-grid interpolation.
- N_num=240 is a finite-grid reference, not a proven asymptotic solution.
- Generic A-D retain the 3 nm filter and uniform final collection; C_lab is not
  selected or modified.

The numerical CSV and raw detector-map archive are SHA-256 guarded. Every
publication run re-derives the displayed metric from those raw maps and aborts
if the validated 20261020 reference differs.
"""
    (out_dir / "PHYSICS_TEAM_REPORT_RESPONSE.md").write_text(
        response, encoding="utf-8"
    )
    reproduce = """# Reproduce Figure 3(e)

Run:

    python tools/figure3e_fixed_detector_evidence.py
    python tools/qiup_figure3_final_article.py

The first command verifies the fixed-detector raw maps and the complete
value-by-value numerical comparison. The second regenerates the standalone
panel and complete Figure 3 composite from the same evidence code.
"""
    (out_dir / "REPRODUCE_FIGURE3e.md").write_text(
        reproduce, encoding="utf-8"
    )


def write_panel_b_evidence(out_dir: Path, x8) -> dict[str, Any]:
    """Record the exact displayed curve and invariant checks for panel (b)."""
    x, response = x8.panel_b_complete_envelope()
    curve_path = out_dir / "data" / "panel_b_complete_sinc2_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["normalized_mismatch_dkl_over_2pi", "normalized_sinc2_response"])
        writer.writerows(zip(x, response))

    op_rows = x8.read_csv(
        x8.X3 / "data" / "U03" / "U03_delta_k_comparison.csv"
    )
    operating_points = [
        {
            "configuration": str(row["profile"]),
            "normalized_mismatch_dkl_over_2pi": float(
                row["normalized_mismatch_dkl_over_2pi"]
            ),
            "T_crystal_C": float(row["T_crystal_C"]),
        }
        for row in op_rows
    ]
    zero_orders = np.asarray(x8.PANEL_B_ZERO_ORDERS, dtype=np.float64)
    checks = {
        "equation": "R(Delta k) = sinc^2(Delta k L/2)",
        "normalized_coordinate": "x = Delta k L/(2 pi); R = numpy.sinc(x)^2",
        "display_x_limits": [-float(x8.PANEL_B_X_LIMIT), float(x8.PANEL_B_X_LIMIT)],
        "sample_count": int(x.size),
        "central_response": float(response[x.size // 2]),
        "maximum_symmetry_residual": float(np.max(np.abs(response - response[::-1]))),
        "zero_orders": [int(v) for v in x8.PANEL_B_ZERO_ORDERS],
        "maximum_analytical_zero_residual": float(np.max(np.sinc(zero_orders) ** 2)),
        "operating_points": operating_points,
        "operating_points_source": (
            "evidence/taskX3_U01_U03_rerun/data/U03/U03_delta_k_comparison.csv"
        ),
        "configuration_coordinates_modified": False,
        "validated_physical_equation_modified": False,
        "physical_parameter_provenance": {
            "configuration": "A generic ppKTP type-0 QPM reference",
            "lambda_p_nm": 532.0,
            "lambda_s_nm": 810.0,
            "lambda_i_nm": 1550.0,
            "crystal_length_mm": 2.0,
            "poling_period_um": 9.675,
            "qpm_order": 1,
            "dispersion": "setup_params.ktp_nz_sellmeier (production); verification.independent_reference (independent)",
            "source_configuration": "configs/setup_config_a.yaml and U03 evidence CSV",
        },
    }
    y5.save_json(out_dir / "data" / "panel_b_complete_sinc2_verification.json", checks)
    return checks


def build_figure(out_dir: Path, t_audit: dict | None = None) -> dict[str, Any]:
    apply_publication_graphics_style()
    x8 = y5._load_x8()
    fig_dir = out_dir / "figure3"
    pub_dir = out_dir / "publication"
    panels_dir = fig_dir / "panels"
    for d in (fig_dir, pub_dir, panels_dir, out_dir / "data", out_dir / "provenance"):
        d.mkdir(parents=True, exist_ok=True)
    panel_e_result = fixed3e.generate_fixed_detector_evidence(out_dir / "data")
    panel_e_rows = panel_e_result["rows"]
    reference_comparison = fixed3e.compare_csv_to_validated_reference(
        out_dir / "data" / fixed3e.CSV_NAME
    )
    y5.save_json(
        out_dir / "data" / "NUMERICAL_REFERENCE_COMPARISON.json",
        reference_comparison,
    )
    if not reference_comparison["exact_numeric_match"]:
        raise AssertionError("Figure 3(e) differs from the validated 20261020 reference")

    # Always regenerate the verification panels from their authoritative code
    # paths.  This prevents a legacy cached display table from restoring the
    # removed panel-(c) offsets or panel-(d) complementary normalization.
    a_prod = figure3a.sweep_production(figure3a.R_GRID_DEFAULT, profile="A")
    a_table = figure3a.build_verification_table(a_prod)
    a_summary = figure3a.summarize_implementation(a_table)
    c_prod = figure3c.sweep_all(
        grid=figure3c.DEFAULT_GRID,
        nbar=figure3c.DEFAULT_NBAR,
        seed=figure3c.DEFAULT_SEED,
    )
    c_overlap = figure3c.overlap_metrics(c_prod)
    c_analytic = figure3c.low_gain_reference_rows(c_prod, c_overlap)
    d_rows = figure3d.sweep_profiles(
        grid=figure3d.DEFAULT_GRID,
        nbar=figure3d.DEFAULT_NBAR,
        seed=figure3d.DEFAULT_SEED,
    )
    d_metrics = figure3d.intensity_metrics(d_rows)
    d_moment = figure3d.moment_verification(d_rows)

    figure3a.save_csv(
        out_dir / "data" / "panel_a_gain_verification.csv",
        a_table,
        list(a_table[0]),
    )
    y5.save_json(out_dir / "data" / "panel_a_gain_verification_summary.json", a_summary)
    figure3c.write_csv(out_dir / "data" / "panel_c_visibility_verification.csv", c_prod)
    figure3c.write_csv(out_dir / "data" / "panel_c_low_gain_reference.csv", c_analytic)
    y5.save_json(out_dir / "data" / "panel_c_overlap_verification.json", c_overlap)
    figure3d.write_csv(out_dir / "data" / "panel_d_raw_phase_response.csv", d_rows)
    figure3d.write_csv(out_dir / "data" / "panel_d_extrema_visibility.csv", d_metrics)
    figure3d.write_csv(out_dir / "data" / "panel_d_moment_verification.csv", d_moment)

    fig = plt.figure(figsize=(12.2, 13.0))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.32,
        wspace=0.28,
        left=0.08,
        right=0.94,
        top=0.96,
        bottom=0.05,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    a_meta = draw_panel_a_verified(ax_a, a_table)
    x8.draw_panel_b(ax_b)
    b_meta = x8.annotate_panel_b_public_temperature(ax_b)
    c_meta = draw_panel_c_article(ax_c, c_prod, c_analytic)
    d_meta = draw_panel_d_simulation_article(ax_d, d_rows, d_metrics)
    e_plot_meta = draw_panel_e_article(ax_e, panel_e_rows)

    # The standalone reference panels use clean white axes without grid lines.
    # Apply that same graphics-only treatment after the legacy panel drawers.
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        style_reference_axes(ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d, ax_e), "abcde"):
        ax.text(
            -0.08 if ax is not ax_e else -0.04,
            1.04,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    stem = "FIGURE3_FINAL_ARTICLE"
    y5.export_figure(fig, fig_dir / stem, dpi=DPI_PNG)
    y5.export_figure(fig, pub_dir / stem, dpi=DPI_PNG)
    plt.close(fig)

    f, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_panel_a_verified(ax, a_table)
    style_reference_axes(ax)
    if ax.get_legend() is not None:
        ax.get_legend().set_frame_on(False)
    ax.text(-0.08, 1.04, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    y5.export_figure(f, panels_dir / "FIGURE3a_FINAL_ARTICLE", dpi=DPI_PNG)
    plt.close(f)

    f, ax = plt.subplots(figsize=(5.2, 4.2))
    x8.draw_panel_b(ax)
    x8.annotate_panel_b_public_temperature(ax)
    style_reference_axes(ax)
    if ax.get_legend() is not None:
        ax.get_legend().set_frame_on(False)
    ax.text(-0.08, 1.04, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    y5.export_figure(f, panels_dir / "FIGURE3b_FINAL_ARTICLE", dpi=DPI_PNG)
    plt.close(f)

    f, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_panel_c_article(ax, c_prod, c_analytic)
    ax.text(-0.08, 1.04, "(c)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    y5.export_figure(f, panels_dir / "FIGURE3c_FINAL_ARTICLE", dpi=DPI_PNG)
    plt.close(f)

    f, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_panel_d_simulation_article(ax, d_rows, d_metrics)
    ax.text(-0.08, 1.04, "(d)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    y5.export_figure(f, panels_dir / "FIGURE3d_FINAL_ARTICLE", dpi=DPI_PNG)
    plt.close(f)

    f, ax = plt.subplots(figsize=(10.5, 3.8))
    draw_panel_e_article(ax, panel_e_rows)
    ax.text(-0.04, 1.04, "(e)", transform=ax.transAxes, fontsize=13, fontweight="bold")
    y5.export_figure(f, panels_dir / "FIGURE3e_FINAL_ARTICLE", dpi=DPI_PNG)
    plt.close(f)

    b_curve_meta = write_panel_b_evidence(out_dir, x8)

    caption = (
        "**Figure 3. Physics and numerical verification.** "
        "**(a)** The production gain-to-occupancy mapping is compared over "
        "$r\\in[0,0.5]$ with the independent relation "
        "$\\bar n=\\sinh^2(r)$; the inset reports pointwise relative residuals.\n\n"
        + CAPTION_PANEL_B_NOTE
        + "\n\n"
        + CAPTION_PANEL_CD_NOTE
        + "\n\n"
        + CAPTION_PANEL_E_NOTE
        + "\n"
    )
    (fig_dir / "FIGURE3_CAPTION_FINAL.md").write_text(caption, encoding="utf-8")
    (pub_dir / "FIGURE3_CAPTION_FINAL.md").write_text(caption, encoding="utf-8")

    y5.save_json(out_dir / "data" / "panel_b_temperature_annotations.json", b_meta)
    y5.save_json(out_dir / "data" / "panel_d_fringe_display.json", d_meta)
    y5.save_json(out_dir / "data" / "panel_e_plot_metadata.json", e_plot_meta)
    y5.save_json(
        out_dir / "provenance" / "source_lineage.json",
        {
            "figure": 3,
            "base_tasks": [
                "independent Figure 3(a), 3(c), and 3(d) generators",
                "fixed-detector Figure 3(e) generator",
            ],
            "change": (
                "panel (a): full-range production-versus-independent gain verification; "
                "panel (b): complete symmetric sinc-squared envelope with explicit zeros, "
                "unchanged calculated operating-point coordinates, and temperature annotations; "
                "panel (c): unmodified physical visibility with overlapping families identified "
                "by combined labels and distinct marker/line encodings; "
                "panel (d): raw plus-port mean intensity over 0–2π without normalization or "
                "complementary transformation; "
                "panel (e): validated fixed-detector raw-response finite-grid L2 "
                "with the approved graphics-only palette"
            ),
            "panel_a_verification": a_summary,
            "panel_c_overlap": c_overlap,
            "panel_c_display": c_meta,
            "panel_d_observable": d_meta,
            "panel_e_numerical_reference": str(fixed3e.REFERENCE_RELEASE),
            "panel_e_validated_csv_sha256": fixed3e.REFERENCE_CSV_SHA256,
            "panel_e_validated_maps_sha256": fixed3e.REFERENCE_MAPS_SHA256,
            "panel_b_annotations": b_meta,
            "panel_b_curve_verification": b_curve_meta,
            "panel_d_display": d_meta,
            "panel_e_audit": panel_e_result["audit"],
            "panel_e_plot": e_plot_meta,
            "panel_e_numerical_comparison": reference_comparison,
        },
    )
    write_panel_e_response(out_dir)
    write_final_decision(out_dir, out_dir.name)

    return {
        "panel_a": a_meta,
        "panel_b": b_meta,
        "panel_b_curve": b_curve_meta,
        "panel_c": c_meta,
        "panel_d": d_meta,
        "panel_e": panel_e_result["audit"],
        "panel_e_plot": e_plot_meta,
        "panel_e_numerical_comparison": reference_comparison,
        "outputs": {"composite": str(fig_dir / f"{stem}.png")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Explicit evidence output directory")
    args = parser.parse_args()
    stamp = utc_stamp()
    out_dir = args.output.resolve() if args.output else EVIDENCE_ROOT / stamp
    result = build_figure(out_dir)
    result["stamp"] = stamp
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_dir.resolve()}")
    print(f"Authoritative pointer: {(EVIDENCE_ROOT / 'LATEST.txt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
