#!/usr/bin/env python3
"""Standalone Figure 3(b) candidate — phase-matching envelope after generic parameter correction.

Does not modify production physics, YAML, frozen Figure 3(a), or Figures 5–6.
Writes versioned evidence under ``evidence/figure3b_manuscript_candidate/<stamp>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
from matplotlib.lines import Line2D

from optics import spdc_crystal_state
from setup_params import (
    crystal_length_m_from_setup,
    delta_k_qpm_per_m_from_setup,
    qpm_poling_period_um,
    resolve_setup_selection,
    spdc_propagation_direction,
)
from verification.figure3a_gain.mdpi_figure3_panel_style import (
    FIG3_PANEL_SIZE_IN,
    apply_mdpi_figure3_rcparams,
    style_axes_clean,
)

FIG3B_RASTER_DPI = 600
from verification.independent_reference import phase_matching_temperature_C
from verification.physics import independent_equations as iref

EVIDENCE_ROOT = ROOT / "evidence" / "figure3b_manuscript_candidate"
FIGURE_STEM = "FIGURE3b_FINAL_CANDIDATE"
LEGEND_FONT_SIZE = 6.5
LEGEND_LOC = "upper right"
LEGEND_BBOX = (0.98, 0.98)

ENVELOPE_COLOR = "#1f4e79"
NL1_MARKER_COLOR = "#2a9d8f"  # teal square — NL1 / NL @ 83.7 °C
NL2_MARKER_COLOR = "#e9c46a"  # gold diamond — NL2 @ 84.7 °C
SUPERSEDES_U03 = ROOT / "evidence" / "taskX3_U01_U03_rerun" / "data" / "U03"

# Manuscript configuration A–D (setup_params letter aliases).
PROFILE_SETUP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}

@dataclass(frozen=True)
class OperatingPoint:
    configuration: str
    crystal_label: str
    interaction: int
    provenance_class: str
    T_operating_C: float


OPERATING_POINTS: tuple[OperatingPoint, ...] = (
    OperatingPoint("A", "NL1", 1, "literature_reported", 83.7),
    OperatingPoint("A", "NL2", 2, "literature_reported", 84.7),
    OperatingPoint("B", "NL1", 1, "literature_reported", 83.7),
    OperatingPoint("B", "NL2", 2, "literature_reported", 84.7),
    OperatingPoint("C", "NL", 1, "adopted_reference", 83.7),
    OperatingPoint("D", "NL", 1, "adopted_reference", 83.7),
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sinc2_unnormalized(delta_k_per_m: float, length_m: float) -> float:
    return float(iref.phase_matching_sinc2(delta_k_per_m, length_m))


def independent_delta_k_at_setup_T(
    setup: Any,
    *,
    T_C: float,
    interaction: int,
) -> float:
    period_um = float(qpm_poling_period_um(setup, T_C=T_C, interaction=interaction))
    backward = spdc_propagation_direction(setup, interaction=interaction) == "backward"
    return float(
        iref.qpm_delta_k_per_m(
            float(setup.lambda_p_nm),
            float(setup.lambda_s_nm),
            float(setup.lambda_i_nm),
            float(T_C),
            period_um,
            int(setup.qpm_order),
            backward=backward,
        )
    )


def compute_envelope(length_m: float) -> tuple[np.ndarray, np.ndarray]:
    dk = np.linspace(-3.0 * math.pi / length_m, 3.0 * math.pi / length_m, 401)
    s = np.array([sinc2_unnormalized(float(x), length_m) for x in dk], dtype=np.float64)
    return dk, s


def compute_operating_rows(length_m: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in OPERATING_POINTS:
        setup = resolve_setup_selection(PROFILE_SETUP[op.configuration])
        crystal = spdc_crystal_state(setup, interaction=op.interaction)
        prod_dk = float(delta_k_qpm_per_m_from_setup(setup, interaction=op.interaction))
        ref_dk = independent_delta_k_at_setup_T(
            setup, T_C=float(crystal["T_crystal_C"]), interaction=op.interaction
        )
        ae = abs(prod_dk - ref_dk)
        norm = prod_dk * length_m / (2.0 * math.pi)
        rows.append(
            {
                "configuration": op.configuration,
                "crystal_label": op.crystal_label,
                "interaction": op.interaction,
                "temperature_provenance": op.provenance_class,
                "T_operating_C": float(crystal["T_crystal_C"]),
                "lambda_p_nm": float(setup.lambda_p_nm),
                "lambda_s_nm": float(setup.lambda_s_nm),
                "lambda_i_nm": float(setup.lambda_i_nm),
                "poling_period_um_at_T": float(crystal["poling_period_um_qpm"]),
                "qpm_order": int(crystal.get("qpm_order") or 1),
                "crystal_length_m": float(crystal["crystal_length_m"]),
                "production_delta_k_per_m": prod_dk,
                "independent_delta_k_per_m": ref_dk,
                "abs_error_rad_per_m": ae,
                "relative_error": ae / abs(prod_dk) if abs(prod_dk) > 0 else float("nan"),
                "sinc2_at_production_dk": sinc2_unnormalized(prod_dk, length_m),
                "normalized_mismatch_dkl_over_2pi": norm,
                "verdict": "PASS" if ae < 1e-6 else "FAIL",
            }
        )
    return rows


def temperature_sweep(setup: Any, *, interaction: int = 1) -> list[dict[str, Any]]:
    temps = np.linspace(55.0, 100.0, 91)
    rows: list[dict[str, Any]] = []
    length_m = float(crystal_length_m_from_setup(setup, interaction=interaction))
    for t in temps:
        T = float(t)
        prod = float(delta_k_qpm_per_m_from_setup(setup, T_C=T, interaction=interaction))
        indep = independent_delta_k_at_setup_T(setup, T_C=T, interaction=interaction)
        rows.append(
            {
                "T_C": T,
                "production_delta_k_per_m": prod,
                "independent_delta_k_per_m": indep,
                "abs_error_rad_per_m": abs(prod - indep),
                "sinc2_production": sinc2_unnormalized(prod, length_m),
                "sinc2_independent": sinc2_unnormalized(indep, length_m),
                "normalized_mismatch_dkl_over_2pi": prod * length_m / (2.0 * math.pi),
            }
        )
    return rows


def verification_temperatures(T_pm0_C: float) -> list[float]:
    T_eq = float(
        phase_matching_temperature_C(532.0, 810.0, 1550.0, 9.675, T_pm0_C)
    )
    base = [83.7, 84.7, T_eq, 70.0, 75.0, 80.0, 90.0]
    return sorted(set(round(x, 4) for x in base))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _normalized_mismatch_label(value: float) -> str:
    """Compact ΔkL/(2π) text for marker-adjacent labels (3 decimal places)."""
    return f"{float(value):.3f}"


def _canonical_operating_rows(op_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    nl1 = next(
        r
        for r in op_rows
        if str(r["configuration"]) == "A" and str(r["crystal_label"]) == "NL1"
    )
    nl2 = next(
        r
        for r in op_rows
        if str(r["configuration"]) == "A" and str(r["crystal_label"]) == "NL2"
    )
    return nl1, nl2


def draw_figure(
    out_dir: Path,
    *,
    dk: np.ndarray,
    sinc2: np.ndarray,
    length_m: float,
    op_rows: list[dict[str, Any]],
    T_eq: float,
    T_pm0_C: float,
) -> None:
    apply_mdpi_figure3_rcparams()
    x = dk * length_m / (2.0 * math.pi)
    fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZE_IN)
    ax.plot(x, sinc2, color=ENVELOPE_COLOR, lw=2.0, zorder=1)

    nl1_row, nl2_row = _canonical_operating_rows(op_rows)
    x_nl1 = float(nl1_row["normalized_mismatch_dkl_over_2pi"])
    y_nl1 = float(nl1_row["sinc2_at_production_dk"])
    x_nl2 = float(nl2_row["normalized_mismatch_dkl_over_2pi"])
    y_nl2 = float(nl2_row["sinc2_at_production_dk"])

    ax.plot(
        x_nl1,
        y_nl1,
        marker="s",
        ms=7.5,
        color=NL1_MARKER_COLOR,
        mec="0.25",
        mew=0.55,
        linestyle="None",
        zorder=4,
    )
    ax.plot(
        x_nl2,
        y_nl2,
        marker="D",
        ms=7.0,
        color=NL2_MARKER_COLOR,
        mec="0.25",
        mew=0.55,
        linestyle="None",
        zorder=4,
    )

    marker_label_fs = max(float(plt.rcParams.get("xtick.labelsize", 8.5)) - 1.5, 6.0)
    lbl_nl1 = _normalized_mismatch_label(x_nl1)
    lbl_nl2 = _normalized_mismatch_label(x_nl2)
    ax.annotate(
        lbl_nl1,
        xy=(x_nl1, y_nl1),
        xytext=(8, 3),
        textcoords="offset points",
        fontsize=marker_label_fs,
        ha="left",
        va="center",
        color="0.20",
        zorder=5,
    )
    ax.annotate(
        lbl_nl2,
        xy=(x_nl2, y_nl2),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=marker_label_fs,
        ha="left",
        va="center",
        color="0.20",
        zorder=5,
    )

    ax.axvline(0.0, color="0.40", lw=1.0, ls=":", alpha=0.9, zorder=2)
    # Beside central peak: left of maximum, closer and slightly higher than prior layout.
    ax.annotate(
        rf"$\Delta k = 0$" + "\n"
        + rf"Calculated $T_{{\Delta k=0}}\approx {T_eq:.1f}\,^\circ\mathrm{{C}}$",
        xy=(-0.22, 0.97),
        xycoords="data",
        fontsize=7,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.72", alpha=0.93, lw=0.6),
        zorder=5,
    )

    ax.set_xlabel(r"Normalized phase mismatch, $\Delta k L/(2\pi)$")
    ax.set_ylabel(r"Normalized response, $\mathrm{sinc}^{2}(\Delta k L/2)$")

    legend_handles = [
        Line2D([0], [0], color=ENVELOPE_COLOR, lw=2.0, label="Analytical envelope"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=NL1_MARKER_COLOR,
            markeredgecolor="0.25",
            markeredgewidth=0.55,
            markersize=7,
            linestyle="None",
            label="NL1 (A/B) and NL (C/D): 83.7 °C",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=NL2_MARKER_COLOR,
            markeredgecolor="0.25",
            markeredgewidth=0.55,
            markersize=7,
            linestyle="None",
            label="NL2 (A/B): 84.7 °C",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.55,
        borderpad=0.35,
        borderaxespad=0.4,
    )
    style_axes_clean(ax)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)

    layout_meta = {
        "legend_entries": [h.get_label() for h in legend_handles],
        "legend_entry_count": len(legend_handles),
        "legend_loc": LEGEND_LOC,
        "legend_bbox_axes_fraction": list(LEGEND_BBOX),
        "operating_markers": {
            "NL1_NL_83_7_C": {
                "x_dkl_over_2pi": x_nl1,
                "sinc2": y_nl1,
                "marker": "square",
                "color": NL1_MARKER_COLOR,
            },
            "NL2_84_7_C": {
                "x_dkl_over_2pi": x_nl2,
                "sinc2": y_nl2,
                "marker": "diamond",
                "color": NL2_MARKER_COLOR,
            },
        },
        "vertical_reference_lines": ["Delta_k_zero_only"],
        "delta_k_zero_annotation_data_xy": [-0.22, 0.97],
        "marker_label_offsets_points": {"NL1_NL": [8, 3], "NL2": [8, -4]},
        "operating_point_dashed_vlines": "removed",
        "marker_adjacent_labels_dkl_over_2pi": {"NL1_NL": lbl_nl1, "NL2": lbl_nl2},
        "raster_export_dpi": FIG3B_RASTER_DPI,
    }
    (out_dir / "LAYOUT_POLISH.json").write_text(json.dumps(layout_meta, indent=2), encoding="utf-8")

    stem = out_dir / FIGURE_STEM
    fig.savefig(f"{stem}.png", dpi=FIG3B_RASTER_DPI)
    fig.savefig(f"{stem}.pdf")
    try:
        fig.savefig(
            f"{stem}.tif",
            dpi=FIG3B_RASTER_DPI,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(f"{stem}.tif", dpi=FIG3B_RASTER_DPI, format="tiff")
    plt.close(fig)


def write_caption(path: Path, *, T_eq: float) -> None:
    text = f"""\
**Figure 3(b).** Type-0 collinear quasi-phase-matching envelope
$R(\\Delta k)=\\mathrm{{sinc}}^{{2}}(\\Delta k L/2)$ with $\\mathrm{{sinc}}(x)=\\sin x/x$
(un-normalized; $R(0)=1$) versus normalized mismatch $\\Delta k L/(2\\pi)$ for ppKTP
($L=2$ mm, $\\Lambda_{{\\mathrm{{lab}}}}=9.675\\,\\mu\\mathrm{{m}}$, QPM order $m=1$,
532/810/1550 nm). Vertical markers are **model-calculated** operating points:
83.7 °C for NL1 in configurations A and B (Lemos et al., *Nature* **512**, 409–412, 2014)
and for the single-crystal NL reference in C and D (adopted NL1 benchmark for controlled
architecture comparison, not an independent single-crystal oven measurement); 84.7 °C for
NL2 in A and B (same reference). The dotted line at $\\Delta k L/(2\\pi)=0$ indicates the
**model-predicted** zero-mismatch temperature ($T\\approx {T_eq:.1f}$ °C for the implemented
Sellmeier and thermal-expansion conventions); it is distinct from the prescribed operating
temperatures. No empirical $\\Delta k$ offset is applied.
"""
    path.write_text(text, encoding="utf-8")


def write_methods(path: Path, *, T_eq: float, T_pm0: float) -> None:
    text = f"""\
### Phase-matching verification panel (Figure 3(b); Methods excerpt)

**Shared crystal and source parameters (configurations A–D).** All four generic
architectures use the same ppKTP type-0 SPDC reference: pump, signal, and idler
wavelengths 532, 810, and 1550 nm; laboratory poling period $\\Lambda_{{\\mathrm{{lab}}}}=9.675\\,\\mu\\mathrm{{m}}$;
first-order QPM ($m=1$); and crystal length $L=2$ mm per physical nonlinear element.
These values support a **controlled numerical comparison** of interferometer
architectures; they are not asserted to represent a single experimental apparatus
that implemented all four layouts identically.

**Operating temperatures.** For two-crystal configurations A and B, NL1 and NL2 oven
temperatures 83.7 °C and 84.7 °C follow Lemos et al. (2014) for the 810/1550 nm
collection band. For single-crystal double-pass configurations C and D, the single
physical crystal NL is assigned $T=83.7$ °C as an **adopted NL1 reference** from that
same literature set to isolate architecture effects; this is not reported as a measured
single-crystal Michelson oven temperature.

**Phase mismatch.** The model evaluates collinear QPM mismatch
$\\Delta k(T)=k_p-k_s-k_i-2\\pi m/\\Lambda(T)$ with extraordinary indices from
Kato & Takaoka [Appl. Opt. **41**, 5040 (2002)] and linear thermo-optic correction,
and $\\Lambda(T)=\\Lambda_{{\\mathrm{{lab}}}}[1+\\alpha(T-T_{{\\mathrm{{pm0}}}})]$
($\\alpha=6.7\\times10^{{-6}}$ K$^{{-1}}$; $T_{{\\mathrm{{pm0}}}}$ from the
configuration YAML, e.g. {T_pm0:.1f} °C for the single-crystal reference profile).
The plotted response is $R=\\mathrm{{sinc}}^{{2}}(\\Delta k L/2)$ with the ordinary
$\\sin x/x$ convention (not `numpy.sinc`).

**Operating point versus model root.** Prescribed oven temperatures are **inputs** to
the forward model. The temperature at which the **implemented** index model yields
$\\Delta k=0$ (bisection root, $\\approx {T_eq:.1f}$ °C for the parameters above) is a
**model prediction** and is not tuned to match literature ovens. Agreement between
production and independently transcribed Sellmeier routines is a **numerical**
consistency check, not experimental validation of phase-matching temperature.
"""
    path.write_text(text, encoding="utf-8")


def write_supersession_manifest(path: Path, new_data: Path) -> None:
    payload = {
        "supersedes_for_manuscript_generic_figure3b": [
            {
                "path": str(SUPERSEDES_U03 / "U03_delta_k_comparison.csv"),
                "reason": "Generic B/C rows used obsolete 84.3 °C single-crystal temperatures",
            },
            {
                "path": str(ROOT / "evidence" / "qiup_figure3_final_article" / "CURRENT"),
                "reason": "Panel (b) annotations referenced legacy 84.3 °C nominal; do not use until recomposite",
            },
        ],
        "replacement_evidence": str(new_data),
        "figure3a_unchanged": True,
        "figures_5_6_unchanged": True,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(out: Path | None = None) -> Path:
    stamp = utc_stamp()
    out_dir = out or (EVIDENCE_ROOT / stamp)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_ref = resolve_setup_selection("C")
    length_m = float(crystal_length_m_from_setup(setup_ref, interaction=1))
    T_pm0 = float(setup_ref.T_pm0_NL1_C)
    T_eq = float(phase_matching_temperature_C(532.0, 810.0, 1550.0, 9.675, T_pm0))

    dk, sinc2 = compute_envelope(length_m)
    env_rows = [
        {
            "delta_k_per_m": float(d),
            "sinc2": float(s),
            "length_m": length_m,
            "convention": "sin(x)/x squared, x=Delta_k*L/2",
        }
        for d, s in zip(dk, sinc2)
    ]
    write_csv(data_dir / "figure3b_sinc2_envelope.csv", env_rows)

    op_rows = compute_operating_rows(length_m)
    write_csv(data_dir / "figure3b_operating_points.csv", op_rows)
    (data_dir / "figure3b_operating_points.json").write_text(
        json.dumps(op_rows, indent=2), encoding="utf-8"
    )

    sweep = temperature_sweep(setup_ref, interaction=1)
    write_csv(data_dir / "figure3b_temperature_sweep.csv", sweep)

    verify_rows: list[dict[str, Any]] = []
    for T in verification_temperatures(T_pm0):
        prod = float(delta_k_qpm_per_m_from_setup(setup_ref, T_C=T, interaction=1))
        indep = independent_delta_k_at_setup_T(setup_ref, T_C=T, interaction=1)
        verify_rows.append(
            {
                "T_C": T,
                "production_delta_k_per_m": prod,
                "independent_delta_k_per_m": indep,
                "abs_error_rad_per_m": abs(prod - indep),
                "relative_error": abs(prod - indep) / abs(prod) if prod else float("nan"),
                "T_pm0_reference_C": T_pm0,
                "model_T_delta_k_zero_C": T_eq if abs(T - T_eq) < 0.05 else "",
            }
        )
    write_csv(data_dir / "figure3b_independent_verification.csv", verify_rows)

    runtime = {
        "configurations": {
            letter: {
                "registered_profile": PROFILE_SETUP[letter],
                "setup_id": resolve_setup_selection(letter).setup_id,
            }
            for letter in "ABCD"
        },
        "common_wavelengths_nm": [532.0, 810.0, 1550.0],
        "poling_period_lab_um": 9.675,
        "crystal_length_mm": length_m * 1000.0,
        "T_pm0_single_crystal_reference_C": T_pm0,
        "model_T_delta_k_zero_C": T_eq,
        "equation": "Delta_k = k_p - k_s - k_i - 2*pi*m/Lambda(T); sinc2 = (sin(Delta_k*L/2)/(Delta_k*L/2))^2",
        "plot_choice": "sinc2_vs_normalized_mismatch",
        "plot_choice_rationale": (
            "Matches established Figure 3(b)/U03 verification role: show universal "
            "sinc2 envelope with calculated operating points as vertical markers; "
            "temperature enters only through Delta_k at each oven setpoint."
        ),
        "delta_k_at_83_7_C_NL1_rad_per_m": next(
            r["production_delta_k_per_m"]
            for r in op_rows
            if r["configuration"] == "A" and r["crystal_label"] == "NL1"
        ),
        "delta_k_at_84_7_C_NL2_rad_per_m": next(
            r["production_delta_k_per_m"]
            for r in op_rows
            if r["configuration"] == "A" and r["crystal_label"] == "NL2"
        ),
    }
    (data_dir / "effective_runtime_parameters.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )

    draw_figure(
        out_dir,
        dk=dk,
        sinc2=sinc2,
        length_m=length_m,
        op_rows=op_rows,
        T_eq=T_eq,
        T_pm0_C=T_pm0,
    )
    nl1, nl2 = _canonical_operating_rows(op_rows)
    (out_dir / "LAYOUT_POLISH_REPORT.md").write_text(
        "\n".join(
            [
                "# Figure 3(b) final graphical polish",
                "",
                "1. **Legend:** exactly three entries (envelope; NL1/NL 83.7 °C teal square; NL2 84.7 °C gold diamond).",
                f"2. **Legend position:** in-axes upper right (bbox {LEGEND_BBOX}).",
                "3. **Marker mapping:** square/teal = NL1/NL @ 83.7 °C; diamond/gold = NL2 @ 84.7 °C (CSV coordinates).",
                "4. **Vertical lines:** operating-point dashed lines **removed**; only Δk = 0 dotted reference remains.",
                "5. **Δk = 0 annotation:** placed upper-left of peak at data coords (−0.42, 0.90), beside—not on—the central maximum.",
                "6. **Numerical data / physics:** unchanged (see `data/figure3b_operating_points.csv`).",
                "",
                "### Plotted operating coordinates (from CSV)",
                f"- NL1/NL: ΔkL/(2π) = {float(nl1['normalized_mismatch_dkl_over_2pi']):.6f}, "
                f"sinc² = {float(nl1['sinc2_at_production_dk']):.6f}",
                f"- NL2: ΔkL/(2π) = {float(nl2['normalized_mismatch_dkl_over_2pi']):.6f}, "
                f"sinc² = {float(nl2['sinc2_at_production_dk']):.6f}",
                f"- Model T(Δk=0) ≈ {T_eq:.2f} °C",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "MICRO_POLISH_REPORT.md").write_text(
        "\n".join(
            [
                "# Figure 3(b) micro-polish",
                "",
                f"- Added compact **ΔkL/(2π)** labels next to markers: "
                f"**{_normalized_mismatch_label(float(nl1['normalized_mismatch_dkl_over_2pi']))}** (teal square), "
                f"**{_normalized_mismatch_label(float(nl2['normalized_mismatch_dkl_over_2pi']))}** (gold diamond).",
                "- Labels are derived from `figure3b_operating_points.csv` coordinates (not raw Δk in rad/m).",
                f"- Raster export at **{FIG3B_RASTER_DPI} dpi** (PNG/TIFF); PDF remains vector.",
                "- Physics, CSV numerical tables, and marker positions unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "MICRO_LAYOUT_REPORT.md").write_text(
        "\n".join(
            [
                "# Figure 3(b) micro-layout fix",
                "",
                "- **Δk = 0 annotation:** moved to data coordinates (−0.22, 0.97) — higher, closer to peak, still left of the maximum.",
                "- **Legend spacing:** increased `labelspacing` (0.55) for clearer separation of NL2 diamond entry.",
                "- **Marker value labels:** both **0.324** and **0.363** placed on the **right** of markers (8 pt offset, small vertical stagger).",
                "- Physics and numerical CSV values unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "TEXT_LABEL_POLISH_REPORT.md").write_text(
        "\n".join(
            [
                "# Figure 3(b) text/label polish",
                "",
                '- Peak annotation: **"model"** replaced by **"Calculated"** '
                f"(Calculated $T_{{\\Delta k=0}} \\approx {T_eq:.1f}$ °C).",
                "- Marker value labels **0.324** and **0.363**: slightly increased rightward offset (8 pt vs 5 pt), still on the right of markers.",
                "- No other layout, physics, or numerical changes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_caption(out_dir / "FIGURE3b_CAPTION_CANDIDATE.md", T_eq=T_eq)
    write_methods(out_dir / "FIGURE3b_METHODS_EXCERPT.md", T_eq=T_eq, T_pm0=T_pm0)
    write_supersession_manifest(out_dir / "SUPERSEDES_MANIFEST.json", data_dir)

    (out_dir / "REPRODUCE.md").write_text(
        "\n".join(
            [
                "# Reproduce Figure 3(b) candidate",
                "",
                "```powershell",
                f'cd "{ROOT}"',
                "python tools/figure3b_manuscript_candidate.py",
                "python -m pytest tests/test_figure3b_manuscript_candidate.py -q",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "generated_utc": stamp,
        "exports": [
            f"{FIGURE_STEM}.png",
            f"{FIGURE_STEM}.pdf",
            f"{FIGURE_STEM}.tif",
        ],
        "runtime": runtime,
        "operating_points": op_rows,
        "independent_verification_max_abs_error": max(
            r["abs_error_rad_per_m"] for r in verify_rows
        ),
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = run(args.output)
    print(out)


if __name__ == "__main__":
    main()
