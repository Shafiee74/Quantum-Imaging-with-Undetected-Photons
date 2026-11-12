#!/usr/bin/env python3
"""Standalone Figure 3(c) — object intensity transmission vs interference visibility.

Canonical configurations A–D (manuscript letters). Does not modify production physics,
YAML, Figure 3(a–b), composite Figure 3, or C_lab.
"""
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
from matplotlib.lines import Line2D

from setup_params import resolve_setup_selection, spdc_crystal_state
from simulate import simulate_forward
from verification.figure3a_gain.mdpi_figure3_panel_style import (
    FIG3_DPI_EXPORT,
    FIG3_PANEL_SIZE_IN,
    apply_mdpi_figure3_rcparams,
    style_axes_clean,
)
from verification.physics import independent_equations as iref
from verification.physics.u1_u9_runner import (
    DEFAULT_GRID,
    DEFAULT_NBAR,
    DEFAULT_SEED,
    quiet_state,
    uniform_object,
    vis_block,
)

FIG3C_RASTER_DPI = FIG3_DPI_EXPORT
FIGURE_STEM = "FIGURE3c_FINAL_CANDIDATE"
EVIDENCE_ROOT = ROOT / "evidence" / "figure3c_manuscript_candidate"
CANONICAL_PROFILES = ("A", "B", "C", "D")
T_GRID = np.linspace(0.0, 1.0, 101)
N_MARKERS = 11

# Harmonized with Figure 3(b) candidate styling.
COLOR_TWO_CRYSTAL_ANALYTIC = "#1f4e79"
COLOR_SINGLE_ANALYTIC = "#264653"
COLOR_TWO_CRYSTAL_MARKER = "#2a9d8f"
COLOR_SINGLE_MARKER = "#e9c46a"
LEGEND_FONT_SIZE = 6.5
LEGEND_LOC = "lower right"
LEGEND_BBOX = (0.98, 0.02)

FAMILY_TWO = "two_crystal_AB"
FAMILY_SINGLE = "single_crystal_CD"
REP_TWO = "A"
REP_SINGLE = "C"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def phase_scan_visibility(N1: float, N2: float, C12: complex) -> tuple[float, float, float]:
    thetas = np.linspace(0.0, 2.0 * math.pi, 721)
    imax, imin = -1.0, 1e30
    for th in thetas:
        ip, im = iref.ideal_interference_intensities(N1, N2, C12, float(th))
        imax = max(imax, float(ip), float(im))
        imin = min(imin, float(ip), float(im))
    return imax, imin, iref.visibility_from_max_min(imax, imin)


def run_canonical_sim(
    profile: str,
    T: float,
    *,
    grid: int,
    nbar: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    setup = resolve_setup_selection(profile)
    state = quiet_state(setup, nbar=nbar)
    ref = uniform_object(grid, T=float(T), phi=0.0)
    result = simulate_forward(ref, state, seed=seed, detector_noise=False)
    vo = vis_block(result)
    meta = result.meta or {}
    vis = meta.get("vis") or {}
    eff = meta.get("effective_object_transfer") or {}
    otype = str(getattr(setup, "object_type", ""))
    t_intensity = float(T)
    t_amp = math.sqrt(max(t_intensity, 0.0))
    if otype == "reflective":
        coeff_label = "r_obj_amplitude_sqrt_R"
    else:
        coeff_label = "t_amplitude_sqrt_T"
    N1 = float(vo.get("N_1") or 0.0)
    N2 = float(vo.get("N_2") or 0.0)
    C12a = float(vo.get("C_12_abs") or 0.0)
    C12 = complex(C12a, 0.0)
    V_prod = float(vo.get("V_50_50") or 0.0)
    V_ind = float(iref.visibility_from_moments(N1, N2, C12))
    imax, imin, v_scan = phase_scan_visibility(N1, N2, C12)
    return setup, {
        "configuration": profile,
        "setup_id": setup.setup_id,
        "intensity_T": float(T),
        "object_type": otype,
        "object_device": getattr(setup, "object_device", None),
        "effective_field_amplitude": t_amp,
        "coefficient_kind": coeff_label,
        "T_seed_mean": vis.get("T_seed_mean"),
        "N1": N1,
        "N2": N2,
        "C12_abs": C12a,
        "I_max": imax,
        "I_min": imin,
        "V_production": V_prod,
        "V_independent_moments": V_ind,
        "V_phase_scan": v_scan,
        "transfer_equation": eff.get("transfer_equation"),
    }


def sweep_all(*, grid: int, nbar: float, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v_one: dict[str, float] = {}
    for profile in CANONICAL_PROFILES:
        _, row = run_canonical_sim(profile, 1.0, grid=grid, nbar=nbar, seed=seed)
        v_one[profile] = row["V_production"]
    for profile in CANONICAL_PROFILES:
        for T in T_GRID:
            _, row = run_canonical_sim(profile, float(T), grid=grid, nbar=nbar, seed=seed)
            V0 = v_one[profile]
            row["V_at_T1"] = V0
            row["V_normalized_diagnostic"] = (
                row["V_production"] / V0 if V0 > 0 else float("nan")
            )
            rows.append(row)
    return rows


def family_for(profile: str) -> str:
    return FAMILY_TWO if profile in ("A", "B") else FAMILY_SINGLE


def overlap_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    def series(p: str) -> np.ndarray:
        return np.array(
            [r["V_production"] for r in rows if r["configuration"] == p],
            dtype=np.float64,
        )

    va, vb, vc, vd = series("A"), series("B"), series("C"), series("D")
    return {
        "max_abs_V_A_minus_B": float(np.max(np.abs(va - vb))),
        "max_abs_V_C_minus_D": float(np.max(np.abs(vc - vd))),
        "max_abs_V_A_minus_C": float(np.max(np.abs(va - vc))),
        "max_abs_V_B_minus_D": float(np.max(np.abs(vb - vd))),
    }


def low_gain_reference_rows(
    rows: list[dict[str, Any]], overlap: dict[str, float]
) -> list[dict[str, Any]]:
    """Endpoint-normalized sqrt(T) benchmark per family (amplitude argument to U6 law)."""
    out: list[dict[str, Any]] = []
    for family, rep in ((FAMILY_TWO, REP_TWO), (FAMILY_SINGLE, REP_SINGLE)):
        v0 = next(
            r["V_production"]
            for r in rows
            if r["configuration"] == rep and abs(r["intensity_T"] - 1.0) < 1e-12
        )
        for T in T_GRID:
            t = math.sqrt(max(float(T), 0.0))
            v_ref = iref.low_gain_visibility_vs_transmission(t, v0)
            out.append(
                {
                    "family": family,
                    "representative_configuration": rep,
                    "intensity_T": float(T),
                    "V_low_gain_sqrt_endpoint_normalized": float(v_ref),
                    "V0_from_production_at_T1": float(v0),
                    "normalization_note": (
                        "Endpoint-normalized low-gain benchmark V0*sqrt(T) with "
                        "V0=V_production(T=1) for family representative; sqrt(T) is "
                        "single-pass amplitude scaling (U6 independent_equations facade)."
                    ),
                }
            )
    return out


def marker_indices(n: int, count: int, phase: int = 0) -> np.ndarray:
    idx = np.unique(np.round(np.linspace(0, n - 1, count)).astype(int))
    if phase:
        idx = np.unique((idx + phase) % n)
    return idx


def draw_figure(
    prod_rows: list[dict[str, Any]],
    analytic_rows: list[dict[str, Any]],
    *,
    y_top: float,
) -> plt.Figure:
    apply_mdpi_figure3_rcparams()
    fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZE_IN)

    def family_analytic(family: str) -> tuple[np.ndarray, np.ndarray]:
        pts = [r for r in analytic_rows if r["family"] == family]
        pts.sort(key=lambda r: r["intensity_T"])
        T = np.array([r["intensity_T"] for r in pts])
        V = np.array([r["V_low_gain_sqrt_endpoint_normalized"] for r in pts])
        return T, V

    T2, Va2 = family_analytic(FAMILY_TWO)
    T1, Va1 = family_analytic(FAMILY_SINGLE)
    ax.plot(
        T2,
        Va2,
        color=COLOR_TWO_CRYSTAL_ANALYTIC,
        lw=2.0,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        T1,
        Va1,
        color=COLOR_SINGLE_ANALYTIC,
        lw=2.0,
        ls=(0, (4, 2)),
        solid_capstyle="round",
        zorder=2,
    )

    def plot_markers(rep: str, color: str, marker: str, phase: int) -> None:
        pts = sorted(
            [r for r in prod_rows if r["configuration"] == rep],
            key=lambda r: r["intensity_T"],
        )
        n = len(pts)
        idx = marker_indices(n, N_MARKERS, phase=phase)
        sub = [pts[i] for i in idx]
        ax.plot(
            [r["intensity_T"] for r in sub],
            [r["V_production"] for r in sub],
            linestyle="none",
            marker=marker,
            color=color,
            mec="0.25",
            mew=0.55,
            ms=6.5,
            mfc=color,
            zorder=5,
        )

    plot_markers(REP_TWO, COLOR_TWO_CRYSTAL_MARKER, "o", phase=0)
    plot_markers(REP_SINGLE, COLOR_SINGLE_MARKER, "D", phase=5)

    ax.set_xlabel(r"Effective idler intensity throughput, $T$")
    ax.set_ylabel(r"Interference visibility, $V$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.02, y_top)
    style_axes_clean(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_TWO_CRYSTAL_ANALYTIC,
            lw=2.0,
            label="Scaled analytical guide (A/B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_TWO_CRYSTAL_MARKER,
            markeredgecolor="0.25",
            markeredgewidth=0.55,
            markersize=6.5,
            linestyle="None",
            label="Two-crystal (A, B) — Simulation",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_SINGLE_ANALYTIC,
            lw=2.0,
            ls=(0, (4, 2)),
            label="Scaled analytical guide (C/D)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COLOR_SINGLE_MARKER,
            markeredgecolor="0.25",
            markeredgewidth=0.55,
            markersize=6.5,
            linestyle="None",
            label="Single-crystal (C, D) — Simulation",
        ),
    ]
    ax.legend(
        handles=handles,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.45,
        borderaxespad=0.35,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    return fig


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=FIG3C_RASTER_DPI)
    fig.savefig(f"{stem}.pdf")
    try:
        fig.savefig(
            f"{stem}.tif",
            dpi=FIG3C_RASTER_DPI,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(f"{stem}.tif", dpi=FIG3C_RASTER_DPI, format="tiff")


def runtime_parameters(rows: list[dict[str, Any]], nbar: float) -> dict[str, Any]:
    setups = {p: resolve_setup_selection(p) for p in CANONICAL_PROFILES}
    out: dict[str, Any] = {
        "nbar_request": nbar,
        "DEFAULT_NBAR": DEFAULT_NBAR,
        "grid": DEFAULT_GRID,
        "seed": DEFAULT_SEED,
        "T_grid_points": len(T_GRID),
        "canonical_profiles": list(CANONICAL_PROFILES),
        "configurations": {},
    }
    for p, s in setups.items():
        st1 = spdc_crystal_state(s, interaction=1)
        st2 = spdc_crystal_state(s, interaction=2)
        out["configurations"][p] = {
            "setup_id": s.setup_id,
            "spdc_architecture": s.spdc_architecture,
            "imaging_mode": s.imaging_mode,
            "object_type": s.object_type,
            "T_crystal_NL1_C": float(st1["T_crystal_C"]),
            "T_crystal_NL2_C": float(st2["T_crystal_C"]),
            "crystal_length_mm_NL1": float(st1["crystal_length_mm"]),
            "lambda_p_nm": float(s.lambda_p_nm),
            "lambda_s_nm": float(s.lambda_s_nm),
            "lambda_i_nm": float(s.lambda_i_nm),
            "poling_period_um": float(s.poling_period_um),
            "qpm_order": int(s.qpm_order),
        }
    v1 = {p: next(r["V_production"] for r in rows if r["configuration"] == p and r["intensity_T"] == 1.0) for p in CANONICAL_PROFILES}
    out["V_at_T1"] = v1
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def verification_report(
    rows: list[dict[str, Any]],
    analytic: list[dict[str, Any]],
    overlap: dict[str, float],
) -> str:
    max_mom = max(abs(r["V_production"] - r["V_independent_moments"]) for r in rows)
    max_scan = max(abs(r["V_production"] - r["V_phase_scan"]) for r in rows)
    bench_errs = []
    for r in rows:
        fam = family_for(r["configuration"])
        ref_pts = [a for a in analytic if a["family"] == fam]
        ref_v = next(
            a["V_low_gain_sqrt_endpoint_normalized"]
            for a in ref_pts
            if abs(a["intensity_T"] - r["intensity_T"]) < 1e-9
        )
        bench_errs.append(abs(r["V_production"] - ref_v))
    max_bench = max(bench_errs)
    rel_bench = [
        abs(r["V_production"] - next(
            a["V_low_gain_sqrt_endpoint_normalized"]
            for a in analytic
            if a["family"] == family_for(r["configuration"])
            and abs(a["intensity_T"] - r["intensity_T"]) < 1e-9
        ))
        / max(r["V_production"], 1e-15)
        for r in rows
        if r["intensity_T"] > 0
    ]
    return "\n".join(
        [
            "# Figure 3(c) verification report",
            "",
            f"- Max |V_production − V_independent_moments|: {max_mom:.3e}",
            f"- Max |V_production − V_phase_scan|: {max_scan:.3e}",
            f"- Max absolute deviation from endpoint-normalized low-gain √T benchmark: {max_bench:.3e}",
            f"- Max relative deviation from benchmark (T>0): {max(rel_bench):.3e}",
            "",
            "## Architecture overlap (production sweep)",
            "",
            f"- max |V_A − V_B|: {overlap['max_abs_V_A_minus_B']:.3e}",
            f"- max |V_C − V_D|: {overlap['max_abs_V_C_minus_D']:.3e}",
            f"- max |V_A − V_C|: {overlap['max_abs_V_A_minus_C']:.3e}",
            "",
            "No display offsets applied. Markers: representative A (two-crystal) and C (single-crystal).",
        ]
    ) + "\n"


def layout_qc_report(y_top: float) -> str:
    return "\n".join(
        [
            "# Figure 3(c) layout QC",
            "",
            "- White background, no grid (mdpi_figure3_panel_style).",
            f"- Panel size 5.6 × 3.8 in; PNG/TIFF {FIG3C_RASTER_DPI} dpi; PDF vector (pdf.fonttype=42).",
            f"- y-axis upper limit: {y_top:.4f} (data-driven padding, not forced to 1.0).",
            "- Legend lower-right; four entries (Analytical + Simulation per family).",
            "- No annotation box, no ±0.022 offsets, no jitter.",
            "- Markers: teal circles (two-crystal), gold diamonds (single-crystal).",
            "",
            "Manual check: inspect PNG at publication size for legend/marker overlap.",
        ]
    ) + "\n"


def write_caption(path: Path) -> None:
    path.write_text(
        """**Figure 3(c).** Interference visibility $V$ versus effective object intensity \
transmission $T$ for the generic QIUP architectures (low-occupancy source, \
$\\bar n \\approx 0.01$). For two-crystal configurations **A** and **B**, $T=|t|^2$ \
where $t$ is the complex single-pass **transmission** coefficient of the transmissive \
object between crystals. For single-crystal Michelson configurations **C** and **D**, \
$T=|r_{\\mathrm{obj}}|^2$ where $r_{\\mathrm{obj}}$ is the complex **reflection** \
coefficient of the reflective idler-end SLM (one effective object interaction; two SPDC \
interactions in the crystal). Solid (dashed) **Analytical** curves: square-root \
approximation $V_{\\mathrm{ana}}(T)=V_{\\mathrm{sim}}(1)\\,\\sqrt{T}$, where \
$V_{\\mathrm{sim}}(1)$ is the full-transmission visibility from the production \
simulation for each architecture family; these describe approximate $T$-dependence \
only and are not independently predicted absolute-visibility curves. **Simulation** \
markers: finite-gain production Gaussian-moment outputs \
$V=2|C_{12}|/(N_1+N_2)$ (configurations **A** and **C** shown; **B** and **D** \
coincide numerically for uniform $T$). Near-field and far-field pairs within each \
family overlap for a uniform object map. No display offsets are applied. Laboratory \
**C\\_lab** (transmissive object with fixed mirror) is excluded.
""",
        encoding="utf-8",
    )


def write_methods(path: Path) -> None:
    path.write_text(
        """### Object transmission and visibility (Figure 3(c); Methods excerpt)

**Common axis $T$.** The abscissa is an effective single-interaction intensity \
throughput: $T=|t|^2$ for transmissive objects (A, B) and $T=|r_{\\mathrm{obj}}|^2$ \
for the reflective SLM object (C, D). Field amplitudes enter the forward model as \
$\\sqrt{T}$ (phase set to zero on the uniform sweep).

**Visibility.** Detector-plane Gaussian moments yield $V=2|C_{12}|/(N_1+N_2)$, \
equivalent to fringe contrast $(I_{\\max}-I_{\\min})/(I_{\\max}+I_{\\min})$ under \
the implemented fixed combiner phase scan.

**Architecture families.** Two-crystal (A, B) and single-crystal Michelson (C, D) \
produce distinct $V(T)$ curves; within each family, near-field and far-field imaging \
coincide for spatially uniform $T$.

**Analytical curves (legend).** Under low-gain assumptions with $|C_{12}|\\propto\\sqrt{T}$ \
and weakly $T$-dependent $N_1+N_2$, $V(T)/V(1)\\approx\\sqrt{T}$. Plotted \
analytical curves use $V_{\\mathrm{ana}}(T)=V_{\\mathrm{sim}}(1)\\,\\sqrt{T}$ with \
$V_{\\mathrm{sim}}(1)$ from the production simulation at $T=1$ per family. They \
approximate functional dependence on $T$ and are not independent absolute-visibility \
predictions. **Simulation** markers are the full finite-gain forward model at \
$\\bar n\\approx 0.01$. Moment-identity and phase-scan checks verify implementation \
consistency separately from the analytical approximation.

**Generic C/D vs C\\_lab.** Stock C/D use a reflective end object, not a transmissive \
double-pass object map. The laboratory C\\_lab configuration (transmissive object plus \
fixed mirror, separate parameters) is documented independently and is not shown here.
""",
        encoding="utf-8",
    )


def write_figure1_note(path: Path) -> None:
    path.write_text(
        """# Figure 1 schematic clarification (manuscript note)

Generic single-crystal panels (C, D) should depict the **reflective SLM** as the \
idler-end element effectively replacing the fixed mirror $M_i$, consistent with \
`object_type: reflective` in `setup_single_crystal_*_field.yaml`. This task does \
not modify Figure 1 artwork.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = utc_stamp()
    out_dir = args.output or (EVIDENCE_ROOT / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    prod = sweep_all(grid=DEFAULT_GRID, nbar=DEFAULT_NBAR, seed=DEFAULT_SEED)
    overlap = overlap_metrics(prod)
    analytic = low_gain_reference_rows(prod, overlap)

    y_max = max(r["V_production"] for r in prod)
    y_top = min(1.08, math.ceil((y_max * 1.06) * 20) / 20.0)
    if y_top < y_max + 0.03:
        y_top = y_max + 0.05

    fig = draw_figure(prod, analytic, y_top=y_top)
    export_figure(fig, out_dir / FIGURE_STEM)
    plt.close(fig)

    write_csv(out_dir / "figure3c_production_visibility.csv", prod)
    write_csv(out_dir / "figure3c_analytical_reference.csv", analytic)
    ind_rows = [
        {
            "configuration": r["configuration"],
            "intensity_T": r["intensity_T"],
            "V_production": r["V_production"],
            "V_independent_moments": r["V_independent_moments"],
            "abs_err_moments": abs(r["V_production"] - r["V_independent_moments"]),
        }
        for r in prod
    ]
    write_csv(out_dir / "figure3c_independent_verification.csv", ind_rows)
    scan_rows = [
        {
            "configuration": r["configuration"],
            "intensity_T": r["intensity_T"],
            "V_production": r["V_production"],
            "V_phase_scan": r["V_phase_scan"],
            "I_max": r["I_max"],
            "I_min": r["I_min"],
            "abs_err_phase_scan": abs(r["V_production"] - r["V_phase_scan"]),
        }
        for r in prod
    ]
    write_csv(out_dir / "figure3c_phase_scan_verification.csv", scan_rows)
    write_csv(
        out_dir / "figure3c_architecture_overlap.csv",
        [{"metric": k, "value": v} for k, v in overlap.items()],
    )

    runtime = runtime_parameters(prod, DEFAULT_NBAR)
    runtime["overlap"] = overlap
    runtime["display_offsets_applied"] = False
    (out_dir / "effective_runtime_parameters.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )

    write_caption(out_dir / "FIGURE3c_CAPTION_CANDIDATE.md")
    write_methods(out_dir / "FIGURE3c_METHODS_EXCERPT.md")
    write_figure1_note(out_dir / "FIGURE1_TOPOLOGY_NOTE.md")
    (out_dir / "FIGURE3c_VERIFICATION_REPORT.md").write_text(
        verification_report(prod, analytic, overlap), encoding="utf-8"
    )
    (out_dir / "LAYOUT_QC_REPORT.md").write_text(layout_qc_report(y_top), encoding="utf-8")
    (out_dir / "LEGEND_LAYOUT_TERMINOLOGY_REPORT.md").write_text(
        "\n".join(
            [
                "# Figure 3(c) legend layout and terminology",
                "",
                "- **Legend position:** lower-right (`bbox_to_anchor=(0.98, 0.02)`), "
                "four entries, no display offsets.",
                "- **Legend labels:** Analytical / Simulation (two-crystal and "
                "single-crystal families).",
                "- **Caption/Methods:** analytical curves "
                "$V_{\\mathrm{ana}}(T)=V_{\\mathrm{sim}}(1)\\sqrt{T}$; simulation "
                "markers unchanged finite-gain production outputs.",
                "- Only wording and documentation changed; numerical arrays unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "REPRODUCE.md").write_text(
        "\n".join(
            [
                "# Reproduce Figure 3(c) candidate",
                "",
                "```powershell",
                f'cd "{ROOT}"',
                "python tools/figure3c_manuscript_candidate.py",
                "python -m pytest tests/test_figure3c_manuscript_candidate.py -q",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
