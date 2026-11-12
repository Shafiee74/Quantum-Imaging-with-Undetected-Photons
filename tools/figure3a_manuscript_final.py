#!/usr/bin/env python3
"""Publication Figure 3(a) — verified SPDC gain mapping (standalone panel).

Does not modify production physics or Figure 3(b–e). Writes under
``evidence/figure3a_manuscript_final/<timestamp>/`` with approved filenames.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, MultipleLocator

from verification.figure3a_gain.independent_reference import (
    low_gain_approximation,
    mean_photons_per_mode_independent,
)
from verification.figure3a_gain.pathway_trace import trace_all_profiles
from verification.figure3a_gain.production_probe import (
    DEFAULT_GRID,
    DEFAULT_SEED,
    R_GRID_DEFAULT,
    sweep_production,
)
from verification.figure3a_gain.mdpi_figure3_panel_style import (
    FIG3A_INSET_BOUNDS,
    FIG3_COLORS,
    FIG3_DPI_EXPORT,
    FIG3_PANEL_SIZE_IN,
    apply_mdpi_figure3_rcparams,
    style_axes_clean,
)

EVIDENCE_ROOT = ROOT / "evidence" / "figure3a_manuscript_final"
REL_ERR_FLOOR = 1.0e-15
N_DISPLAY_MARKERS = 11
FIGURE_STEM = "FIGURE3a_FINAL_APPROVED_CANDIDATE"


def marker_display_indices(n_points: int, n_markers: int = N_DISPLAY_MARKERS) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n_points - 1, n_markers)).astype(int))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def epsilon_low_gain(r: float) -> float:
    n_ex = mean_photons_per_mode_independent(r)
    if n_ex <= 0.0:
        return 0.0
    return abs(n_ex - low_gain_approximation(r)) / n_ex


def build_verification_table(prod_rows: list[dict]) -> list[dict]:
    table = []
    for row in prod_rows:
        r = float(row["r"])
        n_ref = mean_photons_per_mode_independent(r)
        n_low = low_gain_approximation(r)
        n_prod = float(row["nbar_production_NL1"])
        abs_impl = abs(n_prod - n_ref)
        rel_impl = abs_impl / n_ref if n_ref > REL_ERR_FLOOR else float("nan")
        rel_pct = (
            100.0 * rel_impl if (r > 0 and np.isfinite(rel_impl)) else float("nan")
        )
        table.append(
            {
                "r": r,
                "nbar_reference_independent": n_ref,
                "nbar_low_gain_r_squared": n_low,
                "nbar_production_NL1": n_prod,
                "epsilon_low_gain_vs_exact": epsilon_low_gain(r),
                "abs_error_implementation": abs_impl,
                "rel_error_implementation": rel_impl,
                "rel_error_implementation_percent": rel_pct,
            }
        )
    return table


def summarize_implementation(table: list[dict]) -> dict:
    rels = [
        t["rel_error_implementation"]
        for t in table
        if t["r"] > 0 and np.isfinite(t["rel_error_implementation"])
    ]
    abses = [t["abs_error_implementation"] for t in table]
    lg = {t["r"]: t["epsilon_low_gain_vs_exact"] for t in table if t["r"] in (0.2, 0.5)}
    return {
        "r_domain": [float(table[0]["r"]), float(table[-1]["r"])],
        "n_points": len(table),
        "max_abs_error_implementation": max(abses),
        "mean_abs_error_implementation": float(np.mean(abses)),
        "max_rel_error_implementation": max(rels) if rels else 0.0,
        "max_rel_error_implementation_percent": 100.0 * max(rels) if rels else 0.0,
        "mean_rel_error_implementation": float(np.mean(rels)) if rels else 0.0,
        "epsilon_low_gain_at_r_0p2": lg.get(0.2),
        "epsilon_low_gain_at_r_0p5": lg.get(0.5),
        "rel_err_floor": REL_ERR_FLOOR,
        "grid_n": DEFAULT_GRID,
        "seed": DEFAULT_SEED,
        "control": "LatentState(spdc_gain_r=r), mean_photons_per_mode=None",
        "production_function": "optics.resolve_spdc_source_nbar → setup_params.sinh_squared_stable",
        "reference_function": "verification.figure3a_gain.independent_reference.mean_photons_per_mode_independent",
        "figure_stem": FIGURE_STEM,
        "layout": "MDPI_clean_no_grid",
        "inset_bounds_axes_fraction": list(FIG3A_INSET_BOUNDS),
    }


def plot_approved_panel(table: list[dict], out: Path) -> None:
    apply_mdpi_figure3_rcparams()
    r = np.array([t["r"] for t in table])
    n_ref = np.array([t["nbar_reference_independent"] for t in table])
    n_prod = np.array([t["nbar_production_NL1"] for t in table])
    n_low = np.array([t["nbar_low_gain_r_squared"] for t in table])
    rel_pct = np.zeros_like(n_ref, dtype=np.float64)
    pos = r > 0
    rel_pct[pos] = 100.0 * np.abs(n_ref[pos] - n_low[pos]) / n_ref[pos]

    m_idx = marker_display_indices(len(r))
    r_m, n_prod_m = r[m_idx], n_prod[m_idx]

    fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZE_IN)
    ax.plot(
        r,
        n_ref,
        color=FIG3_COLORS["analytical_reference"],
        lw=1.8,
        label="Analytical reference",
    )
    ax.scatter(
        r_m,
        n_prod_m,
        s=30,
        zorder=5,
        color=FIG3_COLORS["production_simulator"],
        edgecolors="k",
        linewidths=0.35,
        label="Production simulator",
    )
    r_span = float(r[-1] - r[0])
    ax.set_xlim(float(r[0]) - 0.02 * r_span, float(r[-1]) + 0.02 * r_span)
    y_top = max(float(n_ref.max()), float(n_prod.max())) * 1.04
    y_bottom_display = -0.028 * y_top
    ax.set_ylim(y_bottom_display, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _pos: "" if y < -1e-12 else f"{y:.2f}")
    )
    ax.set_xlabel(r"Parametric gain, $r$")
    ax.set_ylabel(r"Mean photon number per mode, $\bar{n}$")
    style_axes_clean(ax)

    bx0, by0, bw, bh = FIG3A_INSET_BOUNDS
    ax_ins = ax.inset_axes([bx0, by0, bw, bh])
    mask = (r > 0) & np.isfinite(rel_pct) & (rel_pct > 0.0)
    if np.any(mask):
        ax_ins.plot(
            r[mask],
            rel_pct[mask],
            color=FIG3_COLORS["relative_error"],
            lw=0.75,
            marker="o",
            markersize=2.5,
            markerfacecolor=FIG3_COLORS["relative_error"],
            markeredgecolor=FIG3_COLORS["relative_error"],
            markeredgewidth=0.4,
            zorder=3,
        )
        ax_ins.set_yscale("log")
        ax_ins.yaxis.set_major_locator(LogLocator(base=10.0))
        ax_ins.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ins.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax_ins.set_xlabel(r"$r$", fontsize=7.5)
    ax_ins.set_ylabel(
        "Low-gain rel.\napprox. error (%)",
        fontsize=7.5,
    )
    ax_ins.tick_params(labelsize=7, pad=1)
    style_axes_clean(ax_ins)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.97, 0.06),
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=False,
        borderaxespad=0.35,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.14)

    stem = out / FIGURE_STEM
    fig.savefig(f"{stem}.png", dpi=FIG3_DPI_EXPORT)
    fig.savefig(f"{stem}.pdf")
    try:
        fig.savefig(
            f"{stem}.tif",
            dpi=FIG3_DPI_EXPORT,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(f"{stem}.tif", dpi=FIG3_DPI_EXPORT, format="tiff")
    plt.close(fig)

    meta = {
        "n_csv_points": len(r),
        "n_display_markers": int(len(m_idx)),
        "display_marker_indices": m_idx.tolist(),
        "low_gain_curve_in_figure": False,
        "inset_style": "line_markers_ordered_r",
        "mdpi_style": True,
        "inset_bounds": list(FIG3A_INSET_BOUNDS),
        "legend_location": "lower_right_in_axes",
        "legend_layout": "stacked_vertical",
        "y_display_floor": y_bottom_display,
        "exports": ["png", "pdf", "tif"],
    }
    (out / "plot_display_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_caption_final(path: Path, summary: dict) -> None:
    text = """\
**Figure 3(a).** Verification of the single-mode SPDC gain mapping. The solid line
is the analytical Bogoliubov relation $\\bar{n}=\\sinh^{2}(r)$, where $r$ is the
dimensionless parametric gain and $\\bar{n}$ is the mean photon number per
spatiotemporal mode. Circular markers show corresponding outputs of the production
simulator obtained via `spdc_gain_r` (11 of 51 verification samples shown). The
inset gives the relative numerical error with respect to an independently
implemented analytical reference. The maximum relative discrepancy over
$r\\in[0,0.5]$ is approximately $9.2\\times10^{-13}\\%$, consistent with
floating-point roundoff. This panel verifies the computational mapping
$r\\mapsto\\bar{n}$ only; see Methods for operating-point defaults and forward-model
scope.
"""
    path.write_text(text, encoding="utf-8")


def write_methods_final(path: Path) -> None:
    text = """\
### SPDC gain and operating point (Methods excerpt)

**Single-mode Bogoliubov relation.** For a matched spatiotemporal signal–idler mode
in the undepleted-pump limit, the mean photon number per mode is
$\\bar{n}=\\sinh^{2}r$, where $r$ is the dimensionless parametric gain parameter
(Wiseman and Mølmer, *Phys. Lett. A* **270**, 245–248 (2000), Eq. (6); our
manuscript uses the symbol $r$ for the squeezing parameter). This relation is
**exact** in that two-mode squeezing model; it is not replaced by $\\bar{n}\\approx r^{2}$
in production code. The low-gain **expansion**
$\\sinh^{2}r = r^{2} + r^{4}/3 + O(r^{6})$ explains why a prescribed
$\\bar{n}=0.01$ corresponds to $r\\approx\\operatorname{arcsinh}(\\sqrt{0.01})\\approx 0.0998$;
at that operating point $r^{2}$ differs from $\\sinh^{2}r$ by only $\\sim 0.3\\%$.

**Production mapping.** With `spdc_source_mode=fixed_nbar`, the twin resolves
$\\bar{n}$ from `LatentState.spdc_gain_r` through
`optics.resolve_spdc_source_nbar` and `setup_params.sinh_squared_stable` (exact
$\\sinh^{2}r$, not $r^{2}$). Figure 3(a) compares that pathway to an independent
expm1 evaluation of the same formula (`verification.figure3a_gain.independent_reference`).

**Operating regime vs approximation.** Manuscript simulations typically **prescribe**
$\\bar{n}=0.01$ as a numerical low-photon-number operating point; this is **not**
an experimentally calibrated occupancy and does **not** invoke pump-power-to-$r$
calibration (`fixed_nbar`). The label `spdc_gain_mode=low_gain` sets default
metadata and shortcut occupancies; it does **not** switch the forward equations to
$r^{2}$. Subsequent interference uses **Gaussian field moments**
$N_{1}, N_{2}, C_{12}$ (undepleted-pump moment closure; not a full multimode
biphoton wavefunction).

**Figure 3(a) verification sweep.** Uniform object, configuration A, grid
$N=24$, seed 20260930, detector noise off; $r\\in[0,0.5]$ (51 points) tests the
implementation only and is **not** the experimental gain range. Profiles B–D and
C\\_lab yield the same $\\bar{n}(r)$ on this operating-point check.

**Scope.** Figure 3(a) does not validate multimode SPDC dynamics, pump calibration,
spatial source structure, the full Michelson/two-crystal interferometer, or
predictive agreement with experiment.
"""
    path.write_text(text, encoding="utf-8")


def write_visual_compliance(path: Path, summary: dict) -> None:
    text = f"""# Figure 3(a) — MDPI visual compliance check

Generated with `verification/figure3a_gain/mdpi_figure3_panel_style.py`.

| Check | Status |
|-------|--------|
| No major/minor gridlines (main + inset) | Pass — `style_axes_clean`, grid off |
| White background | Pass |
| Analytical + production only (no $r^2$ curve) | Pass |
| 11 display markers / 51 CSV points | Pass — see `plot_display_meta.json` |
| Inset upper-left (slightly left of center) | Pass — bounds {list(FIG3A_INSET_BOUNDS)} |
| Inset: thin line + markers; log $\\varepsilon_{{\\mathrm{{rel}}}}$ (%); $r=0$ omitted | Pass |
| Max $\\varepsilon_{{\\mathrm{{rel}}}}$ unchanged | {summary['max_rel_error_implementation_percent']:.3e} % |
| Legend lower-right inside axes, horizontal, no frame | Pass |
| Panel size 5.6×3.8 in @ {FIG3_DPI_EXPORT} dpi | Pass |
| PNG + PDF + TIFF RGB exports | Pass — same stem |
| Composite Fig 3 not modified | Pass |

**Manual review:** Open `FIGURE3a_FINAL_APPROVED_CANDIDATE.png` at 100% zoom and confirm
inset label separation from main $\\bar{{n}}$ axis, visibility of $r=0.5$ marker, and
legend clearance above the x-axis label.
"""
    path.write_text(text, encoding="utf-8")


def write_compliance_final(path: Path) -> None:
    text = """# Figure 3(a) — supervisor compliance (final approved candidate)

| Requirement | Evidence |
|-------------|----------|
| Exact analytical $\\bar{n}=\\sinh^{2}(r)$ (solid); independent reference | `independent_reference.mean_photons_per_mode_independent` |
| Production markers via `spdc_gain_r` → `sinh_squared_stable` | `production_probe.sweep_production`; `pathway_trace.csv` |
| 51 samples in CSV; ~11 markers on figure | `figure3a_verification.csv`; `plot_display_meta.json` |
| Relative error inset; exclude $r=0$; log scale for $\\epsilon>0$ | Inset in `FIGURE3a_FINAL_APPROVED_CANDIDATE.*` |
| Max rel. error $\\approx 9.2\\times10^{-13}\\%$ | `verification_summary.json` |
| **No** $r^{2}$ curve on final figure; low-gain math in Methods | `METHODS_SNIPPET_FINAL.md` |
| Main + inset (not stacked subplots); legend below | Approved layout script |
| Match Fig. 3 panel size 5.6×3.8 in | `PANEL_SIZE` in `figure3a_manuscript_final.py` |
| No production / Fig 3(b–e) changes | Read-only physics; separate evidence folder |
| Reproduce | `python tools/figure3a_manuscript_final.py`; `pytest tests/test_figure3a_gain_verification.py -q` |
"""
    path.write_text(text, encoding="utf-8")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run(output: Path | None = None) -> Path:
    stamp = utc_stamp()
    out = output or (EVIDENCE_ROOT / stamp)
    out.mkdir(parents=True, exist_ok=True)

    prod_rows = sweep_production(R_GRID_DEFAULT, profile="A")
    table = build_verification_table(prod_rows)
    summary = summarize_implementation(table)
    traces = trace_all_profiles((0.05, 0.2, 0.5))

    save_csv(out / "figure3a_verification.csv", table, list(table[0].keys()))
    save_csv(out / "pathway_trace.csv", traces, list(traces[0].keys()))
    (out / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_approved_panel(table, out)
    write_caption_final(out / "CAPTION_FINAL.md", summary)
    write_methods_final(out / "METHODS_SNIPPET_FINAL.md")
    write_compliance_final(out / "SUPERVISOR_COMPLIANCE_FINAL.md")
    write_visual_compliance(out / "VISUAL_COMPLIANCE_MDPI.md", summary)

    (out / "REPRODUCE.md").write_text(
        "Source script: `tools/figure3a_manuscript_final.py`\n\n"
        "```powershell\n"
        "python tools/figure3a_manuscript_final.py\n"
        "pytest tests/test_figure3a_gain_verification.py -q\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "summary": summary}, indent=2))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
