#!/usr/bin/env python3
"""Standalone Figure 3(d) — phase-dependent mean signal intensity (raw scale).

Canonical configurations A–D; does not modify production physics or Figures 3(a–c).
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
from setup_params import resolve_setup_selection

FIG3D_RASTER_DPI = FIG3_DPI_EXPORT
FIGURE_STEM = "FIGURE3d_FINAL_CANDIDATE"
EVIDENCE_ROOT = ROOT / "evidence" / "figure3d_manuscript_candidate"
CANONICAL_PROFILES = ("A", "B", "C", "D")
N_PHASE = 361
# Dense uniform sweep including φ=2π for a closed, smooth publication curve.
PHASES = np.linspace(0.0, 2.0 * math.pi, N_PHASE, endpoint=True)
N_MARKERS = 12

# Match Figure 3(c) family colors; distinguish A/B/C/D when plotted.
COLORS = {"A": "#1f4e79", "B": "#2a9d8f", "C": "#264653", "D": "#e9c46a"}
MARKERS = {"A": "o", "B": "s", "C": "D", "D": "^"}
DASH = {"A": "-", "B": (0, (4, 2)), "C": "-", "D": (0, (4, 2))}

# Path-matched fringe reference (X11 convention); local override only.
SWEEP_SETUP_OVERRIDES = {"delta_L_mm": 0.0}
OBJECT_T = 1.0
LEGEND_FONT_SIZE = 6.5
VALIDATION_PHASES = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
    2.0 * math.pi,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_canonical_phase_point(
    profile: str,
    phi_rad: float,
    *,
    grid: int,
    nbar: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    setup = resolve_setup_selection(profile)
    for key, val in SWEEP_SETUP_OVERRIDES.items():
        setattr(setup, key, val)
    state = quiet_state(setup, nbar=nbar)
    ref = uniform_object(grid, T=OBJECT_T, phi=float(phi_rad))
    result = simulate_forward(ref, state, seed=seed, detector_noise=False)
    meta = result.meta or {}
    vo = vis_block(result)
    port = str(meta.get("selected_output_port") or "plus")
    I_mean = float(np.mean(result.ideal_signal_image))
    N1 = float(vo.get("N_1") or 0.0)
    N2 = float(vo.get("N_2") or 0.0)
    C12a = float(vo.get("C_12_abs") or 0.0)
    C12 = complex(C12a, 0.0)
    theta = float(meta.get("interference_phase_rad") or 0.0)
    V_ind = float(iref.visibility_from_moments(N1, N2, C12))
    return setup, {
        "configuration": profile,
        "setup_id": setup.setup_id,
        "phi_rad": float(phi_rad),
        "object_intensity_T": OBJECT_T,
        "object_phase_rad": float(phi_rad),
        "output_port": port,
        "I_raw_mean": I_mean,
        "V_independent_moments": V_ind,
        "N1": N1,
        "N2": N2,
        "C12_abs": C12a,
        "interference_phase_rad": theta,
        "V_moment": float(vo.get("V_50_50") or iref.visibility_from_moments(N1, N2, C12)),
        "detector_noise": False,
        "delta_L_mm": float(getattr(setup, "delta_L_mm", float("nan"))),
        "nbar": nbar,
        "object_type": getattr(setup, "object_type", None),
    }


def i_plus_from_moments(N1: float, N2: float, C12: complex, phi_rad: float) -> float:
    """Combiner plus-port intensity from fixed Gaussian moments (independent of object φ)."""
    return float((N1 + N2) / 2.0 + (C12 * complex(math.cos(phi_rad), math.sin(phi_rad))).real)


def check_moment_phase_invariance(
    profile: str,
    *,
    grid: int,
    nbar: float,
    seed: int,
) -> dict[str, Any]:
    """Confirm N1, N2, |C12| do not vary with uniform object phase in this test."""
    samples: list[dict[str, float]] = []
    for phi in VALIDATION_PHASES:
        _, row = run_canonical_phase_point(
            profile, float(phi), grid=grid, nbar=nbar, seed=seed
        )
        samples.append(
            {
                "phi_rad": float(phi),
                "N1": float(row["N1"]),
                "N2": float(row["N2"]),
                "C12_abs": float(row["C12_abs"]),
            }
        )
    n1 = [s["N1"] for s in samples]
    n2 = [s["N2"] for s in samples]
    c12 = [s["C12_abs"] for s in samples]
    return {
        "configuration": profile,
        "N1_span": float(max(n1) - min(n1)),
        "N2_span": float(max(n2) - min(n2)),
        "C12_abs_span": float(max(c12) - min(c12)),
        "phase_invariant": (
            max(n1) - min(n1) < 1e-15
            and max(n2) - min(n2) < 1e-15
            and max(c12) - min(c12) < 1e-15
        ),
    }


def build_harmonic_phase_rows(
    profile: str,
    *,
    grid: int,
    nbar: float,
    seed: int,
    n_phase: int = N_PHASE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Smooth I_+ curve from φ-invariant moments, scaled to mean(ideal_signal_image) units."""
    _, anchor = run_canonical_phase_point(profile, 0.0, grid=grid, nbar=nbar, seed=seed)
    N1 = float(anchor["N1"])
    N2 = float(anchor["N2"])
    C12 = complex(float(anchor["C12_abs"]), 0.0)
    i_moment_0 = i_plus_from_moments(N1, N2, C12, 0.0)
    i_fwd_0 = float(anchor["I_raw_mean"])
    scale = i_fwd_0 / i_moment_0 if abs(i_moment_0) > 0.0 else 1.0
    phases = np.linspace(0.0, 2.0 * math.pi, int(n_phase), endpoint=True)
    rows: list[dict[str, Any]] = []
    for phi in phases:
        i_m = i_plus_from_moments(N1, N2, C12, float(phi))
        rows.append(
            {
                "configuration": profile,
                "setup_id": anchor["setup_id"],
                "phi_rad": float(phi),
                "object_intensity_T": OBJECT_T,
                "object_phase_rad": float(phi),
                "output_port": anchor["output_port"],
                "I_raw_mean": float(scale * i_m),
                "I_combiner_moment": float(i_m),
                "spatial_mean_scale": float(scale),
                "N1": N1,
                "N2": N2,
                "C12_abs": float(C12.real),
                "V_moment": float(anchor["V_moment"]),
                "V_independent_moments": float(anchor["V_independent_moments"]),
                "curve_source": "harmonic_from_phase_invariant_moments",
            }
        )
    meta = {
        "configuration": profile,
        "spatial_mean_scale": scale,
        "N1": N1,
        "N2": N2,
        "C12_abs": float(C12.real),
        "I_fwd_at_phi0": i_fwd_0,
        "I_moment_at_phi0": i_moment_0,
    }
    return rows, meta


def validate_harmonic_vs_forward(
    profile: str,
    *,
    spatial_mean_scale: float,
    N1: float,
    N2: float,
    C12_abs: float,
    grid: int,
    nbar: float,
    seed: int,
) -> list[dict[str, Any]]:
    C12 = complex(float(C12_abs), 0.0)
    out: list[dict[str, Any]] = []
    for phi in VALIDATION_PHASES:
        _, row = run_canonical_phase_point(
            profile, float(phi), grid=grid, nbar=nbar, seed=seed
        )
        i_fwd = float(row["I_raw_mean"])
        i_harm = spatial_mean_scale * i_plus_from_moments(N1, N2, C12, float(phi))
        abs_err = abs(i_fwd - i_harm)
        rel_err = abs_err / max(abs(i_fwd), 1.0e-30)
        out.append(
            {
                "configuration": profile,
                "phi_rad": float(phi),
                "I_forward_mean": i_fwd,
                "I_harmonic_reconstruction": i_harm,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )
    return out


def build_publication_phase_rows(
    *,
    grid: int = DEFAULT_GRID,
    nbar: float = DEFAULT_NBAR,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Publication Figure 3(d): harmonic curves + forward validation at five phases."""
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"moment_invariance": {}, "harmonic_meta": {}, "validation": []}
    for profile in CANONICAL_PROFILES:
        audit["moment_invariance"][profile] = check_moment_phase_invariance(
            profile, grid=grid, nbar=nbar, seed=seed
        )
        prows, meta = build_harmonic_phase_rows(
            profile, grid=grid, nbar=nbar, seed=seed
        )
        rows.extend(prows)
        audit["harmonic_meta"][profile] = meta
        audit["validation"].extend(
            validate_harmonic_vs_forward(
                profile,
                spatial_mean_scale=float(meta["spatial_mean_scale"]),
                N1=float(meta["N1"]),
                N2=float(meta["N2"]),
                C12_abs=float(meta["C12_abs"]),
                grid=grid,
                nbar=nbar,
                seed=seed,
            )
        )
    vals = audit["validation"]
    audit["max_abs_error"] = max(v["abs_error"] for v in vals)
    finite_rel = [
        v["rel_error"]
        for v in vals
        if float(v["I_forward_mean"]) > 1.0e-9
    ]
    audit["max_rel_error"] = max(finite_rel) if finite_rel else 0.0
    audit["validation_pass"] = audit["max_abs_error"] < 1.0e-5
    return rows, audit


def sweep_profiles(*, grid: int, nbar: float, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in CANONICAL_PROFILES:
        for phi in PHASES:
            _, row = run_canonical_phase_point(profile, float(phi), grid=grid, nbar=nbar, seed=seed)
            rows.append(row)
    return rows


def intensity_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in CANONICAL_PROFILES:
        ser = sorted(
            [r for r in rows if r["configuration"] == profile],
            key=lambda r: r["phi_rad"],
        )
        phi = np.array([r["phi_rad"] for r in ser])
        I = np.array([r["I_raw_mean"] for r in ser])
        imax, imin = float(np.max(I)), float(np.min(I))
        iphi_max = float(phi[int(np.argmax(I))])
        iphi_min = float(phi[int(np.argmin(I))])
        denom = imax + imin
        V_ext = (imax - imin) / denom if denom > 0 else float("nan")
        I_dc_ext = 0.5 * (imax + imin)
        A_f = 0.5 * (imax - imin)
        r0 = min(ser, key=lambda r: abs(r["phi_rad"]))
        out.append(
            {
                "configuration": profile,
                "setup_id": ser[0]["setup_id"],
                "I_DC_extrema": I_dc_ext,
                "A_fringe_extrema": A_f,
                "V_extrema_scan": V_ext,
                "I_max": imax,
                "I_min": imin,
                "phi_at_I_max_rad": iphi_max,
                "phi_at_I_min_rad": iphi_min,
                "I_mean_over_period": float(np.mean(I)),
                "N1_at_phi0": r0["N1"],
                "N2_at_phi0": r0["N2"],
                "C12_abs_at_phi0": r0["C12_abs"],
                "I_DC_moment_phi0": r0["N1"] + r0["N2"],
                "A_fringe_moment_phi0": 2.0 * r0["C12_abs"],
                "V_moment_phi0": r0["V_moment"],
                "output_port": ser[0]["output_port"],
                "intensity_units": "simulation_units_mean_ideal_signal_image",
            }
        )
    return out


def moment_verification(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Moment identity V=2|C12|/(N1+N2) — not spatial-mean combiner equality."""
    out: list[dict[str, Any]] = []
    for r in rows:
        err = abs(r["V_moment"] - r["V_independent_moments"])
        out.append(
            {
                "configuration": r["configuration"],
                "phi_rad": r["phi_rad"],
                "V_production": r["V_moment"],
                "V_independent_moments": r["V_independent_moments"],
                "abs_error_V": err,
                "I_raw_mean": r["I_raw_mean"],
                "N1": r["N1"],
                "N2": r["N2"],
                "C12_abs": r["C12_abs"],
                "interference_phase_rad": r["interference_phase_rad"],
            }
        )
    return out


def visibility_crosscheck(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fig3c = ROOT / "evidence" / "figure3c_manuscript_candidate" / "20261019T140435Z"
    path = fig3c / "figure3c_production_visibility.csv"
    fig3c_v: dict[str, float] = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if abs(float(row["intensity_T"]) - 1.0) < 1e-12:
                    fig3c_v[row["configuration"]] = float(row["V_production"])
    out = []
    for m in metrics:
        p = m["configuration"]
        v3c = fig3c_v.get(p)
        v3d = m["V_moment_phi0"]
        out.append(
            {
                "configuration": p,
                "V_figure3c_T1_moment": v3c,
                "V_figure3d_phi0_moment": v3d,
                "V_figure3d_extrema_scan": m["V_extrema_scan"],
                "abs_err_3d_moment_vs_3c": (
                    abs(v3d - v3c) if v3c is not None else float("nan")
                ),
                "matched_conditions_note": (
                    "T=1, object phi=0, nbar=0.01, delta_L_mm=0 override, plus port, "
                    "detector_noise=False; figure3c had no delta_L override"
                ),
            }
        )
    return out


def max_curve_separation(rows: list[dict[str, Any]], p1: str, p2: str) -> float:
    a = [r["I_raw_mean"] for r in rows if r["configuration"] == p1]
    b = [r["I_raw_mean"] for r in rows if r["configuration"] == p2]
    return float(np.max(np.abs(np.array(a) - np.array(b))))


def marker_indices(n: int, count: int, phase: int = 0) -> np.ndarray:
    idx = np.unique(np.round(np.linspace(0, n - 1, count)).astype(int))
    if phase:
        idx = np.unique((idx + phase) % n)
    return idx


def draw_figure(rows: list[dict[str, Any]], *, y_top: float) -> plt.Figure:
    apply_mdpi_figure3_rcparams()
    fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZE_IN)
    handles: list[Line2D] = []
    for profile in CANONICAL_PROFILES:
        ser = sorted(
            [r for r in rows if r["configuration"] == profile],
            key=lambda r: r["phi_rad"],
        )
        phi = np.array([r["phi_rad"] for r in ser])
        I = np.array([r["I_raw_mean"] for r in ser])
        markevery = max(1, int(round(len(phi) / float(N_MARKERS))))
        mfc = "white" if profile in ("A", "C") else COLORS[profile]
        (line,) = ax.plot(
            phi,
            I,
            color=COLORS[profile],
            ls=DASH[profile],
            lw=1.8,
            marker=MARKERS[profile],
            markevery=markevery,
            ms=5.5,
            mfc=mfc,
            mec=COLORS[profile],
            mew=0.9,
            solid_capstyle="round",
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[profile],
                lw=1.8,
                ls=DASH[profile],
                marker=MARKERS[profile],
                markersize=5.5,
                markerfacecolor=mfc,
                markeredgecolor=COLORS[profile],
                markeredgewidth=0.9,
                label=f"Configuration {profile}",
            )
        )
        line.set_label("_nolegend_")
    ax.set_xlabel(r"Relative phase, $\phi$ (rad)")
    ax.set_ylabel(r"Mean ideal signal intensity, $I_+$ (simulation units)")
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_ylim(0.0, y_top)
    style_axes_clean(ax)
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    return fig


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=FIG3D_RASTER_DPI)
    fig.savefig(f"{stem}.pdf")
    try:
        fig.savefig(
            f"{stem}.tif",
            dpi=FIG3D_RASTER_DPI,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(f"{stem}.tif", dpi=FIG3D_RASTER_DPI, format="tiff")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_audit_markdown(out: Path, sep: dict[str, float]) -> None:
    (out / "FIGURE3D_SOURCE_TRACE.md").write_text(
        """# Figure 3(d) source trace

## Historical pipeline

| Stage | Location | Role |
|-------|----------|------|
| Original dense sweep | `tools/taskX5_phase_sweep.py` | 65-point sweep; **I_raw/mean(I)** normalization |
| Path-matched sweep | `tools/taskX11_Figure3d_pathmatched.py` | `delta_L_mm=0`; object **φ** sweep at **T=1** |
| Final CSV lineage | `evidence/taskX12_final_figure_corrections/data/Figure3d_FINAL_source_data.csv` | Used by X8 / Y5 |
| Composite plot | `tools/qiup_figure3_final_article.py:draw_panel_d_simulation_article` | **Peak/min display map** `I_paper = 1 - I_display` or min–max |

## Old normalization (supervisor concern)

- Per-configuration **mean normalization** `I / mean(I)` (X5).
- Publication panel additionally **min–max or 1−I** display (X12 / qiup article).
- **Independent peak normalization per curve** — hides DC and fringe-contrast differences.

## This candidate

- **New production sweep** (201 φ points, canonical A–D aliases).
- **Raw** `mean(ideal_signal_image)` on selected **plus** port (`detector_noise=False`).
- **No** per-curve peak or mean normalization in the plot.
- Local **ΔL₀=0** override (X11 path-matched convention); Figure 3(c) did not apply this override.
- Swept parameter: uniform **object phase** `ref['phi']` at **T=1** (legacy Figure 3(d) convention).
""",
        encoding="utf-8",
    )
    (out / "FIGURE3D_PHASE_CONVENTION.md").write_text(
        """# Phase convention (Figure 3(d))

- **Swept parameter φ:** encoded in `uniform_object(..., phi=φ)` → object phase map (rad).
- **Object throughput:** fixed **T = 1** (transparent / unit reflectance for generic C/D).
- **Interferometer:** production `interference_phase_rad` in forward model; static OPD phase with **delta_L_mm=0** override for this panel.
- **Output port:** production default **plus** (`ideal_signal_image` = mean over grid of **I_plus**).
- **Observed fringe (plus port, this sweep):** **maximum near φ≈0** and **minimum near φ≈π** on raw ⟨I⟩ (see metrics CSV). Legacy publication display applied `1−I` / min–max maps that inverted the vertical axis relative to raw output.
- **No independent per-configuration phase alignment** applied for plotting.
""",
        encoding="utf-8",
    )
    (out / "FIGURE3D_INTENSITY_SCALE_AUDIT.md").write_text(
        f"""# Intensity scale audit

- **Observable:** spatial mean of `SimResult.ideal_signal_image` (plus port, optical forward, no detector noise).
- **Units:** simulation flux units (same scale across A–D at fixed nbar={DEFAULT_NBAR}).
- **No** independent normalization in the figure; y-axis shows raw ⟨I⟩.
- **Cross-family separation** max|I_A−I_C| ≈ {sep.get('A_C', float('nan')):.3e} (simulation units).
- **Within-family:** max|I_A−I_B| ≈ {sep.get('A_B', float('nan')):.3e}, max|I_C−I_D| ≈ {sep.get('C_D', float('nan')):.3e}.
""",
        encoding="utf-8",
    )


def write_caption(path: Path) -> None:
    path.write_text(
        """**Figure 3(d).** Mean detected **plus-port** signal intensity ⟨I⟩ (simulation units, \
**not** peak-normalized) versus relative object phase φ on a uniform object at **T=1**, \
for generic configurations **A–D** (common **3 nm** signal filter, uniform final camera \
collection **E=1**, n̄≈0.01, `detector_noise=False`). φ enters the production model as the \
spatially uniform object phase map; **ΔL₀=0** mm is applied locally for path-matched fringes \
(Figure 3(c) used the default path mismatch without this override). Curves show DC level and \
fringe contrast; **A≈C** and **B≈D** at matched filter bandwidth with a residual far-field/near-field \
offset from retained spatial-moment and geometry conventions (not per-curve normalization). \
Visibility and moment diagnostics are reported on the original intensity scale in the Methods table.
""",
        encoding="utf-8",
    )


def write_methods(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "### Phase fringe panel (Figure 3(d); Methods excerpt)",
        "",
        "**Intensity.** ⟨I⟩ is the spatial mean of the ideal **I_plus** image before detector noise.",
        "",
        "**Generic benchmark.** Configurations A–D use a common **3 nm** detected-signal filter and "
        "uniform additional camera collection (**E=1**); intrinsic SPDC spatial marginals and propagation are unchanged.",
        "",
        "**Visibility from extrema:** V = (I_max−I_min)/(I_max+I_min) on the raw phase scan;",
        "compare to moment identity V = 2|C₁₂|/(N₁+N₂) at φ=0.",
        "",
        "| Config | I_DC (extrema) | A_fringe | V (scan) | I_max | I_min | φ_max | φ_min | V(φ=0) moment |",
        "|--------|----------------|----------|----------|-------|-------|-------|-------|---------------|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['configuration']} | {m['I_DC_extrema']:.6g} | {m['A_fringe_extrema']:.6g} | "
            f"{m['V_extrema_scan']:.4f} | {m['I_max']:.6g} | {m['I_min']:.6g} | "
            f"{m['phi_at_I_max_rad']:.3f} | {m['phi_at_I_min_rad']:.3f} | {m['V_moment_phi0']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_supervisor_compliance(metrics: list[dict[str, Any]], cross: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Supervisor compliance — Figure 3(d)",
            "",
            "- **Output port:** plus (`ideal_signal_image`).",
            "- **Raw scale:** mean and extrema on unnormalized ⟨I⟩; no per-curve I/I_max.",
            "- **Settings vs 3(c):** same T=1 object throughput and nbar; **delta_L_mm=0** here only.",
            "- **Visibility table:** see `figure3d_intensity_metrics.csv`.",
            "",
            "## Figure 3(c) cross-check (φ=0 moment V)",
            "",
        ]
        + [f"- {r['configuration']}: |V_3d−V_3c|={r['abs_err_3d_moment_vs_3c']}" for r in cross]
        + ["", f"Note: {cross[0]['matched_conditions_note'] if cross else ''}"]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or (EVIDENCE_ROOT / utc_stamp())
    out.mkdir(parents=True, exist_ok=True)

    rows = sweep_profiles(grid=DEFAULT_GRID, nbar=DEFAULT_NBAR, seed=DEFAULT_SEED)
    for r in rows:
        if str(r.get("output_port")) != "plus":
            raise RuntimeError(
                f"Figure 3(d) requires plus-port I_+; got {r.get('output_port')!r} "
                f"for configuration {r.get('configuration')}"
            )
    metrics = intensity_metrics(rows)
    mom_ver = moment_verification(rows)
    cross = visibility_crosscheck(metrics)

    sep = {
        "A_B": max_curve_separation(rows, "A", "B"),
        "C_D": max_curve_separation(rows, "C", "D"),
        "A_C": max_curve_separation(rows, "A", "C"),
    }

    y_max = max(r["I_raw_mean"] for r in rows)
    y_top = y_max * 1.08

    fig = draw_figure(rows, y_top=y_top)
    export_figure(fig, out / FIGURE_STEM)
    plt.close(fig)

    write_csv(out / "figure3d_raw_phase_sweep.csv", rows)
    write_csv(out / "figure3d_intensity_metrics.csv", metrics)
    write_csv(out / "figure3d_moment_verification.csv", mom_ver)
    write_csv(out / "figure3d_visibility_crosscheck.csv", cross)

    runtime = {
        "nbar": DEFAULT_NBAR,
        "grid": DEFAULT_GRID,
        "seed": DEFAULT_SEED,
        "n_phase": N_PHASE,
        "object_T": OBJECT_T,
        "setup_overrides": SWEEP_SETUP_OVERRIDES,
        "canonical_profiles": list(CANONICAL_PROFILES),
        "curve_separation": sep,
        "peak_normalize_in_plot": False,
    }
    (out / "effective_runtime_parameters.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    write_audit_markdown(out, sep)
    write_caption(out / "FIGURE3d_CAPTION_CANDIDATE.md")
    write_methods(out / "FIGURE3d_METHODS_EXCERPT.md", metrics)
    (out / "FIGURE3d_SUPERVISOR_COMPLIANCE.md").write_text(
        write_supervisor_compliance(metrics, cross), encoding="utf-8"
    )
    (out / "LAYOUT_QC_REPORT.md").write_text(
        "\n".join(
            [
                "# Layout QC",
                "",
                "- Raw-scale y-axis; legend lower-right; four configuration entries.",
                f"- No grid; 5.6×3.8 in; {FIG3D_RASTER_DPI} dpi PNG/TIFF; vector PDF.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "REPRODUCE.md").write_text(
        f'cd "{ROOT}"\npython tools/figure3d_manuscript_candidate.py\n',
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
