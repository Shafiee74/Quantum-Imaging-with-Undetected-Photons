#!/usr/bin/env python3
"""Task X8 — rebuild final publication Figure 3 from corrected X2–X7 results.

Panels:
  (a) SPDC gain  (X3/U02)
  (b) Phase matching  (X3/U03)
  (c) Visibility vs T  (X4)
  (d) Phase fringe + fit  (X5)
  (e) Numerical convergence  (X7)

U01 retained as supplementary/table only. No U08/U09 in Figure 3.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
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

from provenance import get_environment, hash_source_modules

OUT = ROOT / "evidence" / "taskX8_final_Figure3"
X3 = ROOT / "evidence" / "taskX3_U01_U03_rerun"
X4 = ROOT / "evidence" / "taskX4_transmission_sweep"
X5 = ROOT / "evidence" / "taskX5_phase_sweep"
X11 = ROOT / "evidence" / "taskX11_Figure3d_pathmatched"
X12 = ROOT / "evidence" / "taskX12_final_figure_corrections"
X7 = ROOT / "evidence" / "taskX7_U14_convergence"

COLORS = {"A": "#e76f51", "B": "#2a9d8f", "C": "#264653", "D": "#e9c46a"}
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
LINESTYLES = {"A": "-", "B": "--", "C": "-.", "D": ":"}
PROFILES = ("A", "B", "C", "D")
PANEL_B_X_LIMIT = 1.5
PANEL_B_ZERO_ORDERS = (-1, 1)
PANEL_B_ENVELOPE_POINTS = 4001
PANEL_B_PHASE_MATCH_TEMP_C = 75.6


def panel_d_paths() -> tuple[Path, str, str, str]:
    """Return (data_dir, sweep_csv, fit_csv, lineage_label). Prefer X12 FINAL, then X11."""
    if (X12 / "data" / "Figure3d_FINAL_source_data.csv").is_file():
        return (
            X12,
            "Figure3d_FINAL_source_data.csv",
            "Figure3d_FINAL_fit_results.csv",
            "evidence/taskX12_final_figure_corrections/data",
        )
    if (X11 / "data" / "Figure3d_phase_sweep.csv").is_file():
        return (
            X11,
            "Figure3d_phase_sweep.csv",
            "Figure3d_fit_results.csv",
            "evidence/taskX11_Figure3d_pathmatched/data",
        )
    return (
        X5,
        "phase_sweep_ABCD.csv",
        "phase_fit_summary.csv",
        "evidence/taskX5_phase_sweep/data",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def export_figure(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> None:
    """Write PNG (high-res), PDF, and SVG for a figure stem."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def _panel_d_source_files() -> list[Path]:
    ddir, sweep, fit, _ = panel_d_paths()
    files = [ddir / "data" / sweep, ddir / "data" / fit]
    for extra in ("Figure3d_phase_sweep.json", "Figure3d_fit_results.json"):
        p = ddir / "data" / extra
        if p.is_file():
            files.append(p)
    return files


def copy_sources() -> dict[str, list[str]]:
    """Copy exact panel source tables into Figure_3_source_data/."""
    src_root = OUT / "Figure_3_source_data"
    mapping = {
        "panel_a_gain": [
            X3 / "data" / "U02" / "U02_independent_gain_curves.csv",
            X3 / "data" / "U02" / "U02_production_nbar_probes.csv",
            X3 / "data" / "U02" / "U02_production_nbar_probes.json",
        ],
        "panel_b_phase_matching": [
            X3 / "data" / "U03" / "U03_independent_sinc2_envelope.csv",
            X3 / "data" / "U03" / "U03_delta_k_comparison.csv",
            X3 / "data" / "U03" / "U03_delta_k_comparison.json",
        ],
        "panel_c_transmission": [
            X4 / "data" / "transmission_sweep_ABCD.csv",
            X4 / "data" / "transmission_sweep_ABCD.json",
            X4 / "data" / "endpoint_verification.csv",
            X4 / "data" / "endpoint_verification.json",
        ],
        "panel_d_phase_fringe": _panel_d_source_files(),
        "panel_e_convergence": [
            X7 / "data" / "per_grid_metrics.csv",
            X7 / "data" / "successive_grid_changes.csv",
            X7 / "data" / "asymptotic_assessment.json",
            X7 / "data" / "vs_finest_secondary.csv",
        ],
        "supplementary_U01": [
            X3 / "data" / "U01" / "U01_energy_residuals.csv",
            X3 / "data" / "U01" / "U01_energy_residuals.json",
            X3 / "data" / "wavelength_authoritative_snapshot.csv",
        ],
    }
    copied: dict[str, list[str]] = {}
    for panel, paths in mapping.items():
        dest = src_root / panel
        dest.mkdir(parents=True, exist_ok=True)
        copied[panel] = []
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"Missing corrected source: {p}")
            target = dest / p.name
            shutil.copy2(p, target)
            copied[panel].append(str(target.relative_to(OUT)).replace("\\", "/"))
            # provenance sidecar
            (dest / f"{p.name}.sha256").write_text(sha256(p) + "\n", encoding="utf-8")
    save_json(src_root / "source_manifest.json", {
        "task": "taskX8_final_Figure3",
        "generated_utc": utc_now(),
        "policy": "DO NOT reuse obsolete pre-X2 numerical results",
        "panels": copied,
        "origins": {
            "panel_a": "evidence/taskX3_U01_U03_rerun/data/U02",
            "panel_b": "evidence/taskX3_U01_U03_rerun/data/U03",
            "panel_c": "evidence/taskX4_transmission_sweep/data",
            "panel_d": panel_d_paths()[3],
            "panel_e": "evidence/taskX7_U14_convergence/data",
            "supplementary_U01": "evidence/taskX3_U01_U03_rerun/data/U01",
        },
    })
    return copied


def style_ax(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.28)
    ax.tick_params(labelsize=8)


def draw_panel_a(ax: plt.Axes) -> None:
    curves = read_csv(X3 / "data" / "U02" / "U02_independent_gain_curves.csv")
    probes = read_csv(X3 / "data" / "U02" / "U02_production_nbar_probes.csv")
    r = np.array([float(r["r"]) for r in curves], dtype=np.float64)
    n_sinh = np.array([float(r["n_sinh2"]) for r in curves], dtype=np.float64)
    n_r2 = np.array([float(r["n_r2"]) for r in curves], dtype=np.float64)
    ax.plot(
        r,
        n_sinh,
        color="#1f4e79",
        lw=2.0,
        label=r"Exact SPDC relation, $\bar{n}=\sinh^{2}(r)$",
    )
    ax.plot(
        r,
        n_r2,
        color="#c45c26",
        ls="--",
        lw=1.6,
        label=r"Low-gain approximation, $\bar{n}\approx r^{2}$",
    )
    unique: dict[float, float] = {}
    for row in probes:
        unique[float(row["r"])] = float(row["nbar_production_meta"])
    ax.scatter(
        list(unique.keys()),
        list(unique.values()),
        s=38,
        zorder=5,
        color="#2a9d8f",
        edgecolors="k",
        linewidths=0.4,
        label="Numerical implementation",
    )
    ax.set_xlabel(r"Parametric gain parameter, $r$")
    ax.set_ylabel(r"Mean photon number per mode, $\bar{n}$")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    style_ax(ax)


def panel_b_complete_envelope() -> tuple[np.ndarray, np.ndarray]:
    """Symmetric normalized sinc^2 envelope for Figure 3(b)."""
    x = np.linspace(-PANEL_B_X_LIMIT, PANEL_B_X_LIMIT, PANEL_B_ENVELOPE_POINTS)
    response = np.sinc(x) ** 2
    return x, response


def annotate_panel_b_public_temperature(ax: plt.Axes) -> dict[str, Any]:
    """Mark Δk=0 and report the public phase-matching temperature annotation."""
    ax.axvline(0.0, color="#4f5b62", ls=":", lw=1.1, alpha=0.95)
    return {
        "phase_match_temperature_C": float(PANEL_B_PHASE_MATCH_TEMP_C),
        "zero_orders": list(PANEL_B_ZERO_ORDERS),
        "display_x_limits": [-float(PANEL_B_X_LIMIT), float(PANEL_B_X_LIMIT)],
    }


def draw_panel_b(ax: plt.Axes) -> None:
    ops = read_csv(X3 / "data" / "U03" / "U03_delta_k_comparison.csv")
    x, response = panel_b_complete_envelope()
    ax.plot(
        x,
        response,
        color="#1f4e79",
        lw=2.0,
        label="Analytical phase-matching envelope",
    )
    # Deduplicate operating points that coincide (B/C, A/D)
    seen: dict[float, list[str]] = {}
    for row in ops:
        xv = float(row["normalized_mismatch_dkl_over_2pi"])
        seen.setdefault(round(xv, 10), []).append(str(row["profile"]))
    for xv_key, profiles in seen.items():
        xv = float(xv_key)
        # pick a representative color (first profile)
        p0 = profiles[0]
        label = "Calculated operating point " + "/".join(profiles)
        ax.axvline(
            xv,
            ls="--",
            lw=1.25,
            alpha=0.9,
            color=COLORS.get(p0, "gray"),
            label=label,
        )
        s_op = float(np.interp(xv, x, response))
        ax.plot(xv, s_op, MARKERS.get(p0, "o"), color=COLORS.get(p0, "gray"), ms=6)
    ax.set_xlim(-PANEL_B_X_LIMIT, PANEL_B_X_LIMIT)
    ax.set_xlabel(r"Normalized phase mismatch, $\Delta k L/(2\pi)$")
    ax.set_ylabel("Normalized phase-matching response")
    ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    style_ax(ax)


def draw_panel_c(ax: plt.Axes) -> None:
    rows = read_csv(X4 / "data" / "transmission_sweep_ABCD.csv")
    for profile in PROFILES:
        pts = [r for r in rows if r["profile"] == profile]
        T = np.array([float(r["T"]) for r in pts], dtype=np.float64)
        V = np.array([float(r["V_production"]) for r in pts], dtype=np.float64)
        ax.plot(
            T,
            V,
            MARKERS[profile] + "-",
            color=COLORS[profile],
            lw=1.5,
            ms=4.5,
            label=f"Configuration {profile}",
        )
    ax.set_xlabel(r"Object amplitude transmission, $T$")
    ax.set_ylabel(r"Visibility, $V$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)
    style_ax(ax)


def draw_panel_d(ax: plt.Axes) -> None:
    ddir, sweep_name, fit_name, _ = panel_d_paths()
    pts = read_csv(ddir / "data" / sweep_name)
    fits = {r["profile"]: r for r in read_csv(ddir / "data" / fit_name)}
    use_display = "I_display_0_1" in pts[0]
    use_x11ish = ddir in (X11, X12) or "normalized_mean_signal_intensity" in pts[0]
    for profile in PROFILES:
        series = [r for r in pts if r["profile"] == profile]
        phi = np.array([float(r["phi_rad"]) for r in series], dtype=np.float64)
        if use_display:
            I = np.array([float(r["I_display_0_1"]) for r in series], dtype=np.float64)
            If = np.array(
                [float(r["I_fit_display_0_1"]) for r in series], dtype=np.float64
            )
        elif use_x11ish and "normalized_mean_signal_intensity" in series[0]:
            I = np.array(
                [float(r["normalized_mean_signal_intensity"]) for r in series],
                dtype=np.float64,
            )
            If = np.array(
                [float(r.get("I_fit_normalized") or r.get("I_fit") or 0) for r in series],
                dtype=np.float64,
            )
        else:
            I = np.array([float(r["I_normalized"]) for r in series], dtype=np.float64)
            If = np.array([float(r["I_fit_normalized"]) for r in series], dtype=np.float64)
        Vf = float(fits[profile].get("V_fit") or fits[profile].get("V_fitted"))
        ax.plot(
            phi,
            I,
            MARKERS[profile],
            color=COLORS[profile],
            ms=3.2,
            alpha=0.75,
            linestyle="none",
            label=f"{profile}: sim.",
        )
        ax.plot(
            phi,
            If,
            LINESTYLES[profile],
            color=COLORS[profile],
            lw=1.4,
            label=f"{profile}: fit ($V$={Vf:.3f})",
        )
    xlab = r"Relative phase, $\varphi$ (rad)"
    ax.set_xlabel(xlab)
    ax.set_ylabel("Normalized mean signal intensity")
    if use_display:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    style_ax(ax)


def draw_panel_e(ax: plt.Axes) -> dict[str, Any]:
    per = read_csv(X7 / "data" / "per_grid_metrics.csv")
    succ = read_csv(X7 / "data" / "successive_grid_changes.csv")
    asym = json.loads(
        (X7 / "data" / "asymptotic_assessment.json").read_text(encoding="utf-8")
    )
    Ns = [int(r["N"]) for r in per]
    psf = [float(r["psf_fwhm_um"]) for r in per]
    mid_N = [0.5 * (int(r["N_from"]) + int(r["N_to"])) for r in succ]
    inten = [float(r["intensity_map_l2"]) for r in succ]

    ax.plot(
        mid_N,
        inten,
        "o-",
        color="#1f4e79",
        lw=1.8,
        ms=6,
        label=r"Successive intensity-map $L_2$ error",
    )
    ax.set_xlabel(r"Grid size, $N$")
    ax.set_ylabel(r"Normalized error")
    ax.set_xticks(Ns)
    style_ax(ax)

    ax2 = ax.twinx()
    ax2.plot(
        Ns,
        psf,
        "s--",
        color="#e76f51",
        lw=1.6,
        ms=6,
        label=r"PSF FWHM",
    )
    ax2.set_ylabel(r"PSF FWHM ($\mathrm{\mu m}$)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51", labelsize=8)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper right")

    note = (
        "Asymptotic convergence not claimed"
        if not asym["asymptotic_convergence_claimed"]
        else "Asymptotic convergence supported"
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        color="#444444",
    )
    return asym


def build_standalone_panels(asym: dict[str, Any]) -> None:
    panel_dir = OUT / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "a": ("panel_a_SPDC_gain", draw_panel_a, (5.6, 3.8)),
        "b": ("panel_b_phase_matching", draw_panel_b, (5.6, 3.8)),
        "c": ("panel_c_visibility_vs_T", draw_panel_c, (5.6, 3.8)),
        "d": ("panel_d_phase_fringe", draw_panel_d, (5.6, 3.8)),
    }
    for letter, (name, fn, figsize) in builders.items():
        fig, ax = plt.subplots(figsize=figsize)
        fn(ax)
        ax.text(
            -0.12,
            1.05,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        fig.tight_layout()
        export_figure(fig, panel_dir / name)
        plt.close(fig)

    # panel e needs twin axis
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    draw_panel_e(ax)
    ax.text(
        -0.10,
        1.05,
        "(e)",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    fig.tight_layout()
    export_figure(fig, panel_dir / "panel_e_numerical_convergence")
    plt.close(fig)


def build_figure3() -> dict[str, Any]:
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

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)
    asym = draw_panel_e(ax_e)

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

    pub = OUT / "publication"
    pub.mkdir(parents=True, exist_ok=True)
    export_figure(fig, pub / "Figure_3_complete", dpi=300)
    # also under figures/
    export_figure(fig, OUT / "figures" / "Figure_3_complete", dpi=300)
    plt.close(fig)
    return asym


def build_u01_supplementary() -> None:
    rows = read_csv(X3 / "data" / "U01" / "U01_energy_residuals.csv")
    supp = OUT / "supplementary" / "U01"
    supp.mkdir(parents=True, exist_ok=True)
    # Publication-facing table
    table = []
    for r in rows:
        eps = abs(float(r["production_residual"]))
        table.append(
            {
                "Configuration": r["profile"],
                "lambda_p_nm": r["lambda_p_nm"],
                "lambda_s_nm": r["lambda_s_nm"],
                "lambda_i_nm": r["lambda_i_nm"],
                "relative_inverse_wavelength_residual_abs_epsilon": f"{eps:.6e}",
                "predeclared_tolerance": "1.0e-4",
                "verdict": r["verdict_production_residual"],
            }
        )
    save_csv(supp / "Table_S_wavelength_energy_consistency.csv", table)
    save_json(supp / "Table_S_wavelength_energy_consistency.json", table)

    # Compact bar figure (×10^5 scale)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = np.arange(len(rows))
    vals = [abs(float(r["production_residual"])) * 1.0e5 for r in rows]
    ax.bar(xs, vals, color="#457b9d", width=0.55)
    ax.axhline(10.0, color="#e76f51", ls="--", lw=1.4, label=r"Tolerance ($10^{-4}\times10^{5}$)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["profile"] for r in rows])
    ax.set_xlabel("Configuration")
    ax.set_ylabel(r"Relative inverse-wavelength residual, $|\varepsilon|\times10^{5}$")
    ax.legend(fontsize=7, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    export_figure(fig, supp / "Figure_S_wavelength_energy_consistency")
    plt.close(fig)

    note = (
        "Supplementary wavelength/energy consistency (U01 verification).\n"
        "Nominal wavelengths lambda_p=532 nm, lambda_s=810 nm, lambda_i=1550 nm.\n"
        f"|epsilon| = {abs(float(rows[0]['production_residual'])):.6e} "
        "(PASS vs 1e-4). Small nonzero residual from nominal wavelength rounding.\n"
        "Not part of main Figure 3; retained as verification evidence.\n"
    )
    (supp / "README.txt").write_text(note, encoding="utf-8")


def write_captions(asym: dict[str, Any]) -> None:
    claim = (
        "Asymptotic numerical convergence is not claimed: successive PSF/FWHM "
        "changes remain material at the finest grids (X7 assessment)."
        if not asym["asymptotic_convergence_claimed"]
        else "Finest-grid successive changes support an asymptotic claim (X7)."
    )
    _, _, _, panel_d_lineage = panel_d_paths()
    if panel_d_lineage.startswith("evidence/taskX12"):
        d_panel = (
            "(d) Phase-dependent interference response under an ideal path-matched "
            "reference condition (ΔL₀ = 0). The normalized mean signal intensity is "
            "shown over one 2π cycle with display normalization "
            "I_norm=(I−I_min)/(I_max−I_min) so the plotted fringe spans 0…1; "
            "reported Visibility is taken from the physical simulator fringe "
            "(fitted modulation depth), not from the display transform. "
            "Simulation points are compared with sinusoidal fits. The zero-path-mismatch "
            "condition is used only for this controlled physics-verification test and "
            "does not represent an assumed experimental path mismatch for Configuration C."
        )
    elif panel_d_lineage.startswith("evidence/taskX11"):
        d_panel = (
            "(d) Phase-dependent interference response under an ideal path-matched "
            "reference condition (ΔL₀ = 0). The normalized mean signal intensity is "
            "shown as a function of the applied relative phase over one 2π cycle. "
            "Simulation points are compared with sinusoidal fits. The zero-path-mismatch "
            "condition is used only for this controlled physics-verification test and "
            "does not represent an assumed experimental path mismatch for Configuration C."
        )
    else:
        d_panel = (
            "(d) Phase-dependent interference. Normalized mean signal intensity versus "
            "relative object phase φ over a dense 0…2π sweep for A–D, with sinusoidal "
            "fits. Reported fringe visibility is extracted from the full-fringe fit "
            "(not a four-point max/min diagnostic)."
        )
    caption = f"""Figure 3. Physics and numerical verification of the QIUP digital twin.

This figure presents software/physics verification of the digital-twin implementation against independent equations and fixed-physics numerical refinement. It is not experimental validation against laboratory measurements.

(a) SPDC gain relation. Exact mean photon number per mode n̄ = sinh²(r) and the low-gain approximation n̄ ≈ r² are compared with the numerical implementation of the r → n̄ mapping. This panel verifies the implemented gain-to-photon-number relation; it does not verify a pump-power chain P_p → r.

(b) Phase-matching response. The analytical phase-matching envelope is shown versus normalized mismatch ΔkL/(2π). Markers/vertical lines are calculated operating points for configurations A–D at the nominal wavelengths 532/810/1550 nm (no Δk calibration offset is applied).

(c) Object-transmission response. Visibility V versus object amplitude transmission T over T = 0…1 for configurations A–D (merged transparent/blocked endpoint controls). Endpoints reproduce the transparent (T = 1) and blocked (T = 0) verification controls.

{d_panel}

(e) Numerical convergence for lab-matched configuration C (f_i = 150 mm) at fixed physical field of view. Left axis: successive-grid intensity-map L₂ error; right axis: PSF FWHM versus grid size N (N = 24, 36, 48, 96, 144, 192). {claim}

Wavelength/energy consistency (relative inverse-wavelength residual |ε| ≈ 1.59×10⁻⁵ for 532/810/1550 nm) is retained as a supplementary verification table/figure and is not shown in the main panel set.
"""
    pub = OUT / "publication"
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "Figure_3_caption.txt").write_text(caption, encoding="utf-8")

    md = f"""# Figure 3 caption

{caption}

## Panel source tasks

| Panel | Corrected source |
|---|---|
| (a) | `evidence/taskX3_U01_U03_rerun` (U02) |
| (b) | `evidence/taskX3_U01_U03_rerun` (U03) |
| (c) | `evidence/taskX4_transmission_sweep` |
| (d) | `{panel_d_lineage}` |
| (e) | `evidence/taskX7_U14_convergence` |
| Supp. U01 | `evidence/taskX3_U01_U03_rerun` (U01) |

## Explicitly excluded from Figure 3

- Old U09 spatial-mapping panel (moved to Figure 4; see task X6)
- Old U08 four-point visibility bar chart (superseded; identity retained as verification only)
- Pre-X2 obsolete numerical results
"""
    (pub / "Figure_3_caption.md").write_text(md, encoding="utf-8")
    # also at evidence root for discoverability
    (OUT / "Figure_3_caption.txt").write_text(caption, encoding="utf-8")
    (OUT / "Figure_3_caption.md").write_text(md, encoding="utf-8")


def write_report(asym: dict[str, Any], copied: dict[str, list[str]]) -> None:
    lines = [
        "# Task X8 — Final publication Figure 3",
        "",
        f"Generated (UTC): `{utc_now()}`",
        "",
        "## Structure",
        "",
        "| Panel | Content | Corrected source |",
        "|---|---|---|",
        "| (a) | SPDC gain sinh²/r² + numerical | X3/U02 |",
        "| (b) | Phase-matching envelope + A–D operating points | X3/U03 |",
        "| (c) | Visibility vs T = 0…1 (A–D) | X4 |",
        "| (d) | Phase fringe + sinusoidal fit + fitted V | X5 |",
        "| (e) | Intensity error + PSF/FWHM (N ≤ 192) | X7 |",
        "| Supp. | U01 wavelength/energy table + figure | X3/U01 |",
        "",
        "## Exclusions",
        "",
        "- U09 not in Figure 3 (Figure 4).",
        "- U08 bar chart not used.",
        "- Obsolete pre-X2 numerics not reused.",
        "",
        "## Convergence statement (from X7)",
        "",
        f"- asymptotic_claimed = `{asym['asymptotic_convergence_claimed']}`",
        f"- {asym['reason']}",
        "",
        "## Outputs",
        "",
        "- `publication/Figure_3_complete.{png,pdf,svg}`",
        "- `panels/panel_{a–e}_*.{png,pdf,svg}`",
        "- `Figure_3_source_data/`",
        "- `Figure_3_caption.txt` / `.md`",
        "- `supplementary/U01/`",
        "",
        "**STOP after X8.**",
        "",
    ]
    (OUT / "report" / "X8_FINAL_FIGURE3_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_replacement_manifest() -> None:
    rows = [
        {
            "OLD_OUTPUT": "evidence/task23_gap_closure/publication/Figure_3/Figure_3_complete.png",
            "NEW_OUTPUT": "evidence/taskX8_final_Figure3/publication/Figure_3_complete.png",
            "REASON": "Full rebuild from X2–X7 corrected results; compact (a–e); no U08/U09",
            "STATUS": "SUPERSEDED_BY_PHYSICS_REVIEW",
        },
        {
            "OLD_OUTPUT": "evidence/task05_u1_u9/figures/U08/U08_visibility_identity_by_configuration.png",
            "NEW_OUTPUT": "evidence/taskX8_final_Figure3/panels/panel_d_phase_fringe.png",
            "REASON": "U08 bar excluded from Figure 3; fringe fit panel (d) used instead",
            "STATUS": "SUPERSEDED_FOR_PUBLICATION",
        },
        {
            "OLD_OUTPUT": "evidence/task05_u1_u9/figures/U09/",
            "NEW_OUTPUT": "evidence/taskX6_U09_Figure4/publication/Figure4_ABCD_spatial_benchmark.png",
            "REASON": "U09 remains out of Figure 3 (Figure 4)",
            "STATUS": "MOVED_TO_FIGURE_4",
        },
    ]
    save_csv(OUT / "replacement_manifest.csv", rows)


def main() -> None:
    for sub in (
        "data",
        "figures",
        "publication",
        "panels",
        "Figure_3_source_data",
        "supplementary",
        "tests",
        "logs",
        "provenance",
        "report",
    ):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # Fail closed if corrected evidence missing
    required = [
        X3 / "data" / "U02" / "U02_independent_gain_curves.csv",
        X3 / "data" / "U03" / "U03_delta_k_comparison.csv",
        X4 / "data" / "transmission_sweep_ABCD.csv",
        panel_d_paths()[0] / "data" / panel_d_paths()[1],
        panel_d_paths()[0] / "data" / panel_d_paths()[2],
        X7 / "data" / "successive_grid_changes.csv",
        X7 / "data" / "asymptotic_assessment.json",
        X3 / "data" / "U01" / "U01_energy_residuals.csv",
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Corrected X2–X7 source missing: {p}")

    print("X8: copy Figure_3_source_data")
    copied = copy_sources()

    print("X8: build Figure 3 (a-e)")
    asym = build_figure3()
    print("X8: export separate panels")
    build_standalone_panels(asym)
    print("X8: U01 supplementary")
    build_u01_supplementary()
    write_captions(asym)
    write_report(asym, copied)
    write_replacement_manifest()

    save_json(
        OUT / "provenance" / "manifest.json",
        {
            "task": "taskX8_final_Figure3",
            "generated_utc": utc_now(),
            "panels": ["a", "b", "c", "d", "e"],
            "excluded": ["U08_bar", "U09_spatial", "pre_X2_obsolete"],
            "sources": {
                "a": "taskX3/U02",
                "b": "taskX3/U03",
                "c": "taskX4",
                "d": "taskX5",
                "e": "taskX7",
                "U01_supp": "taskX3/U01",
            },
            "asymptotic_claimed": asym["asymptotic_convergence_claimed"],
            "environment": get_environment(),
            "source_hashes": hash_source_modules(),
        },
    )

    hash_rows = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".csv",
            ".json",
            ".png",
            ".pdf",
            ".svg",
            ".md",
            ".txt",
            ".log",
            ".py",
            ".npz",
        }:
            hash_rows.append(
                {
                    "path": str(p.relative_to(OUT)).replace("\\", "/"),
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                }
            )
    save_csv(OUT / "provenance" / "output_hashes.csv", hash_rows)

    (OUT / "logs" / "taskX8_final_Figure3.log").write_text(
        f"{utc_now()} | Figure 3 rebuilt from X3/X4/X5/X7; "
        f"asymptotic_claimed={asym['asymptotic_convergence_claimed']}\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")
    print(f"asymptotic_claimed={asym['asymptotic_convergence_claimed']}")


if __name__ == "__main__":
    main()
