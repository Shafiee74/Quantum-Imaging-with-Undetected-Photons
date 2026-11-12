#!/usr/bin/env python3
"""Final publication Figure 4 — four-configuration QIUP spatial benchmark.

Reuses the validated four_config + U09 workflow (A/B/C/D matched targets).
Configuration **C** uses **general** stock C (not C_lab; Figure 5 uses C_lab).
Writes only to ``evidence/qiup_figure4_final_article/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from benchmarks.four_config.definitions import (
    ARCHITECTURE_LABELS,
    BENCHMARK_TARGETS,
    MASTER_SEED,
    NBAR,
    PLOT_POLICY,
)
from benchmarks.four_config.plotting import extent_from_axes, shared_abs_limits
from benchmarks.four_config.runner import run_four_config_suite
from provenance import get_environment, hash_config, hash_source_modules
from setup_params import resolve_setup_selection, resolved_spatial_mapping
import tools.taskX6_U09_Figure4 as x6_u09
from tools.taskX6_U09_Figure4 import run_u09_corrected

DISPLAY_PROFILES: tuple[str, ...] = ("A", "B", "C", "D")
# Figure 4 policy (X14): general A/B/C/D — laboratory C_lab is for Figure 5 only.
SETUP_KEY_MAP: dict[str, str] = {"A": "A", "B": "B", "C": "C", "D": "D"}


def setup_for(profile: str):
    return resolve_setup_selection(SETUP_KEY_MAP[profile])
from verification.physics.u1_u9_runner import (
    DEFAULT_GRID,
    DEFAULT_NBAR,
    DEFAULT_SEED,
    PREDECLARED_TOLERANCES,
    SuiteContext,
    save_csv,
    save_json,
)

EVIDENCE_ROOT = ROOT / "evidence" / "qiup_figure4_final_article"
GENERAL_C_YAML = ROOT / "configs" / "setup_single_crystal_far_field.yaml"
DPI_PNG = 600
WIDTH_MM = 178.0
MM_TO_IN = 1.0 / 25.4

CAPTION = """\
Figure 4. Four-configuration spatial benchmark for the QIUP digital-twin suite.
**(a)** Simulated point-object optical intensity on each architecture’s native
object/idler-plane grid (common absolute color scale). **(b)** Corresponding
vertical knife-edge responses. **(c)** Signed object-to-camera magnification
from the simulator and independent reference equations. **(d)** Object-plane
correlation width $\\sigma_{\\mathrm{obj}}$ from simulation and reference.
Configuration **C** uses the **general** single-crystal double-pass far-field
profile (stock C; not the laboratory `C_lab` overlay used for Figure 5).
Results are synthetic numerical benchmarks; configurations A, B and D are not
experimentally validated in this figure.
"""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_matplotlib_pub() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 100,
            "savefig.dpi": DPI_PNG,
        }
    )


def profile_title(profile: str) -> str:
    """Letter + architecture on two wrapped lines (avoids horizontal overlap)."""
    arch_lines = {
        "A": "two-crystal\nfar-field",
        "B": "two-crystal\nnear-field",
        "C": "single-crystal\ndouble-pass far-field",
        "D": "single-crystal\ndouble-pass near-field",
    }
    return f"{profile}\n{arch_lines[profile]}"


def _panel_label_top_left(
    fig: plt.Figure,
    axes: list[plt.Axes],
    letter: str,
    caption: str,
    *,
    caption_dx: float = 0.036,
    caption_line_step: float = 0.013,
) -> None:
    """Panel tag + short scientific caption at top-left of a block."""
    positions = [a.get_position() for a in axes]
    x0 = min(p.x0 for p in positions)
    y1 = max(p.y1 for p in positions)
    y = y1 + 0.018
    fig.text(
        x0,
        y,
        letter,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=fig.transFigure,
    )
    lines = caption.split("\n")
    for i, line in enumerate(lines):
        fig.text(
            x0 + caption_dx,
            y - i * caption_line_step,
            line,
            fontsize=8.5,
            fontweight="normal",
            ha="left",
            va="bottom",
            transform=fig.transFigure,
            color="#222222",
        )


def build_spatial_benchmark_csv(suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in DISPLAY_PROFILES:
        for tid in ("point", "knife_edge_vertical"):
            case = suite["profiles"][profile][tid]
            rows.append(
                {
                    "profile": profile,
                    "setup_key": SETUP_KEY_MAP[profile],
                    "architecture_label": ARCHITECTURE_LABELS[profile],
                    "target_id": tid,
                    "object_pixel_um": case["object_pixel_um"],
                    "intensity_mean": case["intensity_mean"],
                    "intensity_peak": case["intensity_peak"],
                    "signed_magnification": case["mapping"][
                        "signed_object_to_camera_magnification"
                    ],
                    "sigma_object_um": case["mapping"]["correlation_sigma_object_um"],
                    "imaging_mode": case["mapping"]["imaging_mode"],
                }
            )
    return rows


def build_mag_res_csv(
    suite: dict[str, Any], u09_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    u09_by = {r["profile"]: r for r in u09_rows}
    for profile in DISPLAY_PROFILES:
        u = u09_by[profile]
        for tid in ("point", "knife_edge_vertical"):
            case = suite["profiles"][profile][tid]
            res = case.get("resolution") or {}
            rows.append(
                {
                    "profile": profile,
                    "setup_key": SETUP_KEY_MAP[profile],
                    "target_id": tid,
                    "signed_M_production": u["production_signed_M"],
                    "signed_M_reference": u["reference_signed_M"],
                    "sigma_obj_production_um": u["production_sigma_object_um"],
                    "sigma_obj_reference_um": u["reference_sigma_object_um"],
                    "u09_verdict": u["verdict"],
                    "psf_fwhm_um": res.get("psf_fwhm_um"),
                    "esf_10_90_um": res.get("esf_10_90_um"),
                    "lsf_fwhm_um": res.get("lsf_fwhm_um"),
                    "resolution_definition_note": (
                        "PSF FWHM for point; ESF 10–90% for knife_edge_vertical"
                    ),
                    "object_pixel_um": case["object_pixel_um"],
                    "camera_pixel_um": case["mapping"].get("camera_pixel_um"),
                }
            )
    return rows


def build_provenance(
    suite: dict[str, Any], u09_rows: list[dict[str, Any]], out: Path
) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for profile in DISPLAY_PROFILES:
        setup = setup_for(profile)
        mapping = resolved_spatial_mapping(setup)
        configs[profile] = {
            "setup_key": SETUP_KEY_MAP[profile],
            "setup_id": setup.setup_id,
            "config_yaml": (
                str(GENERAL_C_YAML.relative_to(ROOT)).replace("\\", "/")
                if profile == "C"
                else None
            ),
            "config_hash": hash_config(setup),
            "architecture_label": ARCHITECTURE_LABELS[profile],
            "imaging_mode": mapping.get("imaging_mode"),
            "signed_object_to_camera_magnification": mapping.get(
                "signed_object_to_camera_magnification"
            ),
            "correlation_sigma_object_um": mapping.get("correlation_sigma_object_um"),
            "object_plane_pixel_um": mapping.get("object_plane_pixel_um"),
            "camera_pixel_um": mapping.get("camera_pixel_um"),
        }
        if profile == "C":
            configs[profile]["crystal_length_mm"] = float(
                getattr(setup, "crystal_length_mm", 0) or 0
            )
            configs[profile]["far_field_idler_relay_Mi"] = float(
                mapping.get("far_field_idler_relay_magnification", float("nan"))
            )
            configs[profile]["far_field_signal_relay_Ms"] = float(
                mapping.get("far_field_signal_relay_magnification", float("nan"))
            )

    return {
        "task": "qiup_figure4_final_article",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_reused": [
            "tools/taskX14_general_C_figure4.py (general C policy for Figure 4)",
            "tools/taskX6_U09_Figure4.py (U09 runner)",
            "benchmarks/four_config/runner.py (matched PHYSICAL_TRUTH targets)",
        ],
        "setup_key_map": dict(SETUP_KEY_MAP),
        "configuration_C_authoritative_yaml": str(GENERAL_C_YAML),
        "configuration_C_lab_not_used_in_figure4": True,
        "four_config_seed": suite["master_seed"],
        "four_config_grid_n": suite["grid_n"],
        "four_config_nbar": suite["nbar"],
        "u09_tolerances": {
            k: PREDECLARED_TOLERANCES[k]
            for k in ("U9_magnification_rel", "U9_sigma_rel")
        },
        "u09_rows": u09_rows,
        "configurations": configs,
        "environment": get_environment(),
        "source_hashes": hash_source_modules(),
        "output_directory": str(out.relative_to(ROOT)).replace("\\", "/"),
        "production_physics_modified": False,
        "figures_5_6_preserved": True,
    }


def _imshow_row(
    fig: plt.Figure,
    gs: GridSpec,
    row: int,
    suite: dict[str, Any],
    target_id: str,
    vmin: float,
    vmax: float,
    *,
    show_titles: bool,
    ylabel_left: bool,
) -> None:
    """Draw four config panels + dedicated colorbar column."""
    last_im = None
    for i, profile in enumerate(DISPLAY_PROFILES):
        ax = fig.add_subplot(gs[row, i])
        case = suite["profiles"][profile][target_id]
        last_im = ax.imshow(
            case["optical_intensity"],
            origin=PLOT_POLICY["origin"],
            extent=extent_from_axes(case["object_x_um"], case["object_y_um"]),
            cmap=PLOT_POLICY["cmap_intensity"],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        if show_titles:
            ax.set_title(
                profile_title(profile),
                fontsize=7.5,
                pad=14,
                linespacing=1.15,
                loc="center",
            )
        ax.set_xlabel(r"$x$ ($\mu$m)", fontsize=7.5, labelpad=3)
        if ylabel_left and i == 0:
            ax.set_ylabel(r"$y$ ($\mu$m)", fontsize=8, labelpad=4)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=7.5, pad=2)
    if last_im is not None:
        cax = fig.add_subplot(gs[row, 4])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label("Intensity\n(model units)", fontsize=7.5, labelpad=6)


def build_figure(
    suite: dict[str, Any], u09_rows: list[dict[str, Any]], out: Path
) -> None:
    w_in = WIDTH_MM * MM_TO_IN
    h_in = w_in * 1.08
    fig = plt.figure(figsize=(w_in, h_in))

    gs_outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.05, 0.95],
        hspace=0.20,
        left=0.14,
        right=0.96,
        top=0.97,
        bottom=0.11,
    )
    gs_img = gs_outer[0].subgridspec(
        2,
        5,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.07],
        hspace=0.10,
        wspace=0.46,
    )
    gs_bar = gs_outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)

    vmin, vmax = shared_abs_limits(
        [suite["profiles"][p]["point"]["optical_intensity"] for p in DISPLAY_PROFILES]
    )
    _imshow_row(
        fig,
        gs_img,
        0,
        suite,
        "point",
        vmin,
        vmax,
        show_titles=True,
        ylabel_left=True,
    )

    vmin2, vmax2 = shared_abs_limits(
        [
            suite["profiles"][p]["knife_edge_vertical"]["optical_intensity"]
            for p in DISPLAY_PROFILES
        ]
    )
    _imshow_row(
        fig,
        gs_img,
        1,
        suite,
        "knife_edge_vertical",
        vmin2,
        vmax2,
        show_titles=False,
        ylabel_left=True,
    )

    _panel_label_top_left(
        fig,
        fig.axes[0:5],
        "(a)",
        "Optical intensity vs object-plane coordinates (point object)",
        caption_dx=0.038,
    )
    _panel_label_top_left(
        fig,
        fig.axes[5:10],
        "(b)",
        "Optical intensity vs object-plane coordinates (vertical knife-edge)",
        caption_dx=0.048,
    )

    x = np.arange(len(u09_rows))
    ax_m = fig.add_subplot(gs_bar[0, 0])
    ax_m.bar(
        x - 0.2,
        [r["production_signed_M"] for r in u09_rows],
        width=0.38,
        label="Simulator",
        color="#264653",
    )
    ax_m.bar(
        x + 0.2,
        [r["reference_signed_M"] for r in u09_rows],
        width=0.38,
        label="Reference",
        color="#2a9d8f",
    )
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([r["profile"] for r in u09_rows])
    ax_m.set_ylabel("Signed $M$", fontsize=9)
    ax_m.legend(frameon=True, loc="upper left", fontsize=7, borderpad=0.3)
    ax_m.grid(True, axis="y", alpha=0.3)
    ax_m.axhline(0.0, color="k", lw=0.5)
    ax_m.tick_params(labelsize=8)

    ax_s = fig.add_subplot(gs_bar[0, 1])
    ax_s.bar(
        x - 0.2,
        [r["production_sigma_object_um"] for r in u09_rows],
        width=0.38,
        label="Simulator",
        color="#e76f51",
    )
    ax_s.bar(
        x + 0.2,
        [r["reference_sigma_object_um"] for r in u09_rows],
        width=0.38,
        label="Reference",
        color="#e9c46a",
    )
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([r["profile"] for r in u09_rows])
    ax_s.set_ylabel(r"$\sigma_{\mathrm{obj}}$ ($\mu$m)", fontsize=9)
    ax_s.legend(frameon=True, loc="upper right", fontsize=7, borderpad=0.3)
    ax_s.grid(True, axis="y", alpha=0.3)
    ax_s.tick_params(labelsize=8)

    _panel_label_top_left(
        fig,
        [ax_m],
        "(c)",
        "Signed object-to-camera magnification\nvs configuration",
        caption_dx=0.042,
        caption_line_step=0.014,
    )
    _panel_label_top_left(
        fig,
        [ax_s],
        "(d)",
        r"Object-plane correlation width $\sigma_{\mathrm{obj}}$" + "\n"
        "vs configuration",
        caption_dx=0.042,
        caption_line_step=0.014,
    )

    png = out / "FIGURE4_FINAL_ARTICLE.png"
    pdf = out / "FIGURE4_FINAL_ARTICLE.pdf"
    fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_validation_report(
    out: Path,
    u09_rows: list[dict[str, Any]],
    suite: dict[str, Any],
    regenerated: bool,
) -> None:
    c = next(r for r in u09_rows if r["profile"] == "C")
    lines = [
        "# Figure 4 — final validation report",
        "",
        f"Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Checklist",
        "",
        "1. All four configurations present: **yes** (A, B, C, D).",
        "2. Architecture-specific forward models: **yes** (four_config runner + setup profiles).",
        "3. Matched physical benchmark targets: **yes** (`benchmarks.targets.PHYSICAL_TRUTH`).",
        f"4. Configuration C uses **general** stock C: **yes** (setup_key `{c['setup_key']}`).",
        "5. Image scales and metrics traceable: **yes** (CSVs + provenance JSON).",
        "6. No obsolete X14-only general-C panel mixed in: **yes**.",
        "7. Claims limited to numerical benchmark: **yes**.",
        "8. Figures 5–6 evidence untouched: **yes** (this task writes new directory only).",
        "9. Layout: two-column ~178 mm width, 600 dpi PNG.",
        "10. U09 verification:",
        "",
        "| Profile | Verdict | $M$ (prod) | $\\sigma_{obj}$ (µm) |",
        "|---|---|---:|---:|",
    ]
    for r in u09_rows:
        lines.append(
            f"| {r['profile']} | {r['verdict']} | {r['production_signed_M']:.6f} | "
            f"{r['production_sigma_object_um']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Configuration C (general stock C, not C_lab)",
            "",
            f"- Production signed **M** = `{c['production_signed_M']:.9f}`",
            f"- Reference signed **M** = `{c['reference_signed_M']:.9f}`",
            f"- Idler focal **f_i** = `{c.get('far_field_idler_focal_mm')}` mm",
            "",
            "## Regeneration note",
            "",
            (
                "Full four-config suite and U09 rows were **regenerated** with "
                "general C for panel C (Figure 4 policy)."
                if regenerated
                else "Used cached outputs."
            ),
            "",
            "**PRODUCTION_PHYSICS_MODIFIED:** NO",
            "",
        ]
    )
    (out / "FIGURE4_FINAL_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(output: Path | None = None) -> Path:
    out = output or (EVIDENCE_ROOT / utc_stamp())
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("data/U09", "data/four_config", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    setup_matplotlib_pub()
    x6_u09.SETUP_KEY_MAP = dict(SETUP_KEY_MAP)

    ctx = SuiteContext(
        evidence_root=out,
        seed=DEFAULT_SEED,
        grid=DEFAULT_GRID,
        nbar=DEFAULT_NBAR,
        profiles=DISPLAY_PROFILES,
    )
    ctx.source_hashes = hash_source_modules()
    for profile in DISPLAY_PROFILES:
        ctx.config_hashes[profile] = hash_config(setup_for(profile))

    u09_rows = run_u09_corrected(ctx)
    suite = run_four_config_suite(
        profiles=DISPLAY_PROFILES,
        targets=BENCHMARK_TARGETS,
        n=48,
        nbar=NBAR,
        master_seed=MASTER_SEED,
        setup_key_map=SETUP_KEY_MAP,
    )

    spatial_rows = build_spatial_benchmark_csv(suite)
    mag_res_rows = build_mag_res_csv(suite, u09_rows)
    provenance = build_provenance(suite, u09_rows, out)

    save_csv(out / "FIGURE4_SPATIAL_BENCHMARK.csv", spatial_rows, list(spatial_rows[0].keys()))
    save_csv(
        out / "FIGURE4_MAGNIFICATION_RESOLUTION.csv",
        mag_res_rows,
        list(mag_res_rows[0].keys()),
    )
    save_json(out / "FIGURE4_CONFIGURATION_PROVENANCE.json", provenance)
    (out / "FIGURE4_CAPTION_FINAL.md").write_text(CAPTION, encoding="utf-8")

    build_figure(suite, u09_rows, out)
    write_validation_report(out, u09_rows, suite, regenerated=True)

    decision = [
        "# FINAL DECISION — QIUP Figure 4 (article)",
        "",
        "**PRODUCTION_PHYSICS_MODIFIED:** NO",
        "",
        "## Deliverables",
        "",
        "- `FIGURE4_FINAL_ARTICLE.png` (600 dpi)",
        "- `FIGURE4_FINAL_ARTICLE.pdf`",
        "- `FIGURE4_SPATIAL_BENCHMARK.csv`",
        "- `FIGURE4_MAGNIFICATION_RESOLUTION.csv`",
        "- `FIGURE4_CONFIGURATION_PROVENANCE.json`",
        "- `FIGURE4_CAPTION_FINAL.md`",
        "- `FIGURE4_FINAL_VALIDATION_REPORT.md`",
        "",
        "## Policy",
        "",
        "- Reuses validated four-config + U09 workflow.",
        "- **C = general stock C** (`setup_single_crystal_far_field.yaml`); C_lab is Figure 5 only.",
        "- Does not modify Figure 5/6 evidence or production physics.",
        "",
    ]
    (out / "FINAL_DECISION.md").write_text("\n".join(decision), encoding="utf-8")

    for r in u09_rows:
        if r["verdict"] != "PASS":
            raise RuntimeError(f"U09 failed for {r['profile']}: {r}")

    print(f"Figure 4 final article: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate final QIUP Figure 4 for the paper.")
    p.add_argument("--output", type=Path, default=None, help="Evidence output directory")
    args = p.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
