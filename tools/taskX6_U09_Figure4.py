#!/usr/bin/env python3
"""Task X6 — rerun U09 with X2-corrected Configuration C; regenerate Figure 4.

Publication Configuration C uses setup alias C_lab (f_i=F1=F2=150 mm).
Stock C (f_i=75 mm) is retained only for old_vs_new comparison and must not be
reported as lab-matched. U09 spatial presentation moves from Figure 3 to Figure 4.
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

from benchmarks.four_config.definitions import (
    ARCHITECTURE_LABELS,
    BENCHMARK_TARGETS,
    MASTER_SEED,
    NBAR,
    PLOT_POLICY,
    PROFILES,
)
from benchmarks.four_config.plotting import (
    extent_from_axes,
    save_comparison_2x2,
    save_image_panel,
    save_line_profile_overlay,
    shared_abs_limits,
)
from benchmarks.four_config.runner import run_four_config_suite
from benchmarks.targets.geometry import centered_axis_um
from provenance import get_environment, hash_config, hash_source_modules
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.physics import independent_equations as iref
from verification.physics.u1_u9_runner import (
    DEFAULT_GRID,
    DEFAULT_NBAR,
    DEFAULT_SEED,
    PREDECLARED_TOLERANCES,
    SuiteContext,
    abs_err,
    decide,
    point_object,
    rel_err,
    run_sim,
    save_csv,
    save_json,
    save_npy,
)

OUT = ROOT / "evidence" / "taskX6_U09_Figure4"
OLD_U09 = ROOT / "evidence" / "task05_u1_u9"
OLD_TASK15 = ROOT / "evidence" / "task15_four_config_benchmark"
OLD_FIG4 = ROOT / "evidence" / "task23_gap_closure" / "publication" / "Figure_4"

# Publication A/B/C/D: C resolves to X2-corrected lab-matched geometry.
SETUP_KEY_MAP: dict[str, str] = {"A": "A", "B": "B", "C": "C_lab", "D": "D"}
DISPLAY_PROFILES: tuple[str, ...] = ("A", "B", "C", "D")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def setup_for(profile: str):
    return resolve_setup_selection(SETUP_KEY_MAP[profile])


def physical_extent_mm(axis_um: np.ndarray) -> list[float]:
    """imshow extent in mm from a centered µm axis."""
    x = np.asarray(axis_um, dtype=np.float64)
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    return [
        float((x[0] - 0.5 * dx) * 1.0e-3),
        float((x[-1] + 0.5 * dx) * 1.0e-3),
        float((x[0] - 0.5 * dx) * 1.0e-3),
        float((x[-1] + 0.5 * dx) * 1.0e-3),
    ]


def write_old_vs_new_c_mapping() -> list[dict[str, Any]]:
    old = resolve_setup_selection("C")
    new = resolve_setup_selection("C_lab")
    m_old = resolved_spatial_mapping(old)
    m_new = resolved_spatial_mapping(new)

    def row(
        quantity: str,
        old_v: Any,
        new_v: Any,
        unit: str,
        note: str,
    ) -> dict[str, Any]:
        return {
            "quantity": quantity,
            "old_stock_C_fi75": old_v,
            "new_lab_C_fi150": new_v,
            "unit": unit,
            "note": note,
        }

    rows = [
        row(
            "setup_id",
            old.setup_id,
            new.setup_id,
            "",
            "Stock C retained; publication/lab-matched C is config_c_lab_v1",
        ),
        row(
            "far_field_focal_mm_f_i",
            m_old.get("far_field_idler_focal_mm"),
            m_new.get("far_field_idler_focal_mm"),
            "mm",
            "Idler/Fourier lens; X2 corrected 75→150 for lab C",
        ),
        row(
            "F1_mm",
            float(old.F1_mm),
            float(new.F1_mm),
            "mm",
            "Synced with far_field_focal_mm for M_QIUP",
        ),
        row(
            "F2_mm_f_s_f_c",
            float(old.F2_mm),
            float(new.F2_mm),
            "mm",
            "Signal-path + camera/front lens (unchanged 150)",
        ),
        row(
            "signed_object_to_camera_magnification_M",
            m_old["signed_object_to_camera_magnification"],
            m_new["signed_object_to_camera_magnification"],
            "dimensionless",
            "Do NOT report old M=-1.045 as lab-matched C",
        ),
        row(
            "magnification_magnitude",
            m_old["magnification_magnitude"],
            m_new["magnification_magnitude"],
            "dimensionless",
            "|M| ≈ 0.5226 after lab correction",
        ),
        row(
            "correlation_sigma_object_um",
            m_old["correlation_sigma_object_um"],
            m_new["correlation_sigma_object_um"],
            "µm",
            "Scales ~linearly with f_i (≈2×)",
        ),
        row(
            "correlation_sigma_camera_um",
            m_old["correlation_sigma_camera_um"],
            m_new["correlation_sigma_camera_um"],
            "µm",
            "Camera-plane correlation width",
        ),
        row(
            "object_plane_pixel_um",
            m_old["object_plane_pixel_um"],
            m_new["object_plane_pixel_um"],
            "µm",
            "Declared object-map sampling",
        ),
        row(
            "camera_pixel_um",
            m_old["camera_pixel_um"],
            m_new["camera_pixel_um"],
            "µm",
            "Detector pitch",
        ),
        row(
            "lambda_p_nm",
            float(old.lambda_p_nm),
            float(new.lambda_p_nm),
            "nm",
            "X2 nominal 532 nm",
        ),
        row(
            "lambda_s_nm",
            float(old.lambda_s_nm),
            float(new.lambda_s_nm),
            "nm",
            "",
        ),
        row(
            "lambda_i_nm",
            float(old.lambda_i_nm),
            float(new.lambda_i_nm),
            "nm",
            "",
        ),
    ]
    save_csv(
        OUT / "data" / "old_vs_new_C_spatial_mapping.csv",
        rows,
        list(rows[0].keys()),
    )
    save_json(OUT / "data" / "old_vs_new_C_spatial_mapping.json", rows)
    return rows


def run_u09_corrected(ctx: SuiteContext) -> list[dict[str, Any]]:
    """U09 production vs independent reference with publication C = C_lab."""
    u = "U09"
    ctx.log(f"{u} start spatial mapping (publication C = C_lab)")
    rows: list[dict[str, Any]] = []

    for profile in DISPLAY_PROFILES:
        setup = setup_for(profile)
        mapping = resolved_spatial_mapping(setup)
        mode = mapping["imaging_mode"]
        if mode == "far_field":
            ref_M = iref.far_field_signed_object_to_camera_scale(
                m_i=float(mapping["far_field_idler_relay_magnification"]),
                m_s=float(mapping["far_field_signal_relay_magnification"]),
                f_i_mm=float(mapping["far_field_idler_focal_mm"]),
                f_s_mm=float(mapping["far_field_signal_focal_mm"]),
                lambda_s_nm=float(setup.lambda_s_nm),
                lambda_i_nm=float(setup.lambda_i_nm),
            )
            ref_Mag = iref.far_field_qiup_magnification(
                float(mapping["far_field_idler_focal_mm"]),
                float(mapping["far_field_signal_focal_mm"]),
                float(setup.lambda_s_nm),
                float(setup.lambda_i_nm),
            )
            waist = float(mapping["pump_transverse_profile"]["waist_um"])
            ref_sig = iref.far_field_correlation_sigma_um(
                waist,
                float(mapping["far_field_idler_focal_mm"]),
                float(setup.lambda_i_nm),
                m_i=float(mapping["far_field_idler_relay_magnification"]),
            )
        else:
            mu = float(mapping["near_field_idler_magnification_Mu"])
            ms = float(mapping["near_field_signal_magnification_Ms"])
            ref_M = iref.near_field_magnification(mu, ms)
            ref_Mag = abs(ref_M)
            jg = mapping["joint_gaussian_coefficients"]
            ref_sig = iref.near_field_conditional_sigma_um(
                float(jg["crystal_length_um"]),
                float(jg["lambda_d_um"]),
                float(jg["lambda_u_um"]),
                float(jg["pump_waist_um"]),
            )

        prod_M = float(mapping["signed_object_to_camera_magnification"])
        prod_Mag = float(mapping["magnification_magnitude"])
        prod_sig = float(mapping["correlation_sigma_object_um"])
        re_M = rel_err(prod_M, ref_M)
        re_sig = rel_err(prod_sig, ref_sig)
        v_M = decide("U9_magnification_rel", re_M)
        v_sig = decide("U9_sigma_rel", re_sig)
        verdict = (
            "PASS"
            if v_M == "PASS" and v_sig == "PASS"
            else "FAIL"
            if None not in (v_M, v_sig)
            else None
        )
        row = {
            "profile": profile,
            "setup_key": SETUP_KEY_MAP[profile],
            "setup_id": setup.setup_id,
            "imaging_mode": mode,
            "far_field_idler_focal_mm": mapping.get("far_field_idler_focal_mm"),
            "far_field_signal_focal_mm": mapping.get("far_field_signal_focal_mm"),
            "production_signed_M": prod_M,
            "reference_signed_M": ref_M,
            "production_M_magnitude": prod_Mag,
            "reference_M_magnitude": ref_Mag,
            "production_sigma_object_um": prod_sig,
            "reference_sigma_object_um": ref_sig,
            "rel_error_M": re_M,
            "rel_error_sigma": re_sig,
            "verdict_M": v_M,
            "verdict_sigma": v_sig,
            "verdict": verdict,
        }
        rows.append(row)

        # Point-object mapping image on the signal / camera plane.
        _, _, result = run_sim(
            SETUP_KEY_MAP[profile], point_object(ctx.grid), ctx
        )
        save_npy(
            ctx.u_data(u) / f"{profile}_point_object_ideal_signal.npy",
            result.ideal_signal_image,
        )
        save_json(
            ctx.u_data(u) / f"{profile}_spatial_mapping.json",
            {
                k: mapping[k]
                for k in mapping
                if not isinstance(mapping[k], (np.ndarray,))
            },
        )

        cam_px = float(mapping["camera_pixel_um"])
        n = int(result.ideal_signal_image.shape[0])
        x_um = centered_axis_um(n, cam_px)
        extent_mm = physical_extent_mm(x_um)
        I = np.asarray(result.ideal_signal_image, dtype=np.float64)
        I_norm = I / np.max(I) if np.max(I) > 0 else I

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(
            I_norm,
            origin="lower",
            cmap="magma",
            extent=extent_mm,
            aspect="equal",
        )
        ax.set_xlabel(r"Signal-plane coordinate, $x_s$ (mm)")
        ax.set_ylabel(r"Signal-plane coordinate, $y_s$ (mm)")
        ax.set_title(
            f"U09 point-object map ({profile}; setup={SETUP_KEY_MAP[profile]})"
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Normalized intensity")
        fig.tight_layout()
        fig.savefig(
            ctx.u_fig(u) / f"U09_spatial_mapping_config_{profile}.png", dpi=160
        )
        plt.close(fig)

    save_csv(
        ctx.u_data(u) / "U09_spatial_mapping_comparison.csv",
        rows,
        list(rows[0].keys()),
    )
    save_json(ctx.u_data(u) / "U09_spatial_mapping_comparison.json", rows)

    # Magnification bar — publication Figure 4 companion, not Figure 3.
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(rows))
    ax.bar(
        x - 0.2,
        [r["production_signed_M"] for r in rows],
        width=0.4,
        label="production $M$",
        color="#264653",
    )
    ax.bar(
        x + 0.2,
        [r["reference_signed_M"] for r in rows],
        width=0.4,
        label="independent reference $M$",
        color="#2a9d8f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{r['profile']}"
            + ("\n(C_lab)" if r["profile"] == "C" else "")
            for r in rows
        ]
    )
    ax.set_ylabel("Signed magnification, $M$")
    ax.set_xlabel("Configuration")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.6)
    # Annotate C so old M=-1.045 is not implied
    c_row = next(r for r in rows if r["profile"] == "C")
    ax.annotate(
        f"C: $M$={c_row['production_signed_M']:.4f}\n"
        f"$f_i$={c_row['far_field_idler_focal_mm']:.0f} mm",
        xy=(2, c_row["production_signed_M"]),
        xytext=(2.35, c_row["production_signed_M"] - 0.35),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    fig.tight_layout()
    fig.savefig(
        ctx.u_fig(u) / "U09_signed_magnification_by_configuration.png", dpi=160
    )
    plt.close(fig)

    # Correlation-width companion
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.bar(
        x - 0.2,
        [r["production_sigma_object_um"] for r in rows],
        width=0.4,
        label="production $\\sigma$",
        color="#e76f51",
    )
    ax.bar(
        x + 0.2,
        [r["reference_sigma_object_um"] for r in rows],
        width=0.4,
        label="independent reference $\\sigma$",
        color="#e9c46a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([r["profile"] for r in rows])
    ax.set_ylabel(r"Correlation width, $\sigma$ ($\mathrm{\mu m}$)")
    ax.set_xlabel("Configuration")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        ctx.u_fig(u) / "U09_correlation_sigma_by_configuration.png", dpi=160
    )
    plt.close(fig)

    ctx.log(f"{u} done")
    return rows


def save_four_config_case(case: dict[str, Any]) -> None:
    profile = case["profile"]
    tid = case["target_id"]
    d = OUT / "data" / "four_config" / profile / tid
    d.mkdir(parents=True, exist_ok=True)
    fdir = OUT / "figures" / "four_config" / profile
    fdir.mkdir(parents=True, exist_ok=True)

    np.save(d / "object_T.npy", case["object_T"])
    np.save(d / "optical_intensity.npy", case["optical_intensity"])
    np.save(d / "visibility_map.npy", case["visibility_map"])
    np.save(d / "object_x_um.npy", case["object_x_um"])
    np.save(d / "object_y_um.npy", case["object_y_um"])
    np.save(d / "detector_x_um.npy", case["detector_x_um"])
    np.save(d / "detector_y_um.npy", case["detector_y_um"])

    scalars = {
        k: case[k]
        for k in (
            "profile",
            "architecture_label",
            "target_id",
            "seed_int",
            "nbar",
            "grid_n",
            "object_pixel_um",
            "N_1",
            "N_2",
            "C_12_abs",
            "visibility_V_50_50",
            "visibility_map_mean",
            "intensity_mean",
            "intensity_peak",
            "mapping",
        )
    }
    scalars["setup_key"] = SETUP_KEY_MAP[profile]
    if case["phase"]:
        np.save(
            d / "phase_reconstructed_rad.npy",
            case["phase"]["phase_reconstructed_rad"],
        )
        scalars["phase_rmse_rad"] = case["phase"]["phase_rmse_rad"]
    if case["resolution"]:
        scalars["resolution_metrics"] = {
            k: v
            for k, v in case["resolution"].items()
            if not isinstance(v, dict)
        }
    save_json(d / "metrics.json", scalars)

    save_image_panel(
        fdir / f"{tid}__intensity.png",
        case["optical_intensity"],
        x_um=case["object_x_um"],
        y_um=case["object_y_um"],
        title=f"{profile} ({ARCHITECTURE_LABELS[profile]})\n{tid}: intensity",
        cmap=PLOT_POLICY["cmap_intensity"],
        cbar_label="Normalized intensity",
    )
    save_image_panel(
        fdir / f"{tid}__visibility.png",
        case["visibility_map"],
        x_um=case["object_x_um"],
        y_um=case["object_y_um"],
        title=f"{profile}: {tid} visibility map",
        cmap=PLOT_POLICY["cmap_visibility"],
        vmin=0.0,
        vmax=1.0,
        cbar_label="Visibility, $V$",
    )


def make_figure4(suite: dict[str, Any], u09_rows: list[dict[str, Any]]) -> None:
    """Publication Figure 4: ABCD spatial benchmark + U09 mapping (not Fig 3)."""
    pub = OUT / "publication"
    pub.mkdir(parents=True, exist_ok=True)
    cdir = OUT / "figures" / "four_config" / "comparison_panels"
    cdir.mkdir(parents=True, exist_ok=True)

    # Comparison panels for key targets
    for tid in ("point", "knife_edge_vertical", "transparent", "phase_step"):
        images = {
            p: suite["profiles"][p][tid]["optical_intensity"] for p in DISPLAY_PROFILES
        }
        coords = {
            p: (
                suite["profiles"][p][tid]["object_x_um"],
                suite["profiles"][p][tid]["object_y_um"],
            )
            for p in DISPLAY_PROFILES
        }
        # Peak-normalize each panel for display clarity but keep absolute companion
        # via shared absolute scale as policy.
        save_comparison_2x2(
            cdir / f"{tid}__intensity_ABCD.png",
            images,
            coordinates=coords,
            title=f"Matched {tid}: optical intensity (A–D)",
            cmap=PLOT_POLICY["cmap_intensity"],
            cbar_label="optical intensity [arb]",
            shared_scale=True,
        )
        lines = {
            p: suite["profiles"][p][tid]["line_profile_intensity"]
            for p in DISPLAY_PROFILES
        }
        save_line_profile_overlay(
            cdir / f"{tid}__line_profiles_ABCD.png",
            lines,
            title=f"Matched {tid}: central intensity line profiles",
            ylabel="Optical intensity [arb]",
        )

    # Visibility panel for transparent (control)
    vimages = {
        p: suite["profiles"][p]["transparent"]["visibility_map"]
        for p in DISPLAY_PROFILES
    }
    coords = {
        p: (
            suite["profiles"][p]["transparent"]["object_x_um"],
            suite["profiles"][p]["transparent"]["object_y_um"],
        )
        for p in DISPLAY_PROFILES
    }
    save_comparison_2x2(
        cdir / "transparent__visibility_ABCD.png",
        vimages,
        coordinates=coords,
        title="Matched transparent: visibility map",
        cmap=PLOT_POLICY["cmap_visibility"],
        cbar_label="Visibility, V",
        shared_scale=True,
    )

    # Compose multi-panel Figure 4
    fig = plt.figure(figsize=(12.5, 10.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.85], hspace=0.38, wspace=0.32)

    # Row 0: point-object intensity A–D (object plane, µm)
    vmin, vmax = shared_abs_limits(
        [suite["profiles"][p]["point"]["optical_intensity"] for p in DISPLAY_PROFILES]
    )
    for i, profile in enumerate(DISPLAY_PROFILES):
        ax = fig.add_subplot(gs[0, i])
        case = suite["profiles"][profile]["point"]
        im = ax.imshow(
            case["optical_intensity"],
            origin="lower",
            extent=extent_from_axes(case["object_x_um"], case["object_y_um"]),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        label = profile
        if profile == "C":
            label = "C (lab, $f_i$=150 mm)"
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(r"$x_i$ ($\mathrm{\mu m}$)", fontsize=8)
        ax.set_ylabel(r"$y_i$ ($\mathrm{\mu m}$)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Row 1: knife-edge A–D
    vmin2, vmax2 = shared_abs_limits(
        [
            suite["profiles"][p]["knife_edge_vertical"]["optical_intensity"]
            for p in DISPLAY_PROFILES
        ]
    )
    for i, profile in enumerate(DISPLAY_PROFILES):
        ax = fig.add_subplot(gs[1, i])
        case = suite["profiles"][profile]["knife_edge_vertical"]
        im = ax.imshow(
            case["optical_intensity"],
            origin="lower",
            extent=extent_from_axes(case["object_x_um"], case["object_y_um"]),
            cmap="viridis",
            vmin=vmin2,
            vmax=vmax2,
            aspect="equal",
        )
        ax.set_title(f"{profile}: knife-edge", fontsize=9)
        ax.set_xlabel(r"Idler/object-plane $x_i$ ($\mathrm{\mu m}$)", fontsize=7)
        ax.set_ylabel(r"$y_i$ ($\mathrm{\mu m}$)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Row 2: U09 magnification + sigma + C annotation
    ax_m = fig.add_subplot(gs[2, 0:2])
    x = np.arange(len(u09_rows))
    ax_m.bar(
        x - 0.2,
        [r["production_signed_M"] for r in u09_rows],
        width=0.4,
        label="production",
        color="#264653",
    )
    ax_m.bar(
        x + 0.2,
        [r["reference_signed_M"] for r in u09_rows],
        width=0.4,
        label="reference",
        color="#2a9d8f",
    )
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([r["profile"] for r in u09_rows])
    ax_m.set_ylabel("Signed magnification, $M$")
    ax_m.set_xlabel("Configuration")
    ax_m.legend(fontsize=7)
    ax_m.grid(True, axis="y", alpha=0.3)
    ax_m.axhline(0.0, color="k", lw=0.5)
    ax_m.set_title(
        "U09 spatial magnification (C uses lab $f_i$=150 mm; not $M$=−1.045)",
        fontsize=9,
    )

    ax_s = fig.add_subplot(gs[2, 2:4])
    ax_s.bar(
        x - 0.2,
        [r["production_sigma_object_um"] for r in u09_rows],
        width=0.4,
        label="production",
        color="#e76f51",
    )
    ax_s.bar(
        x + 0.2,
        [r["reference_sigma_object_um"] for r in u09_rows],
        width=0.4,
        label="reference",
        color="#e9c46a",
    )
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([r["profile"] for r in u09_rows])
    ax_s.set_ylabel(r"Correlation width $\sigma$ ($\mathrm{\mu m}$)")
    ax_s.set_xlabel("Configuration")
    ax_s.legend(fontsize=7)
    ax_s.grid(True, axis="y", alpha=0.3)
    ax_s.set_title("U09 object-plane correlation width", fontsize=9)

    fig.suptitle(
        "Figure 4 — Four-configuration spatial benchmark (U09 mapping included)",
        fontsize=12,
        y=0.995,
    )
    out_png = pub / "Figure4_ABCD_spatial_benchmark.png"
    out_pdf = pub / "Figure4_ABCD_spatial_benchmark.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also copy U09 magnification into publication/
    shutil.copy2(
        OUT / "figures" / "U09" / "U09_signed_magnification_by_configuration.png",
        pub / "Figure4_U09_signed_magnification_ABCD.png",
    )
    shutil.copy2(
        OUT / "figures" / "U09" / "U09_correlation_sigma_by_configuration.png",
        pub / "Figure4_U09_correlation_sigma_ABCD.png",
    )


def write_replacement_manifest() -> None:
    pairs = [
        (
            "evidence/task05_u1_u9/figures/U09/U09_signed_magnification_by_configuration.png",
            "evidence/taskX6_U09_Figure4/publication/Figure4_U09_signed_magnification_ABCD.png",
            "U09 moved out of Figure 3; C uses lab f_i=150 mm (M≈-0.523, not -1.045)",
            "MOVED_TO_FIGURE_4",
        ),
        (
            "evidence/task05_u1_u9/figures/U09/U09_spatial_mapping_config_C.png",
            "evidence/taskX6_U09_Figure4/figures/U09/U09_spatial_mapping_config_C.png",
            "Point-object map regenerates with physical signal-plane axes; C=C_lab",
            "SUPERSEDED_BY_PHYSICS_REVIEW",
        ),
        (
            "evidence/task05_u1_u9/data/U09/U09_spatial_mapping_comparison.csv",
            "evidence/taskX6_U09_Figure4/data/U09/U09_spatial_mapping_comparison.csv",
            "Stock-C M=-1.045 superseded for lab-matched/publication C",
            "SUPERSEDED_BY_PHYSICS_REVIEW",
        ),
        (
            "evidence/task15_four_config_benchmark/tables/mapping_resolution_summary.csv",
            "evidence/taskX6_U09_Figure4/data/four_config/mapping_resolution_summary.csv",
            "ABCD spatial benchmark remapped with C_lab focals",
            "SUPERSEDED_BY_PHYSICS_REVIEW",
        ),
        (
            "evidence/task23_gap_closure/publication/Figure_4/Figure_4_complete.png",
            "evidence/taskX6_U09_Figure4/publication/Figure4_ABCD_spatial_benchmark.png",
            "Figure 4 regenerated after X2 C geometry correction",
            "SUPERSEDED_BY_PHYSICS_REVIEW",
        ),
    ]
    rows = []
    for old, new, reason, status in pairs:
        rows.append(
            {
                "OLD_OUTPUT": old,
                "NEW_OUTPUT": new,
                "REASON": reason,
                "STATUS": status,
                "OLD_EXISTS": (ROOT / old).is_file()
                if not old.endswith("**")
                else (ROOT / old.split("**")[0]).exists(),
                "NEW_EXISTS": (ROOT / new).is_file(),
                "NOTE": "U09 presentation OUT of Figure 3; belongs with Figure 4",
            }
        )
    save_csv(OUT / "replacement_manifest.csv", rows, list(rows[0].keys()))


def write_report(u09_rows: list[dict[str, Any]], ovn: list[dict[str, Any]]) -> None:
    c = next(r for r in u09_rows if r["profile"] == "C")
    old_m = next(
        r for r in ovn if r["quantity"] == "signed_object_to_camera_magnification_M"
    )
    lines = [
        "# Task X6 — U09 rerun and Figure 4 spatial benchmark",
        "",
        f"Generated (UTC): `{utc_now()}`",
        "",
        "## Policy",
        "",
        "- Publication Configuration **C** = X2-corrected `C_lab` "
        "(`far_field_focal_mm`=`F1_mm`=`F2_mm`=**150 mm**).",
        "- Stock Configuration C (`f_i`=75 mm) is **not** reported as lab-matched.",
        "- Old lab-facing claim **M = −1.045** is obsolete for corrected C.",
        "- U09 spatial presentation is **moved out of Figure 3** into **Figure 4**.",
        "",
        "## U09 production vs independent equations",
        "",
        "| Config | setup | mode | prod M | ref M | σ_obj (µm) | verdict |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in u09_rows:
        lines.append(
            f"| {r['profile']} | {r['setup_key']} | {r['imaging_mode']} | "
            f"{r['production_signed_M']:.6f} | {r['reference_signed_M']:.6f} | "
            f"{r['production_sigma_object_um']:.4f} | {r['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Configuration C geometry correction",
            "",
            f"- Old stock C signed M: `{old_m['old_stock_C_fi75']}`",
            f"- New lab C signed M: `{old_m['new_lab_C_fi150']}` "
            f"(publication value ≈ **{c['production_signed_M']:.6f}**)",
            f"- New lab C σ_object: `{c['production_sigma_object_um']:.4f}` µm "
            f"(≈2× old 74 µm)",
            "",
            "## Axis-label policy",
            "",
            "- Signal/camera images: `Signal-plane coordinate, x_s (mm)`.",
            "- Object/idler maps: `Idler/object-plane coordinate, x_i (µm)`.",
            "- Color scales: `Normalized intensity` or `Visibility, V` as appropriate.",
            "- Generic `x` / `pixel` / `index` avoided when calibration exists.",
            "",
            "## Deliverables",
            "",
            "- `data/U09/` — U09 comparison + per-config mapping JSON/NPY",
            "- `data/old_vs_new_C_spatial_mapping.csv`",
            "- `data/four_config/` — remapped ABCD spatial benchmark",
            "- `publication/Figure4_ABCD_spatial_benchmark.png`",
            "- `replacement_manifest.csv`",
            "",
            "**STOP after X6.**",
            "",
        ]
    )
    (OUT / "report" / "X6_U09_FIGURE4_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_provenance(u09_rows: list[dict[str, Any]], suite: dict[str, Any]) -> None:
    configs = {}
    for profile in DISPLAY_PROFILES:
        setup = setup_for(profile)
        configs[profile] = {
            "setup_key": SETUP_KEY_MAP[profile],
            "setup_id": setup.setup_id,
            "config_hash": hash_config(setup),
            "far_field_focal_mm": getattr(setup, "far_field_focal_mm", None),
            "F1_mm": float(setup.F1_mm) if setup.F1_mm is not None else None,
            "F2_mm": float(setup.F2_mm) if setup.F2_mm is not None else None,
            "lambda_p_nm": float(setup.lambda_p_nm),
            "lambda_s_nm": float(setup.lambda_s_nm),
            "lambda_i_nm": float(setup.lambda_i_nm),
        }
        mapping = resolved_spatial_mapping(setup)
        excerpt_keys = (
            "imaging_mode",
            "signed_object_to_camera_magnification",
            "magnification_magnitude",
            "correlation_sigma_object_um",
            "correlation_sigma_camera_um",
            "far_field_idler_focal_mm",
            "far_field_signal_focal_mm",
            "object_plane_pixel_um",
            "camera_pixel_um",
        )
        save_json(
            OUT / "configs" / f"setup_{profile}.json",
            {
                "profile": profile,
                **configs[profile],
                "mapping_excerpt": {
                    k: mapping[k] for k in excerpt_keys if k in mapping
                },
            },
        )

    manifest = {
        "task": "taskX6_U09_Figure4",
        "generated_utc": utc_now(),
        "setup_key_map": SETUP_KEY_MAP,
        "u09_verdicts": {r["profile"]: r["verdict"] for r in u09_rows},
        "four_config_grid_n": suite["grid_n"],
        "four_config_nbar": suite["nbar"],
        "tolerances": {
            k: PREDECLARED_TOLERANCES[k]
            for k in ("U9_magnification_rel", "U9_sigma_rel")
        },
        "environment": get_environment(),
        "source_hashes": hash_source_modules(),
        "config_hashes": {p: configs[p]["config_hash"] for p in configs},
        "policy": {
            "publication_C": "C_lab (150/150/150 mm)",
            "stock_C_fi75": "retained for distinguishability; not lab-matched",
            "U09_figure_home": "Figure 4 (moved out of Figure 3)",
            "forbidden_claim": "M=-1.045 as lab-matched C",
        },
    }
    save_json(OUT / "provenance" / "manifest.json", manifest)

    hash_rows = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".csv",
            ".json",
            ".png",
            ".pdf",
            ".md",
            ".npy",
            ".log",
            ".py",
        }:
            hash_rows.append(
                {
                    "path": str(p.relative_to(OUT)).replace("\\", "/"),
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                }
            )
    save_csv(
        OUT / "provenance" / "output_hashes.csv",
        hash_rows,
        ["path", "sha256", "bytes"],
    )


def main() -> None:
    for sub in (
        "data",
        "figures",
        "publication",
        "tests",
        "logs",
        "provenance",
        "report",
        "configs",
        "data/U09",
        "data/four_config",
        "figures/U09",
        "figures/four_config",
    ):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    log_path = OUT / "logs" / "taskX6_U09_Figure4.log"
    print("X6: old_vs_new_C_spatial_mapping")
    ovn = write_old_vs_new_c_mapping()

    ctx = SuiteContext(
        evidence_root=OUT,
        seed=DEFAULT_SEED,
        grid=DEFAULT_GRID,
        nbar=DEFAULT_NBAR,
        profiles=DISPLAY_PROFILES,
    )
    ctx.source_hashes = hash_source_modules()
    for profile in DISPLAY_PROFILES:
        ctx.config_hashes[profile] = hash_config(setup_for(profile))

    print("X6: U09 production vs independent equations")
    u09_rows = run_u09_corrected(ctx)
    for r in u09_rows:
        print(
            f"  {r['profile']}: M={r['production_signed_M']:.6f} "
            f"sig={r['production_sigma_object_um']:.3f} {r['verdict']}"
        )

    print("X6: four-config spatial benchmark (C=C_lab)")
    suite = run_four_config_suite(
        profiles=DISPLAY_PROFILES,
        targets=BENCHMARK_TARGETS,
        n=48,
        nbar=NBAR,
        master_seed=MASTER_SEED,
        setup_key_map=SETUP_KEY_MAP,
    )
    for profile in DISPLAY_PROFILES:
        for tid in BENCHMARK_TARGETS:
            save_four_config_case(suite["profiles"][profile][tid])

    save_csv(
        OUT / "data" / "four_config" / "mapping_resolution_summary.csv",
        suite["mapping_rows"],
        list(suite["mapping_rows"][0].keys()),
    )
    save_csv(
        OUT / "data" / "four_config" / "four_config_metrics.csv",
        suite["metric_rows"],
        list(suite["metric_rows"][0].keys()),
    )
    save_json(OUT / "data" / "four_config" / "suite_summary.json", {
        "generated_utc": suite["generated_utc"],
        "setup_key_map": SETUP_KEY_MAP,
        "grid_n": suite["grid_n"],
        "nbar": suite["nbar"],
        "master_seed": suite["master_seed"],
    })

    print("X6: Figure 4 publication panel")
    make_figure4(suite, u09_rows)
    write_replacement_manifest()
    write_report(u09_rows, ovn)
    write_provenance(u09_rows, suite)

    log_path.write_text(
        "\n".join(ctx.log_lines)
        + "\n"
        + "\n".join(
            f"{r['profile']}: M={r['production_signed_M']} verdict={r['verdict']}"
            for r in u09_rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")
    c = next(r for r in u09_rows if r["profile"] == "C")
    assert abs(c["production_signed_M"] + 0.5225806451612903) < 1e-9, c
    assert abs(c["production_signed_M"] + 1.0451612903225806) > 0.1
    print(
        f"C lab-matched M={c['production_signed_M']:.6f} "
        f"(NOT -1.045); all verdicts="
        f"{ {r['profile']: r['verdict'] for r in u09_rows} }"
    )


if __name__ == "__main__":
    main()
