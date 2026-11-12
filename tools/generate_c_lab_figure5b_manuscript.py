#!/usr/bin/env python3
"""Manuscript Figure 5(b) — image panel (OPD0 I_minus theory + fresh exp TIFFs)."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.c_lab_display_zoom import apply_fixed_crop
from setup_params import resolve_setup_selection
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.c_lab_l2_ms1_mi1_calculated_dk0_rabbit_images import N, object_mask_original
from tools.c_lab_manuscript_convention import (
    COND_SUBDIR,
    CLabManuscriptSettings,
    DEFAULT_MANUSCRIPT_SETTINGS,
    DELTA_L_MM,
    FOLDER_DZ_UM,
    ROI_PIXEL_COUNT,
    ROI_RADIUS_PX,
    THEORY_OUTPUT_PORT,
    THEORY_ROI_PIXELS,
    discover_phase_folders,
    experimental_roi_mask,
    theory_high_low_phases,
    wrapped_phase_rad,
)
from tools.c_lab_rabbit_object_orientation_correction import DISPLAY_ROT90_K
from tools.publication_graphics_style import apply_publication_graphics_style, style_reference_axes
from tools.z2.physics import ref_from_T

ART = ROOT / "article figures"
PIXEL_PITCH_UM = 13.0
# Approved Fig. 5 theory display crop (post rot90 k=2) — 419×455 px.
THEORY_DISPLAY_BOUNDS = {"row0": 302, "row1": 721, "col0": 284, "col1": 739}
CONDITION_LABEL = {"no_object": "without object", "rabbit": "with object"}

# Manuscript Figure 5(b) — exact experimental frames (single TIFF per panel).
EXPERIMENTAL_PANEL_TIFF: dict[tuple[str, str], str] = {
    ("no_object", "high-signal"): "QIUP_Mon Sep 28 2026_12.46.48.tif",
    ("rabbit", "high-signal"): "QIUP_Mon Sep 28 2026_13.06.07.tif",
    ("no_object", "low-signal"): "QIUP_Mon Sep 28 2026_12.50.01.tif",
    ("rabbit", "low-signal"): "QIUP_Mon Sep 28 2026_12.54.45.tif",
}


def resolve_experimental_panel_tiff(
    settings: CLabManuscriptSettings,
    condition: str,
    state: str,
) -> Path:
    fname = EXPERIMENTAL_PANEL_TIFF[(condition, state)]
    sub = COND_SUBDIR[condition]
    root = settings.exp_root / sub
    hits = [p.resolve() for p in root.rglob(fname) if p.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{fname}' under {root} for {condition} {state}; found {len(hits)}: {hits}"
        )
    return hits[0]


def load_experimental_panel_from_tiff(
    settings: CLabManuscriptSettings,
    condition: str,
    state: str,
) -> dict[str, Any]:
    label = CONDITION_LABEL[condition]
    path = resolve_experimental_panel_tiff(settings, condition, state)
    folder = path.parent.name
    if folder not in FOLDER_DZ_UM:
        raise ValueError(f"Unexpected phase folder '{folder}' for {path}")
    img = np.asarray(tifffile.imread(path), dtype=np.float64)
    roi = experimental_roi_mask(settings)
    roi_mean = float(img[roi].mean())
    dz = FOLDER_DZ_UM[folder]
    phi = wrapped_phase_rad(dz)
    return {
        "modality": "experiment",
        "condition": condition,
        "state": state,
        "phase_folder": folder,
        "delta_z_um": dz,
        "phi_rad": phi,
        "n_unique_frames": 1,
        "roi_mean_adu": roi_mean,
        "image": img,
        "caption_title": f"Experiment — {label} — {state}",
        "source_tiff": path.name,
        "source_path": str(path),
    }


def load_figure5b_plan(csv_path: Path) -> tuple[float, float, dict[str, str]]:
    phi_hi = phi_lo = None
    sel: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["modality"] == "theory" and row["condition"] == "no_object":
                phi = float(row["phi_rad"])
                if row["state"] == "high-signal":
                    phi_hi = phi
                elif row["state"] == "low-signal":
                    phi_lo = phi
            if row["modality"] != "experiment":
                continue
            key = f"{'high' if row['state'] == 'high-signal' else 'low'}_{row['condition']}"
            sel[key] = row["phase_folder"]
    if phi_hi is None or phi_lo is None or len(sel) != 4:
        raise ValueError(f"Incomplete Figure 5(b) plan in {csv_path}")
    return phi_hi, phi_lo, sel


def dz_equiv_um(phi_rad: float, lambda_um: float) -> float:
    return float(phi_rad * lambda_um / (4.0 * math.pi))


def theory_roi_cumulative_90(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    s = np.asarray(n1, float) + np.asarray(n2, float)
    flat = s.ravel()
    order = np.argsort(flat)[::-1]
    cs = np.cumsum(flat[order])
    idx = int(np.searchsorted(cs, 0.9 * cs[-1]))
    mask = np.zeros(flat.size, bool)
    mask[order[: idx + 1]] = True
    return mask.reshape(s.shape)


def theory_display_iminus(native: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    oriented = np.rot90(np.asarray(native, dtype=np.float64), k=DISPLAY_ROT90_K)
    return apply_fixed_crop(oriented, bounds)


def forward_iminus_display(
    setup,
    ref,
    dk: DeltaKCalibrationState,
    phi: float,
    bounds: dict[str, int],
    theory_roi: np.ndarray,
) -> tuple[np.ndarray, float]:
    case = run_forward_case(
        setup.overridden(delta_L_mm=DELTA_L_MM, relative_interferometric_phase=float(phi)),
        ref,
        dk,
        grid_n=N,
    )
    im = np.asarray(case["I_minus"], dtype=np.float64)
    disp = theory_display_iminus(im, bounds)
    roi_mean = float(im[theory_roi].mean())
    return disp, roi_mean


def shared_vmax(images: list[np.ndarray], pct: float = 99.5) -> float:
    return float(max(np.percentile(np.asarray(im, dtype=np.float64), pct) for im in images))


def draw_scale_bar(ax: plt.Axes, shape: tuple[int, int]) -> None:
    length_px = 500.0 / PIXEL_PITCH_UM
    y = shape[0] - 19
    x1 = shape[1] - 17
    x0 = x1 - length_px
    ax.plot([x0, x1], [y, y], color="white", lw=3.0, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - 8, "500 µm", color="white", fontsize=9, ha="center", va="bottom")


def set_panel_caption(ax: plt.Axes, lines: list[str], *, row: int) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    y = -0.04 if row == 0 else -0.06
    ax.text(
        0.5,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.35,
        clip_on=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Use verified Figure5b_Final_Data.csv for phases/folders; redraw PNG/PDF only.",
    )
    args = parser.parse_args()

    settings = DEFAULT_MANUSCRIPT_SETTINGS
    ART.mkdir(parents=True, exist_ok=True)
    data_csv = ART / "Figure5b_Final_Data.csv"

    folders = discover_phase_folders(settings)
    if args.figures_only:
        if not data_csv.is_file():
            raise FileNotFoundError(data_csv)
        phi_hi, phi_lo, _sel = load_figure5b_plan(data_csv)
    else:
        phi_hi, phi_lo, _, _ = theory_high_low_phases(settings)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref_clear = ref_from_T(np.ones((N, N), dtype=np.float32))
    ref_rb = ref_from_T(object_mask_original().astype(np.float32))
    bounds = THEORY_DISPLAY_BOUNDS
    case0 = run_forward_case(
        setup.overridden(delta_L_mm=DELTA_L_MM, relative_interferometric_phase=0.0),
        ref_clear,
        dk,
        grid_n=N,
    )
    th_roi = theory_roi_cumulative_90(case0["cap"]["N1"], case0["cap"]["N2"])
    if int(th_roi.sum()) != THEORY_ROI_PIXELS:
        raise RuntimeError(f"Theory ROI pixel count {int(th_roi.sum())} != {THEORY_ROI_PIXELS}")

    def theory_panel(cond: str, ref, state: str, phi: float) -> dict[str, Any]:
        label = CONDITION_LABEL[cond]
        disp, roi_mean = forward_iminus_display(setup, ref, dk, phi, bounds, th_roi)
        return {
            "modality": "theory",
            "condition": cond,
            "state": state,
            "phi_rad": phi,
            "delta_z_um": dz_equiv_um(phi, settings.lambda_i_um),
            "roi_mean_au": roi_mean,
            "image": disp,
            "caption_title": f"Theory — {label} — {state}",
        }

    col_theory = [
        theory_panel("no_object", ref_clear, "high-signal", phi_hi),
        theory_panel("rabbit", ref_rb, "high-signal", phi_hi),
        theory_panel("no_object", ref_clear, "low-signal", phi_lo),
        theory_panel("rabbit", ref_rb, "low-signal", phi_lo),
    ]

    col_exp = [
        load_experimental_panel_from_tiff(settings, "no_object", "high-signal"),
        load_experimental_panel_from_tiff(settings, "rabbit", "high-signal"),
        load_experimental_panel_from_tiff(settings, "no_object", "low-signal"),
        load_experimental_panel_from_tiff(settings, "rabbit", "low-signal"),
    ]

    th_vmax = shared_vmax([p["image"] for p in col_theory])
    ex_vmax = shared_vmax([p["image"] for p in col_exp])

    apply_publication_graphics_style()
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.65))
    for col, (tp, ep) in enumerate(zip(col_theory, col_exp)):
        for row, panel, vmax in ((0, tp, th_vmax), (1, ep, ex_vmax)):
            ax = axes[row, col]
            ax.imshow(panel["image"], cmap="gray", origin="upper", vmin=0, vmax=vmax, aspect="equal")
            draw_scale_bar(ax, panel["image"].shape)
            phi_deg = float(np.degrees(panel["phi_rad"]))
            if panel["modality"] == "theory":
                set_panel_caption(
                    ax,
                    [
                        panel["caption_title"],
                        f"φ = {phi_deg:.1f}°",
                    ],
                    row=row,
                )
            else:
                set_panel_caption(
                    ax,
                    [
                        panel["caption_title"],
                        f"Δz = {panel['delta_z_um']:.0f} µm · φ = {phi_deg:.1f}°",
                    ],
                    row=row,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            style_reference_axes(ax)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.07, hspace=0.27, wspace=0.05)
    fig.text(0.02, 0.70, "Simulation", fontsize=14, va="center", rotation=90)
    fig.text(0.02, 0.33, "Experiment", fontsize=14, va="center", rotation=90)
    fig.savefig(ART / "Figure5b_Final.png", dpi=600)
    try:
        fig.savefig(ART / "Figure5b_Final.pdf")
    except OSError:
        fig.savefig(ART / "Figure5b_Final_regenerated.pdf")
    plt.close(fig)

    if args.figures_only:
        print(f"Figure 5(b) graphics -> {ART}")
        print("Theory phi_hi (deg):", math.degrees(phi_hi), "phi_lo:", math.degrees(phi_lo))
        for p in col_exp:
            print(p["source_path"])
        return

    rows: list[dict] = []
    for p in col_theory + col_exp:
        rows.append(
            {
                "modality": p["modality"],
                "condition": p["condition"],
                "state": p["state"],
                "phase_folder": p.get("phase_folder", ""),
                "delta_z_um": p["delta_z_um"],
                "phi_rad": p["phi_rad"],
                "roi_mean_au": p.get("roi_mean_au", ""),
                "roi_mean_adu": p.get("roi_mean_adu", ""),
                "n_unique_frames": p.get("n_unique_frames", ""),
                "source_tiff": p.get("source_tiff", ""),
                "source_path": p.get("source_path", ""),
            }
        )
    with (ART / "Figure5b_Final_Data.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ts = datetime.now(timezone.utc).isoformat()
    report = f"""# Figure 5(b) — manuscript final

Generated: {ts}  
Location: `article figures/`

## Conventions (matched to Figure 5(a))

| Item | Value |
|------|--------|
| Theory | Joint-SPDC, physical **{THEORY_OUTPUT_PORT}**, **delta_L_mm = {DELTA_L_MM}** (in memory) |
| Theory ROI | **{THEORY_ROI_PIXELS}** px (90% cumulative ∑(N1+N2)) |
| Experimental TIFF root | `{settings.exp_root.resolve()}` |
| Experimental ROI | center **(227, 209)**, radius **{ROI_RADIUS_PX}** px (**{ROI_PIXEL_COUNT}** px) |
| Panel images | **Single named TIFF per panel** (not folder mean); ROI mean recomputed from that frame |
| Cached experiment CSV reused | **NO** |

## Theory panel phases (no-object OPD0 I_minus extrema)

| State | φ (rad) | φ (deg) | Δz equiv. (µm) |
|-------|--------:|--------:|---------------:|
| High-signal | {phi_hi:.6f} | {math.degrees(phi_hi):.2f} | {dz_equiv_um(phi_hi, settings.lambda_i_um):.4f} |
| Low-signal | {phi_lo:.6f} | {math.degrees(phi_lo):.2f} | {dz_equiv_um(phi_lo, settings.lambda_i_um):.4f} |

## Experimental image provenance (Figure 5(b) lower row)

| Panel | Condition | Phase folder | Nominal Δz (µm) | φ (deg) | Filename | Absolute path | ROI mean (ADU/px, this frame) |
|-------|-----------|--------------|----------------:|--------:|----------|---------------|------------------------------:|
"""
    for p in col_exp:
        report += (
            f"| {p['caption_title']} | {CONDITION_LABEL[p['condition']]} | `{p['phase_folder']}` | "
            f"{p['delta_z_um']:.0f} | {math.degrees(p['phi_rad']):.2f} | `{p['source_tiff']}` | "
            f"`{p['source_path']}` | {p['roi_mean_adu']:.6f} |\n"
        )

    report += "\nFigure face omits ROI radius and ROI-mean annotations (manuscript rule).\n"
    (ART / "Figure5b_Final_Report.md").write_text(report, encoding="utf-8")

    print("Figure 5(b) ->", ART)
    print("Theory phi_hi (deg):", math.degrees(phi_hi), "phi_lo:", math.degrees(phi_lo))
    for p in col_exp:
        print(p["source_path"])


if __name__ == "__main__":
    main()
