#!/usr/bin/env python3
"""Publication-ready Figure 5 for the C_lab study (figure production only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from tools.c_lab_figure5_axis_variants import load_assembly
from tools.publication_graphics_style import (
    CONFIGURATION_COLORS,
    CONFIGURATION_LINESTYLES,
    CONFIGURATION_MARKERS,
    apply_publication_graphics_style,
    style_reference_axes,
)
from tools.c_lab_figure5_final_theory_experiment_intensity import (
    EXP_ROOT,
    LAMBDA_I_UM,
    THEORY_EVIDENCE,
    ExpPlotOptions,
    dz_eq_um_from_phase,
    panel_a_folders,
    plot_dz_um,
    setup_phase_axis,
    sinusoid,
    wrapped_phase_rad,
)
from tools.c_lab_manuscript_convention import (
    discover_phase_folders,
    draw_figure5a_manuscript,
    load_opd0_iminus_theory,
)
from tools.visibility import unique_tiff_files

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure5_final_article"
PLOT_OPTS = ExpPlotOptions.exclude_2pi_only()
SIMULATION_LABEL = "Simulation"
EXPERIMENTAL_LABEL = "Experimental"
EXP_DIRS = {"no_object": "witout object", "rabbit": "with object"}
IMAGE_CAPTION_FS = 12.0
IMAGE_CAPTION_PAD = 8
IMAGE_ROW_LABEL_FS = 15.0


def _apply_figure_fonts() -> None:
    apply_publication_graphics_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "axes.unicode_minus": False,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
        }
    )


def _set_image_caption(ax: plt.Axes, lines: list[str]) -> None:
    """Caption below panel (same typography for simulation and experiment)."""
    ax.set_title("")
    text = "\n".join(lines)
    ax.set_xlabel(
        text,
        fontsize=IMAGE_CAPTION_FS,
        labelpad=IMAGE_CAPTION_PAD,
        linespacing=1.45,
        multialignment="center",
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def array_sha256(array: np.ndarray) -> str:
    """Fingerprint an image array without changing its values or dtype."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def list_tiffs(condition_key: str, phase_folder: str) -> list[str]:
    p = EXP_ROOT / EXP_DIRS[condition_key] / phase_folder
    files = unique_tiff_files(p)
    return [str(f.relative_to(ROOT)).replace("\\", "/") for f in files]


def _draw_scale_bar(ax: plt.Axes, image_shape: tuple[int, ...], *, pixel_pitch_um: float = 13.0) -> None:
    """Overlay only the physical scale bar on a publication image.

    The circular ROI remains part of the numerical evidence, but it is not
    drawn over representative images. This matches the approved Figure 5
    presentation and avoids mistaking the orange ROI boundary for image data.
    """
    length_px = 500.0 / pixel_pitch_um
    y = image_shape[0] - 19
    x1 = image_shape[1] - 17
    x0 = x1 - length_px
    ax.plot([x0, x1], [y, y], color="white", lw=3.0, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - 8, "500 µm", color="white", fontsize=9,
            ha="center", va="bottom")


def draw_panel_a_article(ax_left: plt.Axes, payload: dict[str, Any]) -> None:
    exp = payload["exp"]
    phi_th, y_clear, y_rb = load_opd0_iminus_theory()
    folders = [f for f in discover_phase_folders() if f in panel_a_folders(payload["pos"], PLOT_OPTS)]
    ax_right = draw_figure5a_manuscript(
        ax_left,
        phi_th=phi_th,
        y_clear=y_clear,
        y_rb=y_rb,
        exp=exp,
        folders=folders,
    )
    style_reference_axes(ax_left)
    style_reference_axes(ax_right)


def build_article_figure(out_png: Path, out_pdf: Path, payload: dict[str, Any]) -> None:
    _apply_figure_fonts()
    summary = payload["summary"]
    sel = payload["selection"]
    hi, lo = sel["experimental_high_signal"], sel["experimental_low_signal"]

    fig = plt.figure(figsize=(18.0, 13.2))
    gs = GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[1.20, 0.10, 1.16, 1.16],
        hspace=0.30,
        wspace=0.10,
        left=0.13,
        right=0.985,
        top=0.90,
        bottom=0.06,
    )
    draw_panel_a_article(fig.add_subplot(gs[0, :]), payload)

    ax_b = fig.add_subplot(gs[1, :])
    ax_b.axis("off")
    ax_b.text(
        0.0,
        0.55,
        (
            r"$\mathbf{(b)}$ Representative high-signal and low-signal states from simulation (upper row) "
            "and experimental data (lower row) for no-object and rabbit conditions."
        ),
        fontsize=13,
        va="center",
        ha="left",
        transform=ax_b.transAxes,
        wrap=True,
    )

    theory_cols = [
        ("constructive_like", "no_object", 0, "No object — constructive-like"),
        ("constructive_like", "rabbit", 1, "Rabbit — constructive-like"),
        ("destructive_like", "no_object", 2, "No object — destructive-like"),
        ("destructive_like", "rabbit", 3, "Rabbit — destructive-like"),
    ]
    theory_axes: list[plt.Axes] = []
    for col, (state, cond, idx, title) in enumerate(theory_cols):
        ax = fig.add_subplot(gs[2, col])
        theory_axes.append(ax)
        ax.imshow(payload["theory_u16"][idx], cmap="gray", origin="upper", vmin=0, vmax=payload["theory_vmax"])
        _draw_scale_bar(ax, payload["theory_u16"][idx].shape)
        block = summary[state]
        rk = "roi_mean_no_object_au" if cond == "no_object" else "roi_mean_rabbit_au"
        phi_deg = float(np.degrees(block["phi_rad"]))
        _set_image_caption(
            ax,
            [
                title,
                f"φ = {phi_deg:.1f}°",
                f"ROI mean = {block[rk]:.4g} a.u.",
            ],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="x", which="both", length=0)

    exp_cols = [
        (hi, "no_object", 0, "No object — high-signal"),
        (hi, "rabbit", 1, "Rabbit — high-signal"),
        (lo, "no_object", 2, "No object — low-signal"),
        (lo, "rabbit", 3, "Rabbit — low-signal"),
    ]
    experimental_axes: list[plt.Axes] = []
    for col, (blk, cond, idx, title) in enumerate(exp_cols):
        ax = fig.add_subplot(gs[3, col])
        experimental_axes.append(ax)
        ax.imshow(payload["exp_images"][idx], cmap="gray", origin="upper", vmin=0, vmax=payload["exp_vmax"])
        _draw_scale_bar(ax, payload["exp_images"][idx].shape)
        em = blk["conditions"][cond]
        phi_deg = float(blk.get("wrapped_phase_deg", np.degrees(blk.get("wrapped_phase_rad", 0.0))))
        _set_image_caption(
            ax,
            [
                title,
                f"Δz = {blk['mirror_um']:.0f} µm · φ = {phi_deg:.1f}°",
                f"ROI mean = {em['roi_mean_adu_per_pixel']:.4g} ADU/pixel",
            ],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="x", which="both", length=0)

    fig.suptitle(
        "Figure 5. Phase-dependent signal intensity in simulation and experimental results.",
        fontsize=16,
        y=0.985,
        fontweight="normal",
    )
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.13, right=0.985)
    for label, axes in (
        (SIMULATION_LABEL, theory_axes),
        (EXPERIMENTAL_LABEL, experimental_axes),
    ):
        pos = axes[0].get_position()
        fig.text(
            0.018,
            0.5 * (pos.y0 + pos.y1),
            label,
            fontsize=IMAGE_ROW_LABEL_FS,
            fontweight="normal",
            ha="left",
            va="center",
        )
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run(output: Path | None = None, theory_dir: Path = THEORY_EVIDENCE) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    payload = load_assembly(theory_dir)
    sel = payload["selection"]
    hi, lo = sel["experimental_high_signal"], sel["experimental_low_signal"]

    build_article_figure(out / "FIGURE5_FINAL_ARTICLE.png", out / "FIGURE5_FINAL_ARTICLE.pdf", payload)

    theory_src = theory_dir / "FIGURE5_THEORY_SUMMARY.json"
    theory_tiff_dir = theory_dir

    report = {
        "simulation_phase_scan_evidence": str(theory_dir.relative_to(ROOT)),
        "simulation_panels": [
            {
                "role": "constructive_no_object",
                "source_tiff": str((theory_tiff_dir / "THEORY_CONSTRUCTIVE_NO_OBJECT_419x455.tif").relative_to(ROOT)),
                "phi_rad": sel["theory_constructive_like"]["phi_rad"],
                "roi_mean_au": sel["theory_constructive_like"]["roi_mean_no_object_au"],
            },
            {
                "role": "constructive_rabbit",
                "source_tiff": str((theory_tiff_dir / "THEORY_CONSTRUCTIVE_RABBIT_419x455.tif").relative_to(ROOT)),
                "phi_rad": sel["theory_constructive_like"]["phi_rad"],
                "roi_mean_au": sel["theory_constructive_like"]["roi_mean_rabbit_au"],
            },
            {
                "role": "destructive_no_object",
                "source_tiff": str((theory_tiff_dir / "THEORY_DESTRUCTIVE_NO_OBJECT_419x455.tif").relative_to(ROOT)),
                "phi_rad": sel["theory_destructive_like"]["phi_rad"],
                "roi_mean_au": sel["theory_destructive_like"]["roi_mean_no_object_au"],
            },
            {
                "role": "destructive_rabbit",
                "source_tiff": str((theory_tiff_dir / "THEORY_DESTRUCTIVE_RABBIT_419x455.tif").relative_to(ROOT)),
                "phi_rad": sel["theory_destructive_like"]["phi_rad"],
                "roi_mean_au": sel["theory_destructive_like"]["roi_mean_rabbit_au"],
            },
        ],
        "experimental_panels": [
            {
                "role": "high_no_object",
                "phase_folder": hi["phase_folder"],
                "mirror_um": hi["mirror_um"],
                "tiff_files": list_tiffs("no_object", hi["phase_folder"]),
                "n_frames_averaged": hi["conditions"]["no_object"]["n_frames"],
            },
            {
                "role": "high_rabbit",
                "phase_folder": hi["phase_folder"],
                "mirror_um": hi["mirror_um"],
                "tiff_files": list_tiffs("rabbit", hi["phase_folder"]),
                "n_frames_averaged": hi["conditions"]["rabbit"]["n_frames"],
            },
            {
                "role": "low_no_object",
                "phase_folder": lo["phase_folder"],
                "mirror_um": lo["mirror_um"],
                "tiff_files": list_tiffs("no_object", lo["phase_folder"]),
                "n_frames_averaged": lo["conditions"]["no_object"]["n_frames"],
            },
            {
                "role": "low_rabbit",
                "phase_folder": lo["phase_folder"],
                "mirror_um": lo["mirror_um"],
                "tiff_files": list_tiffs("rabbit", lo["phase_folder"]),
                "n_frames_averaged": lo["conditions"]["rabbit"]["n_frames"],
            },
        ],
        "panel_a_experimental_points_um": [0, 25, 50, 75],
        "panel_a_excludes_um": [100],
        "excluded_100um_reason": (
            "No 2pi/100 µm TIFF folder exists for either no-object or rabbit condition; "
            "100 µm appears only in the nominal position mapping JSON."
        ),
        "experimental_phase_status": "nominal from commanded stage displacement; not independently phase-calibrated",
        "comparison_scope": "qualitative; no absolute radiometric or matched-phase calibration",
        "roi_sem_definition": (
            "standard error of the per-frame spatial ROI mean across unique TIFF files; "
            "frames are temporal repeats and are not claimed as independent acquisitions"
        ),
        "image_display": {
            "simulation_vmin_vmax": [0.0, float(payload["theory_vmax"])],
            "experimental_vmin_vmax": [0.0, float(payload["exp_vmax"])],
            "percentile_rule": "separate per-modality maximum of imagewise 99.5th percentiles",
            "pixel_pitch_um": 13.0,
            "display_shape_rows_cols": [419, 455],
            "theory_orientation_crop": "rot90(k=2), native rows 302:721 and columns 284:739",
            "experimental_orientation_crop": "native detector orientation; no crop/resize/flip",
        },
        "image_row_labels": [SIMULATION_LABEL, EXPERIMENTAL_LABEL],
        "image_row_label_position": "beside the corresponding image row (left margin)",
        "source_image_arrays_modified": False,
        "source_image_array_sha256": {
            "simulation": [array_sha256(a) for a in payload["theory_u16"]],
            "experimental": [array_sha256(a) for a in payload["exp_images"]],
        },
        "production_files_modified": False,
    }
    (out / "FIGURE5_FINAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    acquisition_rows = payload["experimental_per_frame"]
    with (out / "FIGURE5_EXPERIMENTAL_ACQUISITION_AUDIT.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fields = list(acquisition_rows[0])
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(acquisition_rows)

    with (out / "FIGURE5_PANEL_A_DATA.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "condition", "phase_rad", "dz_um", "roi_mean", "roi_sem", "units"])
        t = payload["theory_csv"]
        for i in range(len(t["phase_rad"])):
            w.writerow(["simulation", "no_object", t["phase_rad"][i], t["dz_um_one_period"][i], t["roi_mean_no_object_au"][i], "", "a.u."])
            w.writerow(["simulation", "rabbit", t["phase_rad"][i], t["dz_um_one_period"][i], t["roi_mean_rabbit_au"][i], "", "a.u."])
        pos = payload["pos"]
        exp = payload["exp"]
        for folder in panel_a_folders(pos, PLOT_OPTS):
            j = list(pos.keys()).index(folder)
            dz = plot_dz_um(folder, pos, PLOT_OPTS)
            phi = wrapped_phase_rad(dz)
            for cond in ("no_object", "rabbit"):
                w.writerow(
                    [
                        "experimental",
                        cond,
                        phi,
                        dz,
                        float(exp[cond]["roi_mean_adu"][j]),
                        float(exp[cond]["roi_mean_sem_adu"][j]),
                        "ADU/pixel",
                    ]
                )

    img_rows = []
    for p in report["simulation_panels"]:
        img_rows.append(
            {
                "panel_type": "simulation",
                "role": p["role"],
                "source": p["source_tiff"],
                "phi_rad": p["phi_rad"],
                "roi_mean": p["roi_mean_au"],
                "units": "a.u.",
            }
        )
    for p in report["experimental_panels"]:
        cond = "no_object" if "no_object" in p["role"] else "rabbit"
        img_rows.append(
            {
                "panel_type": "experimental",
                "role": p["role"],
                "source_folder": f"exp data/mirror drift/{EXP_DIRS[cond]}/{p['phase_folder']}",
                "mirror_um": p["mirror_um"],
                "n_tiff_files": len(p["tiff_files"]),
                "tiff_files": ";".join(p["tiff_files"]),
                "n_frames_averaged": p["n_frames_averaged"],
            }
        )
    meta_fields = sorted({k for row in img_rows for k in row})
    with (out / "FIGURE5_IMAGE_PANEL_METADATA.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=meta_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(img_rows)

    (out / "FIGURE5_CAPTION_FINAL.md").write_text(
        f"""# Figure 5 caption

**Figure 5. Phase-dependent signal intensity in simulation and experimental results.**

**(a)** ROI mean signal intensity versus interferometric phase φ for no-object and rabbit conditions. Simulation curves show the computed phase scan over one period (arbitrary units). Experimental points correspond to the four available laboratory idler mirror positions (0, 25, 50, and 75 µm) and are plotted at the **nominal**, wrapped phase φ = (4πΔz/λ_i) mod 2π with λ_i = 1.55 µm. The nominal positions file also lists 100 µm, but neither condition contains a corresponding `2pi` TIFF folder; that setting is therefore excluded as missing rather than treated as a measurement. This nominal conversion is not an independent phase calibration; the upper axis gives the equivalent displacement within one 2π period. Experimental intensities are raw ADU per pixel. Error bars are the standard error of the per-frame spatial ROI mean across unique TIFF files; these temporal frames are not asserted to be independent acquisitions.

**Lower panels** show representative constructive-like and destructive-like states from simulation (top row) and high-signal and low-signal experimental mirror positions (bottom row), for no-object and rabbit conditions in each column. The explicit **Simulation** and **Experimental** labels are placed beside their corresponding image rows. Yellow contours mark the fixed ROI and white bars show 500 µm. Simulation TIFFs are a 180° rotation and fixed native crop (rows 302:721, columns 284:739); experimental images retain native orientation and are not cropped, resized, or flipped. Each modality uses its own common grayscale range (0 to the maximum imagewise 99.5th percentile), so the comparison is qualitative rather than absolutely radiometric or independently matched-phase.

Simulation constructive-like phase: φ ≈ {sel['theory_constructive_like']['phi_rad']:.3f} rad; destructive-like: φ ≈ {sel['theory_destructive_like']['phi_rad']:.3f} rad. Experimental high-signal mirror position: {hi['mirror_um']:.0f} µm; low-signal: {lo['mirror_um']:.0f} µm.
""",
        encoding="utf-8",
    )

    (out / "FIGURE5_FINAL_REPORT.md").write_text(
        f"""# Figure 5 final article production

- **Evidence:** `{out}`
- **Simulation phase scan / simulation TIFFs:** `{theory_dir.relative_to(ROOT)}`
- **Production files modified:** NO
- **Image-row layout:** `Simulation` and `Experimental` labels beside their corresponding rows
- **Source image arrays modified:** NO

## Outputs
- FIGURE5_FINAL_ARTICLE.png / .pdf
- FIGURE5_PANEL_A_DATA.csv
- FIGURE5_IMAGE_PANEL_METADATA.csv
- FIGURE5_EXPERIMENTAL_ACQUISITION_AUDIT.csv
- FIGURE5_CAPTION_FINAL.md
- FIGURE5_FINAL_REPORT.json

## Simulation image panels
| Panel | φ (rad) | Source |
|-------|---------|--------|
| Constructive, no object | {sel['theory_constructive_like']['phi_rad']:.4f} | THEORY_CONSTRUCTIVE_NO_OBJECT_419x455.tif |
| Constructive, rabbit | {sel['theory_constructive_like']['phi_rad']:.4f} | THEORY_CONSTRUCTIVE_RABBIT_419x455.tif |
| Destructive, no object | {sel['theory_destructive_like']['phi_rad']:.4f} | THEORY_DESTRUCTIVE_NO_OBJECT_419x455.tif |
| Destructive, rabbit | {sel['theory_destructive_like']['phi_rad']:.4f} | THEORY_DESTRUCTIVE_RABBIT_419x455.tif |

## Experimental image panels
| Panel | Δz (µm) | Folder | Frames (no-obj / rabbit) |
|-------|---------|--------|--------------------------|
| High, no object | {hi['mirror_um']:.0f} | {hi['phase_folder']} | {hi['conditions']['no_object']['n_frames']} |
| High, rabbit | {hi['mirror_um']:.0f} | {hi['phase_folder']} | {hi['conditions']['rabbit']['n_frames']} |
| Low, no object | {lo['mirror_um']:.0f} | {lo['phase_folder']} | {lo['conditions']['no_object']['n_frames']} |
| Low, rabbit | {lo['mirror_um']:.0f} | {lo['phase_folder']} | {lo['conditions']['rabbit']['n_frames']} |

See `FIGURE5_FINAL_REPORT.json` for full TIFF file lists.
""",
        encoding="utf-8",
    )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / "LATEST.txt").write_text(
        f"{out.name}\n{out.resolve()}\n", encoding="utf-8"
    )

    print(f"Done: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--theory-dir", type=Path, default=THEORY_EVIDENCE)
    args = p.parse_args()
    run(args.output, theory_dir=args.theory_dir)


if __name__ == "__main__":
    main()
