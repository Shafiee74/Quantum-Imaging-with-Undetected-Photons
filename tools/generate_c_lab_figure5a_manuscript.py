#!/usr/bin/env python3
"""Regenerate manuscript Figure 5(a) (fresh TIFFs, manuscript convention)."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.c_lab_manuscript_convention import (
    DEFAULT_MANUSCRIPT_SETTINGS,
    DELTA_L_MM,
    FOLDER_DZ_UM,
    ROI_PIXEL_COUNT,
    ROI_RADIUS_PX,
    THEORY_OUTPUT_PORT,
    THEORY_ROI_PIXELS,
    CLabManuscriptSettings,
    compute_experimental_statistics,
    discover_phase_folders,
    draw_figure5a_manuscript,
    load_experiment_from_figure5a_csv,
    load_opd0_iminus_theory,
    wrapped_phase_rad,
)
from tools.publication_graphics_style import apply_publication_graphics_style

ART = ROOT / "article figures"


def sampled_v(means: list[float]) -> float:
    y = means
    return (max(y) - min(y)) / (max(y) + min(y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Redraw PNG/PDF from verified article figures CSV (no TIFF reread).",
    )
    args = parser.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    data_csv = ART / "Figure5a_Final_Data.csv"
    settings = CLabManuscriptSettings(
        exp_root=DEFAULT_MANUSCRIPT_SETTINGS.exp_root,
        theory_csv=data_csv,
    )
    if args.figures_only:
        if not data_csv.is_file():
            raise FileNotFoundError(data_csv)
        phi_th, y_clear, y_rb = load_opd0_iminus_theory(settings)
        exp, folders = load_experiment_from_figure5a_csv(data_csv)
        audit: list = []
    else:
        if not settings.exp_root.is_dir():
            raise FileNotFoundError(settings.exp_root)
        phi_th, y_clear, y_rb = load_opd0_iminus_theory(settings)
        exp, pos, audit = compute_experimental_statistics(settings)
        folders = discover_phase_folders(settings)

    apply_publication_graphics_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    draw_figure5a_manuscript(ax, phi_th=phi_th, y_clear=y_clear, y_rb=y_rb, exp=exp, folders=folders)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.88, bottom=0.14)
    fig.savefig(ART / "Figure5a_Final.png", dpi=600)
    pdf_staged = ART / "_Figure5a_Final_staged.pdf"
    fig.savefig(pdf_staged)
    plt.close(fig)
    final_pdf = ART / "Figure5a_Final.pdf"
    fallback = ART / "Figure5a_Final_regenerated.pdf"
    try:
        if final_pdf.exists():
            final_pdf.unlink()
        pdf_staged.replace(final_pdf)
    except OSError:
        if fallback.exists():
            fallback.unlink()
        pdf_staged.replace(fallback)
        print("Note: Figure5a_Final.pdf locked; wrote Figure5a_Final_regenerated.pdf")

    if args.figures_only:
        print(f"Figure 5(a) graphics -> {ART}")
        return

    rows: list[dict] = []
    meta = (
        f"Joint-SPDC {THEORY_OUTPUT_PORT} delta_L_mm={DELTA_L_MM}; "
        f"theory_roi={THEORY_ROI_PIXELS}; exp={settings.exp_root.relative_to(ROOT)}; "
        f"exp_roi_r={ROI_RADIUS_PX}px n={ROI_PIXEL_COUNT}; fresh_tiff_run={datetime.now(timezone.utc).isoformat()}"
    )
    rows.append(
        {
            "row": "meta",
            "series": "",
            "folder": "",
            "condition": "",
            "dz_um": "",
            "phi_z_rad": "",
            "mean": "",
            "sd": "",
            "sem": "",
            "n_unique": "",
            "theory_I_minus_au": "",
            "note": meta,
        }
    )
    for i in range(len(phi_th)):
        for series, cond, y in (
            ("I_minus_no_object", "theory", y_clear[i]),
            ("I_minus_rabbit70", "theory", y_rb[i]),
        ):
            rows.append(
                {
                    "row": "theory_sample",
                    "series": series,
                    "folder": "",
                    "condition": cond,
                    "dz_um": "",
                    "phi_z_rad": phi_th[i],
                    "mean": "",
                    "sd": "",
                    "sem": "",
                    "n_unique": "",
                    "theory_I_minus_au": y,
                    "note": "",
                }
            )
    for cond in ("no_object", "rabbit"):
        order = exp[cond]["folder_order"]
        for j, folder in enumerate(order):
            dz = FOLDER_DZ_UM[folder]
            rows.append(
                {
                    "row": "experiment_point",
                    "series": "manuscript_exp",
                    "folder": folder,
                    "condition": cond,
                    "dz_um": dz,
                    "phi_z_rad": wrapped_phase_rad(dz),
                    "mean": exp[cond]["roi_mean_adu"][j],
                    "sd": exp[cond]["roi_mean_sd_adu"][j],
                    "sem": exp[cond]["roi_mean_sem_adu"][j],
                    "n_unique": int(
                        sum(
                            1
                            for a in audit
                            if a["condition"] == cond
                            and a["phase_folder"] == folder
                            and a["included_in_statistics"]
                        )
                    ),
                    "theory_I_minus_au": "",
                    "note": f"n_frames_unique_sha={int(sum(1 for a in audit if a['condition']==cond and a['phase_folder']==folder and a['included_in_statistics']))}",
                }
            )

    fields = list(rows[0].keys())
    with (ART / "Figure5a_Final_Data.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    audit_fields = [
        "condition",
        "phase_folder",
        "delta_z_um",
        "phi_z_rad",
        "source_path",
        "sha256",
        "roi_mean_adu_per_pixel",
        "duplicate_in_folder",
        "included_in_statistics",
    ]
    with (ART / "Figure5a_Experimental_Frame_Audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        w.writerows({k: a[k] for k in audit_fields} for a in audit)

    v_no = sampled_v(list(exp["no_object"]["roi_mean_adu"]))
    v_rb = sampled_v(list(exp["rabbit"]["roi_mean_adu"]))
    ts = datetime.now(timezone.utc).isoformat()
    report = f"""# Figure 5(a) — manuscript final (updated TIFFs)

Generated: {ts}  
Output folder: `article figures/`

## Verification

| Check | Result |
|-------|--------|
| Experimental TIFF root | `{settings.exp_root.relative_to(ROOT)}` |
| Intensities recomputed from TIFFs this run | **YES** |
| Prior experimental CSV / cache reused | **NO** |
| Theory | Joint-SPDC **{THEORY_OUTPUT_PORT}**, **δL_mm={DELTA_L_MM}**, theory ROI **{THEORY_ROI_PIXELS}** px |
| Experimental extraction | Fixed circle center **(227, 209)**, radius **{ROI_RADIUS_PX}** px (**{ROI_PIXEL_COUNT}** px) — documented here only |
| Phase offsets / intensity fitting | **None** |
| Phase folders | `{folders}` (dark/ref excluded) |

## Frame counts (unique SHA-256 per folder, after duplicate exclusion)

| Folder | Δz (µm) | φ_z | n (no-object) | n (rabbit) | No-object mean ± SEM | Rabbit mean ± SEM |
|--------|--------:|----:|--------------:|-----------:|---------------------:|------------------:|
"""
    for folder in folders:
        dz = FOLDER_DZ_UM[folder]
        j = exp["no_object"]["folder_order"].index(folder)
        nn = int(
            sum(
                1
                for a in audit
                if a["condition"] == "no_object"
                and a["phase_folder"] == folder
                and a["included_in_statistics"]
            )
        )
        nr = int(
            sum(
                1
                for a in audit
                if a["condition"] == "rabbit"
                and a["phase_folder"] == folder
                and a["included_in_statistics"]
            )
        )
        report += (
            f"| `{folder}` | {dz:.0f} | {wrapped_phase_rad(dz):.6f} | {nn} | {nr} | "
            f"{exp['no_object']['roi_mean_adu'][j]:.2f} ± {exp['no_object']['roi_mean_sem_adu'][j]:.2f} | "
            f"{exp['rabbit']['roi_mean_adu'][j]:.2f} ± {exp['rabbit']['roi_mean_sem_adu'][j]:.2f} |\n"
        )

    report += f"""
## Sampled experimental visibility (4 folder means)

- no-object: **{v_no:.6f}**
- rabbit: **{v_rb:.6f}**

Theory visibility (OPD0 **I_minus** sampled curve): no-object ≈ **1.000**, rabbit ≈ **0.408**.

## Shared code path

Active convention module: `tools/c_lab_manuscript_convention.py`  
Figure 5(b) assembly and mirror-drift experiment hooks: `tools/c_lab_figure5_final_theory_experiment_intensity.py`, `tools/c_lab_figure5_axis_variants.py`, `tools/c_lab_figure5_final_article.py`.

Figure face labels omit experimental aperture radius (manuscript presentation rule).
"""
    (ART / "Figure5a_Final_Report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {ART}")
    print(f"folders={folders} audit_files={len(audit)} V_no={v_no:.4f} V_rb={v_rb:.4f}")


if __name__ == "__main__":
    main()
