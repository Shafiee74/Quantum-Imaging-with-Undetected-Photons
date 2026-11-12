#!/usr/bin/env python3
"""Figure 5 panel-(a) top-axis variants (same data and image panels)."""
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

import numpy as np
import tifffile

from tools.c_lab_figure5_final_theory_experiment_intensity import (
    EVIDENCE_ROOT as _PARENT_EVIDENCE,
    EXP_ROI,
    THEORY_EVIDENCE,
    THEORY_TIFFS,
    ExpPlotOptions,
    build_full_figure,
    dz_eq_um_from_phase,
    experimental_stats,
    load_positions,
    sha256_file,
    shared_display_vmax,
    wrapped_phase_rad,
)
from tools.c_lab_manuscript_convention import theory_csv_dict

HIGH_SIGNAL_FOLDER = "pi"
LOW_SIGNAL_FOLDER = "pi2"

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure5_axis_variants"
PLOT_OPTS = ExpPlotOptions.exclude_2pi_only()
DZ_TABLE = (0.0, 25.0, 50.0, 75.0, 100.0)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_assembly(theory_dir: Path) -> dict[str, Any]:
    theory_csv = theory_csv_dict()
    fit_payload = json.loads((theory_dir / "FIGURE5_THEORY_FIT_REPORT.json").read_text(encoding="utf-8"))
    summary = json.loads((theory_dir / "FIGURE5_THEORY_SUMMARY.json").read_text(encoding="utf-8"))
    exp, per_frame = experimental_stats()
    pos = dict(exp.pop("_manuscript_position_map"))
    pos = {k: v for k, v in pos.items() if k not in PLOT_OPTS.exclude_folders}
    exp_roi = EXP_ROI
    folders = list(pos.keys())
    idx_hi = folders.index(HIGH_SIGNAL_FOLDER)
    idx_lo = folders.index(LOW_SIGNAL_FOLDER)
    hi_folder = HIGH_SIGNAL_FOLDER
    lo_folder = LOW_SIGNAL_FOLDER

    def _exp_state(folder: str, index: int, rule: str) -> dict[str, Any]:
        dz = float(pos[folder])
        phi = wrapped_phase_rad(dz)
        return {
            "selection_rule": rule,
            "phase_folder": folder,
            "mirror_um": dz,
            "wrapped_phase_rad": phi,
            "wrapped_phase_deg": float(np.degrees(phi)),
            "index": index,
            "conditions": {},
        }

    selection: dict[str, Any] = {
        "theory_source_evidence": str(theory_dir.relative_to(ROOT)),
        "plotting_adjustments": {
            "exclude_folders_from_panel_a": ["2pi"],
            "experimental_points_plotted_um": [0.0, 25.0, 50.0, 75.0],
            "note_100um": (
                "Excluded because neither experimental condition contains a 2pi/100 µm folder; "
                "the position exists only in the nominal positions JSON."
            ),
        },
        "theory_constructive_like": summary["constructive_like"],
        "theory_destructive_like": summary["destructive_like"],
        "experimental_high_signal": _exp_state(
            hi_folder,
            idx_hi,
            f"fixed folder {HIGH_SIGNAL_FOLDER} (50 µm high-signal representative)",
        ),
        "experimental_low_signal": _exp_state(
            lo_folder,
            idx_lo,
            f"fixed folder {LOW_SIGNAL_FOLDER} (25 µm low-signal; not 3pi2/75 µm)",
        ),
    }
    for state_key, idx in (("experimental_high_signal", idx_hi), ("experimental_low_signal", idx_lo)):
        for cond in ("no_object", "rabbit"):
            row = exp[cond]["phase_rows"][idx]
            selection[state_key]["conditions"][cond] = {
                "roi_mean_adu_per_pixel": float(exp[cond]["roi_mean_adu"][idx]),
                "roi_mean_sem_adu_per_pixel": float(exp[cond]["roi_mean_sem_adu"][idx]),
                "n_frames": int(row["n_frames"]),
            }

    theory_paths = [theory_dir / name for name in THEORY_TIFFS]
    theory_u16 = [tifffile.imread(p) for p in theory_paths]
    theory_roi_native = np.load(theory_dir / "FIGURE5_THEORY_ROI_MASK.npy").astype(bool)
    theory_roi_display = np.rot90(theory_roi_native, k=2)[302:721, 284:739]
    if theory_roi_display.shape != theory_u16[0].shape:
        raise ValueError("Theory ROI crop does not match displayed theory TIFF dimensions")
    selection["theory_image_sha256"] = {p.name: sha256_file(p) for p in theory_paths}

    exp_images = [
        exp["no_object"]["phase_mean_images"][idx_hi],
        exp["rabbit"]["phase_mean_images"][idx_hi],
        exp["no_object"]["phase_mean_images"][idx_lo],
        exp["rabbit"]["phase_mean_images"][idx_lo],
    ]
    theory_vmax = shared_display_vmax([t.astype(np.float64) for t in theory_u16])
    exp_vmax = shared_display_vmax(exp_images)

    return {
        "theory_csv": theory_csv,
        "fit_payload": fit_payload,
        "summary": {
            "constructive_like": summary["constructive_like"],
            "destructive_like": summary["destructive_like"],
        },
        "exp": exp,
        "experimental_per_frame": per_frame,
        "pos": pos,
        "theory_u16": theory_u16,
        "exp_images": exp_images,
        "exp_roi": exp_roi,
        "theory_roi_display": theory_roi_display,
        "theory_vmax": theory_vmax,
        "exp_vmax": exp_vmax,
        "selection": selection,
    }


def conversion_rows() -> list[dict[str, float]]:
    rows = []
    for dz in DZ_TABLE:
        phi = wrapped_phase_rad(dz)
        rows.append(
            {
                "dz_exp_um": dz,
                "phi_wrapped_rad": phi,
                "phi_wrapped_deg": float(np.degrees(phi)),
                "dz_eq_um_one_period": dz_eq_um_from_phase(phi),
                "plotted_in_panel_a": dz != 100.0,
            }
        )
    return rows


def run(output: Path | None = None, theory_dir: Path = THEORY_EVIDENCE) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    payload = load_assembly(theory_dir)
    conv = conversion_rows()

    with (out / "AXIS_CONVERSION_TABLE.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dz_exp_um",
                "phi_wrapped_rad",
                "phi_wrapped_deg",
                "dz_eq_um_one_period",
                "plotted_in_panel_a",
            ],
        )
        w.writeheader()
        w.writerows(conv)

    common = dict(
        theory=payload["theory_csv"],
        fit=payload["fit_payload"],
        summary=payload["summary"],
        exp=payload["exp"],
        pos=payload["pos"],
        theory_u16=payload["theory_u16"],
        exp_images=payload["exp_images"],
        theory_vmax=payload["theory_vmax"],
        exp_vmax=payload["exp_vmax"],
        selection=payload["selection"],
        plot_opts=PLOT_OPTS,
    )

    build_full_figure(
        out / "FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.png",
        out / "FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.pdf",
        top_axis="equiv_one_period",
        suptitle="Figure 5 — Variant A: phase + equivalent one-period displacement (C_lab Mi=2)",
        **common,
    )
    build_full_figure(
        out / "FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.png",
        out / "FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.pdf",
        top_axis="physical_wrapped_labels",
        suptitle="Figure 5 — Variant B: phase + physical displacement labels (C_lab Mi=2)",
        **common,
    )

    report = {
        "variant_a": {
            "file_png": "FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.png",
            "bottom_axis": "interferometric phase 0 to 2pi rad",
            "top_axis": "equivalent mirror displacement dz_eq 0 to 0.775 um (one 2pi period)",
            "rigor": "preferred for physical consistency with theory fringe period",
        },
        "variant_b": {
            "file_png": "FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.png",
            "bottom_axis": "interferometric phase 0 to 2pi rad",
            "top_axis": "physical dz_exp tick labels 0-100 um at wrapped phase positions",
            "readability": "preferred for direct experimental mirror-step reporting",
        },
        "shared": {
            "theory_evidence": str(theory_dir),
            "panel_a_experimental_points_um": [0, 25, 50, 75],
            "excluded_from_panel_a_um": [100],
            "75um_point": "3pi2 folder, true 75 um displacement, no 85 um remap",
            "physics_or_image_selection_changed": False,
            "production_files_modified": False,
        },
        "conversion_table_csv": "AXIS_CONVERSION_TABLE.csv",
        "theory_tiff_sha256": payload["selection"]["theory_image_sha256"],
    }
    (out / "AXIS_VARIANTS_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    conv_md = "\n".join(
        f"| {r['dz_exp_um']:.0f} | {r['phi_wrapped_rad']:.6f} | {r['phi_wrapped_deg']:.3f} | "
        f"{r['dz_eq_um_one_period']:.6f} | {'yes' if r['plotted_in_panel_a'] else 'no'} |"
        for r in conv
    )
    (out / "AXIS_VARIANTS_REPORT.md").write_text(
        f"""# Figure 5 axis variants

Evidence: `{out}`

## Summary
- **Variant A** (`FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.png`): bottom axis interferometric phase φ ∈ [0, 2π]; top axis **equivalent** displacement Δz_eq ∈ [0, 0.775 µm] for one theoretical 2π period (λ_i = 1.55 µm). This matches the theory curve domain and is the more **physically rigorous** choice for comparing to the simulation phase scan.
- **Variant B** (`FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.png`): same bottom axis; top axis shows **physical** mirror displacement labels (0–100 µm) placed at each step's **wrapped** phase position. More **directly readable** for experimental mirror reporting; points remain at wrapped φ (see note on figure).

Both variants use **identical** theory curves, theory fit, four theory TIFFs, four experimental mean images, ROI extraction, and Mi=2 production baseline. Panel (a) plots experimental points at **0, 25, 50, 75 µm** only (100 µm excluded). The **75 µm** (`3pi2`) point is the experimental 3π/2 step at true 75 µm (no 85 µm reinterpretation).

No production physics, YAML, crop, ROI, rabbit orientation, or image selection logic was changed.

## Axis conversion table

| Δz_exp (µm) | φ_wrapped (rad) | φ_wrapped (deg) | Δz_eq (µm) | Plotted |
|-------------|-----------------|-----------------|------------|---------|
{conv_md}

## Paper recommendation
Use **Variant A** in the main paper figure when the narrative ties experiment to the **single-period theory fringe** and dual y-axes (a.u. vs ADU). Use **Variant B** in supplementary material or when the text emphasizes **nominal mirror steps** (0–100 µm ladder) while still plotting wrapped phase abscissa.
""",
        encoding="utf-8",
    )

    print(f"Created: {out}")
    for name in (
        "FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.png",
        "FIGURE5_VARIANT_A_PHASE_PLUS_EQUIV_DISPLACEMENT.pdf",
        "FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.png",
        "FIGURE5_VARIANT_B_PHASE_PLUS_PHYSICAL_DISPLACEMENT.pdf",
        "AXIS_CONVERSION_TABLE.csv",
        "AXIS_VARIANTS_REPORT.md",
        "AXIS_VARIANTS_REPORT.json",
    ):
        print(f"  - {out / name}")

    print("\nAxis conversion summary:")
    for r in conv:
        print(
            f"  dz_exp={r['dz_exp_um']:5.0f} µm  phi={r['phi_wrapped_deg']:7.3f}°  "
            f"dz_eq={r['dz_eq_um_one_period']:.6f} µm  plotted={'yes' if r['plotted_in_panel_a'] else 'no'}"
        )

    print(
        "\nRecommendation: Variant A for main text (phase-consistent with theory); "
        "Variant B for readability of experimental mirror steps."
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--theory-dir", type=Path, default=THEORY_EVIDENCE)
    args = p.parse_args()
    run(args.output, theory_dir=args.theory_dir)


if __name__ == "__main__":
    main()
