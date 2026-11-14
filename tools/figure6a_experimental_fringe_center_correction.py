#!/usr/bin/env python3
"""Figure 6(a) experimental fringe-center frequency correction (Table S2 path).

Public entry point cited in Supplementary Table S2. Applies the same bright-
fringe center spacing rule used for the simulated correction
(``tools/figure6a_simulated_fringe_frequency_correction.py``) to the local
experimental transverse TIFF stacks and writes:

    Figure6a_Experimental_Fringe_Center_Correction/Experimental_Frequency_Comparison.csv

Original laboratory EMCCD datasets are **not** shipped on GitHub. This script
requires a local experimental TIFF root (default:
``exp data/lens drift/witout object/T drift``). Without that private data it
exits with a clear message and writes no fabricated frequencies.

Simulation-only Figure 6(a) reproduction does not depend on this script;
see ``tools/reproduce_figure6a.py``. Downstream pairing that compares
corrected simulation against corrected experiment loads the CSV produced here.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, X0_MM, group_tiffs
from tools.figure6a_simulated_fringe_frequency_correction import bright_crests
from tools.generate_c_lab_figure6a_manuscript import EXP_ROOT, mean_stack_deduped

OUT = ROOT / "Figure6a_Experimental_Fringe_Center_Correction"
CSV_NAME = "Experimental_Frequency_Comparison.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_experimental_rows(
    exp_root: Path,
    *,
    dx_list: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not exp_root.is_dir():
        raise FileNotFoundError(
            f"Experimental TIFF root not found: {exp_root}\n"
            "Private EMCCD stacks are intentionally not published. Place the "
            "local laboratory transverse TIFF tree at this path (or pass "
            "--exp-root) before regenerating Experimental_Frequency_Comparison.csv."
        )
    groups = group_tiffs(exp_root, ref_mm=X0_MM, rel_scale=1000.0)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for dx in dx_list:
        key = round(float(dx), 6)
        if key not in groups:
            raise KeyError(f"No experimental TIFF group for Δx={dx} µm under {exp_root}")
        image, file_audit = mean_stack_deduped(groups[key])
        for item in file_audit:
            item = dict(item)
            item["lens_displacement_um"] = float(dx)
            audit.append(item)
        det = bright_crests(image, float(dx))
        n_centers = len(det["centers"])
        k = float(det["k"])
        if float(dx) == 0.0:
            k_periodic = float("nan")
            k_plotted = 0.0
            status = "reference_zero_no_fringe_modulation"
        elif math.isfinite(k) and n_centers >= 2:
            k_periodic = k
            k_plotted = k
            status = "measured"
        else:
            k_periodic = float("nan")
            k_plotted = float("nan")
            status = "not_measurable"
        rows.append(
            {
                "lens_displacement_um": float(dx),
                "corrected_n_centers": int(n_centers),
                "corrected_periodic_frequency_cycles_per_mm": k_periodic,
                "corrected_plotted_frequency_cycles_per_mm": k_plotted,
                "mean_spacing_mm": det.get("mean_spacing_mm", float("nan")),
                "spacing_std_mm": det.get("spacing_std_mm", float("nan")),
                "spacing_cv": det.get("spacing_cv", float("nan")),
                "rejected_weak_crests": len(det.get("rejected") or []),
                "measurement_status": status,
                "extractor": "bright_crests_rel_prom_0.25",
                "pixel_um": 13.0,
            }
        )
    return rows, audit


def write_report(path: Path, rows: list[dict[str, Any]], exp_root: Path) -> None:
    lines = [
        "# Figure 6(a) experimental fringe-center correction",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Entry point: `tools/figure6a_experimental_fringe_center_correction.py`",
        f"Experimental root: `{exp_root.as_posix()}`",
        "",
        "Original EMCCD datasets are not published with the repository.",
        "Frequencies use mean adjacent bright-center spacing (25% relative ",
        "prominence gate), matching the simulated correction rule.",
        "",
        "| Δx (µm) | n centers | k (cycles/mm) | status |",
        "|---:|---:|---:|---|",
    ]
    for row in rows:
        k = row["corrected_periodic_frequency_cycles_per_mm"]
        ks = f"{float(k):.6f}" if k != "" and math.isfinite(float(k)) else "—"
        lines.append(
            f"| {row['lens_displacement_um']:.0f} | {row['corrected_n_centers']} | "
            f"{ks} | {row['measurement_status']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=EXP_ROOT,
        help="Local experimental TIFF root (not shipped on GitHub).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Directory for Experimental_Frequency_Comparison.csv",
    )
    args = parser.parse_args(argv)
    try:
        rows, audit = extract_experimental_rows(
            args.exp_root, dx_list=[float(d) for d in TRANS_DX_UM]
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / CSV_NAME, rows)
    write_csv(out_dir / "Experimental_File_Audit.csv", audit)
    write_report(out_dir / "Experimental_Fringe_Center_Correction_Report.md", rows, args.exp_root)
    print("Wrote", out_dir / CSV_NAME)
    for row in rows:
        print(
            row["lens_displacement_um"],
            "n",
            row["corrected_n_centers"],
            "k",
            row["corrected_periodic_frequency_cycles_per_mm"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
