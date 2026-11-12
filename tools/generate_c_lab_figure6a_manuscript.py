#!/usr/bin/env python3
"""Manuscript Figure 6(a) loader and legacy extractor.

The production reproduction entry point is ``tools/reproduce_figure6a.py``.
This module still supplies the TIFF mean, display crop, and forward-model
call used by that entry point. Its own ``main`` no longer writes into
``article figures`` by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.c_lab_display_zoom import apply_fixed_crop
from setup_params import resolve_setup_selection
from tools.c_lab_figure6a_center_spacing import extract_series
from tools.c_lab_figure6_lens_displacement_validation import (
    TRANS_DX_UM,
    X0_MM,
    group_tiffs,
)
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.c_lab_l2_ms1_mi1_calculated_dk0_rabbit_images import N
from tools.c_lab_manuscript_convention import DELTA_L_MM, THEORY_OUTPUT_PORT
from tools.c_lab_rabbit_object_orientation_correction import DISPLAY_ROT90_K
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)
from tools.z2.physics import ref_from_T

ART = ROOT / "article figures"
EXP_ROOT = ROOT / "exp data" / "lens drift" / "witout object" / "T drift"
TRANS_FOLDER = EXP_ROOT
X0_MM_TRANS = X0_MM
PIXEL_UM = 13.0
OPERATING_PHASE_RAD = math.pi
MM_WIDTH = 180.0
DPI_PNG = 600
THEORY_DISPLAY_BOUNDS = {"row0": 302, "row1": 721, "col0": 284, "col1": 739}
SIMULATION_LABEL = "Simulation"
EXPERIMENTAL_LABEL = "Experimental"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_stack_deduped(paths: list[Path]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Temporal mean with SHA-256 dedupe within a displacement group."""
    audit: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    seen: set[str] = set()
    for p in sorted(paths):
        digest = sha256_file(p)
        dup = digest in seen
        if not dup:
            seen.add(digest)
            frames.append(tifffile.imread(p).astype(np.float64))
        audit.append(
            {
                "source_path": str(p.resolve()),
                "filename": p.name,
                "sha256": digest,
                "duplicate_in_group": int(dup),
                "included_in_mean": int(not dup),
            }
        )
    if not frames:
        raise ValueError(f"No unique frames in group: {paths[:1]}")
    stack = np.stack(frames, axis=0)
    return np.mean(stack, axis=0), audit


def theory_display_iminus(native: np.ndarray) -> np.ndarray:
    oriented = np.rot90(np.asarray(native, dtype=np.float64), k=DISPLAY_ROT90_K)
    return apply_fixed_crop(oriented, THEORY_DISPLAY_BOUNDS)


def run_theory_iminus(setup_base, dk: DeltaKCalibrationState, *, dx_um: float) -> np.ndarray:
    setup = setup_base.overridden(
        delta_L_mm=DELTA_L_MM,
        relative_interferometric_phase=OPERATING_PHASE_RAD,
        enable_lens_rings=True,
        lens_shift_x_um=float(dx_um),
        lens_shift_y_um=0.0,
        delta_z_um=0.0,
        use_derived_K_defocus=False,
        defocus_span_px=0.0,
        K_defocus_px_per_um=0.0,
    )
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    case = run_forward_case(setup, ref, dk, grid_n=N)
    return theory_display_iminus(np.asarray(case[THEORY_OUTPUT_PORT], dtype=np.float64))


def verify_displacement_groups(groups: dict[float, list[Path]]) -> list[float]:
    missing = [dx for dx in TRANS_DX_UM if round(dx, 6) not in groups]
    if missing:
        raise FileNotFoundError(
            f"Missing transverse displacement folders/files for Δx (µm): {missing}. "
            f"Found keys: {sorted(groups.keys())}"
        )
    return [float(dx) for dx in TRANS_DX_UM]


def load_transverse_manuscript() -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], list[dict]]:
    if not TRANS_FOLDER.is_dir():
        raise FileNotFoundError(TRANS_FOLDER)
    groups = group_tiffs(TRANS_FOLDER, ref_mm=X0_MM_TRANS, rel_scale=1000.0)
    dx_list = verify_displacement_groups(groups)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    theory: dict[float, np.ndarray] = {}
    exp: dict[float, np.ndarray] = {}
    file_audit: list[dict] = []

    for dx in dx_list:
        key = round(dx, 6)
        exp_img, aud = mean_stack_deduped(groups[key])
        exp[dx] = exp_img
        theory[dx] = run_theory_iminus(setup, dk, dx_um=dx)
        for a in aud:
            file_audit.append({**a, "dx_um": dx, "delta_x_um": dx})

    return theory, exp, file_audit


def build_figure6a(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray],
    th_rows: list[dict[str, Any]],
    ex_rows: list[dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    dx_list = [d for d in TRANS_DX_UM if d in theory]
    w_in = MM_WIDTH / 25.4
    fig = plt.figure(figsize=(w_in, w_in * 0.76))
    gs_img = GridSpec(
        2,
        len(dx_list),
        height_ratios=[1, 1],
        hspace=0.025,
        wspace=0.045,
        left=0.145,
        right=0.995,
        top=0.925,
        bottom=0.485,
    )
    gs_plot = GridSpec(1, 1, left=0.145, right=0.995, top=0.435, bottom=0.105)

    t_v = max(np.percentile(theory[d], 99.5) for d in dx_list)
    e_v = max(np.percentile(exp[d], 99.5) for d in dx_list)

    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs_img[0, j])
        ax.imshow(theory[dx], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δx = {dx:.0f} µm", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.24,
                0.5,
                SIMULATION_LABEL,
                transform=ax.transAxes,
                va="center",
                ha="right",
                fontsize=10,
            )

        ax = fig.add_subplot(gs_img[1, j])
        ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.24,
                0.5,
                EXPERIMENTAL_LABEL,
                transform=ax.transAxes,
                va="center",
                ha="right",
                fontsize=10,
            )

    axp = fig.add_subplot(gs_plot[0, 0])
    xs = list(dx_list)
    y_sim = [float(r["k_plotted_cycles_per_mm"]) for r in th_rows]
    y_ex = [float(r["k_plotted_cycles_per_mm"]) for r in ex_rows]
    axp.plot(
        xs,
        y_sim,
        marker="s",
        linestyle="--",
        color=REFERENCE_TEAL,
        label=SIMULATION_LABEL,
        markersize=6,
        markeredgecolor="0.25",
        markeredgewidth=0.6,
        linewidth=1.35,
    )
    axp.plot(
        xs,
        y_ex,
        marker="D",
        linestyle=(0, (2.5, 2.0, 6.0, 2.0)),
        color=REFERENCE_GOLD,
        label=EXPERIMENTAL_LABEL,
        markersize=6,
        markeredgecolor="0.25",
        markeredgewidth=0.6,
        linewidth=1.35,
    )
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)", labelpad=4)
    axp.set_ylabel("Spatial frequency (cycles mm$^{-1}$)", labelpad=6)
    axp.set_xticks(xs)
    axp.legend(frameon=False, loc="upper left")
    style_reference_axes(axp)
    finite = [v for v in y_sim + y_ex if math.isfinite(v)]
    y_max = max(finite) if finite else 1.0
    axp.set_xlim(-8.0, max(xs) * 1.02 if xs else 180.0)
    axp.set_ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)
    axp.margins(x=0, y=0)

    fig.savefig(out_png, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_data_csv(path: Path, th_rows: list[dict], ex_rows: list[dict]) -> None:
    fields = [
        "modality",
        "dx_um",
        "n_fringe_centers",
        "k_center_spacing_cycles_per_mm",
        "k_plotted_cycles_per_mm",
        "mean_spacing_mm",
        "spacing_std_mm",
        "spacing_cv",
        "n_spacings",
        "center_spacing_measurable",
        "measurement_status",
        "fringe_normal_deg",
        "period_bins_est",
        "profile_pre_smooth_sigma_px",
        "k_fft_diagnostic_cycles_per_mm",
        "peak_prominence",
        "peak_min_distance_bins",
    ]
    rows: list[dict] = []
    for modality, series in ((SIMULATION_LABEL, th_rows), (EXPERIMENTAL_LABEL, ex_rows)):
        for r in series:
            rows.append(
                {
                    "modality": modality,
                    "dx_um": r["dx_um"],
                    "n_fringe_centers": r["n_centers"],
                    "k_center_spacing_cycles_per_mm": r["k_center_spacing_cycles_per_mm"],
                    "k_plotted_cycles_per_mm": r["k_plotted_cycles_per_mm"],
                    "mean_spacing_mm": r["mean_spacing_mm"],
                    "spacing_std_mm": r["spacing_std_mm"],
                    "spacing_cv": r["spacing_cv"],
                    "n_spacings": r["n_spacings"],
                    "center_spacing_measurable": r["center_spacing_measurable"],
                    "measurement_status": r["measurement_status"],
                    "fringe_normal_deg": r["fringe_normal_deg"],
                    "period_bins_est": r["period_bins_est"],
                    "profile_pre_smooth_sigma_px": r["profile_pre_smooth_sigma_px"],
                    "k_fft_diagnostic_cycles_per_mm": r["k_fft_diagnostic_cycles_per_mm"],
                    "peak_prominence": r["peak_prominence"],
                    "peak_min_distance_bins": r["peak_min_distance_bins"],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_audit_csv(path: Path, audit: list[dict]) -> None:
    fields = list(audit[0].keys()) if audit else ["source_path"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(audit)


def write_report(
    path: Path,
    th_rows: list[dict],
    ex_rows: list[dict],
    sim_dx0_mean: float,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    mapping_lines = []
    for dx in TRANS_DX_UM:
        abs_mm = X0_MM_TRANS + dx / 1000.0
        mapping_lines.append(
            f"- Δx = **{dx:.0f} µm** ← stage filename absolute **{abs_mm:.3f} mm** "
            f"(relative to x₀ = {X0_MM_TRANS} mm)"
        )

    def summary_table(rows: list[dict], label: str) -> str:
        lines = [f"### {label}", "", "| Δx (µm) | centers | k (cycles/mm) | status |", "|--------:|--------:|--------------:|--------|"]
        for r in rows:
            k = r["k_center_spacing_cycles_per_mm"]
            ks = f"{k:.4f}" if math.isfinite(float(k)) else "—"
            lines.append(
                f"| {int(r['dx_um'])} | {r['n_centers']} | {ks} | {r['measurement_status']} |"
            )
        return "\n".join(lines)

    text = f"""# Figure 6(a) — manuscript final

Generated: {ts}  
Output folder: `article figures/`

## Physical / modeling convention (aligned with Figure 5)

| Item | Value |
|------|--------|
| Forward model | Joint-SPDC |
| Interference port | **{THEORY_OUTPUT_PORT}** |
| Michelson OPD offset | **delta_L_mm = {DELTA_L_MM}** (in memory) |
| Operating phase | **φ = π rad** (fixed for all Δx) |
| Transverse scan | idler relay lens Δx at fixed Δz = 0 |
| Experimental condition | **without object** (`witout object`) |
| Cached frequency CSV reused | **NO** — TIFFs read fresh |

Simulation mean display **I_minus** at Δx = 0 (φ = π): **{sim_dx0_mean:.6g}** (bright reference check).

## Experimental source

- **Root directory:** `{TRANS_FOLDER.resolve()}`
- **Displacement list (µm):** {list(TRANS_DX_UM)}
- **Reference stage position x₀:** {X0_MM_TRANS} mm
- **Folder mapping (verified from filenames):**

{chr(10).join(mapping_lines)}

## Diagonal stripes

Fine diagonal patterns in experimental frames (including Δx = 0) are **camera/filter artifacts**, not displacement-induced fringes. They can bias orientation and peak picking; images are shown **unmodified**.

## Frequency extraction (manuscript curve)

**Primary:** fringe-center spacing along the fringe-normal profile (scale-aware pre-smoothing σ ≤ 6 px; simulation Δx ≤ 30 µm uses σ = 0 and relative prominence ≥ 3% for weak side lobes).

**Diagnostic only (CSV):** envelope-corrected 1D FFT on the same profile — **not plotted**.

**Plotted ordinate:** at Δx = 0, **0 cycles/mm** (defined reference — no displacement-induced fringe modulation). For Δx > 0, **k = 1 / mean center spacing** when ≥ 2 centers are detected; otherwise the point is omitted (NaN in CSV, gap in line).

Pixel pitch: **{PIXEL_UM} µm**.

{summary_table(ex_rows, "Experimental")}

{summary_table(th_rows, "Simulation")}

## Interpretation

**Semi-quantitative** comparison — morphology and trend; spacing variability and artifact contamination limit strong quantitative claims.

## Validation checklist

| Check | Result |
|-------|--------|
| Joint-SPDC I_minus, δL=0, φ=π all sim panels | **YES** |
| Six mechanical Δx values preserved | **YES** |
| Experimental TIFFs read directly + SHA-256 dedupe | **YES** |
| Frequencies from center spacing (not analytical) | **YES** |
| FFT not on figure | **YES** |
| Figures 5(a), 5(b), 6(b) untouched by this script | **YES** |
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Figure6a_Legacy_Generator_Output",
        help="Legacy extractor output. Manuscript Figure 6(a) is tools/reproduce_figure6a.py.",
    )
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_graphics_style()
    theory, exp, file_audit = load_transverse_manuscript()

    dx_list = list(TRANS_DX_UM)
    th_rows = extract_series(theory, dx_list, "simulation")
    ex_rows = extract_series(exp, dx_list, "experiment")

    sim_dx0_mean = float(np.mean(theory[0.0]))

    png = out_dir / "Figure6a_Final.png"
    pdf = out_dir / "Figure6a_Final.pdf"
    build_figure6a(theory, exp, th_rows, ex_rows, png, pdf)

    write_data_csv(out_dir / "Figure6a_Final_Data.csv", th_rows, ex_rows)
    write_audit_csv(out_dir / "Figure6a_Experimental_File_Audit.csv", file_audit)
    write_report(out_dir / "Figure6a_Final_Report.md", th_rows, ex_rows, sim_dx0_mean)

    print("Figure 6(a) ->", out_dir)
    print("EXP_ROOT:", TRANS_FOLDER.resolve())
    print("Sim dx0 mean I_minus display:", sim_dx0_mean)
    for label, rows in ((SIMULATION_LABEL, th_rows), (EXPERIMENTAL_LABEL, ex_rows)):
        print(label, "centers:", [(int(r["dx_um"]), r["n_centers"]) for r in rows])
        print(label, "k plotted:", [(int(r["dx_um"]), r["k_plotted_cycles_per_mm"]) for r in rows])


if __name__ == "__main__":
    main()
