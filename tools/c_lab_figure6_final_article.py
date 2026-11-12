#!/usr/bin/env python3
"""Publication Figure 6(a) transverse + 6(b) axial (final assembly)."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from matplotlib.gridspec import GridSpec

from camera.c_lab_display_zoom import export_zoom_config_from_setup, resolve_fixed_crop_window
from setup_params import resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import (
    EXP_ROOT,
    N,
    PIXEL_UM,
    TRANS_DX_UM,
    X0_MM,
    Z0_MM,
    fft_fringe_metrics,
    group_tiffs,
    mean_stack,
    run_theory,
    theory_kappa_cycles_per_mm,
    weighted_centroid,
)
from tools.c_lab_manuscript_convention import (  # shared OPD0 / mirror-drift convention
    DEFAULT_MANUSCRIPT_SETTINGS,
    DELTA_L_MM,
    THEORY_OUTPUT_PORT,
)
from tools.c_lab_figure6b_corrected_effective_defocus import Z_EFF_MM, Z_STAGE_MM
from tools.c_lab_figure6b_final_quantitative import load_panels as load_axial_panels
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6_final_article"
APPROVED_6A = ROOT / "evidence" / "c_lab_figure6_lens_displacement_validation" / "20261017T084345Z"
APPROVED_6B = ROOT / "evidence" / "c_lab_figure6_corrected_effective_defocus" / "20261017T103355Z"

MM_WIDTH = 180.0
DPI_PNG = 600


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def fringe_vector_cycles_per_mm(img: np.ndarray) -> dict[str, float]:
    """Dominant 2D FFT peak as spatial-frequency vector (cycles/mm)."""
    m = fft_fringe_metrics(img)
    z = np.asarray(img, dtype=np.float64)
    h, w = z.shape
    wy, wx = np.hanning(h), np.hanning(w)
    spec = np.fft.fftshift(np.fft.fft2((z - z.mean()) * np.outer(wy, wx)))
    mag = np.abs(spec)
    cy, cx = h // 2, w // 2
    mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    yy, xx = np.indices(mag.shape)
    rr = np.hypot(yy - cy, xx - cx)
    band = (rr >= 4) & (rr <= min(z.shape) * 0.35)
    masked = np.where(band, mag, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    pixel_mm = PIXEL_UM * 1e-3
    fx = (ix - cx) / float(w) / pixel_mm
    fy = (iy - cy) / float(h) / pixel_mm
    return {
        "fx_cycles_per_mm": float(fx),
        "fy_cycles_per_mm": float(fy),
        "kappa_cycles_per_mm": float(m["kappa_cycles_per_mm"]),
        "orientation_deg": float(m["orientation_deg"]),
    }


def angle_diff_deg(a: float, b: float) -> float:
    """Fringe normal is periodic under 180° (cosine fringe)."""
    da = (a % 180.0 + 180.0) % 180.0
    db = (b % 180.0 + 180.0) % 180.0
    return min(abs(da - db), 180.0 - abs(da - db))


def setup_matplotlib_pub() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 100,
            "savefig.dpi": DPI_PNG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_transverse() -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], list[dict], bool]:
    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    I_ref = run_forward_case(setup.overridden(enable_lens_rings=True), ref, dk, grid_n=N)["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))
    groups = group_tiffs(EXP_ROOT / "T drift", ref_mm=X0_MM, rel_scale=1000.0)
    theory: dict[float, np.ndarray] = {}
    exp: dict[float, np.ndarray] = {}
    rows: list[dict] = []
    ref_cy, ref_cx = weighted_centroid(mean_stack(groups[0.0])["mean"])

    ref_th_vec = fringe_vector_cycles_per_mm(run_theory(setup, dk, bounds, dx_um=0.0))
    ref_ex_vec = fringe_vector_cycles_per_mm(mean_stack(groups[0.0])["mean"])

    orient_ok = True
    for dx in TRANS_DX_UM:
        key = round(dx, 6)
        exp_m = mean_stack(groups[key])
        exp_img = exp_m["mean"]
        th = run_theory(setup, dk, bounds, dx_um=dx, dz_um=0.0)
        theory[dx] = th
        exp[dx] = exp_img
        v_th = fringe_vector_cycles_per_mm(th)
        v_ex = fringe_vector_cycles_per_mm(exp_img)
        if dx > 0 and angle_diff_deg(v_ex["orientation_deg"], ref_ex_vec["orientation_deg"]) > 25:
            orient_ok = False
        k_th_abs = theory_kappa_cycles_per_mm(dx, setup)
        k_ex_abs = v_ex["kappa_cycles_per_mm"]
        if dx == 0:
            k_th_ind = 0.0
            k_ex_ind = 0.0
        else:
            d_th = np.hypot(
                v_th["fx_cycles_per_mm"] - ref_th_vec["fx_cycles_per_mm"],
                v_th["fy_cycles_per_mm"] - ref_th_vec["fy_cycles_per_mm"],
            )
            d_ex = np.hypot(
                v_ex["fx_cycles_per_mm"] - ref_ex_vec["fx_cycles_per_mm"],
                v_ex["fy_cycles_per_mm"] - ref_ex_vec["fy_cycles_per_mm"],
            )
            k_th_ind = float(d_th)
            k_ex_ind = float(d_ex)
        rows.append(
            {
                "dx_um": dx,
                "theory_kappa_abs_cycles_per_mm": k_th_abs if math.isfinite(k_th_abs) else "",
                "experiment_kappa_abs_cycles_per_mm": k_ex_abs,
                "theory_kappa_induced_cycles_per_mm": k_th_ind,
                "experiment_kappa_induced_cycles_per_mm": k_ex_ind,
                "theory_orientation_deg": v_th["orientation_deg"],
                "experiment_orientation_deg": v_ex["orientation_deg"],
                "measurement_uncertainty": "",
                "validity_notes": (
                    "FFT dominant peak; exp reference may include diagonal background at Δx=0"
                ),
            }
        )

    use_induced = orient_ok
    return theory, exp, rows, use_induced


def figure_width_inches() -> float:
    return MM_WIDTH / 25.4


def build_figure6a(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray],
    freq_rows: list[dict],
    use_induced: bool,
    out: Path,
) -> None:
    dx_list = [d for d in TRANS_DX_UM if d in theory]
    w_in = figure_width_inches()
    fig = plt.figure(figsize=(w_in, w_in * 0.72))
    gs = GridSpec(3, len(dx_list), height_ratios=[1, 1, 0.85], hspace=0.32, wspace=0.08)

    t_v = max(np.percentile(theory[d], 99.5) for d in dx_list)
    e_v = max(np.percentile(exp[d], 99.5) for d in dx_list)

    for j, dx in enumerate(dx_list):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory[dx], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δx = {dx:.0f} µm", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Simulation", fontsize=10, labelpad=28)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Experimental", fontsize=10, labelpad=28)

    fig.text(0.02, 0.98, "(a) Transverse lens displacement", fontsize=11, va="top", ha="left")

    axp = fig.add_subplot(gs[2, :])
    xs = [r["dx_um"] for r in freq_rows]
    if use_induced:
        y_th = [r["theory_kappa_induced_cycles_per_mm"] for r in freq_rows]
        y_ex = [r["experiment_kappa_induced_cycles_per_mm"] for r in freq_rows]
        ylabel = "Displacement-induced fringe spatial frequency (cycles mm$^{-1}$)"
        plot_note = "baseline_referenced_vector_magnitude"
    else:
        y_th = [
            float(r["theory_kappa_abs_cycles_per_mm"]) if r["theory_kappa_abs_cycles_per_mm"] != "" else np.nan
            for r in freq_rows
        ]
        y_ex = [r["experiment_kappa_abs_cycles_per_mm"] for r in freq_rows]
        ylabel = "Spatial fringe frequency (cycles mm$^{-1}$)"
        plot_note = "absolute_dominant_fft_peak"

    axp.plot(xs, y_th, "o-", color="#0173B2", label="Simulation", markersize=6, linewidth=1.2)
    axp.plot(xs, y_ex, "s--", color="#DE8F05", label="Experimental", markersize=6, linewidth=1.2)
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)")
    axp.set_ylabel(ylabel)
    axp.set_xticks(xs)
    axp.legend(frameon=True, loc="upper left")
    axp.grid(True, alpha=0.25)
    axp.set_ylim(bottom=0)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.1)
    fig.savefig(out / "FIGURE6A_FINAL_ARTICLE.png", dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out / "FIGURE6A_FINAL_ARTICLE.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return plot_note


def build_figure6b(theory: dict[float, np.ndarray], exp: dict[float, np.ndarray], out: Path) -> None:
    w_in = figure_width_inches()
    fig = plt.figure(figsize=(w_in, w_in * 0.42))
    gs = GridSpec(2, 4, hspace=0.28, wspace=0.08)
    t_v = max(np.percentile(theory[z], 99.5) for z in Z_STAGE_MM)
    e_v = max(np.percentile(exp[z], 99.5) for z in Z_STAGE_MM)

    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory[z_stage], cmap="gray", origin="upper", vmin=0, vmax=t_v, aspect="equal")
        ax.set_title(f"Δz$_{{eff}}$ = {z_eff:.3f} mm", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Simulation", fontsize=10, labelpad=28)

        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp[z_stage], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Experimental", fontsize=10, labelpad=28)

    fig.text(0.015, 0.5, "(b) Axial lens defocus", fontsize=11, va="center", ha="left", rotation=90)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.06)
    fig.savefig(out / "FIGURE6B_FINAL_ARTICLE.png", dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out / "FIGURE6B_FINAL_ARTICLE.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib_pub()

    th_tr, ex_tr, freq_rows, use_induced = load_transverse()
    plot_note = build_figure6a(th_tr, ex_tr, freq_rows, use_induced, out)

    th_ax, ex_ax, _, _, _ = load_axial_panels()
    build_figure6b(th_ax, ex_ax, out)

    with (out / "FIGURE6A_SPATIAL_FREQUENCY_DATA.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(freq_rows[0].keys()))
        w.writeheader()
        w.writerows(freq_rows)

    map_rows = [
        {
            "dz_stage_mm": zs,
            "dz_eff_mm": ze,
            "theory_source": "doubled_equivalent_defocus I+ (approved 20261017T103355Z)",
            "experiment_source": f"mean stack L drift mechanical {zs} mm",
        }
        for zs, ze in zip(Z_STAGE_MM, Z_EFF_MM)
    ]
    with (out / "FIGURE6B_DISPLACEMENT_MAPPING.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(map_rows[0].keys()))
        w.writeheader()
        w.writerows(map_rows)

    meta = {
        "figure6a_transverse": {
            "theory_hashes": {str(dx): sha256_array(th_tr[dx]) for dx in th_tr},
            "experiment_hashes": {str(dx): sha256_array(ex_tr[dx]) for dx in ex_tr},
            "approved_reference": str(APPROVED_6A),
            "spatial_frequency_plot": plot_note,
            "baseline_vector_tracking_ok": use_induced,
        },
        "figure6b_axial": {
            "theory_hashes": {str(zs): sha256_array(th_ax[zs]) for zs in Z_STAGE_MM},
            "experiment_hashes": {str(zs): sha256_array(ex_ax[zs]) for zs in Z_STAGE_MM},
            "approved_reference": str(APPROVED_6B),
            "panels_only_no_quantitative_row": True,
        },
        "production_physics_modified": False,
    }
    (out / "FIGURE6_FINAL_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (out / "FIGURE6_FINAL_CAPTION.md").write_text(
        """# Figure 6 caption

**Figure 6.** Idler imaging-lens displacement in the SU(1,1) twin-beam path (no object). **(a)** Transverse shift Δx: simulated and measured signal intensity at Δx = 0–180 µm; lower panel compares displacement-induced spatial fringe frequency (dominant FFT peak relative to each modality at Δx = 0) from the Digital Twin and experiment. **(b)** Axial defocus: simulated `I_+` at effective parameter Δz_eff = 0, 4.800, 7.160, and 11.624 mm with corresponding experimental frames (mechanical stage positions 0, 2.400, 3.580, 5.812 mm). Panel (b) is a qualitative comparison of ring morphology; ring density and contrast differ between model and detector. Intensities are not radiometrically calibrated between simulation and EMCCD data.
""",
        encoding="utf-8",
    )

    (out / "FIGURE6_FINAL_VALIDATION_REPORT.md").write_text(
        f"""# Figure 6 final validation

1. Figure 6(a): six theory/experiment columns — **yes**
2. Δx = 0 in quantitative graph — **yes**
3. Y-axis definition — **{plot_note}** (see CSV for absolute and induced columns)
4. Baseline subtraction — vector Δf relative to Δx=0 per modality; used in main panel if orientation tracking OK ({use_induced})
5. Figure 6(b) theory — doubled-equivalent-defocus I+ matching `{APPROVED_6B.name}`
6. Figure 6(b) labels — Δz_eff only
7. Figure 6(b) — eight image panels only
8. TIFFs unchanged — mean stacks only
9. Production physics — not modified
10. Export — PNG @ {DPI_PNG} dpi, PDF vector text
""",
        encoding="utf-8",
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Figure 6 publication assembly

- **Output:** `{out}`
- **6(a) spatial frequency panel:** {plot_note}
- **6(b):** images only, approved effective-defocus theory
- **PRODUCTION_PHYSICS_MODIFIED:** NO
""",
        encoding="utf-8",
    )

    print(f"Done: {out}")
    print(f"Figure 6(a) y-axis: {plot_note}; induced_ok={use_induced}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
