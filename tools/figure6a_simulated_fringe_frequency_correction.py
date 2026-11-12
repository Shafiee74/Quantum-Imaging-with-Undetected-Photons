#!/usr/bin/env python3
"""Correct simulated fringe-center frequencies for Figure 6(a).

The Joint-SPDC intensity panels are recomputed with the manuscript settings
and are not edited. Only the bright-fringe center measurement changes.

At Δx >= 40 µm the approved extractor smoothed the profile and applied
``find_peaks`` distance on the NaN-compressed bin index. That index step is
coarser than a profile bin, so the gate became wider than one bright fringe.
At 40 µm it kept the central crest and the two outer crests, and dropped the
two intermediate crests whose prominence is 84% of the central fringe. The
reported spacing was therefore about two fringe periods, and the frequency
fell from 0.970 cycles/mm at 30 µm to 0.643 cycles/mm at 40 µm.

The corrected measurement averages each column inside the illuminated mask
(the fringes are vertical), merges samples that sit on the same flat crest,
and keeps crests whose prominence is at least 25% of the brightest crest in
that frame. Frequency remains 1 / mean adjacent-center spacing.
"""
from __future__ import annotations

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
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setup_params import resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import TRANS_DX_UM, X0_MM, group_tiffs
from tools.c_lab_figure6a_center_spacing import signal_mask
from tools.c_lab_isolated_pilot import DeltaKCalibrationState
from tools.generate_c_lab_figure6a_manuscript import EXP_ROOT, mean_stack_deduped, run_theory_iminus
from tools.publication_graphics_style import (
    REFERENCE_GOLD,
    REFERENCE_NAVY,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
    style_reference_axes,
)

OUT = ROOT / "Figure6a_Simulated_Fringe_Frequency_Correction"
APPROVED_CSV = ROOT / "article figures" / "Figure6a_Final_Data.csv"
EXP_CORR_CSV = ROOT / "Figure6a_Experimental_Fringe_Center_Correction" / "Experimental_Frequency_Comparison.csv"
PIXEL_UM = 13.0
PIXEL_MM = PIXEL_UM * 1e-3
REL_PROM_MIN = 0.25
VALLEY_KEEP_FRAC = 0.98
LOOSE_PROM_FRAC = 0.01
MIN_COLUMN_SAMPLES = 8


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def column_profile(img: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = signal_mask(img, floor_frac=floor)
    n = img.shape[1]
    cols = np.full(n, np.nan)
    row_at = np.full(n, np.nan)
    for c in range(n):
        sel = np.flatnonzero(mask[:, c])
        if sel.size >= MIN_COLUMN_SAMPLES:
            cols[c] = float(np.mean(img[sel, c]))
            row_at[c] = float(np.mean(sel))
    finite = np.isfinite(cols)
    filled = cols.copy()
    if np.any(~finite) and np.any(finite):
        filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), cols[finite])
    return filled, row_at, finite


def bright_crests(
    img: np.ndarray,
    dx_um: float,
    *,
    rel_prom_min: float = REL_PROM_MIN,
) -> dict[str, Any]:
    floor = 0.08 if dx_um <= 30.0 else 0.12
    y, row_at, finite = column_profile(img, floor)
    y0 = y - float(np.min(y[finite]))
    rng = float(np.ptp(y0[finite])) + 1e-12
    idx2, props2 = find_peaks(y0, prominence=LOOSE_PROM_FRAC * rng, distance=1)
    raw_prom = {
        int(i): float(p)
        for i, p in zip(idx2, props2["prominences"])
        if finite[int(i)]
    }
    idx = np.array(sorted(raw_prom), dtype=int)
    if idx.size == 0:
        return {"centers": [], "rejected": [], "floor": floor, "profile": y0, "finite": finite}

    groups: list[list[int]] = [[int(idx[0])]]
    for i in idx[1:]:
        i = int(i)
        lo = groups[-1][-1]
        valley = float(np.min(y0[lo : i + 1]))
        if valley >= VALLEY_KEEP_FRAC * min(float(y0[lo]), float(y0[i])):
            groups[-1].append(i)
        else:
            groups.append([i])

    crests = []
    for group in groups:
        prom = max(raw_prom[i] for i in group)
        weights = np.array([float(y0[i]) for i in group])
        col = int(np.round(np.average(group, weights=weights)))
        col = int(np.clip(col, 0, y0.size - 1))
        crests.append(
            {
                "col_px": col,
                "row_px": float(row_at[col]) if np.isfinite(row_at[col]) else float(np.nanmean(row_at)),
                "u_mm": col * PIXEL_MM,
                "prominence": prom,
                "height": float(y0[col]),
                "plateau_columns": len(group),
            }
        )
    pmax = max(c["prominence"] for c in crests)
    for c in crests:
        c["relative_prominence"] = c["prominence"] / pmax
        c["accepted"] = c["relative_prominence"] >= rel_prom_min
        c["rejection_reason"] = (
            ""
            if c["accepted"]
            else f"prominence is {c['relative_prominence']:.0%} of the brightest crest; below {rel_prom_min:.0%}"
        )
    accepted = [c for c in crests if c["accepted"]]
    for n, c in enumerate(accepted, start=1):
        c["fringe_index"] = n
    spacings = []
    for a, b in zip(accepted, accepted[1:]):
        dpx = b["col_px"] - a["col_px"]
        spacings.append(
            {
                "left_index": a["fringe_index"],
                "right_index": b["fringe_index"],
                "spacing_px": dpx,
                "spacing_mm": dpx * PIXEL_MM,
            }
        )
    if dx_um == 0.0 or len(spacings) == 0:
        k = float("nan")
        mean_sp = float("nan")
        sd = float("nan")
        cv = float("nan")
    else:
        sp = np.array([s["spacing_mm"] for s in spacings], float)
        mean_sp = float(np.mean(sp))
        sd = float(np.std(sp)) if sp.size > 1 else 0.0
        cv = sd / mean_sp if mean_sp > 0 else float("nan")
        k = 1.0 / mean_sp
    return {
        "centers": accepted,
        "rejected": [c for c in crests if not c["accepted"]],
        "all_crests": crests,
        "spacings": spacings,
        "k": k,
        "mean_spacing_mm": mean_sp,
        "spacing_std_mm": sd,
        "spacing_cv": cv,
        "floor": floor,
        "profile": y0,
        "finite": finite,
        "pmax": pmax,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mae_rmse(sim: list[float], exp: list[float]) -> tuple[float, float]:
    err = np.array(sim, float) - np.array(exp, float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def draw_annotations(frames: list[dict[str, Any]]) -> None:
    apply_publication_graphics_style()
    vmax = max(float(np.percentile(f["image"], 99.5)) for f in frames)
    for frame in frames:
        det = frame["det"]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        ax = axes[0]
        ax.imshow(frame["image"], cmap="gray", origin="upper", vmin=0, vmax=vmax)
        for c in det["centers"]:
            ax.plot(c["col_px"], c["row_px"], "o", ms=6, mfc="none", mec="#d95f02", mew=1.1)
            ax.text(c["col_px"] + 4, c["row_px"] - 10, str(c["fringe_index"]), color="#d95f02", fontsize=8)
        for c in det["rejected"]:
            ax.plot(c["col_px"], c["row_px"], "x", ms=4, color="#7f7f7f")
        n = len(det["centers"])
        noun = "center" if n == 1 else "centers"
        ax.set_title(f"Δx = {frame['dx_um']:.0f} µm  ({n} bright {noun})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1]
        x_mm = np.arange(det["profile"].size) * PIXEL_MM
        ax.plot(x_mm, det["profile"], color="0.2", lw=0.9)
        for c in det["rejected"]:
            ax.plot(c["u_mm"], det["profile"][c["col_px"]], "x", color="#7f7f7f", ms=4)
        for c in det["centers"]:
            ax.plot(c["u_mm"], det["profile"][c["col_px"]], "o", color="#d95f02", ms=4)
        for sp, a, b in zip(det["spacings"], det["centers"], det["centers"][1:]):
            y = min(det["profile"][a["col_px"]], det["profile"][b["col_px"]]) * 0.55
            ax.annotate(
                "",
                xy=(b["u_mm"], y),
                xytext=(a["u_mm"], y),
                arrowprops=dict(arrowstyle="<->", color=REFERENCE_NAVY, lw=0.7),
            )
            ax.text((a["u_mm"] + b["u_mm"]) / 2, y, f"{sp['spacing_mm']:.3f}", ha="center", va="bottom", fontsize=7, color=REFERENCE_NAVY)
        ax.set_xlabel("Detector column (mm)")
        ax.set_ylabel("Column mean (display units)")
        k = det["k"]
        ks = f"{k:.3f} cycles/mm" if math.isfinite(k) else "periodic frequency not measurable"
        ax.set_title(ks)
        fig.tight_layout()
        fig.savefig(OUT / f"Annotated_Simulation_dx{int(frame['dx_um']):03d}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(14, 5.2))
    for j, frame in enumerate(frames):
        axes[0, j].imshow(frame["image"], cmap="gray", origin="upper", vmin=0, vmax=vmax)
        for c in frame["det"]["centers"]:
            axes[0, j].plot(c["col_px"], c["row_px"], "o", ms=3, mfc="none", mec="#d95f02", mew=0.8)
        axes[0, j].set_title(f"Δx={frame['dx_um']:.0f} µm\nn={len(frame['det']['centers'])}", fontsize=8)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        prof = frame["det"]["profile"]
        axes[1, j].plot(np.arange(prof.size) * PIXEL_MM, prof, color="0.2", lw=0.7)
        for c in frame["det"]["centers"]:
            axes[1, j].plot(c["u_mm"], prof[c["col_px"]], "o", color="#d95f02", ms=3)
        axes[1, j].set_xlabel("mm", fontsize=7)
        axes[1, j].tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "Simulated_Fringe_Centers.png", dpi=140)
    plt.close(fig)


def draw_figure(
    theory: dict[float, np.ndarray],
    exp: dict[float, np.ndarray],
    sim_k: list[float],
    exp_k: list[float],
) -> None:
    apply_publication_graphics_style()
    dx_list = [float(d) for d in TRANS_DX_UM]
    w_in = 180.0 / 25.4
    fig = plt.figure(figsize=(w_in, w_in * 0.76))
    gs_img = GridSpec(
        2, len(dx_list), height_ratios=[1, 1], hspace=0.025, wspace=0.045,
        left=0.145, right=0.995, top=0.925, bottom=0.485,
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
            ax.text(-0.24, 0.5, "Simulation", transform=ax.transAxes, va="center", ha="right", fontsize=10)
        ax = fig.add_subplot(gs_img[1, j])
        ax.imshow(exp[dx], cmap="gray", origin="upper", vmin=0, vmax=e_v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.24, 0.5, "Experimental", transform=ax.transAxes, va="center", ha="right", fontsize=10)
    axp = fig.add_subplot(gs_plot[0, 0])
    axp.plot(dx_list, sim_k, marker="s", linestyle="--", color=REFERENCE_TEAL, label="Simulation", markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35)
    axp.plot(dx_list, exp_k, marker="D", linestyle=(0, (2.5, 2.0, 6.0, 2.0)), color=REFERENCE_GOLD, label="Experimental", markersize=6, markeredgecolor="0.25", markeredgewidth=0.6, linewidth=1.35)
    axp.set_xlabel("Transverse idler-lens displacement, Δx (µm)", labelpad=4)
    axp.set_ylabel("Spatial frequency (cycles mm$^{-1}$)", labelpad=6)
    axp.set_xticks(dx_list)
    axp.legend(frameon=False, loc="upper left")
    style_reference_axes(axp)
    finite = [v for v in sim_k + exp_k if math.isfinite(v)]
    axp.set_ylim(0.0, max(finite) * 1.12)
    axp.set_xlim(-8.0, max(dx_list) * 1.02)
    fig.savefig(OUT / "Figure6a_Corrected.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / "Figure6a_Corrected.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    theory = {float(dx): run_theory_iminus(setup, dk, dx_um=float(dx)) for dx in TRANS_DX_UM}
    groups = group_tiffs(EXP_ROOT, ref_mm=X0_MM, rel_scale=1000.0)
    exp = {float(dx): mean_stack_deduped(groups[round(float(dx), 6)])[0] for dx in TRANS_DX_UM}

    approved = {(r["modality"], float(r["dx_um"])): r for r in load_csv(APPROVED_CSV)}
    exp_corr = {float(r["lens_displacement_um"]): r for r in load_csv(EXP_CORR_CSV)}

    frames = []
    for dx in TRANS_DX_UM:
        det = bright_crests(theory[float(dx)], float(dx))
        frames.append({"dx_um": float(dx), "image": theory[float(dx)], "det": det})

    sim_rows = []
    center_rows = []
    spacing_rows = []
    for frame in frames:
        det = frame["det"]
        dx = frame["dx_um"]
        prev = approved[("Simulation", dx)]
        sim_rows.append(
            {
                "dx_um": dx,
                "previous_n_centers": int(float(prev["n_fringe_centers"])),
                "corrected_n_centers": len(det["centers"]),
                "previous_k_cycles_per_mm": float(prev["k_plotted_cycles_per_mm"]) if dx == 0 else float(prev["k_center_spacing_cycles_per_mm"]),
                "corrected_k_cycles_per_mm": 0.0 if dx == 0 else det["k"],
                "corrected_periodic_k_cycles_per_mm": det["k"],
                "mean_spacing_mm": det["mean_spacing_mm"],
                "spacing_std_mm": det["spacing_std_mm"],
                "spacing_cv": det["spacing_cv"],
                "rejected_weak_crests": len(det["rejected"]),
            }
        )
        for c in det["all_crests"]:
            center_rows.append(
                {
                    "dx_um": dx,
                    "accepted": int(c["accepted"]),
                    "fringe_index": c.get("fringe_index", ""),
                    "col_px": c["col_px"],
                    "row_px": f"{c['row_px']:.2f}",
                    "u_mm": f"{c['u_mm']:.6f}",
                    "prominence": f"{c['prominence']:.6f}",
                    "relative_prominence": f"{c['relative_prominence']:.4f}",
                    "rejection_reason": c["rejection_reason"],
                }
            )
        for sp in det["spacings"]:
            spacing_rows.append({"dx_um": dx, **sp})

    exp_periodic = []
    exp_plotted = []
    sim_periodic = []
    sim_plotted = []
    prev_sim = []
    paired = []
    for row, frame in zip(sim_rows, frames):
        dx = row["dx_um"]
        er = exp_corr[dx]
        exp_k = float(er["corrected_periodic_frequency_cycles_per_mm"]) if dx != 0 else float("nan")
        exp_plot = 0.0 if dx == 0 else exp_k
        sim_plot = 0.0 if dx == 0 else float(row["corrected_periodic_k_cycles_per_mm"])
        exp_periodic.append(exp_k)
        exp_plotted.append(exp_plot)
        sim_periodic.append(float(row["corrected_periodic_k_cycles_per_mm"]))
        sim_plotted.append(sim_plot)
        prev_sim.append(float(approved[("Simulation", dx)]["k_plotted_cycles_per_mm"]))
        paired.append(
            {
                "dx_um": dx,
                "previous_simulation_k": float(approved[("Simulation", dx)]["k_plotted_cycles_per_mm"]),
                "corrected_simulation_k": sim_plot,
                "corrected_simulation_periodic_k": row["corrected_periodic_k_cycles_per_mm"],
                "corrected_experimental_k": exp_plot,
                "corrected_experimental_periodic_k": exp_k,
                "previous_simulation_n": row["previous_n_centers"],
                "corrected_simulation_n": row["corrected_n_centers"],
                "corrected_experimental_n": int(er["corrected_n_centers"]),
                "simulation_mean_spacing_mm": row["mean_spacing_mm"],
            }
        )

    nz_sim = [s for d, s in zip(TRANS_DX_UM, sim_periodic) if d > 0]
    nz_exp = [e for d, e in zip(TRANS_DX_UM, exp_periodic) if d > 0]
    nz_prev = [float(approved[("Simulation", float(d))]["k_center_spacing_cycles_per_mm"]) for d in TRANS_DX_UM if d > 0]
    mae_new, rmse_new = mae_rmse(nz_sim, nz_exp)
    mae_old, rmse_old = mae_rmse(nz_prev, nz_exp)

    write_csv(OUT / "Simulated_Frequency_Table.csv", sim_rows)
    write_csv(OUT / "Paired_Frequency_Comparison.csv", paired)
    write_csv(OUT / "Simulated_Fringe_Centers.csv", center_rows)
    write_csv(OUT / "Simulated_Fringe_Spacings.csv", spacing_rows)
    draw_annotations(frames)
    draw_figure(theory, exp, sim_plotted, exp_plotted)

    lines = [
        "# Simulated fringe-frequency correction for Figure 6(a)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The Joint-SPDC panels were recomputed with the manuscript settings (`I_minus`, φ = π, `delta_L_mm` = 0, Δz = 0) and were not edited. Display-image mean at Δx = 0 is "
        f"**{float(np.mean(theory[0.0])):.6f}**.",
        "",
        "## What was wrong at 40 µm",
        "",
        "The unmodified 40 µm image has five bright vertical crests. Their column-mean prominences are 39%, 84%, 100%, 84%, and 39% of the brightest crest, and the four gaps are 0.780, 0.767, 0.780, and 0.780 mm.",
        "",
        "The approved extractor kept three of those crests: the center and the two outer ones. The two intermediate crests, each at 84% of the central prominence, were removed by the minimum-separation gate. That gate was computed in uniform profile bins and then applied after empty bins had been deleted, so its physical width exceeded one fringe. The surviving spacing is about two fringe periods, 1.554 mm, which is 0.6435 cycles/mm. That is lower than the 30 µm result, 0.9700 cycles/mm, even though the 40 µm image is the denser pattern.",
        "",
        "The same skip removes bright crests at 60 µm (3 kept out of 7) and at 180 µm (11 kept out of 21). At 20 and 30 µm the bright-crest set is the same as before. Two baseline ripples at 30 µm, at 11% of the main-fringe prominence, stay excluded.",
        "",
        "## Corrected rule",
        "",
        "For every displacement the fringes are vertical, so the profile is the mean of each detector column inside the illuminated mask (8% of peak for Δx ≤ 30 µm, 12% otherwise, as in the approved simulation mask). Samples on one flat crest are one center. A crest is kept when its prominence is at least 25% of the brightest crest in that frame. Frequency is 1 / mean adjacent spacing, using the 13 µm pixel pitch along the column axis. The FFT is not the frequency.",
        "",
        "At Δx = 0 the single illumination maximum does not define a periodic frequency. The plotted value 0 cycles/mm is the aligned-reference convention.",
        "",
        "## Frequencies",
        "",
        "| Δx (µm) | Previous sim n | Corrected sim n | Previous sim k | Corrected sim k | Mean spacing (mm) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sim_rows:
        k_prev = row["previous_k_cycles_per_mm"]
        k_new = row["corrected_periodic_k_cycles_per_mm"]
        ks_prev = f"{k_prev:.4f}" if math.isfinite(k_prev) else "—"
        ks_new = f"{k_new:.4f}" if math.isfinite(float(k_new)) else "—"
        sp = row["mean_spacing_mm"]
        sps = f"{sp:.4f}" if math.isfinite(float(sp)) else "—"
        lines.append(
            f"| {row['dx_um']:.0f} | {row['previous_n_centers']} | {row['corrected_n_centers']} | {ks_prev} | {ks_new} | {sps} |"
        )
    lines += [
        "",
        "Corrected experimental frequencies are those from the fringe-center correction. At Δx = 40 µm that value is 0.7855 cycles/mm.",
        "",
        "| Δx (µm) | Corrected sim k | Corrected exp k |",
        "|---:|---:|---:|",
    ]
    for row in paired:
        sk = row["corrected_simulation_periodic_k"]
        ek = row["corrected_experimental_periodic_k"]
        sks = f"{float(sk):.4f}" if sk != "" and math.isfinite(float(sk)) else "—"
        eks = f"{float(ek):.4f}" if ek != "" and math.isfinite(float(ek)) else "—"
        lines.append(f"| {row['dx_um']:.0f} | {sks} | {eks} |")
    lines += [
        "",
        f"Nonzero pairs (n = 5). Previous simulation against the corrected experiment: MAE = {mae_old:.4f}, RMSE = {rmse_old:.4f} cycles/mm.",
        f"Corrected simulation against the corrected experiment: MAE = {mae_new:.4f}, RMSE = {rmse_new:.4f} cycles/mm.",
        "",
        "The errors are larger because the simulated frequencies at 40, 60, and 180 µm had been too low. The experimental frequencies were not adjusted. This comparison does not support a claim of quantitative experimental validation. It does remove the non-physical drop in the simulated curve between 30 and 40 µm.",
        "",
        "Mechanical lens displacement Δx and the image-plane motion of a fringe center remain different quantities. This audit measures only the simulated fringe spacing.",
        "",
    ]
    (OUT / "Simulated_Frequency_Correction_Report.md").write_text("\n".join(lines), encoding="utf-8")
    print("dx0 mean", float(np.mean(theory[0.0])))
    for row in sim_rows:
        print(row["dx_um"], "n", row["corrected_n_centers"], "k", row["corrected_periodic_k_cycles_per_mm"])
    print("MAE/RMSE old", mae_old, rmse_old, "new", mae_new, rmse_new)


if __name__ == "__main__":
    main()
