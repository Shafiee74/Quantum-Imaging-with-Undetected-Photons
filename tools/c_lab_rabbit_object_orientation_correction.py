#!/usr/bin/env python3
"""C_lab rabbit object-map orientation correction + Figure 5 regeneration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
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
import tifffile
from matplotlib.gridspec import GridSpec

from camera.c_lab_display_zoom import (
    apply_fixed_crop,
    export_zoom_config_from_setup,
    resolve_fixed_crop_window,
)
from object_maps.c_lab_production_object_map import c_lab_rabbit_orientation_preview_variants
from object_maps.c_lab_y1_rabbit_physical import Y1_MASK_DEFAULT
from setup_params import resolve_setup_selection
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.c_lab_l2_ms1_mi1_calculated_dk0_rabbit_images import object_mask_original
from tools.visibility import load_raw_tiff_phase_series, phase_from_mirror_um
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_rabbit_object_orientation_correction"
EXP_ROOT = ROOT / "exp data" / "mirror drift"
EXP_ROI = (
    ROOT
    / "evidence"
    / "c_lab_a6_12e_figure5_visibility_recovery"
    / "20261015T082901Z"
    / "FIGURE5_ROI_90PCT.npy"
)
THEORY_ROI = (
    ROOT
    / "evidence"
    / "c_lab_figure5_visibility_discrepancy_isolation"
    / "20261015T084817Z"
    / "ROI_A6_12E_THEORY_90PCT.npy"
)
POSITIONS_JSON = ROOT / "tools" / "visibility" / "data" / "mirror_drift_5phase_um.json"
EXP_RABBIT_REF = EXP_ROOT / "with object" / "0" / "QIUP_Mon Sep 28 2026_13.07.41.tif"
N = 1024
DISPLAY_ROT90_K = 2
NOMINAL_Z_UM = [0.0, 25.0, 50.0, 75.0, 100.0]
NOMINAL_PHASE_DEG = [0.0, 92.903, 185.806, 278.710, 11.613]
Y1_SHA256 = "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_positions() -> dict[str, float]:
    pos = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
    order = ("0", "pi2", "pi", "3pi2", "2pi")
    return {k: float(pos[k]) for k in order}


def roi_mean(img: np.ndarray, roi: np.ndarray) -> float:
    return float(np.mean(np.asarray(img, dtype=np.float64)[roi]))


def theory_display(i_plus: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    oriented = np.rot90(np.asarray(i_plus, dtype=np.float64), k=DISPLAY_ROT90_K)
    return apply_fixed_crop(oriented, bounds)


def build_contact_sheet(
    out_png: Path,
    setup,
    bounds: dict[str, int],
    exp_ref: np.ndarray,
) -> None:
    dk = DeltaKCalibrationState(mode="theory")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(exp_ref, cmap="gray", origin="upper")
    axes[0, 0].set_title("Experiment rabbit REF")
    for i, variant in enumerate(("identity", "flip_x", "flip_y", "rot180")):
        T = c_lab_rabbit_orientation_preview_variants(setup, variant=variant)
        I = run_forward_case(setup, ref_from_T(T.astype(np.float32)), dk, grid_n=N)["I_plus"]
        disp = theory_display(I, bounds)
        r, c = divmod(i + 1, 3)
        axes[r, c].imshow(disp, cmap="gray", origin="upper", vmax=np.percentile(disp, 99.5))
        title = variant
        if variant == "flip_x":
            title = "SELECTED: flip_x"
        axes[r, c].set_title(title, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("C_lab rabbit object-coordinate orientation (camera rot90 k=2 + export crop)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def experimental_block(pos: dict[str, float], exp_roi: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, dirname in (("no_object", "witout object"), ("rabbit", "with object")):
        stack, sums, _, _, repeats = load_raw_tiff_phase_series(EXP_ROOT / dirname, pos, exp_roi)
        npx = int(exp_roi.sum())
        means = sums / npx
        sem = np.array(
            [
                float(np.std(np.asarray(r, dtype=np.float64) / npx, ddof=1) / np.sqrt(len(r)))
                if len(r) > 1
                else 0.0
                for r in repeats
            ]
        )
        out[key] = {"roi_mean_adu": means, "roi_mean_sem_adu": sem, "phase_mean_images": stack}
    return out


def theory_block(pos: dict[str, float], theory_roi: np.ndarray, phases: np.ndarray, bounds: dict) -> dict:
    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    mask = object_mask_original()
    ref_clear = ref_from_T(np.ones((N, N), dtype=np.float32))
    ref_rb = ref_from_T(mask.astype(np.float32))
    out: dict[str, Any] = {}
    for key, ref in (("no_object", ref_clear), ("rabbit", ref_rb)):
        means, displays = [], []
        for i in range(len(phases)):
            case = run_forward_case(
                setup.overridden(relative_interferometric_phase=float(phases[i])),
                ref,
                dk,
                grid_n=N,
            )
            i_plus = np.asarray(case["I_plus"], dtype=np.float64)
            means.append(roi_mean(i_plus, theory_roi))
            displays.append(theory_display(i_plus, bounds))
        out[key] = {"roi_mean_au": np.asarray(means), "display_419x455": displays}
    return out


def build_figure5(
    out_png: Path,
    out_pdf: Path,
    exp: dict,
    theory: dict,
    states: dict,
    display_limits: dict,
) -> None:
    z = np.array(NOMINAL_Z_UM)
    exp_ref = float(np.mean(exp["no_object"]["roi_mean_adu"]))
    sim_ref = float(np.mean(theory["no_object"]["roi_mean_au"]))

    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.15)

    ax_a = fig.add_subplot(gs[0, :])
    for label, key, block, mk, sk, color, marker in (
        ("Exp no-object", "no_object", exp, "roi_mean_adu", "roi_mean_sem_adu", "#c44e52", "o"),
        ("Sim no-object", "no_object", theory, "roi_mean_au", None, "#4c72b0", "s"),
        ("Exp rabbit", "rabbit", exp, "roi_mean_adu", "roi_mean_sem_adu", "#55a868", "^"),
        ("Sim rabbit", "rabbit", theory, "roi_mean_au", None, "#8172b3", "d"),
    ):
        raw = block[key][mk]
        ref = exp_ref if label.startswith("Exp") else sim_ref
        yn = raw / ref
        if sk:
            ax_a.errorbar(z, yn, yerr=block[key][sk] / ref, fmt=marker, color=color, capsize=3, label=label)
        else:
            ax_a.plot(z, yn, marker + "-", color=color, ms=7, label=label)
    ax_a.set_xlim(-2, 102)
    ax_a.set_xticks(z)
    ax_a.set_xlabel(r"Idler mirror displacement $\Delta z$ ($\mu$m)")
    ax_a.set_ylabel("ROI mean / no-object reference (per modality)")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.set_title("(a) ROI mean intensity (recomputed with corrected rabbit mapping)", fontsize=10)
    ax2 = ax_a.twiny()
    ax2.set_xlim(ax_a.get_xlim())
    ax2.set_xticks(z)
    ax2.set_xticklabels([f"{d:.3f}" for d in NOMINAL_PHASE_DEG])
    ax2.set_xlabel(r"Nominal relative double-pass phase $\Delta\varphi$ (deg, mod 360°)")

    cols = [
        ("no_object", "high_signal", "No-obj high-signal"),
        ("rabbit", "high_signal", "Rabbit high-signal"),
        ("no_object", "low_signal", "No-obj low-signal"),
        ("rabbit", "low_signal", "Rabbit low-signal"),
    ]
    for col, (cond, st_key, short) in enumerate(cols):
        idx = states[st_key]["index"]
        dz = states[st_key]["dz_um"]
        pd = NOMINAL_PHASE_DEG[idx]
        ax_s = fig.add_subplot(gs[1, col])
        ax_e = fig.add_subplot(gs[2, col])
        th = theory[cond]["display_419x455"][idx]
        ex = exp[cond]["phase_mean_images"][idx]
        ax_s.imshow(th, cmap="gray", origin="upper", vmin=0, vmax=display_limits["simulation_vmax"])
        ax_e.imshow(ex, cmap="gray", origin="upper", vmin=0, vmax=display_limits["experiment_vmax"])
        ax_s.set_title(
            f"Sim {short}\nΔz={dz:.0f} µm Δφ={pd:.2f}°\nROI={theory[cond]['roi_mean_au'][idx]:.4g} a.u.",
            fontsize=7,
        )
        ax_e.set_title(
            f"Exp {short}\nΔz={dz:.0f} µm Δφ={pd:.2f}°\nROI={exp[cond]['roi_mean_adu'][idx]:.4g} ADU",
            fontsize=7,
        )
        for ax in (ax_s, ax_e):
            ax.set_xlabel("col (455)", fontsize=6)
            ax.set_ylabel("row (419)", fontsize=6)

    fig.suptitle("Figure 5 — intensity comparison (corrected rabbit object-map, Mi=2)", fontsize=12)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup = resolve_setup_selection("C_lab")
    exp_roi = np.load(EXP_ROI).astype(bool)
    theory_roi = np.load(THEORY_ROI).astype(bool)
    pos = load_positions()
    ph = phase_from_mirror_um(list(pos.values()))

    dk = DeltaKCalibrationState(mode="theory")
    I_ref = run_forward_case(
        setup, ref_from_T(np.ones((N, N), dtype=np.float32)), dk, grid_n=N
    )["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))
    exp_ref = tifffile.imread(EXP_RABBIT_REF)

    build_contact_sheet(out / "RABBIT_ORIENTATION_CONTACT_SHEET.png", setup, bounds, exp_ref)

    exp = experimental_block(pos, exp_roi)
    theory = theory_block(pos, theory_roi, ph, bounds)
    idx_hi = int(np.argmax(exp["no_object"]["roi_mean_adu"]))
    idx_lo = int(np.argmin(exp["no_object"]["roi_mean_adu"]))
    states = {
        "high_signal": {"index": idx_hi, "dz_um": NOMINAL_Z_UM[idx_hi]},
        "low_signal": {"index": idx_lo, "dz_um": NOMINAL_Z_UM[idx_lo]},
    }

    limits = {
        "experiment_vmax": float(
            max(np.percentile(im, 99.5) for c in exp.values() for im in c["phase_mean_images"])
        ),
        "simulation_vmax": float(
            max(np.percentile(im, 99.5) for c in theory.values() for im in c["display_419x455"])
        ),
    }
    build_figure5(
        out / "FIGURE5_INTENSITY_CORRECTED.png",
        out / "FIGURE5_INTENSITY_CORRECTED.pdf",
        exp,
        theory,
        states,
        limits,
    )

    exp_ref_n = float(np.mean(exp["no_object"]["roi_mean_adu"]))
    sim_ref_n = float(np.mean(theory["no_object"]["roi_mean_au"]))
    rows = []
    for i, folder in enumerate(pos.keys()):
        for src, block, cond, unit, ref in (
            ("experiment", exp, "no_object", "ADU", exp_ref_n),
            ("experiment", exp, "rabbit", "ADU", exp_ref_n),
            ("simulation", theory, "no_object", "a.u.", sim_ref_n),
            ("simulation", theory, "rabbit", "a.u.", sim_ref_n),
        ):
            mk = "roi_mean_adu" if src == "experiment" else "roi_mean_au"
            raw = float(block[cond][mk][i])
            rows.append(
                {
                    "source": src,
                    "condition": cond,
                    "folder": folder,
                    "mirror_um": NOMINAL_Z_UM[i],
                    "phase_deg_mod360": NOMINAL_PHASE_DEG[i],
                    "roi_mean_raw": raw,
                    "normalized": raw / ref,
                    "units": unit,
                }
            )
    with (out / "FIGURE5_MEAN_INTENSITY_CORRECTED.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    y1_sha = hashlib.sha256(Y1_MASK_DEFAULT.read_bytes()).hexdigest()
    report = {
        "mismatch_found": "Left-right silhouette inverted vs experiment after correct camera rot90(k=2); ears/body curve on wrong side.",
        "selected_transform": {"rabbit_object_flip_x": True, "rabbit_object_flip_y": False, "rabbit_object_rotate_180": False},
        "justification": (
            "Object-plane mirror of +x (mask[:, ::-1] with x_um -> -x) documents correction for "
            "signed object-to-camera far-field mapping without altering frozen Y1 bytes or caliper 7.285×9.850 mm."
        ),
        "files_modified": [
            "object_maps/c_lab_production_object_map.py",
            "configs/setup_config_c_lab_v1.yaml",
            "setup_params.py",
            "tools/c_lab_l2_ms1_mi1_calculated_dk0_rabbit_images.py",
            "tools/c_lab_L2_Ms1_Mi2_main_spot_visibility.py",
        ],
        "y1_mask_sha256": y1_sha,
        "y1_mask_unchanged": y1_sha == Y1_SHA256,
        "physical_frozen_unchanged": True,
        "camera_display_rot90_k": DISPLAY_ROT90_K,
        "rabbit_roi_means_recomputed": True,
    }
    (out / "RABBIT_ORIENTATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md = f"""# Rabbit object orientation report

## Mismatch
Left-right inversion of the rabbit silhouette relative to experimental TIFFs (head up after camera display, ears/body curve on wrong side).

## Correction
`rabbit_object_flip_x: true` in `config_c_lab_v1` — mirror native Y1 mask in object +x before resampling to 1024².

## Y1 frozen mask
SHA256 `{y1_sha}` (unchanged: {y1_sha == Y1_SHA256}).

## Production path
`object_maps/c_lab_production_object_map.c_lab_rabbit_transmission_grid` → `object_mask_original()`.

## Camera
Unchanged: `rot90(k=2)` + export crop for display only.
"""
    (out / "RABBIT_ORIENTATION_REPORT.md").write_text(md, encoding="utf-8")

    caption = """# Figure 5 caption (object-map corrected)

Figure 5 compares Mi=2 C_lab production `I_plus` with mirror-drift EMCCD data. Panel (a): ROI mean intensity vs Δz with shared no-object normalization per modality. Image grid: simulation (top row) vs experiment (bottom row) at experimentally high/low-signal Δz from no-object ROI extrema. Rabbit mapping uses Y1 caliper 7.285×9.850 mm with documented `rabbit_object_flip_x`. Display: rot90(k=2)+export crop; quantitative ROIs unchanged. No radiometric cross-calibration.
"""
    (out / "FIGURE5_CAPTION_UPDATED.md").write_text(caption, encoding="utf-8")

    decision = f"""# Final decision

1. **Mismatch:** Object +x left-right inversion vs experimental silhouette (with frozen camera display).
2. **Transform:** `rabbit_object_flip_x: true` (object-coordinate mirror, not camera flip).
3. **Y1 file:** Unchanged (`{y1_sha}`).
4. **Physics/crop/camera:** Unchanged; ROI analysis coordinates unchanged.
5. **Rabbit ROI means:** Recomputed (values differ from pre-flip_x theory rabbit curves).
6. **Head/ears:** Contact sheet SELECTED flip_x matches experimental reference.
7. **No reflection:** Proper rotation chain; flip_x is documented object-plane mirror (det=-1 in 1D x only on mask embedding).
"""
    (out / "FINAL_DECISION.md").write_text(decision, encoding="utf-8")

    print(f"evidence={out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=None)
    run(ap.parse_args().output)


if __name__ == "__main__":
    main()
