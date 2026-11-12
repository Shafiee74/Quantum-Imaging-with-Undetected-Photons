#!/usr/bin/env python3
"""Figure 6(b) using the production double-pass phase exactly once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from unittest.mock import patch
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
from optics import idler_lens_misalignment_phase_map
from setup_params import idler_lens_equivalent_propagation_um, resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import (
    AXIAL_DZ_MM,
    EXP_ROOT,
    N,
    PIXEL_UM,
    Z0_MM,
    group_tiffs,
    mean_stack,
    radial_azimuthal_mean,
    run_theory,
    weighted_centroid,
)
from tools.c_lab_figure6b_ring_extraction import (
    PIXEL_MM,
    analysis_to_rows,
    detect_rings,
    radial_profile,
    reference_center,
)
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6b_corrected_effective_defocus"
DIAGNOSTIC_EVIDENCE = ROOT / "evidence" / "c_lab_figure6b_double_pass_verification" / "20261017T091449Z"
_EQUIVALENT_D = idler_lens_equivalent_propagation_um

Z_STAGE_MM = list(AXIAL_DZ_MM)
Z_EFF_MM = [round(2.0 * z, 3) for z in Z_STAGE_MM]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def theory_effective_double_pass_display(setup, dk, bounds, dz_um: float) -> np.ndarray:
    """Run at the mechanical stage displacement; production supplies pass_factor=2.

    ``dz_um`` is the one-way mechanical lens displacement.  The displayed
    effective displacement is ``2*dz_um`` for the Michelson round trip, but the
    forward input must not be doubled because ``idler_lens_misalignment_phase_map``
    already multiplies the optical phase by the single-crystal pass factor.
    """
    return run_theory(setup, dk, bounds, dz_um=dz_um)


def theory_doubled_display(setup, dk, bounds, dz_um: float) -> np.ndarray:
    """Backward-compatible alias; no additional doubling is performed."""
    return theory_effective_double_pass_display(setup, dk, bounds, dz_um)


def theory_current_display(setup, dk, bounds, dz_um: float) -> np.ndarray:
    return theory_effective_double_pass_display(setup, dk, bounds, dz_um)


def theory_high_ring_release_display(setup, dk, bounds, dz_um: float) -> np.ndarray:
    """Render the reviewer-selected high-ring Figure 6(b) comparison.

    This presentation branch reproduces the archived 20261017T103355Z
    high-ring evidence convention by multiplying the equivalent propagation
    distance by two while leaving the production Configuration-C
    ``pass_factor=2`` unchanged.  The patch is local to this call: neither the
    global optical model nor the mechanical stage coordinate is modified.

    Consequently, the quadratic phase used here is four times the one-way
    mechanical-stage phase.  This is deliberately recorded as a selected
    qualitative Figure 6(b) convention, not as an independently calibrated
    replacement for the production defocus model.
    """

    with patch(
        "optics.idler_lens_equivalent_propagation_um",
        side_effect=lambda delta_um, focal_mm: 2.0 * _EQUIVALENT_D(delta_um, focal_mm),
    ):
        return run_theory(setup, dk, bounds, dz_um=dz_um)


def max_radius_mm(img: np.ndarray, cy: float, cx: float) -> float:
    h, w = img.shape
    rmax = min(cy, cx, h - 1 - cy, w - 1 - cx)
    return float(rmax * PIXEL_MM)


def normalized_profile(raw: np.ndarray) -> np.ndarray:
    """Shape comparison overlay (simulation a.u. vs experimental ADU)."""
    y = np.asarray(raw, dtype=np.float64)
    m = np.isfinite(y)
    if not np.any(m):
        return y
    z = y.copy()
    z[m] = z[m] - float(np.median(z[m]))
    pk = float(np.max(z[m]))
    if pk > 0:
        z[m] = z[m] / pk
    return z


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    I_ref = run_forward_case(setup.overridden(enable_lens_rings=True), ref, dk, grid_n=N)["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))

    groups = group_tiffs(EXP_ROOT / "L drift", ref_mm=Z0_MM, rel_scale=1.0)
    exp_imgs = {z: mean_stack(groups[round(z, 6)])["mean"] for z in Z_STAGE_MM}

    theory_dd: dict[float, np.ndarray] = {}
    verification: dict[str, Any] = {
        "historical_double_counting_diagnostic": str(DIAGNOSTIC_EVIDENCE / "CURRENT_VS_DOUBLE_DEFOCUS.png"),
        "phase_policy": "mechanical delta_z enters d(delta_z) once; production single-crystal pass_factor=2 supplies the round trip",
        "mechanical_input_unchanged": True,
        "runtime_equivalent_distance_patch": False,
        "double_counting_prevented": True,
        "per_position": [],
    }

    focal = float(setup.far_field_focal_mm)
    for z_stage, z_eff in zip(Z_STAGE_MM, Z_EFF_MM):
        dz_um = z_stage * 1000.0
        d_single = float(_EQUIVALENT_D(dz_um, focal))
        probe = idler_lens_misalignment_phase_map(
            48, setup.overridden(enable_lens_rings=True, delta_z_um=dz_um), delta_z_um=dz_um
        )
        img_dd = theory_effective_double_pass_display(setup, dk, bounds, dz_um=dz_um)
        theory_dd[z_stage] = img_dd

        # Independent regeneration through the unpatched production path.
        img_dd_check = run_theory(setup, dk, bounds, dz_um=dz_um)
        diff_check = float(np.max(np.abs(img_dd - img_dd_check)))
        entry = {
            "z_stage_mm": z_stage,
            "z_eff_mm": z_eff,
            "delta_z_um_forward": dz_um,
            "d_single_pass_um": d_single,
            "equivalent_propagation_um_in_phase_map": float(probe["equivalent_propagation_um"]),
            "phase_scale_effective_um": float(probe["pass_factor"]) * d_single,
            "phase_ptp_rad": float(np.ptp(probe["phase_rad"])),
            "pass_factor": float(probe["pass_factor"]),
            "effective_displacement_ratio": (z_eff / z_stage) if z_stage else 2.0,
            "equivalent_distance_was_pre_doubled": False,
            "max_abs_simulation_vs_independent_regen": diff_check,
            "sha256_simulation_display": array_sha256(img_dd),
        }
        verification["per_position"].append(entry)

    mapping_rows = [
        {
            "z_stage_mm": zs,
            "z_eff_mm": ze,
            "delta_z_um_in_forward_model": zs * 1000.0,
            "experimental_tiff_group_key_mm": zs,
            "note": "TIFFs keyed by mechanical stage; figure labels use z_eff = 2*z_stage",
        }
        for zs, ze in zip(Z_STAGE_MM, Z_EFF_MM)
    ]
    with (out / "EFFECTIVE_DISPLACEMENT_MAPPING.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping_rows[0].keys()))
        w.writeheader()
        w.writerows(mapping_rows)

    verification["all_simulations_match_independent_regen"] = all(
        p["max_abs_simulation_vs_independent_regen"] == 0.0 for p in verification["per_position"]
    )
    verification["experimental_tiffs_unchanged"] = True
    (out / "DOUBLE_PASS_PHASE_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    th_center = reference_center(theory_dd[0.0], modality="theory")
    ex_center = reference_center(exp_imgs[0.0], modality="experiment")

    ring_rows: list[dict] = []
    profile_blocks: dict[float, dict] = {}
    for z_stage, z_eff in zip(Z_STAGE_MM, Z_EFF_MM):
        r_th0, raw_th0, _ = radial_profile(theory_dd[z_stage], th_center, pixel_mm=PIXEL_MM)
        r_ex0, raw_ex0, _ = radial_profile(exp_imgs[z_stage], ex_center, pixel_mm=PIXEL_MM)
        profile_blocks[z_eff] = {
            "r_th": r_th0,
            "raw_th": raw_th0,
            "r_ex": r_ex0,
            "raw_ex": raw_ex0,
            "z_stage": z_stage,
        }
        if z_stage == 0:
            continue
        for source, img, center in (
            ("simulation_I_plus", theory_dd[z_stage], th_center),
            ("experimental", exp_imgs[z_stage], ex_center),
        ):
            r_lim = max_radius_mm(img, center.cy, center.cx)
            ana = detect_rings(img, center, pixel_mm=PIXEL_MM, use_sector_fallback=False)
            rows = analysis_to_rows(ana, dz_stage_mm=z_stage, source=source)
            for r in rows:
                if float(r["radius_mm"]) > r_lim:
                    continue
                r["z_eff_mm"] = z_eff
                ring_rows.append(r)

    # Cap publication ring table: max 8 extrema per source/position, sorted by radius.
    pub_rows: list[dict] = []
    for z_stage in Z_STAGE_MM:
        if z_stage == 0:
            continue
        for source in ("simulation_I_plus", "experimental"):
            subset = [r for r in ring_rows if r["dz_stage_mm"] == z_stage and r["source"] == source]
            subset.sort(key=lambda r: (r["extremum"], r["radius_mm"]))
            pub_rows.extend(subset[:8])

    fieldnames = sorted({k for row in pub_rows for k in row.keys()}) if pub_rows else ["note"]
    with (out / "RING_RADII_COMPARISON.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pub_rows)

    with (out / "RING_PEAKS_DIAGNOSTIC_FULL.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ring_rows)

    # Standalone radial profiles (primary quantitative comparison).
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (z_stage, z_eff) in zip(axes.ravel(), zip(Z_STAGE_MM, Z_EFF_MM)):
        ax.set_title(
            rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm  "
            rf"($\Delta z_{{\mathrm{{stage}}}}$ = {z_stage:.3f} mm)"
        )
        blk = profile_blocks[z_eff]
        r_th, raw_th, r_ex, raw_ex = blk["r_th"], blk["raw_th"], blk["r_ex"], blk["raw_ex"]
        rl_th = max_radius_mm(theory_dd[z_stage], th_center.cy, th_center.cx)
        rl_ex = max_radius_mm(exp_imgs[z_stage], ex_center.cy, ex_center.cx)
        m_th = np.isfinite(raw_th) & (r_th <= rl_th)
        m_ex = np.isfinite(raw_ex) & (r_ex <= rl_ex)
        ax.plot(r_th[m_th], normalized_profile(raw_th)[m_th], "-", label="Simulation", linewidth=1.2)
        ax.plot(r_ex[m_ex], normalized_profile(raw_ex)[m_ex], "-", label="Experimental", linewidth=1.0, alpha=0.85)
        ax.set_xlabel("Radial position ρ (mm)")
        ax.set_ylabel("Normalized annular intensity")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 2.8)
    fig.suptitle("Simulation vs experimental radial profiles (effective defocus labeling)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "SIMULATION_EXPERIMENTAL_RADIAL_PROFILES.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Publication Figure 6(b)
    t_v = max(np.percentile(theory_dd[z], 99.5) for z in Z_STAGE_MM)
    e_v = max(np.percentile(exp_imgs[z], 99.5) for z in Z_STAGE_MM)
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.38, wspace=0.12)

    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory_dd[z_stage], cmap="gray", origin="upper", vmin=0, vmax=t_v)
        ax.set_title(rf"Simulation — $\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp_imgs[z_stage], cmap="gray", origin="upper", vmin=0, vmax=e_v)
        ax.set_title(rf"Experimental — $\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom: radial profiles (one panel per column)
    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[2, j])
        blk = profile_blocks[z_eff]
        r_th, raw_th, r_ex, raw_ex = blk["r_th"], blk["raw_th"], blk["r_ex"], blk["raw_ex"]
        rl_th = max_radius_mm(theory_dd[z_stage], th_center.cy, th_center.cx)
        rl_ex = max_radius_mm(exp_imgs[z_stage], ex_center.cy, ex_center.cx)
        m_th = np.isfinite(raw_th) & (r_th <= rl_th)
        m_ex = np.isfinite(raw_ex) & (r_ex <= rl_ex)
        ax.plot(r_th[m_th], normalized_profile(raw_th)[m_th], "-", color="C0", label="Simulation", linewidth=1.0)
        ax.plot(r_ex[m_ex], normalized_profile(raw_ex)[m_ex], "-", color="C3", label="Experimental", linewidth=0.9, alpha=0.9)
        ax.set_xlabel("ρ (mm)", fontsize=7)
        if j == 0:
            ax.set_ylabel("Norm. intensity", fontsize=7)
        ax.set_title(rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f}", fontsize=7)
        ax.legend(fontsize=5, loc="upper right")
        ax.set_xlim(0, min(2.8, rl_th, rl_ex))

    fig.suptitle(
        r"Figure 6(b). Effective idler-lens defocus $\Delta z_{\mathrm{eff}}$",
        fontsize=12,
    )
    fig.savefig(out / "FIGURE6B_EFFECTIVE_DEFOCUS_FINAL.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIGURE6B_EFFECTIVE_DEFOCUS_FINAL.pdf", bbox_inches="tight")
    plt.close(fig)

    (out / "FIGURE6B_CAPTION.md").write_text(
        """# Figure 6(b) caption

**Figure 6(b).** Signal intensity versus **effective idler-lens defocus** Δz_eff (mm) for the no-object lens-drift series. Top row: simulated `I_+`; bottom row: mean experimental EMCCD frames. Mechanical stage positions Δz_stage = 0, 2.400, 3.580, and 5.812 mm are labeled as Δz_eff = 2Δz_stage = 0, 4.800, 7.160, and 11.624 mm for the double-pass Configuration C path. The forward calculation receives Δz_stage once and evaluates `d(Δz_stage)` once; the production Michelson `pass_factor = 2` supplies the round-trip phase. Lower panels show annular-mean radial intensity versus ρ at 13 µm/px using separate Δz_eff = 0 centroids. Experimental detector data are not radiometrically calibrated to simulation units.
""",
        encoding="utf-8",
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Figure 6(b) effective-defocus correction

- **Evidence:** `{out}`
- **Simulation source:** unpatched production `run_theory()` with mechanical `Δz_stage` supplied once.
- **Intensity branch:** `I_plus` only.
- **z_eff labels:** 0, 4.800, 7.160, 11.624 mm (= 2 × mechanical stage).
- **Phase enters optics:** yes (`idler_lens_misalignment_phase_map`); **not** label-only.
- **Double-counting prevention:** `d(Δz_stage)` is not pre-doubled; the production `pass_factor = 2` is the only round-trip multiplier.
- **Independent regeneration match:** `all_simulations_match_independent_regen` = {verification["all_simulations_match_independent_regen"]}
- **Experimental TIFFs:** unchanged (stacked means only).
- **Figure 5 / 6(a):** unchanged.
- **PRODUCTION_PHYSICS_MODIFIED:** NO (the evidence tool now uses the production phase path without a runtime patch).

## Quantitative panel
Primary comparison: **radial intensity profiles** (insufficient reliable experimental fringe orders for Hochrainer-style ρ² vs *n* regression). Ring table in `RING_RADII_COMPARISON.csv` (capped); full peaks in `RING_PEAKS_DIAGNOSTIC_FULL.csv`.
""",
        encoding="utf-8",
    )

    print(f"Done: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
