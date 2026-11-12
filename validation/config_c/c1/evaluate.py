"""Task 18 Stage 4 — with-object evaluation using frozen C1 calibration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.config_c.c1.calibrate import (
    dark_subtract,
    load_group_stack,
    mean_and_uncertainty,
)
from validation.config_c.c1.identify import (
    NOMINAL_PHASE_RAD,
    PHASE_LABELS,
    PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN,
    IdentificationResult,
)
from verification.phase_reconstruction.constants import FOUR_STEP_PHASES_RAD
from verification.phase_reconstruction.metrics import wrap_phase
from verification.phase_reconstruction.reconstruct import reconstruct_phase_steps


def process_with_object(
    identification: IdentificationResult,
    *,
    processed_dir: Path,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    groups = identification.groups

    dark_group = groups.get("with object::dark")
    dark_mean = None
    dark_meta: dict[str, Any] = {"available": False}
    if dark_group and dark_group["n_files_primary_shape"] > 0:
        dark_stack, dark_load = load_group_stack(dark_group)
        if dark_stack is not None:
            stats = mean_and_uncertainty(dark_stack)
            dark_mean = stats["mean"]
            dark_meta = {
                "available": True,
                "n_repeats": stats["n_repeats"],
                "load": dark_load,
            }
            np.save(processed_dir / "obj_dark_mean.npy", dark_mean)

    mean_frames: dict[str, np.ndarray] = {}
    per_label: dict[str, Any] = {}
    for lab in PHASE_LABELS:
        g = groups.get(f"with object::{lab}")
        if not g or g["n_files_primary_shape"] == 0:
            per_label[lab] = {"status": "NO_PRIMARY_SHAPE_FILES"}
            continue
        stack, load_meta = load_group_stack(g)
        if stack is None:
            per_label[lab] = {"status": "LOAD_FAILED", "load": load_meta}
            continue
        stats = mean_and_uncertainty(stack)
        corrected, dark_info = dark_subtract(stats["mean"], dark_mean)
        mean_frames[lab] = corrected
        np.save(processed_dir / f"obj_mean__{lab}.npy", corrected)
        np.save(processed_dir / f"obj_std__{lab}.npy", stats["std"])
        per_label[lab] = {
            "status": "OK",
            "n_repeats": stats["n_repeats"],
            "mean_global": float(np.mean(corrected)),
            "dark": dark_info,
            "load": load_meta,
            "nominal_phase_rad": NOMINAL_PHASE_RAD[lab],
            "provisional_stage_um": PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.get(lab),
            "other_shapes_excluded": len(g.get("files_other_shapes") or []),
        }

    ordered_labs = ["0", "pi2", "pi", "3pi2"]
    if not all(lab in mean_frames for lab in ordered_labs):
        return {
            "status": "MISSING_FOUR_STEP",
            "per_label": per_label,
            "dark": dark_meta,
        }

    # Use frozen effective phase schedule when available; else nominal labels.
    frozen_steps = {
        row["acquisition_label"]: row.get("effective_phi_j_rad")
        for row in (frozen.get("per_step") or [])
    }
    schedule = []
    schedule_source = []
    for lab in ordered_labs:
        eff = frozen_steps.get(lab)
        if eff is not None and np.isfinite(eff):
            schedule.append(float(eff))
            schedule_source.append("frozen_effective_phi_j")
        else:
            schedule.append(float(NOMINAL_PHASE_RAD[lab]))
            schedule_source.append("nominal_folder_label")

    stack = np.stack([mean_frames[lab] for lab in ordered_labs], axis=0)
    # Shared pipeline: reconstruct with frozen schedule AND with nominal for comparison
    rec_frozen = reconstruct_phase_steps(stack, schedule, unwrap=False)
    rec_nominal = reconstruct_phase_steps(stack, FOUR_STEP_PHASES_RAD, unwrap=False)

    for prefix, rec in (("frozen", rec_frozen), ("nominal", rec_nominal)):
        np.save(processed_dir / f"obj_recon_{prefix}__dc_A.npy", rec.dc_A)
        np.save(processed_dir / f"obj_recon_{prefix}__modulation_B.npy", rec.modulation_B)
        np.save(processed_dir / f"obj_recon_{prefix}__visibility_V.npy", rec.visibility_V)
        np.save(
            processed_dir / f"obj_recon_{prefix}__phase_wrapped_rad.npy",
            rec.phase_wrapped_rad,
        )
        np.save(
            processed_dir / f"obj_recon_{prefix}__residual_rms.npy",
            rec.residual_rms,
        )

    h, w = rec_frozen.dc_A.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    summary = {
        "status": "OK",
        "dark": dark_meta,
        "per_label": per_label,
        "schedule_rad_frozen": schedule,
        "schedule_source": schedule_source,
        "metrics_frozen_schedule": {
            "A_median": float(np.median(rec_frozen.dc_A[y0:y1, x0:x1])),
            "B_median": float(np.median(rec_frozen.modulation_B[y0:y1, x0:x1])),
            "V_median": float(np.median(rec_frozen.visibility_V[y0:y1, x0:x1])),
            "residual_rms_median": float(
                np.median(rec_frozen.residual_rms[y0:y1, x0:x1])
            ),
            "method": rec_frozen.method,
        },
        "metrics_nominal_schedule": {
            "A_median": float(np.median(rec_nominal.dc_A[y0:y1, x0:x1])),
            "B_median": float(np.median(rec_nominal.modulation_B[y0:y1, x0:x1])),
            "V_median": float(np.median(rec_nominal.visibility_V[y0:y1, x0:x1])),
            "residual_rms_median": float(
                np.median(rec_nominal.residual_rms[y0:y1, x0:x1])
            ),
            "method": rec_nominal.method,
        },
        "rabbit_object_map": {
            "status": "NOT_FOUND_IN_EXPERIMENTAL_PACKAGE",
            "note": (
                "Freeze plan mentions a rabbit object for C1b, but no object "
                "amplitude/phase map is present under the experimental root. "
                "Object-dependent spatial simulation comparison is STOPPED."
            ),
        },
        "_rec_frozen": rec_frozen,
        "_rec_nominal": rec_nominal,
        "_mean_frames": mean_frames,
    }
    return summary


def run_no_object_simulation(
    *,
    frozen: dict[str, Any],
    sim_dir: Path,
    grid_n: int = 128,
) -> dict[str, Any]:
    """Simulate Config-C four-step acquisition at frozen effective phases.

    Uses stock/lab C profile with phase-step protocol. Pump laser power remains
    unresolved (Task 16); simulation uses profile defaults / placeholders.
    No experimental image resizing is performed.
    """
    from latents import LatentState
    from setup_params import resolve_setup_selection
    from simulate import simulate_forward
    from verification.phase_reconstruction.reconstruct import reconstruct_four_step

    sim_dir.mkdir(parents=True, exist_ok=True)

    # Prefer frozen effective φ_j for schedule; fall back to nominal four-step.
    per_step = frozen.get("per_step") or []
    by_lab = {r["acquisition_label"]: r for r in per_step}
    schedule = []
    for lab in ("0", "pi2", "pi", "3pi2"):
        row = by_lab.get(lab)
        if row and np.isfinite(row.get("effective_phi_j_rad", np.nan)):
            schedule.append(float(row["effective_phi_j_rad"]))
        else:
            schedule.append(float(NOMINAL_PHASE_RAD[lab]))
    schedule_t = tuple(schedule)

    try:
        setup = resolve_setup_selection("C").overridden(
            enable_phase_step_acquisition=True,
            phase_step_schedule_rad=schedule_t,
            enable_lens_rings=False,
            delta_z_um=0.0,
            lens_shift_x_um=0.0,
            lens_shift_y_um=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "SETUP_FAILED",
            "error": repr(exc),
            "schedule_rad": list(schedule_t),
        }

    n = int(grid_n)
    one = np.ones((n, n), dtype=np.float32)
    zero = np.zeros((n, n), dtype=np.float32)
    ref = {
        "M": one,
        "T": one,
        "phi": zero,
        "O_i": one,
        "D": zero,
        "target_size": n,
    }
    state = LatentState(
        setup=setup,
        num_frames=4,
        s_det=0.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )
    try:
        result = simulate_forward(ref, state, seed=18, detector_noise=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "SIMULATE_FAILED",
            "error": repr(exc),
            "schedule_rad": list(schedule_t),
        }

    # Prefer difference / plus frames as intensity proxy for reconstruction
    frames = np.asarray(result.frames_plus, dtype=np.float64)
    if frames.shape[0] != 4:
        return {
            "status": "UNEXPECTED_FRAME_COUNT",
            "n_frames": int(frames.shape[0]),
            "schedule_rad": list(schedule_t),
        }

    rec = reconstruct_phase_steps(frames, schedule_t, unwrap=False)
    np.save(sim_dir / "sim_noobj_frames_plus.npy", frames)
    np.save(sim_dir / "sim_noobj_dc_A.npy", rec.dc_A)
    np.save(sim_dir / "sim_noobj_modulation_B.npy", rec.modulation_B)
    np.save(sim_dir / "sim_noobj_visibility_V.npy", rec.visibility_V)
    np.save(sim_dir / "sim_noobj_phase_wrapped_rad.npy", rec.phase_wrapped_rad)

    return {
        "status": "OK",
        "schedule_rad": list(schedule_t),
        "grid_n": n,
        "profile": "C",
        "pump_power_note": "UNRESOLVED per Task 16; stock C / lab placeholders used",
        "detector_noise": False,
        "metrics": {
            "A_median": float(np.median(rec.dc_A)),
            "B_median": float(np.median(rec.modulation_B)),
            "V_median": float(np.median(rec.visibility_V)),
            "phase_circular_mean_rad": float(
                np.angle(np.mean(np.exp(1j * rec.phase_wrapped_rad)))
            ),
            "method": rec.method,
        },
        "object": "none_uniform_T",
        "with_object_simulation": {
            "status": "STOPPED",
            "reason": "rabbit object map not present in experimental package",
        },
        "_rec": rec,
        "_frames": frames,
    }


def compare_exp_sim_no_object(
    *,
    exp_maps: dict[str, np.ndarray],
    sim_rec,
    residuals_dir: Path,
) -> dict[str, Any]:
    """Compare scalar ROI metrics; do not resize experimental maps to sim grid."""
    residuals_dir.mkdir(parents=True, exist_ok=True)
    exp_A = exp_maps["dc_A"]
    exp_V = exp_maps["visibility_V"]
    exp_phi = exp_maps["phase_wrapped_rad"]
    h, w = exp_A.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    # Profiles: central row/col of experimental reconstruction (native size)
    row = exp_A.shape[0] // 2
    col = exp_A.shape[1] // 2
    np.save(residuals_dir / "exp_profile_row_A.npy", exp_A[row, :])
    np.save(residuals_dir / "exp_profile_col_A.npy", exp_A[:, col])
    np.save(residuals_dir / "exp_profile_row_V.npy", exp_V[row, :])
    np.save(residuals_dir / "exp_profile_row_phase.npy", exp_phi[row, :])

    sim_metrics = {
        "A_median": float(np.median(sim_rec.dc_A)),
        "B_median": float(np.median(sim_rec.modulation_B)),
        "V_median": float(np.median(sim_rec.visibility_V)),
        "phase_circular_mean_rad": float(
            np.angle(np.mean(np.exp(1j * sim_rec.phase_wrapped_rad)))
        ),
        "shape": list(sim_rec.dc_A.shape),
    }
    exp_metrics = {
        "A_median": float(np.median(exp_A[y0:y1, x0:x1])),
        "B_median": float(np.median(exp_maps["modulation_B"][y0:y1, x0:x1])),
        "V_median": float(np.median(exp_V[y0:y1, x0:x1])),
        "phase_circular_mean_rad": float(
            np.angle(np.mean(np.exp(1j * exp_phi[y0:y1, x0:x1])))
        ),
        "shape": list(exp_A.shape),
    }

    # Spatial residual only if shapes match (they generally will not)
    spatial = {
        "status": "SKIPPED_SHAPE_MISMATCH",
        "exp_shape": list(exp_A.shape),
        "sim_shape": list(sim_rec.dc_A.shape),
        "note": (
            "Experimental frames remain at native 419x455; simulation used "
            "an independent N×N grid. No resize/crop was applied to force a "
            "pixel-wise residual."
        ),
    }
    if exp_A.shape == sim_rec.dc_A.shape:
        dV = exp_V - sim_rec.visibility_V
        dphi = wrap_phase(exp_phi - sim_rec.phase_wrapped_rad)
        np.save(residuals_dir / "residual_visibility.npy", dV)
        np.save(residuals_dir / "residual_phase.npy", dphi)
        spatial = {
            "status": "OK",
            "visibility_rmse": float(np.sqrt(np.mean(dV * dV))),
            "phase_rmse_rad": float(np.sqrt(np.mean(dphi * dphi))),
        }

    return {
        "experimental_roi_metrics": exp_metrics,
        "simulation_metrics": sim_metrics,
        "visibility_abs_diff_median": abs(
            exp_metrics["V_median"] - sim_metrics["V_median"]
        ),
        "phase_circular_diff_rad": float(
            wrap_phase(
                exp_metrics["phase_circular_mean_rad"]
                - sim_metrics["phase_circular_mean_rad"]
            )
        ),
        "spatial_residual": spatial,
        "dimension_policy": "no_resize_experimental_to_match_simulation",
    }
