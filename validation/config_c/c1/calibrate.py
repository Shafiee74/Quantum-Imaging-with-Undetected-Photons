"""Task 18 Stage 2 — no-object C1 calibration (READ-ONLY experimental load)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.config_c.c1.identify import (
    NOMINAL_PHASE_RAD,
    PHASE_LABELS,
    PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN,
    IdentificationResult,
)
from verification.phase_reconstruction.constants import FOUR_STEP_PHASES_RAD
from verification.phase_reconstruction.reconstruct import reconstruct_phase_steps


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    return np.asarray(arr, dtype=np.float64)


def load_group_stack(
    group: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load primary-shape repeats → (n_rep, H, W) stack + provenance."""
    files = group.get("files_primary") or []
    meta: dict[str, Any] = {
        "n_requested": len(files),
        "loaded": [],
        "failed": [],
        "shape_filter": group.get("primary_shape"),
        "resize_applied": False,
        "crop_applied": False,
    }
    frames: list[np.ndarray] = []
    for f in files:
        path = Path(f["absolute_path"])
        try:
            img = _read_tiff(path)
            if img.ndim > 2:
                img = np.asarray(img[0], dtype=np.float64)
            h, w = int(img.shape[0]), int(img.shape[1])
            expect = group["primary_shape"]
            eh, ew = map(int, expect.lower().split("x"))
            if (h, w) != (eh, ew):
                meta["failed"].append(
                    {
                        "path": f["relative_path"],
                        "reason": f"shape {h}x{w} != primary {expect}",
                    }
                )
                continue
            frames.append(img)
            meta["loaded"].append(
                {
                    "relative_path": f["relative_path"],
                    "sha256": f["sha256"],
                    "shape": [h, w],
                }
            )
        except Exception as exc:  # noqa: BLE001 — record and continue
            meta["failed"].append(
                {"path": f["relative_path"], "reason": repr(exc)}
            )
    if not frames:
        return None, meta
    stack = np.stack(frames, axis=0)
    meta["n_loaded"] = int(stack.shape[0])
    meta["stack_shape"] = list(stack.shape)
    return stack, meta


def mean_and_uncertainty(stack: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(stack, axis=0)
    if stack.shape[0] > 1:
        std = np.std(stack, axis=0, ddof=1)
        sem = std / np.sqrt(stack.shape[0])
    else:
        std = np.zeros_like(mean)
        sem = np.zeros_like(mean)
    return {"mean": mean, "std": std, "sem": sem, "n_repeats": stack.shape[0]}


def dark_subtract(
    mean_img: np.ndarray,
    dark_mean: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {
        "dark_subtracted": False,
        "raw_vs_solis_preprocessing": "UNKNOWN_PER_TASK16",
    }
    if dark_mean is None:
        info["reason"] = "no primary-shape dark mean available"
        return mean_img, info
    if dark_mean.shape != mean_img.shape:
        info["reason"] = (
            f"dark shape {dark_mean.shape} != image shape {mean_img.shape}; "
            "skipped rather than resizing"
        )
        return mean_img, info
    # Guard against over-subtraction when dark/signal pairing is mismatched
    # (Task 16: raw vs Solis preprocessing unknown).
    if float(np.median(dark_mean)) >= float(np.median(mean_img)):
        info["reason"] = (
            "dark median >= signal median; subtraction skipped to avoid "
            "non-physical negative DC"
        )
        info["dark_median"] = float(np.median(dark_mean))
        info["signal_median"] = float(np.median(mean_img))
        return mean_img, info
    info["dark_subtracted"] = True
    return mean_img - dark_mean, info


def estimate_global_step_phase(
    mean_frames: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """No-object four-step reconstruction + intensity model check.

    Effective interferometer phases φ_j are taken from the **explicit folder
    labels** (0, π/2, π, 3π/2, 2π) — these are acquisition metadata tokens,
    not the ideal geometric mirror law. A, B, V, φ0 are estimated from the
    shared four-step pipeline. Intensity residuals test consistency of that
    schedule with the measured stack.
    """
    labels = [lab for lab in PHASE_LABELS if lab in mean_frames]
    if len(labels) < 3:
        return {"status": "INSUFFICIENT_STEPS", "labels": labels}

    ordered_labs = ["0", "pi2", "pi", "3pi2"]
    if not all(lab in mean_frames for lab in ordered_labs):
        return {
            "status": "MISSING_FOUR_STEP",
            "labels_present": labels,
        }

    stack = np.stack([mean_frames[lab] for lab in ordered_labs], axis=0)
    rec = reconstruct_phase_steps(stack, FOUR_STEP_PHASES_RAD, unwrap=False)
    phi_map = rec.phase_wrapped_rad
    h, w = phi_map.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    roi = np.s_[y0:y1, x0:x1]
    phi0_eff = float(np.angle(np.mean(np.exp(1j * phi_map[roi]))))
    A_med = float(np.median(rec.dc_A[roi]))
    B_med = float(np.median(rec.modulation_B[roi]))
    V_med = float(np.median(rec.visibility_V[roi]))

    per_step: list[dict[str, Any]] = []
    intensity_residuals: list[float] = []
    for lab in labels:
        Ij = float(np.median(mean_frames[lab][roi]))
        # Effective φ_j := folder nominal label (acquisition token).
        # This is NOT φ = 4π Δz/λ_i.
        phi_j = float(NOMINAL_PHASE_RAD[lab])
        I_model = A_med + B_med * float(np.cos(phi_j + phi0_eff))
        resid = Ij - I_model
        intensity_residuals.append(resid)
        per_step.append(
            {
                "acquisition_label": lab,
                "nominal_phase_rad": phi_j,
                "effective_phi_j_rad": phi_j,
                "effective_phi_j_source": "folder_acquisition_label",
                "provisional_stage_um": PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.get(lab),
                "median_intensity": Ij,
                "model_intensity": I_model,
                "intensity_residual": resid,
            }
        )

    return {
        "status": "OK",
        "labels": labels,
        "A_median": A_med,
        "B_median": B_med,
        "V_median": V_med,
        "phi0_circular_mean_rad": phi0_eff,
        "reconstruction_method": rec.method,
        "intensity_model_rmse": float(
            np.sqrt(np.mean(np.square(intensity_residuals)))
        ),
        "effective_phi_j_policy": (
            "Folder tokens are the acquisition phase labels; ideal geometric "
            "mirror law is not used to assign φ_j."
        ),
        "per_step": per_step,
        "maps": {
            "dc_A": rec.dc_A,
            "modulation_B": rec.modulation_B,
            "visibility_V": rec.visibility_V,
            "phase_wrapped_rad": rec.phase_wrapped_rad,
            "residual_rms": rec.residual_rms,
        },
        "roi_box": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
    }


def fit_stage_to_phase(
    per_step: list[dict[str, Any]],
) -> dict[str, Any]:
    """Empirical fit: provisional stage_um → effective φ_j (folder labels).

    Does **not** impose φ = 4π Δz / λ_i. Ideal geometric slope is returned
    separately as a reference line only.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in per_step:
        z = row.get("provisional_stage_um")
        phi = row.get("effective_phi_j_rad")
        if z is None or phi is None or not np.isfinite(phi):
            continue
        xs.append(float(z))
        ys.append(float(phi))
    if len(xs) < 2:
        return {"status": "INSUFFICIENT_POINTS", "n": len(xs)}

    order = np.argsort(xs)
    x = np.asarray(xs, dtype=np.float64)[order]
    y = np.asarray(ys, dtype=np.float64)[order]
    y_unw = np.unwrap(y)
    coef = np.polyfit(x, y_unw, 1)
    y_hat = np.polyval(coef, x)
    resid = y_unw - y_hat
    rmse = float(np.sqrt(np.mean(resid * resid)))

    lambda_i_um = 1.55
    ideal_slope = 4.0 * np.pi / lambda_i_um

    return {
        "status": "OK",
        "model": "phi_eff_rad = p0 + p1 * stage_um",
        "p0_rad": float(coef[1]),
        "p1_rad_per_um": float(coef[0]),
        "rmse_rad": rmse,
        "n_points": int(x.size),
        "stage_um": x.tolist(),
        "phi_eff_unwrapped_rad": y_unw.tolist(),
        "phi_eff_wrapped_rad": y.tolist(),
        "residuals_rad": resid.tolist(),
        "ideal_geometric_slope_rad_per_um": float(ideal_slope),
        "ideal_geometric_law": "phi = 4*pi*delta_z_um / lambda_i_um (reference only)",
        "lambda_i_um_reference": lambda_i_um,
        "forced_ideal_law": False,
        "stage_um_provenance": "PROVISIONAL_FROM_FREEZE_PLAN",
        "phi_j_provenance": "folder_acquisition_label",
        "note": (
            "Slope reflects experimenter-reported 25 µm ≈ quarter-wave "
            "label sequence, not optical geometry."
        ),
    }


def run_no_object_calibration(
    identification: IdentificationResult,
    *,
    processed_dir: Path,
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    groups = identification.groups

    # Dark mean (primary shape only)
    dark_group = groups.get("witout object::dark")
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
            np.save(processed_dir / "noobj_dark_mean.npy", dark_mean)
            np.save(processed_dir / "noobj_dark_std.npy", stats["std"])

    mean_frames: dict[str, np.ndarray] = {}
    per_label: dict[str, Any] = {}
    for lab in PHASE_LABELS:
        g = groups.get(f"witout object::{lab}")
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
        np.save(processed_dir / f"noobj_mean__{lab}.npy", corrected)
        np.save(processed_dir / f"noobj_std__{lab}.npy", stats["std"])
        np.save(processed_dir / f"noobj_sem__{lab}.npy", stats["sem"])
        per_label[lab] = {
            "status": "OK",
            "n_repeats": stats["n_repeats"],
            "mean_global": float(np.mean(corrected)),
            "std_global_of_mean_map": float(np.std(corrected)),
            "repeat_mean_std": float(np.mean(stats["std"])),
            "dark": dark_info,
            "load": load_meta,
            "nominal_phase_rad": NOMINAL_PHASE_RAD[lab],
            "provisional_stage_um": PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.get(lab),
        }

    calib = estimate_global_step_phase(mean_frames)
    stage_fit = None
    if calib.get("status") == "OK":
        stage_fit = fit_stage_to_phase(calib["per_step"])
        # persist maps
        for name, arr in calib["maps"].items():
            np.save(processed_dir / f"noobj_recon__{name}.npy", arr)

    # ref group (indexed only)
    ref_g = groups.get("witout object::ref")
    ref_info = {
        "n_primary": 0 if ref_g is None else ref_g["n_files_primary_shape"],
        "note": "Indexed; not used to alter phase-step calibration.",
    }

    result = {
        "object_token": "witout object",
        "dark": dark_meta,
        "ref": ref_info,
        "per_label": per_label,
        "four_step_calibration": {
            k: v
            for k, v in calib.items()
            if k != "maps"
        },
        "stage_to_phase_fit": stage_fit,
        "targets_stopped": [
            z
            for z, info in identification.stage_um_status["per_target"].items()
            if info["status"] == "NO_DATA_IDENTIFIED"
        ],
    }
    # attach maps separately for freezing (not JSON-serialized as arrays here)
    result["_maps"] = calib.get("maps")
    return result
