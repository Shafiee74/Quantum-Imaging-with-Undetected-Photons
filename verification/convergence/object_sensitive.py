"""Knife-edge R = A_mod_obj/A_mod_ref on fixed physical detector (Figure 3e)."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from benchmarks.targets import PHYSICAL_TRUTH, build_target
from latents import LatentState
from setup_params import resolved_spatial_mapping, validate_generic_benchmark_final_camera_collection, validate_generic_benchmark_spectral_filter
from simulate import simulate_forward
from verification.convergence.definitions import NBAR, SEED
from verification.convergence.fixed_fov import (
    FixedFoVLevel,
    bin_area_average_to_detector,
    detector_axes_um,
    setup_for_fixed_fov,
)
from verification.resolution.profiles import build_esf_profile

ILLUM_FRAC = 0.05
EDGE_X_UM = float(PHYSICAL_TRUTH["knife_edge_offset_um"])


def modulation_from_four_frames(frames_plus: np.ndarray) -> dict[str, np.ndarray]:
    I0 = np.asarray(frames_plus[0], dtype=np.float64)
    I90 = np.asarray(frames_plus[1], dtype=np.float64)
    I180 = np.asarray(frames_plus[2], dtype=np.float64)
    I270 = np.asarray(frames_plus[3], dtype=np.float64)
    a = 0.5 * (I0 - I180)
    b = 0.5 * (I90 - I270)
    amod = np.sqrt(a * a + b * b)
    return {"a": a, "b": b, "A_mod": amod, "I0": I0, "I90": I90, "I180": I180, "I270": I270}


def reference_normalized_response(A_obj: np.ndarray, A_ref: np.ndarray) -> dict[str, Any]:
    thr = ILLUM_FRAC * float(np.max(A_ref))
    valid = A_ref >= thr
    R = np.full_like(A_ref, np.nan, dtype=np.float64)
    np.divide(A_obj, A_ref, out=R, where=valid)
    return {
        "R": R,
        "A_mod_obj": A_obj,
        "A_mod_ref": A_ref,
        "valid_mask": valid,
        "threshold": thr,
        "valid_pixel_count": int(np.sum(valid)),
        "valid_fraction": float(np.mean(valid)),
    }


def _bin_frames(frames: np.ndarray, n_out: int) -> np.ndarray:
    out = []
    for i in range(frames.shape[0]):
        f = np.asarray(frames[i], dtype=np.float64)
        if f.shape[0] == n_out:
            out.append(f)
        else:
            out.append(bin_area_average_to_detector(f, n_out=n_out))
    return np.stack(out, axis=0)


def _seed_for(profile: str, target_id: str, extra: int = 0) -> int:
    # Python's built-in hash is randomized between interpreter processes and
    # therefore cannot be used in a reproducible evidence pipeline.
    target_offset = int.from_bytes(
        hashlib.sha256(target_id.encode("utf-8")).digest()[:4], "big"
    ) % 997
    return (
        int(SEED)
        + 7000 * (ord(profile[0]) - ord("A"))
        + target_offset
        + int(extra)
    )


def simulate_knife_R_on_detector(
    level: FixedFoVLevel,
    *,
    profile: str,
) -> dict[str, Any]:
    setup0 = setup_for_fixed_fov(level)
    validate_generic_benchmark_spectral_filter(setup0)
    validate_generic_benchmark_final_camera_collection(setup0)
    setup = setup0.overridden(enable_phase_step_acquisition=True)
    sm = resolved_spatial_mapping(setup)
    M = float(sm["signed_object_to_camera_magnification"])
    state = LatentState(
        setup=setup,
        mean_photons_per_mode=float(NBAR),
        num_frames=4,
        exposure_s=0.02,
        em_gain=1.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )
    n_num = level.n_num
    p_obj = level.object_pixel_um
    ref_tgt = build_target("transparent", n_num, pixel_um=p_obj)
    knife_tgt = build_target("knife_edge_vertical", n_num, pixel_um=p_obj)
    seed_base = _seed_for(profile, "knife_edge_vertical")
    r_ref = simulate_forward(ref_tgt.to_simulation_reference(), state, seed=seed_base, detector_noise=False)
    r_kn = simulate_forward(knife_tgt.to_simulation_reference(), state, seed=seed_base + 1, detector_noise=False)
    frames_ref = _bin_frames(np.asarray(r_ref.frames_plus), level.n_cam)
    frames_kn = _bin_frames(np.asarray(r_kn.frames_plus), level.n_cam)
    mod_ref = modulation_from_four_frames(frames_ref)
    mod_kn = modulation_from_four_frames(frames_kn)
    rn = reference_normalized_response(mod_kn["A_mod"], mod_ref["A_mod"])
    x_cam, y_cam = detector_axes_um(level)
    x_obj = x_cam / M
    return {
        **rn,
        "x_cam_um": x_cam,
        "y_cam_um": y_cam,
        "x_obj_um": x_obj,
        "signed_M": M,
        "object_T_knife": np.asarray(knife_tgt.T, dtype=np.float64),
        "level": level.to_dict(),
        "profile": profile,
    }


def central_strip(field: np.ndarray, *, half_width: int = 2) -> np.ndarray:
    h = field.shape[0]
    r0, r1 = max(0, h // 2 - half_width), min(h, h // 2 + half_width + 1)
    return np.nanmean(field[r0:r1, :], axis=0)


def edge_roi_mask(x_obj: np.ndarray, *, half_width_um: float = 120.0) -> np.ndarray:
    return np.abs(x_obj) <= float(half_width_um)


def weighted_rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    d = a[m] - b[m]
    ref = b[m]
    rms_d = float(np.sqrt(np.mean(d * d)))
    rms_r = float(np.sqrt(np.mean(ref * ref)))
    return rms_d / rms_r if rms_r > 0 else float("nan")


def esf_width_um(R: np.ndarray, x_obj: np.ndarray, valid: np.ndarray) -> float | None:
    line = central_strip(np.where(valid, R, np.nan))
    fit = build_esf_profile(
        x_obj,
        line,
        quantity="modulation_ratio",
        expected_edge_um=EDGE_X_UM,
    )
    return fit.metric_value


def sensitivity_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
    R = run["R"]
    valid = run["valid_mask"]
    line = central_strip(np.where(valid, R, np.nan))
    return {
        "R_std_global": float(np.nanstd(R)),
        "R_span_central_strip": float(np.nanmax(line) - np.nanmin(line)),
        "valid_fraction": run["valid_fraction"],
        "valid_pixel_count": run["valid_pixel_count"],
        "A_mod_ref_max": float(np.max(run["A_mod_ref"])),
        "passes_spatial_sensitivity": bool(
            run["valid_fraction"] > 0.05
            and np.nanstd(R) > 1.0e-3
            and (float(np.nanmax(line) - np.nanmin(line)) > 0.05)
        ),
    }
