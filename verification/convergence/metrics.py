"""Per-grid-level metric extraction for U14 convergence."""
from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from benchmarks.targets import build_target
from setup_params import resolved_spatial_mapping
from simulate import simulate_forward
from verification.convergence.grids import GridLevel, setup_for_level
from verification.phase_reconstruction import (
    phase_error_metrics,
    reconstruct_phase_steps,
    remove_global_piston,
    circular_phase_error,
)
from verification.physics.u1_u9_runner import quiet_state
from verification.resolution.profiles import (
    average_rows,
    build_esf_profile,
    build_psf_profile,
    extract_central_row,
)
from verification.resolution.quantities import (
    intensity_map_from_result,
    visibility_map_from_result,
)
from verification.spatial.fov import analytical_gaussian_fov_diameter_um
from verification.spatial.analysis import pump_waist_from_setup


def _peak_normalize(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    m = float(np.max(np.abs(a)))
    return a / m if m > 0.0 else a


def _resample_to_axis(
    src: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> np.ndarray:
    """Bilinear resample of a map onto a destination centered grid."""
    from scipy.ndimage import map_coordinates

    # Build destination sample coordinates in source index space.
    # x_um = x0 + i * dx  =>  i = (x - x0) / dx
    sx = np.asarray(src_x, dtype=np.float64)
    sy = np.asarray(src_y, dtype=np.float64)
    dx = float(sx[1] - sx[0]) if sx.size > 1 else 1.0
    dy = float(sy[1] - sy[0]) if sy.size > 1 else 1.0
    X, Y = np.meshgrid(dst_x, dst_y)
    col = (X - sx[0]) / dx
    row = (Y - sy[0]) / dy
    coords = np.vstack([row.ravel(), col.ravel()])
    out = map_coordinates(
        np.asarray(src, dtype=np.float64), coords, order=1, mode="nearest"
    )
    return out.reshape(Y.shape)


def try_resample(
    src: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
) -> np.ndarray:
    try:
        return _resample_to_axis(src, src_x, src_y, dst_x, dst_y)
    except Exception:
        # Fallback without scipy: nearest-neighbor
        sx = np.asarray(src_x, dtype=np.float64)
        sy = np.asarray(src_y, dtype=np.float64)
        X, Y = np.meshgrid(dst_x, dst_y)
        ic = np.clip(np.searchsorted(sx, X) - 1, 0, sx.size - 1)
        ir = np.clip(np.searchsorted(sy, Y) - 1, 0, sy.size - 1)
        return np.asarray(src, dtype=np.float64)[ir, ic]


def evaluate_level(
    level: GridLevel,
    *,
    seed: int,
    nbar: float,
) -> dict[str, Any]:
    """Run forward + derived metrics for one grid level."""
    setup = setup_for_level(level.profile, level)
    sm = resolved_spatial_mapping(setup)
    waist = pump_waist_from_setup(setup)
    D_fov = analytical_gaussian_fov_diameter_um(waist)

    # --- transparent intensity / visibility ---
    t_trans = build_target("transparent", level.n, pixel_um=level.pixel_um)
    state = quiet_state(setup, nbar=nbar)
    state.num_frames = 1

    tracemalloc.start()
    t0 = time.perf_counter()
    r_trans = simulate_forward(
        t_trans.to_simulation_reference(), state, seed=seed, detector_noise=False
    )
    runtime_s = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_peak_mib = peak / (1024.0 * 1024.0)

    I = intensity_map_from_result(r_trans)
    Vmap = visibility_map_from_result(r_trans)
    vo = dict(r_trans.meta.get("visibility_observables") or {})

    # --- point PSF (intensity) ---
    t_pt = build_target("point", level.n, pixel_um=level.pixel_um)
    r_pt = simulate_forward(
        t_pt.to_simulation_reference(),
        quiet_state(setup, nbar=nbar),
        seed=seed,
        detector_noise=False,
    )
    I_pt = intensity_map_from_result(r_pt)
    row = int(np.argmin(np.abs(t_pt.y_um - float(t_pt.metadata.get("point_y_um", 0.0)))))
    psf = build_psf_profile(
        t_pt.x_um, extract_central_row(I_pt, row=row), quantity="intensity"
    )

    # --- knife ESF ---
    t_k = build_target("knife_edge_vertical", level.n, pixel_um=level.pixel_um)
    r_k = simulate_forward(
        t_k.to_simulation_reference(),
        quiet_state(setup, nbar=nbar),
        seed=seed,
        detector_noise=False,
    )
    esf = build_esf_profile(
        t_k.x_um,
        average_rows(intensity_map_from_result(r_k)),
        quantity="intensity",
        expected_edge_um=float(t_k.metadata.get("edge_x_um", 0.0)),
    )

    # --- phase reconstruction (analytical truth) ---
    phi_true = 0.37
    setup_ph = setup.overridden(enable_phase_step_acquisition=True)
    t_ph = build_target("transparent", level.n, pixel_um=level.pixel_um)
    # force constant phase
    ref = t_ph.to_simulation_reference()
    ref["phi"] = np.full((level.n, level.n), phi_true, dtype=np.float32)
    ref["O_i"] = (
        np.sqrt(np.asarray(ref["T"], dtype=np.float64))
        * np.exp(1j * phi_true)
    ).astype(np.complex64)
    st_ph = quiet_state(setup_ph, nbar=nbar)
    st_ph.num_frames = 4
    r_ph = simulate_forward(ref, st_ph, seed=seed, detector_noise=False)
    frames = np.stack(
        [
            np.asarray(r_ph.frames_D[i], dtype=np.float64)
            for i in range(4)
        ],
        axis=0,
    )
    rec = reconstruct_phase_steps(
        {
            "I0": frames[0],
            "I_pi_2": frames[1],
            "I_pi": frames[2],
            "I_3pi_2": frames[3],
        }
    )
    truth = np.full((level.n, level.n), phi_true, dtype=np.float64)
    err_metrics = phase_error_metrics(
        rec.phase_wrapped_rad, truth, remove_piston=True
    )
    aligned = remove_global_piston(rec.phase_wrapped_rad, truth)
    err_map = circular_phase_error(aligned, truth)

    return {
        "level": level.to_dict(),
        "runtime_s": float(runtime_s),
        "memory_peak_mib": float(mem_peak_mib),
        "memory_current_mib": float(current / (1024.0 * 1024.0)),
        "physical_FoV_pump_1e2_um": float(D_fov),
        "sampling_window_um": float(level.domain_um),
        "grid_n": int(level.n),
        "pixel_spacing_um": float(level.pixel_um),
        "x_um": t_trans.x_um,
        "y_um": t_trans.y_um,
        "intensity": I,
        "intensity_peak_normalized": _peak_normalize(I),
        "intensity_mean": float(np.mean(I)),
        "intensity_integral_approx": float(np.sum(I) * level.pixel_um**2),
        "visibility_map": Vmap,
        "visibility_map_mean": float(np.mean(Vmap)),
        "V_50_50": float(vo.get("V_50_50") or 0.0),
        "spatial_mapping": {
            "signed_magnification": float(sm["signed_object_to_camera_magnification"]),
            "sigma_object_um": float(sm["correlation_sigma_object_um"]),
            "sigma_camera_um": float(sm["correlation_sigma_camera_um"]),
            "reference_kind": "analytical_reference",
        },
        "psf_fwhm_um": psf.metric_value,
        "psf_profile": {
            "axis_um": psf.axis_um,
            "processed": psf.processed_profile,
            "original": psf.original_profile,
        },
        "esf_10_90_um": esf.metric_value,
        "esf_profile": {
            "axis_um": esf.axis_um,
            "processed": esf.processed_profile,
            "original": esf.original_profile,
        },
        "phase_true_rad": float(phi_true),
        "phase_reconstructed": rec.phase_wrapped_rad,
        "phase_error_map": err_map,
        "phase_rmse_rad": err_metrics["rmse_rad"],
        "phase_max_abs_rad": err_metrics["max_abs_rad"],
        "reconstruction_method": rec.method,
    }
