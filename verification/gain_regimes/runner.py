"""Gain / pump sweep runner for Task 14."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from latents import LatentState
from setup_params import resolve_setup_selection
from simulate import simulate_forward
from verification.gain_regimes.approximation import (
    abs_approx_error,
    find_approximation_thresholds,
    n_approx_r2,
    n_exact,
    reference_equivalent_pump_mW,
    rel_approx_error,
)
from verification.gain_regimes.definitions import (
    APPROX_ABS_TOLERANCES,
    APPROX_REL_TOLERANCES,
    DETECTOR_EM_GAIN,
    DETECTOR_EXPOSURE_S,
    GRID_N,
    MASTER_SEED,
    NUM_FRAMES,
    PROFILES,
    R_SWEEP,
    REFERENCE_K_PER_SQRT_MW,
)
from verification.monte_carlo.channels import make_phase_reference


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quiet_state(
    setup: Any,
    *,
    r: float,
    detector_noise: bool,
) -> LatentState:
    return LatentState(
        setup=setup,
        spdc_gain_r=float(r),
        mean_photons_per_mode=None,
        num_frames=int(NUM_FRAMES),
        exposure_s=float(DETECTOR_EXPOSURE_S),
        em_gain=float(DETECTOR_EM_GAIN if detector_noise else 1.0),
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def _uniform_ref(setup: Any, n: int) -> dict[str, Any]:
    from setup_params import resolved_spatial_mapping

    pixel_um = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    return make_phase_reference(n, pixel_um, phi_rad=0.0)


def run_sweep_point(
    profile: str,
    r: float,
    *,
    detector_noise: bool,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
    point_index: int = 0,
) -> dict[str, Any]:
    """One gain point through production ``simulate_forward``."""
    setup = resolve_setup_selection(profile)
    state = _quiet_state(setup, r=r, detector_noise=detector_noise)
    ref = _uniform_ref(setup, n)
    # Distinct seeds for clean vs detector channels and sweep index
    seed = (
        int(master_seed)
        + (10_000 if detector_noise else 0)
        + int(point_index)
    )
    result = simulate_forward(
        ref, state, seed=seed, detector_noise=bool(detector_noise)
    )

    vo = dict(result.meta.get("visibility_observables") or {})
    op = dict(result.meta.get("spdc_operating_point") or {})

    n_sinh = n_exact(r)
    n_r2 = n_approx_r2(r)
    abs_err = abs_approx_error(r)
    rel_err = rel_approx_error(r)
    p_ref = reference_equivalent_pump_mW(r)

    # Detector counts (only meaningful / populated when detector enabled)
    plus = np.asarray(result.frames_plus[0], dtype=np.float64)
    minus = np.asarray(result.frames_minus[0], dtype=np.float64)
    ideal = np.asarray(result.ideal_signal_image, dtype=np.float64)
    det_img = getattr(result, "detector_signal_image", None)
    det = (
        np.asarray(det_img, dtype=np.float64)
        if det_img is not None
        else plus
    )

    row: dict[str, Any] = {
        "profile": profile,
        "channel": "detector_enabled" if detector_noise else "clean_optical",
        "detector_noise": bool(detector_noise),
        "r": float(r),
        "n_sinh2": float(n_sinh),
        "n_approx_r2": float(n_r2),
        "abs_approx_error": float(abs_err),
        "rel_approx_error": float(rel_err),
        "nbar_production_NL1": float(
            result.meta.get("nbar_NL1")
            if result.meta.get("nbar_NL1") is not None
            else op.get("mean_photons_per_mode_NL1") or n_sinh
        ),
        "nbar_production_NL2": float(
            result.meta.get("nbar_NL2")
            if result.meta.get("nbar_NL2") is not None
            else op.get("mean_photons_per_mode_NL2") or n_sinh
        ),
        "gain_r_production_1": float(vo.get("gain_r_1") or r),
        "gain_r_production_2": float(vo.get("gain_r_2") or r),
        "pump_power_laser_mW": float(
            result.meta.get("pump_power_laser_mW")
            if result.meta.get("pump_power_laser_mW") is not None
            else op.get("pump_power_laser_mW") or float("nan")
        ),
        "pump_power_NL1_mW": float(
            result.meta.get("pump_power_NL1_mW")
            if result.meta.get("pump_power_NL1_mW") is not None
            else op.get("pump_power_NL1_mW") or float("nan")
        ),
        "pump_power_NL2_mW": float(
            result.meta.get("pump_power_NL2_mW")
            if result.meta.get("pump_power_NL2_mW") is not None
            else op.get("pump_power_NL2_mW") or float("nan")
        ),
        "reference_equivalent_pump_mW": (
            float(p_ref) if p_ref is not None else float("nan")
        ),
        "reference_k_per_sqrt_mW": float(REFERENCE_K_PER_SQRT_MW),
        "N_1": float(vo.get("N_1") or 0.0),
        "N_2": float(vo.get("N_2") or 0.0),
        "C_12_abs": float(vo.get("C_12_abs") or 0.0),
        "visibility_V_50_50": float(vo.get("V_50_50") or 0.0),
        "visibility_image_mean": float(
            vo.get("interference_image_visibility_mean") or 0.0
        ),
        "ideal_signal_mean": float(np.mean(ideal)),
        "seed_int": int(seed),
    }

    if detector_noise:
        row.update(
            {
                "detector_counts_mean_plus": float(np.mean(plus)),
                "detector_counts_mean_minus": float(np.mean(minus)),
                "detector_counts_mean_signal": float(np.mean(det)),
                "detector_counts_std_plus": float(np.std(plus, ddof=1)),
                "detector_counts_std_minus": float(np.std(minus, ddof=1)),
                "em_gain": float(DETECTOR_EM_GAIN),
            }
        )
    else:
        row.update(
            {
                "detector_counts_mean_plus": float("nan"),
                "detector_counts_mean_minus": float("nan"),
                "detector_counts_mean_signal": float("nan"),
                "detector_counts_std_plus": float("nan"),
                "detector_counts_std_minus": float("nan"),
                "em_gain": 1.0,
            }
        )

    # Keep small arrays for evidence (frame 0 maps)
    row["_arrays"] = {
        "ideal_signal_image": ideal,
        "frames_plus_0": plus,
        "frames_minus_0": minus,
    }
    return row


def run_gain_regime_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    r_sweep: np.ndarray | None = None,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
) -> dict[str, Any]:
    rs = np.asarray(R_SWEEP if r_sweep is None else r_sweep, dtype=np.float64)
    suite: dict[str, Any] = {
        "generated_utc": utc_now(),
        "master_seed": int(master_seed),
        "grid_n": int(n),
        "r_sweep": [float(x) for x in rs],
        "profiles": {},
        "clean_rows": [],
        "detector_rows": [],
        "thresholds": find_approximation_thresholds(
            rs,
            abs_tolerances=APPROX_ABS_TOLERANCES,
            rel_tolerances=APPROX_REL_TOLERANCES,
        ),
    }

    for profile in profiles:
        clean_pts = []
        det_pts = []
        for i, r in enumerate(rs):
            clean = run_sweep_point(
                profile,
                float(r),
                detector_noise=False,
                master_seed=master_seed,
                n=n,
                point_index=i,
            )
            det = run_sweep_point(
                profile,
                float(r),
                detector_noise=True,
                master_seed=master_seed,
                n=n,
                point_index=i,
            )
            clean_pts.append(clean)
            det_pts.append(det)
            suite["clean_rows"].append(
                {k: v for k, v in clean.items() if k != "_arrays"}
            )
            suite["detector_rows"].append(
                {k: v for k, v in det.items() if k != "_arrays"}
            )
        suite["profiles"][profile] = {
            "clean": clean_pts,
            "detector_enabled": det_pts,
        }
    return suite
