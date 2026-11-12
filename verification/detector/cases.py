"""Controlled detector cases A–G using the production chain only."""
from __future__ import annotations

from typing import Any

import numpy as np

from detector import (
    _detector_pc_frame,
    _fixed_patterns,
    detector_stages_for_setup,
    spawn_qiup_rngs,
)
from latents import LatentState
from setup_params import SetupParams, resolve_forward
from verification.detector.analytical import analytical_expectations
from verification.detector.definitions import (
    CASE_IDS,
    MASTER_SEED,
    N_FRAMES,
    N_PIXELS,
    SIGNAL_PHOTONS,
)


def _base_setup(**overrides: Any) -> SetupParams:
    values: dict[str, Any] = {
        "detector_mode": "analog_emccd",
        "detector_output_port": "plus",
        "emccd_model": "register",
        "adc_bit_depth": 16,
        "use_physical_spdc_gain": True,
        "use_physical_dark": False,
        "prnu_sigma": 0.0,
        "hot_frac_at_s1": 0.0,
        "optical_background_photons": 0.0,
        "g_adu": 1.0,
        "bias_adu": 100.0,
        "max_adu": 65535.0,
        "full_well_e": 80000.0,
        "QE": 0.9,
        "read_noise_model": "datasheet_fixed",
        "read_noise0_e": 5.0,
        "dark0": 2.0,
        "dark_exp": 0.0,
        "cic0": 0.2,
    }
    values.update(overrides)
    return SetupParams.personal_lab_template().overridden(**values)


def _state(setup: SetupParams, *, em_gain: float, exposure_s: float = 0.05) -> LatentState:
    return LatentState(
        setup=setup,
        num_frames=1,
        exposure_s=float(exposure_s),
        em_gain=float(em_gain),
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def case_configs() -> dict[str, dict[str, Any]]:
    """Return setup/state/I configuration for each controlled case."""
    return {
        "ideal": {
            "label": "A. ideal detector",
            "I": SIGNAL_PHOTONS,
            "setup": _base_setup(
                dark0=0.0, cic0=0.0, read_noise0_e=0.0, QE=1.0
            ),
            "em_gain": 1.0,
        },
        "shot_noise": {
            "label": "B. shot noise only",
            "I": SIGNAL_PHOTONS,
            "setup": _base_setup(
                dark0=0.0, cic0=0.0, read_noise0_e=0.0, QE=1.0
            ),
            "em_gain": 1.0,
        },
        "read_noise": {
            "label": "C. read noise only",
            "I": 0.0,
            "setup": _base_setup(
                dark0=0.0, cic0=0.0, read_noise0_e=8.0, QE=1.0
            ),
            "em_gain": 1.0,
        },
        "dark": {
            "label": "D. dark current only",
            "I": 0.0,
            "setup": _base_setup(
                dark0=5.0, cic0=0.0, read_noise0_e=0.0, QE=1.0
            ),
            "em_gain": 1.0,
            "exposure_s": 0.2,
        },
        "cic": {
            "label": "E. CIC only",
            "I": 0.0,
            "setup": _base_setup(
                dark0=0.0, cic0=1.0, read_noise0_e=0.0, QE=1.0
            ),
            "em_gain": 1.0,
        },
        "gain": {
            "label": "F. gain/EM stage",
            "I": 5.0,
            "setup": _base_setup(
                dark0=0.0, cic0=0.0, read_noise0_e=0.0, QE=1.0
            ),
            "em_gain": 100.0,
        },
        "combined": {
            "label": "G. combined detector model",
            "I": SIGNAL_PHOTONS,
            "setup": _base_setup(
                dark0=2.0, cic0=0.2, read_noise0_e=5.0, QE=0.9
            ),
            "em_gain": 50.0,
            "exposure_s": 0.05,
        },
    }


def run_case(
    case_id: str,
    *,
    master_seed: int = MASTER_SEED,
    n_side: int = N_PIXELS,
    n_frames: int = N_FRAMES,
) -> dict[str, Any]:
    if case_id not in CASE_IDS:
        raise KeyError(case_id)
    cfg = case_configs()[case_id]
    setup = cfg["setup"]
    exposure = float(cfg.get("exposure_s", 0.05))
    state = _state(setup, em_gain=float(cfg["em_gain"]), exposure_s=exposure)
    I = np.full((n_side, n_side), float(cfg["I"]), dtype=np.float32)
    fwd = resolve_forward(setup, n_side, s_focus=0.0)

    # Fixed patterns once (PRNU=1, hot=0 by construction of setup)
    rngs0 = spawn_qiup_rngs(int(master_seed) + hash(case_id) % 10_000)
    prnu, hot = _fixed_patterns((n_side, n_side), state, rngs0["rng_detector"])
    # Force clean PRNU/hot for isolation (still using production sampler path)
    prnu = np.ones_like(prnu)
    hot = np.zeros_like(hot)

    frames = np.zeros((n_frames, n_side, n_side), dtype=np.float32)
    for i in range(n_frames):
        rng = spawn_qiup_rngs(int(master_seed) + 17 * (i + 1) + hash(case_id) % 1000)[
            "rng_plus"
        ]
        frames[i] = _detector_pc_frame(I, state, rng, prnu=prnu, hot=hot)

    flat = frames.reshape(n_frames, -1)
    # Pool all pixels×frames for distribution (iid under flat field)
    samples = flat.ravel()
    mean_s = float(np.mean(samples))
    var_s = float(np.var(samples, ddof=1))
    std_s = float(np.sqrt(var_s))

    expect = analytical_expectations(
        case_id,
        I_value=float(cfg["I"]),
        state=state,
        g_adu=float(fwd.g_adu),
        bias_adu=float(fwd.bias_adu),
        full_well_e=float(fwd.full_well_e),
    )

    abs_err_mean = abs(mean_s - expect["mean_adu"])
    rel_err_mean = abs_err_mean / max(abs(expect["mean_adu"]), 1e-9)
    # Variance: relative where meaningful
    abs_err_var = abs(var_s - expect["var_adu"])
    rel_err_var = abs_err_var / max(abs(expect["var_adu"]), 1e-9)

    return {
        "case_id": case_id,
        "label": cfg["label"],
        "I_photons": float(cfg["I"]),
        "em_gain": float(cfg["em_gain"]),
        "exposure_s": exposure,
        "n_side": n_side,
        "n_frames": n_frames,
        "n_samples": int(samples.size),
        "stages": list(detector_stages_for_setup(setup)),
        "simulated": {
            "mean_adu": mean_s,
            "var_adu": var_s,
            "std_adu": std_s,
            "min_adu": float(np.min(samples)),
            "max_adu": float(np.max(samples)),
            "fraction_negative_pre_bias_proxy": None,
        },
        "analytical": expect,
        "abs_error_mean_adu": float(abs_err_mean),
        "rel_error_mean": float(rel_err_mean),
        "abs_error_var_adu": float(abs_err_var),
        "rel_error_var": float(rel_err_var),
        "frames": frames,
        "samples_flat": samples,
        "setup_snapshot": {
            "QE": float(setup.QE),
            "dark0": float(setup.dark0),
            "cic0": float(setup.cic0),
            "read_noise0_e": float(setup.read_noise0_e),
            "g_adu": float(setup.g_adu),
            "bias_adu": float(setup.bias_adu),
            "emccd_model": str(setup.emccd_model),
            "detector_mode": str(setup.detector_mode),
        },
    }


def run_all_cases(
    *,
    master_seed: int = MASTER_SEED,
    n_side: int = N_PIXELS,
    n_frames: int = N_FRAMES,
) -> dict[str, Any]:
    return {
        case_id: run_case(
            case_id, master_seed=master_seed, n_side=n_side, n_frames=n_frames
        )
        for case_id in CASE_IDS
    }
