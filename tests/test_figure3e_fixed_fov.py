"""Fixed-FoV numerical refinement (Figure 3e) regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.targets import build_target
from setup_params import resolve_setup_selection, validate_generic_benchmark_spectral_filter
from simulate import simulate_forward
from verification.convergence.definitions import NBAR, N_MID_DEFAULT, SEED
from verification.convergence.fixed_fov import (
    build_fixed_fov_levels,
    setup_for_fixed_fov,
    simulate_intensity_on_detector,
)
from verification.convergence.grids import build_explicit_n_ladder, setup_for_level
from verification.convergence.metrics import try_resample
from verification.physics.u1_u9_runner import quiet_state
from verification.resolution.quantities import intensity_map_from_result


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    r = np.sqrt(np.mean(b**2))
    return float(np.sqrt(np.mean(d**2)) / r) if r > 0 else 0.0


@pytest.mark.parametrize("profile", ["A", "B", "C", "D"])
def test_fixed_fov_camera_window_constant(profile: str) -> None:
    levels = build_fixed_fov_levels(profile, (48, 96, 192))
    fovs = {lev.n_num: lev.camera_fov_um for lev in levels}
    assert len(set(fovs.values())) == 1
    assert fovs[48] == pytest.approx(48 * 16.0, rel=0, abs=1e-9)


@pytest.mark.parametrize("profile", ["A", "B", "C", "D"])
def test_production_parity_at_n48(profile: str) -> None:
    """Internal N=48 with hardware pitch must match legacy production grid."""
    setup = resolve_setup_selection(profile)
    validate_generic_benchmark_spectral_filter(setup)
    legacy = build_explicit_n_ladder(profile, (48,), n_mid=N_MID_DEFAULT)[0]
    setup_leg = setup_for_level(profile, legacy)
    tgt = build_target("transparent", 48, pixel_um=legacy.pixel_um)
    state = quiet_state(setup_leg, nbar=NBAR)
    r_leg = simulate_forward(
        tgt.to_simulation_reference(), state, seed=SEED, detector_noise=False
    )
    i_leg = intensity_map_from_result(r_leg)

    ff = build_fixed_fov_levels(profile, (48,))[0]
    state_ff = quiet_state(setup_for_fixed_fov(ff), nbar=NBAR)
    ref = build_target("transparent", 48, pixel_um=ff.object_pixel_um)
    out = simulate_intensity_on_detector(
        ff, ref.to_simulation_reference(), state_ff, seed=SEED
    )
    i_ff = out["intensity_detector"]
    assert i_leg.shape == i_ff.shape
    err = _rel_l2(i_ff, i_leg)
    assert err < 0.02, f"parity rel L2={err}"


def test_c_lab_unchanged_by_verification_fields() -> None:
    from setup_params import internal_camera_sampling_pixel_um

    c = resolve_setup_selection("C_lab")
    assert getattr(c, "numerical_verification_internal_camera_pixel_um", None) is None
    assert internal_camera_sampling_pixel_um(c) == pytest.approx(float(c.pixel_um))


@pytest.mark.parametrize("profile", ["A", "B", "C", "D"])
def test_object_delta_sensitive_nonzero(profile: str) -> None:
    lev = build_fixed_fov_levels(profile, (96,))[0]
    setup = setup_for_fixed_fov(lev)
    state = quiet_state(setup, nbar=NBAR)
    clear = build_target("transparent", lev.n_num, pixel_um=lev.object_pixel_um)
    step = build_target("amplitude_step", lev.n_num, pixel_um=lev.object_pixel_um)
    ref0 = clear.to_simulation_reference()
    out0 = simulate_intensity_on_detector(lev, ref0, state, seed=SEED)
    ref1 = step.to_simulation_reference()
    out1 = simulate_intensity_on_detector(lev, ref1, state, seed=SEED + 1)
    delta = out1["intensity_detector"] - out0["intensity_detector"]
    assert float(np.std(delta)) > 1e-6
