"""Generic uniform final camera collection benchmark (A–D)."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import optics
import qiup_digital_twin_full as qdtf
from diagnostics.fwhm_metrics import fwhm_horizontal_through_peak
from setup_params import (
    FINAL_CAMERA_COLLECTION_MODE_LEGACY,
    FINAL_CAMERA_COLLECTION_UNIFORM,
    GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM,
    GENERIC_BENCHMARK_FINAL_CAMERA_COLLECTION,
    resolve_setup_selection,
    resolved_final_camera_collection_policy,
    validate_setup_params,
)
from simulate import simulate_forward
from verification.physics.u1_u9_runner import DEFAULT_GRID, DEFAULT_NBAR, DEFAULT_SEED, quiet_state, uniform_object


def test_generic_abcd_uniform_collection_policy_yaml() -> None:
    for letter in ("A", "B", "C", "D"):
        setup = resolve_setup_selection(letter)
        assert resolved_final_camera_collection_policy(setup) == FINAL_CAMERA_COLLECTION_UNIFORM
        validate_setup_params(setup)


def test_generic_filter_still_3nm() -> None:
    for letter in ("A", "B", "C", "D"):
        setup = resolve_setup_selection(letter)
        assert np.isclose(float(setup.camera_filter_fwhm_nm), GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM)


def test_c_lab_unchanged() -> None:
    setup = resolve_setup_selection("C_lab")
    assert np.isclose(float(setup.camera_filter_fwhm_nm), 10.0)
    assert np.isclose(float(setup.pixel_um), 13.0)
    assert resolved_final_camera_collection_policy(setup) == FINAL_CAMERA_COLLECTION_MODE_LEGACY


def test_moments_invariant_switching_envelope_policy() -> None:
    ref = uniform_object(DEFAULT_GRID, T=1.0, phi=0.0)
    for letter in ("B", "D"):
        setup_u = resolve_setup_selection(letter)
        setup_u.delta_L_mm = 0.0
        state = quiet_state(setup_u, nbar=DEFAULT_NBAR)
        res_u = simulate_forward(ref, state, seed=DEFAULT_SEED, detector_noise=False)
        vo_u = res_u.meta.get("visibility_observables") or {}
        setup_l = resolve_setup_selection(letter)
        setup_l.delta_L_mm = 0.0
        setup_l.final_camera_collection_policy = FINAL_CAMERA_COLLECTION_MODE_LEGACY
        state_l = quiet_state(setup_l, nbar=DEFAULT_NBAR)
        with mock.patch(
            "setup_params.validate_generic_benchmark_final_camera_collection",
            lambda _s: None,
        ):
            res_l = simulate_forward(ref, state_l, seed=DEFAULT_SEED, detector_noise=False)
        vo_l = res_l.meta.get("visibility_observables") or {}
        for key in ("N_1", "N_2", "C_12_abs"):
            assert np.isclose(float(vo_u.get(key) or 0), float(vo_l.get(key) or 0), rtol=0, atol=1e-12)


def test_legacy_gaussian_restores_nf_intensity_scale() -> None:
    ref = uniform_object(DEFAULT_GRID, T=1.0, phi=0.0)
    setup_u = resolve_setup_selection("B")
    setup_u.delta_L_mm = 0.0
    state_u = quiet_state(setup_u, nbar=DEFAULT_NBAR)
    i_u = float(np.mean(simulate_forward(ref, state_u, seed=DEFAULT_SEED, detector_noise=False).ideal_signal_image))
    setup_l = resolve_setup_selection("B")
    setup_l.delta_L_mm = 0.0
    setup_l.final_camera_collection_policy = FINAL_CAMERA_COLLECTION_MODE_LEGACY
    state_l = quiet_state(setup_l, nbar=DEFAULT_NBAR)
    with mock.patch(
        "setup_params.validate_generic_benchmark_final_camera_collection",
        lambda _s: None,
    ):
        i_l = float(
            np.mean(
                simulate_forward(ref, state_l, seed=DEFAULT_SEED, detector_noise=False).ideal_signal_image
            )
        )
    assert i_u / i_l == pytest.approx(1.796877, rel=1e-4)


def test_monolith_parity_policy() -> None:
    setup = resolve_setup_selection("B")
    assert qdtf.resolved_final_camera_collection_policy(setup) == GENERIC_BENCHMARK_FINAL_CAMERA_COLLECTION


def test_fwhm_rejects_edge_clipped_flat_top() -> None:
    n = 24
    pitch = 16.0
    cx = (n - 1) / 2.0
    x = (np.arange(n) - cx) * pitch
    img = np.ones((n, n), dtype=np.float64)
    info = fwhm_horizontal_through_peak(img, x)
    assert info["status"] == "EDGE_CLIPPED"
    assert np.isnan(info["fwhm_um"])


def test_canonical_letters_abcd() -> None:
    assert resolve_setup_selection("A").setup_id == "two_crystal_2014_reference_v2"
    assert resolve_setup_selection("B").setup_id == "two_crystal_near_field_v1"
    assert resolve_setup_selection("C").setup_id == "single_crystal_far_field_v2"
    assert resolve_setup_selection("D").setup_id == "single_crystal_near_field_v2"
