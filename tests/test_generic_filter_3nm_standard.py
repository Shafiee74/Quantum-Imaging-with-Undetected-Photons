"""Regression: generic A–D use 3 nm filter; C_lab retains laboratory 10 nm."""
from __future__ import annotations

import numpy as np

from setup_params import (
    CAMERA_FILTER_PRESET_LAB,
    CAMERA_FILTER_PRESET_PAPER,
    GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM,
    resolve_setup_selection,
    validate_setup_params,
)


def test_generic_abcd_filter_3nm() -> None:
    for letter in ("A", "B", "C", "D"):
        setup = resolve_setup_selection(letter)
        assert setup.camera_filter_preset == CAMERA_FILTER_PRESET_PAPER
        assert np.isclose(
            float(setup.camera_filter_fwhm_nm),
            GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM,
        )


def test_monolithic_validate_generic_filter() -> None:
    import qiup_digital_twin_full as qdtf

    setup = resolve_setup_selection("C")
    validate_setup_params(setup)
    qdtf.validate_setup_params(setup)


def test_c_lab_filter_unchanged_10nm() -> None:
    setup = resolve_setup_selection("C_lab")
    assert setup.camera_filter_preset == CAMERA_FILTER_PRESET_LAB
    assert np.isclose(float(setup.camera_filter_fwhm_nm), 10.0)
    assert np.isclose(float(setup.pixel_um), 13.0)
    assert float(getattr(setup, "crystal_length_mm", 0)) == 2.0
