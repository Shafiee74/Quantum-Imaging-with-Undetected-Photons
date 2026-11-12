"""C_lab fixed crop camera zoom (display export only)."""
from __future__ import annotations

import numpy as np

from camera.c_lab_display_zoom import (
    apply_fixed_crop,
    crop_bounds_from_center,
    detect_signal_spot_center,
    export_zoom_config_from_setup,
    float_to_display_u16,
    resolve_fixed_crop_window,
)
from setup_params import resolve_setup_selection


def _spot_frame() -> np.ndarray:
    y, x = np.indices((1024, 1024), dtype=np.float64)
    return np.exp(-((x - 520) ** 2 + (y - 498) ** 2) / (2 * 45.0 ** 2))


def test_zoom_config_from_c_lab_yaml() -> None:
    cfg = export_zoom_config_from_setup(resolve_setup_selection("C_lab"))
    assert cfg.enabled is True
    assert cfg.export_width_px == 455
    assert cfg.export_height_px == 419
    assert cfg.resize_after_crop is False
    assert cfg.interpolation == "none"


def test_crop_exact_size_and_equals_slice() -> None:
    native = _spot_frame()
    cfg = export_zoom_config_from_setup(resolve_setup_selection("C_lab"))
    meta, bounds = resolve_fixed_crop_window(native, cfg)
    u16 = float_to_display_u16(native, float(native.max()))
    zoom = apply_fixed_crop(u16, bounds)
    assert zoom.shape == (419, 455)
    sl = u16[bounds["row0"] : bounds["row1"], bounds["col0"] : bounds["col1"]]
    assert np.array_equal(zoom, sl)
    assert zoom.dtype == np.uint16


def test_same_crop_for_second_frame() -> None:
    native_no = _spot_frame()
    native_rb = native_no * 0.7 + 0.05
    cfg = export_zoom_config_from_setup(resolve_setup_selection("C_lab"))
    _, bounds = resolve_fixed_crop_window(native_no, cfg)
    z1 = apply_fixed_crop(native_no, bounds)
    z2 = apply_fixed_crop(native_rb, bounds)
    assert z1.shape == z2.shape == (419, 455)
    b = bounds
    assert np.array_equal(z2, native_rb[b["row0"] : b["row1"], b["col0"] : b["col1"]])


def test_native_unchanged_after_export_ops() -> None:
    native = _spot_frame().copy()
    before = native.copy()
    cfg = export_zoom_config_from_setup(resolve_setup_selection("C_lab"))
    _, bounds = resolve_fixed_crop_window(native, cfg)
    apply_fixed_crop(native, bounds)
    assert np.array_equal(native, before)


def test_center_detection_finite() -> None:
    c = detect_signal_spot_center(_spot_frame())
    assert 0 <= c["center_row"] < 1024
    assert 0 <= c["center_col"] < 1024


def test_disabled_zoom_backward_compat() -> None:
    setup = resolve_setup_selection("C_lab").overridden(export_zoom_crop_enabled=False)
    cfg = export_zoom_config_from_setup(setup)
    assert cfg.enabled is False


def test_c_lab_frozen_slice_bounds() -> None:
    bounds = crop_bounds_from_center(
        511,
        511,
        out_height=419,
        out_width=455,
        native_height=1024,
        native_width=1024,
    )
    assert bounds == {
        "row0": 302,
        "row1": 721,
        "col0": 284,
        "col1": 739,
        "out_height": 419,
        "out_width": 455,
    }
