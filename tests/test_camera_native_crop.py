"""Shared native camera crop (general + C_lab golden bounds)."""
from __future__ import annotations

import numpy as np
import pytest

from camera.native_crop import (
    CropBoundsValidationError,
    NativeCropConfig,
    apply_native_crop,
    crop_bounds_from_center,
    export_gray16_tiff,
    float_to_display_u16,
    image_geometric_center,
    native_crop_config_from_setup,
    resolve_crop_window,
)
from setup_params import SetupParams, resolve_setup_selection


def _gaussian_spot(h: int, w: int, cr: float, cc: float, sigma: float = 45.0) -> np.ndarray:
    y, x = np.indices((h, w), dtype=np.float64)
    return np.exp(-((x - cc) ** 2 + (y - cr) ** 2) / (2 * sigma ** 2))


def test_general_crop_matches_numpy_slice() -> None:
    native = _gaussian_spot(2048, 2048, 1023.5, 1023.5)
    cfg = NativeCropConfig(
        enabled=True,
        mode="fixed_center_native_crop",
        center_mode="image_geometric_center",
        crop_width_px=1024,
        crop_height_px=1024,
        center_row_px=None,
        center_col_px=None,
        native_shape_hw=(2048, 2048),
        preserve_native_frame=True,
        resize=False,
        interpolation="none",
        export_dtype="uint16",
        export_colormap=None,
    )
    _, bounds = resolve_crop_window(native, cfg)
    cropped = apply_native_crop(native, bounds)
    assert cropped.shape == (1024, 1024)
    sl = native[bounds["row0"] : bounds["row1"], bounds["col0"] : bounds["col1"]]
    assert np.array_equal(cropped, sl)


def test_arbitrary_crop_dimensions() -> None:
    native = np.arange(1024 * 1024, dtype=np.float64).reshape(1024, 1024)
    bounds = crop_bounds_from_center(
        511,
        511,
        crop_height_px=480,
        crop_width_px=640,
        native_height=1024,
        native_width=1024,
    )
    out = apply_native_crop(native, bounds)
    assert out.shape == (480, 640)


def test_strict_rejects_out_of_frame_center() -> None:
    with pytest.raises(CropBoundsValidationError):
        crop_bounds_from_center(
            10,
            10,
            crop_height_px=419,
            crop_width_px=455,
            native_height=1024,
            native_width=1024,
        )


def test_c_lab_golden_bounds_511_center() -> None:
    bounds = crop_bounds_from_center(
        511,
        511,
        out_height=419,
        out_width=455,
        native_height=1024,
        native_width=1024,
    )
    assert bounds["row0"] == 302 and bounds["row1"] == 721
    assert bounds["col0"] == 284 and bounds["col1"] == 739
    assert bounds["out_height"] == 419 and bounds["out_width"] == 455


def test_c_lab_production_window_on_simulated_spot() -> None:
    native = _gaussian_spot(1024, 1024, 511.4, 511.4)
    cfg = native_crop_config_from_setup(resolve_setup_selection("C_lab"))
    meta, bounds = resolve_crop_window(native, cfg)
    assert bounds["row0"] == 302 and bounds["col1"] == 739
    assert meta["center_row"] == 511 and meta["center_col"] == 511


def test_same_frozen_bounds_all_conditions() -> None:
    ref = _gaussian_spot(1024, 1024, 511.0, 511.0)
    other = ref * 0.6 + 0.02
    cfg = native_crop_config_from_setup(resolve_setup_selection("C_lab"))
    _, b0 = resolve_crop_window(ref, cfg)
    z1 = apply_native_crop(ref, b0)
    z2 = apply_native_crop(other, b0)
    assert z1.shape == z2.shape == (419, 455)
    assert np.array_equal(z2, other[b0["row0"] : b0["row1"], b0["col0"] : b0["col1"]])


def test_default_setup_crop_disabled() -> None:
    cfg = native_crop_config_from_setup(SetupParams())
    assert cfg.enabled is False


def test_gray16_export_single_channel(tmp_path) -> None:
    u16 = float_to_display_u16(np.ones((8, 10)) * 0.5, 1.0)
    assert u16.dtype == np.uint16
    path = tmp_path / "t.tif"
    export_gray16_tiff(path, u16)
    import tifffile

    loaded = tifffile.imread(path)
    assert loaded.ndim == 2
    assert loaded.dtype == np.uint16


def test_geometric_center_1024() -> None:
    assert image_geometric_center(1024, 1024) == (511, 511)
