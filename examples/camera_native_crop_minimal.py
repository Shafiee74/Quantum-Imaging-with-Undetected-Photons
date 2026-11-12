"""Minimal native camera crop examples (display/export only)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from camera import (
    apply_native_crop,
    crop_bounds_from_center,
    export_zoom_config_from_setup,
    native_crop_config_from_setup,
    resolve_crop_window,
    resolve_fixed_crop_window,
)
from setup_params import resolve_setup_selection


def generic_512_crop() -> None:
    native = np.random.default_rng(0).random((1024, 1024))
    bounds = crop_bounds_from_center(
        511,
        511,
        crop_height_px=512,
        crop_width_px=512,
        native_height=1024,
        native_width=1024,
    )
    zoom = apply_native_crop(native, bounds)
    print("generic 1024→512 shape:", zoom.shape)


def c_lab_from_yaml() -> None:
    setup = resolve_setup_selection("C_lab")
    cfg_yaml = native_crop_config_from_setup(setup)
    cfg_legacy = export_zoom_config_from_setup(setup)
    spot = np.exp(-((np.indices((1024, 1024))[1] - 511) ** 2 + (np.indices((1024, 1024))[0] - 511) ** 2) / (2 * 40.0 ** 2))
    _, bounds = resolve_crop_window(spot, cfg_yaml)
    _, bounds2 = resolve_fixed_crop_window(spot, cfg_legacy)
    assert bounds == bounds2
    out = apply_native_crop(spot, bounds)
    print("C_lab crop shape (H×W):", out.shape)
    print("C_lab bounds:", bounds)


if __name__ == "__main__":
    generic_512_crop()
    c_lab_from_yaml()
