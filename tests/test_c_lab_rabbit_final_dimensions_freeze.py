"""Guards final Y1 rabbit caliper dimensions in production C_lab."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from object_maps.c_lab_y1_rabbit_physical import (
    Y1_MASK_DEFAULT,
    load_c_lab_y1_object_map,
    resolved_aperture_mm,
)
from setup_params import resolve_setup_selection

ROOT = Path(__file__).resolve().parents[1]


def test_c_lab_yaml_rabbit_aperture_mm() -> None:
    setup = resolve_setup_selection("C_lab")
    w, h = resolved_aperture_mm(setup)
    assert w == 7.285
    assert h == 9.850


def test_y1_mask_bytes_unchanged() -> None:
    h = hashlib.sha256(Y1_MASK_DEFAULT.read_bytes()).hexdigest()
    assert h == "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"


def test_calibration_bbox_and_footprint() -> None:
    setup = resolve_setup_selection("C_lab")
    loaded = load_c_lab_y1_object_map(setup)
    cal = loaded["physical_calibration"]
    assert cal["bbox"]["width_px"] == 214
    assert cal["bbox"]["height_px"] == 257
    foot = loaded["predicted_camera_footprint"]
    assert abs(foot["camera_aperture_width_mm"] - 7.285 * foot["M_magnitude"]) < 1e-9
    assert np.array_equal(loaded["mask"], np.load(Y1_MASK_DEFAULT))
