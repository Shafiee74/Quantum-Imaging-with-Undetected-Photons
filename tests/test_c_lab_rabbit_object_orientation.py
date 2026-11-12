"""C_lab rabbit object-plane orientation (production mapping)."""
from __future__ import annotations

import hashlib

import numpy as np

from object_maps.c_lab_production_object_map import (
    apply_rabbit_object_orientation_in_plane,
    c_lab_rabbit_transmission_grid,
)
from object_maps.c_lab_y1_rabbit_physical import (
    Y1_MASK_DEFAULT,
    load_c_lab_y1_object_map,
    resolved_aperture_mm,
)
from setup_params import resolve_setup_selection
from tools.c_lab_a6_2_calibrated_1024_audit import N


def test_y1_frozen_mask_sha256_unchanged() -> None:
    h = hashlib.sha256(Y1_MASK_DEFAULT.read_bytes()).hexdigest()
    assert h == "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"


def test_production_yaml_flip_x_enabled() -> None:
    setup = resolve_setup_selection("C_lab")
    assert setup.rabbit_object_flip_x is True
    assert setup.rabbit_object_flip_y is False
    assert setup.rabbit_object_rotate_180 is False


def test_flip_x_changes_grid_not_frozen_mask() -> None:
    setup = resolve_setup_selection("C_lab")
    loaded = load_c_lab_y1_object_map(setup)
    assert np.array_equal(loaded["mask"], np.load(Y1_MASK_DEFAULT))
    off = setup.overridden(rabbit_object_flip_x=False)
    on = setup.overridden(rabbit_object_flip_x=True)
    g0 = c_lab_rabbit_transmission_grid(off)
    g1 = c_lab_rabbit_transmission_grid(on)
    assert g0.shape == (N, N)
    assert not np.array_equal(g0, g1)


def test_aperture_mm_unchanged_after_flip_x() -> None:
    setup = resolve_setup_selection("C_lab")
    w, h = resolved_aperture_mm(setup)
    assert w == 7.285 and h == 9.850
    off = setup.overridden(rabbit_object_flip_x=False)
    on = setup.overridden(rabbit_object_flip_x=True)
    assert int(c_lab_rabbit_transmission_grid(off).sum()) == int(c_lab_rabbit_transmission_grid(on).sum())


def test_flip_x_is_object_mirror_not_array_only() -> None:
    setup = resolve_setup_selection("C_lab")
    loaded = load_c_lab_y1_object_map(setup)
    m0, sx0, sy0, _ = apply_rabbit_object_orientation_in_plane(
        loaded["mask"], loaded["x_um"], loaded["y_um"], setup.overridden(rabbit_object_flip_x=False)
    )
    m1, sx1, sy1, _ = apply_rabbit_object_orientation_in_plane(
        loaded["mask"], loaded["x_um"], loaded["y_um"], setup.overridden(rabbit_object_flip_x=True)
    )
    assert sx0[0] < 0 < sx0[-1]
    assert sx1[0] < 0 < sx1[-1]
    assert sx1[0] == -sx0[-1]
