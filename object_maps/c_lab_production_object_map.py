"""C_lab production rabbit transmission grid (Y1 physical calibration + orientation)."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from benchmarks.targets.geometry import centered_axis_um
from object_maps.c_lab_y1_rabbit_physical import load_c_lab_y1_object_map
from setup_params import SetupParams, resolve_setup_selection


def apply_rabbit_object_orientation_in_plane(
    mask: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    setup: SetupParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply documented object-coordinate orientation (never mutates frozen Y1 file)."""
    m = np.asarray(mask, dtype=np.float64)
    sx = np.asarray(x_um, dtype=np.float64).copy()
    sy = np.asarray(y_um, dtype=np.float64).copy()
    meta: dict[str, Any] = {
        "rabbit_object_flip_x": bool(getattr(setup, "rabbit_object_flip_x", False)),
        "rabbit_object_flip_y": bool(getattr(setup, "rabbit_object_flip_y", False)),
        "rabbit_object_rotate_180": bool(getattr(setup, "rabbit_object_rotate_180", False)),
    }
    if meta["rabbit_object_flip_x"]:
        m = m[:, ::-1]
        sx = -sx[::-1]
    if meta["rabbit_object_flip_y"]:
        m = m[::-1, :]
        sy = -sy[::-1]
    if meta["rabbit_object_rotate_180"]:
        m = np.rot90(m, k=2)
        sx = -sx[::-1]
        sy = -sy[::-1]
    return m, sx, sy, meta


def resample_transmission_to_grid(
    mask: np.ndarray,
    src_x_um: np.ndarray,
    src_y_um: np.ndarray,
    *,
    grid_n: int,
    pitch_um: float,
) -> np.ndarray:
    x_obj = centered_axis_um(grid_n, pitch_um)
    y_obj = centered_axis_um(grid_n, pitch_um)
    xx, yy = np.meshgrid(x_obj, y_obj)
    dx = float(src_x_um[1] - src_x_um[0]) if len(src_x_um) > 1 else 1.0
    dy = float(src_y_um[1] - src_y_um[0]) if len(src_y_um) > 1 else 1.0
    col = (xx - float(src_x_um[0])) / dx
    row = (yy - float(src_y_um[0])) / dy
    sampled = map_coordinates(mask, [row, col], order=0, mode="constant", cval=0.0)
    return (sampled >= 0.5).astype(np.float64)


def c_lab_rabbit_transmission_grid(
    setup: SetupParams | str | None = None,
    *,
    grid_n: int = 1024,
    pitch_um: float | None = None,
) -> np.ndarray:
    """Authoritative C_lab rabbit T(x,y) on the simulator object grid."""
    if not isinstance(setup, SetupParams):
        setup = resolve_setup_selection(setup or "C_lab")
    pitch = float(pitch_um if pitch_um is not None else setup.object_map_pixel_um)
    loaded = load_c_lab_y1_object_map(setup)
    mask, sx, sy, _ = apply_rabbit_object_orientation_in_plane(
        loaded["mask"], loaded["x_um"], loaded["y_um"], setup
    )
    return resample_transmission_to_grid(mask, sx, sy, grid_n=grid_n, pitch_um=pitch)


def c_lab_rabbit_orientation_preview_variants(
    setup: SetupParams,
    *,
    variant: str,
    grid_n: int = 1024,
    pitch_um: float | None = None,
) -> np.ndarray:
    """Contact-sheet helper: identity | flip_x | flip_y | rot180 (does not use YAML flags)."""
    pitch = float(pitch_um if pitch_um is not None else setup.object_map_pixel_um)
    loaded = load_c_lab_y1_object_map(setup)
    m = np.asarray(loaded["mask"], dtype=np.float64)
    sx = np.asarray(loaded["x_um"], dtype=np.float64)
    sy = np.asarray(loaded["y_um"], dtype=np.float64)

    class _Flags:
        rabbit_object_flip_x = variant == "flip_x"
        rabbit_object_flip_y = variant == "flip_y"
        rabbit_object_rotate_180 = variant == "rot180"

    mask, sx, sy, _ = apply_rabbit_object_orientation_in_plane(m, sx, sy, _Flags())  # type: ignore[arg-type]
    return resample_transmission_to_grid(mask, sx, sy, grid_n=grid_n, pitch_um=pitch)
