"""Reusable loader / resampler for the frozen independent rabbit object map.

Maps a physically calibrated binary amplitude mask onto a requested simulator
grid while preserving physical dimensions, aspect ratio, origin convention,
and orientation. Does not modify the frozen native mask.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

DEFAULT_MASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "taskY1_rabbit_ground_truth"
    / "data"
    / "frozen"
)


def load_object_map(
    mask_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    *,
    aperture_width_mm: float | None = None,
    aperture_height_mm: float | None = None,
) -> dict[str, Any]:
    """Load frozen rabbit mask + metadata.

    Returns dict with keys:
      mask (ndarray uint8 0/1),
      x_mm, y_mm, x_um, y_um,
      metadata (dict),
      mm_per_pixel, um_per_pixel,
      origin_convention
    """
    mask_path = Path(mask_path) if mask_path else DEFAULT_MASK_DIR / "mask_rabbit.npy"
    metadata_path = (
        Path(metadata_path) if metadata_path else DEFAULT_MASK_DIR / "rabbit_mask_metadata.json"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    mask = np.load(mask_path)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    physical_calibration: dict[str, Any] | None = None
    if aperture_width_mm is not None and aperture_height_mm is not None:
        from object_maps.c_lab_y1_rabbit_physical import build_native_physical_axes

        physical_calibration = build_native_physical_axes(
            mask,
            aperture_width_mm=float(aperture_width_mm),
            aperture_height_mm=float(aperture_height_mm),
        )
        x_mm = physical_calibration["x_mm"]
        y_mm = physical_calibration["y_mm"]
        x_um = physical_calibration["x_um"]
        y_um = physical_calibration["y_um"]
    else:
        coords_path = mask_path.parent / "rabbit_mask_physical_coordinates.npz"
        if coords_path.is_file():
            z = np.load(coords_path)
            x_mm = np.asarray(z["x_mm"], dtype=np.float64)
            y_mm = np.asarray(z["y_mm"], dtype=np.float64)
            x_um = np.asarray(z["x_um"], dtype=np.float64)
            y_um = np.asarray(z["y_um"], dtype=np.float64)
        else:
            um_per_px = float(meta["um_per_pixel"])
            h, w = mask.shape
            x_um = (np.arange(w, dtype=np.float64) - 0.5 * (w - 1)) * um_per_px
            y_um = (np.arange(h, dtype=np.float64) - 0.5 * (h - 1)) * um_per_px
            x_mm = x_um * 1.0e-3
            y_mm = y_um * 1.0e-3

    out = {
        "mask": np.asarray(mask),
        "x_mm": x_mm,
        "y_mm": y_mm,
        "x_um": x_um,
        "y_um": y_um,
        "metadata": meta,
        "mm_per_pixel": float(meta["mm_per_pixel"]),
        "um_per_pixel": float(meta["um_per_pixel"]),
        "origin_convention": meta.get("origin_convention", "image_center"),
    }
    if physical_calibration is not None:
        out["physical_calibration"] = physical_calibration
        out["mm_per_pixel"] = float(physical_calibration["mm_per_px_x"])
        out["um_per_pixel"] = float(physical_calibration["um_per_px_x"])
    return out


def resample_object_map_to_grid(
    object_map: dict[str, Any] | None = None,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    method: str = "nearest",
    binarize: bool = True,
    mask_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample physical rabbit T(x,y) onto simulator axes (µm).

    ``x_um``, ``y_um`` are 1-D monotonic physical coordinate vectors for the
    simulator object plane. Output ``T_sim`` has shape (len(y_um), len(x_um)).

    For strict binary aperture geometry, ``method='nearest'`` is preferred.
    """
    if object_map is None:
        object_map = load_object_map(mask_path, metadata_path)

    mask = np.asarray(object_map["mask"], dtype=np.float64)
    src_x = np.asarray(object_map["x_um"], dtype=np.float64)
    src_y = np.asarray(object_map["y_um"], dtype=np.float64)
    x_um = np.asarray(x_um, dtype=np.float64)
    y_um = np.asarray(y_um, dtype=np.float64)
    if src_x.ndim != 1 or src_y.ndim != 1:
        raise ValueError("source axes must be 1-D")
    if x_um.ndim != 1 or y_um.ndim != 1:
        raise ValueError("simulator axes must be 1-D")

    # map_coordinates uses (row, col) = (y, x)
    # Convert target physical coords -> fractional source indices
    def _to_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
        # linear mapping assuming uniform axis
        a0 = float(axis[0])
        da = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
        return (values - a0) / da

    xx, yy = np.meshgrid(x_um, y_um)
    col = _to_index(src_x, xx)
    row = _to_index(src_y, yy)

    order = 0 if method in ("nearest", "nn") else 1
    sampled = map_coordinates(
        mask, [row, col], order=order, mode="constant", cval=0.0, prefilter=False
    )
    if binarize:
        T = (sampled >= 0.5).astype(np.float64)
    else:
        T = sampled.astype(np.float64)

    meta = {
        "method": method,
        "binarize": bool(binarize),
        "source_shape": list(mask.shape),
        "output_shape": list(T.shape),
        "source_x_um_range": [float(src_x[0]), float(src_x[-1])],
        "source_y_um_range": [float(src_y[0]), float(src_y[-1])],
        "sim_x_um_range": [float(x_um[0]), float(x_um[-1])],
        "sim_y_um_range": [float(y_um[0]), float(y_um[-1])],
        "origin_convention": object_map.get("origin_convention"),
        "physical_width_mm_source": float(object_map["metadata"].get("physical_width_mm", float("nan"))),
        "physical_height_mm_source": float(object_map["metadata"].get("physical_height_mm", float("nan"))),
        "aperture_fraction": float(np.mean(T)) if T.size else 0.0,
    }
    return T, meta
