"""C_lab authoritative Y1 rabbit physical calibration (production).

Physical aperture dimensions are read from ``config_c_lab_v1`` YAML only.
The frozen ``mask_rabbit.npy`` pixel data are never modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from setup_params import SetupParams, resolve_setup_selection

Y1_MASK_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "taskY1_rabbit_ground_truth"
    / "data"
    / "frozen"
    / "mask_rabbit.npy"
)
Y1_META_DEFAULT = Y1_MASK_DEFAULT.parent / "rabbit_mask_metadata.json"


def aperture_bbox_px(mask: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    """Transmission aperture bounding box on the native Y1 grid (rows=y, cols=x)."""
    m = np.asarray(mask) >= threshold
    rows, cols = np.where(m)
    if rows.size == 0:
        raise ValueError("empty transmission mask")
    return {
        "col_min": int(cols.min()),
        "col_max": int(cols.max()),
        "row_min": int(rows.min()),
        "row_max": int(rows.max()),
        "width_px": int(cols.max() - cols.min() + 1),
        "height_px": int(rows.max() - rows.min() + 1),
        "native_shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
    }


def resolved_aperture_mm(setup: SetupParams) -> tuple[float, float]:
    """Return (width_mm along +x, height_mm along +y) from setup."""
    w = getattr(setup, "y1_rabbit_aperture_width_mm", None)
    h = getattr(setup, "y1_rabbit_aperture_height_mm", None)
    if w is None or h is None:
        raise ValueError("C_lab Y1 rabbit aperture dimensions missing from setup")
    return float(w), float(h)


def build_native_physical_axes(
    mask: np.ndarray,
    *,
    aperture_width_mm: float,
    aperture_height_mm: float,
) -> dict[str, Any]:
    """Map caliper aperture size onto the transmission bbox (anisotropic µm/px allowed)."""
    bbox = aperture_bbox_px(mask)
    um_per_px_x = aperture_width_mm * 1000.0 / bbox["width_px"]
    um_per_px_y = aperture_height_mm * 1000.0 / bbox["height_px"]
    h, w = mask.shape
    x_um = (np.arange(w, dtype=np.float64) - 0.5 * (w - 1)) * um_per_px_x
    y_um = (np.arange(h, dtype=np.float64) - 0.5 * (h - 1)) * um_per_px_y
    aspect_bbox = bbox["width_px"] / bbox["height_px"]
    aspect_caliper = aperture_width_mm / aperture_height_mm
    return {
        "bbox": bbox,
        "um_per_px_x": float(um_per_px_x),
        "um_per_px_y": float(um_per_px_y),
        "mm_per_px_x": float(um_per_px_x * 1e-3),
        "mm_per_px_y": float(um_per_px_y * 1e-3),
        "aperture_width_mm": float(aperture_width_mm),
        "aperture_height_mm": float(aperture_height_mm),
        "x_um": x_um,
        "y_um": y_um,
        "x_mm": x_um * 1e-3,
        "y_mm": y_um * 1e-3,
        "axis_convention": {
            "x": "+column index (right); assigned shorter caliper dimension (width)",
            "y": "+row index (down in array storage); longer caliper dimension (height)",
            "native_head_direction": "+y toward smaller row index per Y1 metadata (imshow upper)",
        },
        "aspect_ratio_bbox": float(aspect_bbox),
        "aspect_ratio_caliper": float(aspect_caliper),
        "aspect_ratio_discrepancy": float(aspect_bbox - aspect_caliper),
        "note": (
            "Independent x/y scale from bbox→caliper; contour not stretched. "
            "Full native image extent exceeds bbox (image vs aperture distinguished)."
        ),
    }


def predicted_camera_footprint_mm(setup: SetupParams) -> dict[str, float]:
    from setup_params import resolved_spatial_mapping

    w_mm, h_mm = resolved_aperture_mm(setup)
    m = abs(float(resolved_spatial_mapping(setup)["signed_object_to_camera_magnification"]))
    pitch = float(setup.pixel_um)
    cw, ch = w_mm * m, h_mm * m
    return {
        "M_magnitude": m,
        "camera_aperture_width_mm": cw,
        "camera_aperture_height_mm": ch,
        "camera_aperture_width_px": cw / (pitch * 1e-3),
        "camera_aperture_height_px": ch / (pitch * 1e-3),
    }


def load_c_lab_y1_object_map(
    setup: SetupParams | str = "C_lab",
    *,
    mask_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Load frozen Y1 mask with C_lab production physical coordinates."""
    from object_maps.rabbit_loader import load_object_map

    if not isinstance(setup, SetupParams):
        setup = resolve_setup_selection(setup)
    w_mm, h_mm = resolved_aperture_mm(setup)
    mask_path = mask_path or Y1_MASK_DEFAULT
    metadata_path = metadata_path or Y1_META_DEFAULT
    loaded = load_object_map(
        mask_path,
        metadata_path,
        aperture_width_mm=w_mm,
        aperture_height_mm=h_mm,
    )
    loaded["c_lab_calibration"] = loaded.get("physical_calibration", {})
    loaded["predicted_camera_footprint"] = predicted_camera_footprint_mm(setup)
    return loaded
