"""General fixed-center native-pixel camera crop (display/export only)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None

CENTER_MODE_REFERENCE = "reference_signal_spot_center"
CENTER_MODE_GEOMETRIC = "image_geometric_center"
CENTER_MODE_MANUAL = "manual"

_LEGACY_CENTER_SOURCE_TO_MODE = {
    "signal_spot_center_from_reference": CENTER_MODE_REFERENCE,
    "reference_signal_spot_center": CENTER_MODE_REFERENCE,
    "image_geometric_center": CENTER_MODE_GEOMETRIC,
    "manual": CENTER_MODE_MANUAL,
}


class CropBoundsValidationError(ValueError):
    """Crop window cannot be placed at the requested center without leaving the frame."""


@dataclass(frozen=True)
class NativeCropConfig:
    enabled: bool
    mode: str
    center_mode: str
    crop_width_px: int
    crop_height_px: int
    center_row_px: int | None
    center_col_px: int | None
    native_shape_hw: tuple[int, int] | None
    preserve_native_frame: bool
    resize: bool
    interpolation: str
    export_dtype: str
    export_colormap: str | None


def _setup_field(setup: Any, new: str, old: str, default: Any = None) -> Any:
    if hasattr(setup, new):
        v = getattr(setup, new)
        if v is not None:
            return v
    if hasattr(setup, old):
        v = getattr(setup, old)
        if v is not None:
            return v
    return default


def _normalize_center_mode(raw: str | None) -> str:
    key = str(raw or CENTER_MODE_REFERENCE).strip()
    return _LEGACY_CENTER_SOURCE_TO_MODE.get(key, key)


def native_crop_config_from_setup(setup: Any) -> NativeCropConfig:
    nh = _setup_field(setup, "camera_native_height_px", "camera_native_height_px", None)
    nw = _setup_field(setup, "camera_native_width_px", "camera_native_width_px", None)
    native_shape: tuple[int, int] | None = None
    if nh is not None and nw is not None:
        native_shape = (int(nh), int(nw))

    center_raw = _setup_field(
        setup,
        "camera_crop_center_mode",
        "export_zoom_center_source",
        CENTER_MODE_REFERENCE,
    )
    center_mode = _normalize_center_mode(str(center_raw) if center_raw is not None else None)

    enabled = bool(
        _setup_field(setup, "camera_crop_enabled", "export_zoom_crop_enabled", False)
    )
    mode = str(
        _setup_field(
            setup,
            "camera_crop_mode",
            "export_zoom_crop_mode",
            "fixed_center_native_crop",
        )
        or "fixed_center_native_crop"
    )
    width = int(
        _setup_field(setup, "camera_crop_width_px", "export_zoom_export_width_px", 455)
        or 455
    )
    height = int(
        _setup_field(setup, "camera_crop_height_px", "export_zoom_export_height_px", 419)
        or 419
    )
    return NativeCropConfig(
        enabled=enabled,
        mode=mode,
        center_mode=center_mode,
        crop_width_px=width,
        crop_height_px=height,
        center_row_px=_setup_field(setup, "camera_crop_center_row_px", "camera_crop_center_row_px", None),
        center_col_px=_setup_field(setup, "camera_crop_center_col_px", "camera_crop_center_col_px", None),
        native_shape_hw=native_shape,
        preserve_native_frame=bool(
            _setup_field(
                setup,
                "camera_crop_preserve_native_frame",
                "export_zoom_preserve_native_frame",
                True,
            )
        ),
        resize=bool(
            _setup_field(setup, "camera_crop_resize", "export_zoom_resize_after_crop", False)
        ),
        interpolation=str(
            _setup_field(setup, "camera_crop_interpolation", "export_zoom_interpolation", "none")
            or "none"
        ),
        export_dtype=str(
            _setup_field(setup, "camera_crop_export_dtype", "export_zoom_export_dtype", "uint16")
            or "uint16"
        ),
        export_colormap=_setup_field(
            setup, "camera_crop_export_colormap", "camera_crop_export_colormap", None
        ),
    )


def validate_native_crop_config(config: NativeCropConfig) -> None:
    if not config.enabled:
        return
    if config.resize:
        raise CropBoundsValidationError("native crop forbids resize_after_crop")
    if str(config.interpolation).lower() not in ("none", ""):
        raise CropBoundsValidationError(
            f"native crop forbids interpolation (got {config.interpolation!r})"
        )


def detect_signal_spot_center(
    native_frame: np.ndarray,
    *,
    smooth_sigma_px: float = 8.0,
    lobe_radius_px: float = 140.0,
    threshold_frac: float = 0.08,
) -> dict[str, Any]:
    """Intensity-weighted centroid of the main lobe (no-object reference)."""
    if gaussian_filter is None:
        raise RuntimeError("scipy required for signal center detection")
    a = np.asarray(native_frame, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("native_frame must be 2-D")
    smooth = gaussian_filter(a, sigma=float(smooth_sigma_px))
    peak_flat = int(np.argmax(smooth))
    pr, pc = divmod(peak_flat, a.shape[1])
    yy, xx = np.indices(a.shape, dtype=np.float64)
    r = np.hypot(yy - pr, xx - pc)
    mask = (r <= lobe_radius_px) & (smooth >= threshold_frac * float(smooth.max()))
    if int(mask.sum()) < 16:
        return {
            "center_row": int(pr),
            "center_col": int(pc),
            "method": "argmax_fallback",
            "valid_pixels": int(mask.sum()),
        }
    w = smooth[mask]
    rows, cols = np.where(mask)
    cr = float(np.sum(rows * w) / np.sum(w))
    cc = float(np.sum(cols * w) / np.sum(w))
    return {
        "center_row": int(round(cr)),
        "center_col": int(round(cc)),
        "center_row_float": cr,
        "center_col_float": cc,
        "method": "gaussian_smooth_weighted_centroid",
        "peak_row": int(pr),
        "peak_col": int(pc),
        "valid_pixels": int(mask.sum()),
        "smooth_sigma_px": smooth_sigma_px,
        "lobe_radius_px": lobe_radius_px,
        "threshold_frac": threshold_frac,
    }


def image_geometric_center(native_height: int, native_width: int) -> tuple[int, int]:
    """Integer center consistent with 1024² → (511, 511)."""
    cr = (int(native_height) - 1) // 2
    cc = (int(native_width) - 1) // 2
    return cr, cc


def resolve_crop_center(
    reference_native_frame: np.ndarray,
    config: NativeCropConfig,
) -> tuple[int, int, dict[str, Any]]:
    nh, nw = reference_native_frame.shape
    if config.center_mode == CENTER_MODE_REFERENCE:
        det = detect_signal_spot_center(reference_native_frame)
        return det["center_row"], det["center_col"], {"center_detection": det}
    if config.center_mode == CENTER_MODE_GEOMETRIC:
        cr, cc = image_geometric_center(nh, nw)
        return cr, cc, {"center_mode": CENTER_MODE_GEOMETRIC, "center_row": cr, "center_col": cc}
    if config.center_mode == CENTER_MODE_MANUAL:
        if config.center_row_px is None or config.center_col_px is None:
            raise CropBoundsValidationError(
                "manual center_mode requires center_row_px and center_col_px"
            )
        cr, cc = int(config.center_row_px), int(config.center_col_px)
        return cr, cc, {"center_mode": CENTER_MODE_MANUAL, "center_row": cr, "center_col": cc}
    raise CropBoundsValidationError(f"unknown center_mode {config.center_mode!r}")


def crop_bounds_from_center(
    center_row: int,
    center_col: int,
    *,
    crop_height_px: int | None = None,
    crop_width_px: int | None = None,
    out_height: int | None = None,
    out_width: int | None = None,
    native_height: int,
    native_width: int,
) -> dict[str, int]:
    """Return inclusive-exclusive slice bounds row0:row1, col0:col1 (exact output size).

    No clamping, padding, or recentering — raises CropBoundsValidationError if the window
    would leave the source image.
    """
    oh = crop_height_px if crop_height_px is not None else out_height
    ow = crop_width_px if crop_width_px is not None else out_width
    if oh is None or ow is None:
        raise TypeError("crop_bounds_from_center requires crop_height_px/crop_width_px or out_height/out_width")
    oh, ow = int(oh), int(ow)
    nh, nw = int(native_height), int(native_width)
    if oh <= 0 or ow <= 0:
        raise CropBoundsValidationError("crop dimensions must be positive")
    if oh > nh or ow > nw:
        raise CropBoundsValidationError("crop larger than native frame")
    row0 = int(center_row) - oh // 2
    col0 = int(center_col) - ow // 2
    row1 = row0 + oh
    col1 = col0 + ow
    if row0 < 0 or col0 < 0 or row1 > nh or col1 > nw:
        raise CropBoundsValidationError(
            "crop window does not fit native frame at requested center "
            f"(center=({center_row},{center_col}), crop=({oh},{ow}), native=({nh},{nw}), "
            f"bounds=({row0}:{row1},{col0}:{col1}))"
        )
    if row1 - row0 != oh or col1 - col0 != ow:
        raise RuntimeError("failed to produce exact crop size")
    return {
        "row0": row0,
        "row1": row1,
        "col0": col0,
        "col1": col1,
        "out_height": oh,
        "out_width": ow,
    }


def apply_native_crop(native_frame: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    """Native-pixel slice only — no interpolation."""
    sl = native_frame[bounds["row0"] : bounds["row1"], bounds["col0"] : bounds["col1"]]
    return np.asarray(sl)


def resolve_crop_window(
    reference_native_frame: np.ndarray,
    config: NativeCropConfig,
) -> tuple[dict[str, Any], dict[str, int]]:
    validate_native_crop_config(config)
    frame = np.asarray(reference_native_frame)
    if frame.ndim != 2:
        raise ValueError("reference_native_frame must be 2-D")
    nh, nw = frame.shape
    if config.native_shape_hw is not None:
        exp_h, exp_w = config.native_shape_hw
        if (nh, nw) != (exp_h, exp_w):
            raise CropBoundsValidationError(
                f"reference frame shape {(nh, nw)} != configured native {(exp_h, exp_w)}"
            )
    center_row, center_col, center_meta = resolve_crop_center(frame, config)
    bounds = crop_bounds_from_center(
        center_row,
        center_col,
        crop_height_px=config.crop_height_px,
        crop_width_px=config.crop_width_px,
        native_height=nh,
        native_width=nw,
    )
    meta = {
        **center_meta,
        "center_row": center_row,
        "center_col": center_col,
        "bounds": bounds,
        "config": asdict(config),
        "native_shape_hw": (nh, nw),
    }
    return meta, bounds


def float_to_display_u16(frame: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-30)
    return np.clip(np.asarray(frame, dtype=np.float64) / scale * 65535.0, 0, 65535).astype(
        np.uint16
    )


def export_gray16_tiff(path: Path, frame_u16: np.ndarray) -> None:
    import tifffile

    arr = np.asarray(frame_u16)
    if arr.ndim != 2:
        raise ValueError("export_gray16_tiff requires a 2-D grayscale frame")
    if arr.dtype != np.uint16:
        raise ValueError("export_gray16_tiff requires uint16")
    tifffile.imwrite(path, arr, photometric="minisblack")


def experimental_spot_center_offset_report(
    frame: np.ndarray,
) -> dict[str, Any]:
    """Diagnostic for already-cropped experimental frames (read-only)."""
    a = np.asarray(frame)
    h, w = a.shape
    gcr, gcc = image_geometric_center(h, w)
    det = detect_signal_spot_center(a)
    return {
        "image_shape_hw": (h, w),
        "geometric_center_row_col": (gcr, gcc),
        "signal_spot_center_row_col": (det["center_row"], det["center_col"]),
        "offset_row_px": int(det["center_row"]) - gcr,
        "offset_col_px": int(det["center_col"]) - gcc,
        "center_detection": det,
    }
