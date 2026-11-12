"""C_lab display zoom — compatibility wrapper over camera.native_crop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera.native_crop import (
    NativeCropConfig,
    _normalize_center_mode,
    apply_native_crop,
    crop_bounds_from_center,
    detect_signal_spot_center,
    export_gray16_tiff,
    float_to_display_u16,
    native_crop_config_from_setup,
    resolve_crop_window,
)


@dataclass(frozen=True)
class ExportZoomConfig:
    enabled: bool
    mode: str
    center_source: str
    native_shape_hw: tuple[int, int]
    export_width_px: int
    export_height_px: int
    preserve_native_frame: bool
    resize_after_crop: bool
    interpolation: str
    export_dtype: str


def _native_to_export_zoom(cfg) -> ExportZoomConfig:
    nh, nw = cfg.native_shape_hw or (1024, 1024)
    legacy_center = {
        "reference_signal_spot_center": "signal_spot_center_from_reference",
        "image_geometric_center": "image_geometric_center",
        "manual": "manual",
    }.get(cfg.center_mode, cfg.center_mode)
    legacy_mode = (
        "centered_fixed_crop"
        if cfg.mode in ("fixed_center_native_crop", "centered_fixed_crop")
        else cfg.mode
    )
    return ExportZoomConfig(
        enabled=cfg.enabled,
        mode=legacy_mode,
        center_source=legacy_center,
        native_shape_hw=(int(nh), int(nw)),
        export_width_px=int(cfg.crop_width_px),
        export_height_px=int(cfg.crop_height_px),
        preserve_native_frame=bool(cfg.preserve_native_frame),
        resize_after_crop=bool(cfg.resize),
        interpolation=str(cfg.interpolation),
        export_dtype=str(cfg.export_dtype),
    )


def export_zoom_config_from_setup(setup: Any) -> ExportZoomConfig:
    return _native_to_export_zoom(native_crop_config_from_setup(setup))


def apply_fixed_crop(native_frame: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    return apply_native_crop(native_frame, bounds)


def export_zoom_to_native_crop(config: ExportZoomConfig) -> NativeCropConfig:
    return NativeCropConfig(
        enabled=config.enabled,
        mode=config.mode,
        center_mode=_normalize_center_mode(config.center_source),
        crop_width_px=config.export_width_px,
        crop_height_px=config.export_height_px,
        center_row_px=None,
        center_col_px=None,
        native_shape_hw=config.native_shape_hw,
        preserve_native_frame=config.preserve_native_frame,
        resize=config.resize_after_crop,
        interpolation=config.interpolation,
        export_dtype=config.export_dtype,
        export_colormap=None,
    )


def resolve_fixed_crop_window(
    reference_native_frame: np.ndarray,
    config: ExportZoomConfig,
) -> tuple[dict[str, Any], dict[str, int]]:
    meta, bounds = resolve_crop_window(reference_native_frame, export_zoom_to_native_crop(config))
    if "center_source" not in meta.get("config", {}):
        meta["config"] = {
            **meta.get("config", {}),
            "center_source": config.center_source,
        }
    return meta, bounds


__all__ = [
    "ExportZoomConfig",
    "apply_fixed_crop",
    "crop_bounds_from_center",
    "detect_signal_spot_center",
    "export_gray16_tiff",
    "export_zoom_config_from_setup",
    "float_to_display_u16",
    "resolve_fixed_crop_window",
]
