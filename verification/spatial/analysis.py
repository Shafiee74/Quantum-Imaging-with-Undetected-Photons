"""Per-configuration FoV + spatial-mode evaluation."""
from __future__ import annotations

from typing import Any

from setup_params import (
    pump_generation_amplitude_weight_from_setup,
    resolve_setup_selection,
    resolved_spatial_mapping,
)
from verification.spatial.fov import usable_fov_object_plane
from verification.spatial.modes import spatial_mode_metrics


def pump_waist_from_setup(setup: Any) -> float:
    w = getattr(setup, "pump_waist_NL1_um", None)
    if w is None:
        w = getattr(setup, "pump_waist_um", None)
    if w is None:
        raise ValueError("setup lacks pump_waist_um")
    return float(w)


def evaluate_profile(
    profile: str,
    *,
    array_n: int = 64,
    setup: Any | None = None,
) -> dict[str, Any]:
    """Evaluate usable FoV and resolved modes for one A/B/C/D profile."""
    setup = resolve_setup_selection(profile) if setup is None else setup
    sm = resolved_spatial_mapping(setup)
    waist = pump_waist_from_setup(setup)
    pump_meta = pump_generation_amplitude_weight_from_setup(setup, (array_n, array_n))

    fov = usable_fov_object_plane(
        waist,
        center_x_um=float(pump_meta.get("center_x_um") or 0.0),
        center_y_um=float(pump_meta.get("center_y_um") or 0.0),
    )
    modes = spatial_mode_metrics(
        D_FoV_um=float(fov["D_FoV_um"]),
        sigma_object_um=float(sm["correlation_sigma_object_um"]),
        sigma_camera_um=float(sm["correlation_sigma_camera_um"]),
        signed_magnification=float(sm["signed_object_to_camera_magnification"]),
        camera_pixel_um=float(sm["camera_pixel_um"]),
        object_pixel_um=float(sm["object_plane_pixel_um"]),
        array_n=int(array_n),
    )

    return {
        "profile": profile,
        "imaging_mode": sm.get("imaging_mode"),
        "pump_waist_um": waist,
        "crystal_length_mm": float(getattr(setup, "crystal_length_mm")),
        "pixel_um": float(getattr(setup, "pixel_um")),
        "object_map_pixel_um": float(sm["object_plane_pixel_um"]),
        "signed_magnification": float(sm["signed_object_to_camera_magnification"]),
        "correlation_sigma_object_um": float(sm["correlation_sigma_object_um"]),
        "correlation_sigma_camera_um": float(sm["correlation_sigma_camera_um"]),
        "fov": {k: v for k, v in fov.items() if not str(k).startswith("map_")},
        "modes": modes,
        "pump_meta": {
            k: pump_meta[k]
            for k in (
                "profile_type",
                "waist_um",
                "center_x_um",
                "center_y_um",
                "equation",
                "peak_normalized",
            )
            if k in pump_meta
        },
        "fov_maps": {
            "X_um": fov["map_X_um"],
            "Y_um": fov["map_Y_um"],
            "I": fov["map_I"],
            "mask": fov["map_mask"],
            "radial_r_um": fov["map_radial_r_um"],
            "radial_I": fov["map_radial_I"],
        },
        "spatial_mapping_subset": {
            k: sm[k]
            for k in (
                "imaging_mode",
                "object_plane_pixel_um",
                "camera_pixel_um",
                "signed_object_to_camera_magnification",
                "magnification_magnitude",
                "correlation_sigma_object_um",
                "correlation_sigma_camera_um",
                "correlation_sigma_camera_px",
                "M_QIUP",
            )
            if k in sm
        },
    }
