"""Resolved spatial-mode counts from usable FoV and correlation width."""
from __future__ import annotations

import math
from typing import Any

from verification.spatial.definitions import MODE_RADIUS_IN_SIGMA


def mode_area_um2(sigma_object_um: float, *, radius_in_sigma: float = MODE_RADIUS_IN_SIGMA) -> float:
    sig = float(sigma_object_um)
    if sig <= 0.0 or not math.isfinite(sig):
        raise ValueError("sigma_object_um must be finite and > 0")
    r = float(radius_in_sigma) * sig
    return float(math.pi * r * r)


def spatial_mode_metrics(
    *,
    D_FoV_um: float,
    sigma_object_um: float,
    sigma_camera_um: float | None = None,
    signed_magnification: float | None = None,
    camera_pixel_um: float | None = None,
    object_pixel_um: float | None = None,
    array_n: int | None = None,
) -> dict[str, Any]:
    """Compute N_modes and sampling diagnostics (array window ≠ FoV)."""
    D = float(D_FoV_um)
    sig_o = float(sigma_object_um)
    if D <= 0.0 or sig_o <= 0.0:
        raise ValueError("D_FoV_um and sigma_object_um must be > 0")

    A_FoV = math.pi * (0.5 * D) ** 2
    A_mode = mode_area_um2(sig_o)
    N_modes = A_FoV / A_mode
    N_1d = D / (2.0 * sig_o)

    out: dict[str, Any] = {
        "D_FoV_object_um": D,
        "A_FoV_object_um2": float(A_FoV),
        "sigma_object_um": sig_o,
        "mode_radius_um": float(MODE_RADIUS_IN_SIGMA * sig_o),
        "A_mode_um2": float(A_mode),
        "N_modes": float(N_modes),
        "N_modes_1d": float(N_1d),
        "definition": "A_FoV/(pi*sigma_o^2); N_1d=D_FoV/(2*sigma_o)",
    }

    if signed_magnification is not None:
        M = float(signed_magnification)
        D_cam = abs(M) * D
        out["signed_magnification"] = M
        out["D_FoV_camera_um"] = float(D_cam)
        out["A_FoV_camera_um2"] = float(math.pi * (0.5 * D_cam) ** 2)

    if sigma_camera_um is not None:
        sig_c = float(sigma_camera_um)
        out["sigma_camera_um"] = sig_c
        if signed_magnification is not None:
            A_mode_c = mode_area_um2(sig_c)
            D_cam = abs(float(signed_magnification)) * D
            out["N_modes_camera_check"] = float(
                (math.pi * (0.5 * D_cam) ** 2) / A_mode_c
            )

    if camera_pixel_um is not None and sigma_camera_um is not None:
        pix = float(camera_pixel_um)
        out["camera_pixel_um"] = pix
        out["sigma_camera_px"] = float(sigma_camera_um) / pix
        out["samples_per_sigma_camera"] = float(sigma_camera_um) / pix

    # Sampling window diagnostic — explicitly not FoV.
    if array_n is not None and object_pixel_um is not None:
        L = float(array_n) * float(object_pixel_um)
        out["array_n"] = int(array_n)
        out["object_map_pixel_um"] = float(object_pixel_um)
        out["array_sampling_window_um"] = L
        out["array_window_is_not_FoV"] = True
        out["FoV_over_array_window"] = float(D / L) if L > 0 else None

    return out
