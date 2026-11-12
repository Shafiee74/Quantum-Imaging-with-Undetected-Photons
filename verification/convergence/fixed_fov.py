"""Fixed physical camera FOV with independent internal numerical refinement (Figure 3e)."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import numpy as np

from benchmarks.targets.geometry import centered_axis_um
from setup_params import (
    physical_detector_pixel_um,
    resolve_setup_selection,
)
from simulate import simulate_forward
from verification.convergence.definitions import N_MID_DEFAULT
from verification.convergence.grids import domain_um_for_profile, stock_object_pixel_um
from verification.resolution.quantities import intensity_map_from_result


@dataclass(frozen=True)
class FixedFoVLevel:
    profile: str
    n_num: int
    n_cam: int
    hardware_pixel_um: float
    internal_camera_pixel_um: float
    object_pixel_um: float
    camera_fov_um: float
    object_domain_um: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "n_num": self.n_num,
            "n_cam": self.n_cam,
            "hardware_pixel_um": self.hardware_pixel_um,
            "internal_camera_pixel_um": self.internal_camera_pixel_um,
            "object_pixel_um": self.object_pixel_um,
            "camera_fov_um": self.camera_fov_um,
            "object_domain_um": self.object_domain_um,
            "policy": (
                "L_cam = n_cam * hardware_pitch fixed; "
                "dx_num = L_cam / n_num; object dx = L_obj / n_num; "
                "integrate internal maps onto n_cam hardware grid"
            ),
        }


def build_fixed_fov_levels(
    profile: str,
    ns: tuple[int, ...] | list[int],
    *,
    n_cam: int = N_MID_DEFAULT,
    hardware_pixel_um: float | None = None,
) -> list[FixedFoVLevel]:
    base = resolve_setup_selection(profile)
    p_hw = float(
        hardware_pixel_um
        if hardware_pixel_um is not None
        else physical_detector_pixel_um(base)
    )
    l_cam = float(n_cam) * p_hw
    l_obj = domain_um_for_profile(profile, n_mid=n_cam)
    levels: list[FixedFoVLevel] = []
    for n in ns:
        n_num = int(n)
        if n_num < n_cam:
            raise ValueError(f"n_num={n_num} must be >= fixed camera n_cam={n_cam}")
        if n_num % n_cam != 0:
            raise ValueError(f"n_num={n_num} must be divisible by n_cam={n_cam} for binning")
        levels.append(
            FixedFoVLevel(
                profile=profile,
                n_num=n_num,
                n_cam=n_cam,
                hardware_pixel_um=p_hw,
                internal_camera_pixel_um=l_cam / float(n_num),
                object_pixel_um=l_obj / float(n_num),
                camera_fov_um=l_cam,
                object_domain_um=l_obj,
            )
        )
    return levels


def setup_for_fixed_fov(level: FixedFoVLevel):
    base = resolve_setup_selection(level.profile)
    return base.overridden(
        object_map_pixel_um=float(level.object_pixel_um),
        numerical_verification_internal_camera_pixel_um=float(
            level.internal_camera_pixel_um
        ),
        fixed_fov_detector_output_n=int(level.n_cam),
    )


def bin_area_average_to_detector(
    img: np.ndarray,
    *,
    n_out: int,
) -> np.ndarray:
    """Area-average an n_in×n_in map onto n_out×n_out (n_in divisible by n_out)."""
    a = np.asarray(img, dtype=np.float64)
    n_in = int(a.shape[0])
    n_out = int(n_out)
    if a.shape != (n_in, n_in):
        raise ValueError(f"expected square map, got {a.shape}")
    if n_in % n_out != 0:
        raise ValueError(f"cannot bin {n_in} -> {n_out}")
    k = n_in // n_out
    return a.reshape(n_out, k, n_out, k).mean(axis=(1, 3))


def detector_axes_um(level: FixedFoVLevel) -> tuple[np.ndarray, np.ndarray]:
    p = float(level.hardware_pixel_um)
    n = int(level.n_cam)
    x = centered_axis_um(n, p)
    y = centered_axis_um(n, p)
    return x, y


def simulate_intensity_on_detector(
    level: FixedFoVLevel,
    ref: dict[str, Any],
    state,
    *,
    seed: int,
) -> dict[str, Any]:
    setup = setup_for_fixed_fov(level)
    state = replace(state, setup=setup)
    r = simulate_forward(ref, state, seed=seed, detector_noise=False)
    i_int = intensity_map_from_result(r)
    if i_int.shape[0] == level.n_cam:
        i_det = i_int
    else:
        i_det = bin_area_average_to_detector(i_int, n_out=level.n_cam)
    x_um, y_um = detector_axes_um(level)
    vo = dict(r.meta.get("visibility_observables") or {})
    p_hw = float(level.hardware_pixel_um)
    dx_int = float(level.internal_camera_pixel_um)
    return {
        "intensity_detector": i_det,
        "intensity_internal": i_int,
        "x_um": x_um,
        "y_um": y_um,
        "I_mean": float(np.mean(i_det)),
        "I_sum_pixels": float(np.sum(i_det)),
        "I_integral_internal": float(np.sum(i_int) * dx_int**2),
        "I_integral_detector": float(np.sum(i_det) * p_hw**2),
        "V_50_50": float(vo.get("V_50_50") or np.nan),
        "level": level.to_dict(),
        "stock_object_pixel_um": stock_object_pixel_um(level.profile),
    }
