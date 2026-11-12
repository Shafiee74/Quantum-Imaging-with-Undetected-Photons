"""Canonical deterministic benchmark target catalog (config-agnostic).

Physical parameters are declared once. The same ground truth is sampled onto
any ``(n, pixel_um)`` grid — never fork incompatible A/B/C/D object versions.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from benchmarks.targets.geometry import (
    DEFAULT_N,
    DEFAULT_PIXEL_UM,
    coordinate_grids_um,
)
from benchmarks.targets.target import BenchmarkTarget

# ---------------------------------------------------------------------------
# Shared physical ground-truth parameters (µm / rad) — single source of truth
# ---------------------------------------------------------------------------
PHYSICAL_TRUTH: dict[str, Any] = {
    "default_n": DEFAULT_N,
    "default_pixel_um": DEFAULT_PIXEL_UM,
    "point_half_width_um": 0.0,  # single central pixel (nearest-pixel point)
    "slit_width_um": 40.0,  # narrow vertical slit
    "knife_edge_offset_um": 0.0,  # edge through optical axis
    "checker_period_um": 80.0,
    "amplitude_step_T_left": 0.20,
    "amplitude_step_T_right": 0.90,
    "amplitude_step_x0_um": 0.0,
    "phase_step_phi_left_rad": -0.60,
    "phase_step_phi_right_rad": 1.80,
    "phase_step_x0_um": 0.0,
    "phase_ramp_phi0_rad": -float(np.pi),
    "phase_ramp_phi1_rad": float(np.pi),
    "transparent_T": 1.0,
    "transparent_phi_rad": 0.0,
    "blocked_T": 0.0,
    "blocked_phi_rad": 0.0,
}


def _base_maps(
    n: int,
    pixel_um: float,
    *,
    T: np.ndarray,
    phi: np.ndarray,
    target_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> BenchmarkTarget:
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    meta = {
        "physical_truth_keys": sorted(PHYSICAL_TRUTH.keys()),
        "shared_across_configs": ["A", "B", "C", "D"],
    }
    if extra_meta:
        meta.update(extra_meta)
    return BenchmarkTarget(
        target_id=target_id,
        T=T,
        phi_rad=phi,
        x_um=x_um,
        y_um=y_um,
        X_um=X_um,
        Y_um=Y_um,
        pixel_um=float(pixel_um),
        metadata=meta,
    )


def make_transparent(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    T = np.full((n, n), float(PHYSICAL_TRUTH["transparent_T"]), dtype=np.float64)
    phi = np.full(
        (n, n), float(PHYSICAL_TRUTH["transparent_phi_rad"]), dtype=np.float64
    )
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="transparent",
        extra_meta={"family": "control", "description": "fully transmitting, zero phase"},
    )


def make_blocked(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    T = np.full((n, n), float(PHYSICAL_TRUTH["blocked_T"]), dtype=np.float64)
    phi = np.full((n, n), float(PHYSICAL_TRUTH["blocked_phi_rad"]), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="blocked",
        extra_meta={"family": "control", "description": "fully opaque"},
    )


def make_point(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    """Single bright pixel at the optical axis (nearest sample to x=y=0)."""
    T = np.zeros((n, n), dtype=np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    # Nearest pixel to origin.
    iy = int(np.argmin(np.abs(y_um)))
    ix = int(np.argmin(np.abs(x_um)))
    T[iy, ix] = 1.0
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="point",
        extra_meta={
            "family": "resolution",
            "description": "single-pixel point at optical axis",
            "point_pixel_ji": [iy, ix],
            "point_x_um": float(x_um[ix]),
            "point_y_um": float(y_um[iy]),
            "point_half_width_um": float(PHYSICAL_TRUTH["point_half_width_um"]),
        },
    )


def make_narrow_slit(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
    slit_width_um: float | None = None,
) -> BenchmarkTarget:
    width = float(
        PHYSICAL_TRUTH["slit_width_um"] if slit_width_um is None else slit_width_um
    )
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    T = (np.abs(X_um) <= 0.5 * width).astype(np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="narrow_slit",
        extra_meta={
            "family": "resolution",
            "description": "vertical transmitting slit about x=0",
            "slit_width_um": width,
            "orientation": "vertical",
        },
    )


def make_knife_edge_vertical(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
    edge_x_um: float | None = None,
) -> BenchmarkTarget:
    x0 = float(
        PHYSICAL_TRUTH["knife_edge_offset_um"] if edge_x_um is None else edge_x_um
    )
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    # Transmit for x >= edge (right half open).
    T = (X_um >= x0).astype(np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="knife_edge_vertical",
        extra_meta={
            "family": "resolution",
            "description": "vertical knife edge; T=1 for x >= edge_x_um",
            "edge_x_um": x0,
            "orientation": "vertical",
        },
    )


def make_knife_edge_horizontal(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
    edge_y_um: float | None = None,
) -> BenchmarkTarget:
    y0 = float(
        PHYSICAL_TRUTH["knife_edge_offset_um"] if edge_y_um is None else edge_y_um
    )
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    T = (Y_um >= y0).astype(np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="knife_edge_horizontal",
        extra_meta={
            "family": "resolution",
            "description": "horizontal knife edge; T=1 for y >= edge_y_um",
            "edge_y_um": y0,
            "orientation": "horizontal",
        },
    )


def make_checkerboard(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
    period_um: float | None = None,
) -> BenchmarkTarget:
    period = float(
        PHYSICAL_TRUTH["checker_period_um"] if period_um is None else period_um
    )
    if period <= 0.0:
        raise ValueError("period_um must be > 0")
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    # Shift so the origin cell is well-defined for both even/odd N.
    ix = np.floor((X_um + 0.5 * period) / period).astype(np.int64)
    iy = np.floor((Y_um + 0.5 * period) / period).astype(np.int64)
    T = ((ix + iy) % 2 == 0).astype(np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="checkerboard",
        extra_meta={
            "family": "calibration",
            "description": "calibrated checkerboard / bars (binary T)",
            "period_um": period,
            "pattern": "checkerboard",
        },
    )


def make_amplitude_step(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    T_l = float(PHYSICAL_TRUTH["amplitude_step_T_left"])
    T_r = float(PHYSICAL_TRUTH["amplitude_step_T_right"])
    x0 = float(PHYSICAL_TRUTH["amplitude_step_x0_um"])
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    T = np.where(X_um < x0, T_l, T_r).astype(np.float64)
    phi = np.zeros((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="amplitude_step",
        extra_meta={
            "family": "amplitude",
            "description": "left/right intensity transmission step",
            "T_left": T_l,
            "T_right": T_r,
            "step_x_um": x0,
        },
    )


def make_phase_step(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    phi_l = float(PHYSICAL_TRUTH["phase_step_phi_left_rad"])
    phi_r = float(PHYSICAL_TRUTH["phase_step_phi_right_rad"])
    x0 = float(PHYSICAL_TRUTH["phase_step_x0_um"])
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    T = np.ones((n, n), dtype=np.float64)
    phi = np.where(X_um < x0, phi_l, phi_r).astype(np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=phi,
        target_id="phase_step",
        extra_meta={
            "family": "phase",
            "description": "left/right phase step at full transmission",
            "phi_left_rad": phi_l,
            "phi_right_rad": phi_r,
            "step_x_um": x0,
        },
    )


def make_phase_ramp(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> BenchmarkTarget:
    phi0 = float(PHYSICAL_TRUTH["phase_ramp_phi0_rad"])
    phi1 = float(PHYSICAL_TRUTH["phase_ramp_phi1_rad"])
    x_um, y_um, X_um, Y_um = coordinate_grids_um((n, n), pixel_um)
    x_min, x_max = float(x_um[0]), float(x_um[-1])
    if n == 1 or x_max == x_min:
        ramp = np.full((n, n), 0.5 * (phi0 + phi1), dtype=np.float64)
    else:
        alpha = (X_um - x_min) / (x_max - x_min)
        ramp = phi0 + (phi1 - phi0) * alpha
    T = np.ones((n, n), dtype=np.float64)
    return _base_maps(
        n,
        pixel_um,
        T=T,
        phi=ramp.astype(np.float64),
        target_id="phase_ramp",
        extra_meta={
            "family": "phase",
            "description": "horizontal phase ramp across FOV at T=1",
            "phi0_rad": phi0,
            "phi1_rad": phi1,
        },
    )


TARGET_BUILDERS: dict[str, Callable[..., BenchmarkTarget]] = {
    "transparent": make_transparent,
    "blocked": make_blocked,
    "point": make_point,
    "narrow_slit": make_narrow_slit,
    "knife_edge_vertical": make_knife_edge_vertical,
    "knife_edge_horizontal": make_knife_edge_horizontal,
    "checkerboard": make_checkerboard,
    "amplitude_step": make_amplitude_step,
    "phase_step": make_phase_step,
    "phase_ramp": make_phase_ramp,
}

TARGET_IDS: tuple[str, ...] = tuple(TARGET_BUILDERS.keys())


def build_target(
    target_id: str,
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
    **kwargs: Any,
) -> BenchmarkTarget:
    if target_id not in TARGET_BUILDERS:
        raise KeyError(
            f"unknown target_id {target_id!r}; expected one of {list(TARGET_IDS)}"
        )
    return TARGET_BUILDERS[target_id](n, pixel_um=pixel_um, **kwargs)


def build_all_targets(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> dict[str, BenchmarkTarget]:
    return {tid: build_target(tid, n, pixel_um=pixel_um) for tid in TARGET_IDS}


def target_definitions_dict(
    n: int = DEFAULT_N,
    *,
    pixel_um: float = DEFAULT_PIXEL_UM,
) -> dict[str, Any]:
    """JSON-serializable catalog definitions (no arrays)."""
    defs: dict[str, Any] = {
        "physical_truth": {k: _jsonable(v) for k, v in PHYSICAL_TRUTH.items()},
        "sampling": {"n": int(n), "pixel_um": float(pixel_um)},
        "targets": {},
    }
    for tid, tgt in build_all_targets(n, pixel_um=pixel_um).items():
        defs["targets"][tid] = tgt.full_metadata()
    return defs


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v
