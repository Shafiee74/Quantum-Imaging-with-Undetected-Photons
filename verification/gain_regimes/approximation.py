"""Independent gain-law helpers + approximation-threshold scan (Task 14)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from setup_params import (
    k_spdc_per_sqrt_W_from_per_sqrt_mW,
    pump_power_W_from_mW,
)
from verification.gain_regimes.definitions import REFERENCE_K_PER_SQRT_MW
from verification.physics import independent_equations as iref


def n_exact(r: float) -> float:
    return float(iref.mean_photons_sinh_squared(float(r)))


def n_approx_r2(r: float) -> float:
    return float(iref.low_gain_nbar_approximation(float(r)))


def abs_approx_error(r: float) -> float:
    return float(abs(n_exact(r) - n_approx_r2(r)))


def rel_approx_error(r: float) -> float:
    """|(sinh²r − r²) / sinh²r|; 0 at r=0."""
    return float(iref.low_gain_relative_error(float(r)))


def reference_equivalent_pump_mW(
    r: float,
    *,
    k_per_sqrt_mW: float = REFERENCE_K_PER_SQRT_MW,
) -> float | None:
    """Invert r = k_W √P_W for a documented reference k (diagnostic only)."""
    rr = float(r)
    if rr < 0.0 or not np.isfinite(rr):
        return None
    if rr == 0.0:
        return 0.0
    k_W = float(k_spdc_per_sqrt_W_from_per_sqrt_mW(float(k_per_sqrt_mW)))
    if k_W <= 0.0:
        return None
    # P_W = (r / k_W)^2  →  P_mW = 1000 * P_W
    p_W = (rr / k_W) ** 2
    return float(1000.0 * p_W)


def verify_reference_pump_inversion(
    r: float,
    *,
    k_per_sqrt_mW: float = REFERENCE_K_PER_SQRT_MW,
) -> float:
    """Round-trip check: r → P_mW → r' using the same k convention."""
    p_mW = reference_equivalent_pump_mW(r, k_per_sqrt_mW=k_per_sqrt_mW)
    if p_mW is None:
        return float("nan")
    k_W = float(k_spdc_per_sqrt_W_from_per_sqrt_mW(float(k_per_sqrt_mW)))
    return float(k_W * np.sqrt(pump_power_W_from_mW(float(p_mW))))


def find_approximation_thresholds(
    r_values: Sequence[float],
    *,
    abs_tolerances: Sequence[float],
    rel_tolerances: Sequence[float],
) -> list[dict[str, Any]]:
    """Largest contiguous-from-zero region of the sweep satisfying each tol.

    For each (abs_tol, rel_tol) pair, walk the sorted r grid from the origin
    and find the largest index where *all* points with r ≤ r_i satisfy both
    |n−r²| ≤ abs_tol and (r=0 or |(n−r²)/n| ≤ rel_tol).

    No qualitative low/high labels are assigned — only numeric thresholds.
    """
    rs = np.asarray(sorted(float(x) for x in r_values), dtype=np.float64)
    abs_err = np.array([abs_approx_error(float(r)) for r in rs])
    rel_err = np.array([rel_approx_error(float(r)) for r in rs])
    n_vals = np.array([n_exact(float(r)) for r in rs])

    rows: list[dict[str, Any]] = []
    for abs_tol in abs_tolerances:
        for rel_tol in rel_tolerances:
            ok = (abs_err <= float(abs_tol)) & (
                (rs == 0.0) | (rel_err <= float(rel_tol))
            )
            # Contiguous from zero along the sorted grid
            r_max = 0.0
            n_max = 0.0
            n_points = 0
            for i, flag in enumerate(ok):
                if not flag:
                    break
                r_max = float(rs[i])
                n_max = float(n_vals[i])
                n_points = i + 1
            rows.append(
                {
                    "abs_tol": float(abs_tol),
                    "rel_tol": float(rel_tol),
                    "r_max_satisfying": r_max,
                    "n_max_satisfying": n_max,
                    "n_approx_r2_at_r_max": n_approx_r2(r_max),
                    "abs_error_at_r_max": abs_approx_error(r_max),
                    "rel_error_at_r_max": rel_approx_error(r_max),
                    "n_grid_points_in_region": int(n_points),
                    "criterion": (
                        "contiguous_from_r0: |n-r^2|<=abs_tol AND "
                        "(r==0 OR |(n-r^2)/n|<=rel_tol)"
                    ),
                }
            )
    return rows
