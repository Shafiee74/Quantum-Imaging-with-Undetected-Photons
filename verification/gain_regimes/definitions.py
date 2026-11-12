"""SPDC gain-regime benchmark definitions (Task 14).

Uses the existing physical gain law n̄ = sinh²(r). Does **not** invent
regime labels: approximation-validity regions are reported only as the set of
sweep points that satisfy predeclared numerical tolerances on n̄ ≈ r².
"""
from __future__ import annotations

from typing import Any

import numpy as np

PROFILES: tuple[str, ...] = ("A",)
GRID_N: int = 24
MASTER_SEED: int = 20261414
NUM_FRAMES: int = 1

# Sweep control: Bogoliubov gain parameter r (physical SPDC gain path).
# Covers very small r toward higher gain (n̄ = sinh²(r) grows without bound).
# Log-spaced so low-r approximation region is well sampled.
R_SWEEP: np.ndarray = np.unique(
    np.concatenate(
        [
            np.array([0.0]),
            np.geomspace(1.0e-4, 1.5, 48),
        ]
    )
)

# Documented reference k for reporting an equivalent pump proxy only.
# Not used as production control under fixed_nbar; column is clearly labeled.
# r = k_W * sqrt(P_W), k_mW such that k_W = k_mW * sqrt(1000) if P in W from mW/1000
# With k_per_sqrt_mW = 0.05: r = 0.05 * sqrt(P_mW) when using mW-convention
# matching setup_params.spdc_gain_r_from_pump_mW (k_W * sqrt(P_W)).
REFERENCE_K_PER_SQRT_MW: float = 0.05

# Predeclared tolerances for quantitative n̄ ≈ r² identification (not labels).
APPROX_ABS_TOLERANCES: tuple[float, ...] = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3)
APPROX_REL_TOLERANCES: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)

DETECTOR_EM_GAIN: float = 100.0
DETECTOR_EXPOSURE_S: float = 0.02

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "task": "task14_gain_regimes",
    "physical_gain_law": "n = sinh^2(r)",
    "low_gain_approximation": "n_approx = r^2",
    "sweep_control": "spdc_gain_r (fixed_nbar physical gain path)",
    "pump_reporting": (
        "operating-point pump powers from production meta; "
        "reference_equivalent_pump_mW = (r / k_ref)^2 under documented "
        f"k_per_sqrt_mW={REFERENCE_K_PER_SQRT_MW} (diagnostic proxy only)"
    ),
    "profiles": list(PROFILES),
    "r_sweep_n_points": int(R_SWEEP.size),
    "r_min": float(R_SWEEP.min()),
    "r_max": float(R_SWEEP.max()),
    "approximation_abs_tolerances": list(APPROX_ABS_TOLERANCES),
    "approximation_rel_tolerances": list(APPROX_REL_TOLERANCES),
    "labeling_policy": (
        "Do not arbitrarily label points as low/high gain. "
        "Report only tolerance-qualified approximation-validity regions."
    ),
    "channels": ("clean_optical", "detector_enabled"),
    "master_seed": MASTER_SEED,
}
