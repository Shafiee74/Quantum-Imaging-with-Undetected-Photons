"""Phase-step schedule constants for shared reconstruction.

These values match the production four-step PSI schedule numerically but are
owned by the verification package so experimental callers need not import
``setup_params``.
"""
from __future__ import annotations

import math

# I0, Iπ/2, Iπ, I3π/2
FOUR_STEP_PHASES_RAD: tuple[float, ...] = (
    0.0,
    0.5 * math.pi,
    math.pi,
    1.5 * math.pi,
)

FOUR_STEP_LABELS: tuple[str, ...] = ("I0", "I_pi_2", "I_pi", "I_3pi_2")
