"""Independent analytic reference for Figure 3(a) gain verification.

Implements the Bogoliubov vacuum relation for a single spatiotemporal mode:

    n̄ = sinh²(r)

using only the standard library ``math`` module. This module must **not**
import production gain helpers (``setup_params.sinh_squared_stable``,
``optics.resolve_spdc_source_nbar``) or the U2 ``independent_equations`` module.
"""
from __future__ import annotations

import math


def mean_photons_per_mode_independent(r: float) -> float:
    """n̄ = sinh²(r) via the identity sinh²(x) = (e^{2x} + e^{-2x} - 2) / 4.

    Uses ``expm1`` for stability at small |x| without calling ``sinh`` directly,
    so the evaluation path differs from production ``sinh_squared_stable``.
    """
    x = float(r)
    if x == 0.0:
        return 0.0
    # sinh²(x) = (e^{2x} + e^{-2x} - 2) / 4 = (expm1(2x) + expm1(-2x)) / 4
    hyp = math.expm1(2.0 * x) + math.expm1(-2.0 * x)
    return 0.25 * hyp


def low_gain_approximation(r: float) -> float:
    """First-order expansion n̄ ≈ r² (valid for |r| ≪ 1)."""
    x = float(r)
    return x * x


def low_gain_relative_error_vs_exact(r: float) -> float:
    """|(n̄_exact − r²) / n̄_exact|; undefined at r=0 → 0."""
    n = mean_photons_per_mode_independent(r)
    if n <= 0.0:
        return 0.0
    return abs(n - low_gain_approximation(r)) / n
