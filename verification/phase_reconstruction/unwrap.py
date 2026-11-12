"""Optional spatial phase unwrapping helpers.

Unwrapping is applied only when the caller requests it and the map is
physically continuous enough for a 2-D unwrap to be justified (e.g. a
smooth phase ramp). Discontinuous objects should keep wrapped phase.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.phase_reconstruction.metrics import wrap_phase


def unwrap_phase_2d(
    phase_rad: np.ndarray,
    *,
    wrap_around_axes: tuple[bool, bool] | None = None,
) -> np.ndarray:
    """2-D phase unwrap via ``numpy.unwrap`` along rows then columns.

    This is a lightweight, dependency-free unwrap suitable for synthetic
    ramps. It is **not** a quality-guided experimental unwrap engine.
    """
    phi = np.asarray(phase_rad, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("unwrap_phase_2d expects a 2-D phase map")
    # Axis 1 (columns) first, then axis 0 (rows) — classic raster unwrap.
    out = np.unwrap(phi, axis=1)
    out = np.unwrap(out, axis=0)
    if wrap_around_axes is not None:
        # Reserved for future periodic-boundary support; currently unused.
        _ = wrap_around_axes
    return out.astype(np.float64)


def maybe_unwrap(
    phase_rad: np.ndarray,
    *,
    enabled: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    """Optionally unwrap; always returns wrapped phase as well."""
    wrapped = wrap_phase(phase_rad)
    if not enabled:
        return {
            "phase_wrapped_rad": wrapped,
            "phase_unwrapped_rad": None,
            "unwrap_enabled": False,
            "unwrap_reason": reason or "not_requested",
        }
    unwrapped = unwrap_phase_2d(wrapped)
    return {
        "phase_wrapped_rad": wrapped,
        "phase_unwrapped_rad": unwrapped,
        "unwrap_enabled": True,
        "unwrap_reason": reason or "requested",
    }
