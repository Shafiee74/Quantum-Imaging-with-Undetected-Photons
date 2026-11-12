"""Intensity vs visibility quantity maps (never silently equated)."""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.resolution.definitions import (
    QUANTITY_INTENSITY,
    QUANTITY_VISIBILITY,
)


def intensity_map_from_result(result: Any) -> np.ndarray:
    """Selected-port ideal intensity image."""
    img = getattr(result, "ideal_signal_image", None)
    if img is None:
        raise ValueError("SimResult.ideal_signal_image is required for intensity maps")
    return np.asarray(img, dtype=np.float64)


def visibility_map_from_result(result: Any, *, eps: float = 1.0e-15) -> np.ndarray:
    """Local Michelson visibility from ideal ± ports (frame 0 / mid).

    ``V = |I+ − I−| / (I+ + I− + ε)``

    This is **not** the intensity image and must not be analyzed as such.
    """
    Ip = np.asarray(result.frames_plus[0], dtype=np.float64)
    Im = np.asarray(result.frames_minus[0], dtype=np.float64)
    return np.abs(Ip - Im) / (Ip + Im + float(eps))


def quantity_maps(result: Any) -> dict[str, np.ndarray]:
    I = intensity_map_from_result(result)
    V = visibility_map_from_result(result)
    return {
        QUANTITY_INTENSITY: I,
        QUANTITY_VISIBILITY: V,
        # Diagnostic only — not used for gated metrics.
        "amplitude_sqrt_intensity_diagnostic": np.sqrt(np.clip(I, 0.0, None)),
    }
