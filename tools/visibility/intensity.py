"""Intensity preparation for visibility analysis (no artificial DC removal)."""
from __future__ import annotations

from enum import Enum

import numpy as np


class BackgroundMode(str, Enum):
    """How detector frames are prepared before sinusoidal fitting."""

    RAW_NO_DARK = "RAW_NO_DARK"
    # Future: per-acquisition bias map subtraction validated against calibration data.
    CALIBRATED_BIAS = "CALIBRATED_BIAS"


def prepare_intensity_frame(
    raw_uint16: np.ndarray,
    *,
    mode: BackgroundMode = BackgroundMode.RAW_NO_DARK,
    bias_adu: np.ndarray | None = None,
) -> np.ndarray:
    """Return float64 intensities for fitting. RAW: no offset, clip, or scaling."""
    frame = np.asarray(raw_uint16, dtype=np.float64)
    if mode == BackgroundMode.RAW_NO_DARK:
        return frame
    if mode == BackgroundMode.CALIBRATED_BIAS:
        if bias_adu is None:
            raise NotImplementedError(
                "CALIBRATED_BIAS requires a validated per-acquisition bias map; not enabled."
            )
        b = np.asarray(bias_adu, dtype=np.float64)
        if b.shape != frame.shape:
            raise ValueError("bias_adu shape must match frame")
        return frame - b
    raise ValueError(f"Unknown background mode: {mode}")
