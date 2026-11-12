"""Optional final-camera collection envelope policy hooks.

``optics.py`` imports these helpers inside try/except so a missing package is
non-fatal. The helpers below preserve production envelope behaviour while
allowing uniform-collection diagnostics to instrument intensities when present.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_PRE_PSF: dict[str, Any] | None = None
_POST_PSF: dict[str, Any] | None = None


def apply_policy_to_env(
    *,
    env: np.ndarray,
    camera_mode: str,
    final_camera_pump_envelope_applied: bool,
    final_camera_pump_envelope_equation: str,
    N: int,
    cx: float,
    cy: float,
    xx: np.ndarray,
    yy: np.ndarray,
    w_px: float,
    imaging_mode_far: str,
    imaging_mode_near: str,
) -> dict[str, Any] | None:
    """Return ``None`` to keep the caller envelope unchanged."""
    del (
        env,
        camera_mode,
        final_camera_pump_envelope_applied,
        final_camera_pump_envelope_equation,
        N,
        cx,
        cy,
        xx,
        yy,
        w_px,
        imaging_mode_far,
        imaging_mode_near,
    )
    return None


def record_pre_psf_intensity(
    *,
    flux_scale: float,
    env: np.ndarray,
    I_plus_mode: np.ndarray,
    I_plus_pre_psf: np.ndarray,
    I_minus_pre_psf: np.ndarray,
) -> None:
    global _PRE_PSF
    _PRE_PSF = {
        "flux_scale": float(flux_scale),
        "env_mean": float(np.mean(env)),
        "I_plus_mode_mean": float(np.mean(I_plus_mode)),
        "I_plus_pre_psf_mean": float(np.mean(I_plus_pre_psf)),
        "I_minus_pre_psf_mean": float(np.mean(I_minus_pre_psf)),
    }


def record_post_psf_intensity(*, I_plus: np.ndarray, sig_opt: float) -> None:
    global _POST_PSF
    _POST_PSF = {
        "I_plus_mean": float(np.mean(I_plus)),
        "sig_opt": float(sig_opt),
    }
