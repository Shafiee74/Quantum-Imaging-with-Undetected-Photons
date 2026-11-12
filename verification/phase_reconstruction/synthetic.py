"""Synthetic four-step intensity generators for reconstruction demos/tests.

Frames follow the shared model
``I(θ) = A + B cos(θ + φ)``
which is equivalent to ``A + C cos θ + S sin θ`` with
``C = B cos φ``, ``S = -B sin φ``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.phase_reconstruction.constants import FOUR_STEP_PHASES_RAD
from verification.phase_reconstruction.metrics import wrap_phase


def synthesize_four_step_frames(
    dc_A: np.ndarray | float,
    modulation_B: np.ndarray | float,
    phase_rad: np.ndarray | float,
    *,
    phases_rad: tuple[float, ...] = FOUR_STEP_PHASES_RAD,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Build four (or N) intensity frames from known A, B, φ."""
    A = np.asarray(dc_A, dtype=np.float64)
    B = np.asarray(modulation_B, dtype=np.float64)
    phi = np.asarray(phase_rad, dtype=np.float64)
    phases = np.asarray(list(phases_rad), dtype=np.float64)
    frames = np.stack(
        [A + B * np.cos(theta + phi) for theta in phases],
        axis=0,
    )
    if noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        frames = frames + rng.normal(0.0, float(noise_sigma), size=frames.shape)
    return {
        "frames": frames.astype(np.float64),
        "phases_rad": tuple(float(p) for p in phases),
        "truth_dc_A": np.asarray(A + np.zeros_like(phi), dtype=np.float64),
        "truth_modulation_B": np.asarray(B + np.zeros_like(phi), dtype=np.float64),
        "truth_phase_rad": wrap_phase(phi),
        "truth_phase_unwrapped_rad": np.asarray(phi, dtype=np.float64),
        "noise_sigma": float(noise_sigma),
    }


def constant_phase_maps(
    shape: tuple[int, int],
    *,
    phase_rad: float,
    dc_A: float = 1.0,
    modulation_B: float = 0.5,
) -> dict[str, np.ndarray]:
    h, w = shape
    return {
        "dc_A": np.full((h, w), float(dc_A), dtype=np.float64),
        "modulation_B": np.full((h, w), float(modulation_B), dtype=np.float64),
        "phase_rad": np.full((h, w), float(phase_rad), dtype=np.float64),
    }


def phase_ramp_maps(
    shape: tuple[int, int],
    *,
    phi0: float = -np.pi,
    phi1: float = 2.5 * np.pi,
    dc_A: float = 1.0,
    modulation_B: float = 0.45,
) -> dict[str, np.ndarray]:
    """Horizontal unwrapped ramp spanning more than 2π (unwrap justified)."""
    h, w = shape
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)
    ramp = phi0 + (phi1 - phi0) * x[None, :]
    phase = np.broadcast_to(ramp, (h, w)).copy()
    return {
        "dc_A": np.full((h, w), float(dc_A), dtype=np.float64),
        "modulation_B": np.full((h, w), float(modulation_B), dtype=np.float64),
        "phase_rad": phase,
    }


def phase_step_maps(
    shape: tuple[int, int],
    *,
    phi_left: float = -0.6,
    phi_right: float = 1.8,
    dc_A: float = 1.0,
    modulation_B: float = 0.5,
) -> dict[str, np.ndarray]:
    h, w = shape
    phase = np.full((h, w), float(phi_left), dtype=np.float64)
    phase[:, w // 2 :] = float(phi_right)
    return {
        "dc_A": np.full((h, w), float(dc_A), dtype=np.float64),
        "modulation_B": np.full((h, w), float(modulation_B), dtype=np.float64),
        "phase_rad": phase,
    }


def amplitude_phase_maps(
    shape: tuple[int, int],
    *,
    dc_base: float = 1.0,
    mod_base: float = 0.35,
) -> dict[str, np.ndarray]:
    """Smooth amplitude envelope + 2-D phase object."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    yn = (yy - (h - 1) / 2.0) / max(h / 2.0, 1.0)
    xn = (xx - (w - 1) / 2.0) / max(w / 2.0, 1.0)
    envelope = np.exp(-0.5 * (xn * xn + yn * yn) / (0.55**2))
    A = dc_base * (0.55 + 0.45 * envelope)
    B = mod_base * (0.25 + 0.75 * envelope)
    phase = 0.9 * xn + 0.55 * yn + 0.35 * np.sin(2.5 * xn) * np.cos(2.0 * yn)
    return {
        "dc_A": A.astype(np.float64),
        "modulation_B": B.astype(np.float64),
        "phase_rad": phase.astype(np.float64),
    }
