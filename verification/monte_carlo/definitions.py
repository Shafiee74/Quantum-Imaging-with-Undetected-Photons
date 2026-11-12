"""Predeclared Monte-Carlo / frame-count conventions (Task 11 / U13)."""
from __future__ import annotations

from typing import Any

PROFILES: tuple[str, ...] = ("A", "B")

# Realization / frame-count ladder (independent MC draws unless noted).
FRAME_COUNTS: tuple[int, ...] = (8, 16, 32, 64)

# Grid for forward runs (keep modest for suite runtime).
GRID_N: int = 24
NBAR: float = 0.02
PHASE_TRUTH_RAD: float = 0.37

# Master seed — all child seeds spawn from this via SeedSequence.
MASTER_SEED: int = 20261111

# Stochasticity channels (mutually exclusive isolation).
CHANNEL_OPTICAL: str = "optical_stochasticity"
CHANNEL_DETECTOR: str = "detector_stochasticity"
CHANNEL_DRIFT: str = "optional_physical_drift"
CHANNELS: tuple[str, ...] = (
    CHANNEL_OPTICAL,
    CHANNEL_DETECTOR,
    CHANNEL_DRIFT,
)

# Confidence intervals: normal approximation mean ± z * SEM (predeclared).
CI_Z: float = 1.959963984540054  # ~95%
CI_LEVEL: float = 0.95

SEED_POLICY: dict[str, Any] = {
    "master_seed": MASTER_SEED,
    "scheme": "numpy.random.SeedSequence",
    "realization_entropy": (
        "SeedSequence(master).spawn(1)[0].spawn(n_channels)[channel_index]"
        ".spawn(max_frame_count)[realization_index]"
    ),
    "independent_realizations": True,
    "deterministic_replay": True,
    "note": (
        "Each (channel, realization) pair has a unique child SeedSequence; "
        "re-running with the same master seed reproduces all draws."
    ),
}

CHANNEL_POLICY: dict[str, Any] = {
    CHANNEL_OPTICAL: {
        "detector_noise": False,
        "latents": {"s_pump": 0.25, "s_align": 0.20, "s_det": 0.0, "s_thermal": 0.0},
        "description": "Optical RNG only (pump/align jitter); detector chain off",
    },
    CHANNEL_DETECTOR: {
        "detector_noise": True,
        "latents": {"s_pump": 0.0, "s_align": 0.0, "s_det": 0.25, "s_thermal": 0.0},
        "description": "Detector RNG only; optical latents quiet",
    },
    CHANNEL_DRIFT: {
        "detector_noise": False,
        "latents": {"s_pump": 0.0, "s_align": 0.0, "s_det": 0.0, "s_thermal": 0.0},
        "drift_phase_rms_rad": 0.15,
        "description": (
            "Optional physical drift: per-realization object phase offset "
            "drawn from N(0, rms^2); optics/detector quiet otherwise"
        ),
    },
}

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "u_id": "U13",
    "frame_counts": list(FRAME_COUNTS),
    "channels": list(CHANNELS),
    "channel_policy": CHANNEL_POLICY,
    "seed_policy": SEED_POLICY,
    "ci_level": CI_LEVEL,
    "ci_z": CI_Z,
    "grid_n": GRID_N,
    "nbar": NBAR,
    "profiles": list(PROFILES),
    "metrics": [
        "mean_intensity",
        "variance",
        "visibility",
        "covariance",
        "normalized_correlation",
        "reconstructed_phase",
        "reconstruction_error",
        "confidence_intervals",
    ],
}
