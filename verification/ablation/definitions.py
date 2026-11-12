"""Predeclared noise/distortion ablation inventory (Task 13).

Only mechanisms that exist in production ``optics`` / ``detector`` / ``latents``
are listed. No invented physics.
"""
from __future__ import annotations

from typing import Any

PROFILES: tuple[str, ...] = ("A",)
GRID_N: int = 24
MASTER_SEED: int = 20261313
NBAR: float = 0.02
PHI_TRUTH_RAD: float = 0.37

# ---------------------------------------------------------------------------
# Mechanism catalog — each maps to real controls in the codebase
# ---------------------------------------------------------------------------
MECHANISMS: dict[str, dict[str, Any]] = {
    "photon_shot_noise": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {},
        "em_gain": 1.0,
        "source": "detector.sample_input_electrons (Poisson)",
    },
    "read_noise": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "read_noise0_e": 8.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {},
        "em_gain": 1.0,
        "source": "detector signed_read_noise_post_EM",
    },
    "dark_current": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 5.0,
            "cic0": 0.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {},
        "em_gain": 1.0,
        "source": "detector dark_current via dark_rate_from_setup",
    },
    "cic": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 1.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {},
        "em_gain": 1.0,
        "source": "detector CIC_pre_EM (cic0*(0.5+s_det))",
    },
    "prnu": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.08,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {"s_det": 1.0},
        "em_gain": 1.0,
        "source": "detector._fixed_patterns PRNU (σ·s_det)",
    },
    "hot_pixels": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.02,
        },
        "latents": {"s_cool": 1.0},
        "em_gain": 1.0,
        "source": "detector._fixed_patterns hot pixels (frac·s_cool)",
    },
    "optical_blur": {
        "stage_b": True,
        "controls": {"detector_noise": False},
        "setup": {},
        "latents": {"s_focus": 0.6},
        "em_gain": 1.0,
        "source": "optics resolve_forward / imaging PSF vs s_focus",
    },
    "phase_drift": {
        "stage_b": True,
        "controls": {"detector_noise": False},
        "setup": {},
        "latents": {},
        "em_gain": 1.0,
        "object_phase_offset_rad": 0.25,
        "source": "object phi offset (physical phase distortion on supported object map)",
    },
    "overlap_misalignment": {
        "stage_b": True,
        "controls": {"detector_noise": False},
        "setup": {},
        "latents": {"s_align": 0.35},
        "em_gain": 1.0,
        "source": "optics s_align (overlap / V mottling / phase jitter)",
    },
    "pump_fluctuation": {
        "stage_b": True,
        "controls": {"detector_noise": False},
        "setup": {},
        "latents": {"s_pump": 0.35},
        "em_gain": 1.0,
        "source": "optics s_pump (pump power & beam walk RNG)",
    },
    "optical_drift": {
        "stage_b": True,
        "controls": {"detector_noise": False},
        "setup": {},
        "latents": {"s_thermal": 0.5},
        "em_gain": 1.0,
        "source": "optics s_thermal (phase walk / beam drift vs frame index)",
    },
    "detector_em_gain": {
        "stage_b": True,
        "controls": {"detector_noise": True},
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "read_noise0_e": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
        },
        "latents": {},
        "em_gain": 100.0,
        "source": "detector sample_em_register (Gamma EM, F²≈2)",
    },
}

# Stage C — selected combinations of implemented mechanisms only
COMBINATIONS: dict[str, tuple[str, ...]] = {
    "shot_plus_read": ("photon_shot_noise", "read_noise"),
    "detector_pedestal": ("dark_current", "cic", "read_noise"),
    "optics_alignment_pump": ("overlap_misalignment", "pump_fluctuation"),
    "optics_thermal_align": ("optical_drift", "overlap_misalignment"),
    "detector_full_readout": (
        "photon_shot_noise",
        "read_noise",
        "dark_current",
        "cic",
        "prnu",
        "hot_pixels",
        "detector_em_gain",
    ),
}

# Stage D — full configured model (union of Stage-B mechanisms)
FULL_MODEL_MECHANISMS: tuple[str, ...] = tuple(MECHANISMS.keys())

CLEAN_SETUP_BASE: dict[str, Any] = {
    "enable_phase_step_acquisition": True,
    "dark0": 0.0,
    "cic0": 0.0,
    "read_noise_model": "datasheet_fixed",
    "read_noise0_e": 0.0,
    "prnu_sigma": 0.0,
    "hot_frac_at_s1": 0.0,
    "optical_background_photons": 0.0,
}

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "task": "task13_noise_ablation",
    "rewrite_policy": "reuse existing implementations; do not invent missing physics",
    "mechanisms": {k: {"source": v["source"]} for k, v in MECHANISMS.items()},
    "combinations": {k: list(v) for k, v in COMBINATIONS.items()},
    "full_model": list(FULL_MODEL_MECHANISMS),
    "profiles": list(PROFILES),
    "metrics": [
        "intensity",
        "visibility",
        "phase",
        "snr",
        "reconstruction_error",
        "covariance",
        "correlation",
    ],
    "master_seed": MASTER_SEED,
}
