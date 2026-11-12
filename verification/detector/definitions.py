"""Predeclared detector-validation cases and analytical expectations (Task 12).

Uses the **existing** ``detector.py`` chain only. Analytical formulas below are
references for mean/variance checks — not a second detector implementation.
"""
from __future__ import annotations

from typing import Any

MASTER_SEED: int = 20261212
N_PIXELS: int = 128  # square side → N_PIXELS² samples per frame
N_FRAMES: int = 48   # MC frames for distribution estimates

# Optical input used when a non-zero signal is required [photons/px/exposure]
SIGNAL_PHOTONS: float = 20.0

CASE_IDS: tuple[str, ...] = (
    "ideal",
    "shot_noise",
    "read_noise",
    "dark",
    "cic",
    "gain",
    "combined",
)

# Map case id → evidence subdirectory name
CASE_DATA_DIRS: dict[str, str] = {
    "ideal": "ideal",
    "shot_noise": "shot_noise",
    "read_noise": "read_noise",
    "dark": "dark",
    "cic": "cic",
    "gain": "gain",
    "combined": "combined",
}

CHAIN_STAGES_DOCUMENTED: tuple[str, ...] = (
    "optical intensity",
    "photons/photoelectrons via QE·PRNU",
    "dark current",
    "CIC (pre-EM)",
    "Poisson shot",
    "hot pixels (pre-EM)",
    "EM Gamma register (F²≈2)",
    "soft saturation",
    "signed read noise (post-EM)",
    "bias",
    "ADC digitization / clip",
)

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "u_id": "U12",
    "implementation": "detector._detector_pc_frame + helpers",
    "second_detector_model": False,
    "experimental_tuning": False,
    "chain_stages": list(CHAIN_STAGES_DOCUMENTED),
    "cases": list(CASE_IDS),
    "analytical_notes": {
        "pre_poisson_mean": (
            "λ = QE·PRNU·(I + optical_bg) + dark_rate·t_exp + cic0·(0.5+s_det) "
            "[detector_expected_electron_mean]"
        ),
        "poisson": "n ~ Poisson(λ); E=λ, Var=λ",
        "em_register_fixed_n": (
            "n_em | n ~ Gamma(shape=n, scale=G); E=nG, Var=n G²"
        ),
        "em_register_poisson_n": (
            "Law of total variance: E=Gλ, Var=G²λ + G²λ = 2 G² λ (F²≈2)"
        ),
        "read_noise": "additive N(0, σ_read²) in electrons post-EM",
        "adu": "ADU ≈ round((n_sat + read)·g_adu + bias_adu), clipped to [0, max_adu]",
        "ideal_mean": (
            "With dark=cic=read=hot=0, PRNU=1, G=1, λ≪full_well: "
            "E[ADU] ≈ g_adu·QE·I + bias_adu"
        ),
    },
    "n_pixels_side": N_PIXELS,
    "n_frames": N_FRAMES,
    "signal_photons": SIGNAL_PHOTONS,
    "master_seed": MASTER_SEED,
}
