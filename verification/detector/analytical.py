"""Analytical mean/variance references for the existing EMCCD chain.

These are closed-form expectations derived from the documented
``detector.py`` models — not a re-implementation of the stochastic sampler.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from detector import detector_expected_electron_mean
from latents import LatentState
from setup_params import dark_rate_from_setup, read_noise_e_from_setup


def pre_poisson_lambda(
    I: np.ndarray,
    state: LatentState,
    *,
    prnu: np.ndarray | None = None,
) -> np.ndarray:
    """Delegate to production expected-electron mean (single source of truth)."""
    return detector_expected_electron_mean(I, state, prnu=prnu)


def analytical_expectations(
    case_id: str,
    *,
    I_value: float,
    state: LatentState,
    g_adu: float,
    bias_adu: float,
    full_well_e: float,
) -> dict[str, Any]:
    """Return labeled analytical mean/variance targets in ADU (approx).

    Soft saturation is linearized when ``G·λ ≪ full_well``.
    """
    setup = state.get_setup()
    e = state.effective()
    QE = float(setup.QE)
    G = max(float(state.em_gain), 1.0)
    t = float(state.exposure_s)
    dark = float(dark_rate_from_setup(setup, e["s_cool"]))
    cic = float(setup.cic0) * (0.5 + float(e["s_det"]))
    s_read = float(read_noise_e_from_setup(setup, e["s_det"]))
    I = float(I_value)
    optical_bg = float(getattr(setup, "optical_background_photons", 0.0) or 0.0)

    lam = QE * 1.0 * (max(I, 0.0) + optical_bg) + dark * t + cic
    # accidental term only if not physical SPDC — keep 0 for our cases
    if not bool(getattr(setup, "use_physical_spdc_gain", False)):
        lam = lam + float(setup.accidental_e_at_s1) * float(e["s_gain"]) ** 2

    # Linear EM + read (ignore soft-sat when well below full well)
    mean_e = G * lam  # before read (read mean 0)
    # F²≈2 register + read
    var_e = (2.0 * (G ** 2) * lam) + (s_read ** 2)
    # Soft-sat note
    linear_ok = bool(mean_e < 0.05 * float(full_well_e))

    mean_adu = float(g_adu) * mean_e + float(bias_adu)
    var_adu = (float(g_adu) ** 2) * var_e

    # Case-specific refinements / labels
    notes = []
    if case_id == "ideal":
        notes.append("noise sources zeroed; still register EM at G=1 ⇒ Var≈2λ in e-")
        notes.append("mean ADU ≈ g·QE·I + bias")
    elif case_id == "shot_noise":
        notes.append("dark=cic=read=0; Var_e ≈ 2 G² λ with G=1")
    elif case_id == "read_noise":
        notes.append("I=dark=cic=0 ⇒ λ=0, mean_e=0, Var_e=σ_read²")
        mean_e = 0.0
        var_e = s_read ** 2
        mean_adu = float(bias_adu)
        var_adu = (float(g_adu) ** 2) * var_e
    elif case_id == "dark":
        notes.append("I=cic=read=0; λ=dark·t")
    elif case_id == "cic":
        notes.append("I=dark=read=0; λ=cic")
    elif case_id == "gain":
        notes.append("EM gain G≫1; E=Gλ, Var≈2 G² λ (+ read if present)")
    elif case_id == "combined":
        notes.append("all stages enabled at configured levels")

    return {
        "case_id": case_id,
        "lambda_e": float(lam),
        "em_gain": float(G),
        "mean_electrons_pre_read": float(mean_e),
        "var_electrons": float(var_e),
        "mean_adu": float(mean_adu),
        "var_adu": float(var_adu),
        "std_adu": float(np.sqrt(var_adu)),
        "read_noise_e": float(s_read),
        "QE": float(QE),
        "dark_rate_e_per_s": float(dark),
        "cic_e": float(cic),
        "g_adu": float(g_adu),
        "bias_adu": float(bias_adu),
        "linear_em_approx_ok": linear_ok,
        "excess_noise_factor_F2": 2.0,
        "notes": notes,
        "reference_kind": "analytical_expectation_from_documented_detector_model",
    }
