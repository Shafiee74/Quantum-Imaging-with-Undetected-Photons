"""Independent physics equations for U1–U9 style verification (Task 04).

CRITICAL
========
Expected / reference results must **not** call the production implementation of
the same physical equation. This module may import only:

* the Python standard library
* NumPy
* ``verification.independent_reference`` (already an independent solver for
  Configuration D and shared SPDC/QPM/interference identities)

It must **never** import ``setup_params``, ``optics``, ``simulate``, ``latents``,
``detector``, ``provenance``, or ``qiup_digital_twin_full``.

Where an equation already exists in ``verification.independent_reference``, this
module provides a documented U1–U9 facade that **delegates** to that independent
implementation rather than duplicating the arithmetic.

PRODUCTION IMPLEMENTATION (do not call from here)
-------------------------------------------------
``setup_params.py``, ``optics.py``, ``simulate.py``, …

INDEPENDENT REFERENCE IMPLEMENTATION (this package + independent_reference)
---------------------------------------------------------------------------
Closed-form / literature equations used as expected values in verification.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from verification import independent_reference as _ir
from verification.physics.equation_meta import EquationSpec, register

# ---------------------------------------------------------------------------
# U1 — energy / wavelength consistency
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="energy_conservation_residual",
        equation="ε = 1 − λ_p (1/λ_s + 1/λ_i)   equivalently   ε = (1/λ_p − 1/λ_s − 1/λ_i)/(1/λ_p)",
        variables={
            "lambda_p_nm": "pump vacuum wavelength",
            "lambda_s_nm": "signal vacuum wavelength",
            "lambda_i_nm": "idler vacuum wavelength",
            "epsilon": "relative inverse-wavelength residual",
        },
        si_units={
            "lambda_*_nm": "nm (consistent units; value proportional to metres)",
            "epsilon": "1 (dimensionless)",
        },
        assumptions=(
            "Collinear SPDC energy conservation ω_p = ω_s + ω_i in vacuum.",
            "Dispersion ignored at the wavelength-assignment stage.",
        ),
        configurations=("A", "B", "C", "D"),
        source="SPDC energy conservation; Lemos/Lahiri QIUP source physics; Task-plan U1",
        domain="λ_p, λ_s, λ_i > 0; residual typically |ε| ≲ 1e-4 for rounded 532/810/1550 nm",
        u_suite=("U1",),
        reuses_existing_independent="verification.independent_reference.idler_wavelength_nm (for solve)",
    )
)


def energy_conservation_residual(
    lambda_p_nm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
) -> float:
    """Relative residual of 1/λ_p = 1/λ_s + 1/λ_i (independent of production)."""
    lp, ls, li = (float(lambda_p_nm), float(lambda_s_nm), float(lambda_i_nm))
    if min(lp, ls, li) <= 0.0:
        raise ValueError("wavelengths must be > 0")
    inv_p = 1.0 / lp
    return float((inv_p - 1.0 / ls - 1.0 / li) / inv_p)


register(
    EquationSpec(
        function="idler_from_energy_nm",
        equation="λ_i = 1 / (1/λ_p − 1/λ_s)",
        variables={
            "lambda_p_nm": "pump wavelength",
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength implied by energy conservation",
        },
        si_units={"lambda_*_nm": "nm"},
        assumptions=("Same as energy_conservation_residual.",),
        configurations=("A", "B", "C", "D"),
        source="Energy conservation; facade over independent_reference.idler_wavelength_nm",
        domain="λ_p < λ_s (degenerate/non-degenerate SPDC with λ_i > 0)",
        u_suite=("U1",),
        reuses_existing_independent="verification.independent_reference.idler_wavelength_nm",
    )
)


def idler_from_energy_nm(lambda_p_nm: float, lambda_s_nm: float) -> float:
    return float(_ir.idler_wavelength_nm(lambda_p_nm, lambda_s_nm))


# ---------------------------------------------------------------------------
# U2 — SPDC gain
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="gain_parameter_from_pump",
        equation="r = k √P    (phase-matched single-pass gain parameter)",
        variables={
            "k_per_sqrt_mW": "gain coefficient",
            "pump_mW": "pump power on the interaction",
            "r": "dimensionless Bogoliubov gain parameter",
        },
        si_units={
            "k_per_sqrt_mW": "mW^{-1/2}",
            "pump_mW": "mW",
            "r": "1",
        },
        assumptions=(
            "Undepleted pump.",
            "Phase-matched reference definition of k (Δk absorbed elsewhere).",
        ),
        configurations=("A", "B", "C", "D"),
        source="Parametric amplifier gain; independent_reference.gain_from_pump",
        domain="P ≥ 0, k ≥ 0",
        u_suite=("U2",),
        reuses_existing_independent="verification.independent_reference.gain_from_pump",
    )
)


def gain_parameter_from_pump(k_per_sqrt_mW: float, pump_mW: float) -> float:
    return float(_ir.gain_from_pump(k_per_sqrt_mW, pump_mW))


register(
    EquationSpec(
        function="mean_photons_sinh_squared",
        equation="n̄ = sinh²(r)",
        variables={"r": "gain parameter", "nbar": "mean photons per mode"},
        si_units={"r": "1", "nbar": "1"},
        assumptions=(
            "Single-mode Bogoliubov transformation from vacuum.",
            "Phase-matched interaction so the hyperbolic form applies.",
        ),
        configurations=("A", "B", "C", "D"),
        source="Wiseman/Mølmer / SPDC Bogoliubov; Task-plan U2",
        domain="r real; for |r| ≫ 1, n̄ grows exponentially",
        u_suite=("U2",),
    )
)


def mean_photons_sinh_squared(r: float) -> float:
    """n̄ = sinh²(r) with a stable evaluation for large |r|."""
    x = float(r)
    # sinh²(x) = 0.25*(e^{2x} + e^{-2x} - 2); clamp via expm1 for small x.
    if abs(x) < 1.0e-6:
        return x * x  # leading order
    return float(math.sinh(x) ** 2)


register(
    EquationSpec(
        function="low_gain_nbar_approximation",
        equation="n̄_low ≈ r²    (first-order SPDC)",
        variables={"r": "gain parameter", "nbar_approx": "low-gain mean photons"},
        si_units={"r": "1", "nbar_approx": "1"},
        assumptions=("|r| ≪ 1 so sinh(r) ≈ r.",),
        configurations=("A", "B", "C", "D"),
        source="Low-gain expansion of sinh²(r); Task-plan U2",
        domain="recommended |r| ≲ 0.1 for <1% relative error",
        u_suite=("U2",),
    )
)


def low_gain_nbar_approximation(r: float) -> float:
    x = float(r)
    return x * x


register(
    EquationSpec(
        function="low_gain_relative_error",
        equation="e = (sinh²(r) − r²) / sinh²(r)",
        variables={"r": "gain", "e": "relative error of low-gain law"},
        si_units={"r": "1", "e": "1"},
        assumptions=("Exact n̄ = sinh²(r) > 0.",),
        configurations=("A", "B", "C", "D"),
        source="Task-plan U2 low-gain error diagnostic",
        domain="r ≠ 0",
        u_suite=("U2",),
    )
)


def low_gain_relative_error(r: float) -> float:
    n = mean_photons_sinh_squared(r)
    if n <= 0.0:
        return 0.0
    return float((n - low_gain_nbar_approximation(r)) / n)


register(
    EquationSpec(
        function="nbar_phase_mismatched",
        equation="n̄ = κ² |sinh(g L)/g|²,  g = √(κ² − (Δk/2)²)",
        variables={
            "kappa_per_m": "coupling κ",
            "delta_k_per_m": "phase mismatch Δk",
            "length_m": "interaction length L",
            "nbar": "mean photons per mode",
        },
        si_units={
            "kappa_per_m": "m^{-1}",
            "delta_k_per_m": "rad m^{-1}",
            "length_m": "m",
            "nbar": "1",
        },
        assumptions=("Undepleted pump; collinear coupled-mode model.",),
        configurations=("A", "B", "C", "D"),
        source="independent_reference.closed_form_nbar",
        domain="L ≥ 0; covers hyperbolic and oscillatory branches",
        u_suite=("U2", "U3"),
        reuses_existing_independent="verification.independent_reference.closed_form_nbar",
    )
)


def nbar_phase_mismatched(
    kappa_per_m: float,
    delta_k_per_m: float,
    length_m: float,
) -> float:
    return float(_ir.closed_form_nbar(kappa_per_m, delta_k_per_m, length_m))


# ---------------------------------------------------------------------------
# U3 — phase matching
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="phase_matching_sinc2",
        equation="S(Δk,L) = sinc²(Δk L / 2) with sinc(x)=sin(x)/x, sinc(0)=1",
        variables={
            "delta_k_per_m": "wave-vector mismatch",
            "length_m": "crystal length",
            "S": "normalized phase-matching envelope",
        },
        si_units={
            "delta_k_per_m": "rad m^{-1}",
            "length_m": "m",
            "S": "1",
        },
        assumptions=(
            "Uniform pump / undepleted plane-wave interaction.",
            "Common textbook envelope; not a substitute for the full κ,Δk n̄ law.",
        ),
        configurations=("A", "B", "C", "D"),
        source="SPDC phase-matching envelope; Task-plan U3",
        domain="L ≥ 0",
        u_suite=("U3",),
    )
)


def phase_matching_sinc2(delta_k_per_m: float, length_m: float) -> float:
    x = 0.5 * float(delta_k_per_m) * float(length_m)
    if abs(x) < 1.0e-12:
        return 1.0
    return float((math.sin(x) / x) ** 2)


register(
    EquationSpec(
        function="qpm_delta_k_per_m",
        equation="Δk = k_p − k_s − k_i − 2π m / Λ   (optional sign flip if backward)",
        variables={
            "lambda_*_nm": "wavelengths",
            "temperature_C": "crystal temperature",
            "period_um": "poling period Λ",
            "order": "QPM order m",
            "delta_k": "mismatch",
        },
        si_units={
            "lambda_*_nm": "nm",
            "temperature_C": "°C",
            "period_um": "µm",
            "delta_k": "rad m^{-1}",
        },
        assumptions=(
            "Type-0 ppKTP Sellmeier (Kato & Takaoka) as transcribed in independent_reference.",
            "Collinear geometry.",
        ),
        configurations=("A", "B", "C", "D"),
        source="independent_reference.delta_k_per_m",
        domain="physical ppKTP wavelengths / temperatures in Sellmeier validity",
        u_suite=("U3",),
        reuses_existing_independent="verification.independent_reference.delta_k_per_m",
    )
)


def qpm_delta_k_per_m(
    lambda_p_nm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    temperature_C: float,
    period_um: float,
    order: int = 1,
    *,
    backward: bool = False,
) -> float:
    return float(
        _ir.delta_k_per_m(
            lambda_p_nm,
            lambda_s_nm,
            lambda_i_nm,
            temperature_C,
            period_um,
            order=order,
            backward=backward,
        )
    )


# ---------------------------------------------------------------------------
# U4–U8 — interference / visibility / object limits
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="ideal_interference_intensities",
        equation="I± = (N1+N2)/2 ± Re[C12 e^{iθ}]",
        variables={
            "N1": "mean photon number arm 1",
            "N2": "mean photon number arm 2",
            "C12": "complex coherence <s2† s1>",
            "theta": "interferometer / relative phase θ",
            "I_plus": "constructive port intensity",
            "I_minus": "destructive port intensity",
        },
        si_units={"N1": "1", "N2": "1", "C12": "1", "theta": "rad", "I±": "1"},
        assumptions=(
            "50/50 combiner out± = (d1 ± e^{-iθ} d2)/√2 at the operator level.",
            "First-moment intensities only.",
        ),
        configurations=("A", "B", "C", "D"),
        source="independent_reference.combiner_intensities; QIUP two-alternative interference",
        domain="N1,N2 ≥ 0",
        u_suite=("U4", "U5", "U7", "U8"),
        reuses_existing_independent="verification.independent_reference.combiner_intensities",
    )
)


def ideal_interference_intensities(
    N1: float,
    N2: float,
    C12: complex,
    theta: float,
) -> tuple[float, float]:
    return _ir.combiner_intensities(N1, N2, C12, theta)


register(
    EquationSpec(
        function="visibility_from_moments",
        equation="V = 2 |C12| / (N1 + N2)",
        variables={
            "N1": "arm-1 mean",
            "N2": "arm-2 mean",
            "C12": "complex coherence",
            "V": "fringe visibility",
        },
        si_units={"N1": "1", "N2": "1", "C12": "1", "V": "1"},
        assumptions=(
            "Same combiner model as ideal_interference_intensities.",
            "Equivalent to (Imax−Imin)/(Imax+Imin) for balanced scanning of θ.",
        ),
        configurations=("A", "B", "C", "D"),
        source="independent_reference.fixed_combiner_visibility; Task-plan U8",
        domain="N1+N2 > 0; else V=0",
        u_suite=("U8", "U4", "U5", "U6"),
        reuses_existing_independent="verification.independent_reference.fixed_combiner_visibility",
    )
)


def visibility_from_moments(N1: float, N2: float, C12: complex) -> float:
    return float(_ir.fixed_combiner_visibility(N1, N2, C12))


register(
    EquationSpec(
        function="visibility_from_max_min",
        equation="V = (I_max − I_min) / (I_max + I_min)",
        variables={
            "I_max": "maximum intensity over phase",
            "I_min": "minimum intensity over phase",
            "V": "visibility",
        },
        si_units={"I_max": "1", "I_min": "1", "V": "1"},
        assumptions=("I_max ≥ I_min ≥ 0; I_max+I_min > 0.",),
        configurations=("A", "B", "C", "D"),
        source="Classical fringe definition; Task-plan U8 identity check",
        domain="non-negative intensities",
        u_suite=("U8",),
    )
)


def visibility_from_max_min(I_max: float, I_min: float) -> float:
    mx, mn = float(I_max), float(I_min)
    denom = mx + mn
    if denom <= 0.0:
        return 0.0
    return float((mx - mn) / denom)


register(
    EquationSpec(
        function="sinusoidal_intensity",
        equation="I(θ) = A + B cos(θ + φ)",
        variables={
            "A": "DC offset",
            "B": "modulation amplitude",
            "theta": "known phase step",
            "phi": "object / path phase",
            "I": "intensity",
        },
        si_units={"A": "1", "B": "1", "theta": "rad", "phi": "rad", "I": "1"},
        assumptions=("Single-frequency interference; no higher harmonics.",),
        configurations=("A", "B", "C", "D"),
        source="Phase imaging law; Task-plan U7",
        domain="B ≥ 0; A ≥ |B| for non-negative intensity",
        u_suite=("U7",),
    )
)


def sinusoidal_intensity(A: float, B: float, theta: float, phi: float = 0.0) -> float:
    return float(A + B * math.cos(float(theta) + float(phi)))


register(
    EquationSpec(
        function="transparent_object_seed",
        equation="t_seed = η · T e^{iφ} with T=1, φ=0 ⇒ t_seed = η",
        variables={"eta": "path / mode amplitude factor", "t_seed": "complex idler seed"},
        si_units={"eta": "1", "t_seed": "1"},
        assumptions=("Uniform transparent object; seed multiplies NL2 / return path.",),
        configurations=("A", "B", "C", "D"),
        source="Task-plan U4 transparent control",
        domain="|η| ≤ 1 typically",
        u_suite=("U4",),
    )
)


def transparent_object_seed(eta: complex = 1.0) -> complex:
    return complex(eta)


register(
    EquationSpec(
        function="blocked_object_seed",
        equation="T=0 ⇒ t_seed = 0  (object-mediated coherence removed)",
        variables={"t_seed": "complex idler seed"},
        si_units={"t_seed": "1"},
        assumptions=(
            "Blocked / zero-transmission object.",
            "Incoherent background terms may remain in N1,N2 depending on topology.",
        ),
        configurations=("A", "B", "C", "D"),
        source="Task-plan U5 blocked control",
        domain="always",
        u_suite=("U5",),
    )
)


def blocked_object_seed() -> complex:
    return 0.0 + 0.0j


register(
    EquationSpec(
        function="low_gain_visibility_vs_transmission",
        equation="V(T) = V0 · T   (amplitude transmission; fixed phase, low gain, balanced arms)",
        variables={
            "T": "amplitude transmission (0…1)",
            "V0": "visibility at T=1",
            "V": "visibility",
        },
        si_units={"T": "1", "V0": "1", "V": "1"},
        assumptions=(
            "Low-gain induced coherence with |C12| ∝ T and N1≈N2≈const.",
            "No additional loss/ decoherence beyond T.",
        ),
        configurations=("A", "B", "C", "D"),
        source="Lemos / Viswanathan amplitude law; Task-plan U6",
        domain="T ∈ [0,1], V0 ∈ [0,1]",
        u_suite=("U6",),
    )
)


def low_gain_visibility_vs_transmission(T: float, V0: float = 1.0) -> float:
    return float(max(0.0, min(1.0, float(V0) * float(T))))


# ---------------------------------------------------------------------------
# U9 — spatial mapping
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="far_field_qiup_magnification",
        equation="M_QIUP = (F2 / F1) · (λ_s / λ_i)",
        variables={
            "F1_mm": "idler/Fourier focal length f_u (F1)",
            "F2_mm": "signal/camera focal length f_c (F2)",
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength",
            "M": "far-field QIUP magnification magnitude",
        },
        si_units={
            "F1_mm": "mm",
            "F2_mm": "mm",
            "lambda_*_nm": "nm",
            "M": "1",
        },
        assumptions=(
            "Momentum correlation q_s ≈ −q_i.",
            "Thin-lens Fourier mapping in both arms.",
            "This returns the positive magnitude used as M_QIUP in the reduced-order model.",
        ),
        configurations=("A", "C"),
        source="Lahiri et al. PRA 2015 far-field magnification; PDF §5.7 form",
        domain="F1,F2,λ_s,λ_i > 0",
        u_suite=("U9",),
    )
)


def far_field_qiup_magnification(
    F1_mm: float,
    F2_mm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
) -> float:
    f1, f2 = float(F1_mm), float(F2_mm)
    ls, li = float(lambda_s_nm), float(lambda_i_nm)
    if min(f1, f2, ls, li) <= 0.0:
        raise ValueError("focal lengths and wavelengths must be > 0")
    return float(f2 * ls / (f1 * li))


register(
    EquationSpec(
        function="far_field_signed_object_to_camera_scale",
        equation="M_ff = − (M_s/M_i)·(λ_s/λ_i)·(f_s/f_i)   (q_s=−q_i inversion)",
        variables={
            "m_i": "idler relay magnification",
            "m_s": "signal relay magnification",
            "f_i_mm": "idler Fourier focal length",
            "f_s_mm": "signal Fourier focal length",
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength",
            "M_ff": "signed ρ_c/ρ_o scale",
        },
        si_units={
            "m_i": "1",
            "m_s": "1",
            "f_*_mm": "mm",
            "lambda_*_nm": "nm",
            "M_ff": "1",
        },
        assumptions=(
            "ρ_o = M_i λ_i f_i q_i/(2π), ρ_c = M_s λ_s f_s q_s/(2π), q_s=−q_i.",
        ),
        configurations=("A", "C"),
        source="Lahiri / Viswanathan far-field coordinate map; Task-plan U9/A2",
        domain="nonzero relays and wavelengths",
        u_suite=("U9",),
    )
)


def far_field_signed_object_to_camera_scale(
    *,
    m_i: float = 1.0,
    m_s: float = 1.0,
    f_i_mm: float,
    f_s_mm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
) -> float:
    mi, ms = float(m_i), float(m_s)
    fi, fs = float(f_i_mm), float(f_s_mm)
    ls, li = float(lambda_s_nm), float(lambda_i_nm)
    if min(abs(mi), abs(ms), fi, fs, ls, li) <= 0.0:
        raise ValueError("relays, focals, and wavelengths must be nonzero/positive as required")
    return float(-(ms / mi) * (ls / li) * (fs / fi))


register(
    EquationSpec(
        function="near_field_magnification",
        equation="M_near = M_s / M_u",
        variables={
            "m_u": "idler imaging magnification M_u",
            "m_s": "signal imaging magnification M_s",
            "M_near": "object→camera position-correlation magnification",
        },
        si_units={"m_u": "1", "m_s": "1", "M_near": "1"},
        assumptions=(
            "Position correlation ρ_o = M_u ρ_i, ρ_c = M_s ρ_s with ρ_s≈ρ_i.",
            "No explicit wavelength scaling (unlike far field).",
        ),
        configurations=("B", "D"),
        source="Viswanathan position-correlation QIUP; independent_reference.near_field_index_scale",
        domain="M_u ≠ 0",
        u_suite=("U9",),
        reuses_existing_independent="verification.independent_reference.near_field_index_scale (index form)",
    )
)


def near_field_magnification(m_u: float, m_s: float) -> float:
    mu = float(m_u)
    if abs(mu) <= 0.0:
        raise ValueError("m_u must be nonzero")
    return float(m_s) / mu


register(
    EquationSpec(
        function="near_field_index_scale",
        equation="s = (M_s/M_u)·(p_object / p_camera)",
        variables={
            "m_u": "idler magnification",
            "m_s": "signal magnification",
            "p_object_um": "object-plane sample pitch",
            "p_camera_um": "camera pixel pitch",
            "s": "array index scale",
        },
        si_units={
            "m_u": "1",
            "m_s": "1",
            "p_object_um": "µm",
            "p_camera_um": "µm",
            "s": "1",
        },
        assumptions=("Uniform pixel grids; geometric relay only.",),
        configurations=("B", "D"),
        source="independent_reference.near_field_index_scale",
        domain="pitches > 0, M_u ≠ 0",
        u_suite=("U9",),
        reuses_existing_independent="verification.independent_reference.near_field_index_scale",
    )
)


def near_field_index_scale(
    m_u: float,
    m_s: float,
    p_object_um: float,
    p_camera_um: float,
) -> float:
    return float(_ir.near_field_index_scale(m_u, m_s, p_object_um, p_camera_um))


# ---------------------------------------------------------------------------
# Correlation widths
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="far_field_correlation_sigma_um",
        equation="σ_o = m_i · f_i · λ_i / (2π w_p)    with λ_i, f_i, w_p in consistent length units",
        variables={
            "pump_waist_um": "pump 1/e² intensity waist w_p",
            "far_field_focal_mm": "idler Fourier focal length f_i",
            "lambda_i_nm": "idler wavelength",
            "m_i": "idler relay magnification",
            "sigma_um": "object/Fourier-plane Gaussian std width",
        },
        si_units={
            "pump_waist_um": "µm",
            "far_field_focal_mm": "mm",
            "lambda_i_nm": "nm",
            "m_i": "1",
            "sigma_um": "µm",
        },
        assumptions=(
            "Gaussian pump I∝exp(−2r²/w_p²) ⇒ σ_q = 1/w_p (P14).",
            "ρ_o = m_i λ_i f_i q_i/(2π).",
            "Reduced-order intensity kernel (no full biphoton Φ).",
        ),
        configurations=("A", "C"),
        source="P14 / P07 far-field correlation width; Task-plan U9/U10 analytical width",
        domain="w_p, f_i, λ_i > 0",
        u_suite=("U9", "U10"),
    )
)


def far_field_correlation_sigma_um(
    pump_waist_um: float,
    far_field_focal_mm: float,
    lambda_i_nm: float,
    *,
    m_i: float = 1.0,
) -> float:
    w = float(pump_waist_um)
    f_um = float(far_field_focal_mm) * 1.0e3
    lam_i_um = float(lambda_i_nm) * 1.0e-3
    if min(w, f_um, lam_i_um) <= 0.0:
        raise ValueError("waist, focal length, and wavelength must be > 0")
    return float(float(m_i) * f_um * lam_i_um / (2.0 * math.pi * w))


register(
    EquationSpec(
        function="near_field_conditional_sigma_um",
        equation="From P10 joint Gaussian: σ = 1/√(2α) with α = A λ_d² + B, …",
        variables={
            "length_um": "crystal length",
            "lambda_d_um": "detected wavelength",
            "lambda_u_um": "undetected wavelength",
            "waist_um": "pump waist",
            "sigma_um": "conditional width",
        },
        si_units={
            "length_um": "µm",
            "lambda_*_um": "µm",
            "waist_um": "µm",
            "sigma_um": "µm",
        },
        assumptions=("P10 Gaussian biphoton approximation in the near field.",),
        configurations=("B", "D"),
        source="independent_reference.p10_joint_gaussian",
        domain="positive lengths/wavelengths/waist",
        u_suite=("U9", "U10"),
        reuses_existing_independent="verification.independent_reference.p10_joint_gaussian",
    )
)


def near_field_conditional_sigma_um(
    length_um: float,
    lambda_d_um: float,
    lambda_u_um: float,
    waist_um: float,
) -> float:
    return float(
        _ir.p10_joint_gaussian(length_um, lambda_d_um, lambda_u_um, waist_um)["sigma_um"]
    )


# ---------------------------------------------------------------------------
# Longitudinal idler-lens quadratic phase (Hochrainer / P11)
# ---------------------------------------------------------------------------
register(
    EquationSpec(
        function="equivalent_wavelength_nm",
        equation="λ_eq = λ_s² / λ_i",
        variables={
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength",
            "lambda_eq_nm": "equivalent wavelength",
        },
        si_units={"lambda_*_nm": "nm"},
        assumptions=("Far-field momentum correlation with θ_s/θ_i ≈ λ_s/λ_i.",),
        configurations=("A", "C"),
        source="Hochrainer et al. Optica 2017; P11 Eq. (4)",
        domain="λ_s, λ_i > 0",
        u_suite=("U7", "U9"),
        notes="Used for longitudinal (and ideally transverse) far-field lens phase.",
    )
)


def equivalent_wavelength_nm(lambda_s_nm: float, lambda_i_nm: float) -> float:
    ls, li = float(lambda_s_nm), float(lambda_i_nm)
    if ls <= 0.0 or li <= 0.0:
        raise ValueError("wavelengths must be > 0")
    return ls * ls / li


register(
    EquationSpec(
        function="idler_lens_equivalent_propagation_um",
        equation="d(δ) = f_u² δ / (f_u² + δ²)",
        variables={
            "delta_um": "longitudinal lens displacement δ",
            "focal_mm": "idler lens focal length f_u",
            "d_um": "equivalent free-space propagation",
        },
        si_units={"delta_um": "µm", "focal_mm": "mm", "d_um": "µm"},
        assumptions=("Thin lens; paraxial relay model of Hochrainer supplement.",),
        configurations=("A", "C", "B", "D"),
        source="Hochrainer et al.; P11 supplement",
        domain="f_u > 0; δ real",
        u_suite=("U7", "U9"),
    )
)


def idler_lens_equivalent_propagation_um(delta_um: float, focal_mm: float) -> float:
    f_um = float(focal_mm) * 1.0e3
    dlt = float(delta_um)
    denom = f_um * f_um + dlt * dlt
    if denom <= 0.0:
        raise ValueError("equivalent propagation denominator must be > 0")
    return float(f_um * f_um * dlt / denom)


register(
    EquationSpec(
        function="longitudinal_idler_lens_phase_rad",
        equation="φ(ρ) = g_rt · π · d(δ) · ρ² / (f_c² · λ_eq)",
        variables={
            "rho_um": "camera-plane radial coordinate",
            "delta_z_um": "lens axial displacement",
            "f_u_mm": "idler lens focal length",
            "f_c_mm": "camera lens focal length",
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength",
            "pass_factor": "g_rt round-trip factor (1 two-crystal, 2 Michelson)",
            "phi": "phase [rad]",
        },
        si_units={
            "rho_um": "µm",
            "delta_z_um": "µm",
            "f_*_mm": "mm",
            "lambda_*_nm": "nm",
            "pass_factor": "1",
            "phi": "rad",
        },
        assumptions=(
            "Far-field / momentum-correlation geometry.",
            "Self-retracing Michelson uses g_rt=2; two-crystal single pass uses g_rt=1.",
            "Quadratic term only (no transverse shift).",
        ),
        configurations=("A", "C"),
        source="Hochrainer Optica 2017 adapted to single-crystal round trip; Config-C freeze plan §8",
        domain="f_c, λ_eq > 0",
        u_suite=("U7", "U9"),
        notes="Independent of production optics.idler_lens_misalignment_phase_map.",
    )
)


def longitudinal_idler_lens_phase_rad(
    rho_um: float,
    *,
    delta_z_um: float,
    f_u_mm: float,
    f_c_mm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    pass_factor: float = 1.0,
) -> float:
    d_um = idler_lens_equivalent_propagation_um(delta_z_um, f_u_mm)
    lam_eq_um = equivalent_wavelength_nm(lambda_s_nm, lambda_i_nm) * 1.0e-3
    f_c_um = float(f_c_mm) * 1.0e3
    if f_c_um <= 0.0 or lam_eq_um <= 0.0:
        raise ValueError("f_c and λ_eq must be > 0")
    rho2 = float(rho_um) ** 2
    return float(
        float(pass_factor) * math.pi * d_um * rho2 / (f_c_um * f_c_um * lam_eq_um)
    )


register(
    EquationSpec(
        function="longitudinal_quadratic_coefficient_per_um2",
        equation="a(δ) = g_rt · d(δ) / (2 f_c² λ_eq)   so that φ = 2π a ρ²",
        variables={
            "a": "quadratic phase coefficient",
            "delta_z_um": "lens displacement",
            "f_u_mm": "idler focal length",
            "f_c_mm": "camera focal length",
            "lambda_s_nm": "signal wavelength",
            "lambda_i_nm": "idler wavelength",
            "pass_factor": "g_rt",
        },
        si_units={
            "a": "rad / µm²   (via 2π a ρ² with ρ in µm)",
            "delta_z_um": "µm",
            "f_*_mm": "mm",
            "lambda_*_nm": "nm",
        },
        assumptions=("Same as longitudinal_idler_lens_phase_rad.",),
        configurations=("A", "C"),
        source="Configuration C freeze plan §8.1",
        domain="f_c, λ_eq > 0",
        u_suite=("U7", "U9"),
    )
)


def longitudinal_quadratic_coefficient_per_um2(
    *,
    delta_z_um: float,
    f_u_mm: float,
    f_c_mm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    pass_factor: float = 1.0,
) -> float:
    """Return a such that φ = 2π a ρ² with ρ in µm."""
    d_um = idler_lens_equivalent_propagation_um(delta_z_um, f_u_mm)
    lam_eq_um = equivalent_wavelength_nm(lambda_s_nm, lambda_i_nm) * 1.0e-3
    f_c_um = float(f_c_mm) * 1.0e3
    if f_c_um <= 0.0 or lam_eq_um <= 0.0:
        raise ValueError("f_c and λ_eq must be > 0")
    # φ = g π d ρ² / (f_c² λ_eq) = 2π a ρ²  ⇒  a = g d / (2 f_c² λ_eq)
    return float(float(pass_factor) * d_um / (2.0 * f_c_um * f_c_um * lam_eq_um))


def pass_factor_for_configuration(config: str) -> float:
    """Round-trip factor used by the independent longitudinal lens law."""
    key = str(config).strip().upper()
    if key in {"B", "C"}:
        return 2.0
    if key in {"A", "D"}:
        return 1.0
    raise ValueError(f"unknown configuration {config!r}")


# ---------------------------------------------------------------------------
# Analytical case helper (for evidence tables)
# ---------------------------------------------------------------------------
def analytical_reference_casebook() -> list[dict[str, Any]]:
    """Known closed-form numerical cases used by tests and Task 04 evidence."""
    cases: list[dict[str, Any]] = []

    # U1: exact 532/810/1550 is slightly inexact; use energy-exact pump.
    lp, ls = 532.0, 810.0
    li = idler_from_energy_nm(lp, ls)
    cases.append(
        {
            "case_id": "U1_energy_exact_triplet",
            "u_suite": "U1",
            "function": "energy_conservation_residual",
            "inputs": {"lambda_p_nm": lp, "lambda_s_nm": ls, "lambda_i_nm": li},
            "expected": 0.0,
            "actual": energy_conservation_residual(lp, ls, li),
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )
    # Nominal rounded laboratory wavelengths.
    cases.append(
        {
            "case_id": "U1_rounded_532_810_1550_residual",
            "u_suite": "U1",
            "function": "energy_conservation_residual",
            "inputs": {"lambda_p_nm": 532.0, "lambda_s_nm": 810.0, "lambda_i_nm": 1550.0},
            "expected": energy_conservation_residual(532.0, 810.0, 1550.0),
            "actual": energy_conservation_residual(532.0, 810.0, 1550.0),
            "tolerance": 0.0,
            "configurations": "A;B;C;D",
            "note": "self-consistency of residual definition; |ε|~1.6e-5",
        }
    )

    r = 0.05
    n = mean_photons_sinh_squared(r)
    cases.append(
        {
            "case_id": "U2_sinh2_small_r",
            "u_suite": "U2",
            "function": "mean_photons_sinh_squared",
            "inputs": {"r": r},
            "expected": math.sinh(r) ** 2,
            "actual": n,
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )
    cases.append(
        {
            "case_id": "U2_low_gain_approx_r0.01",
            "u_suite": "U2",
            "function": "low_gain_nbar_approximation",
            "inputs": {"r": 0.01},
            "expected": 1.0e-4,
            "actual": low_gain_nbar_approximation(0.01),
            "tolerance": 0.0,
            "configurations": "A;B;C;D",
        }
    )
    cases.append(
        {
            "case_id": "U2_gain_from_pump",
            "u_suite": "U2",
            "function": "gain_parameter_from_pump",
            "inputs": {"k_per_sqrt_mW": 0.2, "pump_mW": 25.0},
            "expected": 0.2 * math.sqrt(25.0),
            "actual": gain_parameter_from_pump(0.2, 25.0),
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )

    cases.append(
        {
            "case_id": "U3_sinc2_zero_mismatch",
            "u_suite": "U3",
            "function": "phase_matching_sinc2",
            "inputs": {"delta_k_per_m": 0.0, "length_m": 0.01},
            "expected": 1.0,
            "actual": phase_matching_sinc2(0.0, 0.01),
            "tolerance": 0.0,
            "configurations": "A;B;C;D",
        }
    )
    # sinc^2(π) = 0
    dk = 2.0 * math.pi / 0.01
    cases.append(
        {
            "case_id": "U3_sinc2_first_null",
            "u_suite": "U3",
            "function": "phase_matching_sinc2",
            "inputs": {"delta_k_per_m": dk, "length_m": 0.01},
            "expected": 0.0,
            "actual": phase_matching_sinc2(dk, 0.01),
            "tolerance": 1e-12,
            "configurations": "A;B;C;D",
        }
    )

    N1, N2, C = 1.0, 1.0, 0.5 + 0.0j
    ip, im = ideal_interference_intensities(N1, N2, C, 0.0)
    cases.append(
        {
            "case_id": "U8_interference_theta0",
            "u_suite": "U8",
            "function": "ideal_interference_intensities",
            "inputs": {"N1": N1, "N2": N2, "C12": [0.5, 0.0], "theta": 0.0},
            "expected": [1.5, 0.5],
            "actual": [ip, im],
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )
    v_mom = visibility_from_moments(N1, N2, C)
    v_mm = visibility_from_max_min(ip, im)
    cases.append(
        {
            "case_id": "U8_visibility_identity",
            "u_suite": "U8",
            "function": "visibility_from_moments",
            "inputs": {"N1": N1, "N2": N2, "C12": [0.5, 0.0]},
            "expected": 0.5,
            "actual": v_mom,
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
            "note": f"max/min visibility={v_mm}",
        }
    )

    cases.append(
        {
            "case_id": "U4_transparent_seed",
            "u_suite": "U4",
            "function": "transparent_object_seed",
            "inputs": {"eta": 1.0},
            "expected": [1.0, 0.0],
            "actual": [transparent_object_seed(1.0).real, transparent_object_seed(1.0).imag],
            "tolerance": 0.0,
            "configurations": "A;B;C;D",
        }
    )
    cases.append(
        {
            "case_id": "U5_blocked_seed",
            "u_suite": "U5",
            "function": "blocked_object_seed",
            "inputs": {},
            "expected": [0.0, 0.0],
            "actual": [blocked_object_seed().real, blocked_object_seed().imag],
            "tolerance": 0.0,
            "configurations": "A;B;C;D",
        }
    )
    cases.append(
        {
            "case_id": "U6_visibility_vs_T",
            "u_suite": "U6",
            "function": "low_gain_visibility_vs_transmission",
            "inputs": {"T": 0.4, "V0": 0.8},
            "expected": 0.32,
            "actual": low_gain_visibility_vs_transmission(0.4, 0.8),
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )
    cases.append(
        {
            "case_id": "U7_sinusoid_pi",
            "u_suite": "U7",
            "function": "sinusoidal_intensity",
            "inputs": {"A": 1.0, "B": 0.5, "theta": math.pi, "phi": 0.0},
            "expected": 0.5,
            "actual": sinusoidal_intensity(1.0, 0.5, math.pi, 0.0),
            "tolerance": 1e-15,
            "configurations": "A;B;C;D",
        }
    )

    M = far_field_qiup_magnification(75.0, 150.0, 810.0, 1550.0)
    cases.append(
        {
            "case_id": "U9_far_field_M_QIUP",
            "u_suite": "U9",
            "function": "far_field_qiup_magnification",
            "inputs": {
                "F1_mm": 75.0,
                "F2_mm": 150.0,
                "lambda_s_nm": 810.0,
                "lambda_i_nm": 1550.0,
            },
            "expected": 150.0 * 810.0 / (75.0 * 1550.0),
            "actual": M,
            "tolerance": 1e-15,
            "configurations": "A;C",
        }
    )
    cases.append(
        {
            "case_id": "U9_near_field_M",
            "u_suite": "U9",
            "function": "near_field_magnification",
            "inputs": {"m_u": 2.0, "m_s": -1.0},
            "expected": -0.5,
            "actual": near_field_magnification(2.0, -1.0),
            "tolerance": 0.0,
            "configurations": "B;D",
        }
    )
    sig = far_field_correlation_sigma_um(250.0, 75.0, 1550.0, m_i=1.0)
    cases.append(
        {
            "case_id": "U9_far_field_sigma",
            "u_suite": "U9",
            "function": "far_field_correlation_sigma_um",
            "inputs": {
                "pump_waist_um": 250.0,
                "far_field_focal_mm": 75.0,
                "lambda_i_nm": 1550.0,
                "m_i": 1.0,
            },
            "expected": (1.0 * 75.0e3 * 1.550) / (2.0 * math.pi * 250.0),
            "actual": sig,
            "tolerance": 1e-12,
            "configurations": "A;C",
        }
    )

    lam_eq = equivalent_wavelength_nm(810.0, 1550.0)
    cases.append(
        {
            "case_id": "U7_lambda_eq",
            "u_suite": "U7",
            "function": "equivalent_wavelength_nm",
            "inputs": {"lambda_s_nm": 810.0, "lambda_i_nm": 1550.0},
            "expected": 810.0 ** 2 / 1550.0,
            "actual": lam_eq,
            "tolerance": 1e-12,
            "configurations": "A;C",
        }
    )
    # δ → 0 ⇒ d ≈ δ
    d0 = idler_lens_equivalent_propagation_um(0.0, 15.0)
    cases.append(
        {
            "case_id": "U7_d_delta_zero",
            "u_suite": "U7",
            "function": "idler_lens_equivalent_propagation_um",
            "inputs": {"delta_um": 0.0, "focal_mm": 15.0},
            "expected": 0.0,
            "actual": d0,
            "tolerance": 0.0,
            "configurations": "A;C",
        }
    )
    # δ = f ⇒ d = f/2
    d_half = idler_lens_equivalent_propagation_um(15.0e3, 15.0)
    cases.append(
        {
            "case_id": "U7_d_delta_equals_f",
            "u_suite": "U7",
            "function": "idler_lens_equivalent_propagation_um",
            "inputs": {"delta_um": 15.0e3, "focal_mm": 15.0},
            "expected": 7.5e3,
            "actual": d_half,
            "tolerance": 1e-9,
            "configurations": "A;C",
        }
    )
    phi = longitudinal_idler_lens_phase_rad(
        100.0,
        delta_z_um=40.0,
        f_u_mm=75.0,
        f_c_mm=150.0,
        lambda_s_nm=810.0,
        lambda_i_nm=1550.0,
        pass_factor=2.0,
    )
    a = longitudinal_quadratic_coefficient_per_um2(
        delta_z_um=40.0,
        f_u_mm=75.0,
        f_c_mm=150.0,
        lambda_s_nm=810.0,
        lambda_i_nm=1550.0,
        pass_factor=2.0,
    )
    cases.append(
        {
            "case_id": "U7_longitudinal_phase_matches_a",
            "u_suite": "U7",
            "function": "longitudinal_idler_lens_phase_rad",
            "inputs": {
                "rho_um": 100.0,
                "delta_z_um": 40.0,
                "f_u_mm": 75.0,
                "f_c_mm": 150.0,
                "lambda_s_nm": 810.0,
                "lambda_i_nm": 1550.0,
                "pass_factor": 2.0,
            },
            "expected": 2.0 * math.pi * a * (100.0 ** 2),
            "actual": phi,
            "tolerance": 1e-10,
            "configurations": "C",
        }
    )
    cases.append(
        {
            "case_id": "U7_pass_factor_C",
            "u_suite": "U7",
            "function": "pass_factor_for_configuration",
            "inputs": {"config": "C"},
            "expected": 2.0,
            "actual": pass_factor_for_configuration("C"),
            "tolerance": 0.0,
            "configurations": "C",
        }
    )
    return cases


__all__ = [
    "analytical_reference_casebook",
    "blocked_object_seed",
    "energy_conservation_residual",
    "equivalent_wavelength_nm",
    "far_field_correlation_sigma_um",
    "far_field_qiup_magnification",
    "far_field_signed_object_to_camera_scale",
    "gain_parameter_from_pump",
    "ideal_interference_intensities",
    "idler_from_energy_nm",
    "idler_lens_equivalent_propagation_um",
    "longitudinal_idler_lens_phase_rad",
    "longitudinal_quadratic_coefficient_per_um2",
    "low_gain_nbar_approximation",
    "low_gain_relative_error",
    "low_gain_visibility_vs_transmission",
    "mean_photons_sinh_squared",
    "nbar_phase_mismatched",
    "near_field_conditional_sigma_um",
    "near_field_index_scale",
    "near_field_magnification",
    "pass_factor_for_configuration",
    "phase_matching_sinc2",
    "qpm_delta_k_per_m",
    "sinusoidal_intensity",
    "transparent_object_seed",
    "visibility_from_max_min",
    "visibility_from_moments",
]
