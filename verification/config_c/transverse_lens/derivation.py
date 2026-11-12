"""Independent derivation of Config-C transverse idler-lens phase (Task 17).

Does **not** import production ``optics`` / ``setup_params`` / ``simulate``.

Physical starting point (thin lens, paraxial)
---------------------------------------------
A thin lens of focal length ``f`` with geometric centre displaced by
``Δ = (Δx, Δy)`` has phase

    φ_L(r) = −(k / 2f) |r − Δ|²
           = −(k / 2f)|r|² + (k / f) Δ·r − (k / 2f)|Δ|²

with ``k = 2π/λ``. The **linear (prism) term** is

    φ_lin(r) = (2π / (λ f)) Δ·r .

Far-field QIUP / Configuration C (single-crystal Michelson)
-----------------------------------------------------------
The axial Haidinger term is unchanged: ``λ_eq = λ_s²/λ_i``, camera focal
length ``f_c``, and ``g_rt = 2`` multiplies only that quadratic term.

The transverse term is the far-field phase of the round-trip image
translation ``2Δ``. On camera coordinates ``(x, y)``:

    Φ_tr(x, y) = 4π (Δx x + Δy y) / (λ_s f_c)

The factor ``4π`` already contains that translation. ``g_rt`` is not applied
again, and ``λ_eq`` is not the wavelength of this linear term.

Dimensional consistency
-----------------------
Numerator ``Δx · x`` has units length². Denominator ``λ_eq · f_c`` has units
length². Ratio is dimensionless → phase argument of ``2π`` is consistent.

The production far-field formula (pre-Task-17) used denominator
``λ_eq · f_c · f_u`` (three lengths) → units of 1/length → **dimensionally
inconsistent**. That legacy form is retained here only as ``LegacyFarFieldBug``
for regression / documentation.

Detector-plane predictions
--------------------------
With ``Δ = (Δx, 0)``:

    ∂φ/∂x = 4π Δx / (λ_s f_c)                 [rad / length]
    κ_x   = |∂φ/∂x| / (2π) = 2 |Δx| / (λ_s f_c)   [cycles / length]
    Λ     = 1/κ_x = λ_s f_c / (2 |Δx|)        [length]
    k⃗    = (∂φ/∂x, ∂φ/∂y)                    [rad / length]

Fringes are oriented **perpendicular** to ``Δ`` (wave vector parallel to ``Δ``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared helpers (duplicated intentionally; no production imports)
# ---------------------------------------------------------------------------


def equivalent_wavelength_nm(lambda_s_nm: float, lambda_i_nm: float) -> float:
    """λ_eq = λ_s² / λ_i (Hochrainer / P11)."""
    ls, li = float(lambda_s_nm), float(lambda_i_nm)
    if ls <= 0.0 or li <= 0.0:
        raise ValueError("wavelengths must be > 0")
    return ls * ls / li


def config_c_pass_factor() -> float:
    """Single-crystal Michelson round-trip factor for Config C."""
    return 2.0


def _lengths_um(
    *,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
) -> tuple[float, float]:
    """Return ``(λ_eq, f_c)`` in µm. Used by the retired prism audit only."""
    lam_eq_um = equivalent_wavelength_nm(lambda_s_nm, lambda_i_nm) * 1.0e-3
    f_c_um = float(f_c_mm) * 1.0e3
    if lam_eq_um <= 0.0 or f_c_um <= 0.0:
        raise ValueError("λ_eq and f_c must be > 0")
    return lam_eq_um, f_c_um


def _signal_lengths_um(
    *,
    lambda_s_nm: float,
    f_c_mm: float,
) -> tuple[float, float]:
    """Return ``(λ_s, f_c)`` in µm for the image-shift transverse phase."""
    lam_s_um = float(lambda_s_nm) * 1.0e-3
    f_c_um = float(f_c_mm) * 1.0e3
    if lam_s_um <= 0.0 or f_c_um <= 0.0:
        raise ValueError("λ_s and f_c must be > 0")
    return lam_s_um, f_c_um


# ---------------------------------------------------------------------------
# Correct thin-lens linear phase (camera plane)
# ---------------------------------------------------------------------------


def thin_lens_linear_phase_rad(
    x_um: float,
    y_um: float,
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float = 2.0,
) -> float:
    """Φ_tr = 4π (Δx x + Δy y) / (λ_s f_c).

    ``pass_factor`` remains in the signature for existing callers. It is the
    axial round-trip factor and is not applied to this transverse term.
    ``lambda_i_nm`` is retained for the same reason; the linear term uses λ_s.
    """
    del pass_factor, lambda_i_nm
    lam_s_um, f_c_um = _signal_lengths_um(lambda_s_nm=lambda_s_nm, f_c_mm=f_c_mm)
    num = float(delta_x_um) * float(x_um) + float(delta_y_um) * float(y_um)
    return float(4.0 * math.pi * num / (lam_s_um * f_c_um))


def phase_slope_rad_per_um(
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float = 2.0,
) -> tuple[float, float]:
    """Return (∂φ/∂x, ∂φ/∂y) in rad/µm for Φ_tr = 4π (Δ·r) / (λ_s f_c)."""
    del pass_factor, lambda_i_nm
    lam_s_um, f_c_um = _signal_lengths_um(lambda_s_nm=lambda_s_nm, f_c_mm=f_c_mm)
    scale = 4.0 * math.pi / (lam_s_um * f_c_um)
    return float(scale * float(delta_x_um)), float(scale * float(delta_y_um))


def transverse_wavevector_rad_per_um(
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float = 2.0,
) -> tuple[float, float]:
    """Detector-plane wave vector k⃗ = ∇φ [rad/µm]."""
    return phase_slope_rad_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=pass_factor,
    )


def fringe_spatial_frequency_per_um(
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float = 2.0,
) -> float:
    """κ = |∇φ| / 2π  [cycles/µm]."""
    kx, ky = transverse_wavevector_rad_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=pass_factor,
    )
    return float(math.hypot(kx, ky) / (2.0 * math.pi))


def fringe_period_um(
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    pass_factor: float = 2.0,
) -> float | None:
    """Λ = 1/κ [µm], or None if Δ = 0."""
    kappa = fringe_spatial_frequency_per_um(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lambda_s_nm,
        lambda_i_nm=lambda_i_nm,
        f_c_mm=f_c_mm,
        pass_factor=pass_factor,
    )
    if kappa <= 0.0:
        return None
    return float(1.0 / kappa)


# ---------------------------------------------------------------------------
# Dimensional audit + legacy (buggy) production formula
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionalCheck:
    name: str
    formula: str
    numerator_units: str
    denominator_units: str
    phase_dimensionless: bool
    notes: str


@dataclass(frozen=True)
class LegacyFarFieldBug:
    """Document the pre-correction production far-field transverse law."""

    equation: str = (
        "φ_legacy = g_rt · 2π (Δx x + Δy y) / (λ_eq · f_c · f_u)"
    )
    problem: str = (
        "Denominator has three length factors (λ_eq, f_c, f_u) so the "
        "argument of 2π has units of 1/length — not dimensionless."
    )
    corrected_equation: str = (
        "Φ_tr = 4π (Δx x + Δy y) / (λ_s · f_c)"
    )


LEGACY_BUG = LegacyFarFieldBug()


def legacy_far_field_linear_phase_rad(
    x_um: float,
    y_um: float,
    *,
    delta_x_um: float,
    delta_y_um: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    f_c_mm: float,
    f_u_mm: float,
    pass_factor: float = 2.0,
) -> float:
    """Reproduce the dimensionally inconsistent production formula (for audit)."""
    lam_eq_um, f_c_um = _lengths_um(
        lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm, f_c_mm=f_c_mm
    )
    f_u_um = float(f_u_mm) * 1.0e3
    if f_u_um <= 0.0:
        raise ValueError("f_u must be > 0")
    num = float(delta_x_um) * float(x_um) + float(delta_y_um) * float(y_um)
    return float(
        float(pass_factor) * 2.0 * math.pi * num / (lam_eq_um * f_c_um * f_u_um)
    )


def validate_dimensional_consistency() -> list[DimensionalCheck]:
    """Static dimensional checks for correct vs legacy laws."""
    return [
        DimensionalCheck(
            name="corrected_far_field_linear",
            formula="4π (Δ·r) / (λ_s f_c)",
            numerator_units="length²",
            denominator_units="length²",
            phase_dimensionless=True,
            notes=(
                "Round-trip image shift 2Δ is inside 4π. "
                "g_rt multiplies only the axial Haidinger term."
            ),
        ),
        DimensionalCheck(
            name="legacy_production_far_field_linear",
            formula="2π (Δ·r) / (λ_eq f_c f_u)",
            numerator_units="length²",
            denominator_units="length³",
            phase_dimensionless=False,
            notes=LEGACY_BUG.problem,
        ),
        DimensionalCheck(
            name="near_field_linear_reference",
            formula="2π (Δ·r) / (λ_i f_u m_nf)",
            numerator_units="length²",
            denominator_units="length²",
            phase_dimensionless=True,
            notes="Production near-field branch is already dimensionally OK.",
        ),
        DimensionalCheck(
            name="far_field_quadratic_p11",
            formula="π d ρ² / (f_c² λ_eq)",
            numerator_units="length · length²",
            denominator_units="length² · length",
            phase_dimensionless=True,
            notes="Longitudinal law (unchanged); sets camera mapping to f_c, λ_eq.",
        ),
    ]


DIMENSIONAL_STATUS: dict[str, Any] = {
    "corrected_ok": True,
    "legacy_ok": False,
    "legacy": LEGACY_BUG,
    "checks": validate_dimensional_consistency(),
    "round_trip": {
        "config_c_g_rt": 2.0,
        "linear_phase": "4π (Δ·r) / (λ_s f_c); g_rt is not applied again",
        "parity": (
            "The symmetric focal Michelson translates the returning idler "
            "image by 2Δ. The camera phase of that shift is "
            "4π (Δ·r) / (λ_s f_c). The axial Haidinger term still uses "
            "g_rt = 2 and λ_eq."
        ),
        "sign_convention": (
            "Δx > 0 ⇒ ∂φ/∂x > 0 on the camera grid."
        ),
    },
}


def derivation_summary() -> dict[str, Any]:
    """Machine-readable derivation checklist for evidence/provenance."""
    checks = validate_dimensional_consistency()
    return {
        "thin_lens_phase": "φ_L = −(k/2f)|r−Δ|² → linear term (k/f) Δ·r",
        "config_c_equation": LEGACY_BUG.corrected_equation,
        "legacy_equation": LEGACY_BUG.equation,
        "legacy_problem": LEGACY_BUG.problem,
        "pass_factor_config_c": config_c_pass_factor(),
        "linear_round_trip": "included in 4π/(λ_s f_c); g_rt not reapplied",
        "fringe_orientation": "wave vector ∥ Δ; fringe normals ∥ Δ; fringes ⊥ Δ",
        "dimensional_checks": [
            {
                "name": c.name,
                "formula": c.formula,
                "phase_dimensionless": c.phase_dimensionless,
                "notes": c.notes,
            }
            for c in checks
        ],
    }
