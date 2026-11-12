"""Frozen production simulation helpers for Task 19 C2 (no parameter retuning)."""
from __future__ import annotations

from typing import Any

import numpy as np

from optics import idler_lens_misalignment_phase_map
from setup_params import resolve_setup_selection
from validation.config_c.c2_validation.longitudinal import (
    analyze_longitudinal_mean,
    theory_a,
)
from validation.config_c.c2_validation.transverse import (
    analyze_transverse_mean,
    theory_transverse,
)
from verification.config_c.transverse_lens.reference import transverse_phase_map
from verification.physics.independent_equations import (
    equivalent_wavelength_nm,
    idler_lens_equivalent_propagation_um,
)


def _quiet_overrides(**extra):
    values = dict(
        enable_lens_rings=True,
        use_derived_K_defocus=False,
        K_defocus_px_per_um=0.0,
        use_derived_diffraction=False,
        sigma_diffraction_px=0.0,
        wavefront_rms_waves=0.0,
        sigma_blur_extra_px=0.0,
        # Operational range only — does not change the phase law.
        delta_z_max_um=20000.0,
        lens_shift_soft_um=500.0,
    )
    values.update(extra)
    return values


def simulate_longitudinal_intensity(
    *,
    delta_z_um: float,
    n: int = 257,
    pixel_um: float = 16.0,
    visibility: float = 0.4,
    dc: float = 1.0,
) -> dict[str, Any]:
    """Production phase map → synthetic camera intensity (noise-free)."""
    setup = resolve_setup_selection("C").overridden(
        **_quiet_overrides(
            delta_z_um=float(delta_z_um),
            lens_shift_x_um=0.0,
            lens_shift_y_um=0.0,
            pixel_um=float(pixel_um),
        )
    )
    state = idler_lens_misalignment_phase_map(n, setup, delta_z_um=float(delta_z_um))
    phi = np.asarray(state["phase_rad"], dtype=np.float64)
    inten = dc * (1.0 + visibility * np.cos(phi))
    # Independent theory phase on same grid for residual
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    x_um = (xx - (n - 1) * 0.5) * pixel_um
    y_um = (yy - (n - 1) * 0.5) * pixel_um
    rho = np.hypot(x_um, y_um)
    from setup_params import resolved_far_field_focal_mm

    fu = float(resolved_far_field_focal_mm(setup))
    # Independent theory (vectorized): φ = g_rt · π · d · ρ² / (f_c² λ_eq), g_rt=2
    d_um = idler_lens_equivalent_propagation_um(float(delta_z_um), fu)
    lam_eq_um = equivalent_wavelength_nm(
        float(setup.lambda_s_nm), float(setup.lambda_i_nm)
    ) * 1.0e-3
    f_c_um = float(setup.F2_mm) * 1.0e3
    phi_th = (
        2.0
        * np.pi
        * d_um
        * (rho * rho)
        / (f_c_um * f_c_um * lam_eq_um)
    )  # = 2 * π * d * ρ² / (f_c² λ_eq)
    metrics = analyze_longitudinal_mean(
        inten, delta_z_um=float(delta_z_um), pixel_um_assumed=pixel_um, f_u_mm=fu
    )
    th = theory_a(float(delta_z_um), f_u_mm=fu, f_c_mm=float(setup.F2_mm))
    # Fit a from production phase: φ = 2π a ρ² → a = median(φ / (2π ρ²))
    rho2 = rho * rho
    mask = rho2 > (5.0 * pixel_um) ** 2
    a_sim = float(np.median(phi[mask] / (2.0 * np.pi * rho2[mask]))) if np.any(mask) else float("nan")
    return {
        "intensity": inten,
        "phase_production": phi,
        "phase_theory": phi_th,
        "metrics": metrics,
        "a_sim_from_phase_per_um2": a_sim,
        "a_theory_per_um2": th["a_theory_per_um2"],
        "phase_max_abs_err_vs_theory": float(np.max(np.abs(phi - phi_th))),
        "pass_factor": float(state["pass_factor"]),
        "f_u_mm": fu,
        "n": n,
        "pixel_um": pixel_um,
    }


def simulate_transverse_intensity(
    *,
    delta_x_um: float,
    n: int = 257,
    pixel_um: float = 16.0,
    visibility: float = 0.4,
    dc: float = 1.0,
) -> dict[str, Any]:
    setup = resolve_setup_selection("C").overridden(
        **_quiet_overrides(
            delta_z_um=0.0,
            lens_shift_x_um=float(delta_x_um),
            lens_shift_y_um=0.0,
            pixel_um=float(pixel_um),
        )
    )
    state = idler_lens_misalignment_phase_map(n, setup, delta_z_um=0.0)
    phi = np.asarray(state["phase_rad"], dtype=np.float64)
    inten = dc * (1.0 + visibility * np.cos(phi))
    ref = transverse_phase_map(
        n,
        pixel_um=pixel_um,
        delta_x_um=float(delta_x_um),
        delta_y_um=0.0,
        lambda_s_nm=float(setup.lambda_s_nm),
        lambda_i_nm=float(setup.lambda_i_nm),
        f_c_mm=float(setup.F2_mm),
        pass_factor=2.0,
    )
    metrics = analyze_transverse_mean(
        inten, delta_x_um=float(delta_x_um), pixel_um_assumed=pixel_um
    )
    th = theory_transverse(float(delta_x_um), f_c_mm=float(setup.F2_mm))
    return {
        "intensity": inten,
        "phase_production": phi,
        "phase_theory": ref["phase_rad"],
        "metrics": metrics,
        "kappa_theory": th["kappa_cycles_per_um"],
        "phase_max_abs_err_vs_theory": float(
            np.max(np.abs(phi - ref["phase_rad"]))
        ),
        "n": n,
        "pixel_um": pixel_um,
    }
