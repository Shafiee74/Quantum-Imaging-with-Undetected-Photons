"""Compare independent transverse law vs production optics (Task 17).

This module is allowed to import production for audit only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from optics import idler_lens_misalignment_phase_map
from setup_params import (
    resolve_setup_selection,
    resolved_far_field_focal_mm,
)
from verification.config_c.transverse_lens.derivation import (
    LEGACY_BUG,
    legacy_far_field_linear_phase_rad,
)
from verification.config_c.transverse_lens.reference import (
    estimate_phase_slope_from_map,
    predicted_observables,
    sample_point_phase,
    transverse_phase_map,
)


def _quiet_c(**extra):
    values = dict(
        enable_lens_rings=True,
        delta_z_um=0.0,
        lens_shift_x_um=0.0,
        lens_shift_y_um=0.0,
        use_derived_K_defocus=False,
        K_defocus_px_per_um=0.0,
        use_derived_diffraction=False,
        sigma_diffraction_px=0.0,
        wavefront_rms_waves=0.0,
        sigma_blur_extra_px=0.0,
    )
    values.update(extra)
    return values


def production_phase_map(
    n: int,
    *,
    delta_x_um: float,
    delta_y_um: float = 0.0,
    delta_z_um: float = 0.0,
    profile: str = "C",
) -> dict[str, Any]:
    setup = resolve_setup_selection(profile).overridden(
        **_quiet_c(
            lens_shift_x_um=float(delta_x_um),
            lens_shift_y_um=float(delta_y_um),
            delta_z_um=float(delta_z_um),
        )
    )
    state = idler_lens_misalignment_phase_map(n, setup, delta_z_um=float(delta_z_um))
    return {
        "phase_rad": np.asarray(state["phase_rad"], dtype=np.float64),
        "pass_factor": float(state["pass_factor"]),
        "phase_law": state["phase_law"],
        "setup": setup,
        "meta": state,
    }


def compare_displacement_case(
    *,
    delta_x_um: float,
    delta_y_um: float = 0.0,
    n: int = 129,
    profile: str = "C",
) -> dict[str, Any]:
    """Independent vs production for one (Δx, Δy); also vs legacy formula."""
    prod = production_phase_map(
        n, delta_x_um=delta_x_um, delta_y_um=delta_y_um, profile=profile
    )
    setup = prod["setup"]
    pixel = float(setup.pixel_um)
    lam_s = float(setup.lambda_s_nm)
    lam_i = float(setup.lambda_i_nm)
    f_c = float(setup.F2_mm)
    f_u = float(resolved_far_field_focal_mm(setup))
    g = float(prod["pass_factor"])

    ref = transverse_phase_map(
        n,
        pixel_um=pixel,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lam_s,
        lambda_i_nm=lam_i,
        f_c_mm=f_c,
        pass_factor=g,
    )
    pred = predicted_observables(
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lam_s,
        lambda_i_nm=lam_i,
        f_c_mm=f_c,
        pass_factor=g,
    )

    phi_p = prod["phase_rad"]
    phi_r = ref["phase_rad"]
    abs_err = float(np.max(np.abs(phi_p - phi_r)))
    rms_err = float(np.sqrt(np.mean((phi_p - phi_r) ** 2)))

    sx_p, sy_p = estimate_phase_slope_from_map(phi_p, pixel_um=pixel)
    sx_r, sy_r = estimate_phase_slope_from_map(phi_r, pixel_um=pixel)

    # Spot-check centre-right pixel vs legacy
    row, col = n // 2, min(n - 1, n // 2 + 20)
    x_um = (col - (n - 1) * 0.5) * pixel
    y_um = (row - (n - 1) * 0.5) * pixel
    phi_ref_pt = sample_point_phase(
        x_um,
        y_um,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lam_s,
        lambda_i_nm=lam_i,
        f_c_mm=f_c,
        pass_factor=g,
    )
    phi_legacy_pt = legacy_far_field_linear_phase_rad(
        x_um,
        y_um,
        delta_x_um=delta_x_um,
        delta_y_um=delta_y_um,
        lambda_s_nm=lam_s,
        lambda_i_nm=lam_i,
        f_c_mm=f_c,
        f_u_mm=f_u,
        pass_factor=g,
    )
    phi_prod_pt = float(phi_p[row, col])

    matches_corrected = abs_err < 1.0e-9
    matches_legacy = abs(phi_prod_pt - phi_legacy_pt) < 1.0e-9

    return {
        "delta_x_um": float(delta_x_um),
        "delta_y_um": float(delta_y_um),
        "n": int(n),
        "pixel_um": pixel,
        "f_c_mm": f_c,
        "f_u_mm": f_u,
        "pass_factor": g,
        "max_abs_err_vs_corrected": abs_err,
        "rms_err_vs_corrected": rms_err,
        "matches_corrected_law": matches_corrected,
        "matches_legacy_buggy_law": matches_legacy,
        "production_slope_x": sx_p,
        "production_slope_y": sy_p,
        "reference_slope_x": sx_r,
        "reference_slope_y": sy_r,
        "predicted": pred,
        "spot_check": {
            "row": row,
            "col": col,
            "x_um": x_um,
            "y_um": y_um,
            "phi_production": phi_prod_pt,
            "phi_corrected": phi_ref_pt,
            "phi_legacy": phi_legacy_pt,
        },
        "legacy_equation": LEGACY_BUG.equation,
        "corrected_equation": LEGACY_BUG.corrected_equation,
        "phase_production": phi_p,
        "phase_reference": phi_r,
    }


def baseline_zero_shift_unchanged(
    *,
    n: int = 65,
    delta_z_um: float = 40.0,
    profiles: tuple[str, ...] = ("A", "C"),
) -> list[dict[str, Any]]:
    """Longitudinal-only maps must be unaffected by the transverse fix."""
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        setup = resolve_setup_selection(profile).overridden(
            **_quiet_c(delta_z_um=delta_z_um, lens_shift_x_um=0.0, lens_shift_y_um=0.0)
        )
        state = idler_lens_misalignment_phase_map(n, setup, delta_z_um=delta_z_um)
        phi = np.asarray(state["phase_rad"], dtype=np.float64)
        # With Δx=Δy=0, transverse contribution is identically zero; centre = 0
        rows.append(
            {
                "profile": profile,
                "delta_z_um": float(delta_z_um),
                "pass_factor": float(state["pass_factor"]),
                "phase_law": state["phase_law"],
                "center_phase": float(phi[n // 2, n // 2]),
                "max_abs_phase": float(np.max(np.abs(phi))),
                "finite": bool(np.all(np.isfinite(phi))),
            }
        )
    return rows
