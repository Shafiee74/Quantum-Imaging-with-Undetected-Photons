"""U14 grid-convergence suite runner."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from verification.convergence.definitions import (
    DEFINITIONS_MANIFEST,
    NBAR,
    PROFILES,
    REFERENCE_POLICY,
    SEED,
)
from verification.convergence.grids import build_grid_levels
from verification.convergence.metrics import evaluate_level, try_resample


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel_err(val: float | None, ref: float | None, *, eps: float = 1e-15) -> float | None:
    if val is None or ref is None:
        return None
    return float(abs(float(val) - float(ref)) / max(abs(float(ref)), eps))


def _abs_err(val: float | None, ref: float | None) -> float | None:
    if val is None or ref is None:
        return None
    return float(abs(float(val) - float(ref)))


def compare_to_references(
    level_result: dict[str, Any],
    *,
    numerical_ref: dict[str, Any] | None,
    analytical_sigma: float,
    analytical_M: float,
) -> dict[str, Any]:
    """Attach analytical and numerical-reference errors with explicit labels."""
    out: dict[str, Any] = {
        "analytical_reference": {},
        "numerical_reference": {},
    }

    # Analytical: spatial mapping
    out["analytical_reference"]["sigma_object_um"] = {
        "value": level_result["spatial_mapping"]["sigma_object_um"],
        "reference": analytical_sigma,
        "abs_error": _abs_err(
            level_result["spatial_mapping"]["sigma_object_um"], analytical_sigma
        ),
        "rel_error": _rel_err(
            level_result["spatial_mapping"]["sigma_object_um"], analytical_sigma
        ),
        "policy": REFERENCE_POLICY["spatial_mapping_sigma_object_um"],
    }
    out["analytical_reference"]["signed_magnification"] = {
        "value": level_result["spatial_mapping"]["signed_magnification"],
        "reference": analytical_M,
        "abs_error": _abs_err(
            level_result["spatial_mapping"]["signed_magnification"], analytical_M
        ),
        "rel_error": _rel_err(
            level_result["spatial_mapping"]["signed_magnification"], analytical_M
        ),
        "policy": REFERENCE_POLICY["spatial_mapping_signed_magnification"],
    }
    out["analytical_reference"]["phase_reconstruction_rmse_rad"] = {
        "value": level_result["phase_rmse_rad"],
        "reference": 0.0,
        "abs_error": float(level_result["phase_rmse_rad"]),
        "policy": REFERENCE_POLICY["phase_reconstruction_rmse_rad"],
    }
    out["analytical_reference"]["visibility_V_50_50"] = {
        "value": level_result["V_50_50"],
        "policy": REFERENCE_POLICY["visibility_V_50_50"],
        "note": "recorded per level; analytical identity holds inside each run",
    }

    if numerical_ref is None:
        out["numerical_reference"]["status"] = "this_level_is_numerical_reference"
        return out

    # Resample intensity / visibility to numerical-ref grid for L2 shape error
    I_r = try_resample(
        level_result["intensity_peak_normalized"],
        level_result["x_um"],
        level_result["y_um"],
        numerical_ref["x_um"],
        numerical_ref["y_um"],
    )
    I_ref = numerical_ref["intensity_peak_normalized"]
    V_r = try_resample(
        level_result["visibility_map"],
        level_result["x_um"],
        level_result["y_um"],
        numerical_ref["x_um"],
        numerical_ref["y_um"],
    )
    V_ref = numerical_ref["visibility_map"]
    inten_l2 = float(np.sqrt(np.mean((I_r - I_ref) ** 2)))
    vis_l2 = float(np.sqrt(np.mean((V_r - V_ref) ** 2)))

    out["numerical_reference"]["intensity_shape_l2"] = {
        "value": inten_l2,
        "reference_level": numerical_ref["level"]["level_id"],
        "policy": REFERENCE_POLICY["intensity_normalized_map"],
        "difference_map": I_r - I_ref,
    }
    out["numerical_reference"]["visibility_map_l2"] = {
        "value": vis_l2,
        "reference_level": numerical_ref["level"]["level_id"],
        "policy": REFERENCE_POLICY["visibility_map"],
        "difference_map": V_r - V_ref,
    }
    out["numerical_reference"]["psf_fwhm_um"] = {
        "value": level_result["psf_fwhm_um"],
        "reference": numerical_ref["psf_fwhm_um"],
        "abs_error": _abs_err(level_result["psf_fwhm_um"], numerical_ref["psf_fwhm_um"]),
        "rel_error": _rel_err(level_result["psf_fwhm_um"], numerical_ref["psf_fwhm_um"]),
        "policy": REFERENCE_POLICY["psf_fwhm_um"],
    }
    out["numerical_reference"]["esf_10_90_um"] = {
        "value": level_result["esf_10_90_um"],
        "reference": numerical_ref["esf_10_90_um"],
        "abs_error": _abs_err(level_result["esf_10_90_um"], numerical_ref["esf_10_90_um"]),
        "rel_error": _rel_err(level_result["esf_10_90_um"], numerical_ref["esf_10_90_um"]),
        "policy": REFERENCE_POLICY["esf_10_90_um"],
    }
    return out


def run_profile_convergence(
    profile: str,
    *,
    seed: int = SEED,
    nbar: float = NBAR,
) -> dict[str, Any]:
    levels = build_grid_levels(profile, include_additional=True)
    raw: dict[str, dict[str, Any]] = {}
    for lev in levels:
        raw[lev.level_id] = evaluate_level(lev, seed=seed, nbar=nbar)

    # Numerical reference = primary finest (grid_level_3)
    ref_id = next(
        lev.level_id for lev in levels if lev.is_numerical_reference
    )
    numerical_ref = raw[ref_id]
    analytical_sigma = numerical_ref["spatial_mapping"]["sigma_object_um"]
    analytical_M = numerical_ref["spatial_mapping"]["signed_magnification"]

    compared: dict[str, Any] = {}
    for lev in levels:
        res = raw[lev.level_id]
        if lev.level_id == ref_id:
            cmp = compare_to_references(
                res,
                numerical_ref=None,
                analytical_sigma=analytical_sigma,
                analytical_M=analytical_M,
            )
        else:
            cmp = compare_to_references(
                res,
                numerical_ref=numerical_ref,
                analytical_sigma=analytical_sigma,
                analytical_M=analytical_M,
            )
        compared[lev.level_id] = {
            "metrics": {
                k: res[k]
                for k in res
                if k
                not in (
                    "intensity",
                    "intensity_peak_normalized",
                    "visibility_map",
                    "x_um",
                    "y_um",
                    "phase_reconstructed",
                    "phase_error_map",
                    "psf_profile",
                    "esf_profile",
                )
            },
            "arrays": {
                "intensity": res["intensity"],
                "intensity_peak_normalized": res["intensity_peak_normalized"],
                "visibility_map": res["visibility_map"],
                "x_um": res["x_um"],
                "y_um": res["y_um"],
                "phase_reconstructed": res["phase_reconstructed"],
                "phase_error_map": res["phase_error_map"],
                "psf_profile": res["psf_profile"],
                "esf_profile": res["esf_profile"],
            },
            "comparison": cmp,
            "level": res["level"],
        }

    return {
        "profile": profile,
        "numerical_reference_level_id": ref_id,
        "levels": compared,
        "definitions": DEFINITIONS_MANIFEST,
    }


def run_convergence_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    seed: int = SEED,
    nbar: float = NBAR,
) -> dict[str, Any]:
    return {
        "utc": utc_now(),
        "definitions": DEFINITIONS_MANIFEST,
        "seed": seed,
        "nbar": nbar,
        "profiles": {
            p: run_profile_convergence(p, seed=seed, nbar=nbar) for p in profiles
        },
    }
