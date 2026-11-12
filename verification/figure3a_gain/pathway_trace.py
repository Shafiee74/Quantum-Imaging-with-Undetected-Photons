"""Document production gain-resolution pathway for Figure 3(a) verification."""
from __future__ import annotations

from typing import Any

from latents import LatentState
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from simulate import simulate_forward
from verification.figure3a_gain.production_probe import (
    DEFAULT_GRID,
    DEFAULT_SEED,
    production_state_for_r,
)
from verification.monte_carlo.channels import make_phase_reference

MANUSCRIPT_PROFILES: tuple[str, ...] = ("A", "B", "C", "D")
OPTIONAL_PROFILES: tuple[str, ...] = ("C_lab",)


def _uniform_ref(setup: Any, n: int) -> dict[str, Any]:
    pixel_um = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    return make_phase_reference(n, pixel_um, phi_rad=0.0)


def trace_gain_pathway(profile: str, r: float, *, point_index: int = 0) -> dict[str, Any]:
    """Run one forward call and return operating-point fields proving r → n̄ resolution."""
    setup = resolve_setup_selection(profile)
    state = production_state_for_r(setup, r)
    assert state.mean_photons_per_mode is None
    assert state.spdc_gain_r == float(r)
    ref = _uniform_ref(setup, DEFAULT_GRID)
    result = simulate_forward(
        ref, state, seed=DEFAULT_SEED + point_index, detector_noise=False
    )
    op = dict(result.meta.get("spdc_operating_point") or {})
    return {
        "profile": profile,
        "r_request": float(r),
        "latent_mean_photons_per_mode": state.mean_photons_per_mode,
        "latent_spdc_gain_r": state.spdc_gain_r,
        "spdc_source_mode": op.get("spdc_source_mode") or getattr(setup, "spdc_source_mode"),
        "source_NL1": op.get("source_NL1") or op.get("source"),
        "source_NL2": op.get("source_NL2"),
        "nbar_NL1": float(result.meta.get("nbar_NL1") or op.get("mean_photons_per_mode_NL1")),
        "nbar_NL2": float(result.meta.get("nbar_NL2") or op.get("mean_photons_per_mode_NL2")),
        "spdc_gain_r_NL1": op.get("spdc_gain_r_NL1"),
        "spdc_gain_r_NL2": op.get("spdc_gain_r_NL2"),
        "predictive_NL1": op.get("predictive") if "predictive" in op else None,
    }


def trace_all_profiles(r_values: tuple[float, ...] = (0.05, 0.2)) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for profile in (*MANUSCRIPT_PROFILES, *OPTIONAL_PROFILES):
        try:
            resolve_setup_selection(profile)
        except Exception:
            continue
        for r in r_values:
            rows.append(trace_gain_pathway(profile, r, point_index=idx))
            idx += 1
    return rows
