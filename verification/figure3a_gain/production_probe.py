"""Probe production SPDC gain resolution through ``simulate_forward``."""
from __future__ import annotations

from typing import Any

import numpy as np

from latents import LatentState
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from simulate import simulate_forward
from verification.monte_carlo.channels import make_phase_reference

# Match U02 / Figure 3(a) manuscript sweep (task X3 evidence).
R_GRID_DEFAULT = np.linspace(0.0, 0.5, 51)
DEFAULT_PROFILE = "A"
DEFAULT_GRID = 24
DEFAULT_SEED = 20260930


def _uniform_ref(setup: Any, n: int) -> dict[str, Any]:
    pixel_um = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    return make_phase_reference(n, pixel_um, phi_rad=0.0)


def production_state_for_r(setup: Any, r: float) -> LatentState:
    """Request gain via ``spdc_gain_r`` (production r → n̄ path in ``optics``)."""
    return LatentState(
        setup=setup,
        spdc_gain_r=float(r),
        mean_photons_per_mode=None,
        num_frames=1,
        exposure_s=0.02,
        em_gain=100.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def legacy_u02_state_for_r(setup: Any, r: float, nbar_from_r: float) -> LatentState:
    """Legacy U02 check: inject n̄ = sinh²(r) directly (not an independent r→n̄ test)."""
    return LatentState(
        setup=setup,
        mean_photons_per_mode=float(nbar_from_r),
        num_frames=1,
        exposure_s=0.02,
        em_gain=100.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def probe_production_nbar_nl1(
    profile: str,
    r: float,
    *,
    grid_n: int = DEFAULT_GRID,
    seed: int = DEFAULT_SEED,
    point_index: int = 0,
) -> dict[str, Any]:
    setup = resolve_setup_selection(profile)
    state = production_state_for_r(setup, r)
    ref = _uniform_ref(setup, grid_n)
    result = simulate_forward(
        ref,
        state,
        seed=int(seed) + int(point_index),
        detector_noise=False,
    )
    op = dict(result.meta.get("spdc_operating_point") or {})
    n_prod = result.meta.get("nbar_NL1")
    if n_prod is None:
        n_prod = op.get("mean_photons_per_mode_NL1")
    n2 = result.meta.get("nbar_NL2")
    if n2 is None:
        n2 = op.get("mean_photons_per_mode_NL2")
    return {
        "profile": profile,
        "r": float(r),
        "nbar_production_NL1": float(n_prod),
        "nbar_production_NL2": float(n2) if n2 is not None else float("nan"),
        "spdc_gain_r_request": float(r),
        "spdc_source_mode": getattr(setup, "spdc_source_mode", None),
        "source_NL1": op.get("source_NL1") or op.get("source"),
        "gain_r_meta_1": float(
            (result.meta.get("visibility_observables") or {}).get("gain_r_1") or r
        ),
    }


def sweep_production(
    r_values: np.ndarray,
    *,
    profile: str = DEFAULT_PROFILE,
) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(np.asarray(r_values, dtype=np.float64)):
        rows.append(probe_production_nbar_nl1(profile, float(r), point_index=i))
    return rows
