"""Build per-channel simulation states (optical / detector / drift isolation)."""
from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.targets import build_target
from latents import LatentState
from setup_params import resolve_setup_selection
from verification.monte_carlo.definitions import (
    CHANNEL_DETECTOR,
    CHANNEL_DRIFT,
    CHANNEL_OPTICAL,
    CHANNEL_POLICY,
    GRID_N,
    NBAR,
    PHASE_TRUTH_RAD,
)


def make_state(
    setup: Any,
    channel: str,
    *,
    nbar: float = NBAR,
    num_frames: int = 1,
) -> LatentState:
    pol = CHANNEL_POLICY[channel]
    lat = dict(pol["latents"])
    return LatentState(
        setup=setup,
        mean_photons_per_mode=float(nbar),
        num_frames=int(num_frames),
        exposure_s=0.02,
        em_gain=100.0,
        s_pump=float(lat.get("s_pump", 0.0)),
        s_gain=0.0,
        s_align=float(lat.get("s_align", 0.0)),
        s_thermal=float(lat.get("s_thermal", 0.0)),
        s_det=float(lat.get("s_det", 0.0)),
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def make_phase_reference(
    n: int,
    pixel_um: float,
    *,
    phi_rad: float,
) -> dict[str, Any]:
    tgt = build_target("transparent", n, pixel_um=pixel_um)
    ref = tgt.to_simulation_reference()
    phi = np.full((n, n), float(phi_rad), dtype=np.float32)
    T = np.asarray(ref["T"], dtype=np.float64)
    ref["phi"] = phi
    ref["O_i"] = (np.sqrt(T) * np.exp(1j * float(phi_rad))).astype(np.complex64)
    return ref


def channel_run_config(
    profile: str,
    channel: str,
    *,
    n: int = GRID_N,
    phi_truth: float = PHASE_TRUTH_RAD,
    drift_offset_rad: float = 0.0,
) -> dict[str, Any]:
    setup = resolve_setup_selection(profile)
    # Enable four-step for phase metrics on every channel.
    setup = setup.overridden(enable_phase_step_acquisition=True)
    from setup_params import resolved_spatial_mapping

    pixel_um = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    phi = float(phi_truth) + float(drift_offset_rad)
    ref = make_phase_reference(n, pixel_um, phi_rad=phi)
    pol = CHANNEL_POLICY[channel]
    return {
        "setup": setup,
        "ref": ref,
        "detector_noise": bool(pol["detector_noise"]),
        "pixel_um": pixel_um,
        "phi_applied_rad": phi,
        "phi_truth_rad": float(phi_truth),
        "drift_offset_rad": float(drift_offset_rad),
        "channel_policy": pol,
    }
