"""Systematic noise/distortion ablation runner (Stages A–D)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from latents import LatentState
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from simulate import simulate_forward
from verification.ablation.config import clean_config, merge_mechanisms
from verification.ablation.definitions import (
    COMBINATIONS,
    FULL_MODEL_MECHANISMS,
    GRID_N,
    MASTER_SEED,
    MECHANISMS,
    NBAR,
    PHI_TRUTH_RAD,
    PROFILES,
)
from verification.ablation.metrics import degradation_row, extract_case_metrics
from verification.monte_carlo.channels import make_phase_reference


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_seed(master_seed: int, case_id: str) -> int:
    """Deterministic per-case seed from master + case name (stable hash)."""
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    case_int = int.from_bytes(digest[:4], "little")
    ss = np.random.SeedSequence([int(master_seed), case_int])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def build_setup(profile: str, merged: dict[str, Any]) -> Any:
    setup = resolve_setup_selection(profile)
    overrides = dict(merged["setup"])
    return setup.overridden(**overrides)


def build_state(setup: Any, merged: dict[str, Any], *, nbar: float) -> LatentState:
    lat = merged["latents"]
    return LatentState(
        setup=setup,
        mean_photons_per_mode=float(nbar),
        num_frames=4,
        exposure_s=0.02,
        em_gain=float(merged["em_gain"]),
        s_pump=float(lat.get("s_pump", 0.0)),
        s_gain=0.0,
        s_align=float(lat.get("s_align", 0.0)),
        s_thermal=float(lat.get("s_thermal", 0.0)),
        s_det=float(lat.get("s_det", 0.0)),
        s_cool=float(lat.get("s_cool", 0.0)),
        s_focus=float(lat.get("s_focus", 0.0)),
        s_slm=0.0,
    )


def build_reference(
    setup: Any,
    *,
    n: int,
    phi_truth_rad: float,
    phase_offset_rad: float,
) -> dict[str, Any]:
    pixel_um = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    phi = float(phi_truth_rad) + float(phase_offset_rad)
    return make_phase_reference(n, pixel_um, phi_rad=phi)


def run_case(
    profile: str,
    case_id: str,
    stage: str,
    mechanism_names: tuple[str, ...] | list[str],
    *,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
    nbar: float = NBAR,
    phi_truth_rad: float = PHI_TRUTH_RAD,
) -> dict[str, Any]:
    """Run one ablation case through production ``simulate_forward``."""
    merged = merge_mechanisms(mechanism_names) if mechanism_names else clean_config()
    setup = build_setup(profile, merged)
    state = build_state(setup, merged, nbar=nbar)
    ref = build_reference(
        setup,
        n=n,
        phi_truth_rad=phi_truth_rad,
        phase_offset_rad=float(merged["object_phase_offset_rad"]),
    )
    seed = case_seed(master_seed, f"{profile}:{stage}:{case_id}")
    result = simulate_forward(
        ref,
        state,
        seed=seed,
        detector_noise=bool(merged["detector_noise"]),
    )
    metrics = extract_case_metrics(
        result,
        n=n,
        phi_truth_rad=phi_truth_rad,
        detector_noise=bool(merged["detector_noise"]),
    )
    return {
        "profile": profile,
        "case_id": case_id,
        "stage": stage,
        "mechanism_names": list(mechanism_names),
        "sources": list(merged["sources"]),
        "seed_int": int(seed),
        "nbar": float(nbar),
        "phi_truth_rad": float(phi_truth_rad),
        "object_phase_offset_rad": float(merged["object_phase_offset_rad"]),
        "em_gain": float(merged["em_gain"]),
        "latents": dict(merged["latents"]),
        "setup_overrides": {
            k: merged["setup"][k]
            for k in (
                "dark0",
                "cic0",
                "read_noise0_e",
                "prnu_sigma",
                "hot_frac_at_s1",
                "read_noise_model",
            )
            if k in merged["setup"]
        },
        "metrics": metrics,
        "result_meta_keys": sorted(result.meta.keys()) if result.meta else [],
    }


def run_ablation_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
    nbar: float = NBAR,
    phi_truth_rad: float = PHI_TRUTH_RAD,
) -> dict[str, Any]:
    """Stages A (clean), B (individual), C (combinations), D (full model)."""
    suite: dict[str, Any] = {
        "generated_utc": utc_now(),
        "master_seed": int(master_seed),
        "grid_n": int(n),
        "nbar": float(nbar),
        "phi_truth_rad": float(phi_truth_rad),
        "profiles": {},
        "ablation_rows": [],
        "degradation_rows": [],
    }

    for profile in profiles:
        # Stage A
        clean = run_case(
            profile,
            "clean",
            "A",
            (),
            master_seed=master_seed,
            n=n,
            nbar=nbar,
            phi_truth_rad=phi_truth_rad,
        )
        clean_m = clean["metrics"]

        # Stage B
        individual: dict[str, Any] = {}
        for name, spec in MECHANISMS.items():
            if not spec.get("stage_b", True):
                continue
            individual[name] = run_case(
                profile,
                name,
                "B",
                (name,),
                master_seed=master_seed,
                n=n,
                nbar=nbar,
                phi_truth_rad=phi_truth_rad,
            )

        # Stage C
        combinations: dict[str, Any] = {}
        for comb_id, names in COMBINATIONS.items():
            combinations[comb_id] = run_case(
                profile,
                comb_id,
                "C",
                names,
                master_seed=master_seed,
                n=n,
                nbar=nbar,
                phi_truth_rad=phi_truth_rad,
            )

        # Stage D
        full = run_case(
            profile,
            "full_model",
            "D",
            FULL_MODEL_MECHANISMS,
            master_seed=master_seed,
            n=n,
            nbar=nbar,
            phi_truth_rad=phi_truth_rad,
        )

        suite["profiles"][profile] = {
            "clean": clean,
            "individual": individual,
            "combinations": combinations,
            "full_model": full,
        }

        all_cases = (
            [clean]
            + list(individual.values())
            + list(combinations.values())
            + [full]
        )
        for case in all_cases:
            m = case["metrics"]
            suite["ablation_rows"].append(
                {
                    "profile": profile,
                    "stage": case["stage"],
                    "case_id": case["case_id"],
                    "mechanisms": ";".join(case["mechanism_names"]),
                    "detector_noise": bool(m["detector_noise"]),
                    "intensity_mean": float(m["intensity_mean"]),
                    "intensity_std": float(m["intensity_std"]),
                    "visibility_frame_michelson": float(
                        m["visibility_frame_michelson"]
                    ),
                    "visibility_V_50_50": float(m["visibility_V_50_50"]),
                    "phase_mean_rad": float(m["phase_mean_rad"]),
                    "phase_rmse_rad": float(m["phase_rmse_rad"]),
                    "phase_rmse_no_piston_rad": float(m["phase_rmse_no_piston_rad"]),
                    "phase_mae_rad": float(m["phase_mae_rad"]),
                    "snr_spatial": float(m["snr_spatial"]),
                    "covariance_plus_minus": float(m["covariance_plus_minus"]),
                    "correlation_plus_minus": float(m["correlation_plus_minus"]),
                    "seed_int": int(case["seed_int"]),
                    "em_gain": float(case["em_gain"]),
                }
            )
            deg = degradation_row(case["case_id"], case["stage"], m, clean_m)
            deg["profile"] = profile
            suite["degradation_rows"].append(deg)

    return suite
