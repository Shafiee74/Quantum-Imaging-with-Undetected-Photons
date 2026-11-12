"""Monte-Carlo / frame-count convergence runner (Task 11)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from simulate import simulate_forward
from verification.monte_carlo.channels import channel_run_config, make_state
from verification.monte_carlo.definitions import (
    CHANNEL_DRIFT,
    CHANNEL_POLICY,
    CHANNELS,
    DEFINITIONS_MANIFEST,
    FRAME_COUNTS,
    GRID_N,
    MASTER_SEED,
    NBAR,
    PHASE_TRUTH_RAD,
    PROFILES,
)
from verification.monte_carlo.seeds import (
    drift_phase_offset_rad,
    realization_seed_int,
    seed_manifest,
)
from verification.monte_carlo.statistics import (
    covariance_pair,
    normalized_correlation,
    summarize_scalar_series,
)
from verification.phase_reconstruction import (
    phase_error_metrics,
    reconstruct_phase_steps,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vis_block(result) -> dict[str, Any]:
    return dict(result.meta.get("visibility_observables") or {})


def run_one_realization(
    profile: str,
    channel: str,
    realization_index: int,
    *,
    master_seed: int,
    max_realizations: int,
    n: int = GRID_N,
    nbar: float = NBAR,
) -> dict[str, Any]:
    """One independent realization (4-step acquisition for phase metrics)."""
    drift = 0.0
    if channel == CHANNEL_DRIFT:
        rms = float(CHANNEL_POLICY[CHANNEL_DRIFT]["drift_phase_rms_rad"])
        drift = drift_phase_offset_rad(
            channel,
            realization_index,
            master_seed=master_seed,
            max_realizations=max_realizations,
            rms_rad=rms,
        )
    cfg = channel_run_config(
        profile, channel, n=n, phi_truth=PHASE_TRUTH_RAD, drift_offset_rad=drift
    )
    seed = realization_seed_int(
        channel,
        realization_index,
        master_seed=master_seed,
        max_realizations=max_realizations,
    )
    state = make_state(cfg["setup"], channel, nbar=nbar, num_frames=4)
    result = simulate_forward(
        cfg["ref"],
        state,
        seed=seed,
        detector_noise=bool(cfg["detector_noise"]),
    )

    # Prefer detector frames when detector noise on; else ideal/optical frames.
    if cfg["detector_noise"]:
        plus = np.asarray(result.frames_plus, dtype=np.float64)
        minus = np.asarray(result.frames_minus, dtype=np.float64)
        D = np.asarray(result.frames_D, dtype=np.float64)
    else:
        # Optical path still writes frames_*; with detector_noise=False they
        # equal the ideal optical intensities.
        plus = np.asarray(result.frames_plus, dtype=np.float64)
        minus = np.asarray(result.frames_minus, dtype=np.float64)
        D = np.asarray(result.frames_D, dtype=np.float64)

    mean_I = float(np.mean(plus))
    mean_Im = float(np.mean(minus))
    vo = _vis_block(result)
    V = float(vo.get("V_50_50") or 0.0)

    rec = reconstruct_phase_steps(
        {"I0": D[0], "I_pi_2": D[1], "I_pi": D[2], "I_3pi_2": D[3]}
    )
    err = phase_error_metrics(
        rec.phase_wrapped_rad,
        np.full((n, n), float(cfg["phi_truth_rad"])),
        remove_piston=True,
    )

    return {
        "profile": profile,
        "channel": channel,
        "realization_index": int(realization_index),
        "seed_int": int(seed),
        "drift_offset_rad": float(drift),
        "phi_applied_rad": float(cfg["phi_applied_rad"]),
        "mean_intensity_plus": mean_I,
        "mean_intensity_minus": mean_Im,
        "visibility_V_50_50": V,
        "phase_mean_rad": float(np.mean(rec.phase_wrapped_rad)),
        "phase_rmse_rad": float(err["rmse_rad"]),
        "phase_max_abs_rad": float(err["max_abs_rad"]),
        "C_12_abs": float(vo.get("C_12_abs") or 0.0),
        "N_1": float(vo.get("N_1") or 0.0),
        "N_2": float(vo.get("N_2") or 0.0),
    }


def aggregate_realizations(
    records: list[dict[str, Any]],
    *,
    n_use: int,
) -> dict[str, Any]:
    """Aggregate the first ``n_use`` realizations into MC summaries."""
    use = records[: int(n_use)]
    I_p = [r["mean_intensity_plus"] for r in use]
    I_m = [r["mean_intensity_minus"] for r in use]
    V = [r["visibility_V_50_50"] for r in use]
    phase_rmse = [r["phase_rmse_rad"] for r in use]
    phase_mean = [r["phase_mean_rad"] for r in use]

    return {
        "n_realizations": int(n_use),
        "mean_intensity": summarize_scalar_series("mean_intensity_plus", I_p),
        "variance_intensity": {
            "metric": "variance_mean_intensity_plus",
            "value": summarize_scalar_series("mean_intensity_plus", I_p)["variance"],
        },
        "visibility": summarize_scalar_series("visibility_V_50_50", V),
        "covariance_Iplus_Iminus": {
            "metric": "cov(mean_I_plus, mean_I_minus)",
            "value": covariance_pair(I_p, I_m),
        },
        "normalized_correlation_Iplus_Iminus": {
            "metric": "corr(mean_I_plus, mean_I_minus)",
            "value": normalized_correlation(I_p, I_m),
        },
        "reconstructed_phase_mean": summarize_scalar_series(
            "phase_mean_rad", phase_mean
        ),
        "reconstruction_error": summarize_scalar_series(
            "phase_rmse_rad", phase_rmse
        ),
        "realization_table": use,
    }


def run_channel_sweep(
    profile: str,
    channel: str,
    *,
    frame_counts: tuple[int, ...] = FRAME_COUNTS,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
    nbar: float = NBAR,
) -> dict[str, Any]:
    max_n = int(max(frame_counts))
    records = [
        run_one_realization(
            profile,
            channel,
            i,
            master_seed=master_seed,
            max_realizations=max_n,
            n=n,
            nbar=nbar,
        )
        for i in range(max_n)
    ]
    by_count = {
        int(k): aggregate_realizations(records, n_use=int(k)) for k in frame_counts
    }
    return {
        "profile": profile,
        "channel": channel,
        "max_realizations": max_n,
        "frame_counts": list(frame_counts),
        "by_frame_count": by_count,
        "all_realizations": records,
    }


def run_monte_carlo_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    channels: tuple[str, ...] = CHANNELS,
    frame_counts: tuple[int, ...] = FRAME_COUNTS,
    master_seed: int = MASTER_SEED,
    n: int = GRID_N,
    nbar: float = NBAR,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "utc": utc_now(),
        "definitions": DEFINITIONS_MANIFEST,
        "master_seed": int(master_seed),
        "seed_manifest": seed_manifest(
            master_seed=master_seed,
            frame_counts=list(frame_counts),
            channels=channels,
        ),
        "profiles": {},
    }
    run_rows = []
    for profile in profiles:
        results["profiles"][profile] = {}
        for channel in channels:
            block = run_channel_sweep(
                profile,
                channel,
                frame_counts=frame_counts,
                master_seed=master_seed,
                n=n,
                nbar=nbar,
            )
            results["profiles"][profile][channel] = block
            for fc, agg in block["by_frame_count"].items():
                run_rows.append(
                    {
                        "profile": profile,
                        "channel": channel,
                        "n_realizations": fc,
                        "mean_intensity": agg["mean_intensity"]["mean"],
                        "var_intensity": agg["mean_intensity"]["variance"],
                        "visibility_mean": agg["visibility"]["mean"],
                        "visibility_ci_low": agg["visibility"]["ci"]["ci_low"],
                        "visibility_ci_high": agg["visibility"]["ci"]["ci_high"],
                        "cov_Iplus_Iminus": agg["covariance_Iplus_Iminus"]["value"],
                        "corr_Iplus_Iminus": agg[
                            "normalized_correlation_Iplus_Iminus"
                        ]["value"],
                        "phase_rmse_mean": agg["reconstruction_error"]["mean"],
                        "phase_rmse_ci_low": agg["reconstruction_error"]["ci"][
                            "ci_low"
                        ],
                        "phase_rmse_ci_high": agg["reconstruction_error"]["ci"][
                            "ci_high"
                        ],
                    }
                )
    results["run_manifest"] = {
        "profiles": list(profiles),
        "channels": list(channels),
        "frame_counts": list(frame_counts),
        "grid_n": n,
        "nbar": nbar,
        "n_summary_rows": len(run_rows),
    }
    results["summary_rows"] = run_rows
    return results
