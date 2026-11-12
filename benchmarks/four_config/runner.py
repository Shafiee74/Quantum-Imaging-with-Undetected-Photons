"""Matched A/B/C/D four-configuration benchmark runner (Task 15)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from benchmarks.four_config.definitions import (
    ARCHITECTURE_LABELS,
    BENCHMARK_TARGETS,
    GRID_N,
    MASTER_SEED,
    NBAR,
    NUM_FRAMES_OPTICAL,
    NUM_FRAMES_PHASE,
    PHASE_TARGETS,
    PROFILES,
    RESOLUTION_TARGETS,
)
from benchmarks.targets import PHYSICAL_TRUTH, build_target
from benchmarks.targets.geometry import centered_axis_um
from latents import LatentState
from setup_params import (
    camera_plane_geometry_from_setup,
    resolve_architecture_profile,
    resolve_setup_selection,
    resolved_spatial_mapping,
)
from simulate import simulate_forward
from verification.phase_reconstruction import (
    phase_error_metrics,
    reconstruct_phase_steps,
)
from verification.resolution.profiles import (
    average_rows,
    build_esf_profile,
    build_lsf_from_esf,
    build_psf_profile,
    extract_central_row,
)
from verification.resolution.quantities import (
    intensity_map_from_result,
    visibility_map_from_result,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quiet_state(
    setup: Any,
    *,
    nbar: float = NBAR,
    num_frames: int = NUM_FRAMES_OPTICAL,
    enable_phase_steps: bool = False,
) -> LatentState:
    st_setup = setup
    if enable_phase_steps:
        st_setup = setup.overridden(enable_phase_step_acquisition=True)
    return LatentState(
        setup=st_setup,
        mean_photons_per_mode=float(nbar),
        num_frames=int(num_frames),
        exposure_s=0.02,
        em_gain=1.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )


def mapping_block(setup: Any, *, n: int) -> dict[str, Any]:
    sm = resolved_spatial_mapping(setup)
    cam = camera_plane_geometry_from_setup(setup, n_pixels=n)
    ap = resolve_architecture_profile(setup)
    obj_px = float(sm["object_plane_pixel_um"])
    cam_px = float(sm["camera_pixel_um"])
    return {
        "architecture_profile_id": ap.profile_id,
        "architecture_label": ARCHITECTURE_LABELS[ap.profile_id],
        "spdc_architecture": ap.spdc_architecture,
        "imaging_mode": ap.imaging_mode,
        "spatial_transfer_path": ap.spatial_transfer_path,
        "n_physical_crystals": ap.n_physical_crystals,
        "n_spdc_interactions": ap.n_spdc_interactions,
        "object_plane_pixel_um": obj_px,
        "camera_pixel_um": cam_px,
        "signed_object_to_camera_magnification": float(
            sm["signed_object_to_camera_magnification"]
        ),
        "magnification_magnitude": float(sm["magnification_magnitude"]),
        "correlation_sigma_object_um": float(sm["correlation_sigma_object_um"]),
        "correlation_sigma_camera_um": float(sm["correlation_sigma_camera_um"]),
        "correlation_type": sm.get("correlation_type"),
        "object_array_coordinates": sm.get("object_array_coordinates"),
        "camera_array_coordinates": sm.get("camera_array_coordinates"),
        "M_QIUP": float(cam["M_QIUP"]),
        "camera_plane_pixel_pitch_um": float(cam["pixel_pitch_um"]),
        "object_x_um": centered_axis_um(n, obj_px),
        "object_y_um": centered_axis_um(n, obj_px),
        "detector_x_um": centered_axis_um(n, cam_px),
        "detector_y_um": centered_axis_um(n, cam_px),
        "detector_x_px": centered_axis_um(n, 1.0),
        "detector_y_px": centered_axis_um(n, 1.0),
    }


def central_line_profile(
    image: np.ndarray, x_um: np.ndarray, y_um: np.ndarray
) -> dict[str, np.ndarray]:
    row = int(np.argmin(np.abs(y_um)))
    return {
        "axis_um": np.asarray(x_um, dtype=np.float64),
        "values": extract_central_row(image, row=row),
        "row_index": np.asarray([row]),
    }


def resolution_metrics_for_target(
    target_id: str,
    intensity: np.ndarray,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    tgt_meta: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {"target_id": target_id}
    if target_id == "point":
        py = float(tgt_meta.get("point_y_um", 0.0))
        px = float(tgt_meta.get("point_x_um", 0.0))
        row = int(np.argmin(np.abs(y_um - py)))
        psf = build_psf_profile(
            x_um, extract_central_row(intensity, row=row), quantity="intensity"
        )
        out.update(
            {
                "psf_fwhm_um": psf.metric_value,
                "centroid_um": psf.extras.get("centroid_um"),
                "expected_centroid_um": px,
                "line_profile": {
                    "axis_um": psf.axis_um,
                    "original": psf.original_profile,
                    "processed": psf.processed_profile,
                },
            }
        )
    elif target_id == "knife_edge_vertical":
        edge = float(tgt_meta.get("edge_x_um", 0.0))
        esf = build_esf_profile(
            x_um,
            average_rows(intensity),
            quantity="intensity",
            expected_edge_um=edge,
        )
        lsf = build_lsf_from_esf(esf)
        out.update(
            {
                "esf_10_90_um": esf.metric_value,
                "lsf_fwhm_um": lsf.metric_value,
                "esf_x50_um": esf.extras.get("x50_um"),
                "spatial_mapping_error_um": esf.extras.get("spatial_mapping_error_um"),
                "line_profile_esf": {
                    "axis_um": esf.axis_um,
                    "original": esf.original_profile,
                    "processed": esf.processed_profile,
                },
                "line_profile_lsf": {
                    "axis_um": lsf.axis_um,
                    "original": lsf.original_profile,
                    "processed": lsf.processed_profile,
                },
            }
        )
    return out


def run_target_case(
    profile: str,
    target_id: str,
    *,
    n: int = GRID_N,
    nbar: float = NBAR,
    master_seed: int = MASTER_SEED,
    setup_key: str | None = None,
) -> dict[str, Any]:
    """Run one matched target.

    ``profile`` is the publication label (A/B/C/D). ``setup_key`` selects the
    YAML/setup alias (defaults to ``profile``). Publication Configuration C
    may pass ``setup_key="C_lab"`` after the X2 lab-matched focal correction.
    """
    setup0 = resolve_setup_selection(setup_key if setup_key is not None else profile)
    mapping = mapping_block(setup0, n=n)
    pixel_um = float(mapping["object_plane_pixel_um"])
    tgt = build_target(target_id, n, pixel_um=pixel_um)
    ref = tgt.to_simulation_reference()

    need_phase = target_id in PHASE_TARGETS
    state = quiet_state(
        setup0,
        nbar=nbar,
        num_frames=NUM_FRAMES_PHASE if need_phase else NUM_FRAMES_OPTICAL,
        enable_phase_steps=need_phase,
    )
    seed = int(master_seed) + 1009 * (ord(profile[0]) - ord("A")) + sum(
        ord(c) for c in target_id
    )

    result = simulate_forward(ref, state, seed=seed, detector_noise=False)
    intensity = intensity_map_from_result(result)
    vis_map = visibility_map_from_result(result)
    vo = dict(result.meta.get("visibility_observables") or {})

    x_um = mapping["object_x_um"]
    y_um = mapping["object_y_um"]
    line = central_line_profile(intensity, x_um, y_um)

    phase_block: dict[str, Any] | None = None
    if need_phase:
        D = np.asarray(result.frames_D, dtype=np.float64)
        rec = reconstruct_phase_steps(
            {"I0": D[0], "I_pi_2": D[1], "I_pi": D[2], "I_3pi_2": D[3]}
        )
        err = phase_error_metrics(
            rec.phase_wrapped_rad, tgt.phi_rad, remove_piston=True
        )
        phase_block = {
            "phase_reconstructed_rad": rec.phase_wrapped_rad,
            "phase_truth_rad": np.asarray(tgt.phi_rad, dtype=np.float64),
            "phase_rmse_rad": float(err["rmse_rad"]),
            "phase_mae_rad": float(err["mae_rad"]),
            "phase_max_abs_rad": float(err["max_abs_rad"]),
            "method": rec.method,
        }

    res_metrics = None
    if target_id in RESOLUTION_TARGETS:
        res_metrics = resolution_metrics_for_target(
            target_id,
            intensity,
            x_um=x_um,
            y_um=y_um,
            tgt_meta=dict(tgt.metadata),
        )

    return {
        "profile": profile,
        "architecture_label": ARCHITECTURE_LABELS[profile],
        "target_id": target_id,
        "seed_int": int(seed),
        "nbar": float(nbar),
        "grid_n": int(n),
        "object_pixel_um": float(pixel_um),
        "mapping": {k: v for k, v in mapping.items() if not isinstance(v, np.ndarray)},
        "object_x_um": x_um,
        "object_y_um": y_um,
        "detector_x_um": mapping["detector_x_um"],
        "detector_y_um": mapping["detector_y_um"],
        "detector_x_px": mapping["detector_x_px"],
        "detector_y_px": mapping["detector_y_px"],
        "object_T": np.asarray(tgt.T, dtype=np.float64),
        "object_phi_rad": np.asarray(tgt.phi_rad, dtype=np.float64),
        "optical_intensity": intensity,
        "visibility_map": vis_map,
        "N_1": float(vo.get("N_1") or 0.0),
        "N_2": float(vo.get("N_2") or 0.0),
        "C_12_abs": float(vo.get("C_12_abs") or 0.0),
        "visibility_V_50_50": float(vo.get("V_50_50") or 0.0),
        "visibility_map_mean": float(np.mean(vis_map)),
        "intensity_mean": float(np.mean(intensity)),
        "intensity_peak": float(np.max(intensity)),
        "line_profile_intensity": line,
        "phase": phase_block,
        "resolution": res_metrics,
        "target_metadata": tgt.full_metadata(),
        "physical_truth_ref": {
            k: PHYSICAL_TRUTH[k]
            for k in (
                "slit_width_um",
                "knife_edge_offset_um",
                "amplitude_step_T_left",
                "amplitude_step_T_right",
                "phase_step_phi_left_rad",
                "phase_step_phi_right_rad",
            )
            if k in PHYSICAL_TRUTH
        },
    }


def run_four_config_suite(
    *,
    profiles: tuple[str, ...] = PROFILES,
    targets: tuple[str, ...] = BENCHMARK_TARGETS,
    n: int = GRID_N,
    nbar: float = NBAR,
    master_seed: int = MASTER_SEED,
    setup_key_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the matched A/B/C/D suite.

    ``setup_key_map`` optionally remaps publication profile labels to setup
    aliases (e.g. ``{"C": "C_lab"}`` for X2-corrected lab-matched C).
    """
    key_map = dict(setup_key_map or {})
    suite: dict[str, Any] = {
        "generated_utc": utc_now(),
        "master_seed": int(master_seed),
        "grid_n": int(n),
        "nbar": float(nbar),
        "setup_key_map": key_map,
        "profiles": {},
        "metric_rows": [],
        "mapping_rows": [],
    }
    for profile in profiles:
        setup_key = key_map.get(profile, profile)
        suite["profiles"][profile] = {}
        # Mapping row once per profile (from transparent / first target)
        first = run_target_case(
            profile,
            targets[0],
            n=n,
            nbar=nbar,
            master_seed=master_seed,
            setup_key=setup_key,
        )
        suite["profiles"][profile][targets[0]] = first
        suite["mapping_rows"].append(
            {
                "profile": profile,
                "setup_key": setup_key,
                "architecture_label": ARCHITECTURE_LABELS[profile],
                **first["mapping"],
                "object_pixel_um": first["object_pixel_um"],
            }
        )
        _append_metric_row(suite, first)

        for tid in targets[1:]:
            case = run_target_case(
                profile,
                tid,
                n=n,
                nbar=nbar,
                master_seed=master_seed,
                setup_key=setup_key,
            )
            suite["profiles"][profile][tid] = case
            _append_metric_row(suite, case)
    return suite


def _append_metric_row(suite: dict[str, Any], case: dict[str, Any]) -> None:
    row: dict[str, Any] = {
        "profile": case["profile"],
        "architecture_label": case["architecture_label"],
        "target_id": case["target_id"],
        "object_pixel_um": case["object_pixel_um"],
        "intensity_mean": case["intensity_mean"],
        "intensity_peak": case["intensity_peak"],
        "N_1": case["N_1"],
        "N_2": case["N_2"],
        "C_12_abs": case["C_12_abs"],
        "visibility_V_50_50": case["visibility_V_50_50"],
        "visibility_map_mean": case["visibility_map_mean"],
        "signed_magnification": case["mapping"][
            "signed_object_to_camera_magnification"
        ],
        "sigma_object_um": case["mapping"]["correlation_sigma_object_um"],
        "sigma_camera_um": case["mapping"]["correlation_sigma_camera_um"],
        "phase_rmse_rad": (
            case["phase"]["phase_rmse_rad"] if case["phase"] else float("nan")
        ),
        "psf_fwhm_um": float("nan"),
        "esf_10_90_um": float("nan"),
        "lsf_fwhm_um": float("nan"),
    }
    if case["resolution"]:
        r = case["resolution"]
        if "psf_fwhm_um" in r:
            row["psf_fwhm_um"] = r["psf_fwhm_um"]
        if "esf_10_90_um" in r:
            row["esf_10_90_um"] = r["esf_10_90_um"]
        if "lsf_fwhm_um" in r:
            row["lsf_fwhm_um"] = r["lsf_fwhm_um"]
    suite["metric_rows"].append(row)
