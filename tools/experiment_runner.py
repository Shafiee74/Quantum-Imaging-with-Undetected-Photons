"""Task 9/10 experiment runner over the shared QIUP forward model.

This module contains orchestration and artifact writing only. All optical,
SPDC, propagation, interference, and detector calculations remain in the six
reviewed shared source modules.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
import uuid

import numpy as np
from PIL import Image, ImageDraw

from detector import validate_detector_operating_state
from latents import LatentState
from provenance import get_environment, hash_array, hash_config, hash_sim_arrays
from setup_params import (
    SetupParams,
    normalize_spdc_source_mode,
    physical_model_support_status,
    resolve_architecture_profile,
    resolve_setup_selection,
    setup_selection_info,
)
from simulate import (
    THREE_CONFIGURATION_PIPELINE_STAGES,
    SimResult,
    atomic_commit_sample_dir,
    cleanup_failed_sample_temp,
    make_sample_temp_dir,
    prepare_dataset_output_dir,
    simulate_forward,
    validate_simulation_request,
)


RUNNER_VERSION = "qiup_three_configuration_experiment_runner_v1"

EXPERIMENT_PARAMETERS: dict[str, frozenset[str]] = {
    "mean_photon_gain": frozenset({
        "mean_photons_per_mode", "mean_photons_per_mode_NL1",
        "mean_photons_per_mode_NL2",
    }),
    "pump_power": frozenset({"pump_power_mW"}),
    "crystal_length": frozenset({
        "crystal_length_mm", "crystal_length_NL2_mm",
        "crystal_length_pair_mm",
    }),
    "object_transmission": frozenset({
        "object_transmission", "object_type", "object_interaction_count",
    }),
    "object_phase": frozenset({
        "object_phase_rad", "object_type", "object_interaction_count",
    }),
    "generation_alignment": frozenset({
        "generation_dx_um", "generation_dy_um",
    }),
    "relay_mode_overlap": frozenset({"idler_mode_matching"}),
    "thermal_opd": frozenset({
        "delta_L_mm", "T_crystal_NL1_C", "T_crystal_NL2_C",
    }),
    "near_field_imaging": frozenset({
        "F1_mm", "F2_mm", "near_field_idler_magnification",
        "near_field_signal_magnification", "pump_waist_um",
        "crystal_length_mm",
    }),
    "far_field_imaging": frozenset({
        "F2_mm", "far_field_focal_mm",
        "far_field_idler_relay_magnification",
        "far_field_signal_relay_magnification", "pump_waist_um",
        "crystal_length_mm", "crystal_length_NL2_mm",
        "wavelength_triplet_nm",
    }),
}

SOFTWARE_SUPPORTED_RANGES: dict[str, str] = {
    "mean_photon_gain": (
        "finite nbar >= 0 with explicit numerical-overflow rejection; "
        "software regression covers 1e-4 through 1e3"
    ),
    "pump_power": (
        "finite pump power >= 0 only with documented per-interaction source "
        "calibration and valid pump topology"
    ),
    "crystal_length": "2 mm <= each physical crystal length <= 20 mm",
    "object_transmission": "0 <= object intensity transmission/reflectance <= 1",
    "object_phase": "0 <= phase <= 2*pi for this runner",
    "generation_alignment": "finite generation offsets accepted by SetupParams",
    "relay_mode_overlap": "0 <= idler mode overlap <= 1",
    "thermal_opd": "signed finite static OPD and validated crystal temperatures",
    "near_field_imaging": "validated positive near-field relay geometry",
    "far_field_imaging": "validated positive far-field relay/lens geometry",
}

_SETUP_FIELDS = frozenset(f.name for f in fields(SetupParams))
_STATE_FIELDS = frozenset({
    "mean_photons_per_mode", "mean_photons_per_mode_NL1",
    "mean_photons_per_mode_NL2", "pump_power_mW",
})
_SPECIAL_FIELDS = frozenset({
    "object_transmission", "object_phase_rad", "crystal_length_pair_mm",
    "wavelength_triplet_nm",
})


@dataclass
class PreparedPoint:
    index: int
    requested: dict[str, Any]
    setup: SetupParams
    state: LatentState
    ref: dict[str, Any]
    seed: int
    preflight: dict[str, Any]


def default_experiment_reference(size: int = 32) -> dict[str, Any]:
    """Deterministic transmissive object used by the command-line examples."""
    n = int(size)
    if n < 4:
        raise ValueError("reference size must be >= 4")
    y, x = np.mgrid[:n, :n]
    cx = cy = (n - 1) / 2.0
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    aperture = (radius <= 0.42 * n).astype(np.float32)
    pattern = 0.25 + 0.75 * (((x // 4) + (y // 6)) % 2)
    T = (aperture * pattern).astype(np.float32)
    phi = (0.25 * np.sin(x / 5.0) * aperture).astype(np.float32)
    one = np.ones((n, n), dtype=np.float32)
    zero = np.zeros((n, n), dtype=np.float32)
    return {
        "M": one, "T": T, "phi": phi, "O_i": one,
        "D": zero, "target_size": n,
    }


def _copy_ref(ref: dict[str, Any]) -> dict[str, Any]:
    out = dict(ref)
    for name in ("M", "T", "phi", "O_i", "D"):
        if name in out and out[name] is not None:
            out[name] = np.array(out[name], copy=True)
    return out


def validate_experiment_points(
    experiment: str,
    points: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject unknown experiments/parameters before any output path is made."""
    name = str(experiment).strip().lower()
    if name not in EXPERIMENT_PARAMETERS:
        raise ValueError(
            f"unknown experiment {experiment!r}; choose from "
            f"{sorted(EXPERIMENT_PARAMETERS)}"
        )
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(points):
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"sweep point {index} must be a non-empty mapping")
        unknown = sorted(set(raw) - EXPERIMENT_PARAMETERS[name])
        if unknown:
            raise ValueError(
                f"unknown {name} sweep parameters: {unknown}; allowed: "
                f"{sorted(EXPERIMENT_PARAMETERS[name])}"
            )
        normalized.append(dict(raw))
    if not normalized:
        raise ValueError("at least one sweep point is required")
    return normalized


def points_from_values(parameter: str, values: Iterable[Any]) -> list[dict[str, Any]]:
    return [{str(parameter): value} for value in values]


def _validate_special_value(name: str, value: Any) -> None:
    if name == "object_transmission":
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("object_transmission must be finite and in [0, 1]")
    elif name == "object_phase_rad":
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 2.0 * np.pi:
            raise ValueError("object_phase_rad must be finite and in [0, 2*pi]")
    elif name == "crystal_length_pair_mm":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("crystal_length_pair_mm must be [L1_mm, L2_mm]")
    elif name == "wavelength_triplet_nm":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(
                "wavelength_triplet_nm must be [pump_nm, signal_nm, idler_nm]"
            )


def prepare_experiment_point(
    *,
    profile: str | Path | SetupParams,
    experiment: str,
    requested: dict[str, Any],
    ref: dict[str, Any],
    index: int,
    seed: int,
    exposure_s: float = 0.05,
    em_gain: float = 250.0,
) -> PreparedPoint:
    """Resolve one point through the same strict setup/simulation validator."""
    points = validate_experiment_points(experiment, [requested])
    requested = points[0]
    setup = resolve_setup_selection(profile)
    architecture = resolve_architecture_profile(setup)
    if experiment == "near_field_imaging" and architecture.imaging_mode != "near_field":
        raise ValueError("near_field_imaging requires configuration B")
    if experiment == "far_field_imaging" and architecture.imaging_mode != "far_field":
        raise ValueError("far_field_imaging requires configuration A or C")
    source_mode = normalize_spdc_source_mode(setup.spdc_source_mode)
    if experiment == "mean_photon_gain" and source_mode != "fixed_nbar":
        raise ValueError("mean_photon_gain requires spdc_source_mode=fixed_nbar")
    if experiment == "pump_power" and source_mode != "calibrated_source":
        raise ValueError("pump_power requires spdc_source_mode=calibrated_source")

    setup_values: dict[str, Any] = {}
    state_values: dict[str, Any] = {}
    point_ref = _copy_ref(ref)
    for key, value in requested.items():
        if key in _SPECIAL_FIELDS:
            _validate_special_value(key, value)
            if key == "object_transmission":
                point_ref["T"] = np.full_like(
                    np.asarray(point_ref["T"], dtype=np.float32), float(value)
                )
            elif key == "object_phase_rad":
                point_ref["phi"] = np.full_like(
                    np.asarray(point_ref["phi"], dtype=np.float32), float(value)
                )
            elif key == "crystal_length_pair_mm":
                if architecture.profile_id != "A":
                    raise ValueError(
                        "crystal_length_pair_mm is only valid for two-crystal A"
                    )
                setup_values["crystal_length_mm"] = value[0]
                setup_values["crystal_length_NL2_mm"] = value[1]
            elif key == "wavelength_triplet_nm":
                setup_values.update({
                    "lambda_p_nm": value[0],
                    "lambda_s_nm": value[1],
                    "lambda_i_nm": value[2],
                })
        elif key in _STATE_FIELDS:
            state_values[key] = value
        elif key in _SETUP_FIELDS:
            setup_values[key] = value
        else:  # guarded by validate_experiment_points
            raise ValueError(f"unsupported experiment parameter {key!r}")

    if "object_type" in setup_values and "object_interaction_count" not in setup_values:
        setup_values["object_interaction_count"] = (
            1 if setup_values["object_type"] == "reflective"
            else (1 if architecture.profile_id == "A" else 2)
        )
    if setup_values:
        setup = setup.overridden(**setup_values)

    state = LatentState(
        setup=setup,
        num_frames=1,
        exposure_s=float(exposure_s),
        em_gain=float(em_gain),
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
        **state_values,
    )
    validate_detector_operating_state(state)
    preflight = validate_simulation_request(point_ref, state)
    return PreparedPoint(
        index=int(index), requested=requested, setup=setup, state=state,
        ref=point_ref, seed=int(seed), preflight=preflight,
    )


def _preview_u8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.round(np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _point_record(
    point: PreparedPoint,
    result: SimResult,
    *,
    experiment: str,
    sample_id: str,
) -> dict[str, Any]:
    ideal = np.asarray(result.ideal_signal_image, dtype=np.float32)
    detector = np.asarray(result.detector_signal_image, dtype=np.float32)
    clean = np.asarray(result.clean_reference_image, dtype=np.float32)
    operating = dict(result.meta["spdc_operating_point"])
    vis = dict(result.meta["visibility_observables"])
    support = dict(result.meta["physical_model_support"])
    architecture = dict(result.meta["architecture_profile"])
    selection = dict(result.meta["setup_selection"])
    array_hashes = {
        **hash_sim_arrays(
            frames_plus=result.frames_plus,
            frames_minus=result.frames_minus,
            frames_D=result.frames_D,
            clean_D=result.clean_D,
        ),
        "ideal_optical_signal": hash_array(ideal),
        "detector_signal": hash_array(detector),
        "clean_reference": hash_array(clean),
    }
    pipeline_status = []
    for stage in result.meta["pipeline_stage_status"]:
        item = dict(stage)
        if item.get("stage") == "validated_atomic_output_transaction":
            item["status"] = "completed_after_precommit_validation"
        pipeline_status.append(item)
    spatial = result.meta.get("qiup_spatial_transfer") or {}
    provenance = dict(result.meta.get("provenance") or {})
    spdc_physics = dict(result.meta.get("spdc_physics") or {})
    return {
        "runner_version": RUNNER_VERSION,
        "runner_sha256": _runner_sha256(),
        "sample_id": sample_id,
        "experiment": experiment,
        "requested_parameters": point.requested,
        "requested_configuration": point.setup.to_dict(),
        "physics_policy_id": getattr(point.setup, "physics_policy_id", None),
        "selected_profile": architecture.get("profile_id"),
        "selected_yaml_path": selection.get("yaml_path"),
        "selected_yaml_sha256": selection.get("yaml_sha256"),
        "registered_profile_name": selection.get("registered_profile_name"),
        "architecture": architecture.get("spdc_architecture"),
        "imaging_mode": architecture.get("imaging_mode"),
        "physical_crystal_count": architecture.get("n_physical_crystals"),
        "spdc_interaction_count": architecture.get("n_spdc_interactions"),
        "blur_model": result.meta.get("blur_model"),
        "source_mode": operating.get("spdc_source_mode"),
        "source_input_type": operating.get("source_input_type"),
        "gain_regime_NL1": operating.get("regime_NL1"),
        "gain_regime_NL2": operating.get("regime_NL2"),
        "crystal_parameters": result.meta["architecture_parameters"][
            "physical_crystals"
        ],
        "interaction_parameters": operating.get("interactions"),
        "pump_powers": {
            "laser_mW": operating.get("pump_power_laser_mW"),
            "interaction_1_mW": operating.get("pump_power_NL1_mW"),
            "interaction_2_mW": operating.get("pump_power_NL2_mW"),
            "P_target_1_mW": operating.get("P_target_1_mW"),
            "P_target_2_mW": operating.get("P_target_2_mW"),
            "P_actual_1_mW": operating.get("P_actual_1_mW"),
            "P_actual_2_mW": operating.get("P_actual_2_mW"),
            "delta_P": operating.get("delta_P"),
            "abs_delta_P": operating.get("abs_delta_P"),
            "hidden_equalization_applied": operating.get(
                "hidden_equalization_applied"
            ),
        },
        "resolved_gain_parameters": {
            "nbar_1": operating.get("mean_photons_per_mode_NL1"),
            "nbar_2": operating.get("mean_photons_per_mode_NL2"),
            "gain_r_1": operating.get("spdc_gain_r_NL1"),
            "gain_r_2": operating.get("spdc_gain_r_NL2"),
        },
        "object_type": point.setup.object_type,
        "object_device": getattr(point.setup, "object_device", None),
        "object_mode": getattr(point.setup, "object_mode", None),
        "object_interaction_count": getattr(
            point.setup, "object_interaction_count", None
        ),
        "object_hash": provenance.get("object_hash"),
        "object_evidence": provenance.get("object_evidence"),
        "effective_object_transfer": result.meta["effective_object_transfer"],
        "spatial_correlation_model": spatial,
        "camera_coordinate_mapping": spatial.get("coordinate_mapping"),
        "spatial_resolution": {
            "correlation_sigma_px": spatial.get("correlation_sigma_px"),
            "sigma_spdc_px": (result.meta.get("resolved_forward") or {}).get(
                "sigma_spdc_px"
            ),
            "sigma_diffraction_px": result.meta.get("sigma_diffraction_px"),
            "sigma_total_px": result.meta.get("sigma_total_px"),
        },
        "signed_optical_path_difference_mm": result.meta.get(
            "signed_optical_path_difference_mm"
        ),
        "interference_phase_rad": result.meta.get("interference_phase_rad"),
        "visibility": vis,
        "mean_ideal_signal_intensity": float(np.mean(ideal)),
        "selected_output_port": result.selected_output_port,
        "detector_parameters": result.meta["detector_parameters"],
        "signal_filter": result.meta.get("signal_filter"),
        "qpm": {
            "interaction_1": spdc_physics.get("qpm_interaction_1"),
            "interaction_2": spdc_physics.get("qpm_interaction_2"),
        },
        "seed": int(point.seed),
        "rng_scheme": provenance.get("rng_scheme"),
        "timestamp": provenance.get("timestamp"),
        "environment": provenance["environment"],
        "configuration_hash": hash_config(point.setup),
        "source_hashes": provenance["source_hashes"],
        "build_hash": provenance["build_hash"],
        "package_hashes": provenance.get("package_hashes"),
        "array_hashes": array_hashes,
        "physical_model_support": support,
        "implemented": support["implemented"],
        "software_tested": support["software_tested"],
        "experimentally_validated": support["experimentally_validated"],
        "pipeline_execution_order": list(THREE_CONFIGURATION_PIPELINE_STAGES),
        "pipeline_stage_status": pipeline_status,
        "clean_reference_definition": {
            "source": result.meta.get("clean_reference_source"),
            "same_architecture": True,
            "removed_effects": [
                "Poisson photoelectron sampling",
                "stochastic EM multiplication",
                "dark current and CIC",
                "hot pixels and PRNU",
                "signed read noise, bias, saturation and ADC digitization",
            ],
        },
        "resolved_parameters": {
            "setup": point.setup.to_dict(),
            "resolved_forward": result.meta.get("resolved_forward"),
            "spdc_operating_point": operating,
            "spdc_physics": result.meta.get("spdc_physics"),
            "detector": result.meta.get("detector_parameters"),
        },
        "output_transaction": {
            "mode": "validated_temp_directory_then_atomic_rename",
            "precommit_validation": True,
            "commit_operation": "os.replace",
        },
        "files": {
            "ideal_optical_signal": "ideal_optical_signal.npy",
            "detector_signal": "detector_signal.npy",
            "clean_reference": "clean_reference.npy",
            "frames_plus": "frames_plus.npy",
            "frames_minus": "frames_minus.npy",
            "frames_D": "frames_D.npy",
            "clean_D": "clean_D.npy",
            "visibility": "visibility.json",
            "metadata": "metadata.json",
        },
    }


def _validate_temp_point(tmp_dir: Path, record: dict[str, Any]) -> None:
    required = tuple(record["files"].values())
    missing = [name for name in required if not (tmp_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"experiment sample missing files before commit: {missing}")
    arrays = {
        "ideal_optical_signal": np.load(tmp_dir / "ideal_optical_signal.npy"),
        "detector_signal": np.load(tmp_dir / "detector_signal.npy"),
        "clean_reference": np.load(tmp_dir / "clean_reference.npy"),
        "frames_plus": np.load(tmp_dir / "frames_plus.npy"),
        "frames_minus": np.load(tmp_dir / "frames_minus.npy"),
        "frames_D": np.load(tmp_dir / "frames_D.npy"),
        "clean_D": np.load(tmp_dir / "clean_D.npy"),
    }
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN/Inf before commit")
        expected = record["array_hashes"][name]
        if hash_array(arr) != expected:
            raise RuntimeError(f"{name} hash mismatch before commit")
    parsed = json.loads((tmp_dir / "metadata.json").read_text(encoding="utf-8"))
    if parsed.get("sample_id") != record["sample_id"]:
        raise RuntimeError("metadata sample_id mismatch before commit")


def write_experiment_point_atomic(
    samples_dir: Path,
    point: PreparedPoint,
    result: SimResult,
    *,
    experiment: str,
    sample_number: int,
) -> tuple[Path, dict[str, Any]]:
    """Write, reload, validate, then atomically rename one sweep point."""
    sample_id = f"sample_{int(sample_number):04d}"
    tmp_dir = make_sample_temp_dir(samples_dir, sample_id)
    final_dir = samples_dir / sample_id
    try:
        record = _point_record(
            point, result, experiment=experiment, sample_id=sample_id
        )
        arrays = {
            "ideal_optical_signal.npy": result.ideal_signal_image,
            "detector_signal.npy": result.detector_signal_image,
            "clean_reference.npy": result.clean_reference_image,
            "frames_plus.npy": result.frames_plus,
            "frames_minus.npy": result.frames_minus,
            "frames_D.npy": result.frames_D,
            "clean_D.npy": result.clean_D,
        }
        for filename, array in arrays.items():
            np.save(tmp_dir / filename, np.asarray(array, dtype=np.float32))
        (tmp_dir / "visibility.json").write_text(
            json.dumps(record["visibility"], indent=2) + "\n", encoding="utf-8"
        )
        (tmp_dir / "metadata.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        Image.fromarray(_preview_u8(result.ideal_signal_image), mode="L").save(
            tmp_dir / "ideal_optical_signal.png"
        )
        Image.fromarray(_preview_u8(result.detector_signal_image), mode="L").save(
            tmp_dir / "detector_signal.png"
        )
        _validate_temp_point(tmp_dir, record)
        committed = atomic_commit_sample_dir(tmp_dir, final_dir)
        return committed, record
    except Exception:
        cleanup_failed_sample_temp(tmp_dir)
        raise


def _plot_x_value(requested: dict[str, Any], index: int) -> float:
    for value in requested.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, (list, tuple)) and value:
            try:
                return float(value[0])
            except (TypeError, ValueError):
                pass
    return float(index)


def _write_visibility_plot(
    out_dir: Path,
    experiment: str,
    records: list[dict[str, Any]],
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([
        _plot_x_value(record["requested_parameters"], index)
        for index, record in enumerate(records)
    ])
    v = np.array([record["visibility"]["V_50_50"] for record in records], dtype=float)
    g = np.array([record["visibility"]["g_1"] for record in records], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(x, v, "o-", label=r"$V_{50:50}$")
    ax.plot(x, g, "s--", label=r"$g^{(1)}$")
    ax.set_xlabel("requested sweep parameter")
    ax.set_ylabel("visibility / coherence")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(experiment.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out_dir / "visibility_vs_parameter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _write_image_grid(out_dir: Path, records: list[dict[str, Any]]) -> str:
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    for record in records:
        sample = out_dir / record["sample_dir"]
        ideal = _preview_u8(np.load(sample / "ideal_optical_signal.npy"))
        detector = _preview_u8(np.load(sample / "detector_signal.npy"))
        previews.append((record["sample_id"], ideal, detector))
    h, w = previews[0][1].shape
    scale = max(1, 128 // max(h, w))
    tile_w, tile_h = w * scale, h * scale
    label_h = 20
    canvas = Image.new("L", (2 * tile_w, len(previews) * (tile_h + label_h)), 255)
    draw = ImageDraw.Draw(canvas)
    for row, (sample_id, ideal, detector) in enumerate(previews):
        top = row * (tile_h + label_h)
        draw.text((3, top + 3), f"{sample_id} ideal", fill=0)
        draw.text((tile_w + 3, top + 3), f"{sample_id} detector", fill=0)
        for col, array in enumerate((ideal, detector)):
            image = Image.fromarray(array, mode="L").resize(
                (tile_w, tile_h), Image.Resampling.NEAREST
            )
            canvas.paste(image, (col * tile_w, top + label_h))
    path = out_dir / "representative_image_grid.png"
    canvas.save(path)
    return path.name


def _write_json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_resolved_table(out_dir: Path, records: list[dict[str, Any]]) -> str:
    path = out_dir / "resolved_parameters.csv"
    rows = []
    for record in records:
        vis = record["visibility"]
        gain = record["resolved_gain_parameters"]
        rows.append({
            "sample_id": record["sample_id"],
            "profile": record["selected_profile"],
            "architecture": record["architecture"],
            "imaging_mode": record["imaging_mode"],
            "source_mode": record["source_mode"],
            "gain_regime_NL1": record["gain_regime_NL1"],
            "gain_regime_NL2": record["gain_regime_NL2"],
            "nbar_1": gain["nbar_1"], "nbar_2": gain["nbar_2"],
            "gain_r_1": gain["gain_r_1"], "gain_r_2": gain["gain_r_2"],
            "N_1": vis["N_1"], "N_2": vis["N_2"],
            "C_12_abs": vis["C_12_abs"], "g_1": vis["g_1"],
            "V_50_50": vis["V_50_50"],
            "mean_ideal_signal_intensity": record["mean_ideal_signal_intensity"],
            "configuration_hash": record["configuration_hash"],
            "seed": record["seed"],
            "support_level": record["physical_model_support"]["support_level"],
            "experimentally_validated": record["experimentally_validated"],
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path.name


def run_parameter_sweep(
    *,
    profile: str | Path | SetupParams,
    experiment: str,
    points: Iterable[dict[str, Any]],
    ref: dict[str, Any],
    out_dir: str | Path,
    seed0: int = 9000,
    detector_noise: bool = True,
    create_run_subdir: bool = True,
) -> dict[str, Any]:
    """Preflight, execute, atomically save, and summarize one experiment."""
    normalized = validate_experiment_points(experiment, points)
    prepared = [
        prepare_experiment_point(
            profile=profile, experiment=experiment, requested=requested,
            ref=ref, index=index, seed=seed0 + 17 * index,
        )
        for index, requested in enumerate(normalized)
    ]
    # Nothing has touched out_dir before all points pass the strict validator.
    output = prepare_dataset_output_dir(
        out_dir, create_run_subdir=create_run_subdir
    )
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=False, exist_ok=False)
    records: list[dict[str, Any]] = []
    for number, point in enumerate(prepared):
        result = simulate_forward(
            point.ref, point.state, seed=point.seed,
            detector_noise=detector_noise, raise_on_forbidden=True,
        )
        committed, record = write_experiment_point_atomic(
            samples_dir, point, result, experiment=experiment,
            sample_number=number,
        )
        record["sample_dir"] = str(committed.relative_to(output)).replace("\\", "/")
        records.append(record)

    visibility_plot = _write_visibility_plot(output, experiment, records)
    image_grid = _write_image_grid(output, records)
    table = _write_resolved_table(output, records)
    profile_info = resolve_architecture_profile(prepared[0].setup)
    manifest = {
        "runner_version": RUNNER_VERSION,
        "runner_sha256": _runner_sha256(),
        "experiment": experiment,
        "selected_profile": profile_info.profile_id,
        "architecture": profile_info.spdc_architecture,
        "imaging_mode": profile_info.imaging_mode,
        "source_mode": prepared[0].setup.spdc_source_mode,
        "n_samples": len(records),
        "strict_validator": "simulate.validate_simulation_request",
        "unknown_parameters_rejected_before_output": True,
        "sample_transaction": "validated temp directory then os.replace",
        "visibility_equation": "2*abs(C_12)/(N_1+N_2)",
        "arbitrary_visibility_coefficient_used": False,
        "range_status": {
            "software_supported_range": SOFTWARE_SUPPORTED_RANGES[experiment],
            "apparatus_operating_range": None,
            "apparatus_operating_range_status": "not supplied",
            "experimentally_validated_range": None,
            "experimentally_validated_range_status": "not supplied",
        },
        "environment": get_environment(),
        "visibility_plot": visibility_plot,
        "representative_image_grid": image_grid,
        "resolved_parameter_table": table,
        "samples": records,
    }
    _write_json_atomic(output / "experiment_manifest.json", manifest)
    return {**manifest, "output_dir": str(output.resolve())}


def _calibrated_source_gaps(setup: SetupParams) -> list[str]:
    """List missing traceability inputs without attempting a prediction."""
    profile = resolve_architecture_profile(setup)
    missing: list[str] = []
    for field_name in ("spdc_gain_per_sqrt_mW", "spdc_gain_calibration_id"):
        value = getattr(setup, field_name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    if profile.profile_id == "A":
        for field_name in (
            "spdc_gain_per_sqrt_mW_NL2",
            "spdc_gain_calibration_id_NL2",
        ):
            value = getattr(setup, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)
        topology = getattr(setup, "two_crystal_pump_topology", None)
        if topology == "sequential":
            if getattr(setup, "pump_between_crystals_transmission", None) is None:
                missing.append("pump_between_crystals_transmission")
    return missing


def run_required_configuration_matrix(
    *,
    ref: dict[str, Any],
    out_dir: str | Path,
    seed0: int = 12000,
    detector_noise: bool = True,
    create_run_subdir: bool = True,
) -> dict[str, Any]:
    """Execute the supported Task 9 matrix and audit unavailable cases.

    The installed A/B/C profiles intentionally contain no apparatus gain
    calibration. Their six calibrated-source rows are therefore recorded as
    not run. They are never counted as a physics pass.
    """
    profile_names = {"A": "A", "B": "B", "C": "C"}
    target_nbar = {"low": 1.0e-3, "high": 5.0, "transition": 0.1}
    matrix_rows: list[dict[str, Any]] = []
    prepared_cases: list[tuple[str, str, PreparedPoint]] = []
    seed_index = 0

    # Preflight every executable case before touching the output directory.
    for profile_id, profile_name in profile_names.items():
        setup = resolve_setup_selection(profile_name)
        calibration_gaps = _calibrated_source_gaps(setup)
        for source_mode in ("fixed_nbar", "calibrated_source"):
            for gain_target in ("low", "high"):
                case_id = f"{profile_id}_{source_mode}_{gain_target}"
                row = {
                    "case_id": case_id,
                    "profile": profile_id,
                    "source_mode": source_mode,
                    "requested_gain_regime": gain_target,
                }
                if source_mode == "calibrated_source":
                    row.update({
                        "status": "not_run_calibration_unavailable",
                        "physics_pass": False,
                        "reason": (
                            "quantitative pump prediction requires documented "
                            "calibration; missing " + ", ".join(calibration_gaps)
                        ),
                        "missing_calibration_parameters": calibration_gaps,
                    })
                    matrix_rows.append(row)
                    continue
                point = prepare_experiment_point(
                    profile=profile_name,
                    experiment="mean_photon_gain",
                    requested={
                        "mean_photons_per_mode": target_nbar[gain_target]
                    },
                    ref=ref,
                    index=seed_index,
                    seed=seed0 + 17 * seed_index,
                )
                seed_index += 1
                row.update({
                    "status": "preflight_complete",
                    "physics_pass": False,
                    "reason": "implemented research-only approximation",
                })
                matrix_rows.append(row)
                prepared_cases.append((case_id, "required_matrix", point))

        transition_id = f"{profile_id}_fixed_nbar_transition"
        transition = prepare_experiment_point(
            profile=profile_name,
            experiment="mean_photon_gain",
            requested={"mean_photons_per_mode": target_nbar["transition"]},
            ref=ref,
            index=seed_index,
            seed=seed0 + 17 * seed_index,
        )
        seed_index += 1
        prepared_cases.append((transition_id, "additional_transition", transition))

        if profile_id in {"B", "C"}:
            object_id = f"{profile_id}_reflective_object"
            object_point = prepare_experiment_point(
                profile=profile_name,
                experiment="object_transmission",
                requested={
                    "object_transmission": 0.64,
                    "object_type": "reflective",
                    "object_interaction_count": 1,
                },
                ref=ref,
                index=seed_index,
                seed=seed0 + 17 * seed_index,
            )
            seed_index += 1
            prepared_cases.append((object_id, "additional_object_type", object_point))

    output = prepare_dataset_output_dir(
        out_dir, create_run_subdir=create_run_subdir
    )
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=False, exist_ok=False)
    records: list[dict[str, Any]] = []
    matrix_by_id = {row["case_id"]: row for row in matrix_rows}
    for number, (case_id, case_kind, point) in enumerate(prepared_cases):
        result = simulate_forward(
            point.ref, point.state, seed=point.seed,
            detector_noise=detector_noise, raise_on_forbidden=True,
        )
        committed, record = write_experiment_point_atomic(
            samples_dir, point, result, experiment=case_kind,
            sample_number=number,
        )
        record["sample_dir"] = str(committed.relative_to(output)).replace("\\", "/")
        record["case_id"] = case_id
        record["case_kind"] = case_kind
        records.append(record)
        if case_id in matrix_by_id:
            row = matrix_by_id[case_id]
            row.update({
                "status": "executed_research_only_approximation",
                "sample_id": record["sample_id"],
                "sample_dir": record["sample_dir"],
                "resolved_gain_regime_NL1": record["gain_regime_NL1"],
                "resolved_gain_regime_NL2": record["gain_regime_NL2"],
                "visibility": record["visibility"]["V_50_50"],
                "implemented": record["implemented"],
                "software_tested": record["software_tested"],
                "experimentally_validated": record["experimentally_validated"],
                "support_level": record["physical_model_support"]["support_level"],
            })

    visibility_plot = _write_visibility_plot(output, "task9_configuration_matrix", records)
    image_grid = _write_image_grid(output, records)
    table = _write_resolved_table(output, records)
    n_executed = sum(row["status"].startswith("executed") for row in matrix_rows)
    n_not_run = sum(row["status"].startswith("not_run") for row in matrix_rows)
    manifest = {
        "runner_version": RUNNER_VERSION,
        "matrix": "Task 9 A/B/C x fixed/calibrated x low/high",
        "required_case_count": 12,
        "required_cases_executed": n_executed,
        "required_cases_not_run": n_not_run,
        "full_12_case_physics_pass": False,
        "full_12_case_physics_pass_reason": (
            "six calibrated-source cases were not run because the supplied "
            "profiles contain no documented apparatus gain calibration; all "
            "executed configurations remain research-only approximations"
        ),
        "calibrated_cases_fabricated": False,
        "strict_validator": "simulate.validate_simulation_request",
        "all_executable_cases_preflighted_before_output": True,
        "sample_transaction": "validated temp directory then os.replace",
        "required_matrix": matrix_rows,
        "additional_cases": [
            {
                "case_id": record["case_id"],
                "case_kind": record["case_kind"],
                "sample_id": record["sample_id"],
                "sample_dir": record["sample_dir"],
                "gain_regime_NL1": record["gain_regime_NL1"],
                "gain_regime_NL2": record["gain_regime_NL2"],
                "object_type": record["object_type"],
                "visibility": record["visibility"]["V_50_50"],
            }
            for record in records if record["case_kind"] != "required_matrix"
        ],
        "visibility_plot": visibility_plot,
        "representative_image_grid": image_grid,
        "resolved_parameter_table": table,
        "samples": records,
        "environment": get_environment(),
    }
    _write_json_atomic(output / "matrix_manifest.json", manifest)
    return {**manifest, "output_dir": str(output.resolve())}


def _parse_values(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--values requires comma-separated numbers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="A/B/C, registered name, or YAML path")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENT_PARAMETERS))
    parser.add_argument("--parameter", help="single parameter name for a one-dimensional sweep")
    parser.add_argument("--values", help="comma-separated numerical values")
    parser.add_argument("--points-json", help="JSON file containing a list of parameter mappings")
    parser.add_argument("--out", required=True, help="new output root")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--direct-output", action="store_true", help="use --out directly and require it empty")
    args = parser.parse_args(argv)
    if args.points_json:
        points = json.loads(Path(args.points_json).read_text(encoding="utf-8"))
    else:
        if not args.parameter or not args.values:
            parser.error("provide --points-json or both --parameter and --values")
        points = points_from_values(args.parameter, _parse_values(args.values))
    manifest = run_parameter_sweep(
        profile=args.profile,
        experiment=args.experiment,
        points=points,
        ref=default_experiment_reference(args.size),
        out_dir=args.out,
        seed0=args.seed,
        create_run_subdir=not args.direct_output,
    )
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "experiment": manifest["experiment"],
        "profile": manifest["selected_profile"],
        "n_samples": manifest["n_samples"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
