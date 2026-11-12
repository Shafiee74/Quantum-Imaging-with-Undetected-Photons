"""
QIUP Digital Twin -- SECTION: simulate.py  (forward orchestrator + dataset)
Ties optics + detector into a full forward pass and generates labeled
datasets. Enforces the optical-before-detector ordering.

Paper-1 scope: single analog frame I+ (ADU), N=1. Covariance / g2 /
grid-search are NOT here (that is Paper-2, and was removed with upi steps 4-8).

CONTENTS: SimResult, simulate_forward, simulate_sequence, sweep_cluster, plots,
  generate_personal_dataset
Provenance: qiup_system_architecture.py
"""
from __future__ import annotations
import os, csv, json, shutil, uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import numpy as np
from setup_params import (
    SetupParams, ResolvedForward, resolve_forward, sensor_temperature_C,
    dark_rate_from_setup, read_noise_e_from_setup,
    load_setup_yaml, save_setup_yaml, setup_overrides_from_sweep,
    validate_sweep_names, LATENT_SWEEP_FIELDS,
    detector_mode_is_photon_counting,
    detector_mode_from_photon_counting_flag,
    ConfigError,
    assert_canonical_setup,
    is_canonical_setup,
    reject_canonical_overrides,
    verify_canonical_file_hashes,
    canonical_config_audit,
    resolve_architecture_profile,
    architecture_parameter_snapshot,
    resolve_setup_selection,
    setup_selection_info,
    validate_forward_profile_supported,
    validate_setup_params,
    physical_model_support_status,
    resolved_phase_step_schedule,
)
from latents import (
    LatentState, apply_preset, check_forbidden, check_state, sample_latents,
    InvalidSampleError, SamplingExhaustedError,
    CLUSTERS, CLUSTER_CONTRACTS, COUPLING_WEIGHTS, COUPLING_WEIGHTS_ROLE,
    GAIN_SEMANTICS, LEMOS_REFERENCE, PHASE4_POLICY, ValidityReport,
    Phase4DeferredError, Phase4PolicyError,
    request_zernike_wavefront, validate_beam_semantics,
    LATENT_KEYS,
)
from optics import (
    OPTICAL_STAGES, VISIBILITY_DEFAULTS, PSF_DEFAULTS, SLM_TAXONOMY,
    _optical_intensity, build_clean_reference, normalize,
    validate_reference_schema, reference_target_hw, ReferenceSchemaError,
    reconstruct_phase_from_phase_steps,
)
from detector import (
    DETECTOR_STAGES_ANALOG,
    DETECTOR_STAGES_PC,
    QIUP_RNG_STREAM_NAMES,
    _detector_pc_frame,
    _fixed_patterns,
    spawn_qiup_rngs,
    resolve_detector_mode,
    detector_stages_for_setup,
    validate_detector_operating_state,
    resolved_detector_parameters,
)
from provenance import (
    assert_execution_evidence,
    build_configuration_evidence,
    build_object_evidence,
    build_run_provenance,
    compare_array_hashes,
    flatten_execution_evidence,
    get_environment,
    hash_array,
    hash_config,
    hash_sim_arrays,
    load_object_ref_from_evidence,
    save_object_evidence,
    MODEL_STATUS,
    MODEL_VERSION,
    RELEASE_STATUS,
    RELEASE_VERSION,
    REQUIRED_EXECUTION_EVIDENCE_KEYS,
)

# ==============================================================================
# RESULT + ORCHESTRATOR
# Role: forward pass
# Models: object + LatentState -> optics -> [detector] -> frame + meta
# Provenance: arch
# ==============================================================================

THREE_CONFIGURATION_PIPELINE_STAGES: tuple[str, ...] = (
    "yaml_selection",
    "configuration_validation",
    "physical_architecture_resolution",
    "physical_crystal_and_interaction_resolution",
    "source_mode_and_gain_resolution",
    "pump_propagation_and_qpm",
    "spdc_field_and_moment_calculation",
    "signal_idler_propagation",
    "object_interaction",
    "architecture_specific_spatial_correlation",
    "interference_and_optical_image_formation",
    "analog_emccd_detector",
    "image_visibility_metadata_and_provenance",
    "validated_atomic_output_transaction",
)

@dataclass
class SimResult:
    frames_plus: np.ndarray
    frames_minus: np.ndarray
    frames_D: np.ndarray
    sample_dots: np.ndarray
    meta: dict[str, Any]
    clean_D: np.ndarray
    state: LatentState
    validity: ValidityReport
    # Task 8 primary scientific image pair. These are the same selected signal
    # port before and after the shared detector chain for the mid sequence frame.
    ideal_signal_image: np.ndarray | None = None
    detector_signal_image: np.ndarray | None = None
    clean_reference_image: np.ndarray | None = None
    reconstructed_phase_rad: np.ndarray | None = None
    selected_output_port: str = "plus"


# Re-export for callers that import schema guards from simulate (V-P0-11).
__all_schema__ = (
    "ReferenceSchemaError",
    "validate_reference_schema",
    "reference_target_hw",
    "validate_result_shapes",
)


def validate_result_shapes(
    ref: dict[str, Any],
    *,
    frames_plus: np.ndarray,
    frames_minus: np.ndarray,
    frames_D: np.ndarray,
    clean_D: np.ndarray,
) -> None:
    """Ensure saved frames match the validated reference grid (V-P0-11)."""
    H, W = reference_target_hw(ref)
    for name, arr in (
        ("frames_plus", frames_plus),
        ("frames_minus", frames_minus),
        ("frames_D", frames_D),
    ):
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape[-2:] != (H, W):
            raise ReferenceSchemaError(
                f"{name} shape mismatch: got {a.shape}, expected (*, {H}, {W})"
            )
        if not np.isfinite(a).all():
            raise ReferenceSchemaError(f"{name} contains NaN/Inf")
    cd = np.asarray(clean_D)
    if cd.shape != (H, W):
        raise ReferenceSchemaError(
            f"clean_D shape mismatch: got {cd.shape}, expected {(H, W)}"
        )
    if not np.isfinite(cd).all():
        raise ReferenceSchemaError("clean_D contains NaN/Inf")


def validate_simulation_request(
    ref: dict[str, Any],
    state: LatentState,
    *,
    raise_on_forbidden: bool = True,
) -> dict[str, Any]:
    """Run the common preflight gate without RNG draws or filesystem writes."""
    setup = validate_setup_params(state.get_setup())
    state.setup = setup
    H, W = validate_reference_schema(ref)
    architecture_profile = validate_forward_profile_supported(setup)
    architecture_snapshot = architecture_parameter_snapshot(setup)
    validate_detector_operating_state(state)
    if bool(getattr(setup, "enable_zernike", False)):
        raise Phase4DeferredError(
            "D1 deferred: enable_zernike=True is not available yet. "
            "Calibrate effective PSF first; use request_zernike_wavefront() "
            "for the policy error."
        )
    beam_bad = validate_beam_semantics(setup)
    if beam_bad:
        raise Phase4PolicyError("; ".join(beam_bad))
    # Resolves both source interactions and fails closed on unavailable pump or
    # calibrated-source parameters before simulation/output.
    operating_point = state.spdc_operating_point(relative_power_fluctuation=0.0)
    validity = check_forbidden(state, fold_spdc_into_optical=False)
    if raise_on_forbidden:
        validity.raise_if_invalid()
    return {
        "shape": (H, W),
        "setup": setup,
        "architecture_profile": architecture_profile,
        "architecture_parameters": architecture_snapshot,
        "operating_point": operating_point,
        "validity": validity,
    }


def visibility_observables(
    vis: dict[str, Any] | None,
    operating_point: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compact detector-plane moment and visibility record for sweeps/runs."""
    vis = dict(vis or {})
    operating_point = dict(operating_point or {})
    N1 = vis.get("N1_mean")
    N2 = vis.get("N2_mean")
    C12 = vis.get("C12_abs_mean")
    v_formula = None
    if N1 is not None and N2 is not None and C12 is not None:
        denominator = float(N1) + float(N2)
        v_formula = 0.0 if denominator <= 0.0 else 2.0 * float(C12) / denominator
    reported = vis.get("visibility_fixed_combiner_mean", vis.get("V_eff"))
    return {
        "nbar_1": operating_point.get("mean_photons_per_mode_NL1"),
        "nbar_2": operating_point.get("mean_photons_per_mode_NL2"),
        "gain_r_1": operating_point.get("spdc_gain_r_NL1"),
        "gain_r_2": operating_point.get("spdc_gain_r_NL2"),
        "gain_regime_1": operating_point.get("regime_NL1"),
        "gain_regime_2": operating_point.get("regime_NL2"),
        "N_1": N1,
        "N_2": N2,
        "C_12_abs": C12,
        "g_1": vis.get("intrinsic_g1_mean"),
        "V_50_50": v_formula,
        "V_50_50_from_moments": v_formula,
        "interference_image_visibility_mean": reported,
        "visibility_equation": "2*abs(C_12)/(N_1+N_2)",
        "arbitrary_visibility_coefficient_used": False,
    }


def simulate_forward(
    ref: dict[str, Any],
    state: LatentState | None = None,
    *,
    profile: str | Path | None = None,
    seed: int | None = None,
    detector_noise: bool = True,
    raise_on_forbidden: bool = True,
    require_canonical: bool = False,
) -> SimResult:
    """Single forward model for noisy and clean twins (V-P0-04).

    Uses the same optical path (``_optical_intensity``): SLM, blur_model,
    gain, magnification, coordinates. When ``detector_noise=False``, frames
    are ideal optical ``I±`` (no shot / read / dark / EM noise). ``clean_D``
    is always the mid-frame optical ``D_ideal`` from that same path.

    ``require_canonical=True`` (CFG-P0-00) refuses any setup that is not the
    frozen approved system and records the canonical hash in the manifest.
    """
    # Task 9 order starts with profile/YAML selection.
    if profile is not None:
        selected_setup = resolve_setup_selection(profile)
        if state is not None and state.setup is not None:
            raise ConfigError("Pass either state.setup or profile, not both")
        if state is None:
            state = apply_preset(
                "well_aligned", setup=selected_setup,
                photon_counting=detector_mode_is_photon_counting(
                    selected_setup.detector_mode
                ),
            )
        else:
            state = replace(state, setup=selected_setup)
    state = state or apply_preset("low_flux_pc")
    preflight = validate_simulation_request(
        ref, state, raise_on_forbidden=raise_on_forbidden
    )
    H, W = preflight["shape"]
    setup = preflight["setup"]
    architecture_profile = preflight["architecture_profile"]
    architecture_snapshot = preflight["architecture_parameters"]
    preflight_operating_point = preflight["operating_point"]
    validity = preflight["validity"]
    # Hash the setup as presented, before the forward can mutate derived fields.
    run_config_hash = hash_config(setup)
    canonical_hashes: dict[str, str] | None = None
    if require_canonical:
        assert_canonical_setup(setup)
        canonical_hashes = verify_canonical_file_hashes(setup)
    # V-P1-07: detector_mode on setup is authoritative; mirror LatentState flag.
    det_mode = resolve_detector_mode(setup)
    state.photon_counting = detector_mode_is_photon_counting(det_mode)

    seed_value = int(seed if seed is not None else ref.get("seed", 42))
    # V-P0-12: independent SeedSequence streams (optics / detector FP / ±).
    rngs = spawn_qiup_rngs(seed_value)
    rng_optics = rngs["rng_optics"]
    rng_detector = rngs["rng_detector"]
    rng_plus = rngs["rng_plus"]
    rng_minus = rngs["rng_minus"]
    Nf = int(state.num_frames)
    phase_schedule = resolved_phase_step_schedule(setup, num_frames=Nf)
    prnu, hot = _fixed_patterns((H, W), state, rng_detector)

    Yp = np.zeros((Nf, H, W), dtype=np.float32)
    Ym = np.zeros((Nf, H, W), dtype=np.float32)
    ideal_plus_stack = np.zeros((Nf, H, W), dtype=np.float32)
    ideal_minus_stack = np.zeros((Nf, H, W), dtype=np.float32)
    clean_stack = np.zeros((Nf, H, W), dtype=np.float32)
    last_opt: dict[str, Any] = {}

    for n in range(Nf):
        step_phase = None if phase_schedule is None else float(phase_schedule[n])
        opt = _optical_intensity(
            ref,
            state,
            n,
            rng_optics,
            interferometric_phase_override=step_phase,
        )
        last_opt = opt
        I_plus = np.asarray(opt["I_plus"], dtype=np.float32)
        I_minus = np.asarray(opt["I_minus"], dtype=np.float32)
        ideal_plus_stack[n] = I_plus
        ideal_minus_stack[n] = I_minus
        clean_stack[n] = np.asarray(opt["D_ideal"], dtype=np.float32)
        if detector_noise:
            Yp[n] = _detector_pc_frame(
                I_plus, state, rng_plus, prnu=prnu, hot=hot
            )
            Ym[n] = _detector_pc_frame(
                I_minus, state, rng_minus, prnu=prnu, hot=hot
            )
        else:
            Yp[n] = I_plus
            Ym[n] = I_minus

    D = (Yp - Ym).astype(np.float32)
    mid = Nf // 2
    detector_parameters = resolved_detector_parameters(state)
    selected_output_port = str(detector_parameters["selected_output_port"])
    if selected_output_port == "minus":
        ideal_signal_image = ideal_minus_stack[mid].copy()
        detector_signal_image = Ym[mid].copy()
    else:
        ideal_signal_image = ideal_plus_stack[mid].copy()
        detector_signal_image = Yp[mid].copy()
    # The clean detector reference is the same selected optical port with all
    # detector nuisance stages bypassed. clean_D remains the ideal balanced
    # difference for the historical interface.
    clean_reference_image = ideal_signal_image.copy()
    sample_dots = detector_signal_image
    clean_D = clean_stack[mid].astype(np.float32)

    reconstructed_phase_rad = None
    phase_reconstruction_meta: dict[str, Any] = {
        "enabled": False,
        "reason": "phase_step_protocol_off",
    }
    if phase_schedule is not None:
        rec = reconstruct_phase_from_phase_steps(D, phase_schedule)
        reconstructed_phase_rad = rec["phase_rad"]
        phase_reconstruction_meta = {
            "enabled": True,
            "method": rec["method"],
            "reference": rec["reference"],
            "n_steps": rec["n_steps"],
            "phases_rad": rec["phases_rad"],
            "input": "frames_D",
            "uses_clean_reference": False,
            "uses_forward_phi_eff": False,
            "modulation_mean": float(np.mean(rec["modulation"])),
            "snr_mean": float(np.mean(rec["snr"])),
            "residual_rms_mean": float(np.mean(rec["residual_rms"])),
        }

    dphi = last_opt.get("delta_phi")
    dphi_std = float(np.std(dphi)) if dphi is not None else 0.0
    setup = state.get_setup()
    fwd = last_opt.get("resolved_forward")
    fwd_audit: dict[str, Any] = {}
    if isinstance(fwd, ResolvedForward):
        fwd_audit = {
            "setup_id": fwd.setup_id,
            "w_px": float(fwd.w_px),
            "pointing_rms_x": float(fwd.pointing_rms_x),
            "pointing_rms_y": float(fwd.pointing_rms_y),
            "sigma_spdc_px": float(fwd.sigma_spdc_px),
            "imaging_mode": str(getattr(fwd, "imaging_mode", "")),
            "correlation_type": str(getattr(fwd, "correlation_type", "")),
            "sigma_diffraction_px": float(fwd.sigma_diffraction_px),
            "defocus_span_px": float(fwd.defocus_span_px),
            "V0": float(fwd.V0),
            "derived": dict(fwd.derived),
        }

    actual_operating_point = last_opt.get(
        "spdc_operating_point", preflight_operating_point
    )
    vis_summary = visibility_observables(
        last_opt.get("vis", {}), actual_operating_point
    )
    support_status = physical_model_support_status(
        setup, actual_operating_point
    )
    T_eff = np.asarray(last_opt.get("T_eff"), dtype=np.float64)
    phi_eff = np.asarray(last_opt.get("phi_eff"), dtype=np.float64)
    effective_object_transfer = {
        "object_type": getattr(setup, "object_type", None),
        "object_device": getattr(setup, "object_device", None),
        "object_mode": getattr(setup, "object_mode", None),
        "interaction_count": getattr(setup, "object_interaction_count", None),
        "intensity_transmission_min": float(np.min(T_eff)),
        "intensity_transmission_max": float(np.max(T_eff)),
        "intensity_transmission_mean": float(np.mean(T_eff)),
        "phase_rad_min": float(np.min(phi_eff)),
        "phase_rad_max": float(np.max(phi_eff)),
        "phase_rad_mean": float(np.mean(phi_eff)),
        "transfer_equation": (
            last_opt.get("qiup_spatial_transfer", {}) or {}
        ).get("object_transfer_equation"),
    }
    pipeline_stage_status = []
    for order, stage in enumerate(THREE_CONFIGURATION_PIPELINE_STAGES, start=1):
        status = "completed"
        if stage == "analog_emccd_detector" and not detector_noise:
            status = "bypassed_for_clean_reference"
        elif stage == "validated_atomic_output_transaction":
            status = "not_requested_in_memory"
        pipeline_stage_status.append(
            {"order": order, "stage": stage, "status": status}
        )

    meta = {
        "photon_counting": bool(state.photon_counting),
        "detector_mode": det_mode,
        "num_frames": Nf,
        "detector_noise": bool(detector_noise),
        "seed": seed_value,
        "rng_streams": list(QIUP_RNG_STREAM_NAMES),
        "rng_mixing": "SeedSequence.spawn(optics,detector,plus,minus)",
        "clean_D_source": "optical_forward_D_ideal",
        "clean_D_frame_index": int(mid),
        "primary_scientific_output": "detector_signal_image",
        "selected_output_port": selected_output_port,
        "selected_optical_quantity": f"I_{selected_output_port}",
        "detected_field": "signal",
        "idler_directly_detected": False,
        "ideal_signal_image_source": "same_forward_selected_port_before_detector",
        "clean_reference_source": "same_forward_selected_port_detector_nuisances_disabled",
        "clean_reference_role": "same_model_ideal_not_experimental_ground_truth",
        "phase_step_acquisition": {
            "enabled": phase_schedule is not None,
            "schedule_rad": (
                None if phase_schedule is None else [float(p) for p in phase_schedule]
            ),
            "num_frames_is_not_a_protocol": True,
        },
        "phase_reconstruction": phase_reconstruction_meta,
        "detector_parameters": detector_parameters,
        "ideal_signal_image_sha256": hash_array(ideal_signal_image),
        "detector_signal_image_sha256": hash_array(detector_signal_image),
        "clean_reference_image_sha256": hash_array(clean_reference_image),
        "latents_raw": state.as_dict(),
        "latents_effective": state.effective(),
        "physical_report": state.physical_report,
        "pipeline_execution_order": list(THREE_CONFIGURATION_PIPELINE_STAGES),
        "pipeline_stage_status": pipeline_stage_status,
        "output_transaction": {
            "mode": "in_memory",
            "atomic_commit": "not_requested",
        },
        "physical_model_support": support_status,
        "physics_policy_id": getattr(setup, "physics_policy_id", None),
        "implemented": support_status["implemented"],
        "software_tested": support_status["software_tested"],
        "experimentally_validated": support_status["experimentally_validated"],
        "vis": last_opt.get("vis", {}),
        "visibility_observables": vis_summary,
        "effective_object_transfer": effective_object_transfer,
        "signed_optical_path_difference_mm": (
            last_opt.get("vis", {}) or {}
        ).get("optical_path_difference_mm"),
        "interference_phase_rad": last_opt.get("theta"),
        "eta_overlap": last_opt.get("eta_overlap"),
        "delta_x_px": last_opt.get("delta_x_px"),
        "delta_y_px": last_opt.get("delta_y_px"),
        "couple_pointing_overlap_visibility": last_opt.get(
            "couple_pointing_overlap_visibility"
        ),
        "use_physical_two_beam": last_opt.get("use_physical_two_beam"),
        "use_field_overlap_visibility_map": last_opt.get(
            "use_field_overlap_visibility_map"
        ),
        "path1_intensity": last_opt.get("path1_intensity"),
        "path2_intensity": last_opt.get("path2_intensity"),
        "signal_path1_transmission": last_opt.get("signal_path1_transmission"),
        "signal_path2_transmission": last_opt.get("signal_path2_transmission"),
        "spdc_operating_point": actual_operating_point,
        "spdc_physics": last_opt.get("spdc_physics", {}),
        "paper_detector_path": det_mode,
        "sig_opt": last_opt.get("sig_opt"),
        "sig_spdc": last_opt.get("sig_spdc"),
        "blur_model": last_opt.get("blur_model"),
        "sigma_total_px": last_opt.get("sigma_total_px"),
        "imaging_mode": last_opt.get("imaging_mode"),
        "architecture_profile": architecture_profile.to_dict(),
        "setup_selection": setup_selection_info(setup),
        "architecture_parameters": architecture_snapshot,
        "correlation_type": last_opt.get("correlation_type"),
        "qiup_spatial_transfer": last_opt.get("qiup_spatial_transfer", {}),
        "signal_filter": last_opt.get("signal_filter", {}),
        "etalon_fringes": last_opt.get("etalon_fringes", {"enabled": False}),
        "sigma_diffraction_px": last_opt.get("sigma_diffraction_px"),
        "sigma_defocus_px": last_opt.get("sigma_defocus_px"),
        "delta_z_um_used": last_opt.get("delta_z_um_used"),
        "lens_rings": last_opt.get("lens_rings", {"enabled": False}),
        "delta_phi_rms": float(getattr(state, "delta_phi_rms", 0.0) or 0.0),
        "delta_phi_corr_px": float(getattr(state, "delta_phi_corr_px", 12.0) or 12.0),
        "delta_phi_realized_std": dphi_std,
        # S4 / S3 stage documentation
        "optical_stages": list(OPTICAL_STAGES),
        "detector_stages": (
            list(detector_stages_for_setup(setup)) if detector_noise else []
        ),
        # S2 gain semantics (never conflate)
        "s_gain_legacy_raw": float(state.s_gain),
        "s_gain_legacy_effective": float(state.effective()["s_gain"]),
        "s_gain_spdc_regime": float(state.s_gain),  # deprecated compatibility alias
        "em_gain_GEM": float(state.em_gain),
        "gain_semantics": dict(GAIN_SEMANTICS),
        # S1 / S7 / S8 / S9 vocab snapshots (defaults; not mutated here)
        "coupling_weights": dict(COUPLING_WEIGHTS),
        "visibility_defaults": {k: float(v) for k, v in VISIBILITY_DEFAULTS.items()},
        "psf_defaults": dict(PSF_DEFAULTS),
        "slm_taxonomy_layers": list(SLM_TAXONOMY.keys()),
        "cluster_contracts": CLUSTER_CONTRACTS,
        "mean_hits_plus": float(Yp.mean()),
        "F2": (1.0 if state.photon_counting else 2.0) if detector_noise else 1.0,
        "forbidden_ok": validity.ok,
        "violations": validity.violations,
        "covariance_mode": False,
        "model_phase": "phase3_physics_upgrades",
        "setup_id": setup.setup_id,
        "model_version": setup.model_version,
        "physics_tier": setup.physics_tier,
        "setup": setup.to_dict(),
        "resolved_forward": fwd_audit,
        "w_px": last_opt.get("w_px"),
        "coupling_weights_effective": state.coupling_table(),
        "coupling_profile_id": setup.coupling_profile_id,
        "emccd_model": setup.emccd_model,
        "read_noise_model": setup.read_noise_model,
        "read_noise_e": read_noise_e_from_setup(setup, state.effective()["s_det"]),
        "read_noise_input_referred_e": (
            read_noise_e_from_setup(setup, state.effective()["s_det"])
            / max(float(state.em_gain), 1.0)
        ),
        "T_sensor_C_effective": sensor_temperature_C(
            setup, state.effective()["s_cool"]
        ),
        "dark_rate_e_per_px_s": dark_rate_from_setup(
            setup, state.effective()["s_cool"]
        ),
        "hot_pixel_model": setup.hot_pixel_model,
        "sampling_regime": setup.sampling_regime,
        "sigma_blur_extra_px": last_opt.get("sigma_blur_extra_px"),
        "image_scale": last_opt.get("image_scale"),
        "image_rotation_deg": last_opt.get("image_rotation_deg"),
        "slm_misreg_x_px": last_opt.get("slm_misreg_x_px"),
        "slm_misreg_y_px": last_opt.get("slm_misreg_y_px"),
        "phase4_policy_ids": list(PHASE4_POLICY.keys()),
        "phase4_governance": "defer_reject_lock",
        "zernike_enabled": False,  # D1 always deferred in this model_version
        "coupling_weights_role": COUPLING_WEIGHTS_ROLE,
        "lemos_reference_role": LEMOS_REFERENCE.get("role"),
    }
    # V-P0-13 / Comment 7.3: mandatory run provenance on every forward.
    provenance = build_run_provenance(
        seed=seed_value,
        setup=setup,
        ref=ref,
        include_package_hashes=True,
        rng_streams=QIUP_RNG_STREAM_NAMES,
        arrays={
            "frames_plus": Yp,
            "frames_minus": Ym,
            "frames_D": D,
            "clean_D": clean_D,
        },
        extra={
            "rng_streams": list(QIUP_RNG_STREAM_NAMES),
            "rng_mixing": "SeedSequence.spawn(optics,detector,plus,minus)",
            "detector_noise": bool(detector_noise),
            "num_frames": Nf,
            "blur_model_resolved": last_opt.get("blur_model"),
            "sigma_total_px": last_opt.get("sigma_total_px"),
            "require_canonical": bool(require_canonical),
            "is_canonical_config": bool(is_canonical_setup(setup)),
            "config_hash": run_config_hash,
            "selected_output_port": selected_output_port,
            "selected_optical_quantity": f"I_{selected_output_port}",
            "detected_field": "signal",
            "idler_directly_detected": False,
            "ideal_signal_image_sha256": hash_array(ideal_signal_image),
            "detector_signal_image_sha256": hash_array(detector_signal_image),
            "clean_reference_image_sha256": hash_array(clean_reference_image),
            "pipeline_execution_order": list(THREE_CONFIGURATION_PIPELINE_STAGES),
            "physical_model_support": support_status,
            "physics_policy_id": getattr(setup, "physics_policy_id", None),
            "pump_balance": {
                key: actual_operating_point.get(key) for key in (
                    "pump_balance_mode", "P_target_1_mW", "P_target_2_mW",
                    "P_actual_1_mW", "P_actual_2_mW", "delta_P", "abs_delta_P",
                    "hidden_equalization_applied",
                )
            },
            "source_input": {
                key: actual_operating_point.get(key) for key in (
                    "source_input_type", "nbar_requested_NL1", "nbar_requested_NL2",
                    "nbar_resolved_NL1", "nbar_resolved_NL2", "measured_power_W",
                    "photon_rate_per_s", "source_conversion_method",
                    "source_input_calibration_id",
                )
            },
            "qpm": (last_opt.get("spdc_physics", {}) or {}).get(
                "qpm_interaction_1"
            ),
            "qpm_interaction_2": (last_opt.get("spdc_physics", {}) or {}).get(
                "qpm_interaction_2"
            ),
            "object_device": getattr(setup, "object_device", None),
            "object_mode": getattr(setup, "object_mode", None),
            "signal_filter": last_opt.get("signal_filter", {}),
            "implemented": support_status["implemented"],
            "software_tested": support_status["software_tested"],
            "experimentally_validated": support_status[
                "experimentally_validated"
            ],
            **(
                canonical_config_audit(verified=True, hashes=canonical_hashes)
                if canonical_hashes is not None
                else {}
            ),
        },
    )
    meta["provenance"] = provenance
    flatten_execution_evidence(meta, provenance)
    # Comment 7.4 — requested vs resolved configuration + derived quantities.
    config_evidence = build_configuration_evidence(
        setup,
        state=state,
        operating_point=meta.get("spdc_operating_point"),
        resolved_forward=fwd_audit,
        last_opt=last_opt if isinstance(last_opt, dict) else {},
        physical_report=meta.get("physical_report"),
    )
    meta["configuration_evidence"] = config_evidence
    meta["requested_configuration"] = config_evidence["requested_configuration"]
    meta["resolved_configuration"] = config_evidence["resolved_configuration"]
    meta["derived_physical_quantities"] = config_evidence[
        "derived_physical_quantities"
    ]
    meta["pump_power_laser_mW"] = config_evidence.get("pump_power_laser_mW")
    meta["pump_power_NL1_mW"] = config_evidence.get("pump_power_NL1_mW")
    meta["pump_power_NL2_mW"] = config_evidence.get("pump_power_NL2_mW")
    meta["nbar_NL1"] = config_evidence.get("nbar_NL1")
    meta["nbar_NL2"] = config_evidence.get("nbar_NL2")
    provenance["configuration_evidence"] = config_evidence
    provenance["requested_configuration"] = config_evidence[
        "requested_configuration"
    ]
    provenance["resolved_configuration"] = config_evidence[
        "resolved_configuration"
    ]
    provenance["derived_physical_quantities"] = config_evidence[
        "derived_physical_quantities"
    ]
    # Comment 7.5 — object hashes / durable reference on every run meta.
    object_evidence = build_object_evidence(ref)
    meta["object_evidence"] = object_evidence
    provenance["object_evidence"] = object_evidence
    provenance["reference_hashes"] = object_evidence.get("hashes") or provenance.get(
        "reference_hashes"
    )
    # Extra compatibility / audit mirrors.
    meta["model_status"] = provenance["model_status"]
    meta["verified_digital_twin"] = provenance["verified_digital_twin"]
    meta["release_version"] = provenance["release_version"]
    meta["hash_algorithm"] = provenance["hash_algorithm"]
    meta["require_canonical"] = bool(require_canonical)
    meta["is_canonical_config"] = bool(is_canonical_setup(setup))
    if canonical_hashes is not None:
        meta.update(canonical_config_audit(verified=True, hashes=canonical_hashes))
    # Prefer setup-authoritative blur/detector already present; keep aliases.
    if meta.get("blur_model") is None:
        meta["blur_model"] = provenance["blur_model"]
    missing_ev = assert_execution_evidence(meta)
    if missing_ev:
        raise RuntimeError(
            f"Comment 7.3: run meta missing execution evidence keys: {missing_ev}"
        )
    return SimResult(
        frames_plus=Yp,
        frames_minus=Ym,
        frames_D=D,
        sample_dots=sample_dots,
        meta=meta,
        clean_D=clean_D,
        state=state,
        validity=validity,
        ideal_signal_image=ideal_signal_image,
        detector_signal_image=detector_signal_image,
        clean_reference_image=clean_reference_image,
        reconstructed_phase_rad=reconstructed_phase_rad,
        selected_output_port=selected_output_port,
    )


def simulate_sequence(
    ref: dict[str, Any],
    state: LatentState | None = None,
    *,
    seed: int | None = None,
    raise_on_forbidden: bool = True,
    detector_noise: bool = True,
    require_canonical: bool = False,
) -> SimResult:
    """Run N-frame QIUP acquisition in analog (thesis) or photon-counting mode.

    Covariance reconstruction is intentionally omitted — thesis/DL path stops at
    continuous measured frames (typically N=1 hologram I+).

    ``clean_D`` uses the same optical forward as the measured frames
    (V-P0-04); set ``detector_noise=False`` for a fully clean optical twin.

    Meta includes V-P0-13 provenance (seed, blur/gain/detector modes,
    ``config_hash``, ``object_hash``, timestamp, environment).
    """
    return simulate_forward(
        ref,
        state,
        seed=seed,
        detector_noise=detector_noise,
        raise_on_forbidden=raise_on_forbidden,
        require_canonical=require_canonical,
    )


# ---------------------------------------------------------------------------
# Cluster sweeps (one latent at a time, others held at well-aligned baseline)
# ---------------------------------------------------------------------------


# ==============================================================================
# ANALYSIS + PLOTS
# Role: tooling
# Models: sweep one latent; inspect clean vs result
# Provenance: arch
# ==============================================================================

def sweep_cluster(
    ref: dict[str, Any],
    cluster_id: str,
    strengths: list[float] | None = None,
    *,
    num_frames: int = 64,
    seed: int = 0,
) -> list[SimResult]:
    """Sweep one cluster latent; keep others near well-aligned; always PC."""
    if cluster_id not in CLUSTERS:
        raise KeyError(cluster_id)
    latent_key = CLUSTERS[cluster_id]["latent"]
    if strengths is None:
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    results: list[SimResult] = []
    for i, s in enumerate(strengths):
        st = apply_preset("low_flux_pc", num_frames=num_frames)
        setattr(st, latent_key, float(np.clip(s, 0.0, 1.0)))
        # F5 soft repair for thermal sweep
        if latent_key == "s_thermal" and s > 0.55:
            st.num_frames = max(st.num_frames, 200)
            st.exposure_s = max(st.exposure_s, 0.25)
        # Don't raise on F3 during intentional align sweep — document instead
        res = simulate_sequence(
            ref, st, seed=seed + 17 * i, raise_on_forbidden=False
        )
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_clean_and_result(
    result: SimResult,
    *,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    frame = result.sample_dots
    clean = result.clean_D
    measured = result.frames_plus[0]
    pc = bool(result.meta.get("photon_counting", False))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].imshow(normalize(clean), cmap="Blues_r", vmin=0, vmax=1)
    axes[0].set_title("Clean D (interference)", fontsize=10)
    if pc:
        axes[1].imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"Raw PC frame (hits={frame.mean():.3f})", fontsize=10)
        axes[2].imshow(normalize(np.abs(result.frames_D.mean(axis=0))), cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("|⟨D⟩| (no cov mode)", fontsize=10)
    else:
        axes[1].imshow(normalize(measured), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Measured I+ (ADU)", fontsize=10)
        axes[2].imshow(normalize(np.abs(result.frames_D[0])), cmap="Blues_r", vmin=0, vmax=1)
        axes[2].set_title("|D| = |I+−I−|", fontsize=10)
    for ax in axes:
        ax.axis("off")
    V = result.meta.get("vis", {}).get("V_eff", float("nan"))
    fig.suptitle(
        title
        or f"QIUP  |  V_eff={V:.3f}  |  N={result.meta['num_frames']}  |  F²={result.meta['F2']}",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print("Saved ->", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return None


def plot_cluster_sweep(
    results: list[SimResult],
    cluster_id: str,
    strengths: list[float],
    *,
    save_path: str | None = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    info = CLUSTERS[cluster_id]
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, (res, s) in enumerate(zip(results, strengths)):
        pc = bool(res.meta.get("photon_counting", False))
        if pc:
            axes[0, i].imshow(res.sample_dots, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            bottom = normalize(np.abs(res.frames_D.mean(axis=0)))
            bot_label = "|⟨D⟩|"
        else:
            axes[0, i].imshow(normalize(res.frames_plus[0]), cmap="gray", vmin=0, vmax=1)
            bottom = normalize(np.abs(res.frames_D[0]))
            bot_label = "|D|"
        axes[0, i].set_title(f"{info['latent']}={s:.2f}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(bottom, cmap="hot", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("I+ / raw frame", fontsize=9)
            axes[1, i].set_ylabel(bot_label, fontsize=9)

    fig.suptitle(
        f"{cluster_id}: {info['name']} — measured intensity (no covariance)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print("Saved ->", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return None


# ==============================================================================
# DATASET GENERATION (single-frame, paper-1)
# Role: data generation
# Models: setup profile -> labeled (object->hologram) dataset
# Provenance: arch
# ==============================================================================

class DatasetExistsError(FileExistsError):
    """Output dataset directory is non-empty (V-P0-14). Refuse reuse/corruption."""


def _dir_is_nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def allocate_dataset_run_dir(base_dir: str | Path) -> Path:
    """Create a fresh empty run folder: ``base/runs/YYYY_MM_DD_NNN`` (V-P0-14).

    Uses exclusive ``mkdir`` so concurrent writers cannot share a directory.
    """
    from datetime import datetime

    runs_root = Path(base_dir) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y_%m_%d")
    for n in range(1, 10_000):
        candidate = runs_root / f"{day}_{n:03d}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise DatasetExistsError(
        f"Could not allocate a free run directory under {runs_root}"
    )


def try_sample_latents(
    rng: np.random.Generator,
    *,
    prefer_pc: bool = True,
    max_tries: int = 40,
    regime: str = "lab",
    setup: SetupParams | None = None,
) -> LatentState | None:
    """Draw a valid latent or skip (V-P0-15).

    Returns ``None`` on ``SamplingExhaustedError`` so dataset writers can
    skip the sample. Never substitutes ``low_flux_pc`` / another regime.
    """
    try:
        return sample_latents(
            rng,
            prefer_pc=prefer_pc,
            max_tries=max_tries,
            regime=regime,
            setup=setup,
        )
    except SamplingExhaustedError:
        return None


def prepare_dataset_output_dir(
    out_dir: str | Path,
    *,
    create_run_subdir: bool = True,
) -> Path:
    """Resolve a writable empty dataset directory (V-P0-14).

    create_run_subdir=True (recommended)
        Always allocate ``out_dir/runs/YYYY_MM_DD_NNN/`` (never reuse).
    create_run_subdir=False
        Use ``out_dir`` directly; raise ``DatasetExistsError`` if it exists
        and is non-empty. Empty/missing dirs are created.
    """
    base = Path(out_dir)
    if create_run_subdir:
        return allocate_dataset_run_dir(base)

    if base.exists():
        if base.is_file():
            raise DatasetExistsError(
                f"Dataset output path is a file, not a directory: {base}"
            )
        if _dir_is_nonempty(base):
            raise DatasetExistsError(
                f"Dataset output directory is not empty: {base}. "
                "Refusing reuse (V-P0-14). Pass create_run_subdir=True "
                "(recommended) or choose a fresh path."
            )
    else:
        base.mkdir(parents=True, exist_ok=False)
    return base


# Comment 5.1 — required artifacts inside a committed sample directory.
SAMPLE_REQUIRED_FILES: tuple[str, ...] = (
    "image.npy",
    "frames_minus.npy",
    "frames_D.npy",
    "clean_D.npy",
    "frames.npz",
    "metadata.json",
)


def purge_abandoned_sample_temps(samples_dir: Path) -> list[Path]:
    """Remove leftover ``.tmp_sample_*`` directories (Comment 5 / 5.5)."""
    removed: list[Path] = []
    if not samples_dir.is_dir():
        return removed
    for path in samples_dir.iterdir():
        if path.is_dir() and path.name.startswith(".tmp_sample_"):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


def cleanup_failed_sample_temp(temp_dir: Path | None) -> None:
    """Comment 5.5 — remove a failed temp sample; never leave ``sample_XXXX``.

    Always safe to call::

        try:
            ...
        except Exception:
            cleanup_failed_sample_temp(temp_dir)
            raise
    """
    if temp_dir is None:
        return
    path = Path(temp_dir)
    # Only delete temp names — never touch a committed sample_XXXX directory.
    if path.is_dir() and path.name.startswith(".tmp_sample_"):
        shutil.rmtree(path, ignore_errors=True)


DATASET_MANIFEST_FILENAME = "dataset_manifest.json"


def is_committed_sample_dir(sample_dir: Path) -> bool:
    """True if ``sample_XXXX`` exists with the full required artifact set."""
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        return False
    if not sample_dir.name.startswith("sample_"):
        return False
    if sample_dir.name.startswith(".tmp_sample_"):
        return False
    return all((sample_dir / name).is_file() for name in SAMPLE_REQUIRED_FILES)


def committed_sample_index(
    out_dir: Path,
    index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Comment 5.6 — keep only index rows that point at committed samples."""
    out_dir = Path(out_dir)
    committed: list[dict[str, Any]] = []
    for row in index:
        rel = str(row.get("sample_dir") or "")
        if not rel:
            continue
        sample_path = out_dir / rel
        if is_committed_sample_dir(sample_path):
            committed.append(row)
    return committed


def write_dataset_manifest(
    out_dir: Path,
    manifest: dict[str, Any],
    *,
    index: list[dict[str, Any]] | None = None,
) -> Path:
    """Comment 5.6 — write the dataset manifest last, indexing committed samples only.

    The on-disk name is ``dataset_manifest.json`` (dataset-level manifest).
    Call this only after every successful sample has been atomically committed
    and appended to the in-memory index.
    """
    out_dir = Path(out_dir)
    rows = committed_sample_index(out_dir, list(index if index is not None else manifest.get("samples") or []))
    payload = dict(manifest)
    payload["samples"] = rows
    payload["n_samples"] = len(rows)
    payload["manifest_indexes_committed_samples_only"] = True
    path = out_dir / DATASET_MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def make_sample_temp_dir(samples_dir: Path, sample_id: str) -> Path:
    """Allocate ``.tmp_sample_XXXX_<UUID>/`` for atomic sample writes (Comment 5.1)."""
    tmp_dir = samples_dir / f".tmp_{sample_id}_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=False, exist_ok=False)
    return tmp_dir


def atomic_commit_sample_dir(tmp_dir: Path, final_dir: Path) -> Path:
    """Comment 5.4 — commit a validated temp sample with ``os.replace``.

    Only call this **after** ``validate_sample_temp_before_commit`` succeeds::

        .tmp_sample_0001_<uuid>   →   sample_0001

    ``os.replace(temp_dir, final_dir)`` is the sole commit operation so readers
    never observe a partially written final ``sample_XXXX`` directory.
    """
    final_dir = Path(final_dir)
    tmp_dir = Path(tmp_dir)
    if not tmp_dir.is_dir():
        raise FileNotFoundError(f"Temp sample directory missing: {tmp_dir}")
    if not tmp_dir.name.startswith(".tmp_sample_"):
        raise ValueError(
            f"Comment 5.4: commit source must be .tmp_sample_XXXX_<UUID>/, "
            f"got {tmp_dir.name!r}"
        )
    if not str(final_dir.name).startswith("sample_"):
        raise ValueError(
            f"Comment 5.4: commit destination must be sample_XXXX/, "
            f"got {final_dir.name!r}"
        )
    if final_dir.exists():
        raise FileExistsError(f"Sample directory already exists: {final_dir}")
    # Atomic commit (same filesystem): temp name disappears; final name appears.
    os.replace(str(tmp_dir), str(final_dir))
    return final_dir


class SampleIntegrityError(RuntimeError):
    """Temp sample failed Comment 5.3 pre-commit validation."""


def required_files_exist(tmp_dir: Path) -> None:
    """Comment 5.3 — every required artifact must be present on disk."""
    tmp_dir = Path(tmp_dir)
    missing = [name for name in SAMPLE_REQUIRED_FILES if not (tmp_dir / name).is_file()]
    if missing:
        raise SampleIntegrityError(
            f"required_files_exist failed under {tmp_dir}: missing {missing}"
        )


def arrays_have_expected_shapes(
    tmp_dir: Path,
    *,
    expected_hw: tuple[int, int],
    expected_n_frames: int,
) -> dict[str, np.ndarray]:
    """Comment 5.3 — reload arrays and check shapes against the reference grid."""
    tmp_dir = Path(tmp_dir)
    H, W = int(expected_hw[0]), int(expected_hw[1])
    N = int(expected_n_frames)
    image = np.load(tmp_dir / "image.npy")
    frames_minus = np.load(tmp_dir / "frames_minus.npy")
    frames_D = np.load(tmp_dir / "frames_D.npy")
    clean_D = np.load(tmp_dir / "clean_D.npy")
    expected_stack = (N, H, W)
    for name, arr in (
        ("image.npy", image),
        ("frames_minus.npy", frames_minus),
        ("frames_D.npy", frames_D),
    ):
        if tuple(arr.shape) != expected_stack:
            raise SampleIntegrityError(
                f"arrays_have_expected_shapes: {name} got {arr.shape}, "
                f"expected {expected_stack}"
            )
    if tuple(clean_D.shape) != (H, W):
        raise SampleIntegrityError(
            f"arrays_have_expected_shapes: clean_D.npy got {clean_D.shape}, "
            f"expected {(H, W)}"
        )
    with np.load(tmp_dir / "frames.npz") as z:
        for key, expected in (
            ("frames_plus", expected_stack),
            ("frames_minus", expected_stack),
            ("frames_D", expected_stack),
            ("clean_D", (H, W)),
        ):
            if key not in z:
                raise SampleIntegrityError(
                    f"arrays_have_expected_shapes: frames.npz missing {key!r}"
                )
            got = tuple(np.asarray(z[key]).shape)
            if got != expected:
                raise SampleIntegrityError(
                    f"arrays_have_expected_shapes: frames.npz[{key!r}] "
                    f"got {got}, expected {expected}"
                )
    return {
        "image": np.asarray(image),
        "frames_minus": np.asarray(frames_minus),
        "frames_D": np.asarray(frames_D),
        "clean_D": np.asarray(clean_D),
    }


def arrays_are_finite(arrays: dict[str, np.ndarray]) -> None:
    """Comment 5.3 — reject NaN/Inf before commit (e.g. ``np.isfinite(frames_D).all()``)."""
    for name, arr in arrays.items():
        a = np.asarray(arr)
        if not np.isfinite(a).all():
            raise SampleIntegrityError(
                f"arrays_are_finite: {name} contains NaN/Inf"
            )


def metadata_is_valid(
    tmp_dir: Path,
    *,
    sample_id: str,
) -> dict[str, Any]:
    """Comment 5.3 — metadata.json must parse and carry required provenance keys."""
    path = Path(tmp_dir) / "metadata.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleIntegrityError(f"metadata_is_valid: cannot load {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SampleIntegrityError("metadata_is_valid: metadata.json root must be an object")
    if meta.get("sample_id") != sample_id:
        raise SampleIntegrityError(
            f"metadata_is_valid: sample_id mismatch "
            f"(meta={meta.get('sample_id')!r}, expected={sample_id!r})"
        )
    files = meta.get("files")
    if not isinstance(files, dict):
        raise SampleIntegrityError("metadata_is_valid: missing files map")
    for key, filename in (
        ("image", "image.npy"),
        ("frames_minus", "frames_minus.npy"),
        ("frames_D", "frames_D.npy"),
        ("clean_D", "clean_D.npy"),
        ("frames_npz", "frames.npz"),
        ("metadata", "metadata.json"),
    ):
        if files.get(key) != filename:
            raise SampleIntegrityError(
                f"metadata_is_valid: files[{key!r}] expected {filename!r}, "
                f"got {files.get(key)!r}"
            )
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        raise SampleIntegrityError("metadata_is_valid: missing provenance object")
    for hash_key in ("image_sha256", "clean_D_sha256"):
        if not prov.get(hash_key):
            raise SampleIntegrityError(
                f"metadata_is_valid: provenance missing {hash_key}"
            )
    # Comment 7.3 — enforce full execution evidence when the sample opts in
    # (official dataset writes always set these keys).
    if any(
        k in meta
        for k in ("version", "release_status", "source_hashes", "array_hashes")
    ):
        missing = assert_execution_evidence(meta)
        if missing:
            raise SampleIntegrityError(
                f"metadata_is_valid: Comment 7.3 missing keys: {missing}"
            )
    # Comment 7.5 — require object evidence when declared on the sample.
    obj_ev = meta.get("object_evidence")
    files_map = meta.get("files") if isinstance(meta.get("files"), dict) else {}
    if obj_ev is not None or files_map.get("object_T") or files_map.get("object_phi"):
        if not isinstance(obj_ev, dict):
            raise SampleIntegrityError(
                "metadata_is_valid: Comment 7.5 missing object_evidence"
            )
        hashes = obj_ev.get("hashes") or {}
        for map_key in ("T", "phi"):
            if not hashes.get(map_key) and not obj_ev.get("durable_object_reference"):
                raise SampleIntegrityError(
                    f"metadata_is_valid: Comment 7.5 object_evidence missing "
                    f"hash for {map_key} (and no durable_object_reference)"
                )
    return meta


def hashes_match(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    """Comment 5.3 — on-disk array hashes must match metadata provenance."""
    prov = metadata.get("provenance") or {}
    image_hash = hash_array(arrays["image"])
    clean_hash = hash_array(arrays["clean_D"])
    expected_image = str(prov.get("image_sha256") or "")
    expected_clean = str(prov.get("clean_D_sha256") or "")
    if image_hash != expected_image:
        raise SampleIntegrityError(
            f"hashes_match: image_sha256 mismatch "
            f"(file={image_hash}, meta={expected_image})"
        )
    if clean_hash != expected_clean:
        raise SampleIntegrityError(
            f"hashes_match: clean_D_sha256 mismatch "
            f"(file={clean_hash}, meta={expected_clean})"
        )


def validate_sample_temp_before_commit(
    tmp_dir: Path,
    *,
    sample_id: str,
    expected_hw: tuple[int, int],
    expected_n_frames: int,
) -> dict[str, Any]:
    """Comment 5.3 — full pre-rename gate for a temp sample directory.

    Order::

        required_files_exist()
        arrays_have_expected_shapes()
        arrays_are_finite()
        metadata_is_valid()
        hashes_match()
    """
    required_files_exist(tmp_dir)
    arrays = arrays_have_expected_shapes(
        tmp_dir,
        expected_hw=expected_hw,
        expected_n_frames=expected_n_frames,
    )
    arrays_are_finite(arrays)
    meta = metadata_is_valid(tmp_dir, sample_id=sample_id)
    hashes_match(arrays, meta)
    return meta


def write_sample_artifacts_to_temp(
    tmp_dir: Path,
    *,
    frames_plus: Any,
    frames_minus: Any,
    frames_D: Any,
    clean_D: Any,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Comment 5.2 — write the full sample file set into ``tmp_dir`` only.

    Artifacts written (never into the final ``sample_XXXX`` name)::

        image.npy
        frames_minus.npy
        frames_D.npy
        clean_D.npy
        frames.npz
        metadata.json

    Returns a mapping of logical name → path under ``tmp_dir``.

    Comment 5.7 — set module global ``SAMPLE_WRITE_FAULT_AFTER`` to an artifact
    filename to raise ``OSError`` immediately after that file is written
    (atomicity tests only).
    """
    tmp_dir = Path(tmp_dir)
    if not tmp_dir.is_dir():
        raise FileNotFoundError(f"Sample temp directory does not exist: {tmp_dir}")
    if not tmp_dir.name.startswith(".tmp_sample_"):
        raise ValueError(
            f"Comment 5.2: sample artifacts must be written under "
            f".tmp_sample_XXXX_<UUID>/, got {tmp_dir.name!r}"
        )

    image = np.asarray(frames_plus, dtype=np.float32)
    paths = {
        "image": tmp_dir / "image.npy",
        "frames_minus": tmp_dir / "frames_minus.npy",
        "frames_D": tmp_dir / "frames_D.npy",
        "clean_D": tmp_dir / "clean_D.npy",
        "frames_npz": tmp_dir / "frames.npz",
        "metadata": tmp_dir / "metadata.json",
    }
    np.save(paths["image"], image)
    _fault_after_sample_write("image.npy")
    np.save(paths["frames_minus"], np.asarray(frames_minus))
    _fault_after_sample_write("frames_minus.npy")
    np.save(paths["frames_D"], np.asarray(frames_D))
    _fault_after_sample_write("frames_D.npy")
    np.save(paths["clean_D"], np.asarray(clean_D))
    _fault_after_sample_write("clean_D.npy")
    np.savez_compressed(
        paths["frames_npz"],
        frames_plus=frames_plus,
        frames_minus=frames_minus,
        frames_D=frames_D,
        clean_D=clean_D,
    )
    _fault_after_sample_write("frames.npz")
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    _fault_after_sample_write("metadata.json")

    missing = [name for name in SAMPLE_REQUIRED_FILES if not (tmp_dir / name).is_file()]
    if missing:
        raise OSError(f"Incomplete temp sample write; missing: {missing}")
    return paths


# Comment 5.7 — when set to a SAMPLE_REQUIRED_FILES name, raise after that write.
SAMPLE_WRITE_FAULT_AFTER: str | None = None


def _fault_after_sample_write(artifact_name: str) -> None:
    """Raise after a named artifact write when fault-injection is armed."""
    if SAMPLE_WRITE_FAULT_AFTER is not None and SAMPLE_WRITE_FAULT_AFTER == artifact_name:
        raise OSError(f"simulated fault after writing {artifact_name}")


def generate_personal_dataset(
    ref: dict[str, Any],
    setup: SetupParams | str | Path | None = None,
    *,
    out_dir: str | Path,
    profile: str | Path | None = None,
    sweeps: dict[str, list[Any]] | None = None,
    base_preset: str = "well_aligned",
    num_frames: int = 16,
    seed0: int = 42,
    photon_counting: bool | None = None,
    create_run_subdir: bool = True,
    latent_sampling: str = "preset",
    sampling_regime: str = "lab",
    sample_max_tries: int = 40,
    require_canonical: bool = False,
) -> dict[str, Any]:
    """Phase-5.3: controlled synthetic dataset from a personal SetupParams.

    Each sample stores frames + full physical meta (setup, model_version, seed).
    Default sweeps: em_gain, delta_z_um, s_slm (checklist §5.3).

    V-P0-07: fail-closed. Forbidden latent/setup combinations raise
    ``InvalidSampleError`` before any sample file is written. Dataset
    generation always keeps forbidden-state enforcement enabled.

    V-P0-11: ``validate_reference_schema(ref)`` runs before the sweep; each
    sample is shape-checked again before write.

    V-P0-13: each sample is a folder with ``image.npy`` + ``metadata.json``
    carrying seed, blur/gain/detector modes, config/object hashes, timestamp.

    V-P0-14: never reuse a non-empty output directory. By default writes under
    ``out_dir/runs/YYYY_MM_DD_NNN/``. Set ``create_run_subdir=False`` only when
    ``out_dir`` is missing or empty (else ``DatasetExistsError``).

    V-P0-15: ``latent_sampling='random'`` uses ``sample_latents``; on
    ``SamplingExhaustedError`` the combo is **skipped** (never replaced by
    ``low_flux_pc`` or another regime).

    Comment 4.5 — ordering is fail-closed::

        load / receive configuration
        → validate reference + keys
        → validate sweep names
        → validate physical domain (overrides / canonical freezes)
        → ONLY THEN create run / samples directories

    Comment 5.1 — each sample is written under
    ``.tmp_sample_XXXX_<UUID>/`` and atomically renamed to ``sample_XXXX``
    only after all artifacts are on disk.

    Comment 5.3 — before rename: required files, shapes, finiteness,
    metadata validity, and hash checks must all pass.

    Comment 5.4 — commit is ``os.replace(.tmp_sample_XXXX_<uuid>, sample_XXXX)``
    and runs only after validation succeeds.

    Comment 5.5 — on any failure inside the sample write/validate/commit
    transaction, ``shutil.rmtree(temp_dir, ignore_errors=True)`` removes the
    temp directory so no incomplete ``sample_XXXX`` survives.

    Comment 5.6 — ``dataset_manifest.json`` is written last and indexes only
    atomically committed ``sample_XXXX`` directories.

    Comment 5.7 — fault injection after each artifact write must leave no
    final ``sample_XXXX`` directory (see ``tests/test_atomic_dataset_write.py``).
    """
    from itertools import product

    if profile is not None and setup is not None:
        raise ConfigError("Pass either setup or profile, not both")
    selected = profile if profile is not None else setup
    if selected is None:
        raise ConfigError("A setup or profile selection is required")
    setup = resolve_setup_selection(selected)
    validate_forward_profile_supported(setup)
    selected_yaml = isinstance(selected, (str, Path))
    configured_pc = detector_mode_is_photon_counting(setup.detector_mode)
    if photon_counting is None:
        photon_counting = configured_pc if selected_yaml else True
    elif selected_yaml and bool(photon_counting) != configured_pc:
        raise ConfigError(
            "photon_counting conflicts with the selected YAML detector_mode"
        )

    sampling = str(latent_sampling or "preset").strip().lower()
    if sampling not in {"preset", "random"}:
        raise ValueError(
            "latent_sampling must be 'preset' or 'random', "
            f"got {latent_sampling!r}"
        )

    # --- Comment 4.5: validate everything before any filesystem writes ---
    validate_reference_schema(ref)

    canonical_hashes: dict[str, str] | None = None
    if require_canonical:
        assert_canonical_setup(setup)
        canonical_hashes = verify_canonical_file_hashes(setup)
        if photon_counting:
            raise ConfigError(
                "CFG-P0-00: official canonical runs use detector_mode="
                "analog_emccd; photon_counting=True is a conflicting override."
            )

    if sweeps is None:
        sweeps = {
            "em_gain": [200.0, 400.0, 800.0],
            "delta_z_um": [0.0, 25.0, 50.0],
            "s_slm": [0.1, 0.35, 0.6],
        }

    # Comment 4.4 / 4.5: reject unknown sweep names before mkdir.
    validate_sweep_names(sweeps, context="generate_personal_dataset")

    keys = list(sweeps.keys())
    values = [list(sweeps[k]) for k in keys]

    # Physical-domain / canonical dry-run for every combo before mkdir.
    for combo in product(*values):
        params = dict(zip(keys, combo))
        setup_kwargs = setup_overrides_from_sweep(params)
        if require_canonical:
            reject_canonical_overrides(setup_kwargs)
            if "photon_counting" in params and bool(params["photon_counting"]):
                raise ConfigError(
                    "CFG-P0-00: cannot sweep photon_counting=True on an "
                    "official canonical analog_emccd run."
                )
        setup_dry = setup
        if setup_kwargs:
            setup_dry = setup.overridden(
                **setup_kwargs,
                require_canonical=require_canonical,
            )
        # Resolve source/gain and detector operating controls during the dry
        # run as well. Missing calibrated-source data therefore fails before
        # prepare_dataset_output_dir creates anything.
        dry_pc = bool(params.get("photon_counting", photon_counting))
        if dry_pc != detector_mode_is_photon_counting(setup_dry.detector_mode):
            setup_dry = setup_dry.overridden(
                detector_mode=detector_mode_from_photon_counting_flag(dry_pc)
            )
        dry_state = apply_preset(
            base_preset,
            num_frames=num_frames,
            photon_counting=dry_pc,
            setup=setup_dry,
            em_gain=float(params.get("em_gain", 800.0 if dry_pc else 250.0)),
        )
        for lat_key in LATENT_SWEEP_FIELDS:
            if lat_key in params and lat_key != "photon_counting":
                setattr(dry_state, lat_key, params[lat_key])
        validate_simulation_request(ref, dry_state, raise_on_forbidden=True)

    # V-P0-14: immutable output — only after config validation succeeds.
    requested_out_dir = Path(out_dir)
    out_dir = prepare_dataset_output_dir(
        requested_out_dir, create_run_subdir=create_run_subdir
    )
    samples_dir = out_dir / "samples"
    # Exclusive samples/ creation inside an already-empty run dir.
    samples_dir.mkdir(parents=False, exist_ok=False)
    purge_abandoned_sample_temps(samples_dir)

    # Comment 7.5 — persist exact object maps (+ SHA-256) for official evidence.
    object_evidence = save_object_evidence(ref, out_dir)

    dataset_config_hash = hash_config(setup)
    dataset_object_hash = (
        object_evidence.get("object_hash")
        or (hash_array(ref["M"]) if ref.get("M") is not None else None)
    )

    index: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    n = 0
    for combo_i, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))
        setup_i = setup
        # Report §7.24: every physical SetupParams key in the sweep must apply.
        setup_kwargs = setup_overrides_from_sweep(params)
        if require_canonical:
            reject_canonical_overrides(setup_kwargs)
            if "photon_counting" in params and bool(params["photon_counting"]):
                raise ConfigError(
                    "CFG-P0-00: cannot sweep photon_counting=True on an "
                    "official canonical analog_emccd run."
                )
        if setup_kwargs:
            setup_i = setup.overridden(
                **setup_kwargs,
                require_canonical=require_canonical,
            )

        seed = int(seed0 + 17 * combo_i)
        if sampling == "random":
            # V-P0-15: never fall back to a different regime on exhaustion.
            rng_lat = np.random.default_rng(seed)
            try:
                st = sample_latents(
                    rng_lat,
                    prefer_pc=photon_counting,
                    max_tries=int(sample_max_tries),
                    regime=sampling_regime,
                    setup=setup_i,
                )
            except SamplingExhaustedError as exc:
                skipped.append(
                    {
                        "sweep": params,
                        "seed": seed,
                        "reason": "sampling_exhausted",
                        "requested_state": dict(exc.requested_state),
                        "violations": list(exc.violations),
                    }
                )
                continue
            st.num_frames = int(num_frames)
            if "em_gain" in params:
                st.em_gain = float(params["em_gain"])
        else:
            st = apply_preset(
                base_preset,
                num_frames=num_frames,
                photon_counting=photon_counting,
                setup=setup_i,
                em_gain=float(
                    params.get("em_gain", 800.0 if photon_counting else 250.0)
                ),
            )
        for lat_key in LATENT_SWEEP_FIELDS:
            if lat_key in params and lat_key != "photon_counting":
                setattr(st, lat_key, params[lat_key])
        if "photon_counting" in params:
            st.photon_counting = bool(params["photon_counting"])
            st.setup = st.get_setup().overridden(
                detector_mode=detector_mode_from_photon_counting_flag(
                    st.photon_counting
                )
            )

        sample_id = f"sample_{n:04d}"
        # V-P0-07: validate before forward/save — never persist forbidden states.
        validity = check_state(st)
        if not validity.ok:
            raise InvalidSampleError(
                validity.violations,
                sample_context=f"{sample_id} sweep={params}",
            )

        # Dataset generation is fail-closed: always raise on forbidden.
        res = simulate_sequence(
            ref,
            st,
            seed=seed,
            raise_on_forbidden=True,
            require_canonical=require_canonical,
        )
        if not res.validity.ok:
            raise InvalidSampleError(
                res.validity.violations,
                sample_context=f"{sample_id} after_forward sweep={params}",
            )

        # V-P0-11: never write frames that disagree with the reference grid.
        validate_result_shapes(
            ref,
            frames_plus=res.frames_plus,
            frames_minus=res.frames_minus,
            frames_D=res.frames_D,
            clean_D=res.clean_D,
        )

        # V-P0-13 / Comment 5.1–5.2: write the full artifact set into
        # ``.tmp_sample_XXXX_<UUID>/``, then atomically rename to ``sample_XXXX``.
        sample_dir_final = samples_dir / sample_id
        tmp_dir = make_sample_temp_dir(samples_dir, sample_id)
        try:
            image = np.asarray(res.frames_plus, dtype=np.float32)
            setup_i_used = st.get_setup()
            provenance = dict(res.meta.get("provenance") or {})
            provenance.update(
                {
                    "sample_id": sample_id,
                    "sweep": params,
                    "preset": base_preset,
                    "seed0": int(seed0),
                    "image_sha256": hash_array(image),
                    "clean_D_sha256": hash_array(res.clean_D),
                    # Keep Comment 7.3 array_hashes in sync with saved files.
                    "array_hashes": hash_sim_arrays(
                        frames_plus=res.frames_plus,
                        frames_minus=res.frames_minus,
                        frames_D=res.frames_D,
                        clean_D=res.clean_D,
                    ),
                }
            )
            meta = {
                "sample_id": sample_id,
                "seed": seed,
                "version": provenance.get("version") or MODEL_VERSION,
                "release_status": provenance.get("release_status") or RELEASE_STATUS,
                "rng_scheme": provenance.get("rng_scheme"),
                "blur_model": getattr(setup_i_used, "blur_model", None),
                "gain_mode": getattr(setup_i_used, "spdc_gain_mode", None),
                "spdc_source_mode": getattr(setup_i_used, "spdc_source_mode", None),
                "architecture_profile": res.meta.get("architecture_profile"),
                "setup_selection": res.meta.get("setup_selection"),
                "detector_mode": getattr(setup_i_used, "detector_mode", None),
                "selected_output_port": res.selected_output_port,
                "selected_optical_quantity": res.meta.get("selected_optical_quantity"),
                "primary_scientific_output": res.meta.get("primary_scientific_output"),
                "detected_field": res.meta.get("detected_field"),
                "idler_directly_detected": res.meta.get("idler_directly_detected"),
                "detector_parameters": res.meta.get("detector_parameters"),
                "pipeline_execution_order": res.meta.get(
                    "pipeline_execution_order"
                ),
                "pipeline_stage_status": res.meta.get("pipeline_stage_status"),
                "output_transaction": {
                    "mode": "validated_temp_directory_then_atomic_rename",
                    "precommit_validation": True,
                    "commit_operation": "os.replace",
                },
                "physical_model_support": res.meta.get(
                    "physical_model_support"
                ),
                "implemented": res.meta.get("implemented"),
                "software_tested": res.meta.get("software_tested"),
                "experimentally_validated": res.meta.get(
                    "experimentally_validated"
                ),
                "visibility_observables": res.meta.get(
                    "visibility_observables"
                ),
                "effective_object_transfer": res.meta.get(
                    "effective_object_transfer"
                ),
                "ideal_signal_image_sha256": res.meta.get(
                    "ideal_signal_image_sha256"
                ),
                "detector_signal_image_sha256": res.meta.get(
                    "detector_signal_image_sha256"
                ),
                "clean_reference_image_sha256": res.meta.get(
                    "clean_reference_image_sha256"
                ),
                "config_hash": provenance.get("config_hash") or hash_config(setup_i_used),
                "object_hash": provenance.get("object_hash") or dataset_object_hash,
                "timestamp": provenance.get("timestamp"),
                "environment": provenance.get("environment"),
                "source_hashes": provenance.get("source_hashes"),
                "build_hash": provenance.get("build_hash"),
                "array_hashes": provenance.get("array_hashes"),
                "provenance": provenance,
                "sweep": params,
                "preset": base_preset,
                "photon_counting": bool(st.photon_counting),
                "num_frames": num_frames,
                "detector_noise": bool(res.meta.get("detector_noise", True)),
                "exposure_s": float(getattr(st, "exposure_s", 0.05)),
                "latents_raw": res.meta["latents_raw"],
                "latents_effective": res.meta["latents_effective"],
                "forbidden_ok": True,
                "violations": [],
                "setup_id": res.meta.get("setup_id"),
                "model_version": res.meta.get("model_version"),
                "physics_tier": res.meta.get("physics_tier"),
                "setup": res.meta.get("setup"),
                "vis": res.meta.get("vis"),
                "spdc_physics": res.meta.get("spdc_physics"),
                "spdc_operating_point": res.meta.get("spdc_operating_point"),
                "sig_spdc": res.meta.get("sig_spdc"),
                "sig_opt": res.meta.get("sig_opt"),
                "em_gain_GEM": res.meta.get("em_gain_GEM"),
                "s_gain_spdc_regime": res.meta.get("s_gain_spdc_regime"),
                "coupling_weights_effective": res.meta.get("coupling_weights_effective"),
                "coupling_profile_id": res.meta.get("coupling_profile_id"),
                "mean_hits_plus": res.meta.get("mean_hits_plus"),
                "phase5_id": MODEL_VERSION,
                "release_version": RELEASE_VERSION,
                "model_status": MODEL_STATUS,
                "verified_digital_twin": False,
                "configuration_evidence": res.meta.get("configuration_evidence"),
                "requested_configuration": res.meta.get("requested_configuration"),
                "resolved_configuration": res.meta.get("resolved_configuration"),
                "derived_physical_quantities": res.meta.get(
                    "derived_physical_quantities"
                ),
                "pump_power_laser_mW": res.meta.get("pump_power_laser_mW"),
                "pump_power_NL1_mW": res.meta.get("pump_power_NL1_mW"),
                "pump_power_NL2_mW": res.meta.get("pump_power_NL2_mW"),
                "nbar_NL1": res.meta.get("nbar_NL1"),
                "nbar_NL2": res.meta.get("nbar_NL2"),
                "object_evidence": object_evidence,
                "files": {
                    "image": "image.npy",
                    "frames_minus": "frames_minus.npy",
                    "frames_D": "frames_D.npy",
                    "clean_D": "clean_D.npy",
                    "frames_npz": "frames.npz",
                    "metadata": "metadata.json",
                    "object_T": object_evidence.get("files", {}).get("T"),
                    "object_phi": object_evidence.get("files", {}).get("phi"),
                    "object_M": object_evidence.get("files", {}).get("M"),
                    "object_manifest": object_evidence.get("manifest_file"),
                },
            }
            provenance["object_evidence"] = object_evidence
            meta["provenance"] = provenance
            flatten_execution_evidence(meta, provenance)
            missing_sample = assert_execution_evidence(meta)
            if missing_sample:
                raise RuntimeError(
                    f"Comment 7.3: sample {sample_id} missing evidence "
                    f"keys: {missing_sample}"
                )
            # Comment 5.2: all six artifacts land only under the temp directory.
            write_sample_artifacts_to_temp(
                tmp_dir,
                frames_plus=res.frames_plus,
                frames_minus=res.frames_minus,
                frames_D=res.frames_D,
                clean_D=res.clean_D,
                metadata=meta,
            )
            # Comment 5.3: verify files / shapes / finiteness / metadata / hashes
            # before the temp directory is renamed to sample_XXXX.
            H_ref, W_ref = reference_target_hw(ref)
            validate_sample_temp_before_commit(
                tmp_dir,
                sample_id=sample_id,
                expected_hw=(H_ref, W_ref),
                expected_n_frames=int(num_frames),
            )
            # Comment 5.4: only after validation — os.replace(temp → sample_XXXX).
            sample_dir = atomic_commit_sample_dir(tmp_dir, sample_dir_final)
        except Exception:
            # Comment 5.5: failed transaction — drop the temp dir; no sample_XXXX.
            cleanup_failed_sample_temp(tmp_dir)
            raise

        index.append({
            "sample_id": sample_id,
            "seed": seed,
            "config_hash": meta["config_hash"],
            "object_hash": meta["object_hash"],
            "blur_model": meta["blur_model"],
            "gain_mode": meta["gain_mode"],
            "detector_mode": meta["detector_mode"],
            **{f"sweep_{k}": params[k] for k in params},
            "V_eff": float((res.meta.get("vis") or {}).get("V_eff", float("nan"))),
            "mean_hits_plus": float(res.meta.get("mean_hits_plus") or 0.0),
            "sig_opt": float(res.meta.get("sig_opt") or 0.0),
            "model_version": meta["model_version"],
            "setup_id": meta["setup_id"],
            "sample_dir": str(sample_dir.relative_to(out_dir)).replace("\\", "/"),
        })
        n += 1

    canonical_audit = (
        None
        if canonical_hashes is None
        else canonical_config_audit(verified=True, hashes=canonical_hashes)
    )
    manifest = {
        "phase5_id": MODEL_VERSION,
        "release_version": RELEASE_VERSION,
        "model_status": MODEL_STATUS,
        "verified_digital_twin": False,
        "provenance_schema": "qiup_v_p0_13_c7",
        "n_samples": n,
        "sweeps": sweeps,
        "setup_id": setup.setup_id,
        "model_version": setup.model_version,
        "config_hash": dataset_config_hash,
        "object_hash": dataset_object_hash,
        "object_evidence": object_evidence,
        "require_canonical": bool(require_canonical),
        "is_canonical_config": bool(is_canonical_setup(setup)),
        "canonical_config_file": (
            None if canonical_audit is None else canonical_audit["canonical_config_file"]
        ),
        "canonical_config_sha256": (
            None if canonical_audit is None else canonical_audit["canonical_config_sha256"]
        ),
        "canonical_config_verified": (
            None if canonical_audit is None else canonical_audit["canonical_config_verified"]
        ),
        "canonical_config_hash": (
            None if canonical_audit is None else canonical_audit["canonical_config_hash"]
        ),
        "canonical_yaml_sha256": (
            None if canonical_audit is None else canonical_audit["canonical_yaml_sha256"]
        ),
        "blur_model": getattr(setup, "blur_model", None),
        "gain_mode": getattr(setup, "spdc_gain_mode", None),
        "spdc_source_mode": getattr(setup, "spdc_source_mode", None),
        "architecture_profile": resolve_architecture_profile(setup).to_dict(),
        "pipeline_execution_order": list(THREE_CONFIGURATION_PIPELINE_STAGES),
        "physical_model_support": physical_model_support_status(setup),
        "implemented": physical_model_support_status(setup)["implemented"],
        "software_tested": physical_model_support_status(setup)["software_tested"],
        "experimentally_validated": physical_model_support_status(setup)[
            "experimentally_validated"
        ],
        "setup_selection": setup_selection_info(setup),
        "setup": setup.to_dict(),
        "detector_mode": getattr(setup, "detector_mode", None),
        "base_preset": base_preset,
        "num_frames": num_frames,
        "seed0": seed0,
        "fail_closed_forbidden": True,
        "output_dir": str(out_dir.resolve()),
        "requested_out_dir": str(requested_out_dir.resolve()),
        "run_id": out_dir.name,
        "create_run_subdir": bool(create_run_subdir),
        "dataset_dir_policy": "reject_nonempty_or_allocate_runs_subdir",
        "latent_sampling": sampling,
        "sampling_regime": sampling_regime,
        "n_skipped_sampling_exhausted": len(skipped),
        "skipped_samples": skipped,
        "samples": index,
    }
    manifest["environment"] = get_environment()
    # Comment 7.5 — re-verify object maps before sealing the dataset manifest.
    from provenance import verify_object_evidence

    obj_mismatches = verify_object_evidence(out_dir, expected=object_evidence)
    if obj_mismatches:
        raise RuntimeError(
            "Comment 7.5: object evidence verification failed: "
            + "; ".join(obj_mismatches)
        )
    # Comment 5.6: write dataset-level manifest last; committed samples only.
    committed_index = committed_sample_index(out_dir, index)
    write_dataset_manifest(out_dir, manifest, index=committed_index)
    # Keep the returned / in-memory manifest aligned with what was written.
    manifest["samples"] = committed_index
    manifest["n_samples"] = len(committed_index)
    manifest["manifest_indexes_committed_samples_only"] = True
    with open(out_dir / "config_hash.txt", "w", encoding="utf-8") as f:
        f.write(f"{dataset_config_hash}\n")
    # CSV index for quick browsing (committed samples only).
    if committed_index:
        csv_path = out_dir / "dataset_index.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(committed_index[0].keys()))
            writer.writeheader()
            writer.writerows(committed_index)
    return manifest


# ==============================================================================
# V-P0-13 — run provenance replay (simulation A → manifest → simulation B)
# ==============================================================================

def build_run_manifest(
    result: SimResult,
    *,
    ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializable run manifest for offline replay (V-P0-13).

    Captures seed, setup, latents, detector_noise, and array hashes from
    simulation A so a later process can reconstruct simulation B.
    """
    meta = dict(result.meta or {})
    prov = dict(meta.get("provenance") or {})
    arrays = hash_sim_arrays(
        frames_plus=result.frames_plus,
        frames_minus=result.frames_minus,
        frames_D=result.frames_D,
        clean_D=result.clean_D,
    )
    return {
        "provenance_schema": "qiup_v_p0_13_c7",
        "version": meta.get("version") or prov.get("version") or MODEL_VERSION,
        "release_status": meta.get("release_status")
        or prov.get("release_status")
        or RELEASE_STATUS,
        "seed": int(meta.get("seed", prov.get("seed", 0))),
        "rng_scheme": meta.get("rng_scheme") or prov.get("rng_scheme"),
        "detector_noise": bool(meta.get("detector_noise", True)),
        "require_canonical": bool(meta.get("require_canonical", False)),
        "num_frames": int(meta.get("num_frames", result.frames_plus.shape[0])),
        "photon_counting": bool(meta.get("photon_counting", False)),
        "em_gain_GEM": float(meta.get("em_gain_GEM", getattr(result.state, "em_gain", 250.0))),
        "exposure_s": float(getattr(result.state, "exposure_s", 0.05)),
        "blur_model": meta.get("blur_model"),
        "gain_mode": meta.get("gain_mode"),
        "spdc_source_mode": meta.get("spdc_source_mode") or prov.get("spdc_source_mode"),
        "detector_mode": meta.get("detector_mode"),
        "config_hash": meta.get("config_hash") or prov.get("config_hash"),
        "object_hash": meta.get("object_hash") or prov.get("object_hash"),
        "timestamp": meta.get("timestamp") or prov.get("timestamp"),
        "source_hashes": meta.get("source_hashes") or prov.get("source_hashes"),
        "build_hash": meta.get("build_hash") or prov.get("build_hash"),
        "environment": meta.get("environment") or prov.get("environment"),
        "setup": meta.get("setup"),
        "latents_raw": meta.get("latents_raw"),
        "provenance": prov,
        "array_hashes": arrays,
        "configuration_evidence": meta.get("configuration_evidence")
        or prov.get("configuration_evidence"),
        "requested_configuration": meta.get("requested_configuration")
        or prov.get("requested_configuration"),
        "resolved_configuration": meta.get("resolved_configuration")
        or prov.get("resolved_configuration"),
        "derived_physical_quantities": meta.get("derived_physical_quantities")
        or prov.get("derived_physical_quantities"),
        "pump_power_laser_mW": meta.get("pump_power_laser_mW"),
        "pump_power_NL1_mW": meta.get("pump_power_NL1_mW"),
        "pump_power_NL2_mW": meta.get("pump_power_NL2_mW"),
        "nbar_NL1": meta.get("nbar_NL1"),
        "nbar_NL2": meta.get("nbar_NL2"),
        "object_evidence": meta.get("object_evidence")
        or prov.get("object_evidence"),
        "reference_hashes": (
            prov.get("reference_hashes")
            if prov.get("reference_hashes") is not None
            else None
        ),
        "ref_seed": None if ref is None else ref.get("seed"),
        "strict_identity_verification": True,
    }


def latent_state_from_run_manifest(manifest: dict[str, Any]) -> LatentState:
    """Rebuild ``LatentState`` from a V-P0-13 run / sample manifest."""
    setup_payload = manifest.get("setup")
    if not isinstance(setup_payload, dict):
        raise ValueError("run manifest missing setup dict for replay")
    setup = SetupParams.from_dict(setup_payload)
    raw = manifest.get("latents_raw") or {}
    missing = [k for k in LATENT_KEYS if k not in raw]
    if missing:
        raise ValueError(f"run manifest latents_raw missing keys: {missing}")
    kwargs = {k: float(raw[k]) for k in LATENT_KEYS}
    optional_names = (
        "delta_phi_rms", "delta_phi_corr_px",
        "mean_photons_per_mode", "spdc_gain_r",
        "mean_photons_per_mode_NL1", "mean_photons_per_mode_NL2",
        "spdc_gain_r_NL1", "spdc_gain_r_NL2", "pump_power_mW",
    )
    optional = {
        name: raw[name]
        for name in optional_names
        if name in raw and raw[name] is not None
    }
    return LatentState(
        s_pump=kwargs["s_pump"],
        s_gain=kwargs["s_gain"],
        s_align=kwargs["s_align"],
        s_thermal=kwargs["s_thermal"],
        s_det=kwargs["s_det"],
        s_cool=kwargs["s_cool"],
        s_focus=kwargs["s_focus"],
        s_slm=kwargs["s_slm"],
        em_gain=float(manifest.get("em_gain_GEM", 250.0)),
        exposure_s=float(manifest.get("exposure_s", 0.05)),
        photon_counting=bool(
            manifest.get(
                "photon_counting",
                detector_mode_is_photon_counting(setup.detector_mode),
            )
        ),
        num_frames=int(manifest.get("num_frames", 1)),
        setup=setup,
        **optional,
    )


def replay_from_run_manifest(
    ref: dict[str, Any],
    manifest: dict[str, Any],
    *,
    raise_on_forbidden: bool = True,
) -> SimResult:
    """Replay simulation B from a saved run manifest (V-P0-13).

    Uses the same seed, setup, latents, and detector_noise flag recorded from
    simulation A. Caller supplies the same object ``ref`` (or an equivalent
    array-identical reference).
    """
    state = latent_state_from_run_manifest(manifest)
    seed = int(manifest["seed"])
    detector_noise = bool(manifest.get("detector_noise", True))
    require_canonical = bool(manifest.get("require_canonical", False))
    return simulate_forward(
        ref,
        state,
        seed=seed,
        detector_noise=detector_noise,
        raise_on_forbidden=raise_on_forbidden,
        require_canonical=require_canonical,
    )


def verify_replay_against_manifest(
    result: SimResult,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare replayed arrays/hashes to the recorded simulation-A manifest."""
    actual = hash_sim_arrays(
        frames_plus=result.frames_plus,
        frames_minus=result.frames_minus,
        frames_D=result.frames_D,
        clean_D=result.clean_D,
    )
    expected = dict(manifest.get("array_hashes") or {})
    mismatches = compare_array_hashes(expected, actual)
    cfg_ok = True
    if manifest.get("config_hash") is not None:
        cfg_ok = str(result.meta.get("config_hash")) == str(manifest["config_hash"])
        if not cfg_ok:
            mismatches.append(
                f"config_hash: expected={manifest['config_hash']} "
                f"actual={result.meta.get('config_hash')}"
            )
    identity: dict[str, bool | None] = {
        "source_hashes_match": None,
        "build_hash_match": None,
        "object_hash_match": None,
        "resolved_configuration_match": None,
    }
    if bool(manifest.get("strict_identity_verification", False)):
        actual_provenance = dict(result.meta.get("provenance") or {})
        comparisons = (
            ("source_hashes", "source_hashes_match"),
            ("build_hash", "build_hash_match"),
            ("object_hash", "object_hash_match"),
            ("resolved_configuration", "resolved_configuration_match"),
        )
        for manifest_key, report_key in comparisons:
            expected_value = manifest.get(manifest_key)
            actual_value = (
                result.meta.get(manifest_key)
                if result.meta.get(manifest_key) is not None
                else actual_provenance.get(manifest_key)
            )
            if expected_value is None:
                identity[report_key] = None
                continue
            matches = json.dumps(
                expected_value, sort_keys=True, default=str, separators=(",", ":")
            ) == json.dumps(
                actual_value, sort_keys=True, default=str, separators=(",", ":")
            )
            identity[report_key] = matches
            if not matches:
                mismatches.append(f"{manifest_key}: recorded identity differs from replay")
    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "expected_hashes": expected,
        "actual_hashes": actual,
        "config_hash_match": cfg_ok,
        "seed": int(manifest.get("seed", -1)),
        "identity_verification": identity,
    }


# ==============================================================================
# Comment 7.6 — replay_from_manifest(manifest_path) (evidence pack → simulation)
# ==============================================================================

RUN_MANIFEST_FILENAME = "run_manifest.json"


def save_run_evidence_pack(
    out_dir: str | Path,
    result: SimResult,
    ref: dict[str, Any],
) -> Path:
    """Persist a self-contained run evidence pack for Comment 7.6 replay.

    Writes::

        out_dir/object/{T,phi,M,...}.npy
        out_dir/object/object_evidence.json
        out_dir/run_manifest.json

    A reviewer can then call ``replay_from_manifest(out_dir/run_manifest.json)``
    with no manual object/config reconstruction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    object_evidence = save_object_evidence(ref, out_dir)
    manifest = build_run_manifest(result, ref=ref)
    manifest["object_evidence"] = object_evidence
    manifest["finding"] = "comment_7_6_replay_from_manifest"
    manifest["evidence_pack"] = True
    path = out_dir / RUN_MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _load_json_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Comment 7.6: cannot load manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("Comment 7.6: manifest root must be a JSON object")
    return data


def _run_manifest_from_sample_metadata(
    sample_meta: dict[str, Any],
    *,
    object_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a sample ``metadata.json`` into a run-manifest shape."""
    prov = dict(sample_meta.get("provenance") or {})
    detector_noise = sample_meta.get("detector_noise")
    if detector_noise is None:
        detector_noise = prov.get("detector_noise")
    if detector_noise is None:
        # Official datasets default to detector-on measured frames.
        detector_noise = True
    return {
        "provenance_schema": prov.get("provenance_schema")
        or sample_meta.get("provenance_schema")
        or "qiup_v_p0_13_c7",
        "version": sample_meta.get("version") or prov.get("version"),
        "release_status": sample_meta.get("release_status")
        or prov.get("release_status"),
        "seed": int(sample_meta["seed"]),
        "rng_scheme": sample_meta.get("rng_scheme") or prov.get("rng_scheme"),
        "detector_noise": bool(detector_noise),
        "require_canonical": bool(
            sample_meta.get("require_canonical", prov.get("require_canonical", False))
        ),
        "num_frames": int(sample_meta.get("num_frames", 1)),
        "photon_counting": bool(sample_meta.get("photon_counting", False)),
        "em_gain_GEM": float(sample_meta.get("em_gain_GEM", 250.0)),
        "exposure_s": float(
            sample_meta.get("exposure_s")
            if sample_meta.get("exposure_s") is not None
            else (
                (sample_meta.get("latents_raw") or {}).get("exposure_s")
                if (sample_meta.get("latents_raw") or {}).get("exposure_s") is not None
                else 0.1
            )
        ),
        "blur_model": sample_meta.get("blur_model"),
        "gain_mode": sample_meta.get("gain_mode"),
        "spdc_source_mode": sample_meta.get("spdc_source_mode"),
        "detector_mode": sample_meta.get("detector_mode"),
        "config_hash": sample_meta.get("config_hash"),
        "object_hash": sample_meta.get("object_hash"),
        "setup": sample_meta.get("setup"),
        "latents_raw": sample_meta.get("latents_raw"),
        "array_hashes": sample_meta.get("array_hashes") or prov.get("array_hashes"),
        "object_evidence": object_evidence
        or sample_meta.get("object_evidence")
        or prov.get("object_evidence"),
        "provenance": prov,
    }


def resolve_run_manifest_from_path(
    manifest_path: str | Path,
    *,
    sample_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load a run or dataset evidence manifest and return (run_manifest, pack_dir)."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comment 7.6: manifest not found: {path}")
    data = _load_json_manifest(path)
    pack_dir = path.parent

    # Dataset pack: dataset_manifest.json with samples[] + shared object/.
    if path.name == DATASET_MANIFEST_FILENAME or (
        isinstance(data.get("samples"), list) and data.get("setup") is None
        and "latents_raw" not in data
    ):
        samples = list(data.get("samples") or [])
        if not samples:
            raise ValueError("Comment 7.6: dataset manifest has no samples")
        chosen = None
        if sample_id is not None:
            for row in samples:
                if row.get("sample_id") == sample_id:
                    chosen = row
                    break
            if chosen is None:
                raise ValueError(
                    f"Comment 7.6: sample_id={sample_id!r} not in dataset manifest"
                )
        else:
            chosen = samples[0]
        rel = chosen.get("sample_dir") or f"samples/{chosen['sample_id']}"
        sample_meta_path = pack_dir / rel / "metadata.json"
        if not sample_meta_path.is_file():
            raise FileNotFoundError(
                f"Comment 7.6: sample metadata missing: {sample_meta_path}"
            )
        sample_meta = _load_json_manifest(sample_meta_path)
        run_manifest = _run_manifest_from_sample_metadata(
            sample_meta,
            object_evidence=data.get("object_evidence")
            or sample_meta.get("object_evidence"),
        )
        return run_manifest, pack_dir

    # Sample metadata.json directly.
    if path.name == "metadata.json" and "sample_id" in data and "setup" in data:
        # Prefer pack root two levels up: samples/sample_XXXX/metadata.json
        pack_guess = path.parent.parent.parent if path.parent.parent.name == "samples" else path.parent
        obj_ev = data.get("object_evidence")
        if obj_ev is None:
            # Walk up looking for object/evidence.
            for candidate in (path.parent, path.parent.parent, pack_guess):
                if (candidate / "object" / "object_evidence.json").is_file():
                    pack_guess = candidate
                    break
        run_manifest = _run_manifest_from_sample_metadata(data, object_evidence=obj_ev)
        return run_manifest, pack_guess

    # Standalone run_manifest.json evidence pack.
    return data, pack_dir


def replay_from_manifest(
    manifest_path: str | Path,
    *,
    sample_id: str | None = None,
    verify: bool = True,
    raise_on_forbidden: bool = True,
) -> SimResult:
    """Comment 7.6 — reconstruct a simulation from an evidence manifest path alone.

    Reconstructs without manual intervention:

    - configuration (``setup`` from the manifest)
    - object maps (``object/T.npy``, ``object/phi.npy``, …)
    - seed / RNG scheme
    - operating modes (blur / gain / detector / source)
    - forward simulation

    Accepts:

    - ``run_manifest.json`` from ``save_run_evidence_pack``
    - ``dataset_manifest.json`` (+ optional ``sample_id``)
    - a sample ``metadata.json`` inside a dataset pack

    When ``verify=True`` (default), array / config hashes are checked against
    the recorded evidence and a mismatch raises ``RuntimeError``.
    """
    run_manifest, pack_dir = resolve_run_manifest_from_path(
        manifest_path, sample_id=sample_id
    )
    if run_manifest.get("setup") is None:
        raise ValueError(
            "Comment 7.6: manifest is missing setup/configuration for replay"
        )
    if run_manifest.get("latents_raw") is None:
        raise ValueError("Comment 7.6: manifest is missing latents_raw for replay")
    if run_manifest.get("seed") is None:
        raise ValueError("Comment 7.6: manifest is missing seed for replay")

    object_evidence = run_manifest.get("object_evidence")
    ref = load_object_ref_from_evidence(pack_dir, object_evidence)
    # Ensure schema / target size for optics.
    if "target_size" not in ref and ref.get("M") is not None:
        ref["target_size"] = int(np.asarray(ref["M"]).shape[0])

    result = replay_from_run_manifest(
        ref,
        run_manifest,
        raise_on_forbidden=raise_on_forbidden,
    )
    report = verify_replay_against_manifest(result, run_manifest)
    result.meta["replay_from_manifest"] = {
        "finding": "comment_7_6_replay_from_manifest",
        "manifest_path": str(Path(manifest_path).resolve()),
        "pack_dir": str(pack_dir),
        "sample_id": sample_id,
        "verification": report,
        "rng_scheme": run_manifest.get("rng_scheme"),
        "blur_model": run_manifest.get("blur_model"),
        "gain_mode": run_manifest.get("gain_mode"),
        "spdc_source_mode": run_manifest.get("spdc_source_mode"),
        "detector_mode": run_manifest.get("detector_mode"),
    }
    if verify and not report["ok"]:
        raise RuntimeError(
            "Comment 7.6: replay verification failed: "
            + "; ".join(report["mismatches"])
        )
    return result
