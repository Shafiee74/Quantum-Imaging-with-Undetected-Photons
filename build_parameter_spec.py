"""Generate the parameter inventory without changing forward physics.

Package release identity is Version 3.4.2. Inventory/audit filenames that still
contain ``3.4.1`` are frozen historical artifact names retained for traceability.
"""

Every ``SetupParams`` and ``LatentState`` field gets exactly one row. The
configurations in which a field is consumed are not hand-written: they are
taken from ``configs/parameter_consumption_audit_3.4.1.json`` (produced by
``parameter_inventory_audit.py``), which traces actual code usage per
registered profile. Derived quantities (rows without a Python field) take
their applicability from the registered profile properties.

The curated rows below keep the physical meaning, unit and range documented
in Version 3.4. Their hand-written 3.4 applicability is retained as
``documented_dependency_v3_4`` so applicability changes are reported, not
silently overwritten.

Outputs:
  configs/three_configuration_parameter_spec.json  (machine-readable)
  configs/PARAMETER_INVENTORY_3.4.1.md             (optional human-readable)
"""
from __future__ import annotations

import ast
from dataclasses import MISSING, fields
import json
from pathlib import Path
import re
from typing import Any

from latents import LATENT_KEYS, REGIME_BOUNDS, LatentState
from provenance import CALIBRATION_STATEMENT, CALIBRATION_STATUS, PACKAGE_VERSION, RELEASE_CLASSIFICATION
import setup_params
from setup_params import SetupParams, resolve_setup_selection

ROOT = Path(__file__).resolve().parent
SPEC_RELATIVE = "configs/three_configuration_parameter_spec.json"
MARKDOWN_RELATIVE = "configs/PARAMETER_INVENTORY_3.4.1.md"
AUDIT_RELATIVE = "configs/parameter_consumption_audit_3.4.1.json"
PROFILES = ("A", "B", "C", "D")
LATENT_SKIP = {"setup", "physical_report"}
GROUPS = ("shared_optics", "physical_crystal", "interaction", "imaging", "source_gain", "detector")

NOT_CALIBRATED = "NOT YET CALIBRATED / EXPERIMENTAL VALUE REQUIRED"
CALIBRATION_STATUS_BY_ROLE = {
    "calibration-dependent": NOT_CALIBRATED,
    "sampled": "SYNTHETIC SAMPLED LATENT (not a laboratory value)",
    "derived": "DERIVED (inherits the calibration status of its inputs)",
    "fixed": "MODEL CONSTANT (assumed software value; not experimentally calibrated)",
    "selector": "NOT APPLICABLE (model / profile selector)",
    "metadata": "NOT APPLICABLE (identifier / metadata)",
}
AUDIT_STATUS_NOTE = {
    "physics": "changes simulated arrays",
    "validated_fixed": "value fixed by profile validation (perturbations rejected)",
    "metadata_only": "recorded in run metadata; no array change detected in audit scenarios",
    "not_consumed": "no effect detected in audit scenarios",
    "not_perturbable": "no admissible perturbation found",
}

# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------
SAMPLED = set(LATENT_KEYS) | {"num_frames", "em_gain", "exposure_s", "photon_counting"}
DERIVED = {
    "NA", "magnification", "magnification_requested", "image_scale", "T_crystal_C", "T_pm_C",
    "d_eff_requested_pm_per_V", "d33_requested_pm_per_V", "d_eff_resolved_pm_per_V", "d_eff_source",
    "d_eff_pm_per_V_requested", "d33_pm_per_V_requested", "d_eff_resolution", "alpha_L_mm_per_C",
    "dlam_s_nm_per_C", "dlam_i_nm_per_C", "dM_over_M_per_C", "spdc_architecture", "n_crystals",
    "K_defocus_px_per_um", "defocus_span_px",
}
METADATA = {
    "setup_id", "model_version", "physics_tier", "notes", "coupling_profile_id", "sampling_regime",
    "physics_policy_id", "pol_pump", "pol_signal", "pol_idler", "pol_pump_NL2", "pol_signal_NL2",
    "pol_idler_NL2", "apparatus_crystal_NL1", "apparatus_crystal_NL2", "calibration_records",
    "n_spdc_interactions", "object_phase_map_path", "object_amplitude_map_path",
    "object_intensity_map_path", "object_transmission_map", "object_phase_map",
}
SELECTORS = {"object_mask_id", "coupling_weights", "fixed_source_input", "phase_step_schedule_rad"}
CALIBRATION_DEPENDENT = {
    # pump and SPDC source
    "lambda_p_nm", "lambda_s_nm", "lambda_i_nm", "pump_power_laser_mW", "pump_waist_um",
    "pump_waist_NL1_um", "pump_waist_NL2_um", "pump_focus_z_NL1_um", "pump_focus_z_NL2_um",
    "pump_center_x_um", "pump_center_y_um", "pointing_alpha", "pump_sigma_rel_at_s1", "flux0_e_per_s",
    "pump_pre_crystal_transmission", "pump_return_transmission", "pump_to_NL2_transmission",
    "pump_bs_reflectance", "pump_between_crystals_transmission", "pump_power_NL2_override_mW",
    "pump_target_power_NL1_mW", "pump_target_power_NL2_mW", "spdc_gain_per_sqrt_mW",
    "spdc_gain_per_sqrt_mW_NL2", "measured_power_W", "measured_power_wavelength_nm",
    "measured_power_to_nbar_per_W", "measured_power_to_nbar_per_W_NL2",
    "spdc_reference_mean_photons_per_mode", "pump_power_mW", "mean_photons_per_mode",
    "mean_photons_per_mode_NL1", "mean_photons_per_mode_NL2", "spdc_gain_r", "spdc_gain_r_NL1", "spdc_gain_r_NL2",
    # crystals
    "crystal_length_mm", "crystal_length_NL2_mm", "poling_period_um", "poling_period_NL2_um",
    "T_pm0_NL1_C", "T_pm0_NL2_C", "dT_NL1_C", "dT_NL2_C", "T_crystal_NL1_C", "T_crystal_NL2_C",
    "delta_T_NL1_C", "delta_T_NL2_C", "d33_pm_per_V", "d_eff_pm_per_V", "d33_NL2_pm_per_V",
    "d_eff_NL2_pm_per_V", "qpm_duty_cycle", "qpm_duty_cycle_NL2", "crystal_orientation_NL1_deg",
    "crystal_orientation_NL2_deg",
    # interferometer, relays, object
    "path1_intensity", "path2_intensity", "idler_path_transmission", "idler_mode_matching",
    "idler_round_trip_loss", "idler_return_phase", "relative_interferometric_phase",
    "signal_return_path_transmission", "signal_return_reflection_amplitude", "signal_return_phase",
    "signal_return_mode_overlap", "idler_waist_um", "idler_waist_2_um", "generation_dx_um",
    "generation_dy_um", "generation_dtheta_mrad", "relay_dx_um", "relay_dy_um", "relay_dtheta_mrad",
    "idler_dx_um", "idler_dy_um", "idler_dtheta_mrad", "delta_L_mm", "L_coh_mm",
    "signal_spectrum_fwhm_nm", "camera_filter_center_nm", "camera_filter_fwhm_nm",
    "camera_filter_wavelengths_nm", "camera_filter_transmissions", "V0",
    "near_field_idler_magnification", "near_field_signal_magnification",
    "far_field_idler_relay_magnification", "far_field_signal_relay_magnification", "object_map_pixel_um",
    "object_intensity_T", "object_intensity_R", "object_phase_rad", "object_reflection_amplitude",
    "object_reflection_phase", "F1_mm", "F2_mm", "far_field_focal_mm", "aperture_diameter_mm",
    "lens_focal_mm", "lens_n", "lens_dndT_per_C", "lens_cte_per_C", "delta_z_um", "wavefront_rms_waves",
    "image_rotation_deg", "delta_M_rel", "camera_center_x_px", "camera_center_y_px",
    "optical_background_photons", "alpha_path_per_C", "L_eff_path_mm", "beam_thermal_alpha_x_um_per_C",
    "beam_thermal_alpha_y_um_per_C", "T_env_C", "lens_shift_x_um", "lens_shift_y_um",
    # SLM
    "slm_levels0", "slm_bit_depth", "slm_pixel_um", "slm_fill_factor", "slm_phi_max_rad", "slm_gamma_span",
    "slm_leak_eta", "slm_misreg_x_px", "slm_misreg_y_px",
    # detector
    "pixel_um", "QE", "read_noise0_e", "read_noise_span_e", "dark0", "dark_exp", "cic0", "prnu_sigma",
    "hot_frac_at_s1", "hot_amp_lo", "hot_amp_hi", "hot_T_coef_per_C", "full_well_e", "g_adu", "bias_adu",
    "max_adu", "adc_bit_depth", "T_sensor_C", "dark_arrhenius_prefactor", "accidental_e_at_s1",
}

SYMBOLS = {
    "lambda_p_nm": "λ_p", "lambda_s_nm": "λ_s", "lambda_i_nm": "λ_i", "pump_power_laser_mW": "P_laser",
    "pump_power_mW": "P_laser", "pump_waist_um": "w_p", "pump_waist_NL1_um": "w_p,1",
    "pump_waist_NL2_um": "w_p,2", "pump_pre_crystal_transmission": "η_pre", "pump_bs_reflectance": "R_BS",
    "pump_to_NL2_transmission": "η_NL2", "pump_between_crystals_transmission": "η_between",
    "pump_return_transmission": "η_ret", "spdc_gain_per_sqrt_mW": "k_1", "spdc_gain_per_sqrt_mW_NL2": "k_2",
    "crystal_length_mm": "L_1", "crystal_length_NL2_mm": "L_2", "poling_period_um": "Λ_1",
    "poling_period_NL2_um": "Λ_2", "T_pm0_NL1_C": "T_pm0,1", "T_pm0_NL2_C": "T_pm0,2",
    "T_crystal_NL1_C": "T_1", "T_crystal_NL2_C": "T_2", "dT_NL1_C": "dT_1", "dT_NL2_C": "dT_2",
    "d33_pm_per_V": "d_33", "d_eff_pm_per_V": "d_eff", "qpm_order": "m", "qpm_duty_cycle": "D",
    "path1_intensity": "η_s1", "path2_intensity": "η_s2", "idler_path_transmission": "η_i",
    "idler_mode_matching": "η_mode", "relative_interferometric_phase": "φ_rel", "delta_L_mm": "ΔL",
    "L_coh_mm": "L_coh", "near_field_idler_magnification": "M_u", "near_field_signal_magnification": "M_s",
    "object_map_pixel_um": "p_obj", "pixel_um": "p_cam", "object_intensity_T": "T_obj",
    "object_intensity_R": "R_obj", "object_phase_rad": "φ_obj", "F1_mm": "F_1", "F2_mm": "F_2",
    "far_field_focal_mm": "f_u", "QE": "QE", "em_gain": "G_EM", "exposure_s": "t_exp",
    "mean_photons_per_mode": "n̄", "mean_photons_per_mode_NL1": "n̄_1", "mean_photons_per_mode_NL2": "n̄_2",
    "spdc_gain_r": "r", "spdc_gain_r_NL1": "r_1", "spdc_gain_r_NL2": "r_2", "NA": "NA",
    "magnification": "M_QIUP", "aperture_diameter_mm": "D_ap", "signal_spectrum_fwhm_nm": "Δλ_src",
    "camera_filter_center_nm": "λ_f", "camera_filter_fwhm_nm": "Δλ_f", "read_noise0_e": "σ_read,0",
    "dark0": "i_dark,0", "cic0": "CIC", "prnu_sigma": "σ_PRNU", "full_well_e": "FW", "g_adu": "g",
    "bias_adu": "b", "image_rotation_deg": "θ_rot", "pointing_alpha": "α_point", "idler_waist_um": "w_i",
    "slm_phi_max_rad": "φ_SLM,max", "slm_levels0": "N_SLM", "flux0_e_per_s": "Φ_0",
}

UNIT_SUFFIXES = (
    ("_rad_per_frame_at_s1", "rad/frame at s=1"), ("_px_per_frame_at_s1", "px/frame at s=1"),
    ("_per_frame_at_s1", "1/frame at s=1"), ("_at_s1", "per unit latent at s=1"),
    ("_mm_per_C", "mm/°C"), ("_um_per_C", "µm/°C"), ("_nm_per_C", "nm/°C"), ("_px_per_um", "px/µm"),
    ("_pm_per_V", "pm/V"), ("_per_sqrt_mW", "1/√mW"), ("_per_W", "1/W"), ("_e_per_s", "e⁻/s"),
    ("_per_C", "1/°C"), ("_mm", "mm"), ("_um", "µm"), ("_nm", "nm"), ("_mW", "mW"), ("_W", "W"),
    ("_C", "°C"), ("_deg", "deg"), ("_mrad", "mrad"), ("_rad", "rad"), ("_px", "px"), ("_eV", "eV"),
    ("_waves", "waves at λ_s"), ("_adu", "ADU"), ("_e", "e⁻"), ("_s", "s"),
)
UNIT_OVERRIDES = {
    "QE": "fraction", "NA": "dimensionless", "em_gain": "dimensionless (G_EM)", "num_frames": "frames",
    "slm_levels0": "levels", "slm_bit_depth": "bits", "slm_levels_floor": "levels", "hot_frac_at_s1": "fraction",
    "hot_amp_lo": "e⁻/frame", "hot_amp_hi": "e⁻/frame", "g_adu": "ADU/e⁻", "max_adu": "ADU",
    "adc_bit_depth": "bits", "dark0": "e⁻/pixel/s", "cic0": "e⁻/pixel", "optical_background_photons":
    "photons/pixel/exposure", "wavefront_rms_waves": "waves at λ_s", "delta_phi_rms": "rad",
    "flux0_e_per_s": "e⁻/s", "dark_arrhenius_prefactor": "e⁻/pixel/s", "full_well_e": "e⁻",
    "accidental_e_at_s1": "e⁻/pixel at s=1", "pc_threshold_k": "σ_read multiples",
}
UNIT_SUFFIX_EXEMPT = {"notes", "physics_policy_id", "coupling_profile_id", "setup_id"}


# ---------------------------------------------------------------------------
# Curated rows retained from Version 3.4. Tuple layout (unchanged):
# (meaning, owner, name, unit, range[, documented_dependency_v3_4[, scope[, value_class[, status]]]])
# ---------------------------------------------------------------------------
def curated_rows() -> dict[str, list[tuple]]:
    return {
        "shared_optics": [
            ("pump wavelength", "SetupParams", "lambda_p_nm", "nm", ">0; energy-conserving triple"),
            ("signal wavelength", "SetupParams", "lambda_s_nm", "nm", ">0; energy-conserving triple"),
            ("idler wavelength", "SetupParams", "lambda_i_nm", "nm", ">0; energy-conserving triple"),
            ("laser output power", "SetupParams", "pump_power_laser_mW", "mW", "nonnegative or null"),
            ("pump beam waist", "SetupParams", "pump_waist_um", "um", "100..1000"),
            ("pump transverse spatial profile", "SetupParams", "pump_profile_type", "name", "gaussian/uniform_effective", "A,B,C", "shared", "assumed", "implemented_reduced_order"),
            ("pump profile center x", "SetupParams", "pump_center_x_um", "um", "finite", "A,B,C", "shared", "assumed", "implemented"),
            ("pump profile center y", "SetupParams", "pump_center_y_um", "um", "finite", "A,B,C", "shared", "assumed", "implemented"),
            ("pump pointing jitter scale", "SetupParams", "pointing_alpha", "dimensionless", "0.01..0.10"),
            ("signal path one intensity transmission", "SetupParams", "path1_intensity", "dimensionless", "0..1"),
            ("signal path two intensity transmission", "SetupParams", "path2_intensity", "dimensionless", "0..1"),
            ("idler path intensity transmission", "SetupParams", "idler_path_transmission", "dimensionless", "0..1"),
            ("idler mode overlap waist", "SetupParams", "idler_waist_um", "um", "positive or inherited"),
            ("second idler mode overlap waist", "SetupParams", "idler_waist_2_um", "um", "positive or inherited"),
            ("signed optical path difference", "SetupParams", "delta_L_mm", "mm", "validated signed domain"),
            ("phase drift scale", "SetupParams", "phase_drift_rad_per_frame_at_s1", "rad/frame", "finite"),
            ("generation alignment x", "SetupParams", "generation_dx_um", "um", "finite", "A,B,C", "interaction geometry"),
            ("relay alignment x", "SetupParams", "relay_dx_um", "um", "finite", "A,B,C", "idler relay"),
            ("detector pixel pitch", "SetupParams", "pixel_um", "um", "6.5..24"),
            ("detector image dimensions", None, None, "px x px", "reference target_size", "A,B,C", "per run", "measured", "runtime_reference"),
        ],
        "physical_crystal": [
            ("NL1 material", "SetupParams", "crystal_material", "name", "ppKTP currently"),
            ("NL / NL1 length", "SetupParams", "crystal_length_mm", "mm", "2..20", "A,B,C,D", "per physical crystal"),
            ("NL2 length equals NL1", "SetupParams", "crystal_length_NL2_mm", "mm", "displayed equal to NL1", "A,D", "two-crystal matched"),
            ("NL / NL1 poling period", "SetupParams", "poling_period_um", "um", "positive", "A,B,C,D", "per physical crystal"),
            ("NL2 poling period equals NL1", "SetupParams", "poling_period_NL2_um", "um", "displayed equal to NL1", "A,D", "two-crystal matched"),
            ("NL1 phase-matching temperature", "SetupParams", "T_pm0_NL1_C", "deg C", "required user input", "A,B,C,D", "per physical crystal"),
            ("NL2 phase-matching temperature", "SetupParams", "T_pm0_NL2_C", "deg C", "required user input, independent of NL1", "A,D", "two-crystal ovens"),
            ("NL1 operating detuning dT", "SetupParams", "dT_NL1_C", "deg C", "user variable; default 0", "A,B,C,D", "per physical crystal"),
            ("NL2 operating detuning dT", "SetupParams", "dT_NL2_C", "deg C", "user variable; default 0", "A,D", "two-crystal ovens"),
            ("NL1 operating temperature", "SetupParams", "T_crystal_NL1_C", "deg C", "validated crystal domain; defaults to T_pm0", "A,B,C,D", "per physical crystal"),
            ("NL2 operating temperature", "SetupParams", "T_crystal_NL2_C", "deg C", "validated crystal domain; defaults to T_pm0_NL2", "A,D", "two-crystal ovens"),
            ("QPM order", "SetupParams", "qpm_order", "integer", ">=1"),
            ("QPM type", "SetupParams", "qpm_type", "name", "type0 currently"),
            ("QPM duty cycle", "SetupParams", "qpm_duty_cycle", "fraction", "0..1"),
            ("pump polarization", "SetupParams", "pol_pump", "name", "supported polarization"),
            ("signal polarization", "SetupParams", "pol_signal", "name", "supported polarization"),
            ("idler polarization", "SetupParams", "pol_idler", "name", "supported polarization"),
            ("d33 nonlinear coefficient", "SetupParams", "d33_pm_per_V", "pm/V", "positive or null"),
            ("effective nonlinear coefficient", "SetupParams", "d_eff_pm_per_V", "pm/V", "positive or null", "A,B,C", "per current shared coefficient", "calibrated"),
            ("NL2 material", "SetupParams", "crystal_material_NL2", "name", "ppKTP currently", "A only", "NL2 independent"),
            ("NL2 QPM type", "SetupParams", "qpm_type_NL2", "name", "type0 currently", "A only", "NL2 independent"),
            ("NL2 QPM order", "SetupParams", "qpm_order_NL2", "integer", "positive or inherited", "A only", "NL2 independent"),
            ("NL2 QPM duty cycle", "SetupParams", "qpm_duty_cycle_NL2", "fraction", "0..1 or inherited", "A only", "NL2 independent"),
            ("NL2 d33", "SetupParams", "d33_NL2_pm_per_V", "pm/V", "finite or inherited", "A only", "NL2 independent", "calibrated"),
            ("NL2 effective nonlinearity", "SetupParams", "d_eff_NL2_pm_per_V", "pm/V", "finite or inherited", "A only", "NL2 independent", "calibrated"),
            ("NL2 pump polarization", "SetupParams", "pol_pump_NL2", "name", "supported or inherited", "A only", "NL2 independent"),
            ("NL2 signal polarization", "SetupParams", "pol_signal_NL2", "name", "supported or inherited", "A only", "NL2 independent"),
            ("NL2 idler polarization", "SetupParams", "pol_idler_NL2", "name", "supported or inherited", "A only", "NL2 independent"),
            ("NL1 orientation", "SetupParams", "crystal_orientation_NL1_deg", "deg", "|theta|<=5.0 or null; Type-0 collinear", "A,B,C", "per physical crystal", "measured", "implemented_reduced_order"),
            ("NL2 orientation", "SetupParams", "crystal_orientation_NL2_deg", "deg", "|theta|<=5.0 or null; Type-0 collinear", "A only", "NL2 independent", "measured", "implemented_reduced_order"),
            ("pump/signal/idler refractive indices", None, None, "dimensionless", "derived from ppKTP Sellmeier and temperature", "A,B,C", "per physical crystal", "derived", "derived_spdc_crystal_state"),
            ("pump waist at interaction", "SetupParams", "pump_waist_NL1_um", "um", "100..1000 or inherited", "A,B,C", "per interaction"),
            ("second interaction pump waist", "SetupParams", "pump_waist_NL2_um", "um", "100..1000 or inherited", "A,B,C", "per interaction"),
            ("pump focus at first interaction", "SetupParams", "pump_focus_z_NL1_um", "um", "finite", "A,B,C", "per interaction"),
            ("pump focus at second interaction", "SetupParams", "pump_focus_z_NL2_um", "um", "finite or inherited", "A,B,C", "per interaction"),
            ("QPM phase mismatch", None, None, "1/m", "derived from wavelengths, temperature, and poling", "A,B,C", "per interaction", "derived", "derived_delta_k_qpm_per_m_from_setup"),
            ("physical generation geometry overlap", None, None, "dimensionless", "derived from pump geometry and generation alignment", "A,B,C", "per interaction", "derived", "derived_spdc_geometry_overlap_from_setup"),
        ],
        "interaction": [
            ("declared interaction count", "SetupParams", "n_spdc_interactions", "integer", "2 or omitted", "A,B,C", "shared", "assumed", "validated_assertion"),
            ("interaction index", None, None, "integer", "1 or 2", "A,B,C", "per interaction", "derived", "derived_profile"),
            ("physical crystal identity", None, None, "NL1 or NL2", "profile mapping", "A,B,C", "per interaction", "derived", "derived_profile"),
            ("propagation direction", None, None, "forward/backward", "profile mapping", "A,B,C", "per interaction", "derived", "profile_only"),
            ("local pump power", None, None, "mW", "nonnegative or unavailable", "A,B,C", "per interaction", "derived", "derived_pump_powers_from_setup"),
            ("relative interaction phase", "SetupParams", "relative_interferometric_phase", "rad", "finite", "A,B,C", "per interaction", "measured", "implemented_complex_phase"),
            ("idler round-trip loss", "SetupParams", "idler_round_trip_loss", "fraction", "0..1", "B,C", "return path", "measured", "implemented_once"),
            ("idler mode matching", "SetupParams", "idler_mode_matching", "fraction", "0..1", "A,B,C", "idler seed", "measured", "implemented_once"),
            ("idler return phase", "SetupParams", "idler_return_phase", "rad", "finite", "B,C", "return path", "measured", "implemented_complex_phase"),
            ("signal return path transmission", "SetupParams", "signal_return_path_transmission", "fraction", "0..1", "B,C", "return path", "measured", "implemented_detector_side"),
            ("signal return reflection amplitude", "SetupParams", "signal_return_reflection_amplitude", "field amplitude", "0..1", "B,C", "return path", "measured", "implemented_once"),
            ("signal return phase", "SetupParams", "signal_return_phase", "rad", "finite", "B,C", "return path", "measured", "implemented_complex_phase"),
            ("signal return mode overlap", "SetupParams", "signal_return_mode_overlap", "fraction", "0..1", "B,C", "return path", "measured", "implemented_once"),
            ("signal seeds second interaction", "SetupParams", "signal_return_seeds_second_interaction", "boolean", "true/false", "B,C", "return topology", "assumed", "continuous_single_mode_bogoliubov"),
            ("squeezing parameter", "LatentState", "spdc_gain_r", "dimensionless", "nonnegative or null", "A,B,C", "per interaction", "calibrated"),
            ("mean occupancy", "LatentState", "mean_photons_per_mode", "photons/mode", "nonnegative or null", "A,B,C", "per interaction"),
            ("interaction one squeezing parameter", "LatentState", "spdc_gain_r_NL1", "dimensionless", "nonnegative or null", "A,B,C", "interaction 1"),
            ("interaction two squeezing parameter", "LatentState", "spdc_gain_r_NL2", "dimensionless", "nonnegative or null", "A,B,C", "interaction 2"),
            ("interaction one fixed occupancy", "LatentState", "mean_photons_per_mode_NL1", "photons/mode", "nonnegative or null", "A,B,C", "interaction 1"),
            ("interaction two fixed occupancy", "LatentState", "mean_photons_per_mode_NL2", "photons/mode", "nonnegative or null", "A,B,C", "interaction 2"),
        ],
        "imaging": [
            ("imaging mode", "SetupParams", "imaging_mode", "name", "near_field/far_field"),
            ("propagation model", "SetupParams", "architecture_propagation_model", "name", "architecture_specific_complex_field_v1 or null", "A,B,C", "per profile", "assumed", "implemented_opt_in"),
            ("spatial transfer kernel", "SetupParams", "spatial_transfer_kernel", "name", "conditional_gaussian_v1", "A,B,C", "per profile", "assumed", "implemented_reduced_order"),
            ("object map sampling", "SetupParams", "object_map_pixel_um", "um/pixel", ">0", "A,B,C", "per object", "measured", "implemented"),
            ("near-field idler relay magnification Mu", "SetupParams", "near_field_idler_magnification", "dimensionless", ">0", "B", "idler relay", "measured", "implemented"),
            ("near-field signal relay magnification Ms", "SetupParams", "near_field_signal_magnification", "dimensionless", ">0", "B", "signal relay", "measured", "implemented"),
            ("far-field idler relay magnification", "SetupParams", "far_field_idler_relay_magnification", "dimensionless", ">0", "A,C", "idler relay", "measured", "implemented"),
            ("far-field signal relay magnification", "SetupParams", "far_field_signal_relay_magnification", "dimensionless", ">0", "A,C", "signal relay", "measured", "implemented"),
            ("object type", "SetupParams", "object_type", "name", "transmissive/reflective", "A,B,C,D", "per profile", "measured", "implemented"),
            ("object mode", "SetupParams", "object_mode", "name", "mask/slm/reference_slm", "A,B,C,D", "per object", "measured", "implemented"),
            ("catalog mask id", "SetupParams", "object_mask_id", "integer", "1..10", "A,B,C,D", "per object", "measured", "implemented"),
            ("object intensity transmission T", "SetupParams", "object_intensity_T", "dimensionless", "0..1 two-crystal transmissive", "A,D", "per object", "measured", "implemented"),
            ("object intensity reflectance R", "SetupParams", "object_intensity_R", "dimensionless", "0..1 single-crystal reflective", "B,C", "per object", "measured", "implemented"),
            ("object phase", "SetupParams", "object_phase_rad", "rad", "0..2pi", "A,B,C,D", "per object", "measured", "implemented"),
            ("object plane", "SetupParams", "object_plane", "name", "profile-specific position/Fourier plane", "A,B,C", "per profile", "measured", "implemented_declared_plane"),
            ("object intensity transmission or reflectance map", "SetupParams", "object_transmission_map", "reference selector", "ref.T", "A,B,C", "per object", "measured", "implemented"),
            ("object phase map", "SetupParams", "object_phase_map", "reference selector", "ref.phi", "A,B,C", "per object", "measured", "implemented"),
            ("object reflection amplitude", "SetupParams", "object_reflection_amplitude", "field amplitude", "0..1", "B,C reflective", "per object", "measured", "implemented_once"),
            ("object reflection phase", "SetupParams", "object_reflection_phase", "rad", "finite", "B,C reflective", "per object", "measured", "implemented_once"),
            ("outbound object coordinate transform", "SetupParams", "object_forward_transform", "name", "identity/flip_x/flip_y/rotate_180", "A,B,C", "per path", "measured", "implemented"),
            ("return object coordinate transform", "SetupParams", "object_backward_transform", "name", "identity/flip_x/flip_y/rotate_180", "B,C", "per path", "measured", "implemented"),
            ("object interaction count", "SetupParams", "object_interaction_count", "integer", "A=1; B/C transmissive=2; reflective=1", "A,B,C", "derived from topology", "derived", "validated"),
            ("camera plane", "SetupParams", "camera_center_x_px", "px", "finite camera origin; geometry via camera_plane_geometry_from_setup", "A,B,C", "per profile", "measured", "implemented"),
            ("camera plane origin y", "SetupParams", "camera_center_y_px", "px", "finite", "A,B,C", "per profile", "measured", "implemented"),
            ("apparatus crystal NL1 record", "SetupParams", "apparatus_crystal_NL1", "mapping", "null or measured slots", "A,B,C", "apparatus P domain", "measured", "structurally_supported_calibration_required"),
            ("apparatus crystal NL2 record", "SetupParams", "apparatus_crystal_NL2", "mapping", "null or measured slots; A only", "A only", "apparatus P domain", "measured", "structurally_supported_calibration_required"),
            ("calibration records", "SetupParams", "calibration_records", "mapping", "null or measured slots", "A,B,C", "calibration", "calibrated", "structurally_supported_calibration_required"),
            ("signal focal length", "SetupParams", "F2_mm", "mm", ">0"),
            ("idler focal length", "SetupParams", "F1_mm", "mm", ">0"),
            ("far-field undetected arm focal length", "SetupParams", "far_field_focal_mm", "mm", ">0 or inherited", "A,C"),
            ("internal QIUP magnification", None, None, "dimensionless", "derived F2*lambda_s/(F1*lambda_i)", "A,B,C", "shared", "derived", "derived_resolve_imaging_magnification"),
            ("signed object-to-camera magnification", None, None, "dimensionless", "derived from explicit relays, wavelengths, and focal lengths", "A,B,C", "per profile", "derived", "derived_resolved_spatial_mapping"),
            ("coordinate orientation", "SetupParams", "image_rotation_deg", "deg", "finite"),
            ("near-field correlation width", None, None, "px", "derived finite-crystal conditional Gaussian sigma", "B", "per imaging mode", "derived", "implemented_reduced_order"),
            ("far-field correlation width", None, None, "px", "derived pump-angular conditional Gaussian sigma", "A,C", "per imaging mode", "derived", "implemented_reduced_order"),
            ("signal and idler spatial coordinates", None, None, "um", "derived from explicit relay maps", "A,B,C", "per field", "derived", "implemented_scaled_resampling"),
            ("signal and idler transverse momentum", None, None, "1/m", "q_s=-q_i with lens coordinate maps", "A,C", "per field", "derived", "implemented_reduced_order"),
            ("ordinary optical PSF", None, None, "px", "derived resolve_blur_sigmas", "A,B,C", "shared", "derived", "implemented"),
            ("SPDC correlation-limited PSF", None, None, "px", "derived resolve_blur_sigmas", "A,B,C", "per imaging mode", "derived", "approximation"),
        ],
        "source_gain": [
            ("source mode", "SetupParams", "spdc_source_mode", "name", "fixed_nbar/calibrated_source"),
            ("gain regime selector", "SetupParams", "spdc_gain_mode", "name", "low_gain/high_gain default; transition from occupancy"),
            ("interaction one pump transmission", "SetupParams", "pump_pre_crystal_transmission", "fraction", "0..1"),
            ("two-crystal pump topology", "SetupParams", "two_crystal_pump_topology", "name", "branched_independent/sequential", "A", "shared"),
            ("two-crystal pump BS reflectance to NL1", "SetupParams", "pump_bs_reflectance", "fraction", "0..1 default 0.5", "A"),
            ("interaction two pump transmission for A", "SetupParams", "pump_to_NL2_transmission", "fraction", "0..1", "A"),
            ("inter-crystal pump transmission for sequential A", "SetupParams", "pump_between_crystals_transmission", "fraction", "0..1 or null", "A", "interaction 2", "measured"),
            ("return pump transmission for B/C", "SetupParams", "pump_return_transmission", "fraction", "0..1", "B,C"),
            ("calibrated gain coefficient", "SetupParams", "spdc_gain_per_sqrt_mW", "1/sqrt(mW)", "positive or null", "A,B,C", "per current shared calibration", "calibrated"),
            ("second interaction calibrated gain coefficient", "SetupParams", "spdc_gain_per_sqrt_mW_NL2", "1/sqrt(mW)", "positive or null", "A", "interaction 2", "calibrated"),
            ("gain calibration identifier", "SetupParams", "spdc_gain_calibration_id", "string", "non-empty or null", "A,B,C", "interaction 1/shared", "calibrated"),
            ("second gain calibration identifier", "SetupParams", "spdc_gain_calibration_id_NL2", "string", "non-empty or null", "A", "interaction 2", "calibrated"),
            ("independent backward/NL2 pump override", "SetupParams", "pump_power_NL2_override_mW", "mW", "nonnegative or null", "A,B,C", "interaction 2", "measured"),
            ("independent pump override calibration identifier", "SetupParams", "pump_power_NL2_override_calibration_id", "string", "non-empty when override is used", "A,B,C", "interaction 2", "calibrated"),
        ],
        "detector": [
            ("detector mode", "SetupParams", "detector_mode", "name", "analog_emccd/photon_counting"),
            ("selected signal output port", "SetupParams", "detector_output_port", "name", "plus/minus; omitted legacy resolves plus"),
            ("quantum efficiency", "SetupParams", "QE", "fraction", "0..1"),
            ("exposure duration", "LatentState", "exposure_s", "s", ">0"),
            ("EM gain", "LatentState", "em_gain", "dimensionless", ">=1"),
            ("EM multiplication model", "SetupParams", "emccd_model", "name", "legacy/register; analog always stochastic register"),
            ("post-EM read noise base", "SetupParams", "read_noise0_e", "e- rms", ">=0"),
            ("post-EM read noise latent span", "SetupParams", "read_noise_span_e", "e- rms", ">=0"),
            ("dark current base", "SetupParams", "dark0", "e-/pixel/s", ">=0"),
            ("clock induced charge", "SetupParams", "cic0", "e-/pixel", ">=0"),
            ("PRNU standard deviation", "SetupParams", "prnu_sigma", "fraction", ">=0"),
            ("hot pixel fraction at latent one", "SetupParams", "hot_frac_at_s1", "fraction", "0..1"),
            ("hot pixel electron amplitude low", "SetupParams", "hot_amp_lo", "e-/frame", ">=0"),
            ("hot pixel electron amplitude high", "SetupParams", "hot_amp_hi", "e-/frame", ">= hot_amp_lo"),
            ("full well saturation", "SetupParams", "full_well_e", "e-", ">0"),
            ("ADC conversion gain", "SetupParams", "g_adu", "ADU/e-", ">0"),
            ("ADC bias", "SetupParams", "bias_adu", "ADU", "finite"),
            ("ADC maximum code", "SetupParams", "max_adu", "ADU", ">0 and within bit depth"),
            ("ADC bit depth", "SetupParams", "adc_bit_depth", "bits", "integer 1..32; omitted legacy derives from max_adu"),
            ("sensor temperature", "SetupParams", "T_sensor_C", "deg C", "sensor operating range"),
        ],
    }


# Applicability rules for derived quantities (rows without a Python field),
# evaluated against the registered profile properties.
DERIVED_RULES = {
    "detector image dimensions": "all",
    "pump/signal/idler refractive indices": "all",
    "QPM phase mismatch": "all",
    "physical generation geometry overlap": "all",
    "interaction index": "all",
    "physical crystal identity": "all",
    "propagation direction": "all",
    "local pump power": "all",
    "internal QIUP magnification": "all",
    "signed object-to-camera magnification": "all",
    "near-field correlation width": "near_field",
    "far-field correlation width": "far_field",
    "signal and idler spatial coordinates": "all",
    "signal and idler transverse momentum": "far_field",
    "ordinary optical PSF": "all",
    "SPDC correlation-limited PSF": "all",
}

# Uncurated fields are grouped by the first matching name pattern.
GROUP_PATTERNS = (
    ("detector", r"^(QE|read_noise|dark|cic0|prnu|hot_|full_well|g_adu|bias_adu|max_adu|adc_|T_sensor|Eg_eV|"
                 r"em_gain|pc_threshold|accidental|emccd|detector_|exposure_s|num_frames|photon_counting|"
                 r"s_det|s_cool|use_physical_dark)"),
    ("physical_crystal", r"^(crystal_|qpm_|d33|d_eff|poling|T_pm|T_crystal|dT_|delta_T_|pol_|spdc_type|"
                         r"n_crystals|use_calibrated_poling|sync_crystal|apparatus|dlam_|s_thermal)"),
    ("source_gain", r"^(spdc_|pump_power|pump_.*transmission|pump_bs|pump_balance|pump_target|"
                    r"use_independent_pump|flux|pump_sigma|measured_power|fixed_source|use_physical_spdc|"
                    r"mean_photons|pump_power_mW|s_gain|s_pump)"),
    ("interaction", r"^(idler_|signal_|relative_|delta_L|L_coh|path\d|V0|V_|overlap|use_physical_two_beam|"
                    r"use_field_overlap|mottle|camera_filter|couple_pointing|enable_phase_step|phase_step|"
                    r"generation_|relay_|inherit_idler|phase_drift|vis_walk|phase_jitter|delta_phi|"
                    r"use_analytic_mode_overlap|s_align|T_env|alpha_|L_eff|beam_|use_physical_thermal|T_ref_span)"),
    ("imaging", r"^(magnification|aperture|NA$|airy|lens|F\d_|blur|sigma|derive_sigma|use_derived|defocus|"
                r"K_defocus|delta_z|n_medium|eta_|etalon|enable_|wavefront|couple_align|delta_M|image_|"
                r"object_|slm_|camera_center|s_focus|s_slm|use_physical_qiup|near_field|far_field|dM_over_M|"
                r"optical_background)"),
)


def _source_comments(path: Path, cls: str) -> dict[str, tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, tuple[int, str]] = {}
    for node in ast.walk(ast.parse("\n".join(lines))):
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                text = lines[st.lineno - 1]
                inline = text.split("#", 1)[1].strip() if "#" in text else ""
                block: list[str] = []
                k = st.lineno - 2
                while k >= 0 and lines[k].strip().startswith("#"):
                    block.insert(0, lines[k].strip().lstrip("#").strip())
                    k -= 1
                comment = " ".join(x for x in (inline, " ".join(block)) if x)
                out[st.target.id] = (st.lineno, re.sub(r"\s+", " ", comment))
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _letters(text: str | None) -> list[str]:
    return sorted(set(re.findall(r"\b[A-D]\b", text or "")))


def _group_for(name: str) -> str:
    for group, pattern in GROUP_PATTERNS:
        if re.search(pattern, name):
            return group
    return "shared_optics"


def _unit_for(name: str, type_text: str) -> str:
    if name in UNIT_OVERRIDES:
        return UNIT_OVERRIDES[name]
    if name in LATENT_KEYS:
        return "normalized latent severity"
    if "bool" in type_text:
        return "boolean"
    if name not in UNIT_SUFFIX_EXEMPT:
        for suffix, unit in UNIT_SUFFIXES:
            if name.endswith(suffix):
                return unit
    if "str" in type_text:
        return "name / identifier"
    if "dict" in type_text or "Mapping" in type_text:
        return "mapping"
    return "dimensionless"


def _enumerated_strings(value: str) -> list[str]:
    for obj in vars(setup_params).values():
        if isinstance(obj, (set, frozenset, tuple, list)) and value in obj and 1 < len(obj) <= 40 \
                and all(isinstance(m, str) for m in obj):
            return sorted(obj)
    return []


def _range_for(owner: str, name: str, type_text: str, default: Any) -> tuple[str, str]:
    if owner == "LatentState" and name in LATENT_KEYS:
        lab, stress = REGIME_BOUNDS["lab"][name], REGIME_BOUNDS["stress"][name]
        return f"lab {list(lab)}; stress {list(stress)}", "latents.REGIME_BOUNDS"
    sampled = {
        "num_frames": "sampled from {48, 64, 96, 128} (>=200 when s_thermal_eff > 0.55)",
        "em_gain": "sampled {200, 400, 800} (photon counting) or U(50, 300) (analog)",
        "exposure_s": "sampled U(0.01, 0.05) s (photon counting) or U(0.05, 0.3) s (analog)",
        "photon_counting": "mirror of SetupParams.detector_mode",
    }
    if name in sampled:
        return sampled[name], "latents.sample_latents"
    if "bool" in type_text:
        return "true/false", "type"
    if isinstance(default, str):
        options = _enumerated_strings(default)
        if options:
            return "one of " + "/".join(options), "setup_params enumeration"
    return "no tabulated range in source; admissibility enforced by validate_setup_params / profile validation", \
        "validate_setup_params"


def _role_for(owner: str, name: str | None, type_text: str) -> str:
    if name is None or name in DERIVED:
        return "derived"
    if owner == "LatentState" and name in SAMPLED:
        return "sampled"
    if name in METADATA or "calibration_id" in name or name == "setup_id":
        return "metadata"
    if name in CALIBRATION_DEPENDENT:
        return "calibration-dependent"
    if name in SELECTORS or "bool" in type_text or "str" in type_text:
        return "selector"
    return "fixed"


def _profile_properties() -> dict[str, dict[str, Any]]:
    out = {}
    for p in PROFILES:
        s = resolve_setup_selection(p)
        out[p] = {"architecture": s.spdc_architecture, "imaging_mode": s.imaging_mode}
    return out


def _derived_profiles(rule: str, props: dict[str, dict[str, Any]]) -> list[str]:
    if rule == "all":
        return list(PROFILES)
    if rule in ("near_field", "far_field"):
        return [p for p in PROFILES if props[p]["imaging_mode"] == rule]
    if rule in ("two_crystal", "single_crystal_double_pass"):
        return [p for p in PROFILES if props[p]["architecture"] == rule]
    raise KeyError(rule)


def load_audit(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / AUDIT_RELATIVE).read_text(encoding="utf-8"))


def build_spec(audit: dict[str, Any] | None = None) -> dict:
    audit = audit if audit is not None else load_audit()
    consumption = audit["fields"]
    setup = SetupParams()
    profile_setups = {p: resolve_setup_selection(p) for p in PROFILES}
    props = _profile_properties()
    owners = {
        "SetupParams": {f.name: f for f in fields(SetupParams)},
        "LatentState": {f.name: f for f in fields(LatentState) if f.name not in LATENT_SKIP},
    }
    comments = {
        "SetupParams": _source_comments(ROOT / "setup_params.py", "SetupParams"),
        "LatentState": _source_comments(ROOT / "latents.py", "LatentState"),
    }
    source_file = {"SetupParams": "setup_params.py", "LatentState": "latents.py"}
    rows: list[dict] = []
    seen: set[str] = set()

    def add(group: str, meaning: str, owner: str | None, name: str | None, unit: str | None,
            permitted_range: str | None, documented_v34: str | None, scope: str,
            value_class: str | None, status: str | None, curated: bool) -> None:
        row: dict[str, Any] = {"group": group, "physical_meaning": meaning}
        if owner is None:
            rule = DERIVED_RULES[meaning]
            consumed = _derived_profiles(rule, props)
            row.update({
                "python_field": None, "symbol": None, "unit": unit, "python_type": None, "default": None,
                "profile_defaults": None, "permitted_range": permitted_range,
                "range_source": "derived quantity", "parameter_role": "derived",
                "source": f"derived from registered profile properties (rule: {rule})",
                "configurations": consumed, "applicability_basis": f"profile property rule '{rule}'",
                "configurations_fixed_by_validation": [], "configurations_metadata_only": [],
                "audit_status": None,
            })
            value_class = value_class or "derived"
            status = status or "derived"
        else:
            f = owners[owner][name]
            key = f"{owner}.{name}"
            type_text = str(f.type)
            if owner == "SetupParams":
                default = getattr(setup, name)
                pdefaults = {p: _jsonable(getattr(profile_setups[p], name)) for p in PROFILES}
            else:
                default = None if f.default is MISSING else f.default
                pdefaults = None
            line, comment = comments[owner].get(name, (0, ""))
            entry = consumption[key]
            role = _role_for(owner, name, type_text)
            rng, rng_src = (permitted_range, "curated Version 3.4 inventory") if permitted_range \
                else _range_for(owner, name, type_text, default)
            consumed = [p for p in PROFILES if entry[p]["status"] == "physics"]
            fixed = [p for p in PROFILES if entry[p]["status"] == "validated_fixed"]
            meta = [p for p in PROFILES if entry[p]["status"] == "metadata_only"]
            if status is None:
                status = ("implemented" if consumed else
                          "validation_constrained" if fixed else
                          "recorded_only" if meta else "not_consumed_by_forward_model")
            row.update({
                "python_field": key, "symbol": SYMBOLS.get(name),
                "unit": unit or _unit_for(name, type_text), "python_type": type_text,
                "default": _jsonable(default), "profile_defaults": pdefaults,
                "permitted_range": rng, "range_source": rng_src, "parameter_role": role,
                "source": f"{source_file[owner]}:{line} {comment}".strip(),
                "configurations": consumed,
                "applicability_basis": "parameter_consumption_audit (fixed-seed perturbation)",
                "configurations_fixed_by_validation": fixed, "configurations_metadata_only": meta,
                "audit_status": {p: entry[p]["status"] for p in PROFILES},
            })
            if value_class is None:
                value_class = {"calibration-dependent": "calibrated", "sampled": "sampled",
                               "derived": "derived", "metadata": "metadata"}.get(role, "assumed")
            seen.add(key)
        role = row["parameter_role"]
        row["profile_defining"] = (not row["configurations"]
                                   and row["configurations_fixed_by_validation"] == list(PROFILES))
        row.update({
            "architecture_dependency": ",".join(row["configurations"]) or (
                "profile-defining (fixed per configuration)" if row["profile_defining"] else "none"),
            "documented_dependency_v3_4": documented_v34,
            "independently_configurable": scope,
            "value_class": value_class,
            "implementation_status": status,
            "calibration_status": CALIBRATION_STATUS_BY_ROLE[role],
            "curated_in_v3_4": curated,
        })
        rows.append(row)

    for group, specs in curated_rows().items():
        for spec in specs:
            meaning, owner, name, unit, rng, *rest = spec
            dep = rest[0] if len(rest) > 0 else "A,B,C,D"
            scope = rest[1] if len(rest) > 1 else "shared"
            vclass = rest[2] if len(rest) > 2 else None
            status = rest[3] if len(rest) > 3 else None
            add(group, meaning, owner, name, unit, rng, dep, scope, vclass, status, True)

    for owner, flds in owners.items():
        for name in flds:
            key = f"{owner}.{name}"
            if key in seen:
                continue
            line, comment = comments[owner].get(name, (0, ""))
            meaning = comment.split(". ")[0][:160] if comment else name.replace("_", " ")
            add(_group_for(name), meaning, owner, name, None, None, None, "shared", None, None, False)

    changes = []
    for row in rows:
        if row["documented_dependency_v3_4"] is None:
            continue
        before = _letters(row["documented_dependency_v3_4"])
        if before != row["configurations"]:
            changes.append({
                "parameter": row["python_field"] or row["physical_meaning"],
                "documented_v3_4": row["documented_dependency_v3_4"],
                "traced_v3_4_1": row["architecture_dependency"],
                "added": [p for p in row["configurations"] if p not in before],
                "removed": [p for p in before if p not in row["configurations"]],
                "basis": row["applicability_basis"],
            })

    return {
        "schema_version": "four_configuration_parameters_v3_4_1",
        "package_version": PACKAGE_VERSION,
        "release_classification": RELEASE_CLASSIFICATION,
        "calibration_status": CALIBRATION_STATUS,
        "calibration_statement": CALIBRATION_STATEMENT,
        "profiles": {
            "A": ["two_crystal", "far_field", 2, 2],
            "B": ["single_crystal_double_pass", "near_field", 1, 2],
            "C": ["single_crystal_double_pass", "far_field", 1, 2],
            "D": ["two_crystal", "near_field", 2, 2],
        },
        "wavelength_energy_relative_tolerance": 1e-4,
        "consumption_audit": {
            "file": AUDIT_RELATIVE,
            "schema": audit.get("schema"),
            "source_sha256": audit.get("source_sha256"),
            "status_meanings": AUDIT_STATUS_NOTE,
        },
        "column_meanings": {
            "configurations": "profiles in which the field changes simulated arrays (traced, not hand-written)",
            "configurations_fixed_by_validation": "profiles whose validation rejects every perturbation",
            "configurations_metadata_only": "profiles where the value is recorded but no array change was detected",
            "profile_defining": "every profile's validation fixes the value (selector owned by the configuration)",
            "documented_dependency_v3_4": "hand-written applicability in the Version 3.4 inventory (historical)",
            "parameter_role": "fixed | sampled | derived | calibration-dependent | selector | metadata",
            "calibration_status": "experimental calibration status; no value in this file is a laboratory measurement",
        },
        "parameters": rows,
        "applicability_changes_vs_v3_4": changes,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ",".join(str(v) for v in value) or "—"
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(spec: dict) -> str:
    rows = spec["parameters"]
    out = [
        f"# QIUP Simulator v{spec['package_version']} — Parameter Inventory",
        "",
        f"**Classification:** {spec['release_classification']}  ",
        f"**Calibration status:** {spec['calibration_status']}",
        "",
        f"> {spec['calibration_statement']}",
        "",
        "Generated by `build_parameter_spec.py` from the source dataclasses and the traced consumption audit "
        f"`{spec['consumption_audit']['file']}`. Do not edit by hand; "
        "`tests/test_parameter_inventory_consistency.py` fails if this file, the JSON specification, the audit, "
        "or the source code disagree.",
        "",
        "## How applicability is determined",
        "",
        "`Consumed in` lists the configurations in which a fixed-seed perturbation of the field changes the "
        "simulated arrays (`parameter_inventory_audit.py`). `Fixed by validation in` lists configurations whose "
        "profile validation rejects every perturbation (the value is set by the configuration). "
        "`Recorded only in` lists configurations where the value appears in run metadata but no array change was "
        "detected in the audit scenarios. Derived quantities take their applicability from the registered profile "
        "properties. Default values are software defaults, not laboratory measurements.",
        "",
        "Profiles: A = two crystal / far field, B = single crystal double pass / near field, "
        "C = single crystal double pass / far field, D = two crystal / near field.",
        "",
        "## Summary",
        "",
    ]
    roles: dict[str, int] = {}
    for r in rows:
        roles[r["parameter_role"]] = roles.get(r["parameter_role"], 0) + 1
    out.append(f"- Rows: {len(rows)} ({sum(1 for r in rows if r['python_field'])} source fields, "
               f"{sum(1 for r in rows if not r['python_field'])} derived quantities)")
    out.append("- Roles: " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))
    for p in PROFILES:
        out.append(f"- Consumed in {p}: {sum(1 for r in rows if p in r['configurations'])}")
    out.append(f"- Rows requiring experimental calibration: "
               f"{sum(1 for r in rows if r['calibration_status'] == NOT_CALIBRATED)} "
               f"(status `{NOT_CALIBRATED}`)")
    out += ["", "## Configuration D", ""]
    d_only = [r for r in rows if r["configurations"] == ["D"]]
    d_and_b = [r for r in rows if r["configurations"] == ["B", "D"]]
    d_and_a = [r for r in rows if r["configurations"] == ["A", "D"]]
    out.append("Fields consumed only by D: " + (", ".join(f"`{_cell(r['python_field'] or r['physical_meaning'])}`"
                                                          for r in d_only) or "none") + ".")
    out.append("")
    out.append("Fields shared by the two near-field profiles (B, D): " + (", ".join(
        f"`{_cell(r['python_field'] or r['physical_meaning'])}`" for r in d_and_b) or "none") + ".")
    out.append("")
    out.append("Fields shared by the two two-crystal profiles (A, D): " + (", ".join(
        f"`{_cell(r['python_field'] or r['physical_meaning'])}`" for r in d_and_a) or "none") + ".")
    out += ["", "## Fields with no detected forward effect in any configuration", "",
            "These fields are profile-defining selectors, identifiers/metadata, legacy aliases, or values whose "
            "effect is gated by a switch not exercised in the audit scenarios. They are reported, not removed. "
            "The sampled latents `s_gain` (legacy severity, unused by the Physical Mode source), `s_thermal` "
            "(acts through latent couplings and acquisition-length repair during sampling) and "
            "`photon_counting` (mirror of `SetupParams.detector_mode`, overwritten by the forward model) "
            "are not read by the forward model itself.", "",
            "| Parameter | Role | Audit status (A/B/C/D) | Source note |", "|---|---|---|---|"]
    for r in rows:
        if r["python_field"] and not r["configurations"]:
            status = "/".join(r["audit_status"][p] for p in PROFILES)
            out.append(f"| `{r['python_field']}` | {r['parameter_role']}"
                       f"{' (profile-defining)' if r['profile_defining'] else ''} | {status} | "
                       f"{_cell(r['source'][:140])} |")
    out += ["", "## Applicability changes relative to the Version 3.4 inventory", "",
            "| Parameter | Documented in 3.4 | Traced in 3.4.1 | Added | Removed |",
            "|---|---|---|---|---|"]
    for c in spec["applicability_changes_vs_v3_4"]:
        out.append(f"| `{_cell(c['parameter'])}` | {_cell(c['documented_v3_4'])} | {_cell(c['traced_v3_4_1'])} | "
                   f"{_cell(c['added'])} | {_cell(c['removed'])} |")
    for group in GROUPS:
        out += ["", f"## {group}", "",
                "| Parameter | Symbol | Description | Unit | Default | Range | Consumed in | Fixed by validation in "
                "| Recorded only in | Role | Calibration status |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["group"] != group:
                continue
            name = f"`{r['python_field']}`" if r["python_field"] else f"*{r['physical_meaning']}* (derived)"
            consumed = "profile-defining" if r["profile_defining"] else _cell(r["configurations"])
            out.append(
                f"| {name} | {_cell(r['symbol'])} | {_cell(r['physical_meaning'])} | {_cell(r['unit'])} | "
                f"{_cell(r['default'])} | {_cell(r['permitted_range'])} | {consumed} | "
                f"{_cell(r['configurations_fixed_by_validation'])} | {_cell(r['configurations_metadata_only'])} | "
                f"{r['parameter_role']} | {r['calibration_status']} |"
            )
    out.append("")
    return "\n".join(out)


def write_outputs(root: Path = ROOT) -> tuple[Path, Path]:
    spec = build_spec(load_audit(root))
    target = root / SPEC_RELATIVE
    # ASCII-escaped: legacy consumers read this file with the platform default codec.
    target.write_text(json.dumps(spec, indent=2, ensure_ascii=True, default=str) + "\n",
                      encoding="utf-8", newline="\n")
    md = root / MARKDOWN_RELATIVE
    md.write_text(render_markdown(spec), encoding="utf-8", newline="\n")
    return target, md


if __name__ == "__main__":
    for path in write_outputs():
        print(path)
