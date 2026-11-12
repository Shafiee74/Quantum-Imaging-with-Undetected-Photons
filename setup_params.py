"""
QIUP SetupParams — physical parameterization for the digital twin.

Goal:
  SetupParams (lab units) → ResolvedForward (pixel-domain numbers) → simulate

Policy:
  Default profile uses physical pump waist ``pump_waist_um`` (100–1000 µm, default
  250 µm), EMCCD ``pixel_um`` (6.5–24 µm, default 16 µm), magnification M=1,
  and crystal length ``crystal_length_mm`` (2–20 mm; default 2 mm for
  ``double_blur``, 10 mm for ``single_blur``).  σ_SPDC is
  derived from PDF §3.1 + eq. 50, not a free constant.  ``legacy_matched()``
  reproduces Phase-0/1 numerical defaults (N/3.2 envelope) for regression only.

Units:
  um, nm, mm, °C, rad, e-, px, s, dimensionless.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from numbers import Integral, Real
import hashlib
import json

import numpy as np

# ---------------------------------------------------------------------------
# Constants used to match legacy envelope (Phase-4 D7):
#   These are LEGACY GEOMETRY labels — NOT a physical waist claim.
#   env = exp( -0.9 * ((x-cx)/(N/3.2))^2 - 0.9 * ((y-cy)/(N/3.2))^2 )
# ↔ I ∝ exp( -2 r^2 / w^2 )  ⇒  w = (N/3.2) * sqrt(2/0.9)
# Physical waist requires use_legacy_beam_match=False + pump_waist_um.
# ---------------------------------------------------------------------------
_LEGACY_BEAM_SIGMA_NORM = 3.2
_LEGACY_BEAM_FALLOFF = 0.9 

# Physical pump waist (user-editable via YAML; validated on load/override)
PUMP_WAIST_UM_DEFAULT = 250.0
PUMP_WAIST_UM_MIN = 100.0
# Extended to support the experimentally reported C_lab pump spot. A measured
# 2 mm 1/e^2 intensity diameter corresponds to w_p=1000 um. This is a stated
# measurement interpretation, not a camera-image fit.
PUMP_WAIST_UM_MAX = 1000.0
IDLER_WAIST_UM_MIN = 100.0
IDLER_WAIST_UM_MAX = 350.0

# EMCCD sensor pixel pitch [µm] — quantum-imaging cameras (Andor iXon, Nüvü, Hamamatsu
# ImagEM / C9100, Photometrics Evolve, etc.). Typical cluster: 13–16 µm; default 16 µm.
PIXEL_UM_DEFAULT = 16.0
PIXEL_UM_MIN = 6.5   # e.g. high-res / small-pitch scientific sensors
PIXEL_UM_MAX = 24.0  # e.g. larger-pixel legacy EMCCDs

# Beam pointing jitter scale (PDF §2.5–2.6): σ_beam = α·w_p,pixel  with 0.01 ≤ α ≤ 0.10
POINTING_ALPHA_DEFAULT = 0.03
POINTING_ALPHA_MIN = 0.01
POINTING_ALPHA_MAX = 0.10

# Nonlinear crystal length L [mm] for SPDC correlation width (PDF §3.1 / eq. 50).
# Physics-team-approved policy: user-editable in [2, 20]. Blur-linked defaults:
#   single_blur  → 10 mm
#   double_blur  → 2 mm per crystal / pass ("each one")
CRYSTAL_LENGTH_POLICY_ID = "physics_team_user_variable_v1"
THREE_CONFIGURATION_PHYSICS_POLICY_ID = "three_configuration_physics_team_v2"
CRYSTAL_LENGTH_MM_MIN = 2.0
CRYSTAL_LENGTH_MM_MAX = 20.0
CRYSTAL_LENGTH_MM_DEFAULT_SINGLE_BLUR = 10.0
CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR = 2.0
# Package default follows BLUR_MODEL_DEFAULT (= double_blur).
CRYSTAL_LENGTH_MM_DEFAULT = CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR

# Temporal coherence length L_c [mm] for V_coh (PDF §4.1 eq. 65–66)
L_COH_MM_DEFAULT = 0.20
L_COH_MM_MIN = 0.02
L_COH_MM_MAX = 0.50

# Path mismatch ΔL [mm] for V_coh (PDF §4.1 eq. 65). Same numeric window as L_c.
DELTA_L_MM_DEFAULT = 0.01
DELTA_L_MM_MIN = -L_COH_MM_MAX
DELTA_L_MM_MAX = L_COH_MM_MAX

# Crystal oven / phase-match temperature (PDF §5 eq. 80: T_pm = T_pm0 + ΔT)
T_CRYSTAL_C_MIN = 20.0
T_CRYSTAL_C_MAX = 90.0
# Generic stock-profile default (single-crystal benchmark NL1 reference); C_lab YAML overrides independently.
T_CRYSTAL_C_DEFAULT = 83.7
T_CRYSTAL_C_STEP = 0.1

# ---------------------------------------------------------------------------
# PDF §5 physical thermal constants (lab-editable defaults = Lemos-like)
# ---------------------------------------------------------------------------
# §5.1 — ppKTP type-0 (Lemos et al.: 1×2×2 mm³, Λ=9.675 µm, H→H+H)
CRYSTAL_MATERIAL_DEFAULT = "ppKTP"
QPM_TYPE_DEFAULT = "type0"
QPM_ORDER_DEFAULT = 1  # m_QPM in K_g=2πm/Λ and d_eff (V-P1-03); usually fixed, not a sweep
QPM_DUTY_CYCLE_DEFAULT = 0.5  # D; first-order square poling, report eq. 41
# Comment 3 — how d_eff_pm_per_V is chosen when d33 and/or direct d_eff are set.
DEFF_PRECEDENCE_AUTO = "auto"
DEFF_PRECEDENCE_DIRECT = "direct"
DEFF_PRECEDENCE_DERIVED = "derived"
DEFF_PRECEDENCE_DEFAULT = DEFF_PRECEDENCE_AUTO
_DEFF_PRECEDENCE_ALIASES = {
    "auto": DEFF_PRECEDENCE_AUTO,
    "default": DEFF_PRECEDENCE_AUTO,
    "direct": DEFF_PRECEDENCE_DIRECT,
    "explicit": DEFF_PRECEDENCE_DIRECT,
    "d_eff": DEFF_PRECEDENCE_DIRECT,
    "deff": DEFF_PRECEDENCE_DIRECT,
    "derived": DEFF_PRECEDENCE_DERIVED,
    "from_d33": DEFF_PRECEDENCE_DERIVED,
    "d33": DEFF_PRECEDENCE_DERIVED,
}
DEFF_RESOLUTION_NONE = "none"
DEFF_RESOLUTION_DIRECT = "direct"
DEFF_RESOLUTION_DERIVED = "derived"  # legacy alias; prefer DEFF_SOURCE_DERIVED
# Comment 3.3 — metadata ``d_eff_source`` vocabulary
DEFF_SOURCE_NONE = "none"
DEFF_SOURCE_DIRECT = "direct"
DEFF_SOURCE_DERIVED = "derived_from_d33"
DEFF_CONSISTENCY_RTOL = 1.0e-6
DEFF_CONSISTENCY_ATOL = 1.0e-9
POLING_PERIOD_UM_DEFAULT = 9.675  # Lemos
POLING_PERIOD_UM_MIN = 1.0
POLING_PERIOD_UM_MAX = 50.0
POLARIZATION_DEFAULT = "H"  # type-0: pump, signal, idler all H
# V-P1-08 — supported physics domain (fail-closed; no silent unsupported configs)
SUPPORTED_CRYSTAL_MATERIALS = frozenset({"ppktp", "pp-ktp", "pp_ktp"})
SUPPORTED_QPM_TYPES = frozenset({"type0", "type-0", "type_0", "0"})
SUPPORTED_POLARIZATIONS = frozenset({"h", "horizontal", "type0"})  # type0 alias → H
_CRYSTAL_MATERIAL_ALIASES = {
    "ppktp": CRYSTAL_MATERIAL_DEFAULT,
    "pp-ktp": CRYSTAL_MATERIAL_DEFAULT,
    "pp_ktp": CRYSTAL_MATERIAL_DEFAULT,
}
_QPM_TYPE_ALIASES = {
    "type0": QPM_TYPE_DEFAULT,
    "type-0": QPM_TYPE_DEFAULT,
    "type_0": QPM_TYPE_DEFAULT,
    "0": QPM_TYPE_DEFAULT,
}
_POLARIZATION_ALIASES = {
    "h": POLARIZATION_DEFAULT,
    "horizontal": POLARIZATION_DEFAULT,
    "type0": POLARIZATION_DEFAULT,  # reviewer shorthand: type-0 ⇒ H/H/H
}
KTP_CTE_PER_C = 6.7e-6  # Λ(T)=Λ0(1+αΔT); order-of-magnitude along poling
# Single material T_ref for ktp_nz(): Kato & Takaoka, Appl. Opt. 41, 5040 (2002)
# [report ref. 17]. Their n_z Sellmeier equations are given at 20 °C; dn_z/dT
# then shifts from this temperature. Do not introduce a second Sellmeier T_ref.
KTP_SELLMEIER_T_REF_C = 20.0

# SI constants for report eq. 24 (k_SPDC,0); c is exact.
SPEED_OF_LIGHT_M_PER_S = 299792458.0
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
# Report eq. 30: P_W = 10^{-3} P_mW. All √10^{-3} factors must use this.
MILLIWATT_TO_WATT = 1.0e-3
# Report §7.8: O_geom must not include e^{i Δk_QPM z}. Δk_QPM enters the
# gain (report §7.9 / eq. 47–51), not A(z). A future Φ(q_s, q_i) model
# (eq. 37) must replace this split as one piece, not add both.
OGEOM_INCLUDES_QPM_PHASE = False
# Step-2: pump transverse profile (distinct from total pump power).
PUMP_PROFILE_GAUSSIAN = "gaussian"
PUMP_PROFILE_UNIFORM_EFFECTIVE = "uniform_effective"
PUMP_PROFILE_MEASURED_2D = "measured_2d"
_PUMP_PROFILE_TYPES = frozenset({
    PUMP_PROFILE_GAUSSIAN,
    PUMP_PROFILE_UNIFORM_EFFECTIVE,
})
# measured_2d is recognized only to fail closed with an explicit deferred reason.
_PUMP_PROFILE_DEFERRED_TYPES = frozenset({
    PUMP_PROFILE_MEASURED_2D,
    "measured",
    "measured_raw",
    "measured_map",
})
PUMP_PROFILE_MEASURED_2D_STATUS = (
    "DEFERRED - PHYSICAL/CALIBRATION SPECIFICATION REQUIRED"
)
# Type-0 collinear QPM: crystal orientation is the tilt of the poling grating
# relative to the lab pump axis. Outside this domain the model fails closed.
CRYSTAL_ORIENTATION_MAX_ABS_DEG = 5.0
CRYSTAL_ORIENTATION_MODEL_ID = "type0_collinear_grating_projection_v1"
# SPDC spatial correlation remains a reduced-order approximation (not a full PSF).
SPDC_CORRELATION_PSF_IMPLEMENTATION_STATUS = "approximation"
SPDC_CORRELATION_PSF_MODEL_ID = "conditional_gaussian_v1"
# Report §7.11: QIUP is one crystal, two SPDC interactions. n_crystals is
# hardware inventory and must not set the number of SPDC operating points.
SPDC_N_INTERACTIONS = 2
TWO_CRYSTAL_PUMP_BRANCHED = "branched_independent"
TWO_CRYSTAL_PUMP_SEQUENTIAL = "sequential"
# Ideal Lemos/Zou–Wang–Mandel BS1: half the laser to each crystal.
PUMP_BS_REFLECTANCE_DEFAULT = 0.5
_TWO_CRYSTAL_PUMP_TOPOLOGIES = {
    TWO_CRYSTAL_PUMP_BRANCHED,
    TWO_CRYSTAL_PUMP_SEQUENTIAL,
}
# Explicit architecture switch. Default keeps the paper path. n_crystals
# must not select this; YAML without the field stays single-crystal.
SPDC_ARCHITECTURE_SINGLE = "single_crystal_double_pass"
SPDC_ARCHITECTURE_TWO = "two_crystal"
SPDC_ARCHITECTURE_DEFAULT = SPDC_ARCHITECTURE_SINGLE
ARCHITECTURE_PROFILE_A = "A"
ARCHITECTURE_PROFILE_B = "B"
ARCHITECTURE_PROFILE_C = "C"
ARCHITECTURE_PROFILE_D = "D"
_SPDC_ARCHITECTURE_ALIASES = {
    "single_crystal_double_pass": SPDC_ARCHITECTURE_SINGLE,
    "single_crystal": SPDC_ARCHITECTURE_SINGLE,
    "double_pass": SPDC_ARCHITECTURE_SINGLE,
    "two_crystal": SPDC_ARCHITECTURE_TWO,
    "two_crystals": SPDC_ARCHITECTURE_TWO,
}

# CFG-P0-00 — frozen approved-system identity (not a free experimental knob).
# Official runs must load this YAML and match the recorded SHA-256 hashes.
CANONICAL_SETUP_ID = "setup_personal_lab_canonical_v1"
CANONICAL_YAML_RELATIVE = "configs/setup_personal_lab.yaml"
CANONICAL_HASH_RELATIVE = "configs/canonical_config_hash.json"
# Canonical YAML uses double_blur → default L = 2 mm. L itself is NOT frozen:
# users may set any crystal_length_mm in [2, 20] on official runs.
CANONICAL_CRYSTAL_LENGTH_MM = CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR
CANONICAL_N_CRYSTALS = 1
CANONICAL_LAMBDA_P_NM = 532.0
CANONICAL_LAMBDA_S_NM = 810.0
CANONICAL_LAMBDA_I_NM = 1550.0
# Mathematically exact pump satisfying 1/λp = 1/λs + 1/λi for 810/1550 nm.
# Use only for energy-identity mathematics — NOT as a substitute for the
# nominal/publication pump wavelength (532.0 nm) to force U01 residual to zero.
ENERGY_EXACT_LAMBDA_P_FROM_810_1550_NM = 1.0 / (
    1.0 / CANONICAL_LAMBDA_S_NM + 1.0 / CANONICAL_LAMBDA_I_NM
)  # ≈ 531.9915254237288 nm
CANONICAL_FROZEN_FIELDS: dict[str, Any] = {
    "n_crystals": CANONICAL_N_CRYSTALS,
    "spdc_architecture": SPDC_ARCHITECTURE_SINGLE,
    "lambda_p_nm": CANONICAL_LAMBDA_P_NM,
    "lambda_s_nm": CANONICAL_LAMBDA_S_NM,
    "lambda_i_nm": CANONICAL_LAMBDA_I_NM,
    "detector_mode": "analog_emccd",
    "blur_model": "double_blur",
    "spdc_gain_mode": "low_gain",
    "spdc_source_mode": "fixed_nbar",
}

# Intensity-level QIUP imaging plane (not a full biphoton Φ model).
# near_field  → position correlations; σ dominated by crystal length L
# far_field   → momentum correlations; σ dominated by pump waist + f_u
IMAGING_MODE_NEAR = "near_field"
IMAGING_MODE_FAR = "far_field"
IMAGING_MODE_DEFAULT = IMAGING_MODE_NEAR
_IMAGING_MODE_ALIASES = {
    "near_field": IMAGING_MODE_NEAR,
    "nearfield": IMAGING_MODE_NEAR,
    "near": IMAGING_MODE_NEAR,
    "nf": IMAGING_MODE_NEAR,
    "position": IMAGING_MODE_NEAR,
    "image_plane": IMAGING_MODE_NEAR,
    "far_field": IMAGING_MODE_FAR,
    "farfield": IMAGING_MODE_FAR,
    "far": IMAGING_MODE_FAR,
    "ff": IMAGING_MODE_FAR,
    "momentum": IMAGING_MODE_FAR,
    "fourier": IMAGING_MODE_FAR,
    "fourier_plane": IMAGING_MODE_FAR,
}

# Architecture-specific complex-field propagation used by profiles
# A/B/C/D.  ``None`` preserves the reviewed reduced-order path for historical
# configurations and therefore is intentionally not normalized to the new
# model implicitly.
ARCHITECTURE_PROPAGATION_COMPLEX_V1 = "architecture_specific_complex_field_v1"
SPATIAL_TRANSFER_CONDITIONAL_GAUSSIAN_V1 = "conditional_gaussian_v1"
OBJECT_TYPE_TRANSMISSIVE = "transmissive"
OBJECT_TYPE_REFLECTIVE = "reflective"
OBJECT_TRANSFORMS = frozenset({"identity", "flip_x", "flip_y", "rotate_180"})
OBJECT_MAP_TRANSMISSION = "ref.T"
OBJECT_MAP_PHASE = "ref.phi"
OBJECT_PLANE_A = "idler_fourier_plane_between_crystals"
OBJECT_PLANE_B = "idler_near_field_image_plane"
OBJECT_PLANE_C = "idler_far_field_fourier_plane"
OBJECT_PLANE_D = "idler_image_plane_between_crystals"
OBJECT_DEVICE_TRANSMISSIVE_SLM = "transmissive_slm"
OBJECT_DEVICE_REFLECTIVE_SLM = "reflective_slm"
OBJECT_DEVICE_MASK = "binary_mask"
OBJECT_MODE_REFERENCE_SLM = "reference_slm"
OBJECT_MODE_SLM = "slm"
OBJECT_MODE_MASK = "mask"
OBJECT_MODE_GENERIC_RESEARCH = "generic_research_object"
OBJECT_MASK_ID_MIN = 1
OBJECT_MASK_ID_MAX = 10
PHYSICAL_CRYSTAL_NL = "NL"
PHYSICAL_CRYSTAL_NL1 = "NL1"
PHYSICAL_CRYSTAL_NL2 = "NL2"
CRYSTAL_MATCHING_MATCHED = "matched_nominal"
CRYSTAL_MATCHING_MISMATCH = "mismatch_study"
PUMP_BALANCE_MATCHED = "matched_nominal"
PUMP_BALANCE_IMBALANCE = "imbalance_study"
FIXED_SOURCE_INPUT_NBAR = "nbar"
FIXED_SOURCE_INPUT_MEASURED_POWER = "measured_power"
CAMERA_FILTER_PRESET_PAPER = "paper_3nm"
CAMERA_FILTER_PRESET_LAB = "laboratory_10nm"
CAMERA_FILTER_PRESET_CUSTOM = "custom"
CAMERA_FILTER_MODEL_GAUSSIAN = "gaussian"
CAMERA_FILTER_MODEL_MEASURED = "measured"
PLANCK_CONSTANT_J_S = 6.62607015e-34

# Blur chain (V-P0-01 companion): keep both setups switchable.
# single_blur  → Object → Optical PSF → Detector  (no SPDC correlation stage)
# double_blur  → Object → SPDC correlation PSF → Optical PSF → Detector
# Effective Gaussian width (diagnostics): σ_total = σ_opt (single) or
# √(σ_SPDC² + σ_opt²) (double). Stages stay separate in the forward path.
BLUR_MODEL_SINGLE = "single_blur"
BLUR_MODEL_DOUBLE = "double_blur"
BLUR_MODEL_DEFAULT = BLUR_MODEL_DOUBLE
_BLUR_MODEL_ALIASES = {
    "single_blur": BLUR_MODEL_SINGLE,
    "single": BLUR_MODEL_SINGLE,
    "optical_only": BLUR_MODEL_SINGLE,
    "double_blur": BLUR_MODEL_DOUBLE,
    "double": BLUR_MODEL_DOUBLE,
    "spdc_and_optical": BLUR_MODEL_DOUBLE,
}

# SPDC gain regime switch (V-P0-03). Independent of blur_model.
# low_gain  → default nbar = 0.001  (regime: nbar < 0.01)
# high_gain → default nbar = 5.0    (regime: nbar >= 1)
SPDC_GAIN_MODE_LOW = "low_gain"
SPDC_GAIN_MODE_HIGH = "high_gain"
SPDC_GAIN_MODE_DEFAULT = SPDC_GAIN_MODE_LOW
SPDC_GAIN_MODE_LOW_NBAR = 0.001
SPDC_GAIN_MODE_HIGH_NBAR = 5.0
_SPDC_GAIN_MODE_ALIASES = {
    "low_gain": SPDC_GAIN_MODE_LOW,
    "low": SPDC_GAIN_MODE_LOW,
    "lg": SPDC_GAIN_MODE_LOW,
    "high_gain": SPDC_GAIN_MODE_HIGH,
    "high": SPDC_GAIN_MODE_HIGH,
    "hg": SPDC_GAIN_MODE_HIGH,
}

# V-P1-01 — SPDC source occupancy policy (explicit; no silent surrogate).
# fixed_nbar         → use requested nbar / spdc_gain_mode shortcut (dataset OK)
# calibrated_source  → pump → SPDC generation model → nbar (physics validation)
SPDC_SOURCE_FIXED = "fixed_nbar"
SPDC_SOURCE_CALIBRATED = "calibrated_source"
SPDC_SOURCE_DEFAULT = SPDC_SOURCE_FIXED
_SPDC_SOURCE_MODE_ALIASES = {
    "fixed_nbar": SPDC_SOURCE_FIXED,
    "fixed": SPDC_SOURCE_FIXED,
    "nbar": SPDC_SOURCE_FIXED,
    "preset": SPDC_SOURCE_FIXED,
    "gain_mode": SPDC_SOURCE_FIXED,
    "calibrated_source": SPDC_SOURCE_CALIBRATED,
    "calibrated": SPDC_SOURCE_CALIBRATED,
    "physical": SPDC_SOURCE_CALIBRATED,
    "predictive": SPDC_SOURCE_CALIBRATED,
    "pump": SPDC_SOURCE_CALIBRATED,
}

# V-P1-07 — explicit detector path (never infer from flux / EM gain / latents).
DETECTOR_MODE_ANALOG = "analog_emccd"
DETECTOR_MODE_PC = "photon_counting"
# Paper / thesis path is analog EMCCD; PC remains an explicit opt-in.
DETECTOR_MODE_DEFAULT = DETECTOR_MODE_ANALOG
_DETECTOR_MODE_ALIASES = {
    "analog_emccd": DETECTOR_MODE_ANALOG,
    "analog": DETECTOR_MODE_ANALOG,
    "emccd": DETECTOR_MODE_ANALOG,
    "adu": DETECTOR_MODE_ANALOG,
    "photon_counting": DETECTOR_MODE_PC,
    "pc": DETECTOR_MODE_PC,
    "photon": DETECTOR_MODE_PC,
    "counting": DETECTOR_MODE_PC,
}

# Task 8: the main product is one signal-port image. Keep the selected port
# explicit in new profiles while resolving omitted legacy values to I_plus.
DETECTOR_OUTPUT_PORT_PLUS = "plus"
DETECTOR_OUTPUT_PORT_MINUS = "minus"
DETECTOR_OUTPUT_PORT_DEFAULT = DETECTOR_OUTPUT_PORT_PLUS
_DETECTOR_OUTPUT_PORT_ALIASES = {
    "plus": DETECTOR_OUTPUT_PORT_PLUS,
    "i_plus": DETECTOR_OUTPUT_PORT_PLUS,
    "+": DETECTOR_OUTPUT_PORT_PLUS,
    "minus": DETECTOR_OUTPUT_PORT_MINUS,
    "i_minus": DETECTOR_OUTPUT_PORT_MINUS,
    "-": DETECTOR_OUTPUT_PORT_MINUS,
}

# Object / SLM preprocessing (V-P0-08 / V-P0-09).
# calibrated          → keep physical units; no content min-max normalize
# visualization_only  → opt-in min-max normalize for display/demo only
OBJECT_PREPROCESS_CALIBRATED = "calibrated"
OBJECT_PREPROCESS_VIS = "visualization_only"
OBJECT_PREPROCESS_DEFAULT = OBJECT_PREPROCESS_CALIBRATED
_OBJECT_PREPROCESS_ALIASES = {
    "calibrated": OBJECT_PREPROCESS_CALIBRATED,
    "calib": OBJECT_PREPROCESS_CALIBRATED,
    "physical": OBJECT_PREPROCESS_CALIBRATED,
    "physics": OBJECT_PREPROCESS_CALIBRATED,
    "visualization_only": OBJECT_PREPROCESS_VIS,
    "visualization": OBJECT_PREPROCESS_VIS,
    "viz": OBJECT_PREPROCESS_VIS,
    "display": OBJECT_PREPROCESS_VIS,
    "minmax": OBJECT_PREPROCESS_VIS,
}

# §5.2 — air-path OPD: ΔL_th = α_air · L · ΔT  (α = dn_air/dT, Edlén-type)
ALPHA_AIR_PATH_PER_C = -9.66e-7  # 1/°C
L_EFF_PATH_MM_DEFAULT = 0.20  # effective mismatched path [mm] (Lemos L_c scale)
# Legacy combined αL kept only for migration / legacy_matched
ALPHA_L_MM_PER_C_DEFAULT = abs(ALPHA_AIR_PATH_PER_C * L_EFF_PATH_MM_DEFAULT)

# §5.3 — oven thermal time constant (FIXED; not a SetupParams knob)
OVEN_TAU_S = 60.0
THETA0_RAD = 0.0  # aligned signal beams → initial interference phase 0

# §5.4 / §5.7 / §5.9 — imaging levers (Lemos: F1=75 mm, F2=150 mm)
F1_MM_DEFAULT = 75.0
F2_MM_DEFAULT = 150.0
CTE_MOUNT_PER_C = 23.0e-6  # Al-scale mount; Δx = α·F1·ΔT

# §5.5 — Thorlabs N-BK7 singlet (common QI / Lemos-style confocal lenses)
LENS_MATERIAL_DEFAULT = "N-BK7"
NBK7_N = 1.5168
NBK7_DNDT_PER_C = 2.4e-6
NBK7_CTE_PER_C = 7.1e-6
LENS_FOCAL_MM_DEFAULT = F1_MM_DEFAULT  # 7.5 cm

# PDF §6.4 eq. 142–144: diffraction floor from Airy + lab geometry
#   r_Airy = 0.61 λ_s / NA
#   NA = D / (2 f)
#   σ_diff = η · M · r_Airy / p
# Defaults match prior NA=0.04751 at f=75 mm → D = 2·NA·f
NA_MATCH_LEGACY = 0.04751
APERTURE_DIAMETER_MM_DEFAULT = 2.0 * NA_MATCH_LEGACY * LENS_FOCAL_MM_DEFAULT  # ≈7.1265 mm
AIRY_TO_GAUSS_ETA_DEFAULT = 1.0  # no extra Airy↔Gaussian scale (calibration knob)
SIGMA_DIFFRACTION_PX_LEGACY = 0.65  # only legacy_matched()

# PDF §6.5–6.9 defocus / circle of confusion
#   c ≈ (NA / n) · Δz
#   K_defocus = η_def · M · NA / (n · p)   [px/µm]
#   σ_defocus = K_defocus · |Δz|
#   s_focus = |Δz| / Δz_max
#   σ_opt = √(σ_diff² + σ_defocus²)
N_MEDIUM_DEFAULT = 1.0  # air
ETA_DEFOCUS_DEFAULT = 1.0
DELTA_Z_MAX_UM_DEFAULT = 100.0
# Legacy robustness span (PDF §6.8 worst-case ≈3.6 px); not typical lab error
DEFOCUS_SPAN_PX_LEGACY = 3.6
K_DEFOCUS_PX_PER_UM_LEGACY = DEFOCUS_SPAN_PX_LEGACY / DELTA_Z_MAX_UM_DEFAULT  # 0.036

# PDF §6.13/6.15 — residual aberration from wavefront RMS (no Zernike)
#   W_rms [waves @ λ_s] → σ_aberr = η_ab · W_waves · (M · r_Airy / p)
# C3 Fan path phase: δφ_rms = 2π W_waves + δφ_align_span · s_align
ETA_ABERRATION_DEFAULT = 1.0
WAVEFRONT_RMS_WAVES_DEFAULT = 0.0
WAVEFRONT_CORR_PX_DEFAULT = 12.0
DELTA_PHI_ALIGN_SPAN_RAD_DEFAULT = 0.35  # extra path-phase RMS at s_align=1 [rad]

# PDF §6.17–6.18 geometric IQ (≠ optical blur):
#   ε_M = ΔM/M → image_scale = 1 + ε_M; θ = image_rotation_deg
DELTA_M_REL_DEFAULT = 0.0  # ε_M relative magnification error

EMCCD_PIXEL_REFERENCE: dict[str, Any] = {
    "role": "hardware",
    "default_um": PIXEL_UM_DEFAULT,
    "allowed_range_um": [PIXEL_UM_MIN, PIXEL_UM_MAX],
    "typical_qi_cluster_um": [13.0, 16.0],
    "examples": {
        "andor_ixon_888": 13.0,
        "andor_ixon_16um": 16.0,
        "hamamatsu_c9100": 13.0,
        "nuvu_hnu": 16.0,
        "photometrics_evolve": 16.0,
    },
    "note": "Edit pixel_um in YAML for your EMCCD datasheet pitch.",
}

LEGACY_BEAM_SEMANTICS: dict[str, Any] = {
    "role": "legacy_geometry",
    "is_physical_waist": False,
    "sigma_norm": _LEGACY_BEAM_SIGMA_NORM,
    "falloff": _LEGACY_BEAM_FALLOFF,
    "phase4_ruling": "D7",
    "note": (
        "Do not call N/3.2 and 0.9 a physical waist without mapping to "
        "(w_p, M, p). Use pump_waist_um + use_legacy_beam_match=False."
    ),
}


def legacy_waist_px(N: int) -> float:
    """Pixel waist that matches the Phase-0 Gaussian envelope at size N."""
    return float((N / _LEGACY_BEAM_SIGMA_NORM) * np.sqrt(2.0 / _LEGACY_BEAM_FALLOFF))


def validate_pump_waist_um(w_um: float) -> float:
    """Ensure pump waist is within the allowed lab range [100, 1000] µm."""
    w = float(w_um)
    if not (PUMP_WAIST_UM_MIN <= w <= PUMP_WAIST_UM_MAX):
        raise ValueError(
            f"pump_waist_um={w} µm is outside allowed range "
            f"[{PUMP_WAIST_UM_MIN}, {PUMP_WAIST_UM_MAX}] µm"
        )
    return w


def validate_pointing_alpha(alpha: float) -> float:
    """Relative beam-pointing scale α in [0.01, 0.10] (PDF §2.5)."""
    a = float(alpha)
    if not (POINTING_ALPHA_MIN <= a <= POINTING_ALPHA_MAX):
        raise ValueError(
            f"pointing_alpha={a} is outside allowed range "
            f"[{POINTING_ALPHA_MIN}, {POINTING_ALPHA_MAX}] "
            f"(default {POINTING_ALPHA_DEFAULT})"
        )
    return a


def sigma_beam_px(w_px: float, *, pointing_alpha: float) -> float:
    """σ_beam = α·w_p,pixel  (PDF eq. σ_beam = α w_p)."""
    return validate_pointing_alpha(pointing_alpha) * float(w_px)


def pump_overlap_eta(delta_x_px: float, delta_y_px: float, w_px: float) -> float:
    """Pump–mode overlap from pointing jitter (PDF §2.6).

    η_overlap = exp(-(Δx² + Δy²) / (2 w_p²))

    Uses beam-pointing displacements only (C1). Thermal beam drift is handled
    separately (C4) and must not be folded into this factor.
    """
    w2 = max(float(w_px), 1e-6) ** 2
    r2 = float(delta_x_px) ** 2 + float(delta_y_px) ** 2
    return float(np.exp(-r2 / (2.0 * w2)))


def validate_idler_waist_um(w_um: float) -> float:
    """Validate the independently specified idler mode waist [100, 350] µm."""
    w = float(w_um)
    if not (IDLER_WAIST_UM_MIN <= w <= IDLER_WAIST_UM_MAX):
        raise ValueError(
            f"idler_waist_um={w} µm is outside allowed range "
            f"[{IDLER_WAIST_UM_MIN}, {IDLER_WAIST_UM_MAX}] µm"
        )
    return w


def eta_mode_gaussian(
    w1_um: float,
    w2_um: float,
    dx_um: float,
    dy_um: float,
    *,
    dtheta_mrad: float = 0.0,
    lambda_i_nm: float = 1550.0,
) -> float:
    """Idler–idler mode overlap η_mode for two Gaussian fields (PDF §4 / eq. 63).

    Field envelopes E ∝ exp(-r²/w²).  Closed form (V-P0-02):

      η = [2 w1 w2 / (w1²+w2²)]²
          · exp(-2 (Δx²+Δy²)/(w1²+w2²))
          · exp(- w1² w2² |Δk_⊥|² / (2(w1²+w2²)))

    with Δk_⊥ = 2π Δθ / λ_i.  Default w1=w2=w ⇒ η = exp(-(Δx²+Δy²)/w²).
    Distinct from pump_overlap_eta (C1 pointing vs crystal).
    """
    w1 = max(float(w1_um), 1e-6)
    w2 = max(float(w2_um), 1e-6)
    w1s = w1 * w1
    w2s = w2 * w2
    den = w1s + w2s
    waist_factor = (2.0 * w1 * w2 / den) ** 2
    r2 = float(dx_um) ** 2 + float(dy_um) ** 2
    spatial = float(np.exp(-2.0 * r2 / den))
    dtheta = float(dtheta_mrad) * 1e-3
    lam_um = max(float(lambda_i_nm), 1e-9) * 1e-3
    dk = 2.0 * np.pi * dtheta / lam_um
    tilt = float(np.exp(-(w1s * w2s) * (dk * dk) / (2.0 * den)))
    return float(np.clip(waist_factor * spatial * tilt, 0.0, 1.0))


def resolved_idler_waist_um(setup: "SetupParams") -> tuple[float, float]:
    """(w1, w2) in µm.  Default w2=w1=idler_waist_um, else inherit pump_waist_um."""
    if setup.idler_waist_um is not None:
        w1 = validate_idler_waist_um(setup.idler_waist_um)
    elif setup.pump_waist_um is not None:
        w1 = validate_pump_waist_um(float(setup.pump_waist_um))
    else:
        w1 = PUMP_WAIST_UM_DEFAULT
    if setup.idler_waist_2_um is not None:
        w2 = validate_idler_waist_um(setup.idler_waist_2_um)
    else:
        w2 = w1
    return w1, w2


def generation_alignment_um(setup: "SetupParams") -> tuple[float, float, float]:
    """Pump→crystal generation misalignment (dx, dy [µm], dθ [mrad]) — V-P1-04.

    Used only by ``O_geom`` / SPDC generation. Never mixed into relay η_mode.
    """
    return (
        float(getattr(setup, "generation_dx_um", 0.0) or 0.0),
        float(getattr(setup, "generation_dy_um", 0.0) or 0.0),
        float(getattr(setup, "generation_dtheta_mrad", 0.0) or 0.0),
    )


def relay_alignment_um(setup: "SetupParams") -> tuple[float, float, float]:
    """Idler→seed relay misalignment (dx, dy [µm], dθ [mrad]) — V-P1-04.

    Used only by η_mode / t_seed. Explicit ``relay_*`` wins; otherwise legacy
    ``idler_*`` is the relay alias (not applied to generation).
    """
    rx = getattr(setup, "relay_dx_um", None)
    ry = getattr(setup, "relay_dy_um", None)
    rt = getattr(setup, "relay_dtheta_mrad", None)
    dx = float(setup.idler_dx_um if rx is None else rx)
    dy = float(setup.idler_dy_um if ry is None else ry)
    dth = float(setup.idler_dtheta_mrad if rt is None else rt)
    return dx, dy, dth


def eta_mode_from_setup(
    setup: "SetupParams",
    *,
    pump_delta_x_px: float = 0.0,
    pump_delta_y_px: float = 0.0,
) -> float:
    """η_mode from relay (idler→seed) geometry only (V-P1-04).

    When ``inherit_idler_displacement_from_pump`` (default), pump pointing
    Δx,Δy [px] are converted to µm and added to **relay** offsets.
    Generation (``generation_*``) is not included — that path is O_geom only.
    Thermal beam drift is not included here (see ``physical_thermal_bundle``).
    """
    w1, w2 = resolved_idler_waist_um(setup)
    dx, dy = idler_displacement_um(
        setup,
        pump_delta_x_px=pump_delta_x_px,
        pump_delta_y_px=pump_delta_y_px,
    )
    _, _, dtheta = relay_alignment_um(setup)
    return eta_mode_gaussian(
        w1,
        w2,
        dx,
        dy,
        dtheta_mrad=dtheta,
        lambda_i_nm=float(setup.lambda_i_nm),
    )


def idler_displacement_um(
    setup: "SetupParams",
    *,
    pump_delta_x_px: float = 0.0,
    pump_delta_y_px: float = 0.0,
    t_s: float | None = None,
) -> tuple[float, float]:
    """Relay idler-2 vs idler-1 (Δx, Δy) [µm] for η_mode (V-P1-04).

    Uses relay / legacy ``idler_*`` only — not ``generation_*``.
    Includes PDF §5.4/§5.9 thermal beam shift when ``use_physical_thermal``.
    """
    extra_x, extra_y, _ = relay_alignment_um(setup)
    if bool(getattr(setup, "use_physical_thermal", False)):
        tx, ty = thermal_beam_shift_um(setup, t_s=t_s)
        extra_x += tx
        extra_y += ty
    if not bool(setup.inherit_idler_displacement_from_pump):
        return extra_x, extra_y
    pump_dx = px_to_um_setup(pump_delta_x_px, setup)
    pump_dy = px_to_um_setup(pump_delta_y_px, setup)
    return extra_x + pump_dx, extra_y + pump_dy


def validate_pixel_um(pixel_um: float) -> float:
    """EMCCD pixel pitch [µm] within the quantum-imaging suggested range."""
    p = float(pixel_um)
    if not (PIXEL_UM_MIN <= p <= PIXEL_UM_MAX):
        raise ValueError(
            f"pixel_um={p} µm is outside suggested EMCCD range "
            f"[{PIXEL_UM_MIN}, {PIXEL_UM_MAX}] µm "
            f"(typical QI cluster 13–16 µm, default {PIXEL_UM_DEFAULT})"
        )
    return p


def validate_internal_sampling_pixel_um(pitch_um: float) -> float:
    """Numerical grid pitch on the camera plane [µm] (verification-only)."""
    p = float(pitch_um)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"internal sampling pitch must be finite and > 0, got {pitch_um!r}")
    return p


def internal_camera_sampling_pixel_um(setup: "SetupParams") -> float:
    """Active camera-plane pitch for forward-model µm↔px conversions."""
    v = getattr(setup, "numerical_verification_internal_camera_pixel_um", None)
    if v is not None:
        return validate_internal_sampling_pixel_um(float(v))
    return validate_pixel_um(setup.pixel_um)


def physical_detector_pixel_um(setup: "SetupParams") -> float:
    """Hardware EMCCD pitch (unchanged under numerical refinement)."""
    return validate_pixel_um(setup.pixel_um)


def validate_crystal_length_mm(L_mm: float) -> float:
    """Nonlinear crystal length L in [2, 20] mm (user-editable; PDF §3.1)."""
    if isinstance(L_mm, bool) or not isinstance(L_mm, Real):
        raise ValueError(f"crystal_length_mm must be a real number, got {L_mm!r}")
    L = float(L_mm)
    if not (CRYSTAL_LENGTH_MM_MIN <= L <= CRYSTAL_LENGTH_MM_MAX):
        raise ValueError(
            f"crystal_length_mm={L} mm is outside allowed range "
            f"[{CRYSTAL_LENGTH_MM_MIN}, {CRYSTAL_LENGTH_MM_MAX}] mm "
            f"(single_blur default {CRYSTAL_LENGTH_MM_DEFAULT_SINGLE_BLUR} mm; "
            f"double_blur default {CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR} mm)"
        )
    return L


def validate_L_coh_mm(L_coh_mm: float) -> float:
    """Coherence length L_c in [0.02, 0.50] mm (PDF §4.1 / eq. 65–66)."""
    Lc = float(L_coh_mm)
    if not (L_COH_MM_MIN <= Lc <= L_COH_MM_MAX):
        raise ValueError(
            f"L_coh_mm={Lc} mm is outside allowed range "
            f"[{L_COH_MM_MIN}, {L_COH_MM_MAX}] mm "
            f"(default {L_COH_MM_DEFAULT})"
        )
    return Lc


def validate_delta_L_mm(delta_L_mm: float) -> float:
    """Signed optical path mismatch in [-0.50, 0.50] mm."""
    dL = float(delta_L_mm)
    if not (DELTA_L_MM_MIN <= dL <= DELTA_L_MM_MAX):
        raise ValueError(
            f"delta_L_mm={dL} mm is outside allowed range "
            f"[{DELTA_L_MM_MIN}, {DELTA_L_MM_MAX}] mm "
            f"(default {DELTA_L_MM_DEFAULT})"
        )
    return dL


def quantize_temperature_C(T_C: float) -> float:
    """Snap temperature to the lab grid (default step 0.1 °C)."""
    step = float(T_CRYSTAL_C_STEP)
    return float(round(float(T_C) / step) * step)


def validate_crystal_temperature_C(T_C: float, *, name: str = "T_C") -> float:
    """Crystal / phase-match temperature in [20, 90] °C, step 0.1 °C."""
    Tq = quantize_temperature_C(T_C)
    if not (T_CRYSTAL_C_MIN <= Tq <= T_CRYSTAL_C_MAX):
        raise ValueError(
            f"{name}={T_C} °C is outside allowed range "
            f"[{T_CRYSTAL_C_MIN}, {T_CRYSTAL_C_MAX}] °C "
            f"(step {T_CRYSTAL_C_STEP}, default {T_CRYSTAL_C_DEFAULT})"
        )
    return Tq


def validate_n_crystals(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError(f"n_crystals must be an integer 1 or 2, got {n!r}")
    n_i = int(n)
    if n_i not in (1, 2):
        raise ValueError(f"n_crystals must be 1 or 2, got {n}")
    return n_i


def validate_qpm_order(value: int, *, name: str = "qpm_order") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def validate_quantum_efficiency(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"QE must be a real number in [0, 1], got {value!r}")
    qe = float(value)
    if not np.isfinite(qe) or not 0.0 <= qe <= 1.0:
        raise ValueError(f"QE must be finite and in [0, 1], got {value!r}")
    return qe


def phase_match_temperature_C(T_pm0_C: float, delta_T_C: float) -> float:
    """PDF §5 eq. 80: T_pm = T_pm0 + ΔT (quantized into the crystal T window)."""
    return validate_crystal_temperature_C(
        float(T_pm0_C) + float(delta_T_C), name="T_pm"
    )


def _resolve_one_crystal_detuning(
    *,
    T_pm_C: float,
    T_crystal_C: float,
    dT_C: float | None,
    name: str,
) -> tuple[float, float]:
    """Resolve lab detuning. ``dT`` is the user variable; default 0.

    ``T_crystal = T_pm + dT``. If only the operating temperature is given,
    ``dT`` is derived. A nonzero ``dT`` with ``T_crystal`` still at ``T_pm``
    is treated as a lab detuning input.
    """
    T_pm = float(T_pm_C)
    T_c = float(T_crystal_C)
    dT = 0.0 if dT_C is None else float(quantize_temperature_C(dT_C))
    tol = 0.5 * float(T_CRYSTAL_C_STEP) + 1.0e-12
    if abs(dT) > tol:
        T_c = T_pm + dT
    elif abs(T_c - T_pm) <= tol:
        T_c = T_pm
        dT = 0.0
    elif (
        abs(T_c - T_CRYSTAL_C_DEFAULT) <= tol
        and abs(T_pm - T_CRYSTAL_C_DEFAULT) > tol
    ):
        T_c = T_pm
        dT = 0.0
    else:
        dT = float(quantize_temperature_C(T_c - T_pm))
    T_c = validate_crystal_temperature_C(T_c, name=name.replace("dT", "T_crystal"))
    return float(quantize_temperature_C(dT)), float(T_c)


def resolve_crystal_operating_detuning(setup: "SetupParams") -> None:
    """Apply user ``dT`` (default 0) and keep ``T_crystal = T_pm + dT``."""
    T_pm1 = phase_match_temperature_C(setup.T_pm0_NL1_C, setup.delta_T_NL1_C)
    dT1, Tc1 = _resolve_one_crystal_detuning(
        T_pm_C=T_pm1,
        T_crystal_C=setup.T_crystal_NL1_C,
        dT_C=getattr(setup, "dT_NL1_C", None),
        name="dT_NL1_C",
    )
    setup.dT_NL1_C = dT1
    setup.T_crystal_NL1_C = Tc1
    T_pm2 = phase_match_temperature_C(setup.T_pm0_NL2_C, setup.delta_T_NL2_C)
    dT2, Tc2 = _resolve_one_crystal_detuning(
        T_pm_C=T_pm2,
        T_crystal_C=setup.T_crystal_NL2_C,
        dT_C=getattr(setup, "dT_NL2_C", None),
        name="dT_NL2_C",
    )
    setup.dT_NL2_C = dT2
    setup.T_crystal_NL2_C = Tc2


def crystal_thermal_state(setup: "SetupParams") -> dict[str, Any]:
    """Resolved crystal temperatures and ΔT for 1- or 2-crystal hardware.

    ``n_crystals`` is hardware inventory, not the SPDC interaction count
    (report §7.11). Eq. 80 per listed oven: ``T_pm = T_pm0 + ΔT``.

    ``T_pm0`` is the equilibrium phase-matching temperature of that crystal
    (QPM design / oven set-point). ``T_crystal`` is the actual operating
    temperature. Detuning is ``dT = T_crystal - T_pm``. Single-crystal
    user-facing labels are ``NL``; two-crystal labels are ``NL1`` / ``NL2``.
    """
    n = validate_n_crystals(setup.n_crystals)
    two = is_two_crystal_architecture(setup)
    label1 = PHYSICAL_CRYSTAL_NL1 if two else PHYSICAL_CRYSTAL_NL
    T_pm1 = phase_match_temperature_C(setup.T_pm0_NL1_C, setup.delta_T_NL1_C)
    T_c1 = validate_crystal_temperature_C(
        setup.T_crystal_NL1_C, name=f"T_crystal_{label1}_C"
    )
    dT1 = float(quantize_temperature_C(getattr(setup, "dT_NL1_C", T_c1 - T_pm1)))
    T_pm0_1 = float(quantize_temperature_C(setup.T_pm0_NL1_C))
    out: dict[str, Any] = {
        "n_crystals": n,
        "eq80": "T_pm = T_pm0 + delta_T",
        "T_pm0_meaning": "equilibrium_phase_matching_temperature",
        "T_crystal_meaning": "actual_operating_temperature",
        "dT_meaning": "lab_operating_detuning_T_crystal_minus_T_pm",
        "dT_user_variable": True,
        "physical_crystal_id_1": label1,
        "T_pm0_NL1_C": T_pm0_1,
        "delta_T_NL1_C": float(quantize_temperature_C(setup.delta_T_NL1_C)),
        "T_pm_NL1_C": float(T_pm1),
        "T_crystal_NL1_C": float(T_c1),
        "dT_NL1_C": dT1,
        "T_pm0_NL2_C": None,
        "delta_T_NL2_C": None,
        "T_pm_NL2_C": None,
        "T_crystal_NL2_C": None,
        "dT_NL2_C": None,
        "dT_rel_C": 0.0,
        "dT_eff_C": dT1,
        "T_pm0_NL_C": None if two else T_pm0_1,
        "T_pm_NL_C": None if two else float(T_pm1),
        "T_crystal_NL_C": None if two else float(T_c1),
        "dT_NL_C": None if two else dT1,
    }
    if not two:
        return out
    T_pm2 = phase_match_temperature_C(setup.T_pm0_NL2_C, setup.delta_T_NL2_C)
    T_c2 = validate_crystal_temperature_C(
        setup.T_crystal_NL2_C, name="T_crystal_NL2_C"
    )
    dT2 = float(quantize_temperature_C(getattr(setup, "dT_NL2_C", T_c2 - T_pm2)))
    out.update(
        {
            "physical_crystal_id_2": PHYSICAL_CRYSTAL_NL2,
            "T_pm0_NL2_C": float(quantize_temperature_C(setup.T_pm0_NL2_C)),
            "delta_T_NL2_C": float(quantize_temperature_C(setup.delta_T_NL2_C)),
            "T_pm_NL2_C": float(T_pm2),
            "T_crystal_NL2_C": float(T_c2),
            "dT_NL2_C": dT2,
            "dT_rel_C": float(dT1 - dT2),
            "dT_eff_C": 0.5 * (dT1 + dT2),
        }
    )
    return out


def n_spdc_interactions(setup: "SetupParams") -> int:
    """Number of SPDC passes. Not ``n_crystals`` (report §7.11).

    Physical Mode: always two interactions (one crystal double-pass, or
    two crystals). Legacy: keep ``n_crystals`` as the old count. Architecture
    is ``spdc_architecture``, not this count.
    """
    if bool(getattr(setup, "use_physical_spdc_gain", False)):
        return SPDC_N_INTERACTIONS
    return validate_n_crystals(setup.n_crystals)


@dataclass(frozen=True)
class ArchitectureProfile:
    """Physical inventory and imaging choice, separate from blur and gain."""

    profile_id: str
    spdc_architecture: str
    imaging_mode: str
    n_physical_crystals: int
    n_spdc_interactions: int
    physical_crystal_ids: tuple[str, ...]
    interaction_crystal_ids: tuple[str, str]
    propagation_directions: tuple[str, str]
    spatial_transfer_path: str
    optical_model_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ARCHITECTURE_PROFILES: dict[tuple[str, str], ArchitectureProfile] = {
    (SPDC_ARCHITECTURE_TWO, IMAGING_MODE_FAR): ArchitectureProfile(
        ARCHITECTURE_PROFILE_A, SPDC_ARCHITECTURE_TWO, IMAGING_MODE_FAR,
        2, 2, (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
        (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
        ("forward", "forward"), "two_crystal_far_field",
        "reduced_order_complex_field_model_available",
    ),
    (SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_NEAR): ArchitectureProfile(
        ARCHITECTURE_PROFILE_D, SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_NEAR,
        1, 2, (PHYSICAL_CRYSTAL_NL,), (PHYSICAL_CRYSTAL_NL, PHYSICAL_CRYSTAL_NL),
        ("forward", "backward"), "double_pass_near_field",
        "reduced_order_complex_field_model_available",
    ),
    (SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_FAR): ArchitectureProfile(
        ARCHITECTURE_PROFILE_C, SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_FAR,
        1, 2, (PHYSICAL_CRYSTAL_NL,), (PHYSICAL_CRYSTAL_NL, PHYSICAL_CRYSTAL_NL),
        ("forward", "backward"), "double_pass_far_field",
        "reduced_order_complex_field_model_available",
    ),
    (SPDC_ARCHITECTURE_TWO, IMAGING_MODE_NEAR): ArchitectureProfile(
        ARCHITECTURE_PROFILE_B, SPDC_ARCHITECTURE_TWO, IMAGING_MODE_NEAR,
        2, 2, (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
        (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
        ("forward", "forward"), "two_crystal_near_field",
        "reduced_order_complex_field_model_available",
    ),
}

_LEGACY_REGRESSION_PROFILE = ArchitectureProfile(
    "legacy_regression", SPDC_ARCHITECTURE_TWO, IMAGING_MODE_FAR,
    2, 2, (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
    (PHYSICAL_CRYSTAL_NL1, PHYSICAL_CRYSTAL_NL2),
    ("unspecified", "unspecified"), "legacy_intensity_regression",
    "numerical_regression_only",
)

REGISTERED_SETUP_PROFILES: dict[str, str] = {
    "two_crystal_2014": "configs/setup_two_crystal_2014.yaml",
    "single_crystal_near_field": "configs/setup_single_crystal_near_field.yaml",
    "single_crystal_far_field": "configs/setup_single_crystal_far_field.yaml",
    "two_crystal_near_field": "configs/setup_two_crystal_near_field.yaml",
    # Task 16 laboratory overlay — does not replace stock Configuration C.
    "config_c_lab_v1": "configs/setup_config_c_lab_v1.yaml",
    # Opt-in Joint-SPDC Rabbit70 pathway (2026-10-27); legacy C_lab unchanged.
    "config_c_lab_joint_spdc_rabbit70_v1": (
        "configs/setup_config_c_lab_joint_spdc_rabbit70_v1.yaml"
    ),
}
# Explicit short-name compatibility mapping; no inference from crystal count.
SETUP_PROFILE_NAME_ALIASES: dict[str, str] = {
    "A": "two_crystal_2014",
    "B": "two_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "single_crystal_near_field",
    "C_lab": "config_c_lab_v1",
    "C_lab_rabbit70": "config_c_lab_joint_spdc_rabbit70_v1",
}

# Canonical generic manuscript configurations (A–D); excludes C_lab.
GENERIC_CANONICAL_SETUP_IDS: frozenset[str] = frozenset({
    "two_crystal_2014_reference_v2",
    "two_crystal_near_field_v1",
    "single_crystal_far_field_v2",
    "single_crystal_near_field_v2",
})

GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM: float = 3.0

# Final camera-plane collection envelope E(x,y) on detected intensity (post-combiner).
FINAL_CAMERA_COLLECTION_UNIFORM: str = "uniform"
FINAL_CAMERA_COLLECTION_MODE_LEGACY: str = "mode_legacy"
FINAL_CAMERA_COLLECTION_MATCHED_GAUSSIAN: str = "matched_gaussian"
_FINAL_CAMERA_COLLECTION_POLICIES: frozenset[str] = frozenset(
    {
        FINAL_CAMERA_COLLECTION_UNIFORM,
        FINAL_CAMERA_COLLECTION_MODE_LEGACY,
        FINAL_CAMERA_COLLECTION_MATCHED_GAUSSIAN,
    }
)
GENERIC_BENCHMARK_FINAL_CAMERA_COLLECTION: str = FINAL_CAMERA_COLLECTION_UNIFORM


def normalize_final_camera_collection_policy(value: Any) -> str:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return FINAL_CAMERA_COLLECTION_MODE_LEGACY
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unity": FINAL_CAMERA_COLLECTION_UNIFORM,
        "u": FINAL_CAMERA_COLLECTION_UNIFORM,
        "legacy": FINAL_CAMERA_COLLECTION_MODE_LEGACY,
        "mode_specific": FINAL_CAMERA_COLLECTION_MODE_LEGACY,
        "production": FINAL_CAMERA_COLLECTION_MODE_LEGACY,
        "gaussian_nf": FINAL_CAMERA_COLLECTION_MODE_LEGACY,
        "matched": FINAL_CAMERA_COLLECTION_MATCHED_GAUSSIAN,
        "g": FINAL_CAMERA_COLLECTION_MATCHED_GAUSSIAN,
    }
    key = aliases.get(key, key)
    if key not in _FINAL_CAMERA_COLLECTION_POLICIES:
        raise ValueError(
            "final_camera_collection_policy must be one of "
            f"{sorted(_FINAL_CAMERA_COLLECTION_POLICIES)}, got {value!r}"
        )
    return key


def resolved_final_camera_collection_policy(setup: "SetupParams") -> str:
    sid = str(getattr(setup, "setup_id", "") or "")
    raw = getattr(setup, "final_camera_collection_policy", None)
    if sid in GENERIC_CANONICAL_SETUP_IDS:
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            return GENERIC_BENCHMARK_FINAL_CAMERA_COLLECTION
        return normalize_final_camera_collection_policy(raw)
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return FINAL_CAMERA_COLLECTION_MODE_LEGACY
    return normalize_final_camera_collection_policy(raw)


def final_camera_collection_envelope(
    setup: "SetupParams",
    *,
    camera_mode: str,
    N: int,
    cx: float,
    cy: float,
    xx: np.ndarray,
    yy: np.ndarray,
    w_px: float,
) -> tuple[np.ndarray, bool, str]:
    """Build the additional final camera collection map E(x,y) on intensity."""
    policy = resolved_final_camera_collection_policy(setup)
    if policy == FINAL_CAMERA_COLLECTION_UNIFORM:
        env = np.ones((N, N), dtype=np.float32)
        return (
            env,
            False,
            "E=1 uniform final camera collection (generic benchmark default)",
        )
    if policy == FINAL_CAMERA_COLLECTION_MATCHED_GAUSSIAN:
        w2 = max(float(w_px), 1e-6) ** 2
        env = np.exp(
            -2.0 * (((xx - cx) ** 2 + (yy - cy) ** 2) / w2)
        ).astype(np.float32)
        return (
            env,
            True,
            "E=exp(-2*r^2/w_px^2) matched Gaussian on all architectures",
        )
    # mode_legacy: far-field unity; near-field pump-waist Gaussian on camera.
    if camera_mode == IMAGING_MODE_FAR:
        env = np.ones((N, N), dtype=np.float32)
        return (
            env,
            False,
            "no final pump-waist Gaussian; pump width in FF spatial kernel/S_cam",
        )
    w2 = max(float(w_px), 1e-6) ** 2
    env = np.exp(
        -2.0 * (((xx - cx) ** 2 + (yy - cy) ** 2) / w2)
    ).astype(np.float32)
    return (
        env,
        True,
        "S_cam=exp(-2*((x-cx)^2+(y-cy)^2)/w_px^2) legacy NF collection",
    )


def resolve_architecture_profile(setup: "SetupParams") -> ArchitectureProfile:
    """Resolve A/B/C/D without inferring physical hardware from blur or passes."""
    architecture = normalize_spdc_architecture(setup.spdc_architecture)
    imaging = normalize_imaging_mode(setup.imaging_mode)
    declared_interactions = getattr(setup, "n_spdc_interactions", None)
    if (declared_interactions is not None
            and (type(declared_interactions) is not int
                 or declared_interactions != SPDC_N_INTERACTIONS)):
        if architecture == SPDC_ARCHITECTURE_SINGLE:
            raise ValueError(
                "single_crystal_double_pass requires one physical nonlinear "
                "crystal and two SPDC interactions through that same crystal; "
                "n_spdc_interactions must be the integer 2"
            )
        raise ValueError(
            "two_crystal requires two physical nonlinear crystals and two "
            "SPDC interactions; n_spdc_interactions must be the integer 2"
        )
    if (getattr(setup, "setup_id", None) == "legacy_matched_v1"
            and getattr(setup, "physics_tier", None) == "legacy"
            and not bool(getattr(setup, "use_physical_spdc_gain", True))
            and architecture == SPDC_ARCHITECTURE_TWO
            and imaging == IMAGING_MODE_FAR
            and validate_n_crystals(setup.n_crystals) == 2):
        return _LEGACY_REGRESSION_PROFILE
    profile = _ARCHITECTURE_PROFILES.get((architecture, imaging))
    if profile is None:
        raise ValueError(
            f"unsupported architecture/imaging pair: {architecture}+{imaging}"
        )
    actual = validate_n_crystals(setup.n_crystals)
    if actual != profile.n_physical_crystals:
        if architecture == SPDC_ARCHITECTURE_SINGLE:
            raise ValueError(
                "single_crystal_double_pass requires one physical nonlinear "
                "crystal and two SPDC interactions through that same crystal; "
                f"got n_crystals={actual}"
            )
        raise ValueError(
            "two_crystal requires n_crystals=2 (two physical nonlinear "
            "crystals) and two SPDC interactions; "
            f"got n_crystals={actual}"
        )
    return profile


def normalize_pump_profile_type(value: Any | None) -> str:
    """Normalize pump transverse profile type (Step 2)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return PUMP_PROFILE_GAUSSIAN
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _PUMP_PROFILE_DEFERRED_TYPES:
        raise ValueError(
            "pump_profile_type='measured_2d' is "
            f"{PUMP_PROFILE_MEASURED_2D_STATUS}; require path, "
            "calibration_id, pixel_size_um, coordinate convention, "
            "normalization convention, and crystal-plane mapping before "
            "execution (got "
            f"{value!r})"
        )
    aliases = {
        "gaussian": PUMP_PROFILE_GAUSSIAN,
        "gauss": PUMP_PROFILE_GAUSSIAN,
        "uniform_effective": PUMP_PROFILE_UNIFORM_EFFECTIVE,
        "uniform": PUMP_PROFILE_UNIFORM_EFFECTIVE,
        "top_hat_effective": PUMP_PROFILE_UNIFORM_EFFECTIVE,
    }
    if key not in aliases:
        raise ValueError(
            "pump_profile_type must be 'gaussian' or 'uniform_effective' "
            f"(measured_2d is deferred), got {value!r}"
        )
    return aliases[key]


def validate_crystal_orientation_deg(
    value: Any | None, *, name: str = "crystal_orientation_deg"
) -> float | None:
    """Accept None or a finite tilt within the Type-0 collinear domain."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number or None, got {value!r}")
    angle = float(value)
    if not np.isfinite(angle):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if abs(angle) > CRYSTAL_ORIENTATION_MAX_ABS_DEG:
        raise ValueError(
            f"{name}={angle} deg is outside the supported Type-0 collinear "
            f"domain |theta|<={CRYSTAL_ORIENTATION_MAX_ABS_DEG} deg "
            f"(model {CRYSTAL_ORIENTATION_MODEL_ID})"
        )
    return angle


def resolved_crystal_orientation_deg(
    setup: "SetupParams", *, interaction: int = 1
) -> float | None:
    """Physical crystal orientation for one interaction (not propagation direction)."""
    if int(interaction) not in (1, 2):
        raise ValueError("interaction must be 1 or 2")
    if int(interaction) == 2 and is_two_crystal_architecture(setup):
        value = getattr(setup, "crystal_orientation_NL2_deg", None)
        if value is None:
            value = getattr(setup, "crystal_orientation_NL1_deg", None)
    else:
        value = getattr(setup, "crystal_orientation_NL1_deg", None)
    return validate_crystal_orientation_deg(
        value, name=f"crystal_orientation_interaction_{int(interaction)}_deg"
    )


def empty_measured_quantity_slot(*, unit: str | None = None) -> dict[str, Any]:
    """R09 measured/calibrated slot — defaults are null, never software guesses."""
    return {
        "value": None,
        "unit": unit,
        "uncertainty": None,
        "evidence_id": None,
        "calibration_id": None,
        "date": None,
        "version": None,
        "source_hash": None,
        "valid_range": None,
    }


def empty_apparatus_crystal_record() -> dict[str, Any]:
    """Installed physical crystal apparatus record (P domain)."""
    return {
        "length_mm": empty_measured_quantity_slot(unit="mm"),
        "poling_period_um": empty_measured_quantity_slot(unit="um"),
        "temperature_C": empty_measured_quantity_slot(unit="deg_C"),
        "orientation_deg": empty_measured_quantity_slot(unit="deg"),
        "evidence_id": None,
    }


def empty_experimental_validation_slot() -> dict[str, Any]:
    """Q domain — experimentally validated range; null when not supplied."""
    return {
        "validated_range": None,
        "evidence_id": None,
        "notes": None,
    }


def software_supported_range_for(parameter: str) -> dict[str, Any] | None:
    """Domain G — software-supported ranges (not apparatus claims)."""
    table = {
        "crystal_length_mm": {
            "min": float(CRYSTAL_LENGTH_MM_MIN),
            "max": float(CRYSTAL_LENGTH_MM_MAX),
            "unit": "mm",
            "policy_id": "physics_team_user_variable_v1",
        },
        "pump_waist_um": {
            "min": float(PUMP_WAIST_UM_MIN),
            "max": float(PUMP_WAIST_UM_MAX),
            "unit": "um",
        },
        "pixel_um": {
            "min": float(PIXEL_UM_MIN),
            "max": float(PIXEL_UM_MAX),
            "unit": "um",
        },
        "QE": {"min": 0.0, "max": 1.0, "unit": "fraction"},
        "crystal_orientation_deg": {
            "min": -float(CRYSTAL_ORIENTATION_MAX_ABS_DEG),
            "max": float(CRYSTAL_ORIENTATION_MAX_ABS_DEG),
            "unit": "deg",
            "model_id": CRYSTAL_ORIENTATION_MODEL_ID,
        },
        "pump_profile_type": {
            "allowed": sorted(_PUMP_PROFILE_TYPES),
        },
    }
    return table.get(parameter)


def gpq_parameter_record(
    parameter: str,
    *,
    simulation_value: Any = None,
    apparatus: dict[str, Any] | None = None,
    experimental_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a G/P/Q triple. Never copies software defaults into apparatus."""
    return {
        "parameter": parameter,
        "software_supported": software_supported_range_for(parameter),
        "apparatus": apparatus if apparatus is not None else {
            "installed_value": None,
            "uncertainty": None,
            "evidence_id": None,
        },
        "experimental_validation": (
            experimental_validation
            if experimental_validation is not None
            else empty_experimental_validation_slot()
        ),
        "simulation_value": simulation_value,
        "simulation_value_is_not_apparatus_claim": True,
    }


def apparatus_records_from_setup(setup: "SetupParams") -> dict[str, Any]:
    """Return apparatus crystal/calibration slots without inventing values."""
    raw_nl1 = getattr(setup, "apparatus_crystal_NL1", None)
    raw_nl2 = getattr(setup, "apparatus_crystal_NL2", None)
    nl1 = empty_apparatus_crystal_record()
    if isinstance(raw_nl1, dict):
        for key in nl1:
            if key in raw_nl1 and raw_nl1[key] is not None:
                nl1[key] = raw_nl1[key]
    nl2 = None
    if is_two_crystal_architecture(setup):
        nl2 = empty_apparatus_crystal_record()
        if isinstance(raw_nl2, dict):
            for key in nl2:
                if key in raw_nl2 and raw_nl2[key] is not None:
                    nl2[key] = raw_nl2[key]
    calib = getattr(setup, "calibration_records", None)
    if calib is None:
        calib = {
            "pump_power": empty_measured_quantity_slot(unit="mW"),
            "spdc_gain": empty_measured_quantity_slot(unit="1/sqrt_mW"),
            "slm_phase": empty_measured_quantity_slot(unit="rad"),
            "emccd_QE": empty_measured_quantity_slot(unit="fraction"),
            "read_noise": empty_measured_quantity_slot(unit="e_rms"),
            "full_well": empty_measured_quantity_slot(unit="e"),
            "pixel_size": empty_measured_quantity_slot(unit="um"),
            "filter_transmission": empty_measured_quantity_slot(unit="fraction"),
        }
    installed_length = None
    length_unc = None
    evidence_id = None
    if isinstance(raw_nl1, dict):
        length_slot = raw_nl1.get("length_mm")
        if isinstance(length_slot, dict):
            installed_length = length_slot.get("value")
            length_unc = length_slot.get("uncertainty")
        evidence_id = raw_nl1.get("evidence_id")
    return {
        "crystal_NL1": nl1,
        "crystal_NL2": nl2,
        "shared_single_crystal": (not is_two_crystal_architecture(setup)),
        "calibration_records": calib,
        "gpq_examples": {
            "crystal_length_mm": gpq_parameter_record(
                "crystal_length_mm",
                simulation_value=getattr(setup, "crystal_length_mm", None),
                apparatus={
                    "installed_value_mm": installed_length,
                    "uncertainty_mm": length_unc,
                    "evidence_id": evidence_id,
                },
            ),
        },
    }


def gaussian_pump_intensity(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    waist_um: float,
    center_x_um: float = 0.0,
    center_y_um: float = 0.0,
) -> np.ndarray:
    """Normalized Gaussian pump intensity I_p(x,y).

    Uses the 1/e^2 intensity waist convention::

        I_p = I0 exp[-2 ((x-x0)^2 + (y-y0)^2) / w_p^2]

    The returned array is scaled so that its discrete integral approximates 1
    (total power remains a separate quantity owned by pump_powers_from_setup).
    """
    w = float(waist_um)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError(f"pump waist must be finite and > 0, got {waist_um}")
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x_um and y_um must have the same shape")
    rho2 = (x - float(center_x_um)) ** 2 + (y - float(center_y_um)) ** 2
    raw = np.exp(-2.0 * rho2 / (w * w))
    if x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] >= 2:
        dx = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0
        dy = float(y[1, 0] - y[0, 0]) if x.shape[0] > 1 else 1.0
        cell = abs(dx * dy)
        integral = float(np.sum(raw) * cell)
    else:
        integral = float(np.sum(raw))
    if integral <= 0.0 or not np.isfinite(integral):
        raise ValueError("pump intensity integral must be positive and finite")
    return (raw / integral).astype(np.float64)


def pump_transverse_profile_from_setup(
    setup: "SetupParams",
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    interaction: int = 1,
) -> dict[str, Any]:
    """Build the pump transverse intensity for one interaction.

    Ownership: this function owns the *spatial distribution*. Total pump power
    is owned by ``pump_powers_from_setup`` and must not be reapplied here.
    """
    profile_type = normalize_pump_profile_type(
        getattr(setup, "pump_profile_type", None)
    )
    wp = resolved_pump_waist_um(setup, interaction=interaction)
    if wp is None:
        wp = float(PUMP_WAIST_UM_DEFAULT)
    else:
        wp = float(validate_pump_waist_um(wp))
    x0 = float(getattr(setup, "pump_center_x_um", 0.0) or 0.0)
    y0 = float(getattr(setup, "pump_center_y_um", 0.0) or 0.0)
    if not np.isfinite(x0) or not np.isfinite(y0):
        raise ValueError("pump_center_x/y_um must be finite")
    if profile_type == PUMP_PROFILE_UNIFORM_EFFECTIVE:
        ones = np.ones(np.asarray(x_um).shape, dtype=np.float64)
        if ones.ndim == 2 and ones.shape[0] >= 2 and ones.shape[1] >= 2:
            x = np.asarray(x_um, dtype=np.float64)
            y = np.asarray(y_um, dtype=np.float64)
            dx = float(x[0, 1] - x[0, 0])
            dy = float(y[1, 0] - y[0, 0])
            integral = float(np.sum(ones) * abs(dx * dy))
        else:
            integral = float(np.sum(ones))
        intensity = ones / integral
        equation = "I_p = constant (effective uniform; waist used only in k_SPDC/sigma)"
    else:
        intensity = gaussian_pump_intensity(
            x_um, y_um, waist_um=wp, center_x_um=x0, center_y_um=y0
        )
        equation = (
            "I_p = I0*exp(-2*((x-x0)^2+(y-y0)^2)/w_p^2); discrete unit integral"
        )
    return {
        "profile_type": profile_type,
        "waist_um": wp,
        "center_x_um": x0,
        "center_y_um": y0,
        "intensity": intensity,
        "equation": equation,
        "total_power_owner": "pump_powers_from_setup",
        "spatial_distribution_owner": "pump_transverse_profile_from_setup",
        "calibrated_measured_profile_supported": False,
        "calibrated_measured_profile_reason": (
            "measured 2-D pump maps require apparatus calibration evidence; "
            "fail closed until a calibration_records.pump_profile entry is supplied"
        ),
        "interaction": int(interaction),
    }


def pump_transverse_profile_evidence(
    setup: "SetupParams", *, interaction: int = 1
) -> dict[str, Any]:
    """Provenance block for the pump transverse profile (no array payload)."""
    profile_type = normalize_pump_profile_type(
        getattr(setup, "pump_profile_type", None)
    )
    wp = resolved_pump_waist_um(setup, interaction=interaction)
    return {
        "profile_type": profile_type,
        "waist_um": None if wp is None else float(wp),
        "center_x_um": float(getattr(setup, "pump_center_x_um", 0.0) or 0.0),
        "center_y_um": float(getattr(setup, "pump_center_y_um", 0.0) or 0.0),
        "equation": (
            "I_p=I0*exp(-2*r^2/w_p^2)" if profile_type == PUMP_PROFILE_GAUSSIAN
            else "I_p=constant_effective"
        ),
        "enters_spdc_via": [
            "generation_amplitude_weight E_p~sqrt(I_p) on object/crystal plane "
            "before conditional correlation kernel",
            "k_SPDC,0 ~ 1/w_p",
            "O_geom Gaussian geometry",
            "far-field correlation sigma = m_i f_i lambda_i/(2 pi w_p)",
        ],
        "not_reapplied_as": "post-hoc image brightness multiplier",
        "total_power_separate": True,
        "measured_2d_status": PUMP_PROFILE_MEASURED_2D_STATUS,
    }


def object_plane_coordinate_grids_um(
    shape: tuple[int, ...],
    setup: "SetupParams",
) -> tuple[np.ndarray, np.ndarray]:
    """Centered object/crystal-plane coordinate grids [µm] for pump sampling."""
    if len(shape) < 2:
        raise ValueError(f"object field shape must be 2-D, got {shape}")
    h, w = int(shape[0]), int(shape[1])
    pitch = float(getattr(setup, "object_map_pixel_um", None) or 0.0)
    if not np.isfinite(pitch) or pitch <= 0.0:
        # Fall back to detector pitch / M_QIUP when object pitch is unset.
        pitch = float(object_pixel_size_um(setup))
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("object-plane pixel pitch must be finite and > 0")
    yy = (np.arange(h, dtype=np.float64) - 0.5 * (h - 1)) * pitch
    xx = (np.arange(w, dtype=np.float64) - 0.5 * (w - 1)) * pitch
    return np.meshgrid(xx, yy)


def _normalize_intensity_to_peak_one(intensity: np.ndarray) -> np.ndarray:
    """Convert a nonnegative intensity map to a peak-one shape weight."""
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("pump intensity must be a 2-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pump intensity must be finite")
    if np.any(arr < 0.0):
        raise ValueError("pump intensity must be nonnegative")
    peak = float(np.max(arr))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("pump intensity peak must be positive and finite")
    return arr / peak


def pump_generation_amplitude_weight_from_setup(
    setup: "SetupParams",
    shape: tuple[int, ...],
    *,
    interaction: int | None = None,
) -> dict[str, Any]:
    """Build the generation amplitude weight E∝√I_p for the object/crystal grid.

    Ownership:
      * spatial *shape* — this function (via pump_transverse_profile_from_setup)
      * scalar pump power — pump_powers_from_setup / gain path only

    The returned amplitude is peak-normalized (max |E_weight| = 1) so a valid
    object field with |t|≤1 remains physically bounded after weighting. Scalar
    pump power is never rescaled here. Waist still enters k_SPDC and σ_SPDC
    separately.
    """
    x_um, y_um = object_plane_coordinate_grids_um(shape, setup)
    profile = resolve_architecture_profile(setup)
    if interaction is None:
        if is_two_crystal_architecture(setup):
            # Branched independent pumps: geometric-mean intensity at the
            # shared object plane between NL1 and NL2 (reduced-order).
            p1 = pump_transverse_profile_from_setup(
                setup, x_um, y_um, interaction=1
            )
            p2 = pump_transverse_profile_from_setup(
                setup, x_um, y_um, interaction=2
            )
            intensity = np.sqrt(
                np.maximum(p1["intensity"], 0.0) * np.maximum(p2["intensity"], 0.0)
            )
            # Re-normalize the geometric mean to unit discrete integral.
            dx = float(x_um[0, 1] - x_um[0, 0]) if x_um.shape[1] > 1 else 1.0
            dy = float(y_um[1, 0] - y_um[0, 0]) if y_um.shape[0] > 1 else 1.0
            cell = abs(dx * dy)
            integral = float(np.sum(intensity) * cell)
            if integral <= 0.0 or not np.isfinite(integral):
                raise ValueError("branched pump geometric-mean integral invalid")
            intensity = intensity / integral
            profile_type = p1["profile_type"]
            waist_um = float(p1["waist_um"])
            center_x = float(p1["center_x_um"])
            center_y = float(p1["center_y_um"])
            interaction_label = "geometric_mean_NL1_NL2"
            equation = (
                "I_eff=normalize(sqrt(I_NL1*I_NL2)); E~sqrt(I_eff/max); "
                "branched_independent reduced-order"
            )
        else:
            # B/C: one physical crystal; use interaction-1 transverse geometry.
            # Return-pass power remains owned by pump_powers_from_setup.
            p1 = pump_transverse_profile_from_setup(
                setup, x_um, y_um, interaction=1
            )
            intensity = np.asarray(p1["intensity"], dtype=np.float64)
            profile_type = p1["profile_type"]
            waist_um = float(p1["waist_um"])
            center_x = float(p1["center_x_um"])
            center_y = float(p1["center_y_um"])
            interaction_label = "interaction_1_shared_crystal"
            equation = p1["equation"] + "; E~sqrt(I_p/max)"
    else:
        p1 = pump_transverse_profile_from_setup(
            setup, x_um, y_um, interaction=int(interaction)
        )
        intensity = np.asarray(p1["intensity"], dtype=np.float64)
        profile_type = p1["profile_type"]
        waist_um = float(p1["waist_um"])
        center_x = float(p1["center_x_um"])
        center_y = float(p1["center_y_um"])
        interaction_label = f"interaction_{int(interaction)}"
        equation = p1["equation"] + "; E~sqrt(I_p/max)"

    shape_weight = _normalize_intensity_to_peak_one(intensity)
    amplitude = np.sqrt(np.maximum(shape_weight, 0.0))
    return {
        "amplitude": amplitude.astype(np.float64),
        "intensity_shape_peak_one": shape_weight.astype(np.float64),
        "profile_type": profile_type,
        "waist_um": waist_um,
        "center_x_um": center_x,
        "center_y_um": center_y,
        "interaction": interaction_label,
        "equation": equation,
        "max_amplitude": float(np.max(amplitude)),
        "total_power_owner": "pump_powers_from_setup",
        "spatial_distribution_owner": "pump_transverse_profile_from_setup",
        "peak_normalized": True,
        "architecture_profile": profile.profile_id,
        "model_status": "reduced_order_generation_envelope",
    }


def apply_pump_generation_weight_to_field(
    field: np.ndarray,
    setup: "SetupParams",
    *,
    interaction: int | None = None,
) -> dict[str, Any]:
    """Helper: multiply a field by the peak-normalized pump amplitude.

    Production object transfer must not use this as t_object / t_seed
    (L02). Pair-generation intensity is applied to I± in optics, not here.
    Peak-normalized weighting keeps |t|≤1 if a caller still needs the helper.
    Scalar pump power is owned elsewhere and is not rescaled.
    """
    value = np.asarray(field, dtype=np.complex128)
    if value.ndim != 2:
        raise ValueError("generation field must be two-dimensional")
    if not np.all(np.isfinite(value)):
        raise ValueError("generation field must be finite")
    weight = pump_generation_amplitude_weight_from_setup(
        setup, value.shape, interaction=interaction
    )
    amp = np.asarray(weight["amplitude"], dtype=np.float64)
    weighted = value * amp
    power_in = float(np.sum(np.abs(value) ** 2))
    power_out = float(np.sum(np.abs(weighted) ** 2))
    meta = {
        "applied": True,
        "profile_type": weight["profile_type"],
        "waist_um": weight["waist_um"],
        "center_x_um": weight["center_x_um"],
        "center_y_um": weight["center_y_um"],
        "interaction": weight["interaction"],
        "equation": weight["equation"],
        "power_in": power_in,
        "power_out": power_out,
        "max_amplitude": weight["max_amplitude"],
        "peak_normalized": True,
        "total_power_owner": weight["total_power_owner"],
        "spatial_distribution_owner": weight["spatial_distribution_owner"],
        "not_reapplied_as": "post-interference image mask",
        "architecture_profile": weight["architecture_profile"],
        "model_status": weight["model_status"],
        "measured_2d_status": PUMP_PROFILE_MEASURED_2D_STATUS,
    }
    return {
        "field": weighted,
        "amplitude_weight": amp,
        "metadata": meta,
    }


def camera_plane_geometry_from_setup(
    setup: "SetupParams",
    *,
    n_pixels: int | None = None,
    phys_T: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit camera-plane geometry, distinct from M_QIUP.

    Concepts kept separate:
      object coordinates, idler coordinates, signal coordinates,
      camera-plane physical coordinates, detector pixel coordinates.
    Pixel pitch samples the mapped plane; it does not redefine M_QIUP.
    """
    mapping = None
    if (
        getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        mapping = resolved_spatial_mapping(setup, phys_T=phys_T)
    m_qiup = resolve_imaging_magnification(setup, phys_T=phys_T)
    p_cam = float(validate_pixel_um(setup.pixel_um))
    n = int(n_pixels) if n_pixels is not None else None
    sensor_um = None if n is None else float(n) * p_cam
    mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    rot = float(getattr(setup, "image_rotation_deg", 0.0) or 0.0)
    if not np.isfinite(rot):
        raise ValueError("image_rotation_deg must be finite")
    cx = float(getattr(setup, "camera_center_x_px", 0.0) or 0.0)
    cy = float(getattr(setup, "camera_center_y_px", 0.0) or 0.0)
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("camera_center_x/y_px must be finite")
    return {
        "imaging_mode": mode,
        "correlation_type": imaging_correlation_type(setup),
        "object_coordinates": "declared_object_plane",
        "idler_coordinates": (
            "near_field_position" if mode == IMAGING_MODE_NEAR else "far_field_momentum"
        ),
        "signal_coordinates": (
            "near_field_position" if mode == IMAGING_MODE_NEAR else "far_field_momentum"
        ),
        "camera_plane_physical_coordinates_um": "detector_plane_physical",
        "detector_pixel_coordinates": "integer_or_fractional_pixel_index",
        "pixel_pitch_um": p_cam,
        "sensor_n_pixels": n,
        "sensor_width_um": sensor_um,
        "sensor_height_um": sensor_um,
        "camera_center_x_px": cx,
        "camera_center_y_px": cy,
        "image_rotation_deg": rot,
        "M_QIUP": float(m_qiup),
        "M_QIUP_is_optical_mapping_not_pixel_pitch": True,
        "pixel_pitch_redefines_M_QIUP": False,
        "spatial_mapping": mapping,
        "object_map_pixel_um": getattr(setup, "object_map_pixel_um", None),
        "relay_magnifications": {
            "near_field_idler": getattr(setup, "near_field_idler_magnification", None),
            "near_field_signal": getattr(setup, "near_field_signal_magnification", None),
            "far_field_idler": getattr(setup, "far_field_idler_relay_magnification", None),
            "far_field_signal": getattr(setup, "far_field_signal_relay_magnification", None),
        },
        "focal_lengths_mm": {
            "F1_mm": getattr(setup, "F1_mm", None),
            "F2_mm": getattr(setup, "F2_mm", None),
            "far_field_focal_mm": getattr(setup, "far_field_focal_mm", None),
        },
    }


def physical_camera_um_to_detector_px(
    x_um: float,
    y_um: float,
    setup: "SetupParams",
) -> tuple[float, float]:
    """Map camera-plane physical micrometres to detector pixel coordinates."""
    p = float(validate_pixel_um(setup.pixel_um))
    cx = float(getattr(setup, "camera_center_x_px", 0.0) or 0.0)
    cy = float(getattr(setup, "camera_center_y_px", 0.0) or 0.0)
    rot = np.deg2rad(float(getattr(setup, "image_rotation_deg", 0.0) or 0.0))
    xr = float(x_um) * np.cos(rot) - float(y_um) * np.sin(rot)
    yr = float(x_um) * np.sin(rot) + float(y_um) * np.cos(rot)
    return cx + xr / p, cy + yr / p


def detector_px_to_physical_camera_um(
    x_px: float,
    y_px: float,
    setup: "SetupParams",
) -> tuple[float, float]:
    """Inverse of :func:`physical_camera_um_to_detector_px`."""
    p = float(validate_pixel_um(setup.pixel_um))
    cx = float(getattr(setup, "camera_center_x_px", 0.0) or 0.0)
    cy = float(getattr(setup, "camera_center_y_px", 0.0) or 0.0)
    rot = np.deg2rad(float(getattr(setup, "image_rotation_deg", 0.0) or 0.0))
    xr = (float(x_px) - cx) * p
    yr = (float(y_px) - cy) * p
    x = xr * np.cos(-rot) - yr * np.sin(-rot)
    y = xr * np.sin(-rot) + yr * np.cos(-rot)
    return float(x), float(y)


def spdc_correlation_psf_status() -> dict[str, Any]:
    """Explicit fidelity label for the spatial SPDC models in the runtime."""
    return {
        "implementation_status": SPDC_CORRELATION_PSF_IMPLEMENTATION_STATUS,
        "model_id": SPDC_CORRELATION_PSF_MODEL_ID,
        "biphoton_wavefunction": "partial_far_field_signal_marginal",
        "assumptions": [
            "near-field joint Gaussian P(ρd,ρu) from P10 Eq. 9 (sinc→Gaussian)",
            "NF image width Δ_G (P10 Eq. 12) and visibility width Δ_V (P10 Eq. 13)",
            "far-field conditional Gaussian in q (P14 σ_q=1/w_p)",
            (
                "config_c_lab_v1 far-field single-photon marginal uses the "
                "pump angular spectrum, paraxial QPM mismatch, sinc-squared "
                "phase matching, and explicit idler-momentum marginalization"
            ),
            "ordinary optical PSF remains a separate convolution",
            (
                "object-conditioned propagation remains a reduced-order "
                "complex-field model rather than a full four-dimensional "
                "biphoton propagation"
            ),
        ],
        "higher_fidelity_physical_model": (
            "signal_marginal_implemented_object_conditioning_reduced_order"
        ),
    }


def validate_forward_profile_supported(setup: "SetupParams") -> ArchitectureProfile:
    """Prevent an unsupported physical profile from generating output arrays."""
    profile = resolve_architecture_profile(setup)
    # Step-2: orientation supported only for Type-0 collinear grating projection.
    for name in ("crystal_orientation_NL1_deg", "crystal_orientation_NL2_deg"):
        validate_crystal_orientation_deg(getattr(setup, name, None), name=name)
    if (
        getattr(setup, "crystal_orientation_NL1_deg", None) is not None
        or getattr(setup, "crystal_orientation_NL2_deg", None) is not None
    ):
        qpm = str(getattr(setup, "qpm_type", "type0") or "type0").lower()
        if qpm not in {"type0", "type_0", "0"}:
            raise NotImplementedError(
                "crystal orientation is only implemented for Type-0 ppKTP "
                f"collinear QPM (model {CRYSTAL_ORIENTATION_MODEL_ID})"
            )
    if getattr(setup, "pump_profile_type", None) is not None:
        normalize_pump_profile_type(setup.pump_profile_type)
    propagation_model = getattr(setup, "architecture_propagation_model", None)
    if (bool(setup.use_physical_spdc_gain)
            and profile.profile_id in {
                ARCHITECTURE_PROFILE_A,
                ARCHITECTURE_PROFILE_B,
                ARCHITECTURE_PROFILE_C,
            }
            and propagation_model != ARCHITECTURE_PROPAGATION_COMPLEX_V1):
        raise NotImplementedError(
            f"Profile {profile.profile_id} requires an explicitly selected "
            f"{profile.spatial_transfer_path} propagation model"
        )
    return profile


def physical_model_support_status(
    setup: "SetupParams",
    operating_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe implementation, software test, and experiment status separately.

    This status is deliberately conservative. The A/B/C/D pipelines are
    implemented and covered by software tests, while no supplied evidence
    experimentally validates a complete configuration.
    """
    profile = validate_forward_profile_supported(setup)
    source_mode = normalize_spdc_source_mode(
        getattr(setup, "spdc_source_mode", None)
    )
    regimes = []
    if isinstance(operating_point, dict):
        regimes = [
            str(operating_point.get("regime_NL1") or "unknown"),
            str(operating_point.get("regime_NL2") or "unknown"),
        ]
    calibration_available = source_mode == SPDC_SOURCE_FIXED
    if source_mode == SPDC_SOURCE_CALIBRATED:
        calibration_available = bool(
            isinstance(operating_point, dict)
            and operating_point.get("predictive_NL1")
            and operating_point.get("predictive_NL2")
        )

    reasons = [
        "conditional-Gaussian spatial correlation is a reduced-order model",
        "no supplied end-to-end experimental validation for this configuration",
    ]
    if profile.profile_id in {ARCHITECTURE_PROFILE_C, ARCHITECTURE_PROFILE_D} \
            and "high_gain" in regimes:
        reasons.append(
            "single-mode Michelson source moments are implemented; "
            "high-gain spatial prediction remains provisional"
        )
    if source_mode == SPDC_SOURCE_CALIBRATED and not calibration_available:
        reasons.append("documented calibrated-source parameters are unavailable")

    implemented = profile.profile_id in {
        ARCHITECTURE_PROFILE_A,
        ARCHITECTURE_PROFILE_B,
        ARCHITECTURE_PROFILE_C,
        ARCHITECTURE_PROFILE_D,
    }
    return {
        "status_schema": "qiup_physical_support_v1",
        "implemented": bool(implemented),
        "software_tested": bool(implemented),
        "experimentally_validated": False,
        "support_level": "research_only_approximation",
        "quantitative_operation_permitted": bool(
            implemented and calibration_available
        ),
        "architecture_profile": profile.profile_id,
        "architecture_model_status": profile.optical_model_status,
        "source_mode": source_mode,
        "source_calibration_available": bool(calibration_available),
        "gain_regime_NL1": regimes[0] if regimes else None,
        "gain_regime_NL2": regimes[1] if regimes else None,
        "source_moment_model": (
            "independent_source_idler_seeding"
            if is_two_crystal_architecture(setup)
            else "single_crystal_two_squeezer_bogoliubov"
        ),
        "spatial_correlation_model": "conditional_gaussian_v1",
        "biphoton_wavefunction": False,
        "detector_model": "shared_analog_emccd",
        "reasons": reasons,
    }


def normalize_architecture_propagation_model(value: Any | None) -> str | None:
    """Select the opt-in A/B/C/D complex-field path without changing old runs."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "architecture_propagation_model must be a non-empty string or null"
        )
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key != ARCHITECTURE_PROPAGATION_COMPLEX_V1:
        raise ValueError(
            "architecture_propagation_model must be "
            f"{ARCHITECTURE_PROPAGATION_COMPLEX_V1!r}, got {value!r}"
        )
    return key


def _normalize_required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return result


def validate_strict_bool(
    value: Any, *, name: str, allow_none: bool = False
) -> bool | None:
    """R03 — accept only real Python ``bool`` values (fail-closed).

    Never use ``bool(user_value)`` for configuration parsing: in Python
    ``bool("false")`` is True because any non-empty string is truthy.
    Strings such as ``"false"``, ``"true"``, ``"0"``, ``"1"``, ``"yes"``,
    and ``"no"`` are rejected. ``numpy.bool_`` is also rejected so YAML/JSON
    numeric coercions cannot slip through silently.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a Boolean True or False, got None")
    if type(value) is not bool:
        raise ValueError(
            f"{name} must be a Boolean True or False, got {value!r} "
            f"(type {type(value).__name__}); string/numeric coercions are rejected"
        )
    return value


# SetupParams Boolean fields audited under R03 strict Boolean policy.
SETUP_BOOLEAN_FIELDS: tuple[str, ...] = (
    "use_legacy_beam_match",
    "couple_pointing_overlap_visibility",
    "use_independent_pump_power_NL2",
    "use_physical_spdc_gain",
    "use_calibrated_poling_period",
    "derive_sigma_spdc",
    "use_derived_diffraction",
    "use_derived_K_defocus",
    "enable_lens_rings",
    "enable_etalon_fringes",
    "enable_phase_step_acquisition",
    "use_derived_aberration",
    "couple_align_to_wavefront",
    "use_analytic_mode_overlap",
    "inherit_idler_displacement_from_pump",
    "use_physical_two_beam",
    "use_field_overlap_visibility_map",
    "sync_crystal_temperatures",
    "use_physical_thermal",
    "use_physical_qiup_magnification",
    "use_physical_dark",
    "enable_zernike",
)

# Optional Boolean fields (None means "not applicable / unset").
SETUP_OPTIONAL_BOOLEAN_FIELDS: tuple[str, ...] = (
    "signal_return_seeds_second_interaction",
)


def validate_etalon_parameters(setup: "SetupParams") -> None:
    """R03 — finiteness and documented ranges for etalon intensity knobs."""
    setup.enable_etalon_fringes = validate_strict_bool(
        getattr(setup, "enable_etalon_fringes", False),
        name="enable_etalon_fringes",
    )
    c = _validate_finite_real(
        getattr(setup, "etalon_contrast", 0.0), name="etalon_contrast"
    )
    if c < 0.0 or c >= 1.0:
        raise ValueError(f"etalon_contrast must satisfy 0 <= c < 1, got {c}")
    setup.etalon_contrast = c
    period = _validate_finite_real(
        getattr(setup, "etalon_period_px", 34.0), name="etalon_period_px"
    )
    if period <= 0.0:
        raise ValueError(f"etalon_period_px must be > 0, got {period}")
    setup.etalon_period_px = period
    setup.etalon_angle_deg = _validate_finite_real(
        getattr(setup, "etalon_angle_deg", -40.0), name="etalon_angle_deg"
    )
    setup.etalon_phase_rad = _validate_finite_real(
        getattr(setup, "etalon_phase_rad", 0.0), name="etalon_phase_rad"
    )


FOUR_STEP_PSI_SCHEDULE_RAD: tuple[float, ...] = (
    0.0,
    0.5 * float(np.pi),
    float(np.pi),
    1.5 * float(np.pi),
)


def validate_optical_background_photons(value: Any) -> float:
    """Non-negative detector-plane optical background [photons / px / exposure]."""
    bg = _validate_finite_real(
        0.0 if value is None else value,
        name="optical_background_photons",
    )
    if bg < 0.0:
        raise ValueError(f"optical_background_photons must be >= 0, got {bg}")
    return bg


def validate_phase_step_schedule(
    value: Any,
) -> tuple[float, ...] | None:
    """Independent interferometer steps; ``None`` means no explicit schedule."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or isinstance(value, bool):
        raise ValueError(
            "phase_step_schedule_rad must be a sequence of finite radians"
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            "phase_step_schedule_rad must be a sequence of finite radians"
        ) from exc
    if len(items) < 3:
        raise ValueError(
            "phase_step_schedule_rad needs at least 3 steps for phase retrieval"
        )
    return tuple(
        _validate_finite_real(item, name="phase_step_schedule_rad")
        for item in items
    )


def validate_l07_acquisition_parameters(setup: "SetupParams") -> None:
    """L07 knobs: optical background and explicit phase-step protocol."""
    setup.enable_phase_step_acquisition = validate_strict_bool(
        getattr(setup, "enable_phase_step_acquisition", False),
        name="enable_phase_step_acquisition",
    )
    setup.optical_background_photons = validate_optical_background_photons(
        getattr(setup, "optical_background_photons", 0.0)
    )
    setup.phase_step_schedule_rad = validate_phase_step_schedule(
        getattr(setup, "phase_step_schedule_rad", None)
    )


def resolved_phase_step_schedule(
    setup: "SetupParams",
    *,
    num_frames: int,
) -> tuple[float, ...] | None:
    """Return the per-frame interferometer schedule, or None if protocol is off.

    Raising ``num_frames`` alone is not a phase-step protocol (L07 / P13).
    """
    enabled = bool(getattr(setup, "enable_phase_step_acquisition", False))
    if not enabled:
        return None
    n = int(num_frames)
    if isinstance(num_frames, bool) or n <= 0:
        raise ValueError("phase-step acquisition requires a positive num_frames")
    schedule = getattr(setup, "phase_step_schedule_rad", None)
    if schedule is None:
        schedule = FOUR_STEP_PSI_SCHEDULE_RAD
    schedule = validate_phase_step_schedule(schedule)
    if schedule is None:
        raise ValueError("phase-step schedule resolved to None")
    if n != len(schedule):
        raise ValueError(
            "enable_phase_step_acquisition requires num_frames to equal "
            f"the schedule length ({len(schedule)}); got {n}. "
            "Increasing num_frames alone is not a phase-step protocol."
        )
    return schedule


def validate_setup_boolean_fields(setup: "SetupParams") -> None:
    """Apply R03 strict Boolean validation to every SetupParams Boolean field."""
    for name in SETUP_BOOLEAN_FIELDS:
        setattr(
            setup,
            name,
            validate_strict_bool(getattr(setup, name), name=name),
        )
    for name in SETUP_OPTIONAL_BOOLEAN_FIELDS:
        setattr(
            setup,
            name,
            validate_strict_bool(
                getattr(setup, name), name=name, allow_none=True
            ),
        )


def validate_architecture_propagation_fields(setup: "SetupParams") -> None:
    """Validate the opt-in object and Michelson return-path state.

    The object-map fields are provenance selectors.  This version supports the
    calibrated arrays already carried by each reference, ``ref.T`` and
    ``ref.phi``.  Unsupported planes, maps, or coordinate transforms fail
    before simulation.
    """
    model = normalize_architecture_propagation_model(
        getattr(setup, "architecture_propagation_model", None)
    )
    setup.architecture_propagation_model = model
    if model is None:
        return

    profile = resolve_architecture_profile(setup)
    object_type = _normalize_required_string(
        getattr(setup, "object_type", None), name="object_type"
    ).lower()
    if object_type not in {OBJECT_TYPE_TRANSMISSIVE, OBJECT_TYPE_REFLECTIVE}:
        raise ValueError("object_type must be 'transmissive' or 'reflective'")
    if is_two_crystal_architecture(setup) and object_type != OBJECT_TYPE_TRANSMISSIVE:
        raise ValueError(
            f"profile {profile.profile_id} supports a transmissive object between crystals"
        )
    setup.object_type = object_type

    object_mode = normalize_named_choice(
        getattr(setup, "object_mode", None),
        name="object_mode",
        allowed=frozenset({
            OBJECT_MODE_REFERENCE_SLM,
            OBJECT_MODE_GENERIC_RESEARCH,
            OBJECT_MODE_SLM,
            OBJECT_MODE_MASK,
        }),
        default=(OBJECT_MODE_REFERENCE_SLM if is_three_configuration_v2(setup) else None),
    )
    setup.object_mode = object_mode
    device = getattr(setup, "object_device", None)
    allowed_devices = {
        OBJECT_DEVICE_TRANSMISSIVE_SLM,
        OBJECT_DEVICE_REFLECTIVE_SLM,
        OBJECT_DEVICE_MASK,
    }
    if device is not None:
        device = normalize_named_choice(
            device,
            name="object_device",
            allowed=frozenset(allowed_devices),
        )
    if object_mode == OBJECT_MODE_MASK:
        device = OBJECT_DEVICE_MASK
        object_type = (
            OBJECT_TYPE_TRANSMISSIVE
            if is_two_crystal_architecture(setup)
            else OBJECT_TYPE_REFLECTIVE
        )
        setup.object_type = object_type
        mask_id = getattr(setup, "object_mask_id", None)
        if mask_id is None:
            raise ValueError("object_mode='mask' requires object_mask_id in 1..10")
        object_mask_intensity(mask_id, (8, 8))
        setup.object_mask_id = int(mask_id)
    setup.object_device = device
    if is_three_configuration_v2(setup) and object_mode in {
        OBJECT_MODE_REFERENCE_SLM, OBJECT_MODE_SLM,
    }:
        required_device = (
            OBJECT_DEVICE_TRANSMISSIVE_SLM
            if is_two_crystal_architecture(setup)
            else OBJECT_DEVICE_REFLECTIVE_SLM
        )
        required_type = (
            OBJECT_TYPE_TRANSMISSIVE
            if is_two_crystal_architecture(setup)
            else OBJECT_TYPE_REFLECTIVE
        )
        if device != required_device:
            raise ValueError(
                f"profile {profile.profile_id} reference_slm requires "
                f"object_device={required_device!r}"
            )
        if object_type != required_type:
            raise ValueError(
                f"profile {profile.profile_id} reference_slm requires "
                f"object_type={required_type!r}; a non-SLM topology must use "
                "object_mode='generic_research_object'"
            )

    # Letter ids follow Figure 1; object-plane constants are physical-topology names.
    expected_plane = {
        ARCHITECTURE_PROFILE_A: OBJECT_PLANE_A,
        ARCHITECTURE_PROFILE_B: OBJECT_PLANE_D,
        ARCHITECTURE_PROFILE_C: OBJECT_PLANE_C,
        ARCHITECTURE_PROFILE_D: OBJECT_PLANE_B,
    }[profile.profile_id]
    plane = _normalize_required_string(
        getattr(setup, "object_plane", None), name="object_plane"
    ).lower()
    if plane != expected_plane:
        raise ValueError(
            f"profile {profile.profile_id} requires object_plane={expected_plane!r}, "
            f"got {plane!r}"
        )
    setup.object_plane = plane

    transmission_map = _normalize_required_string(
        getattr(setup, "object_transmission_map", None),
        name="object_transmission_map",
    )
    phase_map = _normalize_required_string(
        getattr(setup, "object_phase_map", None), name="object_phase_map"
    )
    if transmission_map != OBJECT_MAP_TRANSMISSION:
        raise ValueError("object_transmission_map currently supports only 'ref.T'")
    if phase_map != OBJECT_MAP_PHASE:
        raise ValueError("object_phase_map currently supports only 'ref.phi'")

    for name in ("object_forward_transform", "object_backward_transform"):
        transform = _normalize_required_string(getattr(setup, name, None), name=name)
        transform = transform.lower().replace("-", "_").replace(" ", "_")
        if transform not in OBJECT_TRANSFORMS:
            raise ValueError(
                f"{name} must be one of {sorted(OBJECT_TRANSFORMS)}, got {transform!r}"
            )
        setattr(setup, name, transform)

    count = getattr(setup, "object_interaction_count", None)
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise ValueError("object_interaction_count must be the integer 1 or 2")
    expected_count = 1 if (
        is_two_crystal_architecture(setup)
        or object_type == OBJECT_TYPE_REFLECTIVE
    ) else 2
    if int(count) != expected_count:
        raise ValueError(
            f"profile {profile.profile_id} with object_type={object_type!r} requires "
            f"object_interaction_count={expected_count}"
        )
    setup.object_interaction_count = int(count)

    if is_three_configuration_v2(setup) and object_mode == OBJECT_MODE_REFERENCE_SLM:
        if int(count) != 1:
            raise ValueError(
                "reference SLM topology requires exactly one physical object "
                "interaction; the reflective SLM replaces the terminal idler mirror"
            )
    for name in (
        "object_phase_map_path",
        "object_amplitude_map_path",
        "object_intensity_map_path",
    ):
        value = getattr(setup, name, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty path string or null")
    if is_three_configuration_v2(setup):
        if getattr(setup, "object_phase_units", None) != "rad":
            raise ValueError("object_phase_units must be 'rad' for the v2 profiles")
        if getattr(setup, "object_intensity_convention", None) != "intensity":
            raise ValueError(
                "object_intensity_convention must be 'intensity' "
                "(T transmissive, R reflective, both in [0,1])"
            )
        for name in ("object_intensity_T", "object_intensity_R"):
            value = getattr(setup, name, None)
            if value is None:
                continue
            value = _validate_finite_real(value, name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            setattr(setup, name, value)
        phase = getattr(setup, "object_phase_rad", None)
        if phase is not None:
            phase = _validate_finite_real(phase, name="object_phase_rad")
            if not 0.0 <= phase <= 2.0 * np.pi:
                raise ValueError("object_phase_rad must be in [0, 2π]")
            setup.object_phase_rad = phase
        if object_mode == OBJECT_MODE_SLM and is_two_crystal_architecture(setup):
            if (
                getattr(setup, "object_intensity_T", None) is None
                and getattr(setup, "object_intensity_map_path", None) is None
            ):
                pass  # reference arrays or later scalar override remain valid
        if object_mode == OBJECT_MODE_SLM and not is_two_crystal_architecture(setup):
            if (
                getattr(setup, "object_intensity_R", None) is None
                and getattr(setup, "object_intensity_map_path", None) is None
            ):
                pass
        orientation = normalize_named_choice(
            getattr(setup, "object_coordinate_orientation", None),
            name="object_coordinate_orientation",
            allowed=frozenset({"camera_xy", "object_xy"}),
        )
        setup.object_coordinate_orientation = orientation

    reflection_amplitude = _validate_finite_real(
        getattr(setup, "object_reflection_amplitude", None),
        name="object_reflection_amplitude",
    )
    if not 0.0 <= reflection_amplitude <= 1.0:
        raise ValueError("object_reflection_amplitude must be in [0, 1]")
    setup.object_reflection_amplitude = reflection_amplitude
    setup.object_reflection_phase = _validate_finite_real(
        getattr(setup, "object_reflection_phase", None),
        name="object_reflection_phase",
    )
    setup.idler_round_trip_loss = validate_transmission(
        getattr(setup, "idler_round_trip_loss", None),
        name="idler_round_trip_loss",
    )
    if is_two_crystal_architecture(setup) and setup.idler_round_trip_loss != 0.0:
        raise ValueError(
            f"profile {profile.profile_id} has no idler round trip; "
            "idler_round_trip_loss must be 0"
        )
    setup.idler_mode_matching = validate_transmission(
        getattr(setup, "idler_mode_matching", None),
        name="idler_mode_matching",
    )
    setup.relative_interferometric_phase = _validate_finite_real(
        getattr(setup, "relative_interferometric_phase", None),
        name="relative_interferometric_phase",
    )

    spatial_kernel = _normalize_required_string(
        getattr(setup, "spatial_transfer_kernel", None),
        name="spatial_transfer_kernel",
    ).lower().replace("-", "_").replace(" ", "_")
    if spatial_kernel != SPATIAL_TRANSFER_CONDITIONAL_GAUSSIAN_V1:
        raise ValueError(
            "spatial_transfer_kernel must be "
            f"{SPATIAL_TRANSFER_CONDITIONAL_GAUSSIAN_V1!r}"
        )
    setup.spatial_transfer_kernel = spatial_kernel
    object_pitch = _validate_finite_real(
        getattr(setup, "object_map_pixel_um", None),
        name="object_map_pixel_um",
    )
    if object_pitch <= 0.0:
        raise ValueError("object_map_pixel_um must be > 0")
    setup.object_map_pixel_um = object_pitch

    if profile.imaging_mode == IMAGING_MODE_NEAR:
        for name in (
            "near_field_idler_magnification",
            "near_field_signal_magnification",
        ):
            value = _validate_finite_real(getattr(setup, name, None), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
            setattr(setup, name, value)
        if any(
            getattr(setup, name, None) is not None
            for name in (
                "far_field_idler_relay_magnification",
                "far_field_signal_relay_magnification",
            )
        ):
            raise ValueError(
                "near-field profiles must not declare far-field relay magnifications"
            )
    else:
        for name in (
            "far_field_idler_relay_magnification",
            "far_field_signal_relay_magnification",
        ):
            value = _validate_finite_real(getattr(setup, name, None), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
            setattr(setup, name, value)
        if any(
            getattr(setup, name, None) is not None
            for name in (
                "near_field_idler_magnification",
                "near_field_signal_magnification",
            )
        ):
            raise ValueError(
                "far-field profiles must not declare near-field relay magnifications"
            )

    # B/C return-state variables are explicit.  A uses two independent signal
    # paths and therefore must not declare a Michelson signal return.
    return_fields = (
        "signal_return_path_transmission",
        "signal_return_reflection_amplitude",
        "signal_return_phase",
        "signal_return_mode_overlap",
        "signal_return_seeds_second_interaction",
        "idler_return_phase",
    )
    if is_two_crystal_architecture(setup):
        if any(getattr(setup, name, None) is not None for name in return_fields):
            raise ValueError(
                f"profile {profile.profile_id} must not declare "
                "single-crystal return-field parameters"
            )
        return

    setup.signal_return_path_transmission = validate_transmission(
        getattr(setup, "signal_return_path_transmission", None),
        name="signal_return_path_transmission",
    )
    setup.signal_return_reflection_amplitude = validate_transmission(
        getattr(setup, "signal_return_reflection_amplitude", None),
        name="signal_return_reflection_amplitude",
    )
    setup.signal_return_mode_overlap = validate_transmission(
        getattr(setup, "signal_return_mode_overlap", None),
        name="signal_return_mode_overlap",
    )
    setup.signal_return_phase = _validate_finite_real(
        getattr(setup, "signal_return_phase", None), name="signal_return_phase"
    )
    setup.idler_return_phase = _validate_finite_real(
        getattr(setup, "idler_return_phase", None), name="idler_return_phase"
    )
    setup.signal_return_seeds_second_interaction = validate_strict_bool(
        getattr(setup, "signal_return_seeds_second_interaction", None),
        name="signal_return_seeds_second_interaction",
        allow_none=True,
    )


def normalize_two_crystal_pump_topology(value: Any | None) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _TWO_CRYSTAL_PUMP_TOPOLOGIES:
        raise ValueError(
            "two_crystal_pump_topology must be 'branched_independent' or "
            f"'sequential', got {value!r}"
        )
    return key


def validate_generic_benchmark_spectral_filter(setup: "SetupParams") -> None:
    """Enforce common 3 nm detected-signal filter on generic A–D benchmarks."""
    sid = str(getattr(setup, "setup_id", "") or "")
    if sid not in GENERIC_CANONICAL_SETUP_IDS:
        return
    preset = getattr(setup, "camera_filter_preset", None)
    fwhm = getattr(setup, "camera_filter_fwhm_nm", None)
    if preset != CAMERA_FILTER_PRESET_PAPER:
        raise ValueError(
            f"generic configuration {sid!r} requires camera_filter_preset="
            f"{CAMERA_FILTER_PRESET_PAPER!r} (common 3 nm benchmark)"
        )
    if fwhm is None or not np.isclose(
        float(fwhm), GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM
    ):
        raise ValueError(
            f"generic configuration {sid!r} requires camera_filter_fwhm_nm="
            f"{GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM:g} for controlled comparison"
        )


def validate_generic_benchmark_final_camera_collection(setup: "SetupParams") -> None:
    """Enforce uniform final camera collection on generic A–D benchmarks."""
    sid = str(getattr(setup, "setup_id", "") or "")
    if sid not in GENERIC_CANONICAL_SETUP_IDS:
        return
    policy = resolved_final_camera_collection_policy(setup)
    if policy != FINAL_CAMERA_COLLECTION_UNIFORM:
        raise ValueError(
            f"generic configuration {sid!r} requires "
            f"final_camera_collection_policy="
            f"{FINAL_CAMERA_COLLECTION_UNIFORM!r} (approved benchmark)"
        )


def validate_physics_team_refinements(setup: "SetupParams") -> None:
    """Validate Task 11.5 policy fields without changing historical profiles."""
    policy = getattr(setup, "physics_policy_id", None)
    if policy is not None and policy != THREE_CONFIGURATION_PHYSICS_POLICY_ID:
        raise ValueError(
            "physics_policy_id must be null for a historical profile or "
            f"{THREE_CONFIGURATION_PHYSICS_POLICY_ID!r}"
        )

    # The current material/dispersion/nonlinearity implementation is Type-0.
    qpm_type = normalize_qpm_type(getattr(setup, "qpm_type", None))
    setup.qpm_type = qpm_type
    declared_spdc_type = getattr(setup, "spdc_type", None)
    raw_spdc_type = declared_spdc_type or qpm_type
    try:
        spdc_type = normalize_qpm_type(raw_spdc_type)
    except ValueError as exc:
        raise ValueError(
            f"spdc_type={raw_spdc_type!r} is unsupported in this release; "
            "current polarization-aware SPDC support is Type-0 only "
            "(Type-I/Type-II are future extensions)"
        ) from exc
    if spdc_type != qpm_type:
        raise ValueError("spdc_type and qpm_type must describe the same Type-0 physics")
    setup.spdc_type = (
        spdc_type if declared_spdc_type is not None or is_three_configuration_v2(setup)
        else None
    )

    matching = normalize_named_choice(
        getattr(setup, "crystal_matching_mode", None),
        name="crystal_matching_mode",
        allowed=frozenset({CRYSTAL_MATCHING_MATCHED, CRYSTAL_MATCHING_MISMATCH}),
        default=(CRYSTAL_MATCHING_MATCHED if is_three_configuration_v2(setup)
                 and normalize_spdc_architecture(setup.spdc_architecture)
                 == SPDC_ARCHITECTURE_TWO else None),
    )
    setup.crystal_matching_mode = matching
    if normalize_spdc_architecture(setup.spdc_architecture) == SPDC_ARCHITECTURE_TWO:
        if matching == CRYSTAL_MATCHING_MATCHED:
            L2 = setup.crystal_length_mm if setup.crystal_length_NL2_mm is None \
                else setup.crystal_length_NL2_mm
            P2 = setup.poling_period_um if setup.poling_period_NL2_um is None \
                else setup.poling_period_NL2_um
            material2 = setup.crystal_material_NL2 or setup.crystal_material
            type2 = setup.qpm_type_NL2 or setup.qpm_type
            if float(L2) != float(setup.crystal_length_mm):
                raise ValueError(
                    "crystal_matching_mode='matched_nominal' requires L1=L2; "
                    "use 'mismatch_study' for an intentional length mismatch"
                )
            if float(P2) != float(setup.poling_period_um):
                raise ValueError(
                    "crystal_matching_mode='matched_nominal' requires Lambda1=Lambda2; "
                    "use 'mismatch_study' for an intentional poling mismatch"
                )
            if material2 != setup.crystal_material or type2 != setup.qpm_type:
                raise ValueError(
                    "matched_nominal requires equal material and SPDC type in both crystals"
                )
            # QIUP two-crystal: NL2 length and poling equal NL1 and are displayed.
            setup.crystal_length_NL2_mm = float(setup.crystal_length_mm)
            setup.poling_period_NL2_um = float(setup.poling_period_um)
            setup.crystal_material_NL2 = setup.crystal_material
            setup.qpm_type_NL2 = setup.qpm_type
    elif matching is not None:
        raise ValueError(
            "crystal_matching_mode applies only to two_crystal; "
            "single_crystal_double_pass already shares one physical crystal"
        )

    balance = normalize_named_choice(
        getattr(setup, "pump_balance_mode", None),
        name="pump_balance_mode",
        allowed=frozenset({PUMP_BALANCE_MATCHED, PUMP_BALANCE_IMBALANCE}),
        default=(PUMP_BALANCE_MATCHED if is_three_configuration_v2(setup) else None),
    )
    setup.pump_balance_mode = balance
    targets: list[float | None] = []
    for name in ("pump_target_power_NL1_mW", "pump_target_power_NL2_mW"):
        value = getattr(setup, name, None)
        if value is not None:
            value = _validate_finite_real(value, name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
            setattr(setup, name, value)
        targets.append(value)
    if is_three_configuration_v2(setup) and any(x is None for x in targets):
        raise ValueError("v2 profiles require both target interaction pump powers")
    if balance == PUMP_BALANCE_MATCHED and all(x is not None for x in targets):
        if not np.isclose(float(targets[0]), float(targets[1]), rtol=0.0, atol=1e-12):
            raise ValueError(
                "pump_balance_mode='matched_nominal' requires equal target powers; "
                "actual local powers may still differ after modeled loss"
            )

    source_input = normalize_named_choice(
        getattr(setup, "fixed_source_input", None),
        name="fixed_source_input",
        allowed=frozenset({FIXED_SOURCE_INPUT_NBAR, FIXED_SOURCE_INPUT_MEASURED_POWER}),
        default=(FIXED_SOURCE_INPUT_NBAR if is_three_configuration_v2(setup)
                 and normalize_spdc_source_mode(setup.spdc_source_mode)
                 == SPDC_SOURCE_FIXED else None),
    )
    if normalize_spdc_source_mode(setup.spdc_source_mode) != SPDC_SOURCE_FIXED:
        source_input = None
    setup.fixed_source_input = source_input
    if source_input == FIXED_SOURCE_INPUT_MEASURED_POWER:
        if normalize_spdc_source_mode(setup.spdc_source_mode) != SPDC_SOURCE_FIXED:
            raise ValueError("fixed_source_input applies only to spdc_source_mode='fixed_nbar'")
        if setup.measured_power_W is None:
            raise ValueError("fixed_source_input='measured_power' requires measured_power_W")
        wavelength = setup.measured_power_wavelength_nm
        if wavelength is None:
            wavelength = setup.lambda_s_nm
            setup.measured_power_wavelength_nm = float(wavelength)
        photon_rate_from_power_W(setup.measured_power_W, wavelength)
        if setup.measured_power_to_nbar_per_W is None:
            raise ValueError(
                "measured power determines photon rate, not mean photons per mode; "
                "a calibrated measured_power_to_nbar_per_W mapping is required"
            )
        for name in ("measured_power_to_nbar_per_W", "measured_power_to_nbar_per_W_NL2"):
            value = getattr(setup, name, None)
            if value is not None:
                value = _validate_finite_real(value, name=name)
                if value < 0.0:
                    raise ValueError(f"{name} must be >= 0")
                setattr(setup, name, value)
        if not str(setup.measured_power_to_nbar_calibration_id or "").strip():
            raise ValueError(
                "measured power to nbar conversion requires "
                "measured_power_to_nbar_calibration_id"
            )

    preset = normalize_named_choice(
        getattr(setup, "camera_filter_preset", None),
        name="camera_filter_preset",
        allowed=frozenset({
            CAMERA_FILTER_PRESET_PAPER,
            CAMERA_FILTER_PRESET_LAB,
            CAMERA_FILTER_PRESET_CUSTOM,
        }),
        default=None,
    )
    setup.camera_filter_preset = preset
    if preset is not None:
        preset_values = {
            CAMERA_FILTER_PRESET_PAPER: (810.0, 3.0),
            CAMERA_FILTER_PRESET_LAB: (810.0, 10.0),
        }
        if preset in preset_values:
            center_default, fwhm_default = preset_values[preset]
            if setup.camera_filter_center_nm is None:
                setup.camera_filter_center_nm = center_default
            if setup.camera_filter_fwhm_nm is None:
                setup.camera_filter_fwhm_nm = fwhm_default
            if not np.isclose(float(setup.camera_filter_center_nm), center_default):
                raise ValueError(f"{preset} requires camera_filter_center_nm={center_default:g}")
            if not np.isclose(float(setup.camera_filter_fwhm_nm), fwhm_default):
                raise ValueError(f"{preset} requires camera_filter_fwhm_nm={fwhm_default:g}")
        model = normalize_named_choice(
            getattr(setup, "camera_filter_model", None),
            name="camera_filter_model",
            allowed=frozenset({CAMERA_FILTER_MODEL_GAUSSIAN, CAMERA_FILTER_MODEL_MEASURED}),
            default=CAMERA_FILTER_MODEL_GAUSSIAN,
        )
        setup.camera_filter_model = model
        for name in ("camera_filter_center_nm", "camera_filter_fwhm_nm",
                     "signal_spectrum_fwhm_nm"):
            value = getattr(setup, name, None)
            if value is None:
                if name == "signal_spectrum_fwhm_nm":
                    value = 20.0
                    setattr(setup, name, value)
                else:
                    raise ValueError(f"filter selection requires {name}")
            value = _validate_finite_real(value, name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")
            setattr(setup, name, value)
        waves = setup.camera_filter_wavelengths_nm
        transmissions = setup.camera_filter_transmissions
        if model == CAMERA_FILTER_MODEL_MEASURED:
            if waves is None or transmissions is None:
                raise ValueError("measured camera filter requires wavelength and transmission arrays")
            w = np.asarray(waves, dtype=np.float64)
            t = np.asarray(transmissions, dtype=np.float64)
            if w.ndim != 1 or t.ndim != 1 or w.size < 2 or w.shape != t.shape:
                raise ValueError("measured filter arrays must be matching 1-D arrays with >=2 points")
            if not np.all(np.isfinite(w)) or not np.all(np.diff(w) > 0.0):
                raise ValueError("measured filter wavelengths must be finite and strictly increasing")
            if not np.all(np.isfinite(t)) or np.any(t < 0.0) or np.any(t > 1.0):
                raise ValueError("measured filter transmissions must be finite and in [0,1]")
            setup.camera_filter_wavelengths_nm = [float(x) for x in w]
            setup.camera_filter_transmissions = [float(x) for x in t]
        elif waves is not None or transmissions is not None:
            raise ValueError("measured filter arrays require camera_filter_model='measured'")
    elif is_three_configuration_v2(setup):
        raise ValueError("v2 profiles require an explicit camera_filter_preset")

    if is_three_configuration_v2(setup) and bool(setup.use_calibrated_poling_period):
        raise ValueError(
            "v2 uses the user-requested poling period to calculate delta_k; "
            "use_calibrated_poling_period cannot silently replace it"
        )

    validate_generic_benchmark_spectral_filter(setup)
    validate_generic_benchmark_final_camera_collection(setup)


def validate_spdc_energy_conservation(setup: "SetupParams") -> float:
    """Relative inverse-wavelength residual with all wavelengths in nm.

    Nominal 532/810/1550 nm wavelengths are rounded (residual 1.593e-5).
    The 1e-4 tolerance admits that rounding only.
    """
    wavelengths = (setup.lambda_p_nm, setup.lambda_s_nm, setup.lambda_i_nm)
    if any(isinstance(x, bool) or not isinstance(x, Real)
           or not np.isfinite(float(x)) or float(x) <= 0 for x in wavelengths):
        raise ValueError("SPDC wavelengths must be finite and positive in nm")
    inverse_p = 1.0 / float(setup.lambda_p_nm)
    residual = (inverse_p - 1.0 / float(setup.lambda_s_nm)
                - 1.0 / float(setup.lambda_i_nm)) / inverse_p
    if abs(residual) > 1.0e-4:
        raise ValueError(
            "SPDC energy conservation requires 1/lambda_p = "
            "1/lambda_s + 1/lambda_i (consistent nm units); "
            f"relative residual={residual:.6g}"
        )
    return float(residual)


def spdc_propagation_direction(setup: "SetupParams", *, interaction: int) -> str:
    if int(interaction) not in (1, 2):
        raise ValueError("interaction must be 1 or 2")
    return resolve_architecture_profile(setup).propagation_directions[
        int(interaction) - 1
    ]


def normalize_imaging_mode(value: Any | None) -> str:
    """Normalize YAML/API imaging_mode to near_field | far_field."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return IMAGING_MODE_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _IMAGING_MODE_ALIASES:
        raise ValueError(
            "imaging_mode must be 'near_field' or 'far_field' "
            f"(aliases: nf/ff, position/momentum, image_plane/fourier), got {value!r}"
        )
    return _IMAGING_MODE_ALIASES[key]


def imaging_correlation_type(setup: "SetupParams" | str) -> str:
    """position (near-field) or momentum (far-field)."""
    mode = (
        normalize_imaging_mode(setup)
        if isinstance(setup, str)
        else normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    )
    return "position" if mode == IMAGING_MODE_NEAR else "momentum"


def normalize_blur_model(value: Any | None) -> str:
    """Normalize YAML/API blur_model to single_blur | double_blur."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return BLUR_MODEL_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _BLUR_MODEL_ALIASES:
        raise ValueError(
            "blur_model must be 'single_blur' or 'double_blur' "
            f"(aliases: single/optical_only, double/spdc_and_optical), got {value!r}"
        )
    return _BLUR_MODEL_ALIASES[key]


def default_crystal_length_mm_for_blur(blur_model: Any | None = None) -> float:
    """Blur-linked default nonlinear crystal length [mm].

    ``single_blur`` → 10 mm; ``double_blur`` → 2 mm per crystal/pass.
    User may override to any value in [2, 20].
    """
    model = normalize_blur_model(blur_model)
    if model == BLUR_MODEL_SINGLE:
        return float(CRYSTAL_LENGTH_MM_DEFAULT_SINGLE_BLUR)
    return float(CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR)


def is_three_configuration_v2(setup_or_policy: Any) -> bool:
    """Return whether the frozen v2 physics policy is active.

    The name is historical. The flag is the string
    ``three_configuration_physics_team_v2``, not a lock to three profiles.
    Registered arrangements A, B, C, and D all use this policy id.
    """
    policy = (
        setup_or_policy
        if isinstance(setup_or_policy, str) or setup_or_policy is None
        else getattr(setup_or_policy, "physics_policy_id", None)
    )
    return policy == THREE_CONFIGURATION_PHYSICS_POLICY_ID


def default_crystal_length_mm_for_policy(
    physics_policy_id: str | None,
    blur_model: Any | None = None,
) -> float:
    """Resolve an omitted physical length without changing legacy policy.

    The Task 11.5 three-profile policy uses 2 mm for every blur selection.
    Historical profiles retain the reviewed single/double blur defaults.
    """
    if is_three_configuration_v2(physics_policy_id):
        return float(CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR)
    return default_crystal_length_mm_for_blur(blur_model)


def crystal_length_policy(physics_policy_id: str | None = None) -> dict[str, Any]:
    """Return the approved user-variable crystal-length policy for provenance."""
    if is_three_configuration_v2(physics_policy_id):
        return {
            "policy_id": THREE_CONFIGURATION_PHYSICS_POLICY_ID,
            "user_variable": True,
            "minimum_mm": float(CRYSTAL_LENGTH_MM_MIN),
            "maximum_mm": float(CRYSTAL_LENGTH_MM_MAX),
            "default_mm": float(CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR),
            "default_independent_of_blur_model": True,
            "legacy_policy_preserved": CRYSTAL_LENGTH_POLICY_ID,
        }
    return {
        "policy_id": CRYSTAL_LENGTH_POLICY_ID,
        "user_variable": True,
        "minimum_mm": float(CRYSTAL_LENGTH_MM_MIN),
        "maximum_mm": float(CRYSTAL_LENGTH_MM_MAX),
        "single_blur_default_mm": float(CRYSTAL_LENGTH_MM_DEFAULT_SINGLE_BLUR),
        "double_blur_default_mm_per_interaction": float(
            CRYSTAL_LENGTH_MM_DEFAULT_DOUBLE_BLUR
        ),
    }


def normalize_named_choice(
    value: Any | None,
    *,
    name: str,
    allowed: frozenset[str],
    default: str | None = None,
) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return key


def photon_rate_from_power_W(power_W: float, wavelength_nm: float) -> float:
    """Photon rate [s^-1] = P*lambda/(h*c); this is not modal occupancy."""
    power = _validate_finite_real(power_W, name="measured_power_W")
    wavelength = _validate_finite_real(
        wavelength_nm, name="measured_power_wavelength_nm"
    )
    if power < 0.0:
        raise ValueError("measured_power_W must be >= 0")
    if wavelength <= 0.0:
        raise ValueError("measured_power_wavelength_nm must be > 0")
    return float(
        power * wavelength * 1.0e-9
        / (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S)
    )


def normalize_object_preprocessing(value: Any | None) -> str:
    """Normalize object_preprocessing to calibrated | visualization_only."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return OBJECT_PREPROCESS_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _OBJECT_PREPROCESS_ALIASES:
        raise ValueError(
            "object_preprocessing must be 'calibrated' or 'visualization_only' "
            f"(aliases: physical/calib, viz/display/minmax), got {value!r}"
        )
    return _OBJECT_PREPROCESS_ALIASES[key]


def resolve_blur_sigmas(
    *,
    sigma_spdc_px: float,
    sigma_optical_px: float,
    blur_model: Any | None = None,
) -> dict[str, Any]:
    """Resolved blur widths for the selected blur_model (V-P0-01).

    single_blur: σ_total = σ_optical (SPDC stage off).
    double_blur: σ_total = √(σ_SPDC² + σ_optical²); both stages applied
    sequentially (complex-field SPDC, then intensity optical PSF).
    """
    model = normalize_blur_model(blur_model)
    sig_opt = float(sigma_optical_px)
    sig_spdc = float(sigma_spdc_px)
    if not np.isfinite(sig_opt) or sig_opt < 0.0:
        raise ValueError(f"sigma_optical_px must be finite and >= 0, got {sigma_optical_px}")
    if not np.isfinite(sig_spdc) or sig_spdc < 0.0:
        raise ValueError(f"sigma_spdc_px must be finite and >= 0, got {sigma_spdc_px}")
    if model == BLUR_MODEL_SINGLE:
        return {
            "blur_model": model,
            "apply_spdc_psf": False,
            "sigma_spdc_px": sig_spdc,
            "sigma_spdc_applied_px": 0.0,
            "sigma_optical_px": sig_opt,
            "sigma_total_px": sig_opt,
        }
    return {
        "blur_model": model,
        "apply_spdc_psf": True,
        "sigma_spdc_px": sig_spdc,
        "sigma_spdc_applied_px": sig_spdc,
        "sigma_optical_px": sig_opt,
        "sigma_total_px": float(np.sqrt(sig_spdc * sig_spdc + sig_opt * sig_opt)),
    }


def normalize_spdc_gain_mode(value: Any | None) -> str:
    """Normalize YAML/API spdc_gain_mode to low_gain | high_gain."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return SPDC_GAIN_MODE_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _SPDC_GAIN_MODE_ALIASES:
        raise ValueError(
            "spdc_gain_mode must be 'low_gain' or 'high_gain' "
            f"(aliases: low/lg, high/hg), got {value!r}"
        )
    return _SPDC_GAIN_MODE_ALIASES[key]


def normalize_detector_mode(value: Any | None) -> str:
    """Normalize YAML/API detector_mode to analog_emccd | photon_counting.

    V-P1-07: mode is explicit — callers must set it; do not infer from
    EM gain, flux, or ``LatentState.photon_counting``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return DETECTOR_MODE_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DETECTOR_MODE_ALIASES:
        raise ValueError(
            "detector_mode must be 'analog_emccd' or 'photon_counting' "
            f"(aliases: analog/emccd, pc/photon), got {value!r}"
        )
    return _DETECTOR_MODE_ALIASES[key]


def normalize_detector_output_port(value: Any | None) -> str:
    """Normalize the physically selected signal output to ``plus`` or ``minus``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return DETECTOR_OUTPUT_PORT_DEFAULT
    raw = str(value).strip().lower()
    key = raw if raw in {"+", "-"} else raw.replace("-", "_").replace(" ", "_")
    if key not in _DETECTOR_OUTPUT_PORT_ALIASES:
        raise ValueError(
            "detector_output_port must be 'plus' or 'minus' "
            f"(aliases: I_plus/I_minus, +/-), got {value!r}"
        )
    return _DETECTOR_OUTPUT_PORT_ALIASES[key]


def resolved_adc_bit_depth(setup: "SetupParams") -> int:
    """Return the declared ADC width, or derive the legacy width from max_adu."""
    explicit = getattr(setup, "adc_bit_depth", None)
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, Integral):
            raise ValueError("adc_bit_depth must be an integer")
        bits = int(explicit)
        if not 1 <= bits <= 32:
            raise ValueError("adc_bit_depth must be in [1, 32]")
        return bits
    max_adu = _validate_finite_real(setup.max_adu, name="max_adu")
    if max_adu <= 0.0:
        raise ValueError("max_adu must be > 0")
    return int(np.ceil(np.log2(max_adu + 1.0)))


def validate_detector_configuration(setup: "SetupParams") -> "SetupParams":
    """Fail closed on detector configuration before a forward run can start."""
    setup.detector_mode = normalize_detector_mode(
        getattr(setup, "detector_mode", None)
    )
    # Preserve ``None`` on legacy objects so historical serialized hashes do
    # not change. New profiles store their explicit choice.
    if getattr(setup, "detector_output_port", None) is not None:
        setup.detector_output_port = normalize_detector_output_port(
            setup.detector_output_port
        )

    setup.QE = validate_quantum_efficiency(setup.QE)
    nonnegative = (
        "read_noise0_e", "read_noise_span_e", "cic0", "prnu_sigma",
        "dark0", "dark_exp", "dark_arrhenius_prefactor", "Eg_eV",
        "hot_frac_at_s1", "hot_amp_lo", "hot_amp_hi",
        "hot_T_coef_per_C", "accidental_e_at_s1", "pc_threshold_k",
    )
    for name in nonnegative:
        value = _validate_finite_real(getattr(setup, name), name=name)
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if float(setup.hot_frac_at_s1) > 1.0:
        raise ValueError("hot_frac_at_s1 must be in [0, 1]")
    if float(setup.hot_amp_hi) < float(setup.hot_amp_lo):
        raise ValueError("hot_amp_hi must be >= hot_amp_lo")

    for name in ("full_well_e", "g_adu", "max_adu", "em_gain_default"):
        value = _validate_finite_real(getattr(setup, name), name=name)
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0")
    bias = _validate_finite_real(setup.bias_adu, name="bias_adu")
    if not 0.0 <= bias <= float(setup.max_adu):
        raise ValueError("bias_adu must be in [0, max_adu]")
    bits = resolved_adc_bit_depth(setup)
    if float(setup.max_adu) > float((1 << bits) - 1):
        raise ValueError("max_adu exceeds the range represented by adc_bit_depth")

    if setup.read_noise_model not in {"legacy_s_det", "datasheet_fixed"}:
        raise ValueError(
            "read_noise_model must be 'legacy_s_det' or 'datasheet_fixed'"
        )
    if setup.emccd_model not in {"legacy", "register"}:
        raise ValueError("emccd_model must be 'legacy' or 'register'")
    if setup.hot_pixel_model not in {"legacy", "mask_T"}:
        raise ValueError("hot_pixel_model must be 'legacy' or 'mask_T'")

    for name in ("T_sensor_C", "T_sensor_min_C", "T_sensor_max_C"):
        _validate_finite_real(getattr(setup, name), name=name)
    if float(setup.T_sensor_min_C) > float(setup.T_sensor_max_C):
        raise ValueError("T_sensor_min_C must be <= T_sensor_max_C")
    return setup


def detector_mode_is_photon_counting(value: Any | None = None) -> bool:
    """True iff ``detector_mode`` is the photon-counting path."""
    return normalize_detector_mode(value) == DETECTOR_MODE_PC


def detector_mode_from_photon_counting_flag(photon_counting: bool) -> str:
    """Map an explicit boolean API flag onto ``detector_mode`` (no inference)."""
    return DETECTOR_MODE_PC if bool(photon_counting) else DETECTOR_MODE_ANALOG


def spdc_gain_mode_nbar(gain_mode: Any | None = None) -> float:
    """Default mean photons per mode for ``spdc_gain_mode`` (V-P0-03)."""
    mode = normalize_spdc_gain_mode(gain_mode)
    if mode == SPDC_GAIN_MODE_HIGH:
        return float(SPDC_GAIN_MODE_HIGH_NBAR)
    return float(SPDC_GAIN_MODE_LOW_NBAR)


def normalize_spdc_source_mode(value: Any | None) -> str:
    """Normalize spdc_source_mode to fixed_nbar | calibrated_source (V-P1-01)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return SPDC_SOURCE_DEFAULT
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _SPDC_SOURCE_MODE_ALIASES:
        raise ValueError(
            "spdc_source_mode must be 'fixed_nbar' or 'calibrated_source' "
            f"(aliases: fixed/preset, calibrated/physical/predictive), got {value!r}"
        )
    return _SPDC_SOURCE_MODE_ALIASES[key]


def normalize_spdc_architecture(value: Any | None) -> str:
    """Canonical architecture name. Missing/blank → single-crystal paper path."""
    if value is None:
        return SPDC_ARCHITECTURE_DEFAULT
    key = str(value).strip().lower().replace("-", "_")
    if not key:
        return SPDC_ARCHITECTURE_DEFAULT
    if key not in _SPDC_ARCHITECTURE_ALIASES:
        raise ValueError(
            "spdc_architecture must be "
            f"{SPDC_ARCHITECTURE_SINGLE!r} or {SPDC_ARCHITECTURE_TWO!r}, "
            f"got {value!r}"
        )
    return _SPDC_ARCHITECTURE_ALIASES[key]


def is_two_crystal_architecture(setup: "SetupParams") -> bool:
    """True only when Physical Mode is on and architecture is two_crystal."""
    if not bool(getattr(setup, "use_physical_spdc_gain", False)):
        return False
    arch = normalize_spdc_architecture(getattr(setup, "spdc_architecture", None))
    return arch == SPDC_ARCHITECTURE_TWO


def uses_shared_crystal(setup: "SetupParams") -> bool:
    """Physical Mode paper path: both passes share one crystal state."""
    return bool(getattr(setup, "use_physical_spdc_gain", False)) and (
        not is_two_crystal_architecture(setup)
    )


def physical_crystal_id(setup: "SetupParams", *, interaction: int = 1) -> str:
    """User-facing crystal label: NL for one crystal, NL1/NL2 for two."""
    if is_two_crystal_architecture(setup) and int(interaction) == 2:
        return PHYSICAL_CRYSTAL_NL2
    if is_two_crystal_architecture(setup):
        return PHYSICAL_CRYSTAL_NL1
    return PHYSICAL_CRYSTAL_NL


def object_mask_intensity(mask_id: int, shape: tuple[int, int]) -> np.ndarray:
    """Catalog of 10 binary intensity masks (T or R in {0, 1})."""
    if isinstance(mask_id, bool) or not isinstance(mask_id, Integral):
        raise ValueError("object_mask_id must be an integer in 1..10")
    mid = int(mask_id)
    if not OBJECT_MASK_ID_MIN <= mid <= OBJECT_MASK_ID_MAX:
        raise ValueError(
            f"object_mask_id must be in [{OBJECT_MASK_ID_MIN}, "
            f"{OBJECT_MASK_ID_MAX}], got {mask_id!r}"
        )
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError("object mask shape must be a positive 2-D grid")
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = (ny - 1) * 0.5
    cx = (nx - 1) * 0.5
    ry = np.maximum(ny * 0.22, 1.0)
    rx = np.maximum(nx * 0.22, 1.0)
    if mid == 1:
        mask = np.ones((ny, nx), dtype=np.float64)
    elif mid == 2:
        mask = np.zeros((ny, nx), dtype=np.float64)
    elif mid == 3:
        mask = (xx < nx * 0.5).astype(np.float64)
    elif mid == 4:
        mask = (xx >= nx * 0.5).astype(np.float64)
    elif mid == 5:
        mask = (yy < ny * 0.5).astype(np.float64)
    elif mid == 6:
        mask = (yy >= ny * 0.5).astype(np.float64)
    elif mid == 7:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float64)
    elif mid == 8:
        mask = (
            (np.abs(xx - cx) <= rx) & (np.abs(yy - cy) <= ry)
        ).astype(np.float64)
    elif mid == 9:
        half = max(int(round(nx * 0.08)), 1)
        mask = (np.abs(xx - cx) <= half).astype(np.float64)
    else:
        bar_w = max(int(round(nx * 0.08)), 1)
        gap = max(int(round(nx * 0.10)), 1)
        centers = (cx - (bar_w + gap), cx, cx + (bar_w + gap))
        mask = np.zeros((ny, nx), dtype=bool)
        band = np.abs(yy - cy) <= ry * 1.2
        for c in centers:
            mask |= (np.abs(xx - c) <= bar_w) & band
        mask = mask.astype(np.float64)
    return mask


def resolve_object_intensity_and_phase(
    T: np.ndarray,
    phi: np.ndarray,
    setup: "SetupParams",
) -> dict[str, Any]:
    """Resolve intensity T/R and phase from mask, SLM scalars, or reference maps."""
    T_in = np.asarray(T, dtype=np.float64)
    phi_in = np.asarray(phi, dtype=np.float64)
    if T_in.ndim != 2 or phi_in.shape != T_in.shape:
        raise ValueError("object intensity and phase maps must be matching 2-D arrays")
    mode = getattr(setup, "object_mode", None)
    two = is_two_crystal_architecture(setup)
    coefficient = "intensity_transmission_T" if (
        two or str(getattr(setup, "object_type", "")).lower() == OBJECT_TYPE_TRANSMISSIVE
    ) else "intensity_reflectance_R"
    source = "reference_array"
    if mode == OBJECT_MODE_MASK:
        mask_id = getattr(setup, "object_mask_id", None)
        T_out = object_mask_intensity(mask_id if mask_id is not None else 1, T_in.shape)
        phi_out = np.zeros_like(T_out)
        source = f"catalog_mask_{int(mask_id if mask_id is not None else 1)}"
    else:
        T_out = T_in
        phi_out = phi_in
        scalar = (
            getattr(setup, "object_intensity_T", None)
            if coefficient == "intensity_transmission_T"
            else getattr(setup, "object_intensity_R", None)
        )
        if scalar is not None:
            T_out = np.full(T_in.shape, float(scalar), dtype=np.float64)
            source = "user_scalar_intensity"
        phase = getattr(setup, "object_phase_rad", None)
        if phase is not None:
            phi_out = np.full(T_in.shape, float(phase), dtype=np.float64)
            source = (
                "user_scalar_intensity_and_phase"
                if scalar is not None else "user_scalar_phase"
            )
    if not np.all(np.isfinite(T_out)) or np.any(T_out < 0.0) or np.any(T_out > 1.0):
        raise ValueError(f"{coefficient} must be finite and in [0, 1]")
    if not np.all(np.isfinite(phi_out)):
        raise ValueError("object phase must be finite")
    if mode == OBJECT_MODE_SLM and getattr(setup, "object_phase_rad", None) is not None:
        if np.any(phi_out < 0.0) or np.any(phi_out > 2.0 * np.pi):
            raise ValueError("SLM object_phase_rad must be in [0, 2π]")
    return {
        "T": T_out,
        "phi": phi_out,
        "intensity_coefficient": coefficient,
        "source": source,
        "object_mode": mode,
    }


def lab_poling_period_um(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float:
    """Lab Λ [µm] for one crystal. NL2 uses ``poling_period_NL2_um`` if set."""
    if is_two_crystal_architecture(setup) and int(interaction) == 2:
        p2 = getattr(setup, "poling_period_NL2_um", None)
        if p2 is not None:
            return validate_poling_period_um(p2)
    return validate_poling_period_um(setup.poling_period_um)


def spdc_crystal_state(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> dict[str, Any]:
    """Crystal state for one SPDC interaction.

    Single-crystal double-pass: both interactions return the shared NL
    state (report §7.11). two_crystal: NL2 uses T2; L2 and Λ2 copy NL1
    in matched_nominal QIUP.
    """
    use_nl2 = is_two_crystal_architecture(setup) and int(interaction) == 2
    if use_nl2:
        T = float(setup.T_crystal_NL2_C)
        if setup.crystal_length_NL2_mm is not None:
            L_mm = float(setup.crystal_length_NL2_mm)
        else:
            L_mm = float(setup.crystal_length_mm)
    else:
        T = float(setup.T_crystal_NL1_C)
        L_mm = float(setup.crystal_length_mm)
    L_mm = float(validate_crystal_length_mm(L_mm))
    lp_um = float(setup.lambda_p_nm) * 1.0e-3
    ls_um = float(setup.lambda_s_nm) * 1.0e-3
    li_um = float(setup.lambda_i_nm) * 1.0e-3
    deff = getattr(setup, "d_eff_pm_per_V", None)
    d33 = getattr(setup, "d33_pm_per_V", None)
    qpm_order = setup.qpm_order
    qpm_duty = setup.qpm_duty_cycle
    if use_nl2:
        qpm_order = setup.qpm_order_NL2 or qpm_order
        qpm_duty = (setup.qpm_duty_cycle_NL2 if setup.qpm_duty_cycle_NL2 is not None
                    else qpm_duty)
        if setup.d_eff_NL2_pm_per_V is not None:
            deff = setup.d_eff_NL2_pm_per_V
        elif setup.d33_NL2_pm_per_V is not None:
            deff = derive_d_eff_pm_per_V(
                d33_pm_per_V=setup.d33_NL2_pm_per_V,
                qpm_order=qpm_order,
                qpm_duty_cycle=qpm_duty,
            )
        d33 = setup.d33_NL2_pm_per_V if setup.d33_NL2_pm_per_V is not None else d33
    return {
        "interaction": int(interaction),
        "architecture": normalize_spdc_architecture(
            getattr(setup, "spdc_architecture", None)
        ),
        "shared_crystal": uses_shared_crystal(setup),
        "physical_crystal_id": physical_crystal_id(setup, interaction=interaction),
        "T_crystal_C": T,
        "crystal_length_mm": L_mm,
        "crystal_length_m": L_mm * 1.0e-3,
        "poling_period_um_qpm": qpm_poling_period_um(
            setup, T_C=T, interaction=interaction
        ),
        "d_eff_pm_per_V": None if deff is None else float(deff),
        "d33_pm_per_V": d33,
        "material": (setup.crystal_material_NL2 or setup.crystal_material)
        if use_nl2 else setup.crystal_material,
        "qpm_type": (setup.qpm_type_NL2 or setup.qpm_type) if use_nl2 else setup.qpm_type,
        "qpm_order": qpm_order,
        "qpm_duty_cycle": qpm_duty,
        "polarization": (
            setup.pol_pump_NL2 or setup.pol_pump,
            setup.pol_signal_NL2 or setup.pol_signal,
            setup.pol_idler_NL2 or setup.pol_idler,
        ) if use_nl2 else (setup.pol_pump, setup.pol_signal, setup.pol_idler),
        "crystal_orientation_deg": (
            setup.crystal_orientation_NL2_deg if use_nl2
            else setup.crystal_orientation_NL1_deg
        ),
        "crystal_orientation_model_id": CRYSTAL_ORIENTATION_MODEL_ID,
        "propagation_direction": spdc_propagation_direction(
            setup, interaction=interaction
        ),
        "physical_orientation_distinct_from_propagation_direction": True,
        "n_pump": float(ktp_nz(lp_um, T)),
        "n_signal": float(ktp_nz(ls_um, T)),
        "n_idler": float(ktp_nz(li_um, T)),
        "lambda_p_nm": float(setup.lambda_p_nm),
        "lambda_s_nm": float(setup.lambda_s_nm),
        "lambda_i_nm": float(setup.lambda_i_nm),
        "n_spdc_interactions": n_spdc_interactions(setup),
        "n_crystals_hardware": int(setup.n_crystals),
    }


def spdc_shared_crystal_state(setup: "SetupParams") -> dict[str, Any]:
    """Crystal 1 / shared double-pass state (report §7.11 / §5.10.1).

    Per-pass geometry (w_p, z0, O_geom) and pump powers are not in this state.
    """
    return spdc_crystal_state(setup, interaction=1)


def architecture_parameter_snapshot(setup: "SetupParams") -> dict[str, Any]:
    """Inventory and interaction inputs; no new optical equations are used."""
    profile = resolve_architecture_profile(setup)
    states = [spdc_crystal_state(setup, interaction=i) for i in (1, 2)]
    crystals = []
    for idx, crystal_id in enumerate(profile.physical_crystal_ids):
        state = states[idx]
        crystals.append({
            "physical_crystal_id": crystal_id,
            "material": setup.crystal_material,
            "qpm_type": setup.qpm_type,
            "qpm_order": setup.qpm_order,
            "qpm_duty_cycle": setup.qpm_duty_cycle,
            "polarization": (setup.pol_pump, setup.pol_signal, setup.pol_idler),
            **state,
        })
    interactions = []
    for index in (1, 2):
        interactions.append({
            "interaction_index": index,
            "physical_crystal_id": profile.interaction_crystal_ids[index - 1],
            "propagation_direction": profile.propagation_directions[index - 1],
            "pump_waist_um": resolved_pump_waist_um(setup, interaction=index),
            "pump_focus_um": (setup.pump_focus_z_NL1_um if index == 1
                              else setup.pump_focus_z_NL2_um),
            "local_pump_power_mW": None,
            "relative_interaction_phase_rad": getattr(
                setup, "relative_interferometric_phase", None
            ),
            "generation_alignment": None,
            "physical_geometry_overlap": None,
            "phase_mismatch_per_m": None,
        })
    try:
        pump_snapshot = pump_powers_from_setup(setup)
    except ValueError as exc:
        pump_snapshot = {
            "pump_balance_mode": getattr(setup, "pump_balance_mode", None),
            "P_target_1_mW": getattr(setup, "pump_target_power_NL1_mW", None),
            "P_target_2_mW": getattr(setup, "pump_target_power_NL2_mW", None),
            "P_actual_1_mW": None,
            "P_actual_2_mW": None,
            "delta_P": None,
            "abs_delta_P": None,
            "unresolved_reason": str(exc),
        }
    return {
        "physics_policy_id": getattr(setup, "physics_policy_id", None),
        "profile": profile.to_dict(),
        "spdc_type": getattr(setup, "spdc_type", None) or setup.qpm_type,
        "crystal_matching_mode": getattr(setup, "crystal_matching_mode", None),
        "propagation_model": getattr(setup, "architecture_propagation_model", None),
        "object": {
            "object_type": getattr(setup, "object_type", None),
            "object_device": getattr(setup, "object_device", None),
            "object_mode": getattr(setup, "object_mode", None),
            "object_mask_id": getattr(setup, "object_mask_id", None),
            "object_plane": getattr(setup, "object_plane", None),
            "intensity_transmission_T": getattr(setup, "object_intensity_T", None),
            "intensity_reflectance_R": getattr(setup, "object_intensity_R", None),
            "object_phase_rad": getattr(setup, "object_phase_rad", None),
            "intensity_coefficient": (
                "intensity_transmission_T"
                if (
                    is_two_crystal_architecture(setup)
                    or str(getattr(setup, "object_type", "")).lower()
                    == OBJECT_TYPE_TRANSMISSIVE
                )
                else "intensity_reflectance_R"
            ),
            "transmission_or_reflectance_map": getattr(
                setup, "object_transmission_map", None
            ),
            "phase_map": getattr(setup, "object_phase_map", None),
            "forward_transform": getattr(setup, "object_forward_transform", None),
            "backward_transform": getattr(setup, "object_backward_transform", None),
            "interaction_count": getattr(setup, "object_interaction_count", None),
        },
        "return_fields": {
            "idler_round_trip_loss": getattr(setup, "idler_round_trip_loss", None),
            "idler_mode_matching": getattr(setup, "idler_mode_matching", None),
            "idler_return_phase": getattr(setup, "idler_return_phase", None),
            "signal_return_path_transmission": getattr(
                setup, "signal_return_path_transmission", None
            ),
            "signal_return_reflection_amplitude": getattr(
                setup, "signal_return_reflection_amplitude", None
            ),
            "signal_return_phase": getattr(setup, "signal_return_phase", None),
            "signal_return_mode_overlap": getattr(
                setup, "signal_return_mode_overlap", None
            ),
            "signal_return_seeds_second_interaction": getattr(
                setup, "signal_return_seeds_second_interaction", None
            ),
        },
        "physical_crystals": crystals,
        "two_crystal_length_policy": (
            "NL2_length_equals_NL1"
            if is_two_crystal_architecture(setup)
            else "single_physical_crystal_NL"
        ),
        "crystal_length_NL1_mm": float(setup.crystal_length_mm),
        "crystal_length_NL2_mm": (
            float(setup.crystal_length_mm)
            if is_two_crystal_architecture(setup)
            else None
        ),
        "interactions": interactions,
        "qpm": [
            qpm_configuration_evidence(setup, interaction=1),
            qpm_configuration_evidence(setup, interaction=2),
        ],
        "pump_balance": pump_snapshot,
        "camera_filter": {
            "preset": getattr(setup, "camera_filter_preset", None),
            "center_nm": getattr(setup, "camera_filter_center_nm", None),
            "fwhm_nm": getattr(setup, "camera_filter_fwhm_nm", None),
            "model": getattr(setup, "camera_filter_model", None),
            "spectral_model": (
                "reduced_order_separable_spectral_quadrature"
                if getattr(setup, "camera_filter_preset", None) is not None else None
            ),
        },
        "energy_conservation_relative_residual": validate_spdc_energy_conservation(setup),
        "unsupported_parameters": [
            "crystal_orientation",
        ],
    }


def normalize_crystal_material(value: Any | None) -> str:
    """Normalize crystal material; only ppKTP is supported (V-P1-08)."""
    if value is None:
        return CRYSTAL_MATERIAL_DEFAULT
    key = str(value).strip().lower().replace(" ", "")
    if key not in _CRYSTAL_MATERIAL_ALIASES:
        raise ValueError(
            f"crystal_material={value!r} is unsupported (V-P1-08). "
            f"Supported: '{CRYSTAL_MATERIAL_DEFAULT}' only."
        )
    return _CRYSTAL_MATERIAL_ALIASES[key]


def normalize_qpm_type(value: Any | None) -> str:
    """Normalize QPM type; only type0 is supported (V-P1-08)."""
    if value is None:
        return QPM_TYPE_DEFAULT
    key = str(value).strip().lower().replace(" ", "")
    if key not in _QPM_TYPE_ALIASES:
        raise ValueError(
            f"qpm_type={value!r} is unsupported (V-P1-08). "
            f"Supported: '{QPM_TYPE_DEFAULT}' only."
        )
    return _QPM_TYPE_ALIASES[key]


def normalize_polarization(value: Any | None, *, name: str = "polarization") -> str:
    """Normalize polarization; type-0 domain requires H (V-P1-08)."""
    if value is None:
        return POLARIZATION_DEFAULT
    key = str(value).strip().lower().replace(" ", "")
    if key not in _POLARIZATION_ALIASES:
        raise ValueError(
            f"{name}={value!r} is unsupported for ppKTP type-0 (V-P1-08). "
            f"Supported: '{POLARIZATION_DEFAULT}' (H→H+H)."
        )
    return _POLARIZATION_ALIASES[key]


def validate_qpm_duty_cycle(duty: float) -> float:
    """QPM duty cycle D with 0 < D ≤ 1 (V-P1-08)."""
    if isinstance(duty, bool) or not isinstance(duty, Real):
        raise ValueError(f"qpm_duty_cycle must be a real number, got {duty!r}")
    D = float(duty)
    if not np.isfinite(D) or not (0.0 < D <= 1.0):
        raise ValueError(
            f"qpm_duty_cycle must satisfy 0 < D <= 1 (V-P1-08), got {duty}"
        )
    return D


def validate_transmission(eta: float, *, name: str) -> float:
    """Intensity transmission / path factor in [0, 1] (V-P1-08)."""
    if isinstance(eta, bool) or not isinstance(eta, Real):
        raise ValueError(f"{name} must be a real number in [0, 1], got {eta!r}")
    e = float(eta)
    if not np.isfinite(e) or e < 0.0 or e > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1] (V-P1-08), got {eta}")
    return e


def validate_physics_domain(setup: "SetupParams") -> "SetupParams":
    """Fail-closed supported domain: ppKTP type-0 H→H+H (V-P1-08).

    Rejects unsupported material, QPM type, polarization, duty cycle, and
    out-of-range transmissions so synthetic datasets cannot silently emit
    physically undefined samples.
    """
    setup.crystal_material = normalize_crystal_material(
        getattr(setup, "crystal_material", None)
    )
    setup.qpm_type = normalize_qpm_type(getattr(setup, "qpm_type", None))
    setup.pol_pump = normalize_polarization(
        getattr(setup, "pol_pump", None), name="pol_pump"
    )
    setup.pol_signal = normalize_polarization(
        getattr(setup, "pol_signal", None), name="pol_signal"
    )
    setup.pol_idler = normalize_polarization(
        getattr(setup, "pol_idler", None), name="pol_idler"
    )
    setup.qpm_duty_cycle = validate_qpm_duty_cycle(
        getattr(setup, "qpm_duty_cycle", QPM_DUTY_CYCLE_DEFAULT)
    )
    return setup


def validate_crystal_parameter_sets(setup: "SetupParams") -> None:
    """Validate optional NL2 hardware inputs without changing NL1 defaults."""
    if setup.crystal_material_NL2 is not None:
        setup.crystal_material_NL2 = normalize_crystal_material(
            setup.crystal_material_NL2
        )
    if setup.qpm_type_NL2 is not None:
        setup.qpm_type_NL2 = normalize_qpm_type(setup.qpm_type_NL2)
    if setup.qpm_order_NL2 is not None:
        setup.qpm_order_NL2 = validate_qpm_order(
            setup.qpm_order_NL2, name="qpm_order_NL2"
        )
    if setup.qpm_duty_cycle_NL2 is not None:
        setup.qpm_duty_cycle_NL2 = validate_qpm_duty_cycle(
            setup.qpm_duty_cycle_NL2
        )
    for name in ("pol_pump_NL2", "pol_signal_NL2", "pol_idler_NL2"):
        value = getattr(setup, name)
        if value is not None:
            setattr(setup, name, normalize_polarization(value, name=name))
    for name in ("d33_NL2_pm_per_V", "d_eff_NL2_pm_per_V"):
        value = getattr(setup, name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number")
            setattr(setup, name, float(value))
    for name in ("crystal_orientation_NL1_deg", "crystal_orientation_NL2_deg"):
        setattr(
            setup,
            name,
            validate_crystal_orientation_deg(getattr(setup, name), name=name),
        )
    if setup.d_eff_NL2_pm_per_V is not None:
        setup.d_eff_NL2_pm_per_V = validate_direct_d_eff_pm_per_V(
            setup.d_eff_NL2_pm_per_V
        )
    precedence = normalize_d_eff_precedence(setup.d_eff_precedence_NL2)
    if setup.d_eff_precedence_NL2 is not None:
        setup.d_eff_precedence_NL2 = precedence
    derived_nl2 = None
    if setup.d33_NL2_pm_per_V is not None:
        derived_nl2 = derive_d_eff_pm_per_V(
            d33_pm_per_V=setup.d33_NL2_pm_per_V,
            qpm_order=setup.qpm_order_NL2 or setup.qpm_order,
            qpm_duty_cycle=(setup.qpm_duty_cycle_NL2
                            if setup.qpm_duty_cycle_NL2 is not None
                            else setup.qpm_duty_cycle),
        )
    if precedence == DEFF_PRECEDENCE_DIRECT and setup.d_eff_NL2_pm_per_V is None:
        raise ValueError("d_eff_precedence_NL2='direct' requires d_eff_NL2_pm_per_V")
    if precedence == DEFF_PRECEDENCE_DERIVED:
        if derived_nl2 is None:
            raise ValueError("d_eff_precedence_NL2='derived' requires d33_NL2_pm_per_V")
        setup.d_eff_NL2_pm_per_V = float(derived_nl2)
    elif (derived_nl2 is not None and setup.d_eff_NL2_pm_per_V is None):
        setup.d_eff_NL2_pm_per_V = float(derived_nl2)
    elif (precedence == DEFF_PRECEDENCE_AUTO and derived_nl2 is not None
          and setup.d_eff_NL2_pm_per_V is not None):
        if not _d_eff_values_consistent(setup.d_eff_NL2_pm_per_V, derived_nl2):
            raise ValueError("NL2 direct d_eff conflicts with d33-derived d_eff")
    if normalize_spdc_architecture(setup.spdc_architecture) == SPDC_ARCHITECTURE_SINGLE:
        shared = {
            "crystal_material_NL2": "crystal_material",
            "qpm_type_NL2": "qpm_type",
            "qpm_order_NL2": "qpm_order",
            "qpm_duty_cycle_NL2": "qpm_duty_cycle",
            "d33_NL2_pm_per_V": "d33_pm_per_V",
            "d_eff_NL2_pm_per_V": "d_eff_pm_per_V",
            "pol_pump_NL2": "pol_pump",
            "pol_signal_NL2": "pol_signal",
            "pol_idler_NL2": "pol_idler",
            "crystal_orientation_NL2_deg": "crystal_orientation_NL1_deg",
        }
        for second, first in shared.items():
            value = getattr(setup, second)
            if value is not None and value != getattr(setup, first):
                raise ValueError(
                    f"single-crystal passes must share {first}; {second} differs"
                )


def oven_settling_factor(t_s: float) -> float:
    """Fractional approach to oven setpoint: 1 − exp(−t/τ). τ is fixed."""
    tau = float(OVEN_TAU_S)
    if tau <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-max(float(t_s), 0.0) / tau))


def delta_L_thermal_mm(
    setup: "SetupParams",
    *,
    dT_C: float | None = None,
    t_s: float | None = None,
) -> float:
    """PDF §5.2: ΔL_thermal = α · L · ΔT  (signed air-path thermo-optic OPD).

    ``alpha_path_per_C`` defaults to dn_air/dT; ``L_eff_path_mm`` is the effective
    mismatched path. Optional ``t_s`` applies oven settling (1−e^{−t/τ}).
    Sign is preserved (V-P1-05); do not abs here.
    """
    if dT_C is None:
        dT_C = float(crystal_thermal_state(setup)["dT_eff_C"])
    dT = float(dT_C)
    if t_s is not None:
        dT = dT * oven_settling_factor(t_s)
    alpha = float(getattr(setup, "alpha_path_per_C", ALPHA_AIR_PATH_PER_C))
    L = float(getattr(setup, "L_eff_path_mm", L_EFF_PATH_MM_DEFAULT))
    return float(alpha * L * dT)


def optical_path_difference_mm(
    setup: "SetupParams",
    *,
    t_s: float | None = None,
) -> float:
    """Signed interferometer OPD [mm]: OPD = ΔL₀ + ΔL_thermal (V-P1-05).

    Thermal contribution is added only when ``use_physical_thermal``.
    Never take abs here — magnitude ``|OPD|`` is for coherence only.
    """
    dL0 = validate_delta_L_mm(setup.delta_L_mm)
    if not bool(getattr(setup, "use_physical_thermal", False)):
        return float(dL0)
    return float(dL0 + delta_L_thermal_mm(setup, t_s=t_s))


def effective_delta_L_mm(
    setup: "SetupParams",
    *,
    t_s: float | None = None,
) -> float:
    """Coherence path magnitude |OPD| [mm] (PDF §5.2 → V_coh) — V-P1-05.

    OPD = ΔL₀ + ΔL_thermal is kept signed for phase; only coherence uses
    ``abs(OPD)``, then clipped to the lab ΔL window.
    """
    opd = optical_path_difference_mm(setup, t_s=t_s)
    return float(np.clip(abs(opd), 0.0, DELTA_L_MM_MAX))


def opd_phase_rad(
    setup: "SetupParams",
    *,
    t_s: float | None = None,
) -> float:
    """Interferometer phase from signed OPD: Δφ = k · OPD, k = 2π/λ_s (V-P1-05).

    Uses full ``OPD = ΔL₀ + ΔL_thermal`` (not thermal alone). Does **not**
    bake in θ₀ — callers add ``THETA0_RAD`` once.
    """
    opd_mm = optical_path_difference_mm(setup, t_s=t_s)
    lam_mm = max(float(setup.lambda_s_nm), 1e-9) * 1e-6  # nm → mm
    return float(2.0 * np.pi * opd_mm / lam_mm)


def thermal_phase_offset_rad(
    setup: "SetupParams",
    *,
    t_s: float | None = None,
) -> float:
    """PDF §5.2/§5.3 phase from signed OPD (alias of ``opd_phase_rad``).

    Historically thermal-only; V-P1-05 uses full OPD = ΔL₀ + ΔL_thermal via
    ``k·OPD``. θ₀ is not included (add ``THETA0_RAD`` at the call site).
    """
    if not bool(getattr(setup, "use_physical_thermal", False)):
        # Static lab ΔL₀ still contributes phase when requested via this API
        # only if physical thermal is on; legacy engineered path uses walks.
        return 0.0
    return opd_phase_rad(setup, t_s=t_s)


def v_coh_gaussian(delta_L_mm: float, L_coh_mm: float) -> float:
    """Temporal coherence visibility (PDF §4.1 / §5.2).

    V_coh = exp(-(|ΔL| / L_c)²). Pass |OPD| (or any signed OPD — abs is taken).
    """
    Lc = validate_L_coh_mm(L_coh_mm)
    dL = abs(float(delta_L_mm))
    if not np.isfinite(dL):
        raise ValueError(f"delta_L_mm must be finite, got {delta_L_mm}")
    return float(np.exp(-((dL / Lc) ** 2)))


def validate_poling_period_um(Lambda_um: float) -> float:
    L = float(Lambda_um)
    if not (POLING_PERIOD_UM_MIN <= L <= POLING_PERIOD_UM_MAX):
        raise ValueError(
            f"poling_period_um={L} µm outside "
            f"[{POLING_PERIOD_UM_MIN}, {POLING_PERIOD_UM_MAX}] "
            f"(Lemos default {POLING_PERIOD_UM_DEFAULT})"
        )
    return L


def ktp_nz_sellmeier(lam_um: float) -> float:
    """KTP n_z at ``KTP_SELLMEIER_T_REF_C`` (type-0 / extraordinary).

    Kato & Takaoka, Appl. Opt. 41, 5040 (2002) [report ref. 17, eq. 42]:
        n_z²(λ) = 4.59423 + 0.06206/(λ² − 0.04763)
                          + 110.80672/(λ² − 86.12171)
    λ in µm. Thermo-optic shift is applied in ``ktp_nz``.
    """
    w2 = float(lam_um) ** 2
    nz2 = (
        4.59423
        + 0.06206 / (w2 - 0.04763)
        + 110.80672 / (w2 - 86.12171)
    )
    return float(np.sqrt(max(nz2, 1e-12)))


def ktp_dndT_z_per_C(lam_um: float) -> float:
    """dn_z/dT [1/°C] for KTP z-axis in the working band.

    Kato & Takaoka, Appl. Opt. 41, 5040 (2002) [report ref. 17, eq. 43]:
        dn_z/dT = (0.9221/λ³ − 2.9220/λ² + 3.6677/λ − 0.1897) × 10⁻⁵
    λ in µm.
    """
    lam = max(float(lam_um), 0.4)
    val = (
        0.9221 / (lam ** 3)
        - 2.9220 / (lam ** 2)
        + 3.6677 / lam
        - 0.1897
    ) * 1.0e-5
    return float(val)


def ktp_nz(lam_um: float, T_C: float) -> float:
    """n_z(λ, T) = n_z(λ, T_ref) + (dn_z/dT)(T − T_ref). Report eq. 44."""
    n0 = ktp_nz_sellmeier(lam_um)
    return float(n0 + ktp_dndT_z_per_C(lam_um) * (float(T_C) - KTP_SELLMEIER_T_REF_C))


def poling_period_um_at_T(Lambda0_um: float, T_C: float, T_ref_C: float) -> float:
    """Λ(T) = Λ0 (1 + α_KTP (T − T_ref))."""
    return float(Lambda0_um) * (
        1.0 + KTP_CTE_PER_C * (float(T_C) - float(T_ref_C))
    )


def delta_k_qpm_per_um(
    *,
    lambda_p_nm: float,
    lambda_s_nm: float,
    lambda_i_nm: float,
    T_C: float,
    Lambda_um: float,
    qpm_order: int = QPM_ORDER_DEFAULT,
    orientation_deg: float | None = None,
) -> float:
    """Δk_QPM = k_p − k_s − k_i − K_g  [rad/µm] (V-P1-03).

    With collinear wavevectors k = 2π n / λ (λ in µm)::

        Δk = 2π(n_p/λ_p − n_s/λ_s − n_i/λ_i) − 2π m cos(θ) / Λ

    where ``m = qpm_order`` is the QPM grating order and ``θ`` is the
    Type-0 collinear crystal orientation (poling-grating tilt vs pump axis).
    ``θ = None`` or ``0`` recovers the historical ``2π m / Λ`` term.

    Report §7.8: this mismatch is not an input to O_geom. It currently
    scales flux via ``r_spdc_sinc2``; gain insertion is report §7.9.
    """
    m = int(qpm_order)
    if m < 1:
        raise ValueError(f"qpm_order must be a positive integer, got {qpm_order}")
    Lam = float(Lambda_um)
    if not np.isfinite(Lam) or Lam <= 0.0:
        raise ValueError(f"Lambda_um must be finite and > 0, got {Lambda_um}")
    theta = validate_crystal_orientation_deg(
        orientation_deg, name="orientation_deg"
    )
    cos_th = 1.0 if theta is None else float(np.cos(np.deg2rad(theta)))
    lp = float(lambda_p_nm) * 1e-3
    ls = float(lambda_s_nm) * 1e-3
    li = float(lambda_i_nm) * 1e-3
    np_ = ktp_nz(lp, T_C)
    ns_ = ktp_nz(ls, T_C)
    ni_ = ktp_nz(li, T_C)
    # Grating projection K_g = 2π m cos(θ) / Λ  (Step-2 Type-0 collinear model)
    return float(
        2.0 * np.pi * (
            np_ / lp - ns_ / ls - ni_ / li - float(m) * cos_th / Lam
        )
    )


def delta_k_qpm_per_m_from_setup(
    setup: "SetupParams",
    *,
    T_C: float | None = None,
    interaction: int = 1,
) -> float:
    """Δk_QPM [rad/m] at collection wavelengths (report §7.9 / eq. 45).

    Same kinematics as ``r_spdc_sinc2``: fixed λ filters, Λ(T) from
    ``qpm_poling_period_um``, grating order ``qpm_order`` as 2π m / Λ
    (V-P1-03). Not used by O_geom (report §7.8).
    Single-crystal Physical Mode uses shared T and Λ. two_crystal uses
    that interaction's T and Λ.
    """
    if T_C is None:
        cr = spdc_crystal_state(setup, interaction=interaction)
        T = float(cr["T_crystal_C"])
        Lam = float(cr["poling_period_um_qpm"])
    else:
        T = float(T_C)
        Lam = qpm_poling_period_um(setup, T_C=T, interaction=interaction)
    dk_per_um = delta_k_qpm_per_um(
        lambda_p_nm=float(setup.lambda_p_nm),
        lambda_s_nm=float(setup.lambda_s_nm),
        lambda_i_nm=float(setup.lambda_i_nm),
        T_C=T,
        Lambda_um=Lam,
        qpm_order=(setup.qpm_order_NL2
                   if is_two_crystal_architecture(setup) and int(interaction) == 2
                   and setup.qpm_order_NL2 is not None else setup.qpm_order),
        orientation_deg=resolved_crystal_orientation_deg(
            setup, interaction=int(interaction)
        ),
    )
    # Propagation direction (forward/backward) is distinct from physical
    # crystal orientation; both contribute, neither is an empirical V penalty.
    direction = spdc_propagation_direction(setup, interaction=interaction)
    sign = -1.0 if direction == "backward" else 1.0
    return float(sign * dk_per_um * 1.0e6)


def crystal_length_m_from_setup(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float:
    """Crystal length [m] for one SPDC interaction.

    Single-crystal Physical Mode: shared ``crystal_length_mm`` (§7.11).
    two_crystal / legacy: NL2 may use ``crystal_length_NL2_mm``.
    """
    if uses_shared_crystal(setup):
        return float(spdc_shared_crystal_state(setup)["crystal_length_m"])
    L_mm = float(setup.crystal_length_mm)
    if int(interaction) == 2 and setup.crystal_length_NL2_mm is not None:
        L_mm = float(setup.crystal_length_NL2_mm)
    return float(validate_crystal_length_mm(L_mm) * 1.0e-3)


def d_eff_pm_per_V(
    *,
    d33_pm_per_V: float,
    qpm_order: int = QPM_ORDER_DEFAULT,
    qpm_duty_cycle: float = QPM_DUTY_CYCLE_DEFAULT,
) -> float:
    """Effective quadratic coefficient for square-wave QPM.

    Report eq. 40, Fejer et al. / Hum & Fejer [15, 16]:
        d_eff = d33 · 2 |sin(m π D)| / (m π)
    First-order, D = 0.5 (report eq. 41): d_eff = (2/π) d33.
    """
    m = int(qpm_order)
    if m < 1:
        raise ValueError(f"qpm_order must be a positive integer, got {qpm_order}")
    D = validate_qpm_duty_cycle(qpm_duty_cycle)
    d33 = float(d33_pm_per_V)
    if not np.isfinite(d33):
        raise ValueError(f"d33_pm_per_V must be finite, got {d33_pm_per_V}")
    deff = float(d33 * 2.0 * abs(np.sin(m * np.pi * D)) / (m * np.pi))
    # V-P1-02: never silently zero a non-finite d_eff.
    if not np.isfinite(deff):
        raise ValueError(
            f"d_eff_pm_per_V overflow/non-finite from d33={d33}, m={m}, D={D}"
        )
    return deff


def validate_direct_d_eff_pm_per_V(value: float) -> float:
    """Accept an explicitly supplied effective nonlinearity [pm/V]."""
    deff = float(value)
    if not np.isfinite(deff):
        raise ValueError(f"d_eff_pm_per_V must be finite, got {value}")
    return deff


def derive_d_eff_pm_per_V(
    *,
    d33_pm_per_V: float,
    qpm_order: int = QPM_ORDER_DEFAULT,
    qpm_duty_cycle: float = QPM_DUTY_CYCLE_DEFAULT,
) -> float:
    """Alias for ``d_eff_pm_per_V`` (Comment 3.2 derive helper name)."""
    return d_eff_pm_per_V(
        d33_pm_per_V=d33_pm_per_V,
        qpm_order=qpm_order,
        qpm_duty_cycle=qpm_duty_cycle,
    )


def normalize_d_eff_precedence(value: Any | None = None) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFF_PRECEDENCE_DEFAULT
    key = str(value).strip().lower().replace("-", "_")
    if key not in _DEFF_PRECEDENCE_ALIASES:
        raise ValueError(
            "d_eff_precedence must be one of "
            f"{sorted(set(_DEFF_PRECEDENCE_ALIASES.values()))}, got {value!r}"
        )
    return _DEFF_PRECEDENCE_ALIASES[key]


def _d_eff_values_consistent(a: float, b: float) -> bool:
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=DEFF_CONSISTENCY_RTOL,
            atol=DEFF_CONSISTENCY_ATOL,
        )
    )


def _normalize_stored_d_eff_source(value: Any | None) -> str:
    """Map legacy resolution labels onto Comment-3.3 ``d_eff_source`` values."""
    if value is None:
        return DEFF_SOURCE_NONE
    key = str(value).strip().lower().replace("-", "_")
    if key in {DEFF_SOURCE_NONE, "unresolved", ""}:
        return DEFF_SOURCE_NONE
    if key in {DEFF_SOURCE_DIRECT, DEFF_RESOLUTION_DIRECT}:
        return DEFF_SOURCE_DIRECT
    if key in {
        DEFF_SOURCE_DERIVED,
        DEFF_RESOLUTION_DERIVED,
        "derived_from_d33",
        "from_d33",
    }:
        return DEFF_SOURCE_DERIVED
    return DEFF_SOURCE_NONE


def effective_nonlinearity_metadata(setup: "SetupParams") -> dict[str, Any]:
    """Comment 3.3 audit block: requested vs resolved nonlinearity."""
    source = _normalize_stored_d_eff_source(
        getattr(setup, "d_eff_source", None)
        or getattr(setup, "d_eff_resolution", None)
    )
    d_eff_req = getattr(setup, "d_eff_requested_pm_per_V", None)
    if d_eff_req is None:
        d_eff_req = getattr(setup, "d_eff_pm_per_V_requested", None)
    d33_req = getattr(setup, "d33_requested_pm_per_V", None)
    if d33_req is None:
        d33_req = getattr(setup, "d33_pm_per_V_requested", None)
    resolved = getattr(setup, "d_eff_pm_per_V", None)
    meta: dict[str, Any] = {
        "d_eff_source": source,
        "d_eff_resolved_pm_per_V": None if resolved is None else float(resolved),
        "d_eff_requested_pm_per_V": None if d_eff_req is None else float(d_eff_req),
        "d33_requested_pm_per_V": None if d33_req is None else float(d33_req),
        # Legacy aliases (same values).
        "d_eff_resolution": (
            DEFF_RESOLUTION_DIRECT
            if source == DEFF_SOURCE_DIRECT
            else DEFF_RESOLUTION_DERIVED
            if source == DEFF_SOURCE_DERIVED
            else DEFF_RESOLUTION_NONE
        ),
        "d_eff_pm_per_V_requested": None if d_eff_req is None else float(d_eff_req),
        "d33_pm_per_V_requested": None if d33_req is None else float(d33_req),
        "d_eff_pm_per_V": None if resolved is None else float(resolved),
    }
    return meta


def resolve_effective_nonlinearity(setup: "SetupParams") -> dict[str, Any]:
    """Resolve ``d_eff_pm_per_V`` with Comment 3.2 cases A–D + 3.3 audit fields.

    Auto precedence (``d_eff_precedence='auto'``):
      A  d_eff set, d33 absent  → preserve d_eff (``d_eff_source='direct'``)
      B  d_eff absent, d33 set  → derive (``d_eff_source='derived_from_d33'``)
      C  both absent            → unresolved / None
      D  both set               → require consistency; keep direct d_eff;
                                  reject unless ``d_eff_precedence`` overrides

    Never erases a legitimate direct ``d_eff`` merely because ``d33`` is absent.
    User inputs are recorded under requested_* before any resolve overwrite.
    """
    precedence = normalize_d_eff_precedence(
        getattr(setup, "d_eff_precedence", DEFF_PRECEDENCE_DEFAULT)
    )
    d33_raw = getattr(setup, "d33_pm_per_V", None)
    deff_raw = getattr(setup, "d_eff_pm_per_V", None)
    prev_source = _normalize_stored_d_eff_source(
        getattr(setup, "d_eff_source", None)
        or getattr(setup, "d_eff_resolution", None)
    )
    prev_deff_requested = getattr(setup, "d_eff_requested_pm_per_V", None)
    if prev_deff_requested is None:
        prev_deff_requested = getattr(setup, "d_eff_pm_per_V_requested", None)

    d33_req: float | None
    if d33_raw is None:
        d33_req = None
    else:
        d33_req = float(d33_raw)
        if not np.isfinite(d33_req):
            raise ValueError(f"d33_pm_per_V must be finite, got {d33_raw}")

    # Re-entrant validate after a prior derive: filled d_eff is not user input.
    deff_req: float | None
    if (
        deff_raw is not None
        and prev_source == DEFF_SOURCE_DERIVED
        and prev_deff_requested is None
        and precedence != DEFF_PRECEDENCE_DIRECT
    ):
        deff_req = None
    elif deff_raw is None:
        deff_req = None
    else:
        deff_req = validate_direct_d_eff_pm_per_V(deff_raw)

    qpm_order = int(getattr(setup, "qpm_order", QPM_ORDER_DEFAULT))
    qpm_duty = float(getattr(setup, "qpm_duty_cycle", QPM_DUTY_CYCLE_DEFAULT))
    derived: float | None = None
    if d33_req is not None:
        derived = derive_d_eff_pm_per_V(
            d33_pm_per_V=d33_req,
            qpm_order=qpm_order,
            qpm_duty_cycle=qpm_duty,
        )

    if precedence == DEFF_PRECEDENCE_DIRECT:
        if deff_req is None:
            raise ConfigError(
                "d_eff_precedence='direct' requires a finite d_eff_pm_per_V"
            )
        resolved = deff_req
        source = DEFF_SOURCE_DIRECT
    elif precedence == DEFF_PRECEDENCE_DERIVED:
        if d33_req is None or derived is None:
            raise ConfigError(
                "d_eff_precedence='derived' requires a finite d33_pm_per_V"
            )
        resolved = float(derived)
        source = DEFF_SOURCE_DERIVED
    else:
        # Comment 3.2 Cases A–D
        if deff_req is not None and d33_req is None:
            # Case A
            resolved = deff_req
            source = DEFF_SOURCE_DIRECT
        elif deff_req is None and d33_req is not None:
            # Case B
            resolved = float(derived) if derived is not None else None
            source = DEFF_SOURCE_DERIVED
        elif deff_req is None and d33_req is None:
            # Case C
            resolved = None
            source = DEFF_SOURCE_NONE
        else:
            # Case D — both supplied: verify consistency; keep direct d_eff
            assert deff_req is not None and derived is not None
            if not _d_eff_values_consistent(deff_req, derived):
                raise ConfigError(
                    "Direct d_eff_pm_per_V conflicts with d33-derived value: "
                    f"d_eff_pm_per_V={deff_req} vs derived={derived} "
                    f"from d33_pm_per_V={d33_req} "
                    f"(qpm_order={qpm_order}, qpm_duty_cycle={qpm_duty}). "
                    "Set d_eff_precedence to 'direct' or 'derived'."
                )
            resolved = deff_req
            source = DEFF_SOURCE_DIRECT

    # Record requested inputs BEFORE publishing resolved working value (3.3).
    setup.d_eff_requested_pm_per_V = deff_req
    setup.d33_requested_pm_per_V = d33_req
    setup.d_eff_pm_per_V_requested = deff_req  # legacy alias
    setup.d33_pm_per_V_requested = d33_req  # legacy alias

    setup.d33_pm_per_V = d33_req
    setup.d_eff_pm_per_V = None if resolved is None else float(resolved)
    setup.d_eff_resolved_pm_per_V = setup.d_eff_pm_per_V
    setup.d_eff_source = source
    setup.d_eff_precedence = precedence
    setup.d_eff_resolution = (
        DEFF_RESOLUTION_DIRECT
        if source == DEFF_SOURCE_DIRECT
        else DEFF_RESOLUTION_DERIVED
        if source == DEFF_SOURCE_DERIVED
        else DEFF_RESOLUTION_NONE
    )

    meta = effective_nonlinearity_metadata(setup)
    meta.update(
        {
            "d_eff_precedence": precedence,
            "d_eff_pm_per_V_derived": derived,
            "qpm_order": qpm_order,
            "qpm_duty_cycle": qpm_duty,
        }
    )
    return meta


def angular_frequency_rad_per_s(lambda_nm: float) -> float:
    """ω = 2π c / λ."""
    lam_m = float(lambda_nm) * 1.0e-9
    if lam_m <= 0.0:
        raise ValueError(f"wavelength must be > 0 nm, got {lambda_nm}")
    return float(2.0 * np.pi * SPEED_OF_LIGHT_M_PER_S / lam_m)


def k_spdc_0_per_sqrt_W(
    *,
    d_eff_m_per_V: float,
    crystal_length_m: float,
    pump_waist_m: float,
    n_pump: float,
    n_signal: float,
    n_idler: float,
    omega_s_rad_per_s: float,
    omega_i_rad_per_s: float,
) -> float:
    """Phase-matched base gain coefficient k_SPDC,0  [W^{-1/2}].

    Report eq. 24 / eq. 29, undepleted Gaussian pump, ideal phase matching:
        k_SPDC,0 = (2 d_eff L / w_p)
                   · √(ω_s ω_i / (π n_p n_s n_i ε0 c³))
    so that r = k_SPDC,0 √P_W.
    """
    deff = float(d_eff_m_per_V)
    L = float(crystal_length_m)
    wp = float(pump_waist_m)
    np_ = float(n_pump)
    ns_ = float(n_signal)
    ni_ = float(n_idler)
    ws = float(omega_s_rad_per_s)
    wi = float(omega_i_rad_per_s)
    if wp <= 0.0:
        raise ValueError(f"pump waist must be > 0 m, got {wp}")
    denom = (
        np.pi * np_ * ns_ * ni_ * VACUUM_PERMITTIVITY_F_PER_M
        * (SPEED_OF_LIGHT_M_PER_S ** 3)
    )
    if denom <= 0.0:
        raise ValueError("n_p n_s n_i ε0 c³ must be > 0")
    k0 = float((2.0 * deff * L / wp) * np.sqrt(ws * wi / denom))
    # V-P1-02: keep d_eff-driven k; overflow must error, not become 0.
    if not np.isfinite(k0) or k0 < 0.0:
        raise ValueError(
            f"k_SPDC,0 overflow/non-finite (d_eff={deff}, L={L}, w_p={wp})"
        )
    return k0


def pump_power_W_from_mW(power_mW: float) -> float:
    """Report eq. 30: P_W = 10^{-3} P_mW. The single milliwatt→watt site."""
    return MILLIWATT_TO_WATT * float(power_mW)


def validate_pump_transmission(eta: float, *, name: str) -> float:
    """Intensity transmission in [0, 1] (report §7.12 ηp,pre / ηp,return)."""
    return validate_transmission(eta, name=name)


def validate_pump_bs_reflectance(value: Any | None) -> float:
    """Passive BS intensity reflectance to NL1, in [0, 1]."""
    if value is None:
        return float(PUMP_BS_REFLECTANCE_DEFAULT)
    r = float(value)
    if not np.isfinite(r) or r < 0.0 or r > 1.0:
        raise ValueError(
            f"pump_bs_reflectance must be finite and in [0, 1], got {value!r}"
        )
    return r


def resolved_pump_bs_reflectance(setup: "SetupParams") -> float:
    return validate_pump_bs_reflectance(
        getattr(setup, "pump_bs_reflectance", None)
    )


def _is_branched_two_crystal(setup: "SetupParams") -> bool:
    return (
        is_two_crystal_architecture(setup)
        and normalize_two_crystal_pump_topology(
            getattr(setup, "two_crystal_pump_topology", None)
        )
        == TWO_CRYSTAL_PUMP_BRANCHED
    )


def resolved_pump_laser_mW(
    setup: "SetupParams",
    pump_laser_mW: float | None = None,
) -> float | None:
    """P_p,laser [mW]. Explicit value wins, else setup.pump_power_laser_mW."""
    if pump_laser_mW is not None:
        p = float(pump_laser_mW)
    else:
        raw = getattr(setup, "pump_power_laser_mW", None)
        if raw is None:
            return None
        p = float(raw)
    if not np.isfinite(p) or p < 0.0:
        raise ValueError(f"pump_power_laser_mW must be finite and >= 0, got {p}")
    return p


def pump_fluctuation_scale(relative_power_fluctuation: float = 0.0) -> float:
    """Instantaneous power scale 1+dP, clipped so P >= 0 (report §7.14)."""
    dP = float(relative_power_fluctuation)
    if not np.isfinite(dP):
        raise ValueError(
            f"relative_power_fluctuation must be finite, got {relative_power_fluctuation}"
        )
    return float(max(0.0, 1.0 + dP))


def pump_power_NL1_mW_from_setup(
    setup: "SetupParams",
    pump_laser_mW: float | None = None,
    *,
    relative_power_fluctuation: float = 0.0,
) -> float | None:
    """P_p1 at NL1 [mW].

    Single-crystal or sequential two-crystal: the full laser reaches the first
    crystal, P1 = η_pre P_laser (1+dP)_+.
    Branched two-crystal (passive BS): P1 = R_BS η_pre P_laser (1+dP)_+.
    """
    laser = resolved_pump_laser_mW(setup, pump_laser_mW)
    if laser is None:
        return None
    eta = validate_pump_transmission(
        setup.pump_pre_crystal_transmission, name="pump_pre_crystal_transmission"
    )
    scale = pump_fluctuation_scale(relative_power_fluctuation)
    if _is_branched_two_crystal(setup):
        return float(
            resolved_pump_bs_reflectance(setup) * eta * laser * scale
        )
    return float(eta * laser * scale)


def pump_power_NL2_mW_from_setup(
    setup: "SetupParams",
    pump_laser_mW: float | None = None,
    *,
    relative_power_fluctuation: float = 0.0,
) -> float | None:
    """P_p2 for the second interaction.

    Single-crystal double-pass (report eq. 53 / §7.14):
        P_p2 = ηp,return P_p1(t)
    Branched two-crystal (passive BS, L03):
        P_p2 = (1-R_BS) η_NL2 P_p,laser (1+dP)_+
    Sequential two-crystal:
        P_p2 = η_between P_p1
    Experimental P_p2 override keeps the same relative fluctuation.
    """
    scale = pump_fluctuation_scale(relative_power_fluctuation)
    if bool(getattr(setup, "use_independent_pump_power_NL2", False)):
        override = getattr(setup, "pump_power_NL2_override_mW", None)
        if override is not None:
            p2 = float(override)
            if not np.isfinite(p2) or p2 < 0.0:
                raise ValueError(
                    "pump_power_NL2_override_mW must be finite and >= 0"
                )
            return float(p2 * scale)
    if is_two_crystal_architecture(setup):
        topology = normalize_two_crystal_pump_topology(
            getattr(setup, "two_crystal_pump_topology", None)
        )
        if topology is None:
            raise ValueError(
                "two-crystal local pump power requires explicit "
                "two_crystal_pump_topology"
            )
        if topology == TWO_CRYSTAL_PUMP_BRANCHED:
            laser = resolved_pump_laser_mW(setup, pump_laser_mW)
            if laser is None:
                return None
            eta = validate_pump_transmission(
                getattr(setup, "pump_to_NL2_transmission", 1.0),
                name="pump_to_NL2_transmission",
            )
            transmit = 1.0 - resolved_pump_bs_reflectance(setup)
            return float(transmit * eta * laser * scale)
        p1 = pump_power_NL1_mW_from_setup(
            setup, pump_laser_mW,
            relative_power_fluctuation=relative_power_fluctuation,
        )
        if p1 is None:
            # fixed_nbar does not use a pump-power prediction.  Keep both local
            # powers unavailable instead of requiring an unrelated calibration.
            return None
        eta_between = getattr(setup, "pump_between_crystals_transmission", None)
        if eta_between is None:
            raise ValueError(
                "sequential two-crystal pump topology requires measured "
                "pump_between_crystals_transmission"
            )
        return float(validate_pump_transmission(
            eta_between, name="pump_between_crystals_transmission"
        ) * p1)
    p1 = pump_power_NL1_mW_from_setup(
        setup,
        pump_laser_mW,
        relative_power_fluctuation=relative_power_fluctuation,
    )
    if p1 is None:
        return None
    eta = validate_pump_transmission(
        setup.pump_return_transmission, name="pump_return_transmission"
    )
    return float(eta * p1)


def pump_power_on_pass_mW(
    setup: "SetupParams",
    *,
    interaction: int = 1,
    pump_laser_mW: float | None = None,
    relative_power_fluctuation: float = 0.0,
) -> float | None:
    """Crystal pump power [mW] for pass 1 or 2 (report §7.12 / §7.14)."""
    if int(interaction) == 2:
        return pump_power_NL2_mW_from_setup(
            setup,
            pump_laser_mW,
            relative_power_fluctuation=relative_power_fluctuation,
        )
    return pump_power_NL1_mW_from_setup(
        setup,
        pump_laser_mW,
        relative_power_fluctuation=relative_power_fluctuation,
    )


def pump_powers_from_setup(
    setup: "SetupParams",
    pump_laser_mW: float | None = None,
    *,
    relative_power_fluctuation: float = 0.0,
) -> dict[str, Any]:
    """Laser, transmissions, and P_p1(t), P_p2(t) (report §7.12 / §7.14)."""
    dP = float(relative_power_fluctuation)
    scale = pump_fluctuation_scale(dP)
    laser = resolved_pump_laser_mW(setup, pump_laser_mW)
    laser_t = None if laser is None else float(laser * scale)
    p1 = pump_power_NL1_mW_from_setup(
        setup,
        pump_laser_mW,
        relative_power_fluctuation=dP,
    )
    p2 = pump_power_NL2_mW_from_setup(
        setup,
        pump_laser_mW,
        relative_power_fluctuation=dP,
    )
    delta_p = None
    if p1 is not None and p2 is not None:
        denominator = 0.5 * (float(p1) + float(p2))
        delta_p = 0.0 if denominator <= 0.0 else float((p2 - p1) / denominator)
    target1 = getattr(setup, "pump_target_power_NL1_mW", None)
    target2 = getattr(setup, "pump_target_power_NL2_mW", None)
    branched = _is_branched_two_crystal(setup)
    r_bs = resolved_pump_bs_reflectance(setup) if branched else None
    branch_sum = (
        float(p1) + float(p2)
        if branched and p1 is not None and p2 is not None
        else None
    )
    sum_over_laser = (
        None if branch_sum is None or laser_t is None or float(laser_t) <= 0.0
        else branch_sum / float(laser_t)
    )
    return {
        "pump_power_laser_mW": laser_t,
        "pump_power_laser_mean_mW": laser,
        "pump_relative_power_fluctuation": dP,
        "pump_fluctuation_scale": scale,
        "pump_bs_reflectance": r_bs,
        "pump_pre_crystal_transmission": float(
            setup.pump_pre_crystal_transmission
        ),
        "pump_return_transmission": float(setup.pump_return_transmission),
        "pump_to_NL2_transmission": float(
            getattr(setup, "pump_to_NL2_transmission", 1.0)
        ),
        "two_crystal_pump_topology": normalize_two_crystal_pump_topology(
            getattr(setup, "two_crystal_pump_topology", None)
        ),
        "pump_between_crystals_transmission": getattr(
            setup, "pump_between_crystals_transmission", None
        ),
        "pump_power_NL1_mW": p1,
        "pump_power_NL2_mW": p2,
        "P_target_1_mW": target1,
        "P_target_2_mW": target2,
        "P_actual_1_mW": p1,
        "P_actual_2_mW": p2,
        "pump_branch_sum_mW": branch_sum,
        "pump_branch_sum_over_laser": sum_over_laser,
        "passive_bs_split": branched,
        "delta_P": delta_p,
        "abs_delta_P": None if delta_p is None else abs(delta_p),
        "pump_balance_mode": getattr(setup, "pump_balance_mode", None),
        "pump_imbalance_role": "diagnostic_only_gain_and_moments_own_physics",
        "hidden_equalization_applied": False,
        "use_independent_pump_power_NL2": bool(
            getattr(setup, "use_independent_pump_power_NL2", False)
        ),
        "eq52": (
            "Pp1 = R_BS * eta_p,pre * Pp,laser * (1+dP)_+"
            if branched
            else "Pp1 = eta_p,pre * Pp,laser * (1+dP)_+"
        ),
        "eq53": (
            ("Pp2 = (1-R_BS) * eta_NL2 * Pp,laser * (1+dP)_+"
             if setup.two_crystal_pump_topology == TWO_CRYSTAL_PUMP_BRANCHED
             else "Pp2 = eta_between * Pp1")
            if is_two_crystal_architecture(setup)
            else "Pp2 = eta_p,return * Pp1"
        ),
    }


def k_spdc_per_sqrt_mW_from_per_sqrt_W(k_per_sqrt_W: float) -> float:
    """Report eq. 30: k_√mW = √10^{-3} k_√W, from the same 10^{-3} as P."""
    return float(np.sqrt(MILLIWATT_TO_WATT) * float(k_per_sqrt_W))


def k_spdc_per_sqrt_W_from_per_sqrt_mW(k_per_sqrt_mW: float) -> float:
    """Inverse of eq. 30, so a manual k_√mW can enter the SI r = k √P_W path."""
    return float(k_per_sqrt_mW) / np.sqrt(MILLIWATT_TO_WATT)


def resolved_k_spdc_per_sqrt_W(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float | None:
    """k_SPDC in W^{-1/2} for one pass: k_SPDC,0 · O_geom, else manual k.

    Report eq. 25 / 36 / §7.13. Manual fallback is not multiplied by O_geom
    and is the same for both passes. Δk_QPM is not folded into k.
    """
    k0 = k_spdc_0_from_setup(setup, interaction=interaction)
    if k0 is not None:
        O = spdc_geometry_overlap_from_setup(setup, interaction=interaction)
        return float(k0) * float(O)
    k_mW = (
        getattr(setup, "spdc_gain_per_sqrt_mW_NL2", None)
        if int(interaction) == 2 else None
    )
    if k_mW is None:
        k_mW = getattr(setup, "spdc_gain_per_sqrt_mW", None)
    if k_mW is None:
        return None
    return k_spdc_per_sqrt_W_from_per_sqrt_mW(k_mW)


def spdc_gain_r_from_pump_mW(
    setup: "SetupParams",
    pump_mW: float,
    *,
    interaction: int = 1,
) -> float:
    """r = k_SPDC √P_W for one pass. Phase-matched r0 (report eq. 23, 46).

    Occupancy nbar at nonzero Δk_QPM is ``optics.spdc_nbar_from_setup_pump``
    (report §7.9 / eq. 47–51), not sinh²(r0).
    """
    pump = float(pump_mW)
    if not np.isfinite(pump) or pump < 0.0:
        raise ValueError("pump_power_mW must be finite and >= 0")
    k_W = resolved_k_spdc_per_sqrt_W(setup, interaction=interaction)
    if k_W is None:
        raise ValueError(
            "pump_power_mW requires calibrated setup.spdc_gain_per_sqrt_mW"
        )
    return float(k_W * np.sqrt(pump_power_W_from_mW(pump)))


def validate_calibrated_source_requirements(
    setup: "SetupParams", *, require_coupling: bool = True
) -> None:
    """Fail closed when predictive brightness lacks traceable calibration."""
    mode = normalize_spdc_source_mode(getattr(setup, "spdc_source_mode", None))
    for name in (
        "pump_power_NL2_override_calibration_id",
        "spdc_gain_calibration_id",
        "spdc_gain_calibration_id_NL2",
    ):
        value = getattr(setup, name, None)
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"{name} must be a non-empty string when set")
    if mode != SPDC_SOURCE_CALIBRATED:
        return
    if not str(getattr(setup, "spdc_gain_calibration_id", None) or "").strip():
        raise ValueError(
            "spdc_source_mode='calibrated_source' requires "
            "spdc_gain_calibration_id"
        )
    two_crystal = is_two_crystal_architecture(setup)
    if two_crystal:
        if not str(
            getattr(setup, "spdc_gain_calibration_id_NL2", None) or ""
        ).strip():
            raise ValueError(
                "two-crystal calibrated_source requires independent "
                "spdc_gain_calibration_id_NL2"
            )
        topology = normalize_two_crystal_pump_topology(
            getattr(setup, "two_crystal_pump_topology", None)
        )
        if topology is None:
            raise ValueError(
                "two-crystal calibrated_source requires explicit "
                "two_crystal_pump_topology"
            )
        if (topology == TWO_CRYSTAL_PUMP_SEQUENTIAL
                and getattr(setup, "pump_between_crystals_transmission", None) is None):
            raise ValueError(
                "sequential two-crystal calibrated_source requires measured "
                "pump_between_crystals_transmission"
            )
    if not require_coupling:
        return
    for interaction in (1, 2):
        if resolved_k_spdc_per_sqrt_W(setup, interaction=interaction) is None:
            raise ValueError(
                "spdc_source_mode='calibrated_source' requires a calibrated "
                f"SPDC coupling for interaction {interaction}: supply physical "
                "d_eff/waist inputs or spdc_gain_per_sqrt_mW"
            )


def k_spdc_0_from_setup(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float | None:
    """k_SPDC,0 [W^{-1/2}] from setup (deff, L, w_p, n, ω). Report §7.5.

    Returns None when deff or pump waist is missing so the manual
    ``spdc_gain_per_sqrt_mW`` path remains the explicit fallback.
    Does not include O_geom (applied in ``resolved_k_spdc_per_sqrt_W``).
    Δk_QPM enters nbar in optics.py (report §7.9), not this coefficient.
    Single-crystal Physical Mode shares L, n, deff (§7.11); two_crystal
    uses that crystal's L and n(T). Pump waist stays per-interaction.
    """
    deff_pm = getattr(setup, "d_eff_pm_per_V", None)
    wp_um = resolved_pump_waist_um(setup, interaction=interaction)
    if wp_um is None:
        return None
    L_m = crystal_length_m_from_setup(setup, interaction=interaction)
    if bool(getattr(setup, "use_physical_spdc_gain", False)):
        cr = spdc_crystal_state(setup, interaction=interaction)
        deff_pm = cr["d_eff_pm_per_V"]
        n_pump = float(cr["n_pump"])
        n_signal = float(cr["n_signal"])
        n_idler = float(cr["n_idler"])
    else:
        T = float(setup.T_crystal_NL1_C)
        n_pump = ktp_nz(float(setup.lambda_p_nm) * 1.0e-3, T)
        n_signal = ktp_nz(float(setup.lambda_s_nm) * 1.0e-3, T)
        n_idler = ktp_nz(float(setup.lambda_i_nm) * 1.0e-3, T)
    if deff_pm is None:
        return None
    return k_spdc_0_per_sqrt_W(
        d_eff_m_per_V=float(deff_pm) * 1.0e-12,
        crystal_length_m=L_m,
        pump_waist_m=float(wp_um) * 1.0e-6,
        n_pump=n_pump,
        n_signal=n_signal,
        n_idler=n_idler,
        omega_s_rad_per_s=angular_frequency_rad_per_s(setup.lambda_s_nm),
        omega_i_rad_per_s=angular_frequency_rad_per_s(setup.lambda_i_nm),
    )


def resolved_pump_waist_um(setup: "SetupParams", *, interaction: int = 1) -> float | None:
    """Pump waist [µm] for pass 1 or 2 (report §7.7 / §8). Not n_crystals."""
    if int(interaction) == 2:
        w2 = getattr(setup, "pump_waist_NL2_um", None)
        if w2 is not None:
            return float(w2)
    w1 = getattr(setup, "pump_waist_NL1_um", None)
    if w1 is not None:
        return float(w1)
    w = getattr(setup, "pump_waist_um", None)
    return None if w is None else float(w)


def resolved_pump_focus_z_um(setup: "SetupParams", *, interaction: int = 1) -> float:
    """Pump waist location [µm] vs crystal centre. Default 0."""
    if int(interaction) == 2:
        z2 = getattr(setup, "pump_focus_z_NL2_um", None)
        if z2 is not None:
            return float(z2)
    z1 = getattr(setup, "pump_focus_z_NL1_um", None)
    return 0.0 if z1 is None else float(z1)


def resolved_generation_idler_waist_um(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float | None:
    """Idler waist [µm] in the generation overlap of one pass (report eq. 39).

    NL2 uses ``idler_waist_2_um`` when set; otherwise both passes use
    ``idler_waist_um``, then the pump waist of that pass. No free focusing
    factor.
    """
    if int(interaction) == 2 and setup.idler_waist_2_um is not None:
        return float(setup.idler_waist_2_um)
    if setup.idler_waist_um is not None:
        return float(setup.idler_waist_um)
    return resolved_pump_waist_um(setup, interaction=interaction)


def rayleigh_range_m(*, n: float, waist_m: float, lambda_m: float) -> float:
    """z_R = π n w_0² / λ (report eq. 32)."""
    w = float(waist_m)
    lam = float(lambda_m)
    if w <= 0.0 or lam <= 0.0:
        raise ValueError("waist and wavelength must be > 0")
    return float(np.pi * float(n) * w * w / lam)


def focus_parameter_xi(*, crystal_length_m: float, rayleigh_m: float) -> float:
    """ξ = L / (2 z_R) (report eq. 33)."""
    zR = float(rayleigh_m)
    if zR <= 0.0:
        raise ValueError("Rayleigh range must be > 0")
    return float(crystal_length_m) / (2.0 * zR)


def _gaussian_alpha_C(
    z_m: float,
    *,
    w0_m: float,
    z0_m: float,
    zR_m: float,
    k_med: float,
) -> tuple[complex, complex]:
    """Transverse Gaussian α and amplitude C at z.

    ``k_med`` is the mode wavevector 2πn/λ for Gouy phase and curvature
    (report §5.4.1). It is not Δk_QPM; do not multiply by e^{i Δk_QPM z}
    (report §7.8).
    """
    dz = float(z_m) - float(z0_m)
    zR = float(zR_m)
    w = float(w0_m) * np.sqrt(1.0 + (dz / zR) ** 2)
    if abs(dz) <= 1.0e-18 * max(abs(zR), 1.0):
        alpha: complex = 1.0 / (w * w)
    else:
        R = dz * (1.0 + (zR / dz) ** 2)
        alpha = 1.0 / (w * w) + 1j * float(k_med) / (2.0 * R)
    C = (np.sqrt(2.0 / np.pi) / w) * np.exp(1j * np.arctan2(dz, zR))
    return alpha, C


def transverse_generation_overlap_A(
    z_m: float,
    *,
    w0p_m: float,
    w0s_m: float,
    w0i_m: float,
    z0p_m: float,
    z0s_m: float,
    z0i_m: float,
    zR_p_m: float,
    zR_s_m: float,
    zR_i_m: float,
    k_p: float,
    k_s: float,
    k_i: float,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    dtheta_rad: float = 0.0,
) -> complex:
    """A(z) = ∫∫ u_p u_s* u_i* dx dy (report eq. 31).

    Includes mode size, displacement, tilt, waist location, and geometric
    field phases. The independent QPM factor e^{i Δk_QPM z} is excluded
    here (report §5.4.1 / §7.8) so it is not applied twice when §7.9
    inserts Δk_QPM in the gain.
    """
    ap, Cp = _gaussian_alpha_C(z_m, w0_m=w0p_m, z0_m=z0p_m, zR_m=zR_p_m, k_med=k_p)
    a_s, Cs = _gaussian_alpha_C(z_m, w0_m=w0s_m, z0_m=z0s_m, zR_m=zR_s_m, k_med=k_s)
    ai, Ci = _gaussian_alpha_C(z_m, w0_m=w0i_m, z0_m=z0i_m, zR_m=zR_i_m, k_med=k_i)
    alpha_t = ap + np.conjugate(a_s) + np.conjugate(ai)
    if abs(alpha_t) < 1.0e-30:
        return 0.0j
    dx = float(dx_m)
    dy = float(dy_m)
    beta = float(k_i) * float(dtheta_rad)
    gx = 2.0 * np.conjugate(ai) * dx + 1j * beta
    gy = 2.0 * np.conjugate(ai) * dy
    pref = Cp * np.conjugate(Cs) * np.conjugate(Ci)
    gauss = (np.pi / alpha_t) * np.exp(
        (gx * gx + gy * gy) / (4.0 * alpha_t)
        - np.conjugate(ai) * (dx * dx + dy * dy)
    )
    return pref * gauss


def _trapz_complex(y: np.ndarray, x: np.ndarray) -> complex:
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return complex(trapz(np.real(y), x) + 1j * trapz(np.imag(y), x))


def spdc_geometry_overlap_Ogeom(
    *,
    crystal_length_m: float,
    w0p_m: float,
    w0s_m: float,
    w0i_m: float,
    z0p_m: float,
    z0s_m: float,
    z0i_m: float,
    n_pump: float,
    n_signal: float,
    n_idler: float,
    lambda_p_m: float,
    lambda_s_m: float,
    lambda_i_m: float,
    dx_m: float = 0.0,
    dy_m: float = 0.0,
    dtheta_rad: float = 0.0,
    n_z: int = 97,
) -> float:
    """O_geom = |∫ A dz| / |∫ A_ideal dz| ∈ [0, 1] (report eq. 34–35).

    A_ideal uses the same waists, confocal at the crystal centre, no offset.
    Report §7.8: integrate A(z) only. Do not multiply by e^{i Δk_QPM z}
    and do not call ``delta_k_qpm_per_um``. That factor belongs in the
    gain (report §7.9). A future Φ(q_s, q_i) model (eq. 37) must replace
    this split as one piece.
    """
    if OGEOM_INCLUDES_QPM_PHASE:
        raise RuntimeError(
            "report §7.8 forbids e^{i Δk_QPM z} in O_geom; "
            "Δk_QPM belongs in the gain (report §7.9)"
        )
    L = float(crystal_length_m)
    zR_p = rayleigh_range_m(n=n_pump, waist_m=w0p_m, lambda_m=lambda_p_m)
    zR_s = rayleigh_range_m(n=n_signal, waist_m=w0s_m, lambda_m=lambda_s_m)
    zR_i = rayleigh_range_m(n=n_idler, waist_m=w0i_m, lambda_m=lambda_i_m)
    kp = 2.0 * np.pi * float(n_pump) / float(lambda_p_m)
    ks = 2.0 * np.pi * float(n_signal) / float(lambda_s_m)
    ki = 2.0 * np.pi * float(n_idler) / float(lambda_i_m)
    z = np.linspace(-0.5 * L, 0.5 * L, int(n_z))

    def _A(z0p: float, z0s: float, z0i: float, dx: float, dy: float, th: float) -> np.ndarray:
        return np.array(
            [
                transverse_generation_overlap_A(
                    zz,
                    w0p_m=w0p_m,
                    w0s_m=w0s_m,
                    w0i_m=w0i_m,
                    z0p_m=z0p,
                    z0s_m=z0s,
                    z0i_m=z0i,
                    zR_p_m=zR_p,
                    zR_s_m=zR_s,
                    zR_i_m=zR_i,
                    k_p=kp,
                    k_s=ks,
                    k_i=ki,
                    dx_m=dx,
                    dy_m=dy,
                    dtheta_rad=th,
                )
                for zz in z
            ],
            dtype=np.complex128,
        )

    O_raw = _trapz_complex(_A(z0p_m, z0s_m, z0i_m, dx_m, dy_m, dtheta_rad), z)
    O_id = _trapz_complex(_A(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), z)
    mag_id = abs(O_id)
    if mag_id < 1.0e-30:
        return 0.0
    return float(np.clip(abs(O_raw) / mag_id, 0.0, 1.0))


def spdc_geometry_overlap_from_setup(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> float:
    """O_geom for one pass from real geometry (report eq. 39).

    Single-crystal Physical Mode shares L and n (§7.11); two_crystal uses
    that crystal. Waist and focus remain per-pass.

    V-P1-04: uses ``generation_*`` misalignment only. Relay / ``idler_*``
    offsets belong to η_mode and must not enter O_geom.
    """
    wp_um = resolved_pump_waist_um(setup, interaction=interaction)
    if wp_um is None:
        return 1.0
    w_i_um = resolved_generation_idler_waist_um(setup, interaction=interaction)
    if w_i_um is None:
        w_i_um = float(wp_um)
    z0_um = resolved_pump_focus_z_um(setup, interaction=interaction)
    L_m = crystal_length_m_from_setup(setup, interaction=interaction)
    if bool(getattr(setup, "use_physical_spdc_gain", False)):
        cr = spdc_crystal_state(setup, interaction=interaction)
        n_pump = float(cr["n_pump"])
        n_signal = float(cr["n_signal"])
        n_idler = float(cr["n_idler"])
    else:
        T = float(setup.T_crystal_NL1_C)
        n_pump = ktp_nz(float(setup.lambda_p_nm) * 1.0e-3, T)
        n_signal = ktp_nz(float(setup.lambda_s_nm) * 1.0e-3, T)
        n_idler = ktp_nz(float(setup.lambda_i_nm) * 1.0e-3, T)
    lp_m = float(setup.lambda_p_nm) * 1.0e-9
    ls_m = float(setup.lambda_s_nm) * 1.0e-9
    li_m = float(setup.lambda_i_nm) * 1.0e-9
    w0p_m = float(wp_um) * 1.0e-6
    # V-P1-04: generation alignment only (pump→crystal). Relay offsets stay out.
    gdx, gdy, gdth = generation_alignment_um(setup)
    return spdc_geometry_overlap_Ogeom(
        crystal_length_m=L_m,
        w0p_m=w0p_m,
        w0s_m=w0p_m,
        w0i_m=float(w_i_um) * 1.0e-6,
        z0p_m=z0_um * 1.0e-6,
        z0s_m=z0_um * 1.0e-6,
        z0i_m=z0_um * 1.0e-6,
        n_pump=n_pump,
        n_signal=n_signal,
        n_idler=n_idler,
        lambda_p_m=lp_m,
        lambda_s_m=ls_m,
        lambda_i_m=li_m,
        dx_m=gdx * 1.0e-6,
        dy_m=gdy * 1.0e-6,
        dtheta_rad=gdth * 1.0e-3,
    )


def spdc_gaussian_geometry_from_setup(
    setup: "SetupParams",
    *,
    interaction: int = 1,
) -> dict[str, float | None]:
    """z_R, ξ, z0 and O_geom for one pass (report eq. 32–36 / §7.7).

    Signal waist is the pump waist of that pass (no free focusing factor).
    Phase mismatch is not included.
    """
    wp_um = resolved_pump_waist_um(setup, interaction=interaction)
    wi_um = resolved_generation_idler_waist_um(setup, interaction=interaction)
    z0_um = resolved_pump_focus_z_um(setup, interaction=interaction)
    out: dict[str, float | None] = {
        "pump_waist_um": None if wp_um is None else float(wp_um),
        "signal_waist_um": None if wp_um is None else float(wp_um),
        "idler_waist_um": None if wi_um is None else float(wi_um),
        "pump_focus_z_um": float(z0_um),
        "zR_pump_mm": None,
        "zR_signal_mm": None,
        "zR_idler_mm": None,
        "xi_pump": None,
        "O_geom": spdc_geometry_overlap_from_setup(setup, interaction=interaction),
    }
    if wp_um is None or wi_um is None:
        return out
    L_m = crystal_length_m_from_setup(setup, interaction=interaction)
    if bool(getattr(setup, "use_physical_spdc_gain", False)):
        cr = spdc_crystal_state(setup, interaction=interaction)
        n_pump = float(cr["n_pump"])
        n_signal = float(cr["n_signal"])
        n_idler = float(cr["n_idler"])
    else:
        T = float(setup.T_crystal_NL1_C)
        n_pump = ktp_nz(float(setup.lambda_p_nm) * 1.0e-3, T)
        n_signal = ktp_nz(float(setup.lambda_s_nm) * 1.0e-3, T)
        n_idler = ktp_nz(float(setup.lambda_i_nm) * 1.0e-3, T)
    lp_m = float(setup.lambda_p_nm) * 1.0e-9
    ls_m = float(setup.lambda_s_nm) * 1.0e-9
    li_m = float(setup.lambda_i_nm) * 1.0e-9
    w0p_m = float(wp_um) * 1.0e-6
    w0i_m = float(wi_um) * 1.0e-6
    zR_p = rayleigh_range_m(
        n=n_pump,
        waist_m=w0p_m,
        lambda_m=lp_m,
    )
    zR_s = rayleigh_range_m(
        n=n_signal,
        waist_m=w0p_m,
        lambda_m=ls_m,
    )
    zR_i = rayleigh_range_m(
        n=n_idler,
        waist_m=w0i_m,
        lambda_m=li_m,
    )
    out["zR_pump_mm"] = float(zR_p * 1.0e3)
    out["zR_signal_mm"] = float(zR_s * 1.0e3)
    out["zR_idler_mm"] = float(zR_i * 1.0e3)
    out["xi_pump"] = focus_parameter_xi(
        crystal_length_m=L_m,
        rayleigh_m=zR_p,
    )
    return out


def idler_from_energy_nm(lambda_p_nm: float, lambda_s_nm: float) -> float:
    """1/λ_p = 1/λ_s + 1/λ_i → λ_i."""
    inv = 1.0 / float(lambda_p_nm) - 1.0 / float(lambda_s_nm)
    if inv <= 1e-18:
        return float("inf")
    return float(1.0 / inv)


def calibrated_poling_period_um(setup: "SetupParams") -> float:
    """Λ_eff so Δk(T_pm, λ_s0, λ_i0)=0 — absorbs Sellmeier absolute error.

    Calibration-mode helper only (report §5.7 / §7.3). It must not be called
    from the Physical Mode QPM path; that path uses the lab period via
    ``qpm_poling_period_um``. YAML ``poling_period_um`` remains the published
    crystal period (Lemos 9.675 µm).
    """
    st = crystal_thermal_state(setup)
    T_pm = float(st["T_pm_NL1_C"])
    lp = float(setup.lambda_p_nm) * 1e-3
    ls = float(setup.lambda_s_nm) * 1e-3
    li = float(setup.lambda_i_nm) * 1e-3
    np_ = ktp_nz(lp, T_pm)
    ns_ = ktp_nz(ls, T_pm)
    ni_ = ktp_nz(li, T_pm)
    dk0 = np_ / lp - ns_ / ls - ni_ / li
    if abs(dk0) < 1e-18:
        return validate_poling_period_um(setup.poling_period_um)
    Lam = 1.0 / dk0
    if Lam <= 0:
        return validate_poling_period_um(setup.poling_period_um)
    return float(np.clip(Lam, POLING_PERIOD_UM_MIN, POLING_PERIOD_UM_MAX))


def qpm_poling_period_um(
    setup: "SetupParams",
    *,
    T_C: float | None = None,
    interaction: int = 1,
) -> float:
    """Λ(T) that enters Δk_QPM (report eq. 45 / §7.3).

    Physical Mode (default): lab period after thermal expansion from the
    temperature at which that period is specified (``T_pm0``) to the
    operating crystal temperature:
        Λ(T) = Λ_lab (1 + α_KTP (T − T_pm0))

    Calibration mode (``use_calibrated_poling_period=True``): replace Λ_lab by
    the explicit Λ_eff from ``calibrated_poling_period_um``, then apply the
    same expansion. Opt-in only; NL1 only. two_crystal NL2 always uses its
    lab period (``poling_period_NL2_um`` or crystal-1 Λ).
    """
    st = crystal_thermal_state(setup)
    use_nl2 = is_two_crystal_architecture(setup) and int(interaction) == 2
    if T_C is None:
        T = float(st["T_crystal_NL2_C"] if use_nl2 else st["T_crystal_NL1_C"])
    else:
        T = float(T_C)
    T_ref = float(st["T_pm0_NL2_C"] if use_nl2 else st["T_pm0_NL1_C"])
    calibrated = bool(getattr(setup, "use_calibrated_poling_period", False))
    if calibrated and not use_nl2:
        Lam0 = calibrated_poling_period_um(setup)
    else:
        Lam0 = lab_poling_period_um(setup, interaction=interaction)
    return poling_period_um_at_T(Lam0, T, T_ref)


def phase_match_poling_period_um(
    setup: "SetupParams",
    *,
    T_C: float | None = None,
    interaction: int = 1,
) -> float:
    """Solve the positive collinear Type-0 QPM period at the operating T.

    This helper never changes the configured period. It evaluates
    ``Lambda_pm = m/(n_p/lambda_p-n_s/lambda_s-n_i/lambda_i)`` in micrometres.
    """
    cr = spdc_crystal_state(setup, interaction=interaction)
    T = float(cr["T_crystal_C"] if T_C is None else T_C)
    lp = float(setup.lambda_p_nm) * 1.0e-3
    ls = float(setup.lambda_s_nm) * 1.0e-3
    li = float(setup.lambda_i_nm) * 1.0e-3
    denominator = (
        ktp_nz(lp, T) / lp
        - ktp_nz(ls, T) / ls
        - ktp_nz(li, T) / li
    )
    order = int(cr["qpm_order"])
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise QPMError(
            "the selected wavelength/temperature convention has no positive "
            "collinear QPM period"
        )
    period = float(order / denominator)
    if not np.isfinite(period) or period <= 0.0:
        raise QPMError("phase-match poling solver returned a nonphysical period")
    return period


def qpm_configuration_evidence(
    setup: "SetupParams", *, interaction: int = 1
) -> dict[str, Any]:
    """Requested, thermally resolved, and solved QPM quantities."""
    cr = spdc_crystal_state(setup, interaction=interaction)
    T = float(cr["T_crystal_C"])
    requested = float(lab_poling_period_um(setup, interaction=interaction))
    resolved = float(qpm_poling_period_um(setup, T_C=T, interaction=interaction))
    phase_match = float(
        phase_match_poling_period_um(setup, T_C=T, interaction=interaction)
    )
    order = int(cr["qpm_order"])
    residual_configured = float(
        delta_k_qpm_per_um(
            lambda_p_nm=setup.lambda_p_nm,
            lambda_s_nm=setup.lambda_s_nm,
            lambda_i_nm=setup.lambda_i_nm,
            T_C=T,
            Lambda_um=resolved,
            qpm_order=order,
        )
    )
    residual_solved = float(
        delta_k_qpm_per_um(
            lambda_p_nm=setup.lambda_p_nm,
            lambda_s_nm=setup.lambda_s_nm,
            lambda_i_nm=setup.lambda_i_nm,
            T_C=T,
            Lambda_um=phase_match,
            qpm_order=order,
        )
    )
    direction = spdc_propagation_direction(setup, interaction=interaction)
    sign = -1.0 if direction == "backward" else 1.0
    return {
        "interaction": int(interaction),
        "temperature_C": T,
        "poling_period_requested_um": requested,
        "poling_period_resolved_at_temperature_um": resolved,
        "poling_period_phase_match_um": phase_match,
        "delta_k_QPM_per_um": sign * residual_configured,
        "delta_k_QPM_per_m": sign * residual_configured * 1.0e6,
        "phase_match_residual_per_um": residual_solved,
        "phase_match_residual_per_m": residual_solved * 1.0e6,
        "qpm_order": order,
        "configured_period_used_in_forward": True,
    }


class QPMError(ValueError):
    """Collinear QPM root not found or |Δk| exceeds the solver tolerance."""


class SPDCGainOverflowError(ValueError):
    """High-gain nbar / sinh² exceeded float64 (V-P1-02). Never silent-zero."""


# Exact sinh² below this; log-domain asymptotic above (before float64 sinh blows up).
SINH2_EXACT_MAX = 20.0
# Leave headroom under float64 max for exp(log_nbar).
_LOG_NBAR_MAX = float(np.log(np.finfo(np.float64).max)) - 8.0


def sinh_squared_stable(x: float) -> float:
    """Compute ``sinh(x)**2`` with a log-domain high-gain path (V-P1-02).

    For large ``|x|``, ``sinh(x) ≈ ±½ exp(|x|)``, so::

        log_nbar = 2 * (|x| - log(2))
        nbar = exp(log_nbar)

    Overflow raises ``SPDCGainOverflowError`` (must not become 0).
    """
    return scaled_sinh_squared_stable(1.0, x)


def scaled_sinh_squared_stable(scale: float, x: float) -> float:
    """Return ``scale * sinh(x)**2`` without float overflow→0 (V-P1-02)."""
    ax = abs(float(x))
    sc = float(scale)
    if not np.isfinite(ax) or not np.isfinite(sc):
        raise SPDCGainOverflowError(
            f"non-finite scaled sinh² arguments: scale={scale}, x={x}"
        )
    if sc < 0.0:
        raise ValueError(f"scale for sinh² must be >= 0, got {scale}")
    if sc == 0.0 or ax == 0.0:
        return 0.0

    if ax <= SINH2_EXACT_MAX:
        nbar = sc * float(np.sinh(ax) ** 2)
    else:
        # log(scale * (e^{ax}/2)^2) = log(scale) + 2*(ax - log(2))
        log_nbar = float(np.log(sc) + 2.0 * (ax - np.log(2.0)))
        if log_nbar > _LOG_NBAR_MAX:
            raise SPDCGainOverflowError(
                f"high-gain nbar overflows float64: "
                f"log_nbar={log_nbar:.6g} (scale={sc}, |x|={ax})"
            )
        nbar = float(np.exp(log_nbar))

    if not np.isfinite(nbar) or nbar < 0.0:
        raise SPDCGainOverflowError(
            f"high-gain nbar non-finite after evaluation: {nbar} "
            f"(scale={sc}, x={x})"
        )
    return nbar


# Default |Δk_QPM| acceptance after a local root solve [rad/µm].
QPM_DELTA_K_TOL_PER_UM = 1.0e-8
QPM_SCAN_POINTS = 257


def solve_qpm(
    setup: "SetupParams",
    *,
    T_C: float | None = None,
    lambda_s_min_nm: float | None = None,
    lambda_s_max_nm: float | None = None,
    n_scan: int = QPM_SCAN_POINTS,
    delta_k_tol_per_um: float = QPM_DELTA_K_TOL_PER_UM,
    interaction: int = 1,
) -> dict[str, Any]:
    """Solve collinear type-0 QPM for (λ_s, λ_i) at temperature T (V-P0-05).

    Procedure:
      1. Scan λ_s over a wavelength window
      2. Detect residual sign changes of Δk_QPM(λ_s)
      3. Local root solve on each bracketing interval
      4. Prefer the root closest to the lab λ_s
      5. Validate |Δk|; raise ``QPMError`` if no root or |Δk| > tolerance

    No silent fallback to lab wavelengths.
    """
    from scipy.optimize import brentq

    st = crystal_thermal_state(setup)
    T = float(st["T_crystal_NL1_C"] if T_C is None else T_C)
    Lam = qpm_poling_period_um(setup, T_C=T, interaction=interaction)
    lp = float(setup.lambda_p_nm)
    ls0 = float(setup.lambda_s_nm)
    # Non-degenerate window: λ_s > λ_p and λ_i finite.
    lo_default = max(lp * 1.01, ls0 * 0.70)
    hi_cap = idler_from_energy_nm(lp, lp * 1.01) - 1.0
    hi_default = min(hi_cap, ls0 * 1.40)
    lo = float(lo_default if lambda_s_min_nm is None else lambda_s_min_nm)
    hi = float(hi_default if lambda_s_max_nm is None else lambda_s_max_nm)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise QPMError(
            f"invalid QPM wavelength scan window [{lo}, {hi}] nm "
            f"(lambda_p={lp} nm, lambda_s0={ls0} nm)"
        )
    n_scan = max(int(n_scan), 8)
    tol = float(delta_k_tol_per_um)
    if not np.isfinite(tol) or tol <= 0.0:
        raise QPMError(f"delta_k_tol_per_um must be finite and > 0, got {delta_k_tol_per_um}")

    def residual(ls: float) -> float:
        li = idler_from_energy_nm(lp, ls)
        if not np.isfinite(li) or li <= 0.0:
            return float("nan")
        return delta_k_qpm_per_um(
            lambda_p_nm=lp,
            lambda_s_nm=float(ls),
            lambda_i_nm=float(li),
            T_C=T,
            Lambda_um=Lam,
            qpm_order=int(getattr(setup, "qpm_order", QPM_ORDER_DEFAULT)),
        )

    grid = np.linspace(lo, hi, n_scan, dtype=np.float64)
    vals = np.array([residual(float(x)) for x in grid], dtype=np.float64)

    brackets: list[tuple[float, float]] = []
    for i in range(n_scan - 1):
        a = float(vals[i])
        b = float(vals[i + 1])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        xa = float(grid[i])
        xb = float(grid[i + 1])
        if abs(a) <= tol:
            brackets.append((xa, xa))
        elif a * b < 0.0:
            brackets.append((xa, xb))
        elif abs(b) <= tol and i == n_scan - 2:
            brackets.append((xb, xb))

    if not brackets:
        raise QPMError(
            "QPM scan found no delta_k sign change in "
            f"lambda_s in [{lo:.3f}, {hi:.3f}] nm at T={T:.3f} C "
            f"(Lambda={Lam:.6f} um). Refusing silent lab-wavelength fallback."
        )

    roots: list[dict[str, Any]] = []
    for a, b in brackets:
        if abs(a - b) < 1e-15:
            ls = float(a)
        else:
            try:
                ls = float(
                    brentq(residual, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                )
            except ValueError as exc:
                raise QPMError(
                    f"local QPM root solve failed on bracket [{a:.6f}, {b:.6f}] nm: {exc}"
                ) from exc
        li = float(idler_from_energy_nm(lp, ls))
        dk = float(residual(ls))
        roots.append(
            {
                "lambda_s_nm": ls,
                "lambda_i_nm": li,
                "delta_k_qpm_per_um": dk,
                "bracket_nm": (float(a), float(b)),
                "abs_delta_from_lab_nm": abs(ls - ls0),
            }
        )

    roots.sort(key=lambda r: (r["abs_delta_from_lab_nm"], abs(r["delta_k_qpm_per_um"])))
    best = roots[0]
    dk = float(best["delta_k_qpm_per_um"])
    if abs(dk) > tol:
        raise QPMError(
            f"|delta_k_QPM|={abs(dk):.3e} rad/um exceeds tolerance {tol:.3e} "
            f"at lambda_s={best['lambda_s_nm']:.6f} nm, "
            f"lambda_i={best['lambda_i_nm']:.6f} nm, T={T:.3f} C"
        )

    return {
        "lambda_s_nm": float(best["lambda_s_nm"]),
        "lambda_i_nm": float(best["lambda_i_nm"]),
        "delta_k_qpm_per_um": dk,
        "T_C": float(T),
        "Lambda_um": float(Lam),
        "lambda_p_nm": float(lp),
        "n_roots": len(roots),
        "roots": roots,
        "scan_nm": (float(lo), float(hi)),
        "n_scan": int(n_scan),
        "delta_k_tol_per_um": tol,
        "interaction": int(interaction),
    }


def solve_signal_wavelength_nm(
    setup: "SetupParams",
    *,
    T_C: float | None = None,
) -> tuple[float, float]:
    """Solve collinear type-0 QPM for (λ_s, λ_i) at temperature T.

    Thin wrapper around ``solve_qpm`` (V-P0-05). Energy conservation +
    Δk(T, Λ(T))=0. Returns nm. Raises ``QPMError`` if no validated root.
    """
    out = solve_qpm(setup, T_C=T_C)
    return float(out["lambda_s_nm"]), float(out["lambda_i_nm"])


def dlambda_dT_nm_per_C(setup: "SetupParams", *, dT_step_C: float = 0.5) -> dict[str, float]:
    """Numerical dλ_s/dT, dλ_i/dT from Sellmeier QPM [nm/°C]."""
    st = crystal_thermal_state(setup)
    T0 = float(st["T_crystal_NL1_C"])
    ls1, li1 = solve_signal_wavelength_nm(setup, T_C=T0 - 0.5 * dT_step_C)
    ls2, li2 = solve_signal_wavelength_nm(setup, T_C=T0 + 0.5 * dT_step_C)
    return {
        "dlam_s_nm_per_C": float((ls2 - ls1) / dT_step_C),
        "dlam_i_nm_per_C": float((li2 - li1) / dT_step_C),
        "lambda_s_nm": float(solve_signal_wavelength_nm(setup, T_C=T0)[0]),
        "lambda_i_nm": float(solve_signal_wavelength_nm(setup, T_C=T0)[1]),
    }


def r_spdc_sinc2(setup: "SetupParams", *, T_C: float | None = None) -> float:
    """PDF §5.1: R_SPDC ∝ |sinc(Δk L / 2)|² at current (λ_s, λ_i, T).

    Diagnostic / low-gain comparison only (report §7.10). Physical Mode
    flux uses YAML flux0; Δk_QPM is in nbar (report §7.9), not here.
    Must not be multiplied into O_geom (report §7.8).
    """
    T = T_C
    dk = delta_k_qpm_per_m_from_setup(setup, T_C=T)
    L = crystal_length_m_from_setup(setup, interaction=1)
    x = 0.5 * dk * L
    if abs(x) < 1e-12:
        return 1.0
    return float((np.sin(x) / x) ** 2)


def thermal_beam_shift_um(
    setup: "SetupParams",
    *,
    dT_C: float | None = None,
    t_s: float | None = None,
) -> tuple[float, float]:
    """PDF §5.4/§5.9: Δx = α_x ΔT, Δy = α_y ΔT [µm].

    Defaults: α_{x,y} = CTE_mount · F1 (Al × 75 mm lever).
    """
    if dT_C is None:
        dT_C = float(crystal_thermal_state(setup)["dT_eff_C"])
    dT = float(dT_C)
    if t_s is not None:
        dT *= oven_settling_factor(t_s)
    ax = float(getattr(setup, "beam_thermal_alpha_x_um_per_C", CTE_MOUNT_PER_C * F1_MM_DEFAULT * 1e3))
    ay = float(getattr(setup, "beam_thermal_alpha_y_um_per_C", CTE_MOUNT_PER_C * F1_MM_DEFAULT * 1e3))
    return float(ax * dT), float(ay * dT)


def lens_focal_shift_um(setup: "SetupParams", *, dT_C: float | None = None) -> float:
    """PDF §5.5: Δf = f [(dn/dT)/(n−1) − α_glass] ΔT  for N-BK7 (no housing)."""
    if dT_C is None:
        dT_C = float(crystal_thermal_state(setup)["dT_eff_C"])
    f_um = float(getattr(setup, "lens_focal_mm", LENS_FOCAL_MM_DEFAULT)) * 1e3
    n = float(getattr(setup, "lens_n", NBK7_N))
    dndT = float(getattr(setup, "lens_dndT_per_C", NBK7_DNDT_PER_C))
    alpha = float(getattr(setup, "lens_cte_per_C", NBK7_CTE_PER_C))
    beta = dndT / max(n - 1.0, 1e-6) - alpha
    return float(f_um * beta * float(dT_C))


def qiup_magnification(setup: "SetupParams", *, lambda_s_nm: float, lambda_i_nm: float) -> float:
    """PDF §5.7: M_QIUP = F2 λ_s / (F1 λ_i). Single source of truth (V-P1-06)."""
    F1 = float(getattr(setup, "F1_mm", F1_MM_DEFAULT))
    F2 = float(getattr(setup, "F2_mm", F2_MM_DEFAULT))
    M = float(F2 * float(lambda_s_nm) / max(F1 * float(lambda_i_nm), 1e-12))
    if not np.isfinite(M) or M <= 0.0:
        raise ValueError(f"M_QIUP must be finite and > 0, got {M}")
    return M


def resolve_imaging_magnification(
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> float:
    """Return ``M_QIUP`` for all coordinate transforms (V-P1-06).

    Always ``M_QIUP = F2 λ_s / (F1 λ_i)`` (§5.7). Temperature-tuned λ from
    ``phys_T`` when present. Legacy YAML ``magnification`` is **not** used
    for physics — it is synced to this value in ``validate_setup_params``.
    """
    if phys_T is not None and "M_QIUP" in phys_T:
        M = float(phys_T["M_QIUP"])
        if not np.isfinite(M) or M <= 0.0:
            raise ValueError(f"phys_T M_QIUP must be finite and > 0, got {M}")
        return M
    ls = float(phys_T["lambda_s_nm_T"]) if phys_T and "lambda_s_nm_T" in phys_T else float(setup.lambda_s_nm)
    li = float(phys_T["lambda_i_nm_T"]) if phys_T and "lambda_i_nm_T" in phys_T else float(setup.lambda_i_nm)
    return qiup_magnification(setup, lambda_s_nm=ls, lambda_i_nm=li)


def object_pixel_size_um(
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
    magnification: float | None = None,
) -> float:
    """Object-plane size of one detector pixel [µm]: p_obj = p_det / M_QIUP (V-P1-06)."""
    M = (
        float(magnification)
        if magnification is not None
        else resolve_imaging_magnification(setup, phys_T=phys_T)
    )
    return float(internal_camera_sampling_pixel_um(setup) / max(M, 1e-12))


def beam_size_px(
    setup: "SetupParams",
    *,
    N: int | None = None,
    phys_T: dict[str, Any] | None = None,
    magnification: float | None = None,
) -> float:
    """Pump beam waist on the sensor [px], derived from M_QIUP (V-P1-06)."""
    M = (
        float(magnification)
        if magnification is not None
        else resolve_imaging_magnification(setup, phys_T=phys_T)
    )
    if bool(setup.use_legacy_beam_match) or setup.pump_waist_um is None:
        if N is None:
            raise ValueError("beam_size_px legacy path requires N")
        return float(legacy_waist_px(int(N)))
    w_um = validate_pump_waist_um(float(setup.pump_waist_um))
    return um_to_px(
        w_um,
        pixel_um=internal_camera_sampling_pixel_um(setup),
        magnification=M,
    )


def psf_size_px(
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
    magnification: float | None = None,
) -> float:
    """Correlation / SPDC PSF σ [px] from M_QIUP (V-P1-06)."""
    if not bool(setup.derive_sigma_spdc):
        return float(setup.sigma_spdc_px)
    if (
        getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        return float(
            resolved_spatial_mapping(setup, phys_T=phys_T)[
                "correlation_sigma_camera_px"
            ]
        )
    M = (
        float(magnification)
        if magnification is not None
        else resolve_imaging_magnification(setup, phys_T=phys_T)
    )
    return um_to_px(
        sigma_correlation_um(setup),
        pixel_um=internal_camera_sampling_pixel_um(setup),
        magnification=M,
    )


def um_to_px_setup(
    length_um: float,
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> float:
    """µm → px using detector pitch and M_QIUP (V-P1-06)."""
    return um_to_px(
        length_um,
        pixel_um=internal_camera_sampling_pixel_um(setup),
        magnification=resolve_imaging_magnification(setup, phys_T=phys_T),
    )


def px_to_um_setup(
    length_px: float,
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> float:
    """px → µm using detector pitch and M_QIUP (V-P1-06)."""
    return px_to_um(
        length_px,
        pixel_um=internal_camera_sampling_pixel_um(setup),
        magnification=resolve_imaging_magnification(setup, phys_T=phys_T),
    )


def resolve_image_scale(setup: "SetupParams") -> float:
    """PDF §6.17: geometric scale factor = 1 + ε_M, ε_M = ΔM/M.

    Magnification error is a coordinate stretch — not folded into σ_opt.
    """
    eps = float(getattr(setup, "delta_M_rel", DELTA_M_REL_DEFAULT))
    if eps <= -1.0:
        raise ValueError(f"delta_M_rel must be > -1 (so scale > 0), got {eps}")
    return float(1.0 + eps)


def physical_thermal_bundle(
    setup: "SetupParams",
    *,
    t_s: float | None = None,
) -> dict[str, Any]:
    """Resolve all §5.1–§5.9 physical thermal quantities for forward/meta."""
    st = crystal_thermal_state(setup)
    dT = float(st["dT_eff_C"])
    dL_th = delta_L_thermal_mm(setup, dT_C=dT, t_s=t_s)
    opd = optical_path_difference_mm(setup, t_s=t_s)
    dL_eff = effective_delta_L_mm(setup, t_s=t_s)
    V_coh = v_coh_gaussian(dL_eff, setup.L_coh_mm)
    dphi = opd_phase_rad(setup, t_s=t_s)
    dx_um, dy_um = thermal_beam_shift_um(setup, dT_C=dT, t_s=t_s)
    w1, w2 = resolved_idler_waist_um(setup)
    rdx, rdy, rdth = relay_alignment_um(setup)
    eta_th = eta_mode_gaussian(
        w1,
        w2,
        dx_um + rdx,
        dy_um + rdy,
        dtheta_mrad=rdth,
        lambda_i_nm=float(setup.lambda_i_nm),
    )
    lam = dlambda_dT_nm_per_C(setup)
    ls_T, li_T = float(lam["lambda_s_nm"]), float(lam["lambda_i_nm"])
    M = qiup_magnification(setup, lambda_s_nm=ls_T, lambda_i_nm=li_T)
    M0 = qiup_magnification(
        setup, lambda_s_nm=float(setup.lambda_s_nm), lambda_i_nm=float(setup.lambda_i_nm)
    )
    df_um = lens_focal_shift_um(setup, dT_C=dT)
    R = r_spdc_sinc2(setup)
    V0 = float(setup.V0)
    V_phys = float(np.clip(V0 * V_coh * eta_th, 0.0, 1.0))
    return {
        "dT_eff_C": dT,
        "delta_L_thermal_mm": dL_th,
        "optical_path_difference_mm": opd,
        "delta_L_eff_mm": dL_eff,
        "V_coh": V_coh,
        "thermal_phase_offset_rad": dphi,
        "opd_phase_rad": dphi,
        "theta0_rad": THETA0_RAD,
        "oven_tau_s": OVEN_TAU_S,
        "formula_OPD": "delta_L0 + delta_L_thermal (signed)",
        "formula_V_coh": "exp(-(abs(OPD)/L_coh)^2)",
        "formula_dphi": "k*OPD with k=2*pi/lambda_s",
        "beam_dx_um": dx_um,
        "beam_dy_um": dy_um,
        "eta_mode_thermal": eta_th,
        "lambda_s_nm_T": ls_T,
        "lambda_i_nm_T": li_T,
        "dlam_s_nm_per_C": lam["dlam_s_nm_per_C"],
        "dlam_i_nm_per_C": lam["dlam_i_nm_per_C"],
        "M_QIUP": M,
        "M_QIUP_0": M0,
        "dM_over_M": float((M - M0) / max(M0, 1e-12)),
        "lens_delta_f_um": df_um,
        "R_SPDC_sinc2": R,
        "V0": V0,
        "V_physical": V_phys,
        "alpha_path_per_C": float(setup.alpha_path_per_C),
        "L_eff_path_mm": float(setup.L_eff_path_mm),
        "formula_delta_L_th": "alpha_path * L_eff_path * dT",
        "formula_V": "V0 * V_coh(T) * V_overlap(T)",
    }


def sigma_r_um(*, crystal_length_mm: float, lambda_p_nm: float) -> float:
    """Legacy PDF §3.1 estimate σ_r = √(2 L λ_p / 3π) [µm].

    Kept for audit of the superseded near-field kernel. Active near-field
    transfer uses :func:`sigma_near_field_um` (P07/P10 visibility width).
    """
    L_m = validate_crystal_length_mm(crystal_length_mm) * 1e-3
    lam_m = float(lambda_p_nm) * 1e-9
    if lam_m <= 0:
        raise ValueError(f"lambda_p_nm must be > 0, got {lambda_p_nm}")
    sigma_r_m = float(np.sqrt(2.0 * L_m * lam_m / (3.0 * np.pi)))
    return sigma_r_m * 1e6


def _resolved_pump_waist_um(setup: "SetupParams") -> float:
    w_um = float(
        setup.pump_waist_um
        if setup.pump_waist_um is not None
        else PUMP_WAIST_UM_DEFAULT
    )
    return validate_pump_waist_um(w_um)


def near_field_visibility_1e_um(
    setup: "SetupParams",
    *,
    lambda_s_nm: float | None = None,
    lambda_i_nm: float | None = None,
) -> float:
    """Near-field visibility 1/e half-width [µm] at the crystal plane.

    P07 Eq. (4.9) / P10 Eq. (13). ``w_p`` is the 1/e² intensity radius.
    The large-waist limit is √[L(λ_s+λ_i)/4π].
    """
    L_um = validate_crystal_length_mm(float(setup.crystal_length_mm)) * 1.0e3
    lam_d_um = float(
        setup.lambda_s_nm if lambda_s_nm is None else lambda_s_nm
    ) * 1.0e-3
    lam_u_um = float(
        setup.lambda_i_nm if lambda_i_nm is None else lambda_i_nm
    ) * 1.0e-3
    if lam_d_um <= 0.0:
        raise ValueError(f"lambda_s_nm must be > 0, got {lam_d_um * 1.0e3}")
    if lam_u_um <= 0.0:
        raise ValueError(f"lambda_i_nm must be > 0, got {lam_u_um * 1.0e3}")
    w_um = _resolved_pump_waist_um(setup)
    lam_sum = lam_d_um + lam_u_um
    a = 2.0 * np.pi * w_um * w_um * lam_sum
    denom = a - lam_d_um * lam_u_um * L_um
    if abs(float(denom)) <= 1.0e-30 * max(1.0, abs(float(a))):
        raise ValueError(
            "near-field visibility width diverges: pump waist is at the "
            "separable-state singularity of P07 Eq. (4.9)"
        )
    delta = (
        np.sqrt(L_um * lam_sum / (4.0 * np.pi))
        * a
        * np.sqrt(1.0 + (lam_d_um * lam_d_um) * L_um / a)
        / abs(float(denom))
    )
    return float(delta)


def near_field_gain_1e_um(
    setup: "SetupParams",
    *,
    lambda_s_nm: float | None = None,
    lambda_i_nm: float | None = None,
) -> float:
    """Near-field image-function 1/e half-width [µm] at the crystal plane.

    P10 Eq. (12) for G_PSF. ``λ_d=λ_s`` (detected), ``λ_u=λ_i`` (undetected).
    Large-waist limit is the same √[L(λ_s+λ_i)/4π] as Δ_V.
    """
    L_um = validate_crystal_length_mm(float(setup.crystal_length_mm)) * 1.0e3
    lam_d_um = float(
        setup.lambda_s_nm if lambda_s_nm is None else lambda_s_nm
    ) * 1.0e-3
    lam_u_um = float(
        setup.lambda_i_nm if lambda_i_nm is None else lambda_i_nm
    ) * 1.0e-3
    if lam_d_um <= 0.0:
        raise ValueError(f"lambda_s_nm must be > 0, got {lam_d_um * 1.0e3}")
    if lam_u_um <= 0.0:
        raise ValueError(f"lambda_i_nm must be > 0, got {lam_u_um * 1.0e3}")
    w_um = _resolved_pump_waist_um(setup)
    lam_sum = lam_d_um + lam_u_um
    a = 2.0 * np.pi * w_um * w_um * lam_sum
    return float(
        np.sqrt(L_um * lam_sum / (4.0 * np.pi))
        / np.sqrt(1.0 + (lam_u_um * lam_u_um) * L_um / a)
    )


def sigma_near_field_gain_um(
    setup: "SetupParams",
    *,
    lambda_s_nm: float | None = None,
    lambda_i_nm: float | None = None,
) -> float:
    """Gaussian-std of G_PSF: σ_G = Δ_G / √2."""
    return float(
        near_field_gain_1e_um(
            setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
        )
        / np.sqrt(2.0)
    )


def near_field_joint_gaussian_coefficients(
    setup: "SetupParams",
    *,
    lambda_s_nm: float | None = None,
    lambda_i_nm: float | None = None,
) -> dict[str, Any]:
    """Quadratic coefficients of P10 Eq. (9), lengths in µm.

    P(ρd,ρu) ∝ exp[−A |λu ρd + λd ρu|² − B |ρd − ρu|²]
    with A = 2 / [w_p² (λd+λu)²] and B = 4π / [L(λd+λu)].
    """
    L_um = validate_crystal_length_mm(float(setup.crystal_length_mm)) * 1.0e3
    lam_d_um = float(
        setup.lambda_s_nm if lambda_s_nm is None else lambda_s_nm
    ) * 1.0e-3
    lam_u_um = float(
        setup.lambda_i_nm if lambda_i_nm is None else lambda_i_nm
    ) * 1.0e-3
    if lam_d_um <= 0.0:
        raise ValueError(f"lambda_s_nm must be > 0, got {lam_d_um * 1.0e3}")
    if lam_u_um <= 0.0:
        raise ValueError(f"lambda_i_nm must be > 0, got {lam_u_um * 1.0e3}")
    w_um = _resolved_pump_waist_um(setup)
    lam_sum = lam_d_um + lam_u_um
    a = 2.0 * np.pi * w_um * w_um * lam_sum
    A = 2.0 / (w_um * w_um * lam_sum * lam_sum)
    B = 4.0 * np.pi / (L_um * lam_sum)
    alpha = A * lam_d_um * lam_d_um + B
    beta = A * lam_u_um * lam_u_um + B
    gamma = A * lam_d_um * lam_u_um - B
    return {
        "A_per_um2": float(A),
        "B_per_um2": float(B),
        "alpha_uu_per_um2": float(alpha),
        "beta_dd_per_um2": float(beta),
        "gamma_du_per_um2": float(gamma),
        "conditional_center_scale": float(-gamma / alpha),
        "lambda_d_um": float(lam_d_um),
        "lambda_u_um": float(lam_u_um),
        "crystal_length_um": float(L_um),
        "pump_waist_um": float(w_um),
        "a_wp_um3": float(a),
        "equation": "P10 Eq. 9 Gaussian approximation of sinc",
        "sinc_replaced_by_gaussian": True,
        "biphoton_wavefunction": False,
    }


def sigma_near_field_um(
    setup: "SetupParams",
    *,
    lambda_s_nm: float | None = None,
    lambda_i_nm: float | None = None,
) -> float:
    """Near-field Gaussian-std blur width [µm] at the crystal/object plane.

    Converts the P07/P10 1/e visibility half-width to the σ of
    exp(−r²/(2σ²)) used by ``gaussian_filter``: σ = Δ_V / √2.
    """
    return float(
        near_field_visibility_1e_um(
            setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
        )
        / np.sqrt(2.0)
    )


def equivalent_wavelength_nm(
    lambda_s_nm: float,
    lambda_i_nm: float,
) -> float:
    """P11 Eq. (4): λ_eq = λ_s² / λ_i [nm]."""
    ls = float(lambda_s_nm)
    li = float(lambda_i_nm)
    if ls <= 0.0 or li <= 0.0:
        raise ValueError("signal and idler wavelengths must be > 0")
    return ls * ls / li


def idler_lens_equivalent_propagation_um(
    delta_um: float,
    focal_mm: float,
) -> float:
    """P11 supplement: d = f² δ / (f² + δ²) [µm]."""
    f_um = float(focal_mm) * 1.0e3
    dlt = float(delta_um)
    denom = f_um * f_um + dlt * dlt
    if denom <= 0.0:
        raise ValueError("idler lens equivalent propagation denominator must be > 0")
    return float(f_um * f_um * dlt / denom)


def resolved_far_field_focal_mm(setup: "SetupParams") -> float:
    """Undetected-arm focal length f_u [mm] for far-field angular→spatial map."""
    fu = getattr(setup, "far_field_focal_mm", None)
    if fu is None:
        fu = getattr(setup, "F1_mm", F1_MM_DEFAULT)
    fu = float(fu)
    if fu <= 0.0:
        raise ValueError(f"far_field_focal_mm must be > 0, got {fu}")
    return fu


def sigma_far_field_um(
    setup: "SetupParams",
    *,
    lambda_i_nm: float | None = None,
    idler_relay_magnification: float | None = None,
) -> float:
    """Far-field Gaussian-std blur width [µm] in the object/Fourier plane.

    P14: σ_q = 1/w_p for I∝exp(−2r²/w_p²). P07 Eq. (2.30):
    ρ_o = m_i λ_i f_i q_i / (2π), so
        σ_o = m_i f_i λ_i / (2π w_p)
    Reduced-order intensity kernel (no biphoton Φ).
    """
    w_um = _resolved_pump_waist_um(setup)
    lam_i = float(setup.lambda_i_nm if lambda_i_nm is None else lambda_i_nm)
    lam_i_um = lam_i * 1.0e-3  # nm → µm
    if lam_i_um <= 0.0:
        raise ValueError(f"lambda_i_nm must be > 0, got {lam_i}")
    f_um = resolved_far_field_focal_mm(setup) * 1.0e3  # mm → µm
    if idler_relay_magnification is None:
        raw_m = getattr(setup, "far_field_idler_relay_magnification", None)
        m_i = 1.0 if raw_m is None else float(raw_m)
    else:
        m_i = float(idler_relay_magnification)
    return float(m_i * f_um * lam_i_um / (2.0 * np.pi * w_um))


def sigma_correlation_um(setup: "SetupParams") -> float:
    """Mode-dispatched correlation width [µm] for the QIUP spatial transfer kernel."""
    mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    if mode == IMAGING_MODE_FAR:
        return sigma_far_field_um(setup)
    return sigma_near_field_um(setup)


def resolved_spatial_mapping(
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the physical object-to-camera correlation coordinates.

    Near field uses ``rho_o=M_u rho_i`` and ``rho_c=M_s rho_s`` with
    position correlation, so ``M_near=M_s/M_u``. Far field uses
    ``rho_o=M_i lambda_i f_i q_i/(2 pi)`` and
    ``rho_c=M_s lambda_s f_s q_s/(2 pi)`` with ``q_s=-q_i``. The resulting
    far-field magnification is signed and records the inversion.

    The Gaussian width is a conditional-correlation width. It remains
    separate from the ordinary optical PSF applied after interference.
    """
    if (
        getattr(setup, "architecture_propagation_model", None)
        != ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        raise ValueError(
            "resolved_spatial_mapping requires the opt-in architecture model"
        )
    kernel = getattr(setup, "spatial_transfer_kernel", None)
    if kernel != SPATIAL_TRANSFER_CONDITIONAL_GAUSSIAN_V1:
        raise ValueError(
            "unsupported spatial transfer kernel; expected "
            f"{SPATIAL_TRANSFER_CONDITIONAL_GAUSSIAN_V1!r}"
        )
    p_object = float(setup.object_map_pixel_um)
    p_camera = internal_camera_sampling_pixel_um(setup)
    mode = normalize_imaging_mode(setup.imaging_mode)
    lambda_s_nm = float(
        phys_T.get("lambda_s_nm_T", setup.lambda_s_nm)
        if phys_T is not None else setup.lambda_s_nm
    )
    lambda_i_nm = float(
        phys_T.get("lambda_i_nm_T", setup.lambda_i_nm)
        if phys_T is not None else setup.lambda_i_nm
    )
    m_qiup = qiup_magnification(
        setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
    )

    common: dict[str, Any] = {
        "kernel": kernel,
        "imaging_mode": mode,
        "object_array_coordinates": "declared_object_plane",
        "camera_array_coordinates": "detector_plane",
        "object_plane_pixel_um": p_object,
        "camera_pixel_um": p_camera,
        "M_QIUP": m_qiup,
        "direct_camera_pixel_equivalence": False,
        "biphoton_wavefunction": False,
    }
    if mode == IMAGING_MODE_NEAR:
        m_u = float(setup.near_field_idler_magnification)
        m_s = float(setup.near_field_signal_magnification)
        signed_m = m_s / m_u
        sigma_object_um = sigma_near_field_um(
            setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
        )
        sigma_gain_object_um = sigma_near_field_gain_um(
            setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
        )
        sigma_camera_um = abs(m_s) * sigma_object_um
        joint = near_field_joint_gaussian_coefficients(
            setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
        )
        common.update({
            "correlation_type": "position",
            "near_field_idler_magnification_Mu": m_u,
            "near_field_signal_magnification_Ms": m_s,
            "correlation_relation": "rho_c=(M_s/M_u)*rho_o",
            "joint_probability": "P10 Eq. 9 Gaussian P(rho_d, rho_u)",
            "image_1e_width_um": float(
                near_field_gain_1e_um(
                    setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
                )
            ),
            "visibility_1e_width_um": float(
                near_field_visibility_1e_um(
                    setup, lambda_s_nm=lambda_s_nm, lambda_i_nm=lambda_i_nm
                )
            ),
            "correlation_sigma_gain_object_um": float(sigma_gain_object_um),
            "joint_gaussian_coefficients": joint,
            "conditional_kernel": (
                "G=∫ P(ρd,ρu) t(ρo) dρo; "
                "V=G/∫ P(ρd,ρu) dρo  (P10 Eqs 7-8)"
            ),
            "object_to_camera_index_map": "position_correlation_scaled",
        })
    else:
        m_i = float(setup.far_field_idler_relay_magnification)
        m_s = float(setup.far_field_signal_relay_magnification)
        f_i_mm = resolved_far_field_focal_mm(setup)
        f_s_mm = float(setup.F2_mm)
        signed_m = -(
            m_s * lambda_s_nm * f_s_mm
            / (m_i * lambda_i_nm * f_i_mm)
        )
        sigma_object_um = sigma_far_field_um(
            setup,
            lambda_i_nm=lambda_i_nm,
            idler_relay_magnification=m_i,
        )
        sigma_camera_um = abs(signed_m) * sigma_object_um
        q_i_step_per_m = (
            2.0 * np.pi * p_object * 1.0e-6
            / (m_i * lambda_i_nm * 1.0e-9 * f_i_mm * 1.0e-3)
        )
        common.update({
            "correlation_type": "momentum",
            "far_field_idler_focal_mm": f_i_mm,
            "far_field_signal_focal_mm": f_s_mm,
            "far_field_idler_relay_magnification": m_i,
            "far_field_signal_relay_magnification": m_s,
            "idler_transverse_wavevector_step_per_m": float(q_i_step_per_m),
            "correlation_relation": "q_signal=-q_idler",
            "coordinate_equations": (
                "rho_o=M_i*lambda_i*f_i*q_i/(2*pi); "
                "rho_c=M_s*lambda_s*f_s*q_s/(2*pi)"
            ),
            "object_to_camera_index_map": "momentum_anticorrelation_scaled",
        })

    index_scale = signed_m * p_object / p_camera
    common.update({
        "signed_object_to_camera_magnification": float(signed_m),
        "magnification_magnitude": float(abs(signed_m)),
        "orientation_sign": -1 if signed_m < 0.0 else 1,
        "object_to_camera_index_scale": float(index_scale),
        "correlation_sigma_object_um": float(sigma_object_um),
        "correlation_sigma_camera_um": float(sigma_camera_um),
        "correlation_sigma_camera_px": float(sigma_camera_um / p_camera),
        "pump_transverse_profile": pump_transverse_profile_evidence(setup),
        "spdc_correlation_psf": spdc_correlation_psf_status(),
        "camera_center_x_px": float(getattr(setup, "camera_center_x_px", 0.0) or 0.0),
        "camera_center_y_px": float(getattr(setup, "camera_center_y_px", 0.0) or 0.0),
    })
    return common


def spatial_object_um_to_camera_px(
    coordinate_um: float,
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> float:
    """Map an object coordinate relative to the optical axis to camera pixels."""
    mapping = resolved_spatial_mapping(setup, phys_T=phys_T)
    return float(
        float(coordinate_um)
        * mapping["signed_object_to_camera_magnification"]
        / mapping["camera_pixel_um"]
    )


def spatial_camera_px_to_object_um(
    coordinate_px: float,
    setup: "SetupParams",
    *,
    phys_T: dict[str, Any] | None = None,
) -> float:
    """Invert :func:`spatial_object_um_to_camera_px`."""
    mapping = resolved_spatial_mapping(setup, phys_T=phys_T)
    return float(
        float(coordinate_px)
        * mapping["camera_pixel_um"]
        / mapping["signed_object_to_camera_magnification"]
    )


def validate_setup_params(setup: "SetupParams") -> "SetupParams":
    """Validate user-editable physical knobs after YAML load or override."""
    # R03: Boolean fields fail closed before any numeric coercion or simulation.
    validate_setup_boolean_fields(setup)
    # Resolve the physics-team-approved blur-linked default before validating L.
    # ``None`` means the user omitted L; an explicit value (including 2 mm for
    # single_blur or 10 mm for double_blur) is always preserved.
    setup.blur_model = normalize_blur_model(getattr(setup, "blur_model", None))
    if setup.crystal_length_mm is None:
        setup.crystal_length_mm = default_crystal_length_mm_for_policy(
            getattr(setup, "physics_policy_id", None), setup.blur_model
        )
    # V-P1-08: fail-closed physics domain before any derived quantities.
    validate_physics_domain(setup)
    validate_pixel_um(setup.pixel_um)
    validate_pointing_alpha(setup.pointing_alpha)
    setup.crystal_length_mm = validate_crystal_length_mm(
        setup.crystal_length_mm
    )
    if setup.crystal_length_NL2_mm is not None:
        setup.crystal_length_NL2_mm = validate_crystal_length_mm(
            setup.crystal_length_NL2_mm
        )
    validate_poling_period_um(setup.poling_period_um)
    setup.qpm_order = validate_qpm_order(setup.qpm_order)
    if setup.qpm_order_NL2 is not None:
        setup.qpm_order_NL2 = validate_qpm_order(
            setup.qpm_order_NL2, name="qpm_order_NL2"
        )
    setup.qpm_duty_cycle = validate_qpm_duty_cycle(setup.qpm_duty_cycle)
    setup.QE = validate_quantum_efficiency(setup.QE)
    # Comment 3.1: never erase a direct d_eff when d33 is absent.
    resolve_effective_nonlinearity(setup)
    validate_L_coh_mm(setup.L_coh_mm)
    validate_delta_L_mm(setup.delta_L_mm)
    validate_n_crystals(setup.n_crystals)
    setup.spdc_architecture = normalize_spdc_architecture(
        getattr(setup, "spdc_architecture", None)
    )
    setup.imaging_mode = normalize_imaging_mode(
        getattr(setup, "imaging_mode", None)
    )
    validate_crystal_parameter_sets(setup)
    setup.object_preprocessing = normalize_object_preprocessing(
        getattr(setup, "object_preprocessing", None)
    )
    setup.spdc_gain_mode = normalize_spdc_gain_mode(
        getattr(setup, "spdc_gain_mode", None)
    )
    setup.spdc_source_mode = normalize_spdc_source_mode(
        getattr(setup, "spdc_source_mode", None)
    )
    setup.detector_mode = normalize_detector_mode(
        getattr(setup, "detector_mode", None)
    )
    validate_detector_configuration(setup)
    validate_architecture_propagation_fields(setup)
    sid = str(getattr(setup, "setup_id", ""))
    if sid == "config_c_lab_v1":
        w = getattr(setup, "y1_rabbit_aperture_width_mm", None)
        h = getattr(setup, "y1_rabbit_aperture_height_mm", None)
        if w is None or h is None:
            raise ValueError(
                "config_c_lab_v1 requires y1_rabbit_aperture_width_mm and "
                "y1_rabbit_aperture_height_mm (authoritative caliper aperture)."
            )
        w_f, h_f = float(w), float(h)
        if not (w_f < h_f):
            raise ValueError(
                "y1_rabbit_aperture_width_mm must be the shorter caliper axis "
                "(x / columns); height_mm is the longer axis (y / rows)."
            )
        setup.y1_rabbit_aperture_width_mm = w_f
        setup.y1_rabbit_aperture_height_mm = h_f
        setup.rabbit_object_flip_x = bool(getattr(setup, "rabbit_object_flip_x", False))
        setup.rabbit_object_flip_y = bool(getattr(setup, "rabbit_object_flip_y", False))
        setup.rabbit_object_rotate_180 = bool(getattr(setup, "rabbit_object_rotate_180", False))
    elif sid == "config_c_lab_joint_spdc_rabbit70_v1":
        # SPDC-designed synthetic rabbit; caliper Y1 sizes are not authoritative.
        model = getattr(setup, "c_lab_spatial_object_model", None) or "joint_spdc_rabbit70_v1"
        setup.c_lab_spatial_object_model = str(model)
        w = getattr(setup, "y1_rabbit_aperture_width_mm", None)
        h = getattr(setup, "y1_rabbit_aperture_height_mm", None)
        if w is not None and h is not None:
            setup.y1_rabbit_aperture_width_mm = float(w)
            setup.y1_rabbit_aperture_height_mm = float(h)
        setup.rabbit_object_flip_x = bool(getattr(setup, "rabbit_object_flip_x", False))
        setup.rabbit_object_flip_y = bool(getattr(setup, "rabbit_object_flip_y", False))
        setup.rabbit_object_rotate_180 = bool(getattr(setup, "rabbit_object_rotate_180", False))
        if abs(float(setup.far_field_idler_relay_magnification) - 1.0) > 1e-9:
            raise ValueError(
                "config_c_lab_joint_spdc_rabbit70_v1 baseline requires Mi=1 "
                "(do not use Mi=2 as a hidden resize)"
            )
        if abs(float(setup.far_field_signal_relay_magnification) - 1.0) > 1e-9:
            raise ValueError(
                "config_c_lab_joint_spdc_rabbit70_v1 baseline requires Ms=1"
            )
    # V-P0-03: gain mode owns the default nbar used when LatentState has no override.
    # Only applies under spdc_source_mode=fixed_nbar (V-P1-01 shortcut).
    setup.spdc_default_mean_photons_per_mode = spdc_gain_mode_nbar(
        setup.spdc_gain_mode
    )
    resolve_architecture_profile(setup)
    validate_spdc_energy_conservation(setup)
    if (setup.spdc_architecture == SPDC_ARCHITECTURE_SINGLE
            and setup.crystal_length_NL2_mm is not None
            and setup.crystal_length_NL2_mm != setup.crystal_length_mm):
        raise ValueError("single-crystal passes must share crystal_length_mm")
    if (setup.spdc_architecture == SPDC_ARCHITECTURE_SINGLE
            and setup.poling_period_NL2_um is not None
            and setup.poling_period_NL2_um != setup.poling_period_um):
        raise ValueError("single-crystal passes must share poling_period_um")
    if getattr(setup, "poling_period_NL2_um", None) is not None:
        setup.poling_period_NL2_um = validate_poling_period_um(
            setup.poling_period_NL2_um
        )
    if float(setup.L_eff_path_mm) <= 0.0:
        raise ValueError(f"L_eff_path_mm must be > 0, got {setup.L_eff_path_mm}")
    if float(setup.lens_focal_mm) <= 0.0:
        raise ValueError(f"lens_focal_mm must be > 0, got {setup.lens_focal_mm}")
    if float(setup.aperture_diameter_mm) <= 0.0:
        raise ValueError(
            f"aperture_diameter_mm must be > 0, got {setup.aperture_diameter_mm}"
        )
    if float(setup.F1_mm) <= 0.0 or float(setup.F2_mm) <= 0.0:
        raise ValueError("F1_mm and F2_mm must be > 0")
    if getattr(setup, "far_field_focal_mm", None) is not None:
        if float(setup.far_field_focal_mm) <= 0.0:
            raise ValueError(
                f"far_field_focal_mm must be > 0, got {setup.far_field_focal_mm}"
            )
    # Etalon / Fabry–Pérot fringe knobs (intensity-level; optional)
    validate_etalon_parameters(setup)
    validate_l07_acquisition_parameters(setup)
    if float(setup.airy_to_gauss_eta) <= 0.0:
        raise ValueError(
            f"airy_to_gauss_eta must be > 0, got {setup.airy_to_gauss_eta}"
        )
    if float(setup.n_medium) <= 0.0:
        raise ValueError(f"n_medium must be > 0, got {setup.n_medium}")
    if float(setup.eta_defocus) <= 0.0:
        raise ValueError(f"eta_defocus must be > 0, got {setup.eta_defocus}")
    if float(setup.delta_z_max_um) <= 0.0:
        raise ValueError(f"delta_z_max_um must be > 0, got {setup.delta_z_max_um}")
    if float(getattr(setup, "wavefront_rms_waves", 0.0)) < 0.0:
        raise ValueError(
            f"wavefront_rms_waves must be >= 0, got {setup.wavefront_rms_waves}"
        )
    if float(getattr(setup, "eta_aberration", 1.0)) <= 0.0:
        raise ValueError(f"eta_aberration must be > 0, got {setup.eta_aberration}")
    # PDF §6.4: NA = D/(2f) — always derived from lab aperture + focal length
    setup.NA = resolve_numerical_aperture(setup)
    # Quantize / validate crystal temperatures (PDF §5 eq. 80).
    setup.T_pm0_NL1_C = validate_crystal_temperature_C(
        setup.T_pm0_NL1_C, name="T_pm0_NL1_C"
    )
    setup.T_pm0_NL2_C = validate_crystal_temperature_C(
        setup.T_pm0_NL2_C, name="T_pm0_NL2_C"
    )
    setup.delta_T_NL1_C = quantize_temperature_C(setup.delta_T_NL1_C)
    setup.delta_T_NL2_C = quantize_temperature_C(setup.delta_T_NL2_C)
    resolve_crystal_operating_detuning(setup)
    if setup.sync_crystal_temperatures and not is_two_crystal_architecture(setup):
        setup.T_pm0_NL2_C = setup.T_pm0_NL1_C
        setup.delta_T_NL2_C = setup.delta_T_NL1_C
        setup.dT_NL2_C = setup.dT_NL1_C
        setup.T_crystal_NL2_C = setup.T_crystal_NL1_C
    if setup.spdc_architecture == SPDC_ARCHITECTURE_SINGLE:
        if (setup.T_pm0_NL2_C != setup.T_pm0_NL1_C
                or setup.delta_T_NL2_C != setup.delta_T_NL1_C
                or setup.dT_NL2_C != setup.dT_NL1_C
                or setup.T_crystal_NL2_C != setup.T_crystal_NL1_C):
            raise ValueError(
                "single-crystal passes must share physical crystal temperature"
            )
    # Legacy aliases mirror NL1 (and eq. 80 result).
    setup.T_crystal_C = setup.T_crystal_NL1_C
    setup.T_pm_C = phase_match_temperature_C(setup.T_pm0_NL1_C, setup.delta_T_NL1_C)
    # Keep deprecated αL product consistent with α·L when physical path knobs set.
    setup.alpha_L_mm_per_C = float(setup.alpha_path_per_C) * float(setup.L_eff_path_mm)
    # Touch eq. 80 / ΔL_eff so invalid combos fail early.
    crystal_thermal_state(setup)
    effective_delta_L_mm(setup)
    # V-P1-06: M_QIUP is SoT; preserve the requested YAML/API magnification for
    # Comment 7.4 / R09 provenance before syncing the legacy field.
    requested_mag = _validate_finite_real(
        getattr(setup, "magnification", 1.0), name="magnification"
    )
    if getattr(setup, "magnification_requested", None) is None:
        setup.magnification_requested = float(requested_mag)
    M_qiup = resolve_imaging_magnification(setup)
    setup.magnification = float(M_qiup)
    if setup.magnification <= 0:
        raise ValueError(f"M_QIUP/magnification must be > 0, got {setup.magnification}")
    # PDF §6.17: sync geometric scale from ε_M (never fold into σ_opt)
    setup.image_scale = resolve_image_scale(setup)
    t1 = validate_path_intensity(setup.path1_intensity, name="path1_intensity")
    t2 = validate_path_intensity(setup.path2_intensity, name="path2_intensity")
    if t1 + t2 <= 0.0:
        raise ValueError("path1_intensity + path2_intensity must be > 0")
    setup.idler_path_transmission = validate_path_intensity(
        getattr(setup, "idler_path_transmission", 1.0),
        name="idler_path_transmission",
    )
    n0 = float(setup.spdc_default_mean_photons_per_mode)
    nref = float(setup.spdc_reference_mean_photons_per_mode)
    nlow = float(setup.spdc_low_gain_max_mean_photons)
    nhigh = float(setup.spdc_high_gain_min_mean_photons)
    if not np.isfinite(n0) or n0 < 0.0:
        raise ValueError("spdc_default_mean_photons_per_mode must be finite and >= 0")
    if not np.isfinite(nref) or nref <= 0.0:
        raise ValueError("spdc_reference_mean_photons_per_mode must be finite and > 0")
    if not (0.0 <= nlow < nhigh):
        raise ValueError(
            "SPDC regime thresholds must satisfy "
            "0 <= spdc_low_gain_max_mean_photons < spdc_high_gain_min_mean_photons"
        )
    k_si = resolved_k_spdc_per_sqrt_W(setup)
    if k_si is not None and k_spdc_0_from_setup(setup) is not None:
        setup.spdc_gain_per_sqrt_mW = k_spdc_per_sqrt_mW_from_per_sqrt_W(k_si)
    if setup.spdc_gain_per_sqrt_mW is not None:
        k = float(setup.spdc_gain_per_sqrt_mW)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("spdc_gain_per_sqrt_mW must be finite and > 0 when set")
    if setup.spdc_gain_per_sqrt_mW_NL2 is not None:
        k2 = float(setup.spdc_gain_per_sqrt_mW_NL2)
        if not np.isfinite(k2) or k2 <= 0.0:
            raise ValueError(
                "spdc_gain_per_sqrt_mW_NL2 must be finite and > 0 when set"
            )
    setup.pump_pre_crystal_transmission = validate_pump_transmission(
        setup.pump_pre_crystal_transmission,
        name="pump_pre_crystal_transmission",
    )
    setup.pump_return_transmission = validate_pump_transmission(
        setup.pump_return_transmission, name="pump_return_transmission"
    )
    setup.pump_to_NL2_transmission = validate_pump_transmission(
        getattr(setup, "pump_to_NL2_transmission", 1.0),
        name="pump_to_NL2_transmission",
    )
    if getattr(setup, "pump_bs_reflectance", None) is not None:
        setup.pump_bs_reflectance = validate_pump_bs_reflectance(
            setup.pump_bs_reflectance
        )
    setup.two_crystal_pump_topology = normalize_two_crystal_pump_topology(
        getattr(setup, "two_crystal_pump_topology", None)
    )
    if setup.pump_between_crystals_transmission is not None:
        setup.pump_between_crystals_transmission = validate_pump_transmission(
            setup.pump_between_crystals_transmission,
            name="pump_between_crystals_transmission",
        )
    if setup.pump_power_laser_mW is not None:
        resolved_pump_laser_mW(setup, setup.pump_power_laser_mW)
    if setup.pump_power_NL2_override_mW is not None:
        if not setup.use_independent_pump_power_NL2:
            raise ValueError(
                "pump_power_NL2_override_mW requires "
                "use_independent_pump_power_NL2=True "
                "(single-crystal: P_p2 from the return path; "
                "two_crystal: P_p2 from (1-R_BS)*pump_to_NL2_transmission*P_laser)"
            )
        p2 = float(setup.pump_power_NL2_override_mW)
        if not np.isfinite(p2) or p2 < 0.0:
            raise ValueError(
                "pump_power_NL2_override_mW must be finite and >= 0"
            )
        if not str(
            getattr(setup, "pump_power_NL2_override_calibration_id", "") or ""
        ).strip():
            raise ValueError(
                "independent NL2/backward pump override requires "
                "pump_power_NL2_override_calibration_id"
            )
    validate_calibrated_source_requirements(setup)
    validate_physics_team_refinements(setup)
    if setup.use_legacy_beam_match:
        return setup
    if setup.pump_waist_um is None:
        raise ValueError(
            "use_legacy_beam_match=False requires pump_waist_um "
            f"in [{PUMP_WAIST_UM_MIN}, {PUMP_WAIST_UM_MAX}] µm"
        )
    validate_pump_waist_um(setup.pump_waist_um)
    if setup.pump_waist_NL1_um is not None:
        validate_pump_waist_um(setup.pump_waist_NL1_um)
    if setup.pump_waist_NL2_um is not None:
        validate_pump_waist_um(setup.pump_waist_NL2_um)
    if setup.pump_profile_type is not None:
        setup.pump_profile_type = normalize_pump_profile_type(setup.pump_profile_type)
    for name in ("pump_center_x_um", "pump_center_y_um",
                 "camera_center_x_px", "camera_center_y_px"):
        value = getattr(setup, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real number, got {value!r}")
        setattr(setup, name, float(value))
    if setup.apparatus_crystal_NL1 is not None and not isinstance(
        setup.apparatus_crystal_NL1, dict
    ):
        raise ValueError("apparatus_crystal_NL1 must be a mapping or None")
    if setup.apparatus_crystal_NL2 is not None and not isinstance(
        setup.apparatus_crystal_NL2, dict
    ):
        raise ValueError("apparatus_crystal_NL2 must be a mapping or None")
    if setup.calibration_records is not None and not isinstance(
        setup.calibration_records, dict
    ):
        raise ValueError("calibration_records must be a mapping or None")
    # B/C: one physical crystal — reject a distinct NL2 apparatus record.
    if (
        not is_two_crystal_architecture(setup)
        and setup.apparatus_crystal_NL2 is not None
    ):
        raise ValueError(
            "single_crystal_double_pass uses one physical crystal; "
            "apparatus_crystal_NL2 must remain null"
        )
    if setup.idler_waist_um is not None:
        validate_idler_waist_um(setup.idler_waist_um)
    if setup.idler_waist_2_um is not None:
        validate_idler_waist_um(setup.idler_waist_2_um)
    return setup


def um_to_px(length_um: float, *, pixel_um: float, magnification: float = 1.0) -> float:
    return float(magnification * length_um / pixel_um)


def px_to_um(length_px: float, *, pixel_um: float, magnification: float = 1.0) -> float:
    """Inverse of um_to_px: pixel displacement → lab µm."""
    M = max(float(magnification), 1e-12)
    return float(length_px) * float(pixel_um) / M


def two_beam_visibility(I1: float, I2: float) -> float:
    """Two-path fringe visibility (PDF §4.2 eq. 70).

    V = 2 √(I1 I2) / (I1 + I2)

    ``I1`` and ``I2`` are the two interferometer-path intensities (relative
    transmittances are fine).  Not derived from a ratio ``r`` or from
    ``s_align``.
    """
    a = max(float(I1), 0.0)
    b = max(float(I2), 0.0)
    s = a + b
    if s <= 0.0:
        return 0.0
    return float(2.0 * np.sqrt(a * b) / s)


def validate_path_intensity(value: float, *, name: str) -> float:
    """Path intensity transmission in [0, 1] (V-P1-08)."""
    return validate_transmission(value, name=name)


def signal_path_transmissions_from_setup(setup: "SetupParams") -> tuple[float, float]:
    """ηs1, ηs2 (report §4.7 / §7.18). Stored as path1/path2_intensity."""
    return (
        validate_path_intensity(setup.path1_intensity, name="path1_intensity"),
        validate_path_intensity(setup.path2_intensity, name="path2_intensity"),
    )


def idler_path_transmission_from_setup(setup: "SetupParams") -> float:
    """ηi,path (report §4.4 / §7.15). Default 1: no extra idler-path loss."""
    return validate_path_intensity(
        getattr(setup, "idler_path_transmission", 1.0),
        name="idler_path_transmission",
    )


SETUP_SWEEP_ALIASES: dict[str, str] = {
    "signal_path1_transmission": "path1_intensity",
    "signal_path2_transmission": "path2_intensity",
    "eta_i_path": "idler_path_transmission",
}


# Comment 4.4 — explicit latent allow-list (cluster s_* + LatentState knobs).
# SETUP_SWEEP_FIELDS is assigned after SetupParams is defined (see below).
LATENT_SWEEP_FIELDS: frozenset[str] = frozenset(
    {
        "s_pump",
        "s_gain",
        "s_align",
        "s_thermal",
        "s_det",
        "s_cool",
        "s_focus",
        "s_slm",
        # LatentState / detector knobs (not SetupParams fields)
        "em_gain",
        "exposure_s",
        "mean_photons_per_mode",
        "spdc_gain_r",
        "mean_photons_per_mode_NL1",
        "mean_photons_per_mode_NL2",
        "spdc_gain_r_NL1",
        "spdc_gain_r_NL2",
        "pump_power_mW",
        "photon_counting",
    }
)

# Populated after SetupParams dataclass exists.
SETUP_SWEEP_FIELDS: frozenset[str] = frozenset()


def _compute_setup_sweep_fields() -> frozenset[str]:
    return frozenset({f.name for f in fields(SetupParams)}) | frozenset(
        SETUP_SWEEP_ALIASES.keys()
    )


def validate_sweep_names(
    sweeps: dict[str, Any] | None,
    *,
    context: str = "sweep",
) -> None:
    """Fail-closed sweep-name gate (Comments 4.4 / 4.5).

    Rejects unknown names before simulation or directory creation.
    """
    if sweeps is None:
        return
    if not isinstance(sweeps, dict):
        raise ConfigError(
            f"{context}: expected a mapping of sweep names → value lists, "
            f"got {type(sweeps).__name__}"
        )
    allowed = SETUP_SWEEP_FIELDS | LATENT_SWEEP_FIELDS
    unknown = sorted({str(name) for name in sweeps if str(name) not in allowed})
    if unknown:
        raise ConfigError(
            f"Unknown sweep variable: {unknown[0]}"
            + (f" (also: {unknown[1:]})" if len(unknown) > 1 else "")
        )


def setup_overrides_from_sweep(params: dict[str, Any]) -> dict[str, Any]:
    """Map dataset/sweep keys onto real SetupParams fields (report §7.24).

    Comment 4.4 — fail-closed::

        if name in SETUP_SWEEP_FIELDS: → setup override
        elif name in LATENT_SWEEP_FIELDS: → latent (skipped here)
        else: raise ConfigError
    """
    out: dict[str, Any] = {}
    for key, value in params.items():
        name = str(key)
        if name in SETUP_SWEEP_FIELDS:
            field_name = SETUP_SWEEP_ALIASES.get(name, name)
            out[field_name] = value
        elif name in LATENT_SWEEP_FIELDS:
            continue
        else:
            raise ConfigError(f"Unknown sweep variable: {name}")
    if "T_crystal_C" in out:
        out.setdefault("T_crystal_NL1_C", out["T_crystal_C"])
        out.setdefault("T_crystal_NL2_C", out["T_crystal_C"])
    return out


def gaussian_field_amplitude(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    cx: float,
    cy: float,
    w_px: float,
) -> np.ndarray:
    """Field amplitude envelope E ∝ exp(-r²/w²) on a pixel grid."""
    w2 = max(float(w_px), 1e-6) ** 2
    return np.exp(-(((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / w2)).astype(
        np.float32
    )


def mode_tilt_factor(
    w1_um: float,
    w2_um: float,
    *,
    dtheta_mrad: float,
    lambda_i_nm: float,
) -> float:
    """Angular-mismatch factor from PDF §4 / eq. 63 (scalar)."""
    w1 = max(float(w1_um), 1e-6)
    w2 = max(float(w2_um), 1e-6)
    den = w1 * w1 + w2 * w2
    dtheta = float(dtheta_mrad) * 1e-3
    lam_um = max(float(lambda_i_nm), 1e-9) * 1e-3
    dk = 2.0 * np.pi * dtheta / lam_um
    return float(np.exp(-(w1 * w1 * w2 * w2) * (dk * dk) / (2.0 * den)))


def field_overlap_visibility_map(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    cx1: float,
    cy1: float,
    w1_px: float,
    cx2: float,
    cy2: float,
    w2_px: float,
    tilt_factor: float = 1.0,
) -> np.ndarray:
    """Spatial visibility from two fields (PDF §4.3 eq. 74).

    Builds amplitudes A_j = |E_j| with Gaussian envelopes, then the local
    two-beam form (consistent with eq. 70):

      V(x,y) = 2 A1 A2 / (A1² + A2²) · tilt_factor

    which equals |E1 E2*| / ((|E1|²+|E2|²)/2) when phases cancel in the
    magnitude product.  Replaces engineered mottling (eq. 73).
    """
    A1 = gaussian_field_amplitude(xx, yy, cx=cx1, cy=cy1, w_px=w1_px)
    A2 = gaussian_field_amplitude(xx, yy, cx=cx2, cy=cy2, w_px=w2_px)
    num = 2.0 * A1 * A2
    den = A1 * A1 + A2 * A2
    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(den > 1e-30, num / den, 0.0)
    return np.clip(V * float(np.clip(tilt_factor, 0.0, 1.0)), 0.0, 1.0).astype(
        np.float32
    )


def numerical_aperture_from_aperture(
    aperture_diameter_mm: float,
    focal_length_mm: float,
) -> float:
    """PDF §6.4: NA = D / (2 f)  (paraxial air, lab aperture + imaging lens)."""
    D = float(aperture_diameter_mm)
    f = float(focal_length_mm)
    if D <= 0.0 or f <= 0.0:
        raise ValueError(
            f"aperture_diameter_mm and focal_length_mm must be > 0, got D={D}, f={f}"
        )
    return float(D / (2.0 * f))


def resolve_numerical_aperture(setup: "SetupParams") -> float:
    """NA from lab aperture D and imaging focal length f."""
    f_mm = float(getattr(setup, "lens_focal_mm", LENS_FOCAL_MM_DEFAULT))
    D_mm = float(
        getattr(setup, "aperture_diameter_mm", APERTURE_DIAMETER_MM_DEFAULT)
    )
    return numerical_aperture_from_aperture(D_mm, f_mm)


def airy_radius_um(*, lambda_s_nm: float, NA: float) -> float:
    """PDF §6.4 eq. 142: r_Airy = 0.61 λ_s / NA  [µm]."""
    na = float(NA)
    if na <= 0.0:
        raise ValueError(f"NA must be > 0, got {NA}")
    return float(0.61 * (float(lambda_s_nm) * 1e-3) / na)


def airy_sigma_px(
    *,
    lambda_s_nm: float,
    NA: float,
    pixel_um: float,
    magnification: float = 1.0,
    eta: float = 1.0,
) -> float:
    """PDF §6.4 eq. 144: σ_diffraction = η · M · r_Airy / p  [px].

    r_Airy from eq. 142.  ``eta`` (default 1) is an optional Airy→Gaussian
    calibration scale — not a first-principles constant.
    """
    if float(pixel_um) <= 0.0:
        raise ValueError(f"pixel_um must be > 0, got {pixel_um}")
    r_um = airy_radius_um(lambda_s_nm=lambda_s_nm, NA=NA)
    return um_to_px(
        float(eta) * r_um,
        pixel_um=float(pixel_um),
        magnification=float(magnification),
    )


def sigma_diffraction_from_setup(
    setup: "SetupParams",
    *,
    magnification: float | None = None,
) -> float:
    """Eq. 144 using lab knobs: λ_s, NA=D/(2f), p, M_QIUP (§5.7), η."""
    NA = resolve_numerical_aperture(setup)
    if magnification is None:
        M = resolve_imaging_magnification(setup)
    else:
        M = float(magnification)
    return airy_sigma_px(
        lambda_s_nm=float(setup.lambda_s_nm),
        NA=NA,
        pixel_um=internal_camera_sampling_pixel_um(setup),
        magnification=M,
        eta=float(setup.airy_to_gauss_eta),
    )


def circle_of_confusion_um(
    *,
    delta_z_um: float,
    NA: float,
    n_medium: float = N_MEDIUM_DEFAULT,
) -> float:
    """PDF §6.7: c ≈ (NA / n) · Δz  [µm]."""
    n = float(n_medium)
    if n <= 0.0:
        raise ValueError(f"n_medium must be > 0, got {n_medium}")
    return float(abs(float(NA) / n * float(delta_z_um)))


def K_defocus_from_optics(
    *,
    NA: float,
    magnification: float,
    pixel_um: float,
    n_medium: float = N_MEDIUM_DEFAULT,
    eta_defocus: float = ETA_DEFOCUS_DEFAULT,
) -> float:
    """PDF §6.7: K_defocus = η M NA / (n p)  [px/µm].

    With σ_defocus = K · |Δz| and c = (NA/n)·Δz this is
    σ = η · M · c / p.
    """
    p = float(pixel_um)
    n = float(n_medium)
    eta = float(eta_defocus)
    if p <= 0.0:
        raise ValueError(f"pixel_um must be > 0, got {pixel_um}")
    if n <= 0.0:
        raise ValueError(f"n_medium must be > 0, got {n_medium}")
    if eta <= 0.0:
        raise ValueError(f"eta_defocus must be > 0, got {eta_defocus}")
    if float(NA) <= 0.0:
        raise ValueError(f"NA must be > 0, got {NA}")
    return float(eta * float(magnification) * float(NA) / (n * p))


def resolve_K_defocus(
    setup: "SetupParams",
    *,
    magnification: float | None = None,
) -> float:
    """Physical K(NA,M,p,n,η) when ``use_derived_K_defocus``; else YAML K."""
    if not bool(getattr(setup, "use_derived_K_defocus", False)):
        return float(setup.K_defocus_px_per_um)
    if magnification is None:
        M = resolve_imaging_magnification(setup)
    else:
        M = float(magnification)
    return K_defocus_from_optics(
        NA=resolve_numerical_aperture(setup),
        magnification=M,
        pixel_um=internal_camera_sampling_pixel_um(setup),
        n_medium=float(getattr(setup, "n_medium", N_MEDIUM_DEFAULT)),
        eta_defocus=float(getattr(setup, "eta_defocus", ETA_DEFOCUS_DEFAULT)),
    )


def sigma_defocus_from_delta_z(
    delta_z_um: float,
    K_defocus_px_per_um: float,
) -> float:
    """PDF §6.7: σ_defocus = K_defocus · |Δz|  [px]."""
    return float(abs(float(K_defocus_px_per_um) * float(delta_z_um)))


def s_focus_from_delta_z(
    delta_z_um: float,
    delta_z_max_um: float,
) -> float:
    """PDF §6.10 / twin map: s_focus = |Δz| / Δz_max ∈ [0, 1]."""
    zmax = abs(float(delta_z_max_um))
    if zmax <= 0.0:
        raise ValueError(f"delta_z_max_um must be > 0, got {delta_z_max_um}")
    return float(np.clip(abs(float(delta_z_um)) / zmax, 0.0, 1.0))


def resolve_delta_z_um(
    setup: "SetupParams",
    *,
    s_focus: float = 0.0,
    phys_T: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Resolve axial focus error Δz [µm] for C4↔C7 (PDF §6.9–6.10).

    Physical chain (eq. 160), never eq. 162 (``s_focus += β_T s_thermal``):

      ΔT → Δn, ΔL → Δf → Δz_focus → σ_defocus

    Rules:
      * ``use_physical_thermal``: ``Δz = Δf(T) + Δz_mech``
        where ``Δz_mech = delta_z_um`` if set, else 0.
      * else if ``delta_z_um`` set: absolute mechanical Δz.
      * else (general synthetic): ``Δz = s_focus · Δz_max`` (inverse of eq. 166).
    """
    mech = float(setup.delta_z_um) if setup.delta_z_um is not None else 0.0
    has_mech = setup.delta_z_um is not None

    if phys_T is not None and bool(setup.use_physical_thermal):
        df = float(phys_T["lens_delta_f_um"])
        delta_z = float(df + mech)
        if has_mech:
            return delta_z, "eq160_df_plus_mech"
        return delta_z, "eq160_thermal_df"

    if has_mech:
        return float(mech), "explicit_delta_z"

    # General / legacy sampling only — not a thermal coupling (eq. 162 unused).
    return (
        float(s_focus) * float(setup.delta_z_max_um),
        "eq166_inverse_s_focus_scaled",
    )


def wavefront_rms_rad(wavefront_rms_waves: float) -> float:
    """Convert OPD RMS in waves → radians: δφ = 2π W_waves."""
    return float(2.0 * np.pi * float(wavefront_rms_waves))


def sigma_aberration_from_wavefront(
    *,
    wavefront_rms_waves: float,
    lambda_s_nm: float,
    NA: float,
    pixel_um: float,
    magnification: float,
    eta_aberration: float = ETA_ABERRATION_DEFAULT,
) -> float:
    """PDF §6.13/6.15: effective residual blur from wavefront RMS (no Zernike).

    σ_aberration = η_ab · W_waves · (M · r_Airy / p)  [px]

    ``W_waves`` is lab OPD RMS in waves at λ_s.  η_ab is a calibration scale
    (default 1 = residual blur proportional to the diffraction Airy width).
    """
    W = float(wavefront_rms_waves)
    if W <= 0.0:
        return 0.0
    r_um = airy_radius_um(lambda_s_nm=lambda_s_nm, NA=NA)
    return um_to_px(
        float(eta_aberration) * W * r_um,
        pixel_um=validate_pixel_um(pixel_um),
        magnification=float(magnification),
    )


def resolve_sigma_aberration_px(
    setup: "SetupParams",
    *,
    magnification: float | None = None,
) -> float:
    """Derived σ_aberr from ``wavefront_rms_waves``, else explicit YAML px."""
    if not bool(getattr(setup, "use_derived_aberration", True)):
        return float(getattr(setup, "sigma_aberration_px", 0.0) or 0.0)
    if magnification is None:
        M = resolve_imaging_magnification(setup)
    else:
        M = float(magnification)
    return sigma_aberration_from_wavefront(
        wavefront_rms_waves=float(getattr(setup, "wavefront_rms_waves", 0.0)),
        lambda_s_nm=float(setup.lambda_s_nm),
        NA=resolve_numerical_aperture(setup),
        pixel_um=validate_pixel_um(setup.pixel_um),
        magnification=M,
        eta_aberration=float(getattr(setup, "eta_aberration", ETA_ABERRATION_DEFAULT)),
    )


def resolve_c3_fan_delta_phi_rms(
    setup: "SetupParams",
    *,
    s_align: float = 0.0,
    delta_phi_rms_extra: float = 0.0,
) -> tuple[float, float, str]:
    """C3 path-interference Fan-map RMS [rad] and correlation length [px].

    δφ_rms = 2π·W_waves + (δφ_align_span·s_align if coupled) + extra

    Static lab wavefront feeds both imaging σ_aberration and base Fan phase;
    alignment (``s_align``) adds path-phase disturbance — not Zernike PSF.
    """
    W_rad = wavefront_rms_rad(float(getattr(setup, "wavefront_rms_waves", 0.0)))
    corr = float(
        getattr(setup, "wavefront_corr_px", WAVEFRONT_CORR_PX_DEFAULT)
    )
    if bool(getattr(setup, "couple_align_to_wavefront", True)):
        align_term = float(getattr(setup, "delta_phi_align_span_rad", 0.0)) * float(
            s_align
        )
        mode = "c3_wavefront_plus_align"
    else:
        align_term = 0.0
        mode = "c3_wavefront_static"
    total = float(W_rad + align_term + max(0.0, float(delta_phi_rms_extra)))
    return total, corr, mode


def c4_c7_focus_audit(
    setup: "SetupParams",
    *,
    delta_z_um: float,
    delta_z_mode: str,
    K_defocus_px_per_um: float,
    sigma_defocus_px: float,
    s_focus_eq166: float,
    phys_T: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Document C4→C7 link: eq.160 + eq.164 + eq.166 (eq.162 not used)."""
    dT = float(phys_T["dT_eff_C"]) if phys_T is not None else 0.0
    df = float(phys_T["lens_delta_f_um"]) if phys_T is not None else 0.0
    mech = float(setup.delta_z_um) if setup.delta_z_um is not None else 0.0
    return {
        "c4_c7_chain": "eq160+eq164+eq166",
        "eq160": "dT -> dn,dL -> df -> dz_focus -> sigma_defocus",
        "eq164": "sigma_defocus = K_defocus * |dz|",
        "eq166": "s_focus = |dz| / dz_max",
        "eq162_used": False,
        "eq162_note": "phenomenological s_focus=s0+beta_T*s_thermal NOT defined",
        "delta_z_mode": delta_z_mode,
        "dT_eff_C": dT,
        "lens_delta_f_um": df,
        "delta_z_mech_um": mech if setup.delta_z_um is not None else None,
        "delta_z_um": float(delta_z_um),
        "K_defocus_px_per_um": float(K_defocus_px_per_um),
        "sigma_defocus_px": float(sigma_defocus_px),
        "s_focus_eq166": float(s_focus_eq166),
        "delta_z_max_um": float(setup.delta_z_max_um),
    }


@dataclass
class SetupParams:
    """Physical / engineering setup for a QIUP digital twin.

    Defaults use a **physical** pump profile: ``pump_waist_um=250`` µm (range
    100–1000), ``pixel_um`` from the EMCCD model, and ``pointing_alpha`` jitter
    relative to waist.  Call ``legacy_matched()`` only for Phase-0 regression.
    """

    # Task 11.5 policy is opt-in so historical profiles keep byte/numeric
    # compatibility. The four registered A/B/C/D profiles select v2 explicitly.
    physics_policy_id: str | None = None

    # --- geometry / imaging (L1,L2,L6,L7) ---
    # V-P1-06: M_QIUP = F2·λ_s/(F1·λ_i) is the magnification SoT.
    # ``magnification`` is kept as a synced mirror of M_QIUP (not an independent knob).
    magnification: float = 1.0
    # Captured before M_QIUP sync so Comment 7.4 can show requested vs resolved.
    magnification_requested: float | None = None
    pixel_um: float = PIXEL_UM_DEFAULT  # EMCCD pitch [µm]; YAML-editable [6.5, 24]
    lambda_s_nm: float = 810.0
    lambda_i_nm: float = 1550.0
    # PDF §6.4: NA = D/(2f). ``NA`` is synced from D,f in validate_setup_params.
    aperture_diameter_mm: float = APERTURE_DIAMETER_MM_DEFAULT
    airy_to_gauss_eta: float = AIRY_TO_GAUSS_ETA_DEFAULT  # η in eq. 144 (default 1)
    NA: float = NA_MATCH_LEGACY  # derived cache; do not treat as independent lab knob

    # Pump beam (L1/L2): physical waist in µm; YAML-editable [100, 1000].
    pump_waist_um: float | None = PUMP_WAIST_UM_DEFAULT
    # Per-pass geometry (report §5.4 / §7.7 / §8). None → inherit pump_waist_um / NL1.
    pump_waist_NL1_um: float | None = None
    pump_waist_NL2_um: float | None = None
    pump_focus_z_NL1_um: float = 0.0  # waist vs crystal centre
    pump_focus_z_NL2_um: float | None = None
    # Step-2 pump transverse profile (spatial distribution ≠ total power).
    pump_profile_type: str | None = None  # None → gaussian
    pump_center_x_um: float = 0.0
    pump_center_y_um: float = 0.0
    use_legacy_beam_match: bool = False

    # Pointing (L3 / PDF §2.4): σ_beam = α·w_px; c = c0 + Δ, Δ~N(0,(s_pump·σ_beam)²)
    pointing_alpha: float = POINTING_ALPHA_DEFAULT
    # PDF §2.6: η_overlap(Δx,Δy) couples pointing jitter → visibility (not flux, not drift)
    couple_pointing_overlap_visibility: bool = True

    # Pump flux scale: flux0 is calibrated at spdc_reference_mean_photons_per_mode.
    # s_pump controls only instability/pointing, never mean power.
    flux0_e_per_s: float = 40.0
    pump_sigma_rel_at_s1: float = 0.30
    flux_boost_at_s1: float = 1.4
    # Report §7.12 / eq. 52–53. No numeric lab P or η is given in the report;
    # transmissions default to 1 (no extra loss). Laser power is None until set.
    # P_p2 is derived from the return path; not a free field unless the
    # explicit experimental flag is on.
    pump_power_laser_mW: float | None = None
    pump_pre_crystal_transmission: float = 1.0
    pump_return_transmission: float = 1.0
    # two_crystal only: explicit topology is required for calibrated_source.
    # Branched passive BS (L03): P1=R_BS*η_pre*P_laser,
    # P2=(1-R_BS)*η_NL2*P_laser. Sequential: P2=η_between*P1.
    two_crystal_pump_topology: str | None = None
    pump_to_NL2_transmission: float = 1.0
    # Passive BS intensity reflectance toward NL1. NL2 gets (1-R). Used only
    # for two-crystal branched topology (L03). None → 0.5 when branched.
    # Ignored for B/C return pass and sequential two-crystal.
    pump_bs_reflectance: float | None = None
    pump_between_crystals_transmission: float | None = None
    use_independent_pump_power_NL2: bool = False
    pump_power_NL2_override_mW: float | None = None
    pump_power_NL2_override_calibration_id: str | None = None
    # Reference target and actual propagation are distinct. Targets are
    # diagnostics/experimental objectives and never overwrite local powers.
    pump_balance_mode: str | None = None
    pump_target_power_NL1_mW: float | None = None
    pump_target_power_NL2_mW: float | None = None

    # Physical SPDC gain (Wiseman–Mølmer). Default architecture is the paper
    # path: one crystal, two passes. ``n_crystals`` is hardware inventory.
    # Operating-point priority is: LatentState.mean_photons_per_mode, then
    # LatentState.spdc_gain_r, then P_p1 from P_p,laser (latent or setup),
    # then this default. LatentState.pump_power_mW is P_p,laser (§7.12).
    use_physical_spdc_gain: bool = True
    spdc_architecture: str = SPDC_ARCHITECTURE_DEFAULT
    # Intensity-level imaging plane (QIUP): near_field | far_field.
    # Changes the correlation PSF kernel before interference; not a Φ biphoton model.
    imaging_mode: str = IMAGING_MODE_DEFAULT
    # Undetected-arm focal length f_u [mm] for far-field σ = m_i f_u λ_i/(2π w_p).
    # None → use F1_mm (Lemos-like default 75 mm).
    far_field_focal_mm: float | None = None
    # Opt-in architecture-specific complex-field path. ``None`` retains the
    # historical reduced-order propagation and old configuration hashes.
    architecture_propagation_model: str | None = None
    # Conditional-correlation coordinate kernel for the opt-in A/B/C/D path.
    # Object samples and both relay magnifications are explicit so a reference
    # array is never silently treated as camera pixels.
    spatial_transfer_kernel: str | None = None
    object_map_pixel_um: float | None = None
    # Opt-in U14 / Figure 3(e): refine internal grid while holding physical detector ROI.
    numerical_verification_internal_camera_pixel_um: float | None = None
    fixed_fov_detector_output_n: int | None = None
    near_field_idler_magnification: float | None = None
    near_field_signal_magnification: float | None = None
    far_field_idler_relay_magnification: float | None = None
    far_field_signal_relay_magnification: float | None = None
    # Object fields use reference-array selectors in this version. For a
    # reflective object, ref.T is interpreted as intensity reflectance R.
    object_type: str | None = None
    object_device: str | None = None
    object_mode: str | None = None
    object_mask_id: int | None = None
    # Intensity coefficients: T for a transmissive (two-crystal) object,
    # R for a reflective (single-crystal) object. Phase is radians in [0, 2π].
    object_intensity_T: float | None = None
    object_intensity_R: float | None = None
    object_phase_rad: float | None = None
    object_plane: str | None = None
    object_transmission_map: str | None = None
    object_phase_map: str | None = None
    object_reflection_amplitude: float | None = None
    object_reflection_phase: float | None = None
    object_forward_transform: str | None = None
    object_backward_transform: str | None = None
    object_interaction_count: int | None = None
    object_phase_map_path: str | None = None
    object_amplitude_map_path: str | None = None
    object_intensity_map_path: str | None = None
    object_phase_units: str | None = None
    object_intensity_convention: str | None = None
    object_coordinate_orientation: str | None = None
    # C_lab Y1 rabbit caliper aperture (authoritative for production mapping).
    y1_rabbit_aperture_width_mm: float | None = None
    y1_rabbit_aperture_height_mm: float | None = None
    # C_lab object-plane orientation of frozen Y1 mask (display/mapping only; not SPDC).
    rabbit_object_flip_x: bool = False
    rabbit_object_flip_y: bool = False
    rabbit_object_rotate_180: bool = False
    # Opt-in spatial object pathway selector (legacy default: None / Y1 caliper).
    c_lab_spatial_object_model: str | None = None
    rabbit70_mask_npz: str | None = None
    # External idler round-trip loss is an intensity loss fraction. It stays
    # separate from the object coefficient and relay mode matching.
    idler_round_trip_loss: float | None = None
    idler_mode_matching: float | None = None
    idler_return_phase: float | None = None
    relative_interferometric_phase: float | None = None
    # L07: explicit phase-step protocol. Off by default; num_frames is not a
    # substitute. None + enabled uses FOUR_STEP_PSI_SCHEDULE_RAD.
    enable_phase_step_acquisition: bool = False
    phase_step_schedule_rad: tuple[float, ...] | None = None
    # P04-style additive optical background at the detector plane
    # [photons / pixel / exposure], applied before QE. Default 0.
    optical_background_photons: float = 0.0
    # Explicit single-crystal signal return state. Reflection amplitude is a
    # field coefficient; the other transmission/overlap terms are intensities.
    signal_return_path_transmission: float | None = None
    signal_return_reflection_amplitude: float | None = None
    signal_return_phase: float | None = None
    signal_return_mode_overlap: float | None = None
    signal_return_seeds_second_interaction: bool | None = None
    # V-P0-01 blur chain switch (independent of high/low gain).
    # single_blur | double_blur — see BLUR_MODEL_* constants.
    blur_model: str = BLUR_MODEL_DEFAULT
    # V-P0-08/09: object/SLM preprocessing. calibrated keeps physical units;
    # visualization_only may min-max normalize (display/demo only).
    object_preprocessing: str = OBJECT_PREPROCESS_DEFAULT
    # V-P0-03 gain regime switch. Sets default nbar (0.001 low / 5 high).
    # LatentState / PhysicalParams overrides still win at the operating point
    # when spdc_source_mode=fixed_nbar (dataset shortcut).
    spdc_gain_mode: str = SPDC_GAIN_MODE_DEFAULT
    # V-P1-01: how nbar is obtained — fixed_nbar (requested) or calibrated_source
    # (pump → SPDC model → nbar). Never silently keep nbar constant under
    # calibrated_source while pump/T/QPM change.
    spdc_source_mode: str = SPDC_SOURCE_DEFAULT
    fixed_source_input: str | None = None
    measured_power_W: float | None = None
    measured_power_wavelength_nm: float | None = None
    measured_power_to_nbar_per_W: float | None = None
    measured_power_to_nbar_per_W_NL2: float | None = None
    measured_power_to_nbar_calibration_id: str | None = None
    spdc_default_mean_photons_per_mode: float = SPDC_GAIN_MODE_LOW_NBAR
    spdc_reference_mean_photons_per_mode: float = 0.01
    # r = k √P_W. Physical Mode: k = kSPDC,0 · O_geom (report §7.5 / §7.7).
    # Manual value remains only as explicit fallback when deff/waist is missing.
    spdc_gain_per_sqrt_mW: float | None = None
    spdc_gain_per_sqrt_mW_NL2: float | None = None
    spdc_gain_calibration_id: str | None = None
    spdc_gain_calibration_id_NL2: str | None = None
    spdc_low_gain_max_mean_photons: float = 0.01
    spdc_high_gain_min_mean_photons: float = 1.0

    # SPDC correlation PSF: mode-dispatched reduced-order kernel
    #   NF: σ = Δ_V(L, λ_s, λ_i, w_p)/√2   [P07 Eq. 4.9 / P10 Eq. 13]
    #   FF: σ = m_i f_u λ_i / (2π w_p)     [P14 σ_q=1/w_p; P07 Eq. 2.30]
    #   σ_SPDC = M · σ / p                 [pixels]
    crystal_material: str = CRYSTAL_MATERIAL_DEFAULT  # ppKTP
    qpm_type: str = QPM_TYPE_DEFAULT  # type0
    # User-facing SPDC polarization class. qpm_type remains the compatibility
    # field used by the reviewed Type-0 material/QPM implementation.
    spdc_type: str | None = None
    crystal_matching_mode: str | None = None
    # Hardware QPM constants (report §5.6 / §7.4). Not random-sweep variables.
    # d33 may be omitted; d_eff may be supplied directly (Comment 3 / 3.1).
    # When only d33 is set, d_eff is derived from eq. 40 (m, D).
    d33_pm_per_V: float | None = None
    d_eff_pm_per_V: float | None = None
    # NL2 physical crystal overrides for A; None explicitly inherits NL1.
    crystal_material_NL2: str | None = None
    qpm_type_NL2: str | None = None
    qpm_order_NL2: int | None = None
    qpm_duty_cycle_NL2: float | None = None
    d33_NL2_pm_per_V: float | None = None
    d_eff_NL2_pm_per_V: float | None = None
    d_eff_precedence_NL2: str | None = None
    pol_pump_NL2: str | None = None
    pol_signal_NL2: str | None = None
    pol_idler_NL2: str | None = None
    crystal_orientation_NL1_deg: float | None = None
    crystal_orientation_NL2_deg: float | None = None
    # Step-2 camera-plane sampling origin (does not redefine M_QIUP).
    camera_center_x_px: float = 0.0
    camera_center_y_px: float = 0.0
    # R09 apparatus / calibration slots (null by default — never fill from software).
    apparatus_crystal_NL1: dict[str, Any] | None = None
    apparatus_crystal_NL2: dict[str, Any] | None = None
    calibration_records: dict[str, Any] | None = None
    # auto | direct | derived — conflict policy when both d33 and d_eff are set.
    d_eff_precedence: str = DEFF_PRECEDENCE_DEFAULT
    # Comment 3.3 audit: requested inputs vs resolved working value.
    d_eff_requested_pm_per_V: float | None = None
    d33_requested_pm_per_V: float | None = None
    d_eff_resolved_pm_per_V: float | None = None
    d_eff_source: str = DEFF_SOURCE_NONE  # direct | derived_from_d33 | none
    # Legacy aliases kept in sync by resolve_effective_nonlinearity.
    d_eff_pm_per_V_requested: float | None = None
    d33_pm_per_V_requested: float | None = None
    d_eff_resolution: str = DEFF_RESOLUTION_NONE
    qpm_order: int = QPM_ORDER_DEFAULT  # m_QPM = 1
    qpm_duty_cycle: float = QPM_DUTY_CYCLE_DEFAULT  # D = 0.5
    poling_period_um: float = POLING_PERIOD_UM_DEFAULT  # Lemos 9.675 µm
    # two_crystal only. None → same lab Λ as crystal 1.
    poling_period_NL2_um: float | None = None
    # False = Physical Mode: lab Λ after thermal expansion enters Δk (report §7.3).
    # True  = explicit calibration mode: Λ_eff that forces Δk(T_pm)=0.
    use_calibrated_poling_period: bool = False
    # User-editable L [mm] in [2, 20]. ``None`` is an internal construction
    # sentinel resolved immediately by __post_init__ using blur_model:
    #   single_blur → 10 mm; double_blur → 2 mm per crystal/pass.
    crystal_length_mm: float | None = None
    # two_crystal / legacy. Single-crystal Physical Mode ignores this;
    # both SPDC passes share ``crystal_length_mm`` (report §7.11).
    # When set, each NL crystal uses its own L ("2 mm each" for double_blur).
    crystal_length_NL2_mm: float | None = None
    lambda_p_nm: float = 532.0
    # Polarizations are metadata only (report §7.25); they do not enter the gain.
    pol_pump: str = POLARIZATION_DEFAULT
    pol_signal: str = POLARIZATION_DEFAULT
    pol_idler: str = POLARIZATION_DEFAULT
    derive_sigma_spdc: bool = True
    # Used only when derive_sigma_spdc=False (legacy regression).
    sigma_spdc_px: float = 1.15

    # Diffraction / defocus (L6/L7 / PDF §6.4–6.9)
    # Physical default: σ_diff from λ_s, NA=D/(2f), p, M, η — not a fixed px.
    use_derived_diffraction: bool = True
    # Legacy-only explicit floor (used when use_derived_diffraction=False).
    sigma_diffraction_px: float = SIGMA_DIFFRACTION_PX_LEGACY
    # §6.5–6.10 C4↔C7: eq.160 (ΔT→Δf→Δz) + eq.164 (σ=K|Δz|) + eq.166 (s=|Δz|/Δz_max).
    # Eq.162 (s_focus = s0 + β_T s_thermal) is intentionally NOT defined.
    use_derived_K_defocus: bool = True
    n_medium: float = N_MEDIUM_DEFAULT
    eta_defocus: float = ETA_DEFOCUS_DEFAULT
    # σ_defocus,max for synthetic/legacy span (= K·Δz_max when derived).
    # YAML 3.6 is PDF §6.8 robustness worst-case — used only if use_derived_K_defocus=False.
    defocus_span_px: float = DEFOCUS_SPAN_PX_LEGACY
    K_defocus_px_per_um: float = K_DEFOCUS_PX_PER_UM_LEGACY  # override if not derived
    delta_z_max_um: float = DELTA_Z_MAX_UM_DEFAULT
    # Optional mechanical focus offset [µm]. Physical: Δz = Δf(T) + (this or 0).
    delta_z_um: float | None = None

    # Idler-lens misalignment phase (L06). Off by default.
    # Far field: P11 φ_I(ρ) with λ_eq=λ_s²/λ_i. Near field: object-plane φ_I
    # with λ_i and M_NF. The phase enters C12; it is not an I± texture.
    enable_lens_rings: bool = False
    lens_shift_x_um: float = 0.0  # idler-lens lateral shift → linear tilt [μm]
    lens_shift_y_um: float = 0.0
    lens_delta_z_soft_um: float = 80.0  # |Δz|≲80 μm: aligned-enough operating window
    lens_shift_soft_um: float = 40.0  # |shift|≲40 μm: mild lateral, not out of aperture
    lens_ring_aperture_px: float = 22.0  # effective clear-aperture / pupil radius [px]
    lens_ring_contrast_max: float = 0.45
    # When rings are active, residual Gaussian uses this fraction of σ_defocus
    # so Fresnel structure is not fully washed out.
    lens_ring_gaussian_residual: float = 0.15

    # Unwanted Fabry–Pérot / etalon fringes (filters, windows, weak cavities).
    # Intensity-level multiplicative map after I±, before EMCCD. Off by default.
    # Angle = orientation of fringe bands relative to horizontal
    # (0° ≈ horizontal bands, 90° ≈ vertical; lab often ~10–20°).
    enable_etalon_fringes: bool = False
    etalon_contrast: float = 0.28
    etalon_period_px: float = 34.0
    etalon_angle_deg: float = -40.0
    etalon_phase_rad: float = 0.0

    # Optional residual optical aberration (PDF §6.13/6.15 eq.180–181) — NO Zernike.
    # Physical default: σ_aberr = η_ab · W_waves · (M · r_Airy / p) from wavefront RMS.
    use_derived_aberration: bool = True
    wavefront_rms_waves: float = WAVEFRONT_RMS_WAVES_DEFAULT  # OPD RMS [waves @ λ_s]
    eta_aberration: float = ETA_ABERRATION_DEFAULT
    # Explicit px only when use_derived_aberration=False (legacy / override).
    sigma_aberration_px: float = 0.0
    # C3 Fan path-phase map (interference Δφ): corr length + align coupling
    wavefront_corr_px: float = WAVEFRONT_CORR_PX_DEFAULT
    couple_align_to_wavefront: bool = True
    delta_phi_align_span_rad: float = DELTA_PHI_ALIGN_SPAN_RAD_DEFAULT
    # Geometric IQ — Magnification/Rotation ≠ Optical Blur (PDF §6.17–6.18)
    # ε_M = ΔM/M (relative magnification error). Effective scale = 1 + ε_M.
    # image_scale is synced from delta_M_rel in validate (legacy YAML alias only).
    sigma_blur_extra_px: float = 0.0
    delta_M_rel: float = DELTA_M_REL_DEFAULT
    image_scale: float = 1.0  # = 1 + delta_M_rel after validate
    image_rotation_deg: float = 0.0  # θ [deg], mechanical misalignment

    # Visibility / path (L4 / PDF §4)
    # Residual calibration scale; default 1 → V_eff from physical factors only.
    V0: float = 1.0
    # Legacy only: δ = overlap_delta_scale · s_align  (PDF eq. 64). Unused when
    # use_analytic_mode_overlap=True.
    overlap_delta_scale: float = 1.15
    # Idler–idler Gaussian overlap (PDF eq. 63). Default w2=w1=idler_waist_um,
    # and idler_waist_um inherits pump_waist_um in the crystal when None.
    use_analytic_mode_overlap: bool = True
    idler_waist_um: float | None = None
    idler_waist_2_um: float | None = None
    # V-P1-04 — split alignment so the same offset is not applied twice:
    #   generation_* → pump→crystal O_geom (SPDC generation)
    #   relay_* / legacy idler_* → idler→seed η_mode (relay / t_seed)
    generation_dx_um: float = 0.0
    generation_dy_um: float = 0.0
    generation_dtheta_mrad: float = 0.0
    # Explicit relay offsets (None → fall back to legacy idler_* below).
    relay_dx_um: float | None = None
    relay_dy_um: float | None = None
    relay_dtheta_mrad: float | None = None
    # Legacy relay alias [µm]. Per-frame pump pointing Δx,Δy (px→µm) is
    # added when inherit_idler_displacement_from_pump=True (default).
    # Not used by O_geom (V-P1-04).
    idler_dx_um: float = 0.0
    idler_dy_um: float = 0.0
    idler_dtheta_mrad: float = 0.0
    inherit_idler_displacement_from_pump: bool = True
    # PDF §4.1 eq. 65: V_coh = exp(-(ΔL/L_c)²). Lab path mismatch [mm], not s_align.
    # Editable in [-0.50, 0.50]; sign is retained in phase. Default 0.01.
    delta_L_mm: float = DELTA_L_MM_DEFAULT
    L_coh_mm: float = L_COH_MM_DEFAULT
    # Optional signal filter before the shared detector. None preserves the
    # historical coherence model. The v2 profiles select an explicit preset.
    camera_filter_preset: str | None = None
    camera_filter_center_nm: float | None = None
    camera_filter_fwhm_nm: float | None = None
    camera_filter_model: str | None = None
    signal_spectrum_fwhm_nm: float | None = None
    camera_filter_wavelengths_nm: list[float] | None = None
    camera_filter_transmissions: list[float] | None = None
    # Post-combiner camera collection E(x,y) on detected intensity (not S_cam).
    final_camera_collection_policy: str | None = None
    # Report §4.7 / §7.18: signal-path intensity transmissions ηs1, ηs2.
    # Historical YAML name pathN_intensity is the stored field.
    path1_intensity: float = 1.0
    path2_intensity: float = 1.0
    # Report §4.4 / §7.15: idler-path intensity transmission ηi,path (not the object).
    idler_path_transmission: float = 1.0
    # True: I± = I1+I2 ± 2√(I1 I2) Γ √T cosφ  (eq. 69). False: legacy 1±V cos.
    use_physical_two_beam: bool = True
    # PDF §4.3: V(x,y)∝|E1 E2*| from physical fields (eq. 74). False → legacy
    # mottling V_eff(1 - 0.25 s_align m̃) (eq. 73).
    use_field_overlap_visibility_map: bool = True
    V_gain_drop: float = 0.40
    # Legacy eq. 73 only (ignored when use_field_overlap_visibility_map=True).
    V_map_mottle: float = 0.25
    mottle_smooth_sigma_px: float = 8.0

    # Thermal (L5 / PDF §5) — physical chain; engineered rates disabled when
    # use_physical_thermal=True.
    # Hardware crystal count (inventory), independent of SPDC interaction count.
    # The default single-crystal double-pass architecture has one crystal.
    n_crystals: int = 1
    # Declarative YAML assertion only; the architecture resolver owns the count.
    n_spdc_interactions: int | None = None
    sync_crystal_temperatures: bool = True
    use_physical_thermal: bool = True
    T_pm0_NL1_C: float = T_CRYSTAL_C_DEFAULT
    T_pm0_NL2_C: float = T_CRYSTAL_C_DEFAULT
    delta_T_NL1_C: float = 0.0
    delta_T_NL2_C: float = 0.0
    # Lab detuning dT = T_crystal - T_pm. User variable; default 0.
    dT_NL1_C: float = 0.0
    dT_NL2_C: float = 0.0
    T_crystal_NL1_C: float = T_CRYSTAL_C_DEFAULT
    T_crystal_NL2_C: float = T_CRYSTAL_C_DEFAULT
    T_crystal_C: float = T_CRYSTAL_C_DEFAULT
    T_env_C: float = 23.0
    T_pm_C: float = T_CRYSTAL_C_DEFAULT
    T_ref_span_C: float = 8.0  # |ΔT_eff|=8 °C ↔ s_thermal≈1 (latent map only)
    # §5.2: ΔL_th = α_path · L_eff · ΔT  (α_air = dn_air/dT)
    alpha_path_per_C: float = ALPHA_AIR_PATH_PER_C
    L_eff_path_mm: float = L_EFF_PATH_MM_DEFAULT
    # Deprecated alias (migrated → alpha_path * L_eff); kept for old YAML
    alpha_L_mm_per_C: float = ALPHA_L_MM_PER_C_DEFAULT
    # §5.4/§5.9 beam shift [µm/°C]; default = CTE_Al · F1
    beam_thermal_alpha_x_um_per_C: float = CTE_MOUNT_PER_C * F1_MM_DEFAULT * 1e3
    beam_thermal_alpha_y_um_per_C: float = CTE_MOUNT_PER_C * F1_MM_DEFAULT * 1e3
    # §5.5 / §5.7 imaging (Lemos F1=75 mm, F2=150 mm; N-BK7)
    lens_focal_mm: float = LENS_FOCAL_MM_DEFAULT
    lens_material: str = LENS_MATERIAL_DEFAULT
    lens_n: float = NBK7_N
    lens_dndT_per_C: float = NBK7_DNDT_PER_C
    lens_cte_per_C: float = NBK7_CTE_PER_C
    F1_mm: float = F1_MM_DEFAULT
    F2_mm: float = F2_MM_DEFAULT
    # Deprecated flag (V-P1-06): M_QIUP is always used. Kept for YAML schema compat.
    use_physical_qiup_magnification: bool = True
    # Legacy latent-scale rates (IGNORED when use_physical_thermal=True)
    phase_drift_rad_per_frame_at_s1: float = 0.0
    vis_walk_per_frame_at_s1: float = 0.0
    beam_drift_x_px_per_frame_at_s1: float = 0.0
    beam_drift_y_px_per_frame_at_s1: float = 0.0
    phase_jitter_align_at_s1: float = 0.03
    # Audit-only caches; forward uses Sellmeier dλ/dT when physical thermal
    dlam_s_nm_per_C: float = 0.0
    dlam_i_nm_per_C: float = 0.0
    dM_over_M_per_C: float = 0.0

    # Detector (L10 / M1 / M2 / M5 / M6)
    QE: float = 0.65
    read_noise0_e: float = 8.0
    read_noise_span_e: float = 40.0
    # M1: "legacy_s_det" → σ=8+40·s_det; "datasheet_fixed" → σ=read_noise0_e (post-EM)
    read_noise_model: str = "legacy_s_det"
    # M1/M2: "legacy" keeps Phase-0 PC path; "register" = EM then read then threshold/ADU
    emccd_model: str = "legacy"
    # V-P1-07: explicit detector path — analog_emccd | photon_counting.
    # Never infer from EM gain, flux, or other latents.
    detector_mode: str = DETECTOR_MODE_DEFAULT
    # Task 8: selected signal port. None preserves historical serialization
    # and resolves to ``plus``; new profiles set it explicitly.
    detector_output_port: str | None = None
    cic0: float = 0.006
    prnu_sigma: float = 0.03
    dark0: float = 0.0025
    dark_exp: float = 7.2
    # M5 Arrhenius dark (opt-in)
    use_physical_dark: bool = False
    T_sensor_C: float = -80.0
    T_sensor_max_C: float = -40.0
    T_sensor_min_C: float = -100.0
    dark_arrhenius_prefactor: float = 0.0025
    Eg_eV: float = 1.12
    # M6 hot pixels
    hot_frac_at_s1: float = 0.002
    hot_amp_lo: float = 2.0
    hot_amp_hi: float = 8.0
    hot_pixel_model: str = "legacy"  # "legacy" | "mask_T"
    hot_T_coef_per_C: float = 0.04   # amp ∝ exp(coef·(T-T_min))
    accidental_e_at_s1: float = 0.02
    pc_threshold_k: float = 4.0
    full_well_e: float = 8.0e4
    g_adu: float = 0.45
    bias_adu: float = 100.0
    max_adu: float = 65535.0
    # None derives the historical width from max_adu (65535 -> 16 bits).
    adc_bit_depth: int | None = None
    # C_lab display/export zoom (centered crop only; quantitative native grid unchanged).
    camera_native_height_px: int | None = None
    camera_native_width_px: int | None = None
    export_zoom_crop_enabled: bool = False
    export_zoom_crop_mode: str | None = None
    export_zoom_center_source: str | None = None
    export_zoom_export_width_px: int | None = None
    export_zoom_export_height_px: int | None = None
    export_zoom_preserve_native_frame: bool = True
    export_zoom_resize_after_crop: bool = False
    export_zoom_interpolation: str | None = None
    export_zoom_export_dtype: str | None = None
    # Generalized camera crop (aliases export_zoom_* when unset; display/export only).
    camera_crop_enabled: bool | None = None
    camera_crop_mode: str | None = None
    camera_crop_width_px: int | None = None
    camera_crop_height_px: int | None = None
    camera_crop_center_mode: str | None = None
    camera_crop_center_row_px: int | None = None
    camera_crop_center_col_px: int | None = None
    camera_crop_preserve_native_frame: bool | None = None
    camera_crop_resize: bool | None = None
    camera_crop_interpolation: str | None = None
    camera_crop_export_dtype: str | None = None
    camera_crop_export_colormap: str | None = None
    em_gain_default: float = 250.0

    # SLM (L9) + M9 misregistration
    slm_bit_depth: int = 8
    slm_levels0: float = 256.0
    slm_pixel_um: float = 8.0
    slm_fill_factor: float = 0.90
    slm_phi_max_rad: float = float(2.0 * np.pi)
    slm_quant_coef: float = 0.985
    slm_levels_floor: int = 3
    slm_gamma_span: float = 0.9
    slm_leak_eta: float = 0.40
    slm_misreg_x_px: float = 0.0
    slm_misreg_y_px: float = 0.0

    # M10 — optional per-setup coupling overrides (None → module COUPLING_WEIGHTS)
    coupling_weights: dict[str, float] | None = None
    coupling_profile_id: str = "phase0_frozen_v1"

    # M8 — sampling regime label (lab vs stress); used by sample_latents
    sampling_regime: str = "lab"

    # Phase-4 D1 — Zernike deferred; if True, simulate_sequence refuses
    enable_zernike: bool = False

    # Metadata / versioning
    setup_id: str = "physical_default_v1"
    model_version: str = "qiup_spdc_simulator_v3.3.1-remediated"
    physics_tier: str = "phase3"  # "legacy" | "phase3"
    notes: str = (
        "Physical beam default: pump_waist_um=250 µm (allowed 100–350), "
        "pixel_um=16 µm, M=1, crystal L=2 mm (2–20). "
        "σ_SPDC = M·σ/p; NF uses P07/P10 Δ_V/√2, FF uses m_i f_u λ_i/(2π w_p)."
    )

    def __post_init__(self) -> None:
        """Resolve and validate the user-facing crystal-length inputs.

        This makes direct construction consistent with YAML/from_dict and
        ``overridden``: omitted L follows blur_model, while every explicit
        user value in [2, 20] mm is preserved.
        """
        self.blur_model = normalize_blur_model(self.blur_model)
        if self.crystal_length_mm is None:
            self.crystal_length_mm = default_crystal_length_mm_for_policy(
                self.physics_policy_id, self.blur_model
            )
        self.crystal_length_mm = validate_crystal_length_mm(
            self.crystal_length_mm
        )
        if self.crystal_length_NL2_mm is not None:
            self.crystal_length_NL2_mm = validate_crystal_length_mm(
                self.crystal_length_NL2_mm
            )
        # R03: reject string/numeric Boolean coercions on every entry path.
        validate_setup_boolean_fields(self)
        validate_etalon_parameters(self)
        validate_l07_acquisition_parameters(self)
        profile = resolve_architecture_profile(self)
        validate_architecture_propagation_fields(self)
        validate_spdc_energy_conservation(self)
        self.qpm_order = validate_qpm_order(self.qpm_order)
        if self.qpm_order_NL2 is not None:
            self.qpm_order_NL2 = validate_qpm_order(
                self.qpm_order_NL2, name="qpm_order_NL2"
            )
        self.QE = validate_quantum_efficiency(self.QE)
        validate_detector_configuration(self)
        validate_crystal_parameter_sets(self)
        if self.pump_profile_type is not None:
            self.pump_profile_type = normalize_pump_profile_type(
                self.pump_profile_type
            )
        if self.apparatus_crystal_NL1 is not None and not isinstance(
            self.apparatus_crystal_NL1, dict
        ):
            raise ValueError("apparatus_crystal_NL1 must be a mapping or None")
        if self.apparatus_crystal_NL2 is not None and not isinstance(
            self.apparatus_crystal_NL2, dict
        ):
            raise ValueError("apparatus_crystal_NL2 must be a mapping or None")
        if self.calibration_records is not None and not isinstance(
            self.calibration_records, dict
        ):
            raise ValueError("calibration_records must be a mapping or None")
        if (
            not is_two_crystal_architecture(self)
            and self.apparatus_crystal_NL2 is not None
        ):
            raise ValueError(
                "single_crystal_double_pass uses one physical crystal; "
                "apparatus_crystal_NL2 must remain null"
            )
        self.two_crystal_pump_topology = normalize_two_crystal_pump_topology(
            self.two_crystal_pump_topology
        )
        if self.pump_bs_reflectance is not None:
            self.pump_bs_reflectance = validate_pump_bs_reflectance(
                self.pump_bs_reflectance
            )
        for name in (
            "pump_power_NL2_override_calibration_id",
            "spdc_gain_calibration_id",
            "spdc_gain_calibration_id_NL2",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string when set")
        for name in (
            "path1_intensity", "path2_intensity", "idler_path_transmission",
            "pump_pre_crystal_transmission", "pump_return_transmission",
            "pump_to_NL2_transmission",
        ):
            validate_transmission(getattr(self, name), name=name)
        if self.pump_between_crystals_transmission is not None:
            validate_transmission(
                self.pump_between_crystals_transmission,
                name="pump_between_crystals_transmission",
            )
        if self.pump_power_NL2_override_mW is not None:
            if not self.use_independent_pump_power_NL2:
                raise ValueError(
                    "pump_power_NL2_override_mW requires "
                    "use_independent_pump_power_NL2=True"
                )
            if not str(self.pump_power_NL2_override_calibration_id or "").strip():
                raise ValueError(
                    "independent NL2/backward pump override requires "
                    "pump_power_NL2_override_calibration_id"
                )
        resolve_crystal_operating_detuning(self)
        self.T_pm_C = phase_match_temperature_C(self.T_pm0_NL1_C, self.delta_T_NL1_C)
        self.T_crystal_C = self.T_crystal_NL1_C
        if profile.spdc_architecture == SPDC_ARCHITECTURE_SINGLE:
            if (self.crystal_length_NL2_mm is not None
                    and self.crystal_length_NL2_mm != self.crystal_length_mm):
                raise ValueError("single-crystal passes must share crystal_length_mm")
            if (self.poling_period_NL2_um is not None
                    and self.poling_period_NL2_um != self.poling_period_um):
                raise ValueError("single-crystal passes must share poling_period_um")
            if self.sync_crystal_temperatures:
                self.dT_NL2_C = self.dT_NL1_C
                self.T_crystal_NL2_C = self.T_crystal_NL1_C
            if not self.sync_crystal_temperatures and (
                    self.T_crystal_NL2_C != self.T_crystal_NL1_C
                    or self.T_pm0_NL2_C != self.T_pm0_NL1_C
                    or self.delta_T_NL2_C != self.delta_T_NL1_C
                    or float(getattr(self, "dT_NL2_C", 0.0))
                    != float(getattr(self, "dT_NL1_C", 0.0))):
                raise ValueError(
                    "single-crystal passes must share physical crystal temperature"
                )
        # NL1 d_eff derived from d33 is resolved by validate_setup_params after
        # construction, so coupling availability is checked there.
        validate_calibrated_source_requirements(self, require_coupling=False)
        validate_physics_team_refinements(self)

    @classmethod
    def legacy_matched(cls) -> "SetupParams":
        """Factory: bit-compatible with Phase-0/1/2 forward numerics (regression only)."""
        return cls(
            spdc_architecture=SPDC_ARCHITECTURE_TWO,
            imaging_mode=IMAGING_MODE_FAR,
            n_crystals=2,
            use_legacy_beam_match=True,
            pump_waist_um=None,
            pointing_alpha=POINTING_ALPHA_DEFAULT,
            derive_sigma_spdc=False,
            sigma_spdc_px=1.15,
            crystal_length_mm=CRYSTAL_LENGTH_MM_DEFAULT,
            use_analytic_mode_overlap=False,
            overlap_delta_scale=1.15,
            use_physical_two_beam=False,
            use_physical_spdc_gain=False,
            path1_intensity=1.0,
            path2_intensity=1.0,
            use_field_overlap_visibility_map=False,
            use_physical_thermal=False,
            use_physical_qiup_magnification=False,
            use_derived_diffraction=False,
            sigma_diffraction_px=SIGMA_DIFFRACTION_PX_LEGACY,
            use_derived_K_defocus=False,
            K_defocus_px_per_um=K_DEFOCUS_PX_PER_UM_LEGACY,
            defocus_span_px=DEFOCUS_SPAN_PX_LEGACY,
            delta_z_max_um=DELTA_Z_MAX_UM_DEFAULT,
            n_medium=N_MEDIUM_DEFAULT,
            eta_defocus=ETA_DEFOCUS_DEFAULT,
            use_derived_aberration=False,
            wavefront_rms_waves=0.0,
            sigma_aberration_px=0.0,
            couple_align_to_wavefront=False,
            delta_phi_align_span_rad=0.0,
            phase_drift_rad_per_frame_at_s1=0.08,
            vis_walk_per_frame_at_s1=0.02,
            beam_drift_x_px_per_frame_at_s1=0.15,
            beam_drift_y_px_per_frame_at_s1=0.10,
            V_map_mottle=0.25,
            setup_id="legacy_matched_v1",
            model_version="qiup_arch_phase2_v1",
            physics_tier="legacy",
            notes="Phase-0/1/2 numerical regression profile (N/3.2 envelope).",
        )

    @classmethod
    def phase3_physics(cls) -> "SetupParams":
        """Phase-3 calibratable profile (intentional numeric differences vs Phase-0)."""
        return cls(
            setup_id="phase3_physics_v1",
            model_version="qiup_arch_phase3_v1",
            physics_tier="phase3",
            notes=(
                "Phase-3 physics: EMCCD register chain, datasheet read noise, "
                "Arrhenius dark, T-dependent hot mask, derived σ_SPDC."
            ),
            emccd_model="register",
            read_noise_model="datasheet_fixed",
            use_physical_dark=True,
            hot_pixel_model="mask_T",
            derive_sigma_spdc=True,
            sampling_regime="lab",
            coupling_profile_id="phase0_frozen_v1",
        )

    @classmethod
    def personal_lab_template(cls) -> "SetupParams":
        """Personal lab setup template (physical beam + phase-3 physics flags).

        Default pump waist 250 µm (range 100–1000 µm). Edit YAML for your lab.
        """
        base = cls.phase3_physics()
        return validate_setup_params(base.overridden(
            setup_id=CANONICAL_SETUP_ID,
            model_version="qiup_spdc_simulator_v3.3.1-remediated",
            notes=(
                "CFG-P0-00 canonical lab: n_crystals=1, "
                "single_crystal_double_pass, analog_emccd, double_blur "
                f"(default crystal_length_mm={CANONICAL_CRYSTAL_LENGTH_MM}), "
                "low_gain, fixed_nbar. User may set L in [2, 20] mm. "
                "pump_waist_um=250 µm, pixel_um=16 µm. "
                "Release 3.3.1-remediated (Comment 7.2 / 8.2). "
                "Status: Physics-Informed Reduced-Order SPDC-QIUP "
                "Forward Simulator Prototype "
                "(not a Verified/Validated Digital Twin; "
                "official dataset generation blocked until sign-off)."
            ),
            use_legacy_beam_match=False,
            pump_waist_um=PUMP_WAIST_UM_DEFAULT,
            pixel_um=PIXEL_UM_DEFAULT,
            magnification=1.0,
            pointing_alpha=POINTING_ALPHA_DEFAULT,
            crystal_length_mm=default_crystal_length_mm_for_blur(BLUR_MODEL_DOUBLE),
            derive_sigma_spdc=True,
            use_analytic_mode_overlap=True,
            idler_waist_um=None,  # inherit pump_waist_um in crystal
            generation_dx_um=0.0,
            generation_dy_um=0.0,
            generation_dtheta_mrad=0.0,
            relay_dx_um=None,
            relay_dy_um=None,
            relay_dtheta_mrad=None,
            idler_dx_um=0.0,
            idler_dy_um=0.0,
            idler_dtheta_mrad=0.0,
            inherit_idler_displacement_from_pump=True,
            use_physical_two_beam=True,
            use_physical_spdc_gain=True,
            path1_intensity=1.0,
            path2_intensity=1.0,
            use_field_overlap_visibility_map=True,
            sampling_regime="lab",
            V0=1.0,
            n_crystals=CANONICAL_N_CRYSTALS,
            spdc_architecture=SPDC_ARCHITECTURE_SINGLE,
            blur_model=BLUR_MODEL_DOUBLE,
            spdc_gain_mode=SPDC_GAIN_MODE_LOW,
            spdc_source_mode=SPDC_SOURCE_FIXED,
            detector_mode=DETECTOR_MODE_ANALOG,
            imaging_mode=IMAGING_MODE_NEAR,
            sync_crystal_temperatures=True,
            use_physical_thermal=True,
            use_physical_qiup_magnification=True,
            crystal_material=CRYSTAL_MATERIAL_DEFAULT,
            qpm_type=QPM_TYPE_DEFAULT,
            poling_period_um=POLING_PERIOD_UM_DEFAULT,
            lambda_p_nm=CANONICAL_LAMBDA_P_NM,
            pol_pump=POLARIZATION_DEFAULT,
            pol_signal=POLARIZATION_DEFAULT,
            pol_idler=POLARIZATION_DEFAULT,
            alpha_path_per_C=ALPHA_AIR_PATH_PER_C,
            L_eff_path_mm=L_EFF_PATH_MM_DEFAULT,
            L_coh_mm=L_COH_MM_DEFAULT,
            lambda_s_nm=CANONICAL_LAMBDA_S_NM,
            lambda_i_nm=CANONICAL_LAMBDA_I_NM,
            F1_mm=F1_MM_DEFAULT,
            F2_mm=F2_MM_DEFAULT,
            lens_focal_mm=LENS_FOCAL_MM_DEFAULT,
            lens_material=LENS_MATERIAL_DEFAULT,
            aperture_diameter_mm=APERTURE_DIAMETER_MM_DEFAULT,
            airy_to_gauss_eta=AIRY_TO_GAUSS_ETA_DEFAULT,
            use_derived_diffraction=True,
            use_derived_K_defocus=True,
            n_medium=N_MEDIUM_DEFAULT,
            eta_defocus=ETA_DEFOCUS_DEFAULT,
            delta_z_max_um=DELTA_Z_MAX_UM_DEFAULT,
            delta_z_um=None,
            use_derived_aberration=True,
            wavefront_rms_waves=WAVEFRONT_RMS_WAVES_DEFAULT,
            eta_aberration=ETA_ABERRATION_DEFAULT,
            couple_align_to_wavefront=True,
            delta_phi_align_span_rad=DELTA_PHI_ALIGN_SPAN_RAD_DEFAULT,
            wavefront_corr_px=WAVEFRONT_CORR_PX_DEFAULT,
            T_pm0_NL1_C=T_CRYSTAL_C_DEFAULT,
            T_pm0_NL2_C=T_CRYSTAL_C_DEFAULT,
            delta_T_NL1_C=0.0,
            delta_T_NL2_C=0.0,
            T_crystal_NL1_C=T_CRYSTAL_C_DEFAULT,
            T_crystal_NL2_C=T_CRYSTAL_C_DEFAULT,
            T_crystal_C=T_CRYSTAL_C_DEFAULT,
            T_pm_C=T_CRYSTAL_C_DEFAULT,
            T_env_C=23.0,
            coupling_profile_id="phase0_frozen_v1",
        ))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Old canonical and legacy hashes predate the optional YAML assertion.
        for name in (
            "physics_policy_id", "spdc_type", "crystal_matching_mode",
            "pump_balance_mode", "pump_target_power_NL1_mW",
            "pump_target_power_NL2_mW", "fixed_source_input",
            "measured_power_W", "measured_power_wavelength_nm",
            "measured_power_to_nbar_per_W",
            "measured_power_to_nbar_per_W_NL2",
            "measured_power_to_nbar_calibration_id",
            "camera_filter_preset", "camera_filter_center_nm",
            "camera_filter_fwhm_nm", "camera_filter_model",
            "signal_spectrum_fwhm_nm", "camera_filter_wavelengths_nm",
            "camera_filter_transmissions", "object_device", "object_mode",
            "object_mask_id", "object_intensity_T", "object_intensity_R",
            "object_phase_rad",
            "object_phase_map_path", "object_amplitude_map_path",
            "object_intensity_map_path", "object_phase_units",
            "object_intensity_convention", "object_coordinate_orientation",
            "n_spdc_interactions", "crystal_material_NL2", "qpm_type_NL2",
            "qpm_order_NL2", "qpm_duty_cycle_NL2", "d33_NL2_pm_per_V",
            "d_eff_NL2_pm_per_V", "d_eff_precedence_NL2",
            "pol_pump_NL2", "pol_signal_NL2",
            "pol_idler_NL2", "crystal_orientation_NL1_deg",
            "crystal_orientation_NL2_deg", "two_crystal_pump_topology",
            "pump_bs_reflectance",
            "pump_between_crystals_transmission",
            "pump_power_NL2_override_calibration_id",
            "spdc_gain_per_sqrt_mW_NL2", "spdc_gain_calibration_id",
            "spdc_gain_calibration_id_NL2",
            "architecture_propagation_model", "object_type", "object_plane",
            "object_transmission_map", "object_phase_map",
            "object_reflection_amplitude", "object_reflection_phase",
            "object_forward_transform", "object_backward_transform",
            "object_interaction_count", "idler_round_trip_loss",
            "idler_mode_matching", "idler_return_phase",
            "relative_interferometric_phase",
            "phase_step_schedule_rad",
            "signal_return_path_transmission",
            "signal_return_reflection_amplitude", "signal_return_phase",
            "signal_return_mode_overlap",
            "signal_return_seeds_second_interaction",
            "spatial_transfer_kernel", "object_map_pixel_um",
            "near_field_idler_magnification",
            "near_field_signal_magnification",
            "far_field_idler_relay_magnification",
            "far_field_signal_relay_magnification",
            "detector_output_port", "adc_bit_depth",
            "pump_profile_type", "apparatus_crystal_NL1",
            "apparatus_crystal_NL2", "calibration_records",
        ):
            if data[name] is None:
                data.pop(name)
        # L07 opt-in defaults stay out of the frozen config hash until used.
        if not bool(data.get("enable_phase_step_acquisition", False)):
            data.pop("enable_phase_step_acquisition", None)
        if abs(float(data.get("optical_background_photons", 0.0) or 0.0)) == 0.0:
            data.pop("optical_background_photons", None)
        # Default lab detuning is 0 and stays out of the frozen hash until set.
        for name in ("dT_NL1_C", "dT_NL2_C"):
            if abs(float(data.get(name, 0.0) or 0.0)) == 0.0:
                data.pop(name, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetupParams":
        """Build from a mapping (Comment 4.2 — fail-closed keys).

        Unknown names raise ``ConfigError``. There is no silent
        ``{k: v for k, v in data.items() if k in valid_fields}`` filter.
        Declared legacy aliases may appear briefly, are migrated onto real
        fields, then removed before construction.
        """
        data = dict(data)
        # Comment 4.1 / 4.2: reject typos before any construction.
        validate_setup_keys(data, allow_legacy=True, context="SetupParams.from_dict")
        data = _migrate_setup_yaml_aliases(data)
        if "crystal_length_mm" not in data and "crystal_length_um" in data:
            Lmm = float(data["crystal_length_um"]) / 1000.0
            # Previous default was 1000 µm (1 mm); physical range starts at 2 mm.
            if Lmm < CRYSTAL_LENGTH_MM_MIN:
                Lmm = CRYSTAL_LENGTH_MM_DEFAULT
            data["crystal_length_mm"] = Lmm
        # If L omitted, apply the active versioned policy.
        if "crystal_length_mm" not in data:
            data["crystal_length_mm"] = default_crystal_length_mm_for_policy(
                data.get("physics_policy_id"), data.get("blur_model")
            )
        # Migrate deprecated engineered ΔL = ΔL0 + span·s_align → lab ΔL.
        if "delta_L_mm" not in data and "delta_L0_mm" in data:
            data["delta_L_mm"] = float(data["delta_L0_mm"])
        # Migrate old combined αL → α_path · L_eff (air path).
        if "alpha_path_per_C" not in data and "alpha_L_mm_per_C" in data:
            L_eff = float(data.get("L_eff_path_mm", L_EFF_PATH_MM_DEFAULT))
            aL = float(data["alpha_L_mm_per_C"])
            if abs(L_eff) > 1e-18:
                data["alpha_path_per_C"] = aL / L_eff
        # Migrate legacy NA → aperture D = 2·NA·f when D omitted.
        if "aperture_diameter_mm" not in data and "NA" in data:
            f_mm = float(data.get("lens_focal_mm", LENS_FOCAL_MM_DEFAULT))
            data["aperture_diameter_mm"] = 2.0 * float(data["NA"]) * f_mm
        # Migrate legacy single T_crystal_C / T_pm_C → per-crystal + eq. 80.
        if "T_crystal_NL1_C" not in data and "T_crystal_C" in data:
            Tc = float(data["T_crystal_C"])
            data["T_crystal_NL1_C"] = Tc
            data["T_crystal_NL2_C"] = Tc
        if "T_pm0_NL1_C" not in data and "T_pm_C" in data:
            # Old absolute T_pm → treat as T_pm0 with ΔT=0.
            Tpm = float(data["T_pm_C"])
            data["T_pm0_NL1_C"] = Tpm
            data["T_pm0_NL2_C"] = Tpm
            data.setdefault("delta_T_NL1_C", 0.0)
            data.setdefault("delta_T_NL2_C", 0.0)
        # PDF §6.17: prefer ε_M; legacy image_scale → ε_M = scale − 1
        if "delta_M_rel" not in data and "image_scale" in data:
            data["delta_M_rel"] = float(data["image_scale"]) - 1.0
        # Sweep / report aliases → stored SetupParams fields (§7.24).
        for alias, target in SETUP_SWEEP_ALIASES.items():
            if alias in data and target not in data:
                data[target] = data[alias]
        # Remove only declared legacy alias keys (never unknown typos).
        for alias in SETUP_YAML_LEGACY_ALIASES:
            data.pop(alias, None)
        # Every remaining key must be a real SetupParams field.
        validate_setup_keys(
            data, allow_legacy=False, context="SetupParams.from_dict"
        )
        return validate_setup_params(cls(**data))

    def overridden(self, **kwargs: Any) -> "SetupParams":
        """Return a copy with selected fields changed (Comment 4.3 — fail-closed).

        Unknown override names raise ``ConfigError`` *before* any merge, so a
        typo such as ``crystal_lenght_mm=10`` cannot silently no-op.
        """
        require_canonical = bool(kwargs.pop("require_canonical", False))
        # Comment 4.3: validate overrides first, then apply.
        validate_setup_keys(
            kwargs,
            allow_legacy=True,
            extra_allowed=SETUP_OVERRIDE_CONVENIENCE_KEYS,
            context="SetupParams.overridden",
        )
        d = self.to_dict()
        # Convenience: editing shared T_crystal_C / T_pm0 updates both crystals.
        if "T_crystal_C" in kwargs and "T_crystal_NL1_C" not in kwargs:
            Tc = float(kwargs["T_crystal_C"])
            kwargs.setdefault("T_crystal_NL1_C", Tc)
            kwargs.setdefault("T_crystal_NL2_C", Tc)
        if "T_pm0_C" in kwargs:
            T0 = float(kwargs.pop("T_pm0_C"))
            kwargs.setdefault("T_pm0_NL1_C", T0)
            if normalize_spdc_architecture(
                kwargs.get("spdc_architecture", d.get("spdc_architecture"))
            ) != SPDC_ARCHITECTURE_TWO:
                kwargs.setdefault("T_pm0_NL2_C", T0)
        if "delta_T_C" in kwargs:
            oven = float(kwargs.pop("delta_T_C"))
            kwargs.setdefault("delta_T_NL1_C", oven)
            kwargs.setdefault("delta_T_NL2_C", oven)
        if "dT_C" in kwargs:
            det = float(kwargs.pop("dT_C"))
            kwargs.setdefault("dT_NL1_C", det)
            if normalize_spdc_architecture(
                kwargs.get("spdc_architecture", d.get("spdc_architecture"))
            ) != SPDC_ARCHITECTURE_TWO:
                kwargs.setdefault("dT_NL2_C", det)
        # PDF §6.17: ε_M is source of truth; legacy image_scale → ε_M
        if "image_scale" in kwargs and "delta_M_rel" not in kwargs:
            kwargs["delta_M_rel"] = float(kwargs["image_scale"]) - 1.0
        if "delta_M_rel" in kwargs:
            kwargs["image_scale"] = 1.0 + float(kwargs["delta_M_rel"])
        # Blur switch: if L was still the old blur default and user did not
        # set L, adopt the new blur default (user L in [2,20] always wins).
        if ("blur_model" in kwargs and "crystal_length_mm" not in kwargs
                and not is_three_configuration_v2(self)):
            old_default = default_crystal_length_mm_for_blur(self.blur_model)
            new_default = default_crystal_length_mm_for_blur(kwargs["blur_model"])
            if abs(float(self.crystal_length_mm) - old_default) < 1.0e-12:
                kwargs["crystal_length_mm"] = new_default
        if require_canonical:
            reject_canonical_overrides(kwargs)
        d.update(kwargs)
        # Drop prior nonlinearity audit so a fresh resolve sees the merged
        # field values (avoids treating a previously derived d_eff as "direct").
        d["d_eff_resolution"] = DEFF_RESOLUTION_NONE
        d["d_eff_source"] = DEFF_SOURCE_NONE
        d["d_eff_pm_per_V_requested"] = None
        d["d_eff_requested_pm_per_V"] = None
        d["d33_pm_per_V_requested"] = None
        d["d33_requested_pm_per_V"] = None
        d["d_eff_resolved_pm_per_V"] = None
        setup = SetupParams.from_dict(d)  # validates once
        if hasattr(self, "_setup_selection"):
            setup._setup_selection = dict(self._setup_selection)
        if require_canonical:
            assert_canonical_setup(setup)
        return setup


@dataclass
class ResolvedForward:
    """Pixel-domain numbers consumed by the forward model for one frame size N."""

    N: int
    w_px: float
    pointing_rms_x: float
    pointing_rms_y: float
    sigma_spdc_px: float
    imaging_mode: str
    correlation_type: str
    blur_model: str
    spdc_gain_mode: str
    sigma_diffraction_px: float
    defocus_span_px: float  # = σ_defocus,max = K · Δz_max
    K_defocus_px_per_um: float
    delta_z_um: float
    coc_um: float
    s_focus_from_dz: float
    n_medium: float
    eta_defocus: float
    sigma_aberration_px: float
    delta_M_rel: float
    image_scale: float  # 1 + ε_M (PDF §6.17)
    image_rotation_deg: float  # θ [deg] (PDF §6.18)
    # visibility
    V0: float
    overlap_delta_scale: float
    use_analytic_mode_overlap: bool
    eta_mode: float
    delta_L_mm: float
    delta_L_eff_mm: float
    L_coh_mm: float
    path1_intensity: float
    path2_intensity: float
    idler_path_transmission: float
    use_physical_two_beam: bool
    use_field_overlap_visibility_map: bool
    V_gain_drop: float
    V_map_mottle: float
    mottle_smooth_sigma_px: float
    # thermal (PDF §5 physical bundle)
    use_physical_thermal: bool
    n_crystals: int
    dT_eff_C: float
    thermal_phase_offset_rad: float
    beam_thermal_dx_um: float
    beam_thermal_dy_um: float
    beam_thermal_dx_px: float
    beam_thermal_dy_px: float
    lens_delta_f_um: float
    R_SPDC_sinc2: float
    M_QIUP: float
    # thermal rates at s=1 (legacy latent scales; 0 when physical)
    phase_drift_rad_per_frame: float
    vis_walk_per_frame: float
    beam_drift_x_px_per_frame: float
    beam_drift_y_px_per_frame: float
    phase_jitter_align: float
    # pump
    flux0_e_per_s: float
    pump_sigma_rel: float
    flux_boost: float
    # detector
    QE: float
    read_noise0_e: float
    read_noise_span_e: float
    cic0: float
    prnu_sigma: float
    dark0: float
    dark_exp: float
    use_physical_dark: bool
    T_sensor_C: float
    hot_frac: float
    hot_amp_lo: float
    hot_amp_hi: float
    accidental_e: float
    pc_threshold_k: float
    full_well_e: float
    g_adu: float
    bias_adu: float
    max_adu: float
    # SLM
    slm_levels0: float
    slm_quant_coef: float
    slm_levels_floor: int
    slm_gamma_span: float
    slm_leak_eta: float
    # audit
    setup_id: str
    derived: dict[str, Any] = field(default_factory=dict)


def resolve_forward(
    setup: SetupParams,
    N: int,
    *,
    s_focus: float = 0.0,
) -> ResolvedForward:
    """Map physical setup → forward scales for image size N."""
    N = int(N)

    # --- thermal bundle first (needed for T-tuned M_QIUP / Δz from Δf) ---
    phys_T = None
    if bool(setup.use_physical_thermal):
        phys_T = physical_thermal_bundle(setup)

    # --- V-P1-06: M_QIUP is the single magnification for all µm↔px transforms ---
    mag_used = resolve_imaging_magnification(setup, phys_T=phys_T)
    setup.magnification = float(mag_used)  # keep legacy field synced
    p_obj_um = object_pixel_size_um(setup, magnification=mag_used)

    # --- beam waist (L1/L2) from M_QIUP ---
    if setup.use_legacy_beam_match or setup.pump_waist_um is None:
        w_px = legacy_waist_px(N)
        beam_mode = "legacy_match"
    else:
        w_px = beam_size_px(setup, N=N, magnification=mag_used)
        beam_mode = "physical_waist_M_QIUP"

    # --- pointing (L3 / PDF §2.4–2.5): σ_beam = α·w_px ---
    sigma_beam = sigma_beam_px(w_px, pointing_alpha=setup.pointing_alpha)
    prx = sigma_beam
    pry = sigma_beam
    pointing_mode = "alpha_waist"

    # --- SPDC / QIUP correlation width (mode-dependent NF/FF kernel) ---
    imaging_mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    correlation_type = imaging_correlation_type(imaging_mode)
    blur_model = normalize_blur_model(getattr(setup, "blur_model", None))
    spdc_gain_mode = normalize_spdc_gain_mode(getattr(setup, "spdc_gain_mode", None))
    nbar_gain_mode = spdc_gain_mode_nbar(spdc_gain_mode)
    # Keep setup default nbar synced with the selected gain mode (V-P0-03).
    setup.spdc_default_mean_photons_per_mode = float(nbar_gain_mode)
    if setup.derive_sigma_spdc:
        sig_spdc = psf_size_px(setup, magnification=mag_used)
        if imaging_mode == IMAGING_MODE_FAR:
            spdc_mode = "far_field_momentum_fu_lambda_over_pi_wp"
        else:
            spdc_mode = "near_field_position_pdf_eq35_eq50"
    else:
        sig_spdc = float(setup.sigma_spdc_px)
        spdc_mode = "explicit_px_legacy"

    # --- diffraction (PDF §6.4 eq. 144): σ = η M_QIUP r_Airy / p, NA = D/(2f) ---
    NA_used = resolve_numerical_aperture(setup)
    if bool(setup.use_derived_diffraction):
        sig_diff = sigma_diffraction_from_setup(setup, magnification=mag_used)
        diff_mode = "eq144_airy_M_QIUP_NA_from_D_over_2f"
    else:
        sig_diff = float(setup.sigma_diffraction_px)
        diff_mode = "explicit_px_legacy"

    # --- defocus (PDF §6.5–6.9): c=(NA/n)Δz; K=η M NA/(n p); σ=K|Δz| ---
    K_used = resolve_K_defocus(setup, magnification=mag_used)
    delta_z, delta_z_mode = resolve_delta_z_um(
        setup, s_focus=float(s_focus), phys_T=phys_T
    )
    if bool(setup.use_derived_K_defocus):
        defocus_span = float(K_used) * float(setup.delta_z_max_um)
        defocus_mode = f"derived_K_{delta_z_mode}"
    else:
        defocus_span = float(setup.defocus_span_px)
        # Keep YAML K consistent with span when not derived
        if abs(float(setup.delta_z_max_um)) > 1e-18:
            K_used = float(setup.defocus_span_px) / float(setup.delta_z_max_um)
        defocus_mode = f"legacy_span_{delta_z_mode}"
    coc_um = circle_of_confusion_um(
        delta_z_um=delta_z,
        NA=NA_used,
        n_medium=float(setup.n_medium),
    )
    s_focus_dz = s_focus_from_delta_z(delta_z, setup.delta_z_max_um)
    sigma_defocus_now = sigma_defocus_from_delta_z(delta_z, K_used)
    sig_aberr = resolve_sigma_aberration_px(setup, magnification=mag_used)
    c4_c7 = c4_c7_focus_audit(
        setup,
        delta_z_um=delta_z,
        delta_z_mode=delta_z_mode,
        K_defocus_px_per_um=K_used,
        sigma_defocus_px=sigma_defocus_now,
        s_focus_eq166=s_focus_dz,
        phys_T=phys_T,
    )

    w1_um, w2_um = resolved_idler_waist_um(setup)
    eta_mode = eta_mode_from_setup(setup)
    overlap_mode = (
        "analytic_gaussian_eq63" if setup.use_analytic_mode_overlap else "legacy_delta_s_align"
    )
    therm = crystal_thermal_state(setup)
    if phys_T is None:
        dL_eff = effective_delta_L_mm(setup)
        dphi_th = thermal_phase_offset_rad(setup)
        dx_um = dy_um = 0.0
        R_spdc = 1.0
        M_qiup = float(mag_used)
        df_um = 0.0
    else:
        dL_eff = float(phys_T["delta_L_eff_mm"])
        dphi_th = float(phys_T["thermal_phase_offset_rad"])
        dx_um = float(phys_T["beam_dx_um"])
        dy_um = float(phys_T["beam_dy_um"])
        R_spdc = float(phys_T["R_SPDC_sinc2"])
        M_qiup = float(phys_T["M_QIUP"])
        df_um = float(phys_T["lens_delta_f_um"])
        eta_mode = float(phys_T["eta_mode_thermal"])

    p_int = internal_camera_sampling_pixel_um(setup)
    dx_px = um_to_px(dx_um, pixel_um=p_int, magnification=mag_used)
    dy_px = um_to_px(dy_um, pixel_um=p_int, magnification=mag_used)
    k0_si = k_spdc_0_from_setup(setup)
    spatial_mapping = (
        resolved_spatial_mapping(setup, phys_T=phys_T)
        if setup.architecture_propagation_model
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
        else None
    )

    # Physical thermal: no engineered per-frame walks
    if phys_T is not None:
        phase_walk = 0.0
        vis_walk = 0.0
        beam_walk_x = 0.0
        beam_walk_y = 0.0
    else:
        phase_walk = float(setup.phase_drift_rad_per_frame_at_s1)
        vis_walk = float(setup.vis_walk_per_frame_at_s1)
        beam_walk_x = float(setup.beam_drift_x_px_per_frame_at_s1)
        beam_walk_y = float(setup.beam_drift_y_px_per_frame_at_s1)

    flux0 = float(setup.flux0_e_per_s)
    # Report §7.10: Physical Mode Δk_QPM is already in nbar (§7.9).
    # Do not also scale flux0 by R_SPDC_sinc2. Keep R as diagnostics only.
    flux0_includes_R = not bool(getattr(setup, "use_physical_spdc_gain", False))
    if flux0_includes_R:
        flux0 *= float(R_spdc)

    return ResolvedForward(
        N=N,
        w_px=float(w_px),
        pointing_rms_x=float(prx),
        pointing_rms_y=float(pry),
        sigma_spdc_px=float(sig_spdc),
        imaging_mode=str(imaging_mode),
        correlation_type=str(correlation_type),
        blur_model=str(blur_model),
        spdc_gain_mode=str(spdc_gain_mode),
        sigma_diffraction_px=float(sig_diff),
        defocus_span_px=float(defocus_span),
        K_defocus_px_per_um=float(K_used),
        delta_z_um=float(delta_z),
        coc_um=float(coc_um),
        s_focus_from_dz=float(s_focus_dz),
        n_medium=float(setup.n_medium),
        eta_defocus=float(setup.eta_defocus),
        sigma_aberration_px=float(sig_aberr),
        delta_M_rel=float(setup.delta_M_rel),
        image_scale=float(resolve_image_scale(setup)),
        image_rotation_deg=float(setup.image_rotation_deg),
        V0=float(setup.V0),
        overlap_delta_scale=float(setup.overlap_delta_scale),
        use_analytic_mode_overlap=bool(setup.use_analytic_mode_overlap),
        eta_mode=float(eta_mode),
        delta_L_mm=float(validate_delta_L_mm(setup.delta_L_mm)),
        delta_L_eff_mm=float(dL_eff),
        L_coh_mm=float(validate_L_coh_mm(setup.L_coh_mm)),
        path1_intensity=float(setup.path1_intensity),
        path2_intensity=float(setup.path2_intensity),
        idler_path_transmission=float(idler_path_transmission_from_setup(setup)),
        use_physical_two_beam=bool(setup.use_physical_two_beam),
        use_field_overlap_visibility_map=bool(setup.use_field_overlap_visibility_map),
        V_gain_drop=float(setup.V_gain_drop),
        V_map_mottle=float(setup.V_map_mottle),
        mottle_smooth_sigma_px=float(setup.mottle_smooth_sigma_px),
        use_physical_thermal=bool(setup.use_physical_thermal),
        n_crystals=int(therm["n_crystals"]),
        dT_eff_C=float(therm["dT_eff_C"]),
        thermal_phase_offset_rad=float(dphi_th),
        beam_thermal_dx_um=float(dx_um),
        beam_thermal_dy_um=float(dy_um),
        beam_thermal_dx_px=float(dx_px),
        beam_thermal_dy_px=float(dy_px),
        lens_delta_f_um=float(df_um),
        R_SPDC_sinc2=float(R_spdc),
        M_QIUP=float(M_qiup),
        phase_drift_rad_per_frame=float(phase_walk),
        vis_walk_per_frame=float(vis_walk),
        beam_drift_x_px_per_frame=float(beam_walk_x),
        beam_drift_y_px_per_frame=float(beam_walk_y),
        phase_jitter_align=float(setup.phase_jitter_align_at_s1),
        flux0_e_per_s=float(flux0),
        pump_sigma_rel=float(setup.pump_sigma_rel_at_s1),
        flux_boost=float(setup.flux_boost_at_s1),
        QE=float(setup.QE),
        read_noise0_e=float(setup.read_noise0_e),
        read_noise_span_e=float(setup.read_noise_span_e),
        cic0=float(setup.cic0),
        prnu_sigma=float(setup.prnu_sigma),
        dark0=float(setup.dark0),
        dark_exp=float(setup.dark_exp),
        use_physical_dark=bool(setup.use_physical_dark),
        T_sensor_C=float(setup.T_sensor_C),
        hot_frac=float(setup.hot_frac_at_s1),
        hot_amp_lo=float(setup.hot_amp_lo),
        hot_amp_hi=float(setup.hot_amp_hi),
        accidental_e=float(setup.accidental_e_at_s1),
        pc_threshold_k=float(setup.pc_threshold_k),
        full_well_e=float(setup.full_well_e),
        g_adu=float(setup.g_adu),
        bias_adu=float(setup.bias_adu),
        max_adu=float(setup.max_adu),
        slm_levels0=float(setup.slm_levels0),
        slm_quant_coef=float(setup.slm_quant_coef),
        slm_levels_floor=int(setup.slm_levels_floor),
        slm_gamma_span=float(setup.slm_gamma_span),
        slm_leak_eta=float(setup.slm_leak_eta),
        setup_id=str(setup.setup_id),
        derived={
            "beam_mode": beam_mode,
            "pointing_mode": pointing_mode,
            "diffraction_mode": diff_mode,
            "defocus_mode": defocus_mode,
            "delta_z_mode": delta_z_mode,
            "spdc_mode": spdc_mode,
            "imaging_mode": imaging_mode,
            "correlation_type": correlation_type,
            "blur_model": blur_model,
            "blur_model_single": BLUR_MODEL_SINGLE,
            "blur_model_double": BLUR_MODEL_DOUBLE,
            "spdc_gain_mode": spdc_gain_mode,
            "spdc_gain_mode_nbar": float(nbar_gain_mode),
            "spdc_default_mean_photons_per_mode": float(
                setup.spdc_default_mean_photons_per_mode
            ),
            "far_field_focal_mm": resolved_far_field_focal_mm(setup),
            "sigma_correlation_um": float(sigma_correlation_um(setup)),
            "sigma_near_field_um": float(sigma_near_field_um(setup)),
            "sigma_far_field_um": float(sigma_far_field_um(setup)),
            "spatial_mapping": spatial_mapping,
            "enable_etalon_fringes": bool(setup.enable_etalon_fringes),
            "etalon_contrast": float(setup.etalon_contrast),
            "etalon_period_px": float(setup.etalon_period_px),
            "etalon_angle_deg": float(setup.etalon_angle_deg),
            "etalon_phase_rad": float(setup.etalon_phase_rad),
            "etalon_model": "multiplicative_cosine_intensity",
            "delta_z_um": delta_z,
            "coc_um": coc_um,
            "coc_formula": "(NA/n)*delta_z",
            "K_defocus_px_per_um": K_used,
            "K_defocus_formula": "eta_def*M*NA/(n*pixel_um)",
            "use_derived_K_defocus": setup.use_derived_K_defocus,
            "sigma_defocus_px": sigma_defocus_now,
            "sigma_defocus_max_px": defocus_span,
            "sigma_defocus_eq": "K_defocus*|delta_z|",
            "s_focus_from_dz": s_focus_dz,
            "s_focus_eq": "|delta_z|/delta_z_max",
            "sigma_opt_eq": "sqrt(sigma_diffraction^2 + sigma_defocus^2 + sigma_aberration^2)",
            "optical_psf_model": "effective_gaussian",
            "eq181_components": {
                "sigma_diffraction_px": float(sig_diff),
                "sigma_defocus_px": float(sigma_defocus_now),
                "sigma_aberration_px": float(sig_aberr),
            },
            "sigma_aberration_px": float(sig_aberr),
            "sigma_aberration_eq": "eta_ab*W_waves*(M*r_Airy/p)",
            "use_derived_aberration": setup.use_derived_aberration,
            "wavefront_rms_waves": setup.wavefront_rms_waves,
            "wavefront_rms_rad": wavefront_rms_rad(setup.wavefront_rms_waves),
            "eta_aberration": setup.eta_aberration,
            "couple_align_to_wavefront": setup.couple_align_to_wavefront,
            "delta_phi_align_span_rad": setup.delta_phi_align_span_rad,
            "wavefront_corr_px": setup.wavefront_corr_px,
            "vignetting_enabled": False,
            "delta_M_rel": float(setup.delta_M_rel),
            "epsilon_M_eq": "delta_M / M",
            "image_scale": float(resolve_image_scale(setup)),
            "image_scale_eq": "1 + delta_M_rel",
            "image_rotation_deg": float(setup.image_rotation_deg),
            "geometric_iq_note": "scale/rotation are geometric (PDF §6.17-6.18), not folded into sigma_opt",
            "zernike_enabled": False,
            "c4_c7": c4_c7,
            "n_medium": setup.n_medium,
            "eta_defocus": setup.eta_defocus,
            "delta_z_max_um": setup.delta_z_max_um,
            "legacy_waist_px_for_N": legacy_waist_px(N),
            "NA": NA_used,
            "NA_formula": "D/(2*f)",
            "aperture_diameter_mm": setup.aperture_diameter_mm,
            "lens_focal_mm": setup.lens_focal_mm,
            "airy_to_gauss_eta": setup.airy_to_gauss_eta,
            "r_Airy_um": airy_radius_um(lambda_s_nm=setup.lambda_s_nm, NA=NA_used),
            "sigma_diffraction_eq": "eta * M * r_Airy / pixel_um",
            "use_derived_diffraction": setup.use_derived_diffraction,
            "pixel_um": setup.pixel_um,
            "pixel_um_min": PIXEL_UM_MIN,
            "pixel_um_max": PIXEL_UM_MAX,
            "pixel_um_default": PIXEL_UM_DEFAULT,
            "object_pixel_size_um": float(p_obj_um),
            "beam_size_px": float(w_px),
            "PSF_size_px": float(sig_spdc),
            "magnification": mag_used,
            "magnification_synced_from_M_QIUP": True,
            "M_QIUP": M_qiup,
            "M_QIUP_formula": "F2*lambda_s/(F1*lambda_i)",
            "M_QIUP_is_source_of_truth": True,
            "use_physical_qiup_magnification": True,  # retained for schema; always on (V-P1-06)
            "F1_mm": setup.F1_mm,
            "F2_mm": setup.F2_mm,
            "lambda_s_nm": setup.lambda_s_nm,
            "pump_waist_um": setup.pump_waist_um,
            "overlap_mode": overlap_mode,
            "use_analytic_mode_overlap": setup.use_analytic_mode_overlap,
            "idler_waist_um": w1_um,
            "idler_waist_2_um": w2_um,
            "idler_dx_um": setup.idler_dx_um,
            "idler_dy_um": setup.idler_dy_um,
            "idler_dtheta_mrad": setup.idler_dtheta_mrad,
            "generation_dx_um": float(getattr(setup, "generation_dx_um", 0.0) or 0.0),
            "generation_dy_um": float(getattr(setup, "generation_dy_um", 0.0) or 0.0),
            "generation_dtheta_mrad": float(
                getattr(setup, "generation_dtheta_mrad", 0.0) or 0.0
            ),
            "relay_dx_um": getattr(setup, "relay_dx_um", None),
            "relay_dy_um": getattr(setup, "relay_dy_um", None),
            "relay_dtheta_mrad": getattr(setup, "relay_dtheta_mrad", None),
            "relay_alignment_um": list(relay_alignment_um(setup)),
            "generation_alignment_um": list(generation_alignment_um(setup)),
            "alignment_split": "generation_Ogeom_vs_relay_eta_mode",
            "inherit_idler_displacement_from_pump": setup.inherit_idler_displacement_from_pump,
            "eta_mode": eta_mode,
            "lambda_i_nm": setup.lambda_i_nm,
            "path1_intensity": setup.path1_intensity,
            "path2_intensity": setup.path2_intensity,
            "idler_path_transmission": setup.idler_path_transmission,
            "signal_path1_transmission": setup.path1_intensity,
            "signal_path2_transmission": setup.path2_intensity,
            "use_physical_two_beam": setup.use_physical_two_beam,
            "use_field_overlap_visibility_map": setup.use_field_overlap_visibility_map,
            "V_port": two_beam_visibility(setup.path1_intensity, setup.path2_intensity),
            "L_coh_mm": setup.L_coh_mm,
            "L_coh_mm_min": L_COH_MM_MIN,
            "L_coh_mm_max": L_COH_MM_MAX,
            "L_coh_mm_default": L_COH_MM_DEFAULT,
            "delta_L_mm": setup.delta_L_mm,
            "delta_L_mm_min": DELTA_L_MM_MIN,
            "delta_L_mm_max": DELTA_L_MM_MAX,
            "delta_L_mm_default": DELTA_L_MM_DEFAULT,
            "delta_L_eff_mm": dL_eff,
            "V_coh_formula": "exp(-(abs(OPD)/L_coh)^2)",
            "OPD_formula": "delta_L0 + delta_L_thermal (signed; abs only for V_coh)",
            "pointing_alpha": setup.pointing_alpha,
            "pointing_alpha_min": POINTING_ALPHA_MIN,
            "pointing_alpha_max": POINTING_ALPHA_MAX,
            "sigma_beam_px": sigma_beam,
            "eta_overlap_formula": "exp(-(dx^2+dy^2)/(2*w_px^2))",
            "couple_pointing_overlap_visibility": setup.couple_pointing_overlap_visibility,
            "T_env_C": setup.T_env_C,
            "n_crystals": therm["n_crystals"],
            "n_crystals_hardware": int(setup.n_crystals),
            "n_spdc_interactions": n_spdc_interactions(setup),
            "spdc_architecture": normalize_spdc_architecture(
                getattr(setup, "spdc_architecture", None)
            ),
            "spdc_shared_crystal": (
                spdc_shared_crystal_state(setup)
                if bool(setup.use_physical_spdc_gain)
                else None
            ),
            "spdc_crystal_NL1": (
                spdc_crystal_state(setup, interaction=1)
                if bool(setup.use_physical_spdc_gain)
                else None
            ),
            "spdc_crystal_NL2": (
                spdc_crystal_state(setup, interaction=2)
                if bool(setup.use_physical_spdc_gain)
                else None
            ),
            "use_physical_thermal": setup.use_physical_thermal,
            "sync_crystal_temperatures": setup.sync_crystal_temperatures,
            "T_pm0_NL1_C": therm["T_pm0_NL1_C"],
            "T_pm0_NL2_C": therm["T_pm0_NL2_C"],
            "delta_T_NL1_C": therm["delta_T_NL1_C"],
            "delta_T_NL2_C": therm["delta_T_NL2_C"],
            "T_pm_NL1_C": therm["T_pm_NL1_C"],
            "T_pm_NL2_C": therm["T_pm_NL2_C"],
            "T_crystal_NL1_C": therm["T_crystal_NL1_C"],
            "T_crystal_NL2_C": therm["T_crystal_NL2_C"],
            "dT_NL1_C": therm["dT_NL1_C"],
            "dT_NL2_C": therm["dT_NL2_C"],
            "dT_NL_C": therm["dT_NL_C"],
            "dT_eff_C": therm["dT_eff_C"],
            "dT_rel_C": therm["dT_rel_C"],
            "T_pm_C": setup.T_pm_C,
            "T_crystal_C": setup.T_crystal_C,
            "T_crystal_C_min": T_CRYSTAL_C_MIN,
            "T_crystal_C_max": T_CRYSTAL_C_MAX,
            "T_crystal_C_default": T_CRYSTAL_C_DEFAULT,
            "T_crystal_C_step": T_CRYSTAL_C_STEP,
            "eq80": "T_pm = T_pm0 + delta_T",
            "alpha_path_per_C": setup.alpha_path_per_C,
            "L_eff_path_mm": setup.L_eff_path_mm,
            "alpha_L_mm_per_C": setup.alpha_L_mm_per_C,
            "thermal_phase_offset_rad": dphi_th,
            "theta0_rad": THETA0_RAD,
            "oven_tau_s": OVEN_TAU_S,
            "beam_thermal_dx_um": dx_um,
            "beam_thermal_dy_um": dy_um,
            "lens_delta_f_um": df_um,
            "R_SPDC_sinc2": R_spdc,
            "flux0_includes_R_SPDC_sinc2": flux0_includes_R,
            "M_QIUP": M_qiup,
            "crystal_material": setup.crystal_material,
            "qpm_type": setup.qpm_type,
            "d33_pm_per_V": setup.d33_pm_per_V,
            "d_eff_pm_per_V": setup.d_eff_pm_per_V,
            **effective_nonlinearity_metadata(setup),
            "d_eff_precedence": setup.d_eff_precedence,
            "d_eff_pm_per_V_derived": (
                None
                if setup.d33_pm_per_V is None
                else d_eff_pm_per_V(
                    d33_pm_per_V=float(setup.d33_pm_per_V),
                    qpm_order=int(setup.qpm_order),
                    qpm_duty_cycle=float(setup.qpm_duty_cycle),
                )
            ),
            "qpm_order": setup.qpm_order,
            "qpm_duty_cycle": setup.qpm_duty_cycle,
            "k_spdc_0_per_sqrt_W": k0_si,
            "k_spdc_per_sqrt_W": resolved_k_spdc_per_sqrt_W(setup, interaction=1),
            "k_spdc_per_sqrt_W_NL1": resolved_k_spdc_per_sqrt_W(
                setup, interaction=1
            ),
            "k_spdc_per_sqrt_W_NL2": resolved_k_spdc_per_sqrt_W(
                setup, interaction=2
            ),
            "spdc_geometry_NL1": spdc_gaussian_geometry_from_setup(
                setup, interaction=1
            ),
            "spdc_geometry_NL2": spdc_gaussian_geometry_from_setup(
                setup, interaction=2
            ),
            "spdc_geometry_overlap_NL1": spdc_geometry_overlap_from_setup(
                setup, interaction=1
            ),
            "spdc_geometry_overlap_NL2": spdc_geometry_overlap_from_setup(
                setup, interaction=2
            ),
            "O_geom_includes_qpm_phase": OGEOM_INCLUDES_QPM_PHASE,
            "pump_waist_NL1_um": resolved_pump_waist_um(setup, interaction=1),
            "pump_waist_NL2_um": resolved_pump_waist_um(setup, interaction=2),
            "pump_focus_z_NL1_um": resolved_pump_focus_z_um(setup, interaction=1),
            "pump_focus_z_NL2_um": resolved_pump_focus_z_um(setup, interaction=2),
            **pump_powers_from_setup(setup),
            "spdc_gain_per_sqrt_mW": setup.spdc_gain_per_sqrt_mW,
            "spdc_gain_per_sqrt_mW_source": (
                "derived_kSPDC0_Ogeom" if k0_si is not None else "manual_fallback"
            ),
            "poling_period_um": setup.poling_period_um,
            "use_calibrated_poling_period": setup.use_calibrated_poling_period,
            "poling_period_um_qpm": (
                qpm_poling_period_um(setup) if phys_T is not None else None
            ),
            "poling_period_um_eff": (
                calibrated_poling_period_um(setup)
                if (phys_T is not None and setup.use_calibrated_poling_period)
                else None
            ),
            "pol_pump": setup.pol_pump,
            "pol_signal": setup.pol_signal,
            "pol_idler": setup.pol_idler,
            "F1_mm": setup.F1_mm,
            "F2_mm": setup.F2_mm,
            "lens_focal_mm": setup.lens_focal_mm,
            "lens_material": setup.lens_material,
            "physical_thermal": phys_T,
            "slm_bit_depth": setup.slm_bit_depth,
            "slm_fill_factor": setup.slm_fill_factor,
            "slm_pixel_um": setup.slm_pixel_um,
            "slm_phi_max_rad": setup.slm_phi_max_rad,
            "emccd_model": setup.emccd_model,
            "detector_mode": normalize_detector_mode(
                getattr(setup, "detector_mode", None)
            ),
            "read_noise_model": setup.read_noise_model,
            "hot_pixel_model": setup.hot_pixel_model,
            "physics_tier": setup.physics_tier,
            "model_version": setup.model_version,
            "sigma_blur_extra_px": setup.sigma_blur_extra_px,
            "delta_M_rel": setup.delta_M_rel,
            "image_scale": setup.image_scale,
            "image_rotation_deg": setup.image_rotation_deg,
            "slm_misreg_x_px": setup.slm_misreg_x_px,
            "slm_misreg_y_px": setup.slm_misreg_y_px,
            "sampling_regime": setup.sampling_regime,
            "coupling_profile_id": setup.coupling_profile_id,
            "crystal_length_mm": setup.crystal_length_mm,
            "crystal_length_mm_min": CRYSTAL_LENGTH_MM_MIN,
            "crystal_length_mm_max": CRYSTAL_LENGTH_MM_MAX,
            "crystal_length_policy": crystal_length_policy(
                getattr(setup, "physics_policy_id", None)
            ),
            "lambda_p_nm": setup.lambda_p_nm,
            "sigma_r_um": sigma_r_um(
                crystal_length_mm=setup.crystal_length_mm,
                lambda_p_nm=setup.lambda_p_nm,
            ),
            "sigma_spdc_px": float(sig_spdc),
        },
    )


def derive_sigma_spdc_px(setup: SetupParams) -> float:
    """Correlation PSF width [px] for the active imaging_mode (V-P1-06).

    Near-field (default): P07/P10 — σ = M_QIUP · [Δ_V(L,λ_s,λ_i,w_p)/√2] / p
    Far-field: P14+P07 — σ = M_QIUP · [m_i f_u λ_i/(2π w_p)] / p

    This is an intensity-level QIUP spatial-transfer width, not a biphoton Φ model.
    Always uses ``M_QIUP`` (never a separate legacy magnification).
    """
    return psf_size_px(setup)


def sensor_temperature_C(setup: SetupParams, s_cool: float) -> float:
    """M5: map cooling latent → sensor temperature [°C].

    s_cool=0 → T_sensor_min (best cool); s_cool=1 → T_sensor_max.
    """
    return float(
        setup.T_sensor_min_C
        + float(s_cool) * (setup.T_sensor_max_C - setup.T_sensor_min_C)
    )


def read_noise_e_from_setup(setup: SetupParams, s_det: float) -> float:
    """M1: output-referred read noise [e-] after EM register."""
    if setup.read_noise_model == "datasheet_fixed":
        return float(setup.read_noise0_e)
    return float(setup.read_noise0_e) + float(setup.read_noise_span_e) * float(s_det)


def dark_rate_from_setup(setup: SetupParams, s_cool: float) -> float:
    """Dark current [e-/px/s]. Legacy exponential unless use_physical_dark (M5)."""
    if not setup.use_physical_dark:
        return float(setup.dark0 * np.exp(setup.dark_exp * float(s_cool)))
    T_C = sensor_temperature_C(setup, s_cool)
    T_K = T_C + 273.15
    k_eV = 8.617333262145e-5
    # Normalize so at s_cool=0 (T_min) rate ≈ dark0 (calibration anchor)
    T0 = setup.T_sensor_min_C + 273.15
    scale = setup.dark0 / max(
        (T0 ** 1.5) * np.exp(-setup.Eg_eV / (2.0 * k_eV * T0)), 1e-30
    )
    return float(scale * (T_K ** 1.5) * np.exp(-setup.Eg_eV / (2.0 * k_eV * T_K)))


# ---------------------------------------------------------------------------
# YAML I/O (L12) — fail-closed schema (V-P0-10)
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Setup YAML/JSON failed schema checks (unknown keys, bad structure)."""


def package_dir() -> Path:
    """Directory that contains the five source-of-truth modules."""
    return Path(__file__).resolve().parent


def canonical_yaml_path() -> Path:
    """Authoritative CFG-P0-00 YAML (frozen approved system)."""
    return package_dir() / CANONICAL_YAML_RELATIVE


def canonical_hash_path() -> Path:
    """Recorded SHA-256 fixture for the frozen canonical YAML + SetupParams."""
    return package_dir() / CANONICAL_HASH_RELATIVE


def hash_yaml_file(path: str | Path) -> str:
    """SHA-256 of UTF-8 YAML with newlines normalized to ``\\n`` (CFG-P0-00)."""
    text = Path(path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_recorded_canonical_hashes() -> dict[str, Any]:
    """Load the frozen hash fixture. Missing/malformed file is fail-closed."""
    path = canonical_hash_path()
    if not path.is_file():
        raise ConfigError(
            f"CFG-P0-00: missing canonical hash fixture {path}. "
            "Record yaml_sha256 and config_hash before official runs."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"CFG-P0-00: invalid hash fixture {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"CFG-P0-00: hash fixture must be a mapping: {path}")
    yaml_h = data.get("yaml_sha256")
    cfg_h = data.get("config_hash")
    length_policy = data.get("crystal_length_policy")
    if not isinstance(yaml_h, str) or not yaml_h.strip():
        raise ConfigError(f"CFG-P0-00: hash fixture missing yaml_sha256: {path}")
    if not isinstance(cfg_h, str) or not cfg_h.strip():
        raise ConfigError(f"CFG-P0-00: hash fixture missing config_hash: {path}")
    if not isinstance(length_policy, dict):
        raise ConfigError(
            f"CFG-P0-00: hash fixture missing crystal_length_policy: {path}"
        )
    return {
        "yaml_sha256": yaml_h.strip().lower(),
        "config_hash": cfg_h.strip().lower(),
        "yaml_path": str(data.get("yaml_path", CANONICAL_YAML_RELATIVE)),
        "finding": str(data.get("finding", "CFG-P0-00")),
        "crystal_length_policy": length_policy,
    }


def _canonical_field_value(name: str, value: Any) -> Any:
    """Normalize a frozen identity field for comparison."""
    if name == "n_crystals":
        return int(value)
    if name in {
        "crystal_length_mm",
        "lambda_p_nm",
        "lambda_s_nm",
        "lambda_i_nm",
    }:
        return float(value)
    if name == "spdc_architecture":
        return normalize_spdc_architecture(value)
    if name == "detector_mode":
        return normalize_detector_mode(value)
    if name == "blur_model":
        return normalize_blur_model(value)
    if name == "spdc_gain_mode":
        return normalize_spdc_gain_mode(value)
    if name == "spdc_source_mode":
        return normalize_spdc_source_mode(value)
    return value


def reject_canonical_overrides(kwargs: dict[str, Any]) -> None:
    """Fail-closed if a caller tries to change a frozen CFG-P0-00 field."""
    conflicts: list[str] = []
    for name, approved in CANONICAL_FROZEN_FIELDS.items():
        if name not in kwargs:
            continue
        got = _canonical_field_value(name, kwargs[name])
        want = _canonical_field_value(name, approved)
        if got != want:
            conflicts.append(f"{name}={got!r} (canonical {want!r})")
    if conflicts:
        raise ConfigError(
            "CFG-P0-00: refusing conflicting override of frozen canonical "
            f"fields: {', '.join(conflicts)}"
        )


def canonical_identity_errors(setup: "SetupParams") -> list[str]:
    """Return frozen-field mismatches vs the approved system (empty ⇒ pass)."""
    bad: list[str] = []
    for name, approved in CANONICAL_FROZEN_FIELDS.items():
        got = _canonical_field_value(name, getattr(setup, name))
        want = _canonical_field_value(name, approved)
        if got != want:
            bad.append(f"{name}={got!r} (canonical {want!r})")
    return bad


def is_canonical_setup(setup: "SetupParams") -> bool:
    return not canonical_identity_errors(setup)


def assert_canonical_setup(setup: "SetupParams") -> "SetupParams":
    """Require the approved-system frozen identity (CFG-P0-00)."""
    bad = canonical_identity_errors(setup)
    if bad:
        raise ConfigError(
            "CFG-P0-00: setup is not the approved canonical configuration: "
            + "; ".join(bad)
        )
    return setup


def load_canonical_setup() -> "SetupParams":
    """Load the frozen YAML and assert CFG-P0-00 identity."""
    path = canonical_yaml_path()
    if not path.is_file():
        raise ConfigError(f"CFG-P0-00: missing canonical YAML {path}")
    setup = load_setup_yaml(path)
    return assert_canonical_setup(setup)


def resolved_canonical_config_hash(setup: "SetupParams" | None = None) -> str:
    """SHA-256 of the resolved SetupParams used in run manifests."""
    from provenance import hash_config

    return hash_config(setup if setup is not None else load_canonical_setup())


def verify_canonical_file_hashes(
    setup: "SetupParams" | None = None,
) -> dict[str, str]:
    """Compare the frozen YAML file + resolved canonical SetupParams to the fixture.

    The live ``setup`` (if given) must still match the frozen identity fields.
    Its full ``config_hash`` may differ after later copies/derived fills; the
    recorded hash is always that of ``load_canonical_setup()``.
    """
    recorded = load_recorded_canonical_hashes()
    recorded_policy = recorded.get("crystal_length_policy")
    live_policy = crystal_length_policy()
    if recorded_policy != live_policy:
        raise ConfigError(
            "CFG-P0-00: recorded crystal-length policy does not match the "
            f"physics-team-approved code policy. live={live_policy!r} "
            f"recorded={recorded_policy!r}"
        )
    yaml_path = canonical_yaml_path()
    live_yaml = hash_yaml_file(yaml_path)
    if live_yaml != recorded["yaml_sha256"]:
        raise ConfigError(
            "CFG-P0-00: canonical YAML SHA-256 does not match the frozen "
            f"fixture. live={live_yaml} recorded={recorded['yaml_sha256']}"
        )
    live_cfg = resolved_canonical_config_hash()
    if live_cfg != recorded["config_hash"]:
        raise ConfigError(
            "CFG-P0-00: resolved config_hash does not match the frozen "
            f"fixture. live={live_cfg} recorded={recorded['config_hash']}"
        )
    if setup is not None:
        assert_canonical_setup(setup)
    return {
        "yaml_sha256": live_yaml,
        "config_hash": live_cfg,
        "yaml_path": str(yaml_path.as_posix()),
    }


def canonical_config_audit(
    *,
    verified: bool = True,
    hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """CFG-P0-00 audit block for run / dataset manifests.

    Always records ``SHA256(canonical YAML bytes)`` under both the comment-1
    keys and the legacy aliases used by existing consumers.
    """
    verified_hashes = hashes if hashes is not None else verify_canonical_file_hashes()
    yaml_sha = str(verified_hashes["yaml_sha256"])
    cfg_sha = str(verified_hashes["config_hash"])
    return {
        "canonical_config_file": CANONICAL_YAML_RELATIVE.replace("\\", "/"),
        "canonical_config_sha256": yaml_sha,
        "canonical_config_verified": bool(verified),
        # Legacy / internal aliases (same bytes / same fixture).
        "canonical_yaml_sha256": yaml_sha,
        "canonical_config_hash": cfg_sha,
        "canonical_finding": "CFG-P0-00",
        "crystal_length_policy": crystal_length_policy(),
    }


SETUP_YAML_NESTED_SECTIONS: frozenset[str] = frozenset(
    {
        "architecture",
        "source",
        "pump",
        "physical_crystal",
        "interaction_1",
        "interaction_2",
        "object",
        "signal_filter",
        "geometry",
        "beam",
        "visibility",
        "thermal",
        "detector",
        "camera_display",
        "slm",
        "image_quality",
        "couplings",
        "meta",
        "phase3",
        "apparatus",
        "calibration",
    }
)

# Non-schema keys still accepted briefly so from_dict migrations can run, then
# they are dropped before construction. Unknown names outside this set raise
# ConfigError via ``validate_setup_keys`` (Comment 4.1).
SETUP_YAML_LEGACY_ALIASES: frozenset[str] = frozenset(
    {
        "crystal_length_um",
        "delta_L0_mm",
        "n_physical_crystals",
        "crystal_temperature_C",
        "crystal_1_temperature_C",
        "crystal_2_temperature_C",
        "crystal_1_length_mm",
        "crystal_2_length_mm",
        "crystal_1_poling_period_um",
        "crystal_2_poling_period_um",
        "T_crystal_NL_C",
        "T_pm0_NL_C",
        "operating_temperature_NL_C",
        "operating_temperature_NL1_C",
        "operating_temperature_NL2_C",
        "phase_matching_temperature_C",
        "phase_matching_temperature_NL_C",
        "phase_matching_temperature_NL1_C",
        "phase_matching_temperature_NL2_C",
        "crystal_1_phase_matching_temperature_C",
        "crystal_2_phase_matching_temperature_C",
        "dT_C",
        "dT_NL_C",
        "operating_detuning_C",
        "operating_detuning_NL_C",
        "operating_detuning_NL1_C",
        "operating_detuning_NL2_C",
        "crystal_1_dT_C",
        "crystal_2_dT_C",
        *SETUP_SWEEP_ALIASES.keys(),
    }
)

# Convenience kwargs accepted by ``SetupParams.overridden`` then rewritten.
SETUP_OVERRIDE_CONVENIENCE_KEYS: frozenset[str] = frozenset(
    {
        "T_pm0_C",
        "delta_T_C",
        "dT_C",
    }
)


def setup_schema_keys() -> set[str]:
    """Canonical SetupParams field names (YAML / dict / override schema)."""
    return {f.name for f in fields(SetupParams)}


# Comment 4.4 — freeze SetupParams sweep allow-list now that the dataclass exists.
SETUP_SWEEP_FIELDS = _compute_setup_sweep_fields()


def validate_setup_keys(
    data: dict[str, Any] | Any,
    *,
    allow_legacy: bool = True,
    extra_allowed: frozenset[str] | set[str] | None = None,
    context: str = "SetupParams",
) -> None:
    """Fail-closed key gate for programmatic configuration (Comment 4.1).

    Rejects unknown names before they can be silently dropped by ``from_dict``
    / ``overridden``. Legacy migration aliases may be allowed briefly; they
    must be consumed by migrations and never treated as free-form extras.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{context}: expected a mapping of fields, got {type(data).__name__}")
    valid = setup_schema_keys()
    if allow_legacy:
        valid |= set(SETUP_YAML_LEGACY_ALIASES)
    if extra_allowed:
        valid |= {str(x) for x in extra_allowed}
    unknown = {str(k) for k in data} - valid
    if unknown:
        raise ConfigError(
            f"Unknown SetupParams field(s): {sorted(unknown)}"
        )


def _flatten_setup_yaml_mapping(
    data: dict[str, Any], *, source: str | Path
) -> dict[str, Any]:
    """Flatten nested setup sections; reject unknown section names."""
    flat: dict[str, Any] = {}
    for k, v in data.items():
        key = str(k)
        if isinstance(v, dict) and key in SETUP_YAML_NESTED_SECTIONS:
            for nk, nv in v.items():
                if isinstance(nv, dict):
                    raise ConfigError(
                        f"Nested mapping not allowed under '{key}.{nk}' in {source}"
                    )
                nested_key = str(nk)
                if nested_key in flat:
                    raise ConfigError(
                        f"Duplicate setup field {nested_key!r} while flattening {source}"
                    )
                flat[nested_key] = nv
        elif isinstance(v, dict):
            raise ConfigError(
                f"Unknown nested setup section {key!r} in {source}. "
                f"Allowed sections: {sorted(SETUP_YAML_NESTED_SECTIONS)}"
            )
        else:
            if key in flat:
                raise ConfigError(f"Duplicate setup field {key!r} in {source}")
            flat[key] = v
    return flat


def _migrate_setup_yaml_aliases(flat: dict[str, Any]) -> dict[str, Any]:
    """Apply legacy YAML aliases, then drop non-schema alias keys."""
    data = dict(flat)
    aliases = {
        "n_physical_crystals": "n_crystals",
        "crystal_temperature_C": "T_crystal_NL1_C",
        "crystal_1_temperature_C": "T_crystal_NL1_C",
        "crystal_2_temperature_C": "T_crystal_NL2_C",
        "T_crystal_NL_C": "T_crystal_NL1_C",
        "operating_temperature_NL_C": "T_crystal_NL1_C",
        "operating_temperature_NL1_C": "T_crystal_NL1_C",
        "operating_temperature_NL2_C": "T_crystal_NL2_C",
        "T_pm0_NL_C": "T_pm0_NL1_C",
        "phase_matching_temperature_C": "T_pm0_NL1_C",
        "phase_matching_temperature_NL_C": "T_pm0_NL1_C",
        "phase_matching_temperature_NL1_C": "T_pm0_NL1_C",
        "phase_matching_temperature_NL2_C": "T_pm0_NL2_C",
        "crystal_1_phase_matching_temperature_C": "T_pm0_NL1_C",
        "crystal_2_phase_matching_temperature_C": "T_pm0_NL2_C",
        "dT_C": "dT_NL1_C",
        "dT_NL_C": "dT_NL1_C",
        "operating_detuning_C": "dT_NL1_C",
        "operating_detuning_NL_C": "dT_NL1_C",
        "operating_detuning_NL1_C": "dT_NL1_C",
        "operating_detuning_NL2_C": "dT_NL2_C",
        "crystal_1_dT_C": "dT_NL1_C",
        "crystal_2_dT_C": "dT_NL2_C",
        "crystal_1_length_mm": "crystal_length_mm",
        "crystal_2_length_mm": "crystal_length_NL2_mm",
        "crystal_1_poling_period_um": "poling_period_um",
        "crystal_2_poling_period_um": "poling_period_NL2_um",
    }
    for alias, target in aliases.items():
        if alias not in data:
            continue
        if target in data and data[target] != data[alias]:
            raise ConfigError(
                f"Conflicting values for {alias!r} and canonical field {target!r}"
            )
        data[target] = data[alias]
        del data[alias]
    # A shared physical temperature applies to both passes unless an explicit
    # crystal-2 temperature is declared for the two-crystal architecture.
    if "crystal_temperature_C" in flat:
        value = flat["crystal_temperature_C"]
        if "T_crystal_NL2_C" in data and data["T_crystal_NL2_C"] != value:
            raise ConfigError(
                "crystal_temperature_C conflicts with crystal-2 temperature"
            )
        data["T_crystal_NL2_C"] = value
    if "crystal_length_mm" not in data and "crystal_length_um" in data:
        Lmm = float(data["crystal_length_um"]) / 1000.0
        if Lmm < CRYSTAL_LENGTH_MM_MIN:
            Lmm = CRYSTAL_LENGTH_MM_DEFAULT
        data["crystal_length_mm"] = Lmm
    if "delta_L_mm" not in data and "delta_L0_mm" in data:
        data["delta_L_mm"] = float(data["delta_L0_mm"])
    for alias, target in SETUP_SWEEP_ALIASES.items():
        if alias not in data:
            continue
        if target not in data:
            data[target] = data[alias]
        del data[alias]
    for alias in ("crystal_length_um", "delta_L0_mm"):
        data.pop(alias, None)
    return data


def _require_yaml_phase_match_temperatures(flat: dict[str, Any]) -> dict[str, Any]:
    """User YAML must declare equilibrium phase-matching temperature(s)."""
    data = dict(flat)
    architecture = normalize_spdc_architecture(
        data.get("spdc_architecture", SPDC_ARCHITECTURE_DEFAULT)
    )
    if "T_pm0_NL1_C" not in data:
        raise ConfigError(
            "phase-matching temperature is required "
            "(T_pm0_NL_C / T_pm0_NL1_C / phase_matching_temperature_C). "
            "This is the equilibrium QPM temperature, not the operating T_crystal."
        )
    if architecture == SPDC_ARCHITECTURE_TWO and "T_pm0_NL2_C" not in data:
        raise ConfigError(
            "two_crystal requires a separate phase-matching temperature for "
            "NL1 and NL2 (T_pm0_NL1_C and T_pm0_NL2_C)"
        )
    if "T_crystal_NL1_C" not in data and "T_crystal_C" not in data:
        data["T_crystal_NL1_C"] = data["T_pm0_NL1_C"]
        if architecture != SPDC_ARCHITECTURE_TWO:
            data.setdefault("T_crystal_NL2_C", data["T_pm0_NL1_C"])
    if architecture == SPDC_ARCHITECTURE_TWO and "T_crystal_NL2_C" not in data:
        data["T_crystal_NL2_C"] = data["T_pm0_NL2_C"]
    return data


def load_setup_yaml(path: str | Path) -> SetupParams:
    """Load a setup profile; unknown keys raise ``ConfigError`` (V-P0-10)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any]
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImportError(
                "PyYAML is required to load YAML setup profiles "
                "(pip install pyyaml), or provide JSON."
            ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Setup file must contain a mapping: {path}")

    flat = _flatten_setup_yaml_mapping(data, source=path)
    flat = _migrate_setup_yaml_aliases(flat)
    flat = _require_yaml_phase_match_temperatures(flat)
    if "setup_id" not in flat:
        flat["setup_id"] = path.stem

    schema = setup_schema_keys()
    validate_setup_keys(
        flat,
        allow_legacy=False,
        context=f"setup YAML {path}",
    )
    setup = SetupParams.from_dict(flat)
    setup._setup_selection = {
        "yaml_path": str(path.resolve()),
        "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "registered_profile_name": None,
    }
    return setup


def resolve_setup_selection(selection: "SetupParams | str | Path") -> "SetupParams":
    """Resolve a registered name or YAML path before simulation or mkdir."""
    if isinstance(selection, SetupParams):
        return validate_setup_params(selection)
    if not isinstance(selection, (str, Path)):
        raise ConfigError("setup selection must be SetupParams, YAML path, or registered name")
    name = str(selection)
    registered = SETUP_PROFILE_NAME_ALIASES.get(name, name)
    if registered in REGISTERED_SETUP_PROFILES:
        path = Path(__file__).resolve().parent / REGISTERED_SETUP_PROFILES[registered]
        setup = load_setup_yaml(path)
        setup._setup_selection["registered_profile_name"] = registered
        return setup
    path = Path(selection)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError(
            f"Unknown setup profile {name!r}; choose from "
            f"{sorted(REGISTERED_SETUP_PROFILES)} or provide a YAML path"
        )
    return load_setup_yaml(path)


def setup_selection_info(setup: "SetupParams") -> dict[str, Any]:
    """Non-hashed YAML source evidence; resolved values remain in to_dict()."""
    return dict(getattr(setup, "_setup_selection", {
        "yaml_path": None, "yaml_sha256": None,
        "registered_profile_name": None,
    }))


def save_setup_yaml(setup: SetupParams, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = setup.to_dict()
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        import json

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def default_setup_profile_dict() -> dict[str, Any]:
    """Documented default profile (physical beam, YAML-editable)."""
    s = SetupParams()
    return {
        "setup_id": s.setup_id,
        "notes": s.notes,
        "geometry": {
            "magnification": s.magnification,
            "pixel_um": s.pixel_um,
            "pixel_um_min": PIXEL_UM_MIN,
            "pixel_um_max": PIXEL_UM_MAX,
            "pixel_um_default": PIXEL_UM_DEFAULT,
            "lambda_s_nm": s.lambda_s_nm,
            "lambda_i_nm": s.lambda_i_nm,
            "NA": s.NA,
            "aperture_diameter_mm": s.aperture_diameter_mm,
            "lens_focal_mm": s.lens_focal_mm,
            "airy_to_gauss_eta": s.airy_to_gauss_eta,
            "use_derived_diffraction": s.use_derived_diffraction,
            "sigma_diffraction_px": s.sigma_diffraction_px,
            "use_derived_K_defocus": s.use_derived_K_defocus,
            "n_medium": s.n_medium,
            "eta_defocus": s.eta_defocus,
            "defocus_span_px": s.defocus_span_px,
            "K_defocus_px_per_um": s.K_defocus_px_per_um,
            "delta_z_max_um": s.delta_z_max_um,
            "delta_z_um": s.delta_z_um,
            "K_defocus_formula": "eta_def*M*NA/(n*pixel_um)",
            "coc_formula": "(NA/n)*delta_z",
            "sigma_defocus_eq": "K_defocus*|delta_z|",
            "s_focus_eq": "|delta_z|/delta_z_max",
            "sigma_opt_eq": "sqrt(sigma_diffraction^2 + sigma_defocus^2)",
            "c4_c7_chain": "eq160+eq164+eq166",
            "eq162_used": False,
        },
        "beam": {
            "use_legacy_beam_match": s.use_legacy_beam_match,
            "pump_waist_um": s.pump_waist_um,
            "pump_waist_NL1_um": s.pump_waist_NL1_um,
            "pump_waist_NL2_um": s.pump_waist_NL2_um,
            "pump_focus_z_NL1_um": s.pump_focus_z_NL1_um,
            "pump_focus_z_NL2_um": s.pump_focus_z_NL2_um,
            "pointing_alpha": s.pointing_alpha,
            "couple_pointing_overlap_visibility": s.couple_pointing_overlap_visibility,
            "flux0_e_per_s": s.flux0_e_per_s,
            "crystal_length_mm": s.crystal_length_mm,
            "derive_sigma_spdc": s.derive_sigma_spdc,
            "blur_model": normalize_blur_model(getattr(s, "blur_model", None)),
            "object_preprocessing": normalize_object_preprocessing(
                getattr(s, "object_preprocessing", None)
            ),
            "spdc_gain_mode": normalize_spdc_gain_mode(
                getattr(s, "spdc_gain_mode", None)
            ),
            "spdc_gain_mode_nbar": spdc_gain_mode_nbar(
                getattr(s, "spdc_gain_mode", None)
            ),
            "spdc_source_mode": normalize_spdc_source_mode(
                getattr(s, "spdc_source_mode", None)
            ),
            "lambda_p_nm": s.lambda_p_nm,
            "use_analytic_mode_overlap": s.use_analytic_mode_overlap,
            "idler_waist_um": s.idler_waist_um,
            "generation_dx_um": float(getattr(s, "generation_dx_um", 0.0) or 0.0),
            "generation_dy_um": float(getattr(s, "generation_dy_um", 0.0) or 0.0),
            "generation_dtheta_mrad": float(
                getattr(s, "generation_dtheta_mrad", 0.0) or 0.0
            ),
            "relay_dx_um": getattr(s, "relay_dx_um", None),
            "relay_dy_um": getattr(s, "relay_dy_um", None),
            "relay_dtheta_mrad": getattr(s, "relay_dtheta_mrad", None),
            "idler_dx_um": s.idler_dx_um,
            "idler_dy_um": s.idler_dy_um,
            "idler_dtheta_mrad": s.idler_dtheta_mrad,
            "inherit_idler_displacement_from_pump": s.inherit_idler_displacement_from_pump,
        },
        "spdc_gain": {
            "use_physical_spdc_gain": s.use_physical_spdc_gain,
            "spdc_architecture": normalize_spdc_architecture(
                getattr(s, "spdc_architecture", None)
            ),
            "spdc_source_mode": normalize_spdc_source_mode(
                getattr(s, "spdc_source_mode", None)
            ),
            "default_mean_photons_per_mode": s.spdc_default_mean_photons_per_mode,
            "reference_mean_photons_per_mode": s.spdc_reference_mean_photons_per_mode,
            "spdc_gain_mode": normalize_spdc_gain_mode(
                getattr(s, "spdc_gain_mode", None)
            ),
            "spdc_gain_mode_nbar": spdc_gain_mode_nbar(
                getattr(s, "spdc_gain_mode", None)
            ),
            "gain_per_sqrt_mW": s.spdc_gain_per_sqrt_mW,
            "k_spdc_0_per_sqrt_W": k_spdc_0_from_setup(s),
            "k_spdc_per_sqrt_W": resolved_k_spdc_per_sqrt_W(s, interaction=1),
            "k_spdc_0_per_sqrt_W_NL1": k_spdc_0_from_setup(s, interaction=1),
            "k_spdc_0_per_sqrt_W_NL2": k_spdc_0_from_setup(s, interaction=2),
            "k_spdc_per_sqrt_W_NL1": resolved_k_spdc_per_sqrt_W(s, interaction=1),
            "k_spdc_per_sqrt_W_NL2": resolved_k_spdc_per_sqrt_W(s, interaction=2),
            "O_geom_NL1": spdc_geometry_overlap_from_setup(s, interaction=1),
            "O_geom_NL2": spdc_geometry_overlap_from_setup(s, interaction=2),
            "O_geom_includes_qpm_phase": OGEOM_INCLUDES_QPM_PHASE,
            "n_spdc_interactions": n_spdc_interactions(s),
            "n_crystals_hardware": int(s.n_crystals),
            "pump_power_laser_mW": s.pump_power_laser_mW,
            "pump_pre_crystal_transmission": s.pump_pre_crystal_transmission,
            "pump_return_transmission": s.pump_return_transmission,
            "pump_to_NL2_transmission": s.pump_to_NL2_transmission,
            "pump_bs_reflectance": s.pump_bs_reflectance,
            "pump_power_NL1_mW": pump_power_NL1_mW_from_setup(s),
            "pump_power_NL2_mW": pump_power_NL2_mW_from_setup(s),
            "use_independent_pump_power_NL2": s.use_independent_pump_power_NL2,
            "crystal_length_NL1_mm": float(
                spdc_crystal_state(s, interaction=1)["crystal_length_mm"]
            )
            if s.use_physical_spdc_gain
            else float(s.crystal_length_mm),
            "crystal_length_NL2_mm": float(
                spdc_crystal_state(s, interaction=2)["crystal_length_mm"]
            )
            if s.use_physical_spdc_gain
            else (
                float(s.crystal_length_NL2_mm)
                if s.crystal_length_NL2_mm is not None
                else float(s.crystal_length_mm)
            ),
            "delta_k_qpm_per_m_NL1": delta_k_qpm_per_m_from_setup(s, interaction=1),
            "delta_k_qpm_per_m_NL2": delta_k_qpm_per_m_from_setup(s, interaction=2),
            "shared_crystal_length_mm": (
                float(spdc_shared_crystal_state(s)["crystal_length_mm"])
                if s.use_physical_spdc_gain
                else float(s.crystal_length_mm)
            ),
            "zR_pump_NL1_mm": spdc_gaussian_geometry_from_setup(
                s, interaction=1
            )["zR_pump_mm"],
            "xi_pump_NL1": spdc_gaussian_geometry_from_setup(
                s, interaction=1
            )["xi_pump"],
            "low_gain_max_mean_photons": s.spdc_low_gain_max_mean_photons,
            "high_gain_min_mean_photons": s.spdc_high_gain_min_mean_photons,
            "gain_equations": (
                "r0=k*sqrt(P_W); k=kSPDC,0*O_geom (eq.24,36); "
                "nbar from r0,L,Δk_QPM hyperbolic/oscillatory (§7.9 eq.47-51); "
                "O_geom from Gaussian A(z) without e^{i Δk_QPM z} (§7.8); "
                "Physical Mode flux0 is YAML flux0, not YAML*R_SPDC (§7.10); "
                "default: one crystal state, two interactions; n_crystals is hardware (§7.11); "
                "two_crystal: independent L,T,Λ,Δk; Pp2=η_NL2 Pp,laser; "
                "single-crystal: Pp1=ηp,pre Pp,laser, Pp2=ηp,return Pp1 (§7.12); "
                "operating point is k1,k2,r1,r2,nbar1,nbar2; s_gain not in source (§7.13); "
                "dP scales Pp1 then Pp2; Physical Mode flux has no extra (1+dP) (§7.14); "
                "Tseed=ηmode ηi,path To; tseed→N1,N2,C12; pump overlap not on V (§7.15-7.21)"
            ),
        },
        "visibility": {
            "V0": s.V0,
            "overlap_delta_scale": s.overlap_delta_scale,
            "delta_L_mm": s.delta_L_mm,
            "delta_L_mm_min": DELTA_L_MM_MIN,
            "delta_L_mm_max": DELTA_L_MM_MAX,
            "L_coh_mm": s.L_coh_mm,
            "L_coh_mm_min": L_COH_MM_MIN,
            "L_coh_mm_max": L_COH_MM_MAX,
            "path1_intensity": s.path1_intensity,
            "path2_intensity": s.path2_intensity,
            "idler_path_transmission": s.idler_path_transmission,
            "signal_path1_transmission": s.path1_intensity,
            "signal_path2_transmission": s.path2_intensity,
            "use_physical_two_beam": s.use_physical_two_beam,
            "use_field_overlap_visibility_map": s.use_field_overlap_visibility_map,
            "V_gain_drop": s.V_gain_drop,
        },
        "thermal": {
            "n_crystals": s.n_crystals,
            "sync_crystal_temperatures": s.sync_crystal_temperatures,
            "use_physical_thermal": s.use_physical_thermal,
            "use_physical_qiup_magnification": s.use_physical_qiup_magnification,
            "T_crystal_C": s.T_crystal_C,
            "T_crystal_NL1_C": s.T_crystal_NL1_C,
            "T_crystal_NL2_C": s.T_crystal_NL2_C,
            "T_pm0_NL1_C": s.T_pm0_NL1_C,
            "T_pm0_NL2_C": s.T_pm0_NL2_C,
            "delta_T_NL1_C": s.delta_T_NL1_C,
            "delta_T_NL2_C": s.delta_T_NL2_C,
            "dT_NL1_C": s.dT_NL1_C,
            "dT_NL2_C": s.dT_NL2_C,
            "T_pm_C": s.T_pm_C,
            "T_env_C": s.T_env_C,
            "T_min_C": T_CRYSTAL_C_MIN,
            "T_max_C": T_CRYSTAL_C_MAX,
            "T_default_C": T_CRYSTAL_C_DEFAULT,
            "T_step_C": T_CRYSTAL_C_STEP,
            "eq80": "T_pm = T_pm0 + delta_T",
            "crystal_material": s.crystal_material,
            "qpm_type": s.qpm_type,
            "d33_pm_per_V": s.d33_pm_per_V,
            "d_eff_pm_per_V": s.d_eff_pm_per_V,
            **effective_nonlinearity_metadata(s),
            "d_eff_precedence": s.d_eff_precedence,
            "qpm_order": s.qpm_order,
            "qpm_duty_cycle": s.qpm_duty_cycle,
            "poling_period_um": s.poling_period_um,
            "use_calibrated_poling_period": s.use_calibrated_poling_period,
            "pol_pump": s.pol_pump,
            "pol_signal": s.pol_signal,
            "pol_idler": s.pol_idler,
            "alpha_path_per_C": s.alpha_path_per_C,
            "L_eff_path_mm": s.L_eff_path_mm,
            "alpha_L_mm_per_C": s.alpha_L_mm_per_C,
            "F1_mm": s.F1_mm,
            "F2_mm": s.F2_mm,
            "lens_focal_mm": s.lens_focal_mm,
            "lens_material": s.lens_material,
            "beam_thermal_alpha_x_um_per_C": s.beam_thermal_alpha_x_um_per_C,
            "beam_thermal_alpha_y_um_per_C": s.beam_thermal_alpha_y_um_per_C,
            "oven_tau_s": OVEN_TAU_S,
            "theta0_rad": THETA0_RAD,
        },
        "detector": {
            "QE": s.QE,
            "detector_mode": normalize_detector_mode(
                getattr(s, "detector_mode", None)
            ),
            "detector_output_port": normalize_detector_output_port(
                getattr(s, "detector_output_port", None)
            ),
            "emccd_model": s.emccd_model,
            "read_noise0_e": s.read_noise0_e,
            "read_noise_span_e": s.read_noise_span_e,
            "cic0": s.cic0,
            "prnu_sigma": s.prnu_sigma,
            "dark0": s.dark0,
            "dark_exp": s.dark_exp,
            "use_physical_dark": s.use_physical_dark,
            "T_sensor_C": s.T_sensor_C,
            "T_sensor_min_C": s.T_sensor_min_C,
            "T_sensor_max_C": s.T_sensor_max_C,
            "hot_frac_at_s1": s.hot_frac_at_s1,
            "hot_amp_lo": s.hot_amp_lo,
            "hot_amp_hi": s.hot_amp_hi,
            "hot_pixel_model": s.hot_pixel_model,
            "full_well_e": s.full_well_e,
            "g_adu": s.g_adu,
            "bias_adu": s.bias_adu,
            "max_adu": s.max_adu,
            "adc_bit_depth": resolved_adc_bit_depth(s),
            "pc_threshold_k": s.pc_threshold_k,
            "em_gain_default": s.em_gain_default,
        },
        "slm": {
            "slm_bit_depth": s.slm_bit_depth,
            "slm_levels0": s.slm_levels0,
            "slm_pixel_um": s.slm_pixel_um,
            "slm_fill_factor": s.slm_fill_factor,
            "slm_quant_coef": s.slm_quant_coef,
            "slm_gamma_span": s.slm_gamma_span,
            "slm_leak_eta": s.slm_leak_eta,
        },
    }
