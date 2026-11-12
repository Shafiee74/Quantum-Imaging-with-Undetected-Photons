"""
QIUP Digital Twin -- SECTION: latents.py
Operating-point layer: how far each subsystem drifts from ideal, and the
physical->latent bridge. 8 correlated clusters, each driven by one latent.

CONTENTS
  - Governance (lean): Phase-4 rulings, role legend
  - Taxonomy: LATENT_KEYS, CLUSTERS, FORBIDDEN, COUPLINGS, CLUSTER_CONTRACTS
  - Calibration knobs: COUPLING_WEIGHTS (NOT first-principles), PRESETS
  - Physical bridge: PhysicalParams, apply_physical, physical_ref_from_setup
  - State: LatentState(.effective applies couplings), apply_preset
  - Validity: ValidityReport, check_forbidden / check_state, InvalidSampleError,
    SamplingExhaustedError (alias SamplingError), sample_latents
Provenance: qiup_system_architecture.py
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import numpy as np
from setup_params import (
    SetupParams, ResolvedForward, resolve_forward, LEGACY_BEAM_SEMANTICS,
    PUMP_WAIST_UM_MIN, PUMP_WAIST_UM_MAX, POINTING_ALPHA_DEFAULT,
    T_CRYSTAL_C_DEFAULT,
    validate_pump_waist_um, validate_pointing_alpha, sigma_beam_px,
    dark_rate_from_setup, derive_sigma_spdc_px, read_noise_e_from_setup,
    sensor_temperature_C, um_to_px, default_setup_profile_dict,
    load_setup_yaml, save_setup_yaml, eta_mode_from_setup,
    effective_delta_L_mm, crystal_thermal_state, sigma_diffraction_from_setup,
    resolve_c3_fan_delta_phi_rms, sinh_squared_stable,
    normalize_detector_mode, detector_mode_is_photon_counting,
    detector_mode_from_photon_counting_flag, validate_setup_params,
)

# ==============================================================================
# GOVERNANCE (lean) -- digital-twin rulings
# Role: documentation + runtime errors
# Models: Phase-4 defer/reject policy; roles
# Provenance: arch
# ==============================================================================

class Phase4PolicyError(ValueError):
    """Raised when a Phase-4 rejected action is attempted (D2/D4/D5/D6/D7)."""


class Phase4DeferredError(RuntimeError):
    """Raised when a deferred feature is requested too early (D1)."""


PARAMETER_ROLE_LEGEND = {
    "hardware": "device/setup quantity (datasheet or design)",
    "derived": "computed from other parameters",
    "calibration": "tunable / to-be-calibrated phenomenological weight",
    "legacy-latent-scale": "simulation scale not yet mapped to lab units",
}

# ---------------------------------------------------------------------------
# Cluster definitions (from qiup_noise_clusters.html)
# ---------------------------------------------------------------------------


LEMOS_REFERENCE: dict[str, Any] = {
    "role": "experimental_reference_only",
    "not_universal_constants": True,
    "V_ref": 0.77,
    "T_pm_NL1_C": 83.7,
    "T_pm_NL2_C": 84.7,
    "lam_s0_nm": 810.0,
    "lam_i0_nm": 1550.0,
    "note": "Use SetupParams for your lab; do not treat Lemos temps as global laws (D5).",
}


PHASE4_POLICY: dict[str, dict[str, Any]] = {
    "D1": {
        "ruling": "defer",
        "proposal": "Full Zernike wavefront in the first step",
        "reason": "Wait until effective PSF is calibrated; not needed to start.",
        "enforcement": "request_zernike_wavefront / enable_zernike → Phase4DeferredError",
    },
    "D2": {
        "ruling": "reject",
        "proposal": "Claim all couplings are first-principles constants",
        "reason": "Couplings are calibratable, not uniquely derivable from theory.",
        "enforcement": "COUPLING_WEIGHTS_ROLE must remain 'calibration'",
    },
    "D4": {
        "ruling": "forbid",
        "proposal": "Fold SPDC PSF into optical PSF",
        "reason": "Different mechanisms; ruins calibration (also F11).",
        "enforcement": "check_forbidden(fold_spdc_into_optical=True) + optical path assert",
    },
    "D5": {
        "ruling": "reject",
        "proposal": "Treat Lemos temperatures as universal constants",
        "reason": "Reference only; your setup has its own parameters.",
        "enforcement": "LEMOS_REFERENCE.not_universal_constants + assert_phase4_policy",
    },
    "D6": {
        "ruling": "reject",
        "proposal": "Blindly retune coupling coefficients without data",
        "reason": "That is not calibration; it is arbitrary shifting.",
        "enforcement": "set_coupling_weights requires evidence (or explicit allow_blind)",
    },
    "D7": {
        "ruling": "reject_formulation",
        "proposal": "Call N/3.2 and 0.9 a physical waist without mapping",
        "reason": "Either map to (w_p,M,p) or keep as legacy geometry label.",
        "enforcement": "validate_beam_semantics / resolve_forward beam_mode labels",
    },
}


def request_zernike_wavefront(**_kwargs: Any) -> None:
    """D1: Zernike wavefront is deferred — do not enable in Phase 4."""
    raise Phase4DeferredError(
        "D1 deferred: full Zernike wavefront is postponed until the effective "
        "PSF is calibrated. Use delta_phi_rms (Fan map) or s_align for now."
    )


def set_coupling_weights(
    weights: dict[str, float],
    *,
    evidence: str | None = None,
    allow_blind: bool = False,
    profile_id: str = "manual_override",
) -> dict[str, float]:
    """M10/D6: update module COUPLING_WEIGHTS only with evidence (or explicit blind OK).

    Returns the new table. Refuses blind retunes by default (Phase-4 D6).
    """
    if not evidence and not allow_blind:
        raise Phase4PolicyError(
            "D6 rejected: refusing blind coupling retune without evidence. "
            "Pass evidence='...' (lab/proxy note) or allow_blind=True (explicit)."
        )
    if COUPLING_WEIGHTS_ROLE != "calibration":
        raise Phase4PolicyError(
            "D2 rejected: COUPLING_WEIGHTS_ROLE must be 'calibration', "
            f"got {COUPLING_WEIGHTS_ROLE!r}"
        )
    unknown = set(weights) - set(COUPLING_WEIGHTS)
    if unknown:
        raise KeyError(f"Unknown coupling keys: {sorted(unknown)}")
    for k, v in weights.items():
        if not np.isfinite(float(v)):
            raise ValueError(f"Non-finite coupling weight {k}={v}")
        COUPLING_WEIGHTS[k] = float(v)
    # Audit trail on module (lightweight)
    set_coupling_weights.last_evidence = evidence  # type: ignore[attr-defined]
    set_coupling_weights.last_profile_id = profile_id  # type: ignore[attr-defined]
    set_coupling_weights.last_allow_blind = bool(allow_blind)  # type: ignore[attr-defined]
    return dict(COUPLING_WEIGHTS)


def validate_beam_semantics(setup: SetupParams) -> list[str]:
    """D7: legacy N/3.2 envelope must not be labeled as physical waist."""
    bad: list[str] = []
    if setup.use_legacy_beam_match:
        if setup.pump_waist_um is not None and setup.physics_tier == "phase3":
            pass
    else:
        if setup.pump_waist_um is None:
            bad.append(
                "D7: use_legacy_beam_match=False requires pump_waist_um "
                "(cannot claim physical waist without mapping)"
            )
        else:
            try:
                validate_pump_waist_um(setup.pump_waist_um)
            except ValueError as exc:
                bad.append(str(exc))
    return bad


def assert_phase4_policy(
    result: SimResult | None = None,
    setup: SetupParams | None = None,
    *,
    fold_spdc_into_optical: bool = False,
    claim_lemos_universal: bool = False,
    claim_couplings_first_principles: bool = False,
) -> list[str]:
    """Return Phase-4 policy violations (empty ⇒ pass)."""
    bad: list[str] = []

    # D1 — Zernike must stay deferred (no live API path)
    if setup is not None and bool(getattr(setup, "enable_zernike", False)):
        bad.append("D1: enable_zernike=True is deferred; unset the flag")

    # D2
    if COUPLING_WEIGHTS_ROLE != "calibration":
        bad.append(f"D2: COUPLING_WEIGHTS_ROLE={COUPLING_WEIGHTS_ROLE!r} ≠ 'calibration'")
    if claim_couplings_first_principles:
        bad.append("D2: rejected claim that couplings are first-principles constants")

    # D4 / F11
    if fold_spdc_into_optical:
        bad.append("D4/F11: folding SPDC PSF into optical PSF is forbidden")
    if result is not None:
        sig_s = result.meta.get("sig_spdc")
        sig_o = result.meta.get("sig_opt")
        sig_d = result.meta.get("sigma_diffraction_px")
        sig_f = result.meta.get("sigma_defocus_px")
        if None not in (sig_s, sig_o, sig_d, sig_f):
            sig_a = float(result.meta.get("sigma_aberration_px") or 0.0)
            expect = float(
                np.sqrt(float(sig_d) ** 2 + float(sig_f) ** 2 + sig_a ** 2)
            )
            if abs(float(sig_o) - expect) > 1e-5:
                bad.append(f"D4: sig_opt={sig_o} != diff⊕defocus⊕aberr={expect}")
            folded = float(
                np.sqrt(
                    float(sig_s) ** 2 + float(sig_d) ** 2 + float(sig_f) ** 2 + sig_a ** 2
                )
            )
            if abs(float(sig_o) - folded) < 1e-5 and float(sig_s) > 0:
                bad.append("D4: sig_opt appears to include sig_spdc")

    # D5
    if not LEMOS_REFERENCE.get("not_universal_constants", False):
        bad.append("D5: LEMOS_REFERENCE missing not_universal_constants flag")
    if claim_lemos_universal:
        bad.append("D5: rejected claim that Lemos temperatures are universal constants")
    if setup is not None:
        # Setup may equal Lemos numbers, but must not use setup_id implying universal law
        sid = str(setup.setup_id).lower()
        if "universal" in sid or "lemos_law" in sid:
            bad.append(f"D5: setup_id={setup.setup_id!r} implies universal Lemos law")

    # D7
    if setup is not None:
        bad.extend(validate_beam_semantics(setup))
        if result is not None:
            mode = (result.meta.get("resolved_forward") or {}).get("derived", {}).get(
                "beam_mode"
            )
            if setup.use_legacy_beam_match and mode not in (None, "legacy_match"):
                bad.append(f"D7: legacy setup but beam_mode={mode!r}")
            if (
                not setup.use_legacy_beam_match
                and setup.pump_waist_um is not None
                and mode not in (None, "physical_waist")
            ):
                bad.append(f"D7: physical waist expected, got beam_mode={mode!r}")
    if not LEGACY_BEAM_SEMANTICS.get("role") == "legacy_geometry":
        bad.append("D7: LEGACY_BEAM_SEMANTICS.role must be 'legacy_geometry'")
    if LEGACY_BEAM_SEMANTICS.get("is_physical_waist", True):
        bad.append("D7: LEGACY_BEAM_SEMANTICS must not claim is_physical_waist=True")

    return bad

# S2 — explicit separation: SPDC gain regime latent ≠ EMCCD register gain
#   s_gain  → cluster C2 (pair-generation / multi-pair / visibility drop)


# ==============================================================================
# TAXONOMY -- latents, clusters, forbidden combos, contracts
# Role: structure
# Models: 8 clusters; which knobs each latent controls
# Provenance: arch
# ==============================================================================

LATENT_KEYS = (
    "s_pump", "s_gain", "s_align", "s_thermal", "s_det", "s_cool", "s_focus", "s_slm"
)


GAIN_SEMANTICS = {
    "s_gain": "Legacy phenomenological SPDC severity; unused in Physical Mode source (§7.13)",
    "mean_photons_per_mode": "Alias of nbar1; Physical Mode also has nbar2 (§7.13)",
    "spdc_gain_r": "Alias of r1; Physical Mode also has r2 (§7.13)",
    "pump_power_mW": "P_p,laser [mW]; P_p1 and P_p2 are derived (§7.12)",
    "em_gain": "EMCCD electron-multiplication gain G_EM (detector knob)",
}

# S6 — contract: cooling vs detector operating point must stay separate


CLUSTER_CONTRACTS = {
    "s_cool": {
        "controls": ("dark_current", "hot_pixels"),
        "must_not_control": ("read_noise", "cic", "prnu", "em_gain"),
    },
    "s_det": {
        "controls": ("read_noise", "cic", "prnu"),
        "must_not_control": ("dark_current", "hot_pixels"),
    },
}

# S7 — visibility calibration scale (targets ~0.77 Lemos when well aligned)


CLUSTERS: dict[str, dict[str, Any]] = {
    "C1": {
        "name": "Pump stability",
        "latent": "s_pump",
        "members": [
            "pump_power_fluctuation",
            "beam_pointing_jitter",
            "pump_overlap_visibility",
            "spdc_flux_fluctuation",
        ],
        "always_on": ["gaussian_envelope"],
        "drives": ["C2", "C5"],
    },
    "C2": {
        "name": "SPDC gain regime",
        "latent": "s_gain",
        "members": [
            "spdc_gain_regime",
            "multi_pair_generation",
            "accidental_correlations",
        ],
        "always_on": ["spdc_psf"],
        "drives": ["C3"],
    },
    "C3": {
        "name": "Mode overlap / alignment",
        "latent": "s_align",
        "members": [
            "spatial_misalignment",
            "path_length_mismatch",
            "wavefront_distortion",  # Fan Δφ on interference path (not Zernike PSF)
            "fan_path_phase_map",
            "spatial_mismatch",
            "mean_visibility_loss",
            "spatial_visibility_map",
            "coherence_envelope",
            "unequal_path_intensity",
            "interferometer_phase_offset",
        ],
        "always_on": [],
        "drives": ["C5"],
    },
    "C4": {
        "name": "Thermal drift",
        "latent": "s_thermal",
        "members": [
            "crystal_temperature_drift",
            "phase_matching_instability",
            "alignment_drift",
            "phase_drift",
            "beam_center_drift",
            "background_drift",
            "visibility_fluctuation",
            "thermal_drift",
        ],
        "always_on": [],
        "drives": ["C3", "C7"],
    },
    "C5": {
        "name": "Detector operating point",
        "latent": "s_det",
        "members": [
            "shot_noise",
            "em_gain_excess_noise",
            "read_noise",
            "cic",
            "clipping_saturation",
            "digitization",
        ],
        "always_on": [],
        "drives": [],
    },
    "C6": {
        "name": "Sensor cooling",
        "latent": "s_cool",
        "members": ["dark_current", "hot_pixels", "detector_background"],
        "always_on": [],
        "drives": ["C5"],
    },
    "C7": {
        "name": "Focus / imaging quality",
        "latent": "s_focus",
        "members": [
            "defocus",
            "optical_blur",
            "aberration",  # σ_aberr from wavefront_rms_waves (no Zernike)
            "rotation_affine",
            "magnification_error",
            "lens_misalignment_rings",  # L06: idler-lens φ_I on seed before C12
        ],
        "always_on": ["diffraction_limited_psf"],
        "drives": [],
    },
    "C8": {
        "name": "SLM calibration",
        "latent": "s_slm",
        "members": [
            "gray_level_nonlinearity",
            "phase_amplitude_coupling",
            "slm_amplitude_leakage",
            "object_misregistration",
        ],
        "always_on": ["slm_pixelation", "finite_fill_factor", "phase_quantization"],
        "drives": [],
    },
}

# Forbidden combinations (reject / repair — never sample)


FORBIDDEN: list[dict[str, str]] = [
    {"id": "F1", "cluster": "C1", "rule": "stable_pump_with_flux_jitter",
     "desc": "Stable pump with fluctuating SPDC flux (same quantity)"},
    {"id": "F2", "cluster": "C2", "rule": "low_gain_high_multipair",
     "desc": "Low gain with high multi-pair rate"},
    {"id": "F3", "cluster": "C3", "rule": "high_V_severe_wavefront",
     "desc": "High visibility with severe wavefront distortion"},
    {"id": "F4", "cluster": "C3", "rule": "high_V_large_deltaL",
     "desc": "High visibility with delta_L >> L_coh"},
    {"id": "F5", "cluster": "C4", "rule": "thermal_in_short_acq",
     "desc": "Thermal drift inside a short acquisition"},
    {"id": "F6", "cluster": "C5", "rule": "pc_with_excess_noise",
     "desc": "Photon counting with excess noise factor active"},
    {"id": "F7", "cluster": "C5", "rule": "high_G_read_dominated",
     "desc": "High EM gain with read-noise-dominated statistics"},
    {"id": "F8", "cluster": "C5", "rule": "high_G_flux_no_clip",
     "desc": "High gain+flux with clipping disabled"},
    {"id": "F9", "cluster": "C6", "rule": "high_dark_deep_cool",
     "desc": "High dark current with deeply cooled sensor"},
    {"id": "F10", "cluster": "C7", "rule": "defocus_block5_vs_7",
     "desc": "Defocus at block 5 != block 7 (one knob)"},
    {"id": "F11", "cluster": "C7", "rule": "spdc_folded_into_optical",
     "desc": "SPDC sigma folded into optical PSF sum"},
    {"id": "F12", "cluster": "C8", "rule": "good_cal_large_leakage",
     "desc": "Good calibration with large amplitude leakage"},
]

# Cross-cluster edges that must be enforced when sampling latents


COUPLINGS: list[tuple[str, str, str]] = [
    ("s_pump", "s_gain", "Pump intensity sets gain regime"),
    ("s_pump", "s_det", "Incident flux sets detector noise dominance"),
    ("s_gain", "s_align", "High gain lowers visibility via induced emission"),
    ("s_thermal", "s_align", "Thermal drift degrades overlap (physical Δx/η when enabled)"),
    # C4→C7 is physical eq.160+164+166 in forward — NOT latent β_T (eq.162 unused)
    ("s_thermal", "s_focus", "PDF §6.9: ΔT→Δf→Δz→σ_defocus (physical chain; no eq.162)"),
    ("s_align", "s_det", "Low V → small modulation vs shot noise"),
]


# ==============================================================================
# CALIBRATION KNOBS (phenomenological, NOT first-principles)
# Role: calibration
# Models: cross-cluster coupling weights; regime bounds; presets
# Provenance: arch
# ==============================================================================

COUPLING_WEIGHTS: dict[str, float] = {
    "pump_to_gain": 0.35,       # s_gain  += w * s_pump
    # PDF §5 / §6.9: thermal→align is physical (Δx, η); keep 0 here.
    "thermal_to_align": 0.0,
    "gain_to_align": 0.30,      # s_align += w * s_gain
    # Eq.162 (s_focus = s0 + β_T s_thermal) intentionally omitted — C4→C7 via Δf→Δz.
    "pump_to_det": 0.25,        # s_det   += w * s_pump
    "align_to_det": 0.20,       # s_det   += w * s_align_eff
}
COUPLING_WEIGHTS_ROLE = "calibration"  # D2: never "first_principles"


REGIME_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "lab": {
        "s_pump": (0.0, 0.35),
        "s_gain": (0.0, 0.30),
        "s_align": (0.0, 0.40),
        "s_thermal": (0.0, 0.35),
        "s_det": (0.0, 0.40),
        "s_cool": (0.0, 0.35),
        "s_focus": (0.0, 0.40),
        "s_slm": (0.0, 0.40),
    },
    "stress": {
        "s_pump": (0.0, 1.0),
        "s_gain": (0.0, 1.0),
        "s_align": (0.0, 1.0),
        "s_thermal": (0.0, 1.0),
        "s_det": (0.0, 1.0),
        "s_cool": (0.0, 1.0),
        "s_focus": (0.0, 1.0),
        "s_slm": (0.0, 1.0),
    },
}

# ---------------------------------------------------------------------------
# Phase-1 named defaults (numerical values frozen at Phase-0 baseline)
# Roles: hardware | derived | calibration | legacy-latent-scale
# ---------------------------------------------------------------------------

# S1 — phenomenological cross-cluster coupling weights (calibration knobs)
# Phase-4 D2: these are NOT first-principles constants of QIUP/SPDC.


FORWARD_LEGACY_DEFAULTS: dict[str, float] = {
    "phase_drift_per_frame": 0.08,
    "phase_jitter_align": 0.03,
    "vis_walk_per_frame": 0.02,
    "beam_drift_x_px_per_frame": 0.15,
    "beam_drift_y_px_per_frame": 0.10,
    "mottle_smooth_sigma_px": 8.0,
}


PRESETS: dict[str, dict[str, float]] = {
    "well_aligned": {
        "s_pump": 0.10, "s_gain": 0.05, "s_align": 0.08, "s_thermal": 0.05,
        "s_cool": 0.08, "s_focus": 0.10, "s_slm": 0.10, "s_det": 0.15,
    },
    "poor_alignment": {
        "s_pump": 0.12, "s_gain": 0.06, "s_align": 0.72, "s_thermal": 0.10,
        "s_cool": 0.10, "s_focus": 0.15, "s_slm": 0.12, "s_det": 0.20,
    },
    "thermal_drift": {
        "s_pump": 0.15, "s_gain": 0.06, "s_align": 0.10, "s_thermal": 0.85,
        "s_cool": 0.20, "s_focus": 0.18, "s_slm": 0.12, "s_det": 0.25,
    },
    "low_flux_pc": {
        "s_pump": 0.12, "s_gain": 0.05, "s_align": 0.12, "s_thermal": 0.08,
        "s_cool": 0.10, "s_focus": 0.12, "s_slm": 0.10, "s_det": 0.10,
    },
    "defocus_miscal": {
        "s_pump": 0.15, "s_gain": 0.08, "s_align": 0.15, "s_thermal": 0.12,
        "s_cool": 0.15, "s_focus": 0.80, "s_slm": 0.75, "s_det": 0.20,
    },
}


# ---------------------------------------------------------------------------
# Physical parameter layer → cluster latents
# ---------------------------------------------------------------------------
# Units (canonical for this model):
#   °C, nm, mm (path), rad, px, s, e- (electrons), dimensionless fractions.
# Each quantity is inverted from the existing forward map at s=1 (PHYSICAL_REF).
# Within a cluster, contributions combine by max (worst physical driver sets s).
# Couplings and forward physics after LatentState.effective() are unchanged.


PHYSICAL_REF: dict[str, dict[str, float]] = {
    # scale such that physical_value / scale → contribution to s at the code's s=1 end
    "C1": {
        "pump_sigma_rel": 0.30,       # relative flux rms at s_pump=1
        "pointing_alpha": POINTING_ALPHA_DEFAULT,
    },
    "C2": {
        "flux_boost": 1.4,            # Φ ← Φ·(1 + 1.4 s)
        "V_gain_drop": 0.40,          # V_gain = 1 - 0.40 s
        "accidental_e": 0.02,         # λ_acc = 0.02 s²
    },
    "C3": {
        "overlap_delta": 1.15,        # legacy δ = 1.15 s_align only
        "idler_dx_um": 50.0,          # |Δx| scale at s_align=1 if mapped (audit)
        "delta_L_mm": 0.01,           # PDF §4.1 lab ΔL [mm] (not via s_align)
        "L_coh_mm": 0.20,             # PDF §4.1 L_c [mm]; default fixed
        "phase_jitter_rad": 0.03,
        "V_map_mottle": 0.25,
    },
    "C4": {
        "T_ref_C": 8.0,               # |ΔT_eff| = 8 °C ↔ s_thermal=1 (HTML demo)
        "dlam_s_nm_per_C": 0.0,       # filled from Sellmeier when physical
        "dlam_i_nm_per_C": 0.0,
        "dM_over_M_per_C": 0.0,
        "phase_drift_rad_per_frame": 0.0,
        "vis_walk_per_frame": 0.0,
        "beam_drift_px_per_frame": 0.0,
        "T_pm_NL1_C": T_CRYSTAL_C_DEFAULT,
        "T_pm_NL2_C": T_CRYSTAL_C_DEFAULT,
        "lam_s0_nm": 810.0,
        "lam_i0_nm": 1550.0,
    },
    "C5": {
        "read_noise0_e": 8.0,         # σ = 8 + 40 s
        "read_noise_span_e": 40.0,
        "cic0": 0.006,                # cic = 0.006·(0.5 + s)
        "prnu_sigma": 0.03,
    },
    "C6": {
        "dark0": 0.0025,              # dark = 0.0025·exp(7.2 s)  [e-/px/s]
        "dark_exp": 7.2,
        "hot_frac": 0.002,            # fraction at s=1
    },
    "C7": {
        "diffraction_px": 0.65,
        "defocus_span_px": 3.6,       # σ_opt² = 0.65² + (3.6 s)²
    },
    "C8": {
        "gamma_span": 0.9,            # γ = 1 + 0.9 s
        "levels0": 256.0,
        "quant_coef": 0.985,          # levels = 256·(1 - 0.985 s)²
        "leak_eta": 0.40,
    },
}


# ==============================================================================
# PHYSICAL BRIDGE -- real units (C,nm,mm,rad) -> latents
# Role: physics->latent map
# Models: PhysicalParams contributions; apply_physical
# Provenance: arch
# ==============================================================================

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _max_contrib(*vals: float | None) -> float:
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return _clip01(max(xs)) if xs else 0.0


@dataclass
class PhysicalParams:
    """
    Physical / engineering knobs in real units.

    Only fields that are set (not None) contribute to that cluster's latent.
    Within each cluster, contributions combine by max → s_cluster ∈ [0, 1].
    Detector knobs (em_gain, exposure_s, …) pass through unchanged (not latents).

    Use ``PhysicalParams.lab_defaults()`` for a mild full-lab starting point, or
    set only the fields you want to drive (others keep the base preset).
    """

    # --- C1 Pump [relative, α] ---
    pump_sigma_rel: float | None = None          # relative power/flux rms
    pointing_alpha: float | None = None          # σ_beam = α·w_px

    # --- C2 Gain [dimensionless, e-/px] ---
    flux_boost_factor: float | None = None       # Φ multiplier = 1 + 1.4 s
    V_gain: float | None = None                  # visibility factor from gain
    accidental_e_per_px: float | None = None     # accidental rate on λ

    # --- C3 Align [µm, mm, rad] ---
    overlap_delta: float | None = None           # legacy δ (unused if analytic η_mode)
    idler_waist_um: float | None = None          # idler Gaussian waist in crystal
    idler_dx_um: float | None = None             # idler-2 vs idler-1 Δx [µm]
    idler_dy_um: float | None = None
    idler_dtheta_mrad: float | None = None
    delta_L_mm: float | None = None              # path mismatch [mm]
    L_coh_mm: float | None = None                # coherence length [mm]
    path1_intensity: float | None = None         # PDF §4.2 I1 (relative)
    path2_intensity: float | None = None         # PDF §4.2 I2 (relative)
    phase_jitter_rad: float | None = None        # interferometer phase jitter [rad]
    visibility_mottle: float | None = None       # spatial V map amplitude

    # --- C4 Thermal [°C, rad/frame, 1/frame, px/frame] ---
    T_set_C: float | None = None                 # oven setpoint [°C]
    T_pm_C: float | None = None                  # phase-match temp (NL1 default) [°C]
    delta_T_C: float | None = None               # oven stability / drift amp [°C]
    phase_drift_rad_per_frame: float | None = None
    vis_walk_per_frame: float | None = None
    beam_drift_px_per_frame: float | None = None

    # --- C5 Detector latent drivers [e-, 1] ---
    read_noise_e: float | None = None            # read noise rms [e-]
    cic_e_per_px: float | None = None            # CIC [e-/px/frame]
    prnu_sigma: float | None = None              # PRNU relative σ

    # --- C6 Cooling [e-/px/s, fraction] ---
    dark_rate_e_per_px_s: float | None = None    # dark current rate
    hot_pixel_fraction: float | None = None      # hot pixel fraction

    # --- C7 Focus [px] ---
    sigma_opt_px: float | None = None            # total optical blur σ [px]

    # --- C8 SLM [1, levels, 1] ---
    slm_gamma: float | None = None               # gray→phase gamma
    phase_levels: float | None = None            # effective phase quantization levels
    slm_leak_eta: float | None = None            # amplitude leakage η

    # --- Auxiliary knobs (not cluster latents; forwarded to LatentState) ---
    em_gain: float | None = None
    exposure_s: float | None = None
    photon_counting: bool | None = None
    num_frames: int | None = None
    # V-P0-03: SPDC operating point. None = keep previous LatentState value.
    mean_photons_per_mode: float | None = None
    spdc_gain_r: float | None = None
    mean_photons_per_mode_NL1: float | None = None
    mean_photons_per_mode_NL2: float | None = None
    spdc_gain_r_NL1: float | None = None
    spdc_gain_r_NL2: float | None = None
    pump_power_mW: float | None = None

    @classmethod
    def lab_defaults(cls) -> "PhysicalParams":
        """Mild lab setpoints (real units) that map to small latents."""
        return cls(
            pump_sigma_rel=0.03,
            pointing_alpha=0.01,
            flux_boost_factor=1.07,
            V_gain=0.97,
            accidental_e_per_px=5e-5,
            overlap_delta=None,
            idler_dx_um=0.0,
            idler_dy_um=0.0,
            delta_L_mm=0.05,
            L_coh_mm=0.20,
            path1_intensity=1.0,
            path2_intensity=0.95,
            phase_jitter_rad=0.005,
            visibility_mottle=0.03,
            T_set_C=83.4,
            T_pm_C=83.7,
            delta_T_C=0.1,
            read_noise_e=12.0,
            cic_e_per_px=0.004,
            prnu_sigma=0.005,
            dark_rate_e_per_px_s=0.004,
            hot_pixel_fraction=2e-4,
            sigma_opt_px=0.9,
            slm_gamma=1.1,
            phase_levels=200.0,
            slm_leak_eta=0.04,
            em_gain=800.0,
            exposure_s=0.02,
            photon_counting=True,
            num_frames=128,
        )

    def contributions(
        self, *, setup: SetupParams | None = None
    ) -> dict[str, dict[str, float]]:
        """Per-cluster physical → s contributions (before max / clip).

        If ``setup`` is given, inversion scales come from that profile (L11);
        otherwise module ``PHYSICAL_REF`` (legacy-matched) is used.
        """
        ref = physical_ref_from_setup(setup) if setup is not None else PHYSICAL_REF
        r1, r2, r3 = ref["C1"], ref["C2"], ref["C3"]
        r4, r5, r6 = ref["C4"], ref["C5"], ref["C6"]
        r7, r8 = ref["C7"], ref["C8"]
        out: dict[str, dict[str, float]] = {
            "s_pump": {}, "s_gain": {}, "s_align": {}, "s_thermal": {},
            "s_det": {}, "s_cool": {}, "s_focus": {}, "s_slm": {},
        }

        # C1
        if self.pump_sigma_rel is not None:
            out["s_pump"]["pump_sigma_rel"] = abs(self.pump_sigma_rel) / r1["pump_sigma_rel"]
        if self.pointing_alpha is not None:
            out["s_pump"]["pointing_alpha"] = (
                abs(self.pointing_alpha) / r1["pointing_alpha"]
            )

        # C2
        if self.flux_boost_factor is not None:
            out["s_gain"]["flux_boost"] = max(0.0, self.flux_boost_factor - 1.0) / r2["flux_boost"]
        if self.V_gain is not None:
            out["s_gain"]["V_gain"] = max(0.0, 1.0 - float(self.V_gain)) / r2["V_gain_drop"]
        if self.accidental_e_per_px is not None:
            out["s_gain"]["accidental"] = float(np.sqrt(
                max(0.0, self.accidental_e_per_px) / r2["accidental_e"]
            ))

        # C3 — ΔL / L_c are lab setup fields (eq. 65), not mapped to s_align
        if self.overlap_delta is not None:
            out["s_align"]["overlap_delta"] = abs(self.overlap_delta) / r3["overlap_delta"]
        if self.phase_jitter_rad is not None:
            out["s_align"]["phase_jitter_rad"] = abs(self.phase_jitter_rad) / r3["phase_jitter_rad"]
        if self.visibility_mottle is not None:
            # Legacy eq. 73 only; field-overlap map (eq. 74) ignores this path.
            if setup is None or not bool(
                getattr(setup, "use_field_overlap_visibility_map", False)
            ):
                out["s_align"]["visibility_mottle"] = (
                    abs(self.visibility_mottle) / r3["V_map_mottle"]
                )

        # C4 — temperature in °C → equivalent |ΔT| from phase-match + stability
        # dT_eff [°C] = |T_set - T_pm| + |delta_T|; s = dT_eff / T_ref (T_ref=8°C ↔ s=1)
        if self.T_set_C is not None or self.delta_T_C is not None:
            T_pm = float(self.T_pm_C) if self.T_pm_C is not None else r4["T_pm_NL1_C"]
            T_set = float(self.T_set_C) if self.T_set_C is not None else T_pm
            dT_eff = abs(T_set - T_pm) + (
                abs(float(self.delta_T_C)) if self.delta_T_C is not None else 0.0
            )
            out["s_thermal"]["delta_T_eff_C"] = dT_eff / r4["T_ref_C"]
        if self.phase_drift_rad_per_frame is not None:
            out["s_thermal"]["phase_drift"] = (
                abs(self.phase_drift_rad_per_frame) / r4["phase_drift_rad_per_frame"]
            )
        if self.vis_walk_per_frame is not None:
            out["s_thermal"]["vis_walk"] = abs(self.vis_walk_per_frame) / r4["vis_walk_per_frame"]
        if self.beam_drift_px_per_frame is not None:
            out["s_thermal"]["beam_drift"] = (
                abs(self.beam_drift_px_per_frame) / r4["beam_drift_px_per_frame"]
            )

        # C5
        if self.read_noise_e is not None:
            out["s_det"]["read_noise_e"] = max(
                0.0, (float(self.read_noise_e) - r5["read_noise0_e"]) / r5["read_noise_span_e"]
            )
        if self.cic_e_per_px is not None:
            # cic = 0.006*(0.5 + s) → s = cic/0.006 - 0.5
            out["s_det"]["cic"] = max(0.0, float(self.cic_e_per_px) / r5["cic0"] - 0.5)
        if self.prnu_sigma is not None:
            out["s_det"]["prnu"] = abs(self.prnu_sigma) / r5["prnu_sigma"]

        # C6
        if self.dark_rate_e_per_px_s is not None:
            ratio = max(float(self.dark_rate_e_per_px_s), 1e-12) / r6["dark0"]
            out["s_cool"]["dark_rate"] = float(np.log(ratio) / r6["dark_exp"])
        if self.hot_pixel_fraction is not None:
            out["s_cool"]["hot_frac"] = abs(self.hot_pixel_fraction) / r6["hot_frac"]

        # C7
        if self.sigma_opt_px is not None:
            sig = float(self.sigma_opt_px)
            floor = r7["diffraction_px"]
            out["s_focus"]["sigma_opt"] = float(
                np.sqrt(max(0.0, sig * sig - floor * floor)) / r7["defocus_span_px"]
            )

        # C8
        if self.slm_gamma is not None:
            out["s_slm"]["gamma"] = max(0.0, float(self.slm_gamma) - 1.0) / r8["gamma_span"]
        if self.phase_levels is not None:
            lev = max(float(self.phase_levels), 3.0)
            # levels = 256*(1 - 0.985 s)^2 → s = (1 - sqrt(levels/256)) / 0.985
            out["s_slm"]["phase_levels"] = max(
                0.0, (1.0 - float(np.sqrt(lev / r8["levels0"]))) / r8["quant_coef"]
            )
        if self.slm_leak_eta is not None:
            out["s_slm"]["leak_eta"] = abs(self.slm_leak_eta) / r8["leak_eta"]

        return out

    def to_latents(self, *, setup: SetupParams | None = None) -> dict[str, float]:
        """Map physical params → raw cluster latents in [0, 1] (before coupling)."""
        contrib = self.contributions(setup=setup)
        return {k: _max_contrib(*v.values()) if v else 0.0 for k, v in contrib.items()}

    def thermal_optics(self, *, setup: SetupParams | None = None) -> dict[str, float]:
        """Derived λ / M / path shifts from temperature (PDF §5 physical)."""
        from setup_params import physical_thermal_bundle

        s = setup if setup is not None else self.get_setup()
        if s is not None and bool(getattr(s, "use_physical_thermal", False)):
            b = physical_thermal_bundle(s)
            return {
                "T_set_C": float(s.T_crystal_NL1_C),
                "T_pm_C": float(s.T_pm_C),
                "delta_T_C": float(s.delta_T_NL1_C),
                "dT_bias_C": float(b["dT_eff_C"]),
                "dT_eff_C": float(b["dT_eff_C"]),
                "lam_s_nm": float(b["lambda_s_nm_T"]),
                "lam_i_nm": float(b["lambda_i_nm_T"]),
                "dlam_s_nm_per_C": float(b["dlam_s_nm_per_C"]),
                "dlam_i_nm_per_C": float(b["dlam_i_nm_per_C"]),
                "dM_over_M": float(b["dM_over_M"]),
                "delta_L_thermal_mm": float(b["delta_L_thermal_mm"]),
                "optical_path_difference_mm": float(
                    b.get("optical_path_difference_mm", b["delta_L_thermal_mm"])
                ),
                "opd_phase_rad": float(b.get("opd_phase_rad", 0.0)),
                "V_coh": float(b["V_coh"]),
                "eta_mode_thermal": float(b["eta_mode_thermal"]),
                "R_SPDC_sinc2": float(b["R_SPDC_sinc2"]),
                "units": "C,nm,1",
            }
        ref = physical_ref_from_setup(setup) if setup is not None else PHYSICAL_REF
        r4 = ref["C4"]
        T_set = float(self.T_set_C) if self.T_set_C is not None else r4["T_pm_NL1_C"]
        T_pm = float(self.T_pm_C) if self.T_pm_C is not None else r4["T_pm_NL1_C"]
        dT_amp = float(self.delta_T_C) if self.delta_T_C is not None else 0.0
        dT = (T_set - T_pm)
        dT_eff = abs(dT) + abs(dT_amp)
        return {
            "T_set_C": T_set,
            "T_pm_C": T_pm,
            "delta_T_C": dT_amp,
            "dT_bias_C": dT,
            "dT_eff_C": dT_eff,
            "lam_s_nm": r4["lam_s0_nm"] + r4["dlam_s_nm_per_C"] * dT,
            "lam_i_nm": r4["lam_i0_nm"] + r4["dlam_i_nm_per_C"] * dT,
            "dM_over_M": r4["dM_over_M_per_C"] * dT,
            "units": "C,nm,1",
        }

    def report(self, *, setup: SetupParams | None = None) -> dict[str, Any]:
        contrib = self.contributions(setup=setup)
        latents = self.to_latents(setup=setup)
        scales = physical_ref_from_setup(setup) if setup is not None else PHYSICAL_REF
        return {
            "contributions": contrib,
            "latents_raw": latents,
            "thermal_optics": self.thermal_optics(setup=setup),
            "physical_ref_scales": scales,
            "setup_id": setup.setup_id if setup is not None else "PHYSICAL_REF_module",
        }


def _pointing_alpha_from_setup(setup: SetupParams) -> float:
    return float(validate_pointing_alpha(setup.pointing_alpha))


def physical_ref_from_setup(setup: SetupParams) -> dict[str, dict[str, float]]:
    """Build PHYSICAL_REF-compatible scales from a SetupParams profile (L11)."""
    therm = crystal_thermal_state(setup)
    t_pm2 = therm["T_pm_NL2_C"] if therm["T_pm_NL2_C"] is not None else therm["T_pm_NL1_C"]
    return {
        "C1": {
            "pump_sigma_rel": float(setup.pump_sigma_rel_at_s1),
            "pointing_alpha": _pointing_alpha_from_setup(setup),
        },
        "C2": {
            "flux_boost": float(setup.flux_boost_at_s1),
            "V_gain_drop": float(setup.V_gain_drop),
            "accidental_e": float(setup.accidental_e_at_s1),
        },
        "C3": {
            "overlap_delta": float(setup.overlap_delta_scale),
            "idler_dx_um": 50.0,
            "delta_L_mm": float(setup.delta_L_mm),
            "L_coh_mm": float(setup.L_coh_mm),
            "phase_jitter_rad": float(setup.phase_jitter_align_at_s1),
            "V_map_mottle": float(setup.V_map_mottle),
        },
        "C4": {
            "T_ref_C": float(setup.T_ref_span_C),
            "dlam_s_nm_per_C": float(setup.dlam_s_nm_per_C),
            "dlam_i_nm_per_C": float(setup.dlam_i_nm_per_C),
            "dM_over_M_per_C": float(setup.dM_over_M_per_C),
            "phase_drift_rad_per_frame": float(setup.phase_drift_rad_per_frame_at_s1),
            "vis_walk_per_frame": float(setup.vis_walk_per_frame_at_s1),
            "beam_drift_px_per_frame": float(setup.beam_drift_x_px_per_frame_at_s1),
            "T_pm_NL1_C": float(therm["T_pm_NL1_C"]),
            "T_pm_NL2_C": float(t_pm2),
            "lam_s0_nm": float(setup.lambda_s_nm),
            "lam_i0_nm": float(setup.lambda_i_nm),
        },
        "C5": {
            "read_noise0_e": float(setup.read_noise0_e),
            "read_noise_span_e": float(setup.read_noise_span_e),
            "cic0": float(setup.cic0),
            "prnu_sigma": float(setup.prnu_sigma),
        },
        "C6": {
            "dark0": float(setup.dark0),
            "dark_exp": float(setup.dark_exp),
            "hot_frac": float(setup.hot_frac_at_s1),
        },
        "C7": {
            "diffraction_px": float(
                sigma_diffraction_from_setup(setup)
                if bool(setup.use_derived_diffraction)
                else setup.sigma_diffraction_px
            ),
            "defocus_span_px": float(setup.defocus_span_px),
        },
        "C8": {
            "gamma_span": float(setup.slm_gamma_span),
            "levels0": float(setup.slm_levels0),
            "quant_coef": float(setup.slm_quant_coef),
            "leak_eta": float(setup.slm_leak_eta),
        },
    }


def apply_physical(
    phys: PhysicalParams,
    *,
    base: LatentState | str | None = "well_aligned",
    override_unset_latents: bool = False,
    setup: SetupParams | str | Path | None = None,
) -> LatentState:
    """
    Build LatentState from physical params.

    - Clusters with at least one set physical field get s from physics (max of contribs).
    - Other clusters keep values from ``base`` (preset name or LatentState), unless
      ``override_unset_latents`` forces all eight from physics (missing → 0).
    - Couplings are NOT applied here; use ``state.effective()`` as before.
    - ``setup`` (L11/L12) attaches a physical profile; scales for →s mapping follow it.
    - V-P0-03: SPDC gain fields (``mean_photons_per_mode``, ``spdc_gain_r``,
      ``pump_power_mW``) are copied from ``base`` and only overwritten when the
      corresponding ``PhysicalParams`` field is not None (None keeps previous).
    """
    resolved_setup: SetupParams | None = None
    if setup is not None:
        if isinstance(setup, (str, Path)):
            resolved_setup = load_setup_yaml(setup)
        else:
            resolved_setup = setup

    if base is None:
        st = LatentState()
    elif isinstance(base, str):
        st = apply_preset(base, setup=resolved_setup)
    else:
        # Full identity copy of LatentState (including SPDC gain fields).
        st = replace(
            base,
            setup=resolved_setup if resolved_setup is not None else base.setup,
        )

    if resolved_setup is not None:
        st.setup = resolved_setup

    setup_ov: dict[str, Any] = {}
    if phys.idler_waist_um is not None:
        setup_ov["idler_waist_um"] = float(phys.idler_waist_um)
    if phys.idler_dx_um is not None:
        setup_ov["idler_dx_um"] = float(phys.idler_dx_um)
    if phys.idler_dy_um is not None:
        setup_ov["idler_dy_um"] = float(phys.idler_dy_um)
    if phys.idler_dtheta_mrad is not None:
        setup_ov["idler_dtheta_mrad"] = float(phys.idler_dtheta_mrad)
    if phys.path1_intensity is not None:
        setup_ov["path1_intensity"] = float(phys.path1_intensity)
    if phys.path2_intensity is not None:
        setup_ov["path2_intensity"] = float(phys.path2_intensity)
    if phys.delta_L_mm is not None:
        setup_ov["delta_L_mm"] = float(phys.delta_L_mm)
    if phys.L_coh_mm is not None:
        setup_ov["L_coh_mm"] = float(phys.L_coh_mm)
    if phys.T_set_C is not None:
        setup_ov["T_crystal_C"] = float(phys.T_set_C)
        setup_ov["T_crystal_NL1_C"] = float(phys.T_set_C)
        setup_ov["T_crystal_NL2_C"] = float(phys.T_set_C)
    if phys.T_pm_C is not None:
        setup_ov["T_pm0_NL1_C"] = float(phys.T_pm_C)
        setup_ov["T_pm0_NL2_C"] = float(phys.T_pm_C)
        setup_ov["delta_T_NL1_C"] = 0.0
        setup_ov["delta_T_NL2_C"] = 0.0
    if phys.delta_T_C is not None:
        setup_ov["delta_T_NL1_C"] = float(phys.delta_T_C)
        setup_ov["delta_T_NL2_C"] = float(phys.delta_T_C)
    if setup_ov:
        st.setup = st.get_setup().overridden(**setup_ov)

    contrib = phys.contributions(setup=st.get_setup())
    mapped = phys.to_latents(setup=st.get_setup())
    for key in LATENT_KEYS:
        if contrib[key] or override_unset_latents:
            setattr(st, key, mapped[key])

    if phys.em_gain is not None:
        st.em_gain = float(phys.em_gain)
    if phys.exposure_s is not None:
        st.exposure_s = float(phys.exposure_s)
    if phys.photon_counting is not None:
        st.photon_counting = bool(phys.photon_counting)
        st.setup = st.get_setup().overridden(
            detector_mode=detector_mode_from_photon_counting_flag(
                st.photon_counting
            )
        )
    if phys.num_frames is not None:
        st.num_frames = int(phys.num_frames)

    # V-P0-03: None means keep previous value (do not clear gain state).
    if phys.mean_photons_per_mode is not None:
        nbar = float(phys.mean_photons_per_mode)
        if not np.isfinite(nbar) or nbar < 0.0:
            raise ValueError("mean_photons_per_mode must be finite and >= 0")
        st.mean_photons_per_mode = nbar
    if phys.spdc_gain_r is not None:
        r = float(phys.spdc_gain_r)
        if not np.isfinite(r) or r < 0.0:
            raise ValueError("spdc_gain_r must be finite and >= 0")
        st.spdc_gain_r = r
    for name in (
        "mean_photons_per_mode_NL1", "mean_photons_per_mode_NL2",
        "spdc_gain_r_NL1", "spdc_gain_r_NL2",
    ):
        value = getattr(phys, name)
        if value is not None:
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            setattr(st, name, numeric)
    if phys.pump_power_mW is not None:
        p = float(phys.pump_power_mW)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError("pump_power_mW must be finite and >= 0")
        st.pump_power_mW = p

    # Soft-repair F5 if physics pushes thermal high
    if st.s_thermal > 0.55:
        T_acq = st.num_frames * (st.exposure_s + 0.01)
        if T_acq < 30.0:
            st.num_frames = max(st.num_frames, 200)
            st.exposure_s = max(st.exposure_s, 0.2)

    st.physical_report = phys.report(setup=st.get_setup())
    return st


# ---------------------------------------------------------------------------
# Latent state
# ---------------------------------------------------------------------------


# ==============================================================================
# STATE + PRESETS -- LatentState.effective() applies couplings
# Role: operating point
# Models: latent vector + em_gain + exposure + setup
# Provenance: arch
# ==============================================================================

@dataclass
class LatentState:
    s_pump: float = 0.12
    s_gain: float = 0.08
    s_align: float = 0.12
    s_thermal: float = 0.10
    s_det: float = 0.15
    s_cool: float = 0.10
    s_focus: float = 0.15
    s_slm: float = 0.15
    # Detector operating knobs (not cluster latents, but coupled)
    em_gain: float = 250.0
    exposure_s: float = 0.05
    # Mirror of setup.detector_mode (V-P1-07). Prefer setting detector_mode on
    # SetupParams; this flag is kept in sync by the detector path.
    photon_counting: bool = False
    num_frames: int = 64
    # Fan-style spatial phase disturbance Δφ(x,y) inside cos(φ_eff+θ+Δφ)
    # Additive extra on top of C3 derived Fan: 2π·W_waves + span·s_align.
    # rms=0 means "no extra" (setup/C3 still drive the map when wavefront/align > 0).
    delta_phi_rms: float = 0.0
    delta_phi_corr_px: float = 0.0  # 0 → use setup.wavefront_corr_px
    # Physical SPDC operating point. Direct nbar has highest priority, followed
    # by r, followed by P_p,laser (this field or setup.pump_power_laser_mW).
    # All None uses setup default. s_pump is instability, not mean power.
    mean_photons_per_mode: float | None = None
    spdc_gain_r: float | None = None
    mean_photons_per_mode_NL1: float | None = None
    mean_photons_per_mode_NL2: float | None = None
    spdc_gain_r_NL1: float | None = None
    spdc_gain_r_NL2: float | None = None
    pump_power_mW: float | None = None
    # Optional audit trail from PhysicalParams → s (not used by forward)
    physical_report: dict[str, Any] | None = None
    # Phase-2 physical setup (digital twin). None → legacy_matched defaults.
    setup: SetupParams | None = None

    def get_setup(self) -> SetupParams:
        return self.setup if self.setup is not None else SetupParams.legacy_matched()

    def as_dict(self) -> dict[str, Any]:
        """Serialize every value needed to reproduce this operating point.

        Historical manifests contained only the eight cluster latents, which
        could silently lose an explicit fixed-nbar or calibrated pump request
        during replay.  Keep those keys and add the detector, phase-map and
        SPDC operating-point controls.
        """
        out: dict[str, Any] = {k: float(getattr(self, k)) for k in LATENT_KEYS}
        for name in (
            "em_gain", "exposure_s", "photon_counting", "num_frames",
            "delta_phi_rms", "delta_phi_corr_px",
            "mean_photons_per_mode", "spdc_gain_r",
            "mean_photons_per_mode_NL1", "mean_photons_per_mode_NL2",
            "spdc_gain_r_NL1", "spdc_gain_r_NL2", "pump_power_mW",
        ):
            value = getattr(self, name)
            out[name] = value if value is None or isinstance(value, bool) else float(value)
        return out

    def coupling_table(self) -> dict[str, float]:
        """M10: per-setup coupling overrides, else module defaults.

        Ignores obsolete ``thermal_to_focus`` (eq.162 removed; C4→C7 is physical).
        """
        setup = self.get_setup()
        base = dict(COUPLING_WEIGHTS)
        if setup.coupling_weights:
            for k, v in setup.coupling_weights.items():
                if k == "thermal_to_focus":
                    continue  # eq.162 not defined
                base[k] = float(v)
        return base

    def spdc_operating_point(
        self,
        *,
        relative_power_fluctuation: float = 0.0,
    ) -> dict[str, Any]:
        """Two independent SPDC operating points (report §7.13 / §7.14).

        Physical Mode returns k1, k2, r1, r2, nbar1, nbar2. Source moments
        N1, N2, C12 use both occupancies (§7.17). Aliases ``gain_r`` and
        ``mean_photons_per_mode`` remain pass 1.

        V-P1-01 ``spdc_source_mode``:
          - ``calibrated_source``: pump → SPDC model → nbar (P/L/T/QPM matter)
          - ``fixed_nbar``: requested nbar / ``spdc_gain_mode`` shortcut
        """
        setup = self.get_setup()
        from setup_params import resolve_architecture_profile
        architecture_profile = resolve_architecture_profile(setup)
        if not bool(getattr(setup, "use_physical_spdc_gain", False)):
            return {
                "architecture_profile": architecture_profile.to_dict(),
                "physical": False,
                "regime": "legacy_phenomenological",
                "mean_photons_per_mode": None,
                "gain_r": None,
                "pump_power_mW": self.pump_power_mW,
                "source": "s_gain",
                "spdc_source_mode": None,
                "predictive": False,
            }
        from setup_params import (
            resolved_k_spdc_per_sqrt_W,
            normalize_spdc_source_mode,
            SPDC_SOURCE_CALIBRATED,
            crystal_length_m_from_setup,
            delta_k_qpm_per_m_from_setup,
            pump_powers_from_setup,
            resolved_pump_focus_z_um,
            resolved_pump_waist_um,
            spdc_crystal_state,
            spdc_geometry_overlap_from_setup,
            spdc_propagation_direction,
        )
        from optics import resolve_spdc_source_nbar

        k1 = resolved_k_spdc_per_sqrt_W(setup, interaction=1)
        k2 = resolved_k_spdc_per_sqrt_W(setup, interaction=2)
        pump_laser = self.pump_power_mW
        if pump_laser is None:
            pump_laser = getattr(setup, "pump_power_laser_mW", None)
        dP = float(relative_power_fluctuation)
        source_mode = normalize_spdc_source_mode(
            getattr(setup, "spdc_source_mode", None)
        )

        nbar1_request = (
            self.mean_photons_per_mode_NL1
            if self.mean_photons_per_mode_NL1 is not None
            else self.mean_photons_per_mode
        )
        r1_request = (
            self.spdc_gain_r_NL1
            if self.spdc_gain_r_NL1 is not None else self.spdc_gain_r
        )
        nbar2_request = (
            self.mean_photons_per_mode_NL2
            if self.mean_photons_per_mode_NL2 is not None
            else self.mean_photons_per_mode
        )
        r2_request = (
            self.spdc_gain_r_NL2
            if self.spdc_gain_r_NL2 is not None else self.spdc_gain_r
        )

        gain1 = resolve_spdc_source_nbar(
            setup,
            pump_mW=pump_laser,
            mean_photons_per_mode=nbar1_request,
            spdc_gain_r=r1_request,
            interaction=1,
            relative_power_fluctuation=dP,
        )
        pump_powers = gain1.get("pump_powers")
        gain2 = resolve_spdc_source_nbar(
            setup,
            pump_mW=pump_laser,
            mean_photons_per_mode=nbar2_request,
            spdc_gain_r=r2_request,
            interaction=2,
            relative_power_fluctuation=dP,
        )
        if pump_powers is None:
            pump_powers = gain2.get("pump_powers")
        pump_metadata_error = None
        if pump_powers is None and pump_laser is not None:
            try:
                pump_powers = pump_powers_from_setup(
                    setup, pump_laser, relative_power_fluctuation=dP
                )
            except ValueError as exc:
                # fixed_nbar remains valid because pump does not define nbar.
                pump_metadata_error = str(exc)

        n1 = float(gain1["mean_photons_per_mode"])
        r1 = float(gain1["r0"])
        n2 = float(gain2["mean_photons_per_mode"])
        r2 = float(gain2["r0"])
        source1 = str(gain1.get("source", source_mode))
        source2 = str(gain2.get("source", source_mode))
        predictive1 = bool(gain1.get("predictive", False))
        predictive2 = bool(gain2.get("predictive", False))

        def _regime(nbar: float) -> str:
            if nbar <= float(setup.spdc_low_gain_max_mean_photons):
                return "low_gain"
            if nbar >= float(setup.spdc_high_gain_min_mean_photons):
                return "high_gain"
            return "transition"

        out = {
            "architecture_profile": architecture_profile.to_dict(),
            "physical": True,
            "regime": _regime(n1),
            "regime_NL1": _regime(n1),
            "regime_NL2": _regime(n2),
            "mean_photons_per_mode": n1,
            "mean_photons_per_mode_NL1": n1,
            "mean_photons_per_mode_NL2": n2,
            "gain_r": r1,
            "spdc_gain_r_NL1": r1,
            "spdc_gain_r_NL2": r2,
            "k_spdc_per_sqrt_W_NL1": None if k1 is None else float(k1),
            "k_spdc_per_sqrt_W_NL2": None if k2 is None else float(k2),
            "pump_power_mW": None if pump_laser is None else float(pump_laser),
            "pump_relative_power_fluctuation": dP,
            "source": source1,
            "source_NL1": source1,
            "source_NL2": source2,
            "spdc_source_mode": source_mode,
            "source_input_type": gain1.get(
                "source_input_type",
                "pump_power" if source_mode == SPDC_SOURCE_CALIBRATED else "nbar",
            ),
            "nbar_requested_NL1": gain1.get("nbar_requested"),
            "nbar_requested_NL2": gain2.get("nbar_requested"),
            "nbar_resolved_NL1": gain1.get("nbar_resolved", n1),
            "nbar_resolved_NL2": gain2.get("nbar_resolved", n2),
            "measured_power_W": gain1.get("measured_power_W"),
            "photon_rate_per_s": gain1.get("photon_rate_per_s"),
            "source_conversion_method": gain1.get("conversion_method"),
            "source_input_calibration_id": gain1.get(
                "calibration_id", gain1.get("spdc_gain_calibration_id")
            ),
            "predictive": predictive1 and predictive2,
            "predictive_NL1": predictive1,
            "predictive_NL2": predictive2,
        }
        if pump_metadata_error is not None:
            out["pump_metadata_unresolved_reason"] = pump_metadata_error
        if pump_powers is not None:
            out["pump_power_laser_mW"] = pump_powers["pump_power_laser_mW"]
            out["pump_power_laser_mean_mW"] = pump_powers["pump_power_laser_mean_mW"]
            out["pump_power_NL1_mW"] = pump_powers["pump_power_NL1_mW"]
            out["pump_power_NL2_mW"] = pump_powers["pump_power_NL2_mW"]
            out["pump_fluctuation_scale"] = pump_powers["pump_fluctuation_scale"]
            for key in (
                "P_target_1_mW", "P_target_2_mW",
                "P_actual_1_mW", "P_actual_2_mW",
                "delta_P", "abs_delta_P", "pump_balance_mode",
                "pump_imbalance_role", "hidden_equalization_applied",
            ):
                out[key] = pump_powers.get(key)
        interactions: list[dict[str, Any]] = []
        for interaction, gain in ((1, gain1), (2, gain2)):
            crystal = spdc_crystal_state(setup, interaction=interaction)
            length_m = crystal_length_m_from_setup(setup, interaction=interaction)
            direction = spdc_propagation_direction(setup, interaction=interaction)
            signed_dk = delta_k_qpm_per_m_from_setup(
                setup, interaction=interaction
            )
            local_power = None
            if pump_powers is not None:
                local_power = pump_powers[f"pump_power_NL{interaction}_mW"]
            r0 = float(gain["r0"])
            kappa = gain.get("kappa0_per_m")
            if kappa is None:
                kappa = r0 / length_m
            generation_overlap = spdc_geometry_overlap_from_setup(
                setup, interaction=interaction
            )
            configured_input_overlap = getattr(
                setup, "idler_mode_matching", None
            )
            input_mode_overlap = (
                1.0
                if interaction == 1 or configured_input_overlap is None
                else float(configured_input_overlap)
            )
            signal_transmission = float(
                setup.path1_intensity if interaction == 1 else setup.path2_intensity
            )
            relative_phase = (
                0.0
                if interaction == 1
                else float(getattr(setup, "relative_interferometric_phase", None) or 0.0)
            )
            interactions.append({
                "interaction": interaction,
                "physical_crystal_id": (
                    architecture_profile.interaction_crystal_ids[interaction - 1]
                ),
                "propagation_direction": direction,
                "pump_power_local_mW": local_power,
                "crystal_length_m": length_m,
                "crystal_length_mm": crystal["crystal_length_mm"],
                "temperature_C": crystal["T_crystal_C"],
                "poling_period_um": crystal["poling_period_um_qpm"],
                "qpm_order": crystal["qpm_order"],
                "qpm_type": crystal["qpm_type"],
                "d_eff_pm_per_V": crystal["d_eff_pm_per_V"],
                "pump_waist_um": resolved_pump_waist_um(
                    setup, interaction=interaction
                ),
                "pump_focus_z_um": resolved_pump_focus_z_um(
                    setup, interaction=interaction
                ),
                "geometry_overlap": generation_overlap,
                "generation_mode_overlap": generation_overlap,
                "input_mode_overlap": input_mode_overlap,
                "signal_transmission": signal_transmission,
                "relative_phase_rad": relative_phase,
                "delta_k_qpm_per_m": signed_dk,
                "k_spdc_per_sqrt_W": (
                    None if (k1 if interaction == 1 else k2) is None
                    else float(k1 if interaction == 1 else k2)
                ),
                "kappa_per_m": float(kappa),
                "kappa_basis": (
                    "calibrated_source" if source_mode == SPDC_SOURCE_CALIBRATED
                    else "occupancy_equivalent"
                ),
                "gain_r": r0,
                "mean_photons_per_mode": float(gain["mean_photons_per_mode"]),
                "gain_branch": gain.get("branch"),
                "g_delta_k_per_m": gain.get("g_delta_k_per_m"),
                "beta_per_m": gain.get("beta_per_m"),
                "calibration_id": gain.get("spdc_gain_calibration_id"),
                "source": gain.get("source", source_mode),
                "source_input_type": gain.get("source_input_type"),
                "nbar_requested": gain.get("nbar_requested"),
                "nbar_resolved": gain.get(
                    "nbar_resolved", gain.get("mean_photons_per_mode")
                ),
                "measured_power_W": gain.get("measured_power_W"),
                "photon_rate_per_s": gain.get("photon_rate_per_s"),
                "conversion_method": gain.get("conversion_method"),
                "source_input_calibration_id": gain.get(
                    "calibration_id", gain.get("spdc_gain_calibration_id")
                ),
                "predictive": bool(gain.get("predictive", False)),
                "equations": {
                    "gain": "r = k_spdc * sqrt(P_local_W)",
                    "occupancy_phase_matched": "nbar = sinh(r)^2",
                    "qpm": "delta_k = k_p - k_s - k_i - 2*pi*m/Lambda",
                },
            })
        out["interactions"] = interactions
        if source_mode == SPDC_SOURCE_CALIBRATED:
            out["spdc_gain_branch"] = gain1.get("branch")
            out["spdc_gain_branch_NL1"] = gain1.get("branch")
            out["spdc_gain_branch_NL2"] = gain2.get("branch")
            out["delta_k_qpm_per_m"] = interactions[0]["delta_k_qpm_per_m"]
            out["delta_k_qpm_per_m_NL2"] = interactions[1]["delta_k_qpm_per_m"]
            out["kappa0_per_m"] = interactions[0]["kappa_per_m"]
            out["kappa0_per_m_NL1"] = interactions[0]["kappa_per_m"]
            out["kappa0_per_m_NL2"] = interactions[1]["kappa_per_m"]
        return out

    def effective(self) -> dict[str, float]:
        """Apply cross-cluster couplings (phenomenological calibration weights).

        C4→C7 focus link is **not** done here (eq.162 unused).  Physical path:
        ΔT → Δf → Δz → σ_defocus (eq.160+164) with s_focus = |Δz|/Δz_max (eq.166)
        inside ``resolve_forward`` / optics.  When ``use_physical_thermal``,
        thermal→align latent weight stays 0 (overlap via Δη(T) instead).
        """
        s = self.as_dict()
        w = self.coupling_table()
        setup = self.get_setup()
        if setup is not None and bool(getattr(setup, "use_physical_thermal", False)):
            w = dict(w)
            w["thermal_to_align"] = 0.0
        if setup is not None and bool(getattr(setup, "use_physical_spdc_gain", False)):
            # Mean pump power/gain is resolved by spdc_operating_point().
            # s_pump remains instability only; induced emission is not misalignment.
            w = dict(w)
            w["pump_to_gain"] = 0.0
            w["gain_to_align"] = 0.0
        s_align = min(
            1.0,
            s["s_align"]
            + w.get("thermal_to_align", 0.0) * s["s_thermal"]
            + w["gain_to_align"] * s["s_gain"],
        )
        # s_focus is never boosted by s_thermal (no eq.162 / beta_T).
        s_focus = float(s["s_focus"])
        s_gain = min(1.0, s["s_gain"] + w["pump_to_gain"] * s["s_pump"])
        s_det = min(
            1.0,
            s["s_det"] + w["pump_to_det"] * s["s_pump"] + w["align_to_det"] * s_align,
        )
        return {
            **s,
            "s_align": s_align,
            "s_focus": s_focus,
            "s_gain": s_gain,
            "s_det": s_det,
        }


def apply_preset(name: str, **overrides: Any) -> LatentState:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Choose from {list(PRESETS)}")
    base = {k: v for k, v in PRESETS[name].items()}
    latent_over = {k: overrides[k] for k in LATENT_KEYS if k in overrides}
    base.update(latent_over)
    setup = overrides.get("setup", None)
    if isinstance(setup, (str, Path)):
        setup = load_setup_yaml(setup)
    if setup is not None and not isinstance(setup, SetupParams):
        raise TypeError("setup must be SetupParams, path, or None")

    # V-P1-07: detector_mode is explicit. Map legacy photon_counting override
    # onto detector_mode; never infer from EM gain / flux.
    if "detector_mode" in overrides:
        mode = normalize_detector_mode(overrides["detector_mode"])
        pc = detector_mode_is_photon_counting(mode)
    elif "photon_counting" in overrides:
        pc = bool(overrides["photon_counting"])
        mode = detector_mode_from_photon_counting_flag(pc)
    else:
        pc = name == "low_flux_pc"
        mode = detector_mode_from_photon_counting_flag(pc)

    if setup is None:
        setup = SetupParams(detector_mode=mode)
    else:
        setup = setup.overridden(detector_mode=mode)
    setup = validate_setup_params(setup)

    nf = int(overrides.get("num_frames", 128 if name == "low_flux_pc" else 64))
    return LatentState(
        s_pump=base["s_pump"],
        s_gain=base["s_gain"],
        s_align=base["s_align"],
        s_thermal=base["s_thermal"],
        s_cool=base["s_cool"],
        s_focus=base["s_focus"],
        s_slm=base["s_slm"],
        s_det=base.get("s_det", 0.15),
        photon_counting=pc,
        num_frames=nf,
        em_gain=float(overrides.get("em_gain", 800.0 if pc else 250.0)),
        exposure_s=float(overrides.get("exposure_s", 0.02 if pc else 0.10)),
        delta_phi_rms=float(overrides.get("delta_phi_rms", 0.0)),
        delta_phi_corr_px=float(overrides.get("delta_phi_corr_px", 0.0)),
        mean_photons_per_mode=overrides.get("mean_photons_per_mode", None),
        spdc_gain_r=overrides.get("spdc_gain_r", None),
        mean_photons_per_mode_NL1=overrides.get("mean_photons_per_mode_NL1", None),
        mean_photons_per_mode_NL2=overrides.get("mean_photons_per_mode_NL2", None),
        spdc_gain_r_NL1=overrides.get("spdc_gain_r_NL1", None),
        spdc_gain_r_NL2=overrides.get("spdc_gain_r_NL2", None),
        pump_power_mW=overrides.get("pump_power_mW", None),
        setup=setup,
    )


# ---------------------------------------------------------------------------
# Forbidden combination checker
# ---------------------------------------------------------------------------


# ==============================================================================
# VALIDITY -- runtime forbidden-combo guard + regime sampling
# Role: runtime guard
# Models: reject invalid latent combos; sample within regime
# Provenance: arch
# ==============================================================================

@dataclass
class ValidityReport:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError(
                "Forbidden QIUP combination(s):\n  - " + "\n  - ".join(self.violations)
            )


class InvalidSampleError(ValueError):
    """Dataset sample failed forbidden-state validation (V-P0-07).

    Dataset writers must fail closed: never persist a sample when
    ``check_state(...).ok`` is False.
    """

    def __init__(
        self,
        violations: list[str] | None = None,
        *,
        sample_context: str | None = None,
    ) -> None:
        self.violations = list(violations or [])
        self.sample_context = sample_context
        msg = "Invalid sample (forbidden QIUP state)"
        if sample_context:
            msg += f" [{sample_context}]"
        if self.violations:
            msg += ":\n  - " + "\n  - ".join(self.violations)
        super().__init__(msg)


class SamplingExhaustedError(RuntimeError):
    """Could not draw a valid latent within ``max_tries`` (V-P0-15).

    Must not fall back to another regime/preset (e.g. ``low_flux_pc``): that
    silently replaces requested analog/high-gain with photon-counting low-flux.
    """

    def __init__(
        self,
        requested_state: dict[str, Any],
        violations: list[str] | None = None,
        *,
        max_tries: int | None = None,
    ) -> None:
        self.requested_state = dict(requested_state or {})
        self.violations = list(violations or [])
        self.max_tries = max_tries
        bits = [
            "Sampling exhausted without a valid latent (V-P0-15)",
            f"requested={self.requested_state!r}",
        ]
        if max_tries is not None:
            bits.append(f"max_tries={max_tries}")
        if self.violations:
            bits.append("last_violations:\n  - " + "\n  - ".join(self.violations))
        super().__init__(" | ".join(bits))


# Comment 6.17 / checklist wording: SamplingError ≡ SamplingExhaustedError.
SamplingError = SamplingExhaustedError


def check_forbidden(
    state: LatentState,
    *,
    acquisition_seconds: float | None = None,
    clip_enabled: bool = True,
    defocus_b5: float | None = None,
    defocus_b7: float | None = None,
    fold_spdc_into_optical: bool = False,
) -> ValidityReport:
    """Return validity against the 12 HTML forbidden combinations."""
    e = state.effective()
    v: list[str] = []
    T_acq = acquisition_seconds
    if T_acq is None:
        T_acq = state.num_frames * (state.exposure_s + 0.01)

    # F1: stable pump + independent flux jitter — we derive flux from pump, so OK
    # (flag only if user explicitly set contradictory free params — not exposed)

    # F2 / F12: multi-pair ∝ s_gain and leakage ∝ s_slm by construction —
    # independent contradictory draws are not exposed as free parameters.

    # F3: high V with severe wavefront (C3 Fan path phase / legacy overlap).
    setup = state.get_setup()
    if setup.use_analytic_mode_overlap:
        V_proxy = float(eta_mode_from_setup(setup))
    else:
        d_ov = setup.overlap_delta_scale * e["s_align"]
        V_proxy = float(np.exp(-(d_ov ** 2) / 2))
        if V_proxy > 0.55 and e["s_align"] > 0.55:
            v.append(FORBIDDEN[2]["desc"])
    dphi_rms, _, _ = resolve_c3_fan_delta_phi_rms(
        setup,
        s_align=float(e["s_align"]),
        delta_phi_rms_extra=float(getattr(state, "delta_phi_rms", 0.0) or 0.0),
    )
    if V_proxy > 0.55 and dphi_rms > 1.0:
        v.append(FORBIDDEN[2]["desc"])

    # F4: high spatial V with delta_L_eff >> L_coh (includes thermal ΔL when enabled)
    delta_L = float(effective_delta_L_mm(setup))
    L_coh = float(setup.L_coh_mm)
    if V_proxy > 0.55 and delta_L > 1.5 * L_coh:
        v.append(FORBIDDEN[3]["desc"])

    # F5: thermal drift in short acquisition
    if e["s_thermal"] > 0.55 and T_acq < 30.0:
        v.append(FORBIDDEN[4]["desc"] + f" (T_acq={T_acq:.1f}s)")

    # F6: PC + ENF are mutually exclusive — PC path forces F²=1.
    # Exposed only if a caller requests analog ENF semantics while PC=True:
    if state.photon_counting and getattr(state, "force_enf_with_pc", False):
        v.append(FORBIDDEN[5]["desc"])

    # F7: high G should not be paired with read-noise-dominated analog stats
    if state.em_gain > 200 and e["s_det"] > 0.85 and not state.photon_counting:
        v.append(FORBIDDEN[6]["desc"])

    # F8: high G + high flux without clip
    flux_proxy = (1.0 + 0.3 * e["s_pump"]) * state.exposure_s * state.em_gain
    if flux_proxy > 80 and state.em_gain > 100 and not clip_enabled:
        v.append(FORBIDDEN[7]["desc"])

    # F9: high dark with deep cool — dark ∝ s_cool, so deep cool (low s_cool)
    # cannot coexist with an externally forced high dark flag:
    if e["s_cool"] < 0.15 and getattr(state, "force_high_dark", False):
        v.append(FORBIDDEN[8]["desc"])

    # F10: defocus block5 ≠ block7 (one physical knob)
    if defocus_b5 is not None and defocus_b7 is not None:
        if abs(defocus_b5 - defocus_b7) > 1e-6:
            v.append(FORBIDDEN[9]["desc"])

    # F11: SPDC PSF must not be folded into the optical PSF sum
    if fold_spdc_into_optical:
        v.append(FORBIDDEN[10]["desc"])

    return ValidityReport(ok=len(v) == 0, violations=v)


def check_state(
    state: LatentState,
    *,
    acquisition_seconds: float | None = None,
    clip_enabled: bool = True,
    defocus_b5: float | None = None,
    defocus_b7: float | None = None,
    fold_spdc_into_optical: bool = False,
) -> ValidityReport:
    """Dataset/runtime validity gate (V-P0-07).

    Alias of ``check_forbidden`` with the same arguments. Dataset writers must
    call this before persisting a sample and raise ``InvalidSampleError`` when
    ``ok`` is False.
    """
    return check_forbidden(
        state,
        acquisition_seconds=acquisition_seconds,
        clip_enabled=clip_enabled,
        defocus_b5=defocus_b5,
        defocus_b7=defocus_b7,
        fold_spdc_into_optical=fold_spdc_into_optical,
    )


def sample_latents(
    rng: np.random.Generator,
    *,
    prefer_pc: bool = True,
    max_tries: int = 40,
    regime: str = "lab",
    setup: SetupParams | None = None,
) -> LatentState:
    """Sample a physically valid latent state with couplings (M8 regimes).

    V-P0-15: if no valid draw is found within ``max_tries``, raise
    ``SamplingExhaustedError``. Never fall back to ``low_flux_pc`` (or any
    other preset) — that would silently change detector/gain regime.
    """
    if regime not in REGIME_BOUNDS:
        raise KeyError(f"Unknown regime {regime!r}; choose from {list(REGIME_BOUNDS)}")
    bounds = REGIME_BOUNDS[regime]
    setup = setup if setup is not None else SetupParams.legacy_matched()
    mode = detector_mode_from_photon_counting_flag(prefer_pc)
    setup = validate_setup_params(
        setup.overridden(sampling_regime=regime, detector_mode=mode)
    )
    requested_state = {
        "regime": regime,
        "prefer_pc": bool(prefer_pc),
        "detector_mode": mode,
        "sampling_regime": getattr(setup, "sampling_regime", regime),
        "spdc_gain_mode": getattr(setup, "spdc_gain_mode", None),
        "blur_model": getattr(setup, "blur_model", None),
        "setup_id": getattr(setup, "setup_id", None),
        "max_tries": int(max_tries),
    }
    last_violations: list[str] = []
    for _ in range(max_tries):
        raw = {
            k: float(rng.uniform(bounds[k][0], bounds[k][1])) for k in LATENT_KEYS
        }
        if regime == "lab":
            raw["s_gain"] *= 0.85
        st = LatentState(
            **raw,
            photon_counting=prefer_pc,
            num_frames=int(rng.choice([48, 64, 96, 128])),
            em_gain=float(rng.choice([200, 400, 800])) if prefer_pc else float(rng.uniform(50, 300)),
            exposure_s=float(rng.uniform(0.01, 0.05)) if prefer_pc else float(rng.uniform(0.05, 0.3)),
            setup=setup,
        )
        if st.effective()["s_thermal"] > 0.55:
            st.num_frames = max(st.num_frames, 200)
            st.exposure_s = max(st.exposure_s, 0.2)
        report = check_forbidden(st)
        if report.ok:
            return st
        last_violations = list(report.violations)
    raise SamplingExhaustedError(
        requested_state,
        last_violations,
        max_tries=int(max_tries),
    )


# ---------------------------------------------------------------------------
# Clean reference (same path as MSKHAKI)
# ---------------------------------------------------------------------------
