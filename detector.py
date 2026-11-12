"""
QIUP Digital Twin -- SECTION: detector.py  (signal path step 4)
EMCCD readout chain, physically ordered. Converts an ideal intensity field
into a realistic ADU frame.

Order (Block 8): QE.PRNU -> dark(M5)+CIC(pre-EM) -> Poisson -> hot
  -> stochastic Gamma EM (excess-noise F^2~2) -> soft saturation
  -> signed read noise (post-EM) -> bias -> ADC clip / 16-bit ADU.
Two critical correctnesses live here: EM gain is STOCHASTIC (rng.gamma,
giving F^2=2), and read noise is signed AFTER multiplication and BEFORE
bias/ADC clip (V-P0-06; do not rectify noise before bias). The optical
PSF is applied upstream in optics.py, so Poisson sees an already-blurred
field (shot noise stays spatially white).

CONTENTS: DETECTOR_STAGES_*, spawn_qiup_rngs, _fixed_patterns, _detector_pc_frame
Provenance: qiup_system_architecture.py (fixed chain)
"""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter
from setup_params import (
    SetupParams, resolve_forward, dark_rate_from_setup,
    read_noise_e_from_setup, sensor_temperature_C,
    normalize_detector_mode, detector_mode_is_photon_counting,
    normalize_detector_output_port, resolved_adc_bit_depth,
    validate_detector_configuration, resolve_architecture_profile,
)
from latents import LatentState

# ==============================================================================
# STAGE ORDER (documented in meta)
# Role: documentation
# Models: analog vs photon-counting order
# Provenance: arch
# ==============================================================================

DETECTOR_STAGES_ANALOG = (
    "QE_PRNU",
    "dark_current",                # M5: D(T) or legacy exp(s_cool)
    "CIC_pre_EM",                  # M2: CIC enters rate before EM
    "Poisson_shot",
    "hot_pixels_pre_EM",           # M6
    "EM_Gamma_F2approx2",
    "soft_saturation",
    "signed_read_noise_post_EM",   # V-P0-06: no rectifying clip before bias
    "bias_offset",
    "adc_clip_16bit",
)


DETECTOR_STAGES_PC = (
    "QE_PRNU",
    "dark_current",
    "CIC_pre_EM",
    "Poisson_shot",
    "hot_pixels_pre_EM",
    "EM_then_read_then_threshold",  # M2 ordering
)


# Named streams spawned from one master seed (V-P0-12). Order is part of the
# contract — do not reorder without a seed-version bump.
QIUP_RNG_STREAM_NAMES: tuple[str, ...] = (
    "rng_optics",
    "rng_detector",
    "rng_plus",
    "rng_minus",
)


def spawn_qiup_rngs(
    seed: int | np.random.SeedSequence,
) -> dict[str, np.random.Generator]:
    """Create independent RNGs via ``SeedSequence.spawn`` (V-P0-12).

    Streams
    -------
    rng_optics
        Optical path (pump jitter, phase noise, …).
    rng_detector
        Fixed detector patterns (PRNU / hot mask) shared across the sequence.
    rng_plus / rng_minus
        Per-arm shot/EM/read noise — never share a single stream.
    """
    ss = (
        seed
        if isinstance(seed, np.random.SeedSequence)
        else np.random.SeedSequence(int(seed))
    )
    children = ss.spawn(len(QIUP_RNG_STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(QIUP_RNG_STREAM_NAMES, children, strict=True)
    }


def resolve_detector_mode(setup: SetupParams) -> str:
    """Return explicit ``setup.detector_mode`` (V-P1-07). Never infer."""
    return normalize_detector_mode(getattr(setup, "detector_mode", None))


def detector_stages_for_setup(setup: SetupParams) -> tuple[str, ...]:
    """Stage list for the explicit detector_mode."""
    resolve_architecture_profile(setup)
    if detector_mode_is_photon_counting(resolve_detector_mode(setup)):
        return DETECTOR_STAGES_PC
    bits = resolved_adc_bit_depth(setup)
    if bits == 16:
        return DETECTOR_STAGES_ANALOG
    return (*DETECTOR_STAGES_ANALOG[:-1], f"adc_clip_{bits}bit")


def validate_detector_operating_state(state: LatentState) -> LatentState:
    """Validate acquisition knobs before optics, RNG draws, or output writes."""
    setup = state.get_setup()
    validate_detector_configuration(setup)
    for name in ("exposure_s", "em_gain"):
        value = getattr(state, name)
        if isinstance(value, bool) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite real number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        invalid = numeric < 1.0 if name == "em_gain" else numeric <= 0.0
        if invalid:
            relation = ">= 1" if name == "em_gain" else "> 0"
            raise ValueError(f"{name} must be {relation}")
    if isinstance(state.num_frames, bool) or not isinstance(
        state.num_frames, (int, np.integer)
    ):
        raise ValueError("num_frames must be a positive integer")
    if int(state.num_frames) <= 0:
        raise ValueError("num_frames must be a positive integer")
    return state


def resolved_detector_parameters(state: LatentState) -> dict[str, Any]:
    """Return the detector settings actually used by the shared readout chain."""
    validate_detector_operating_state(state)
    setup = state.get_setup()
    e = state.effective()
    selected_port = normalize_detector_output_port(
        getattr(setup, "detector_output_port", None)
    )
    return {
        "implementation": "detector._detector_pc_frame",
        "detector_mode": resolve_detector_mode(setup),
        "selected_output_port": selected_port,
        "selected_optical_quantity": f"I_{selected_port}",
        "detected_field": "signal",
        "idler_directly_detected": False,
        "optical_input_units": "incident_signal_photons_per_pixel_per_exposure",
        "signal_exposure_application_count": 1,
        "dark_exposure_application_count": 1,
        "quantum_efficiency": float(setup.QE),
        "exposure_s": float(state.exposure_s),
        "electron_multiplication_gain": float(state.em_gain),
        "emccd_model": str(setup.emccd_model),
        "read_noise_e_rms": read_noise_e_from_setup(setup, e["s_det"]),
        "read_noise_model": str(setup.read_noise_model),
        "dark_current_e_per_px_s": dark_rate_from_setup(setup, e["s_cool"]),
        "cic_mean_e_per_px_per_frame": float(setup.cic0) * (0.5 + e["s_det"]),
        "optical_background_photons": float(
            getattr(setup, "optical_background_photons", 0.0) or 0.0
        ),
        "optical_background_applied_before_qe": True,
        "shot_noise_model": "poisson_on_electron_mean",
        "shot_noise_status": (
            "approximation_not_validated_multimode_spdc_variance"
        ),
        "prnu_fractional_sigma": float(setup.prnu_sigma) * e["s_det"],
        "hot_pixel_fraction": float(setup.hot_frac_at_s1) * e["s_cool"],
        "hot_pixel_amplitude_e_range": [
            float(setup.hot_amp_lo), float(setup.hot_amp_hi)
        ],
        "hot_pixel_model": str(setup.hot_pixel_model),
        "sensor_temperature_C": sensor_temperature_C(setup, e["s_cool"]),
        "saturation_level_e": float(setup.full_well_e),
        "saturation_model": "soft_full_well_before_read_noise",
        "adc_conversion_gain_adu_per_e": float(setup.g_adu),
        "adc_bias_adu": float(setup.bias_adu),
        "adc_bit_depth": resolved_adc_bit_depth(setup),
        "adc_max_adu": float(setup.max_adu),
        "stage_order": list(detector_stages_for_setup(setup)),
        "operating_latent": {
            "name": "s_det",
            "value": float(e["s_det"]),
            "controls": ["read_noise", "CIC", "PRNU"],
        },
        "cooling_latent": {
            "name": "s_cool",
            "value": float(e["s_cool"]),
            "controls": ["sensor_temperature", "dark_current", "hot_pixels"],
        },
    }


def detector_expected_electron_mean(
    I_signal_photons: np.ndarray,
    state: LatentState,
    *,
    prnu: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the pre-Poisson electron mean for an integrated signal image.

    ``I_signal_photons`` already contains the exposure integration performed by
    the optical model. Exposure therefore appears only on the dark-current
    term here. Optional ``optical_background_photons`` is P04-style additive
    light at the detector plane and is *not* dark current or read noise.
    """
    validate_detector_operating_state(state)
    I = np.asarray(I_signal_photons)
    if I.ndim != 2:
        raise ValueError("I_signal_photons must be a two-dimensional image")
    if not np.all(np.isfinite(I)):
        raise ValueError("I_signal_photons contains NaN/Inf")
    if prnu is None:
        prnu_arr = np.ones(I.shape, dtype=np.float32)
    else:
        prnu_arr = np.asarray(prnu)
        if prnu_arr.shape != I.shape:
            raise ValueError("prnu must have the same shape as I_signal_photons")
        if not np.all(np.isfinite(prnu_arr)) or np.any(prnu_arr < 0.0):
            raise ValueError("prnu must be finite and non-negative")

    e = state.effective()
    setup = state.get_setup()
    QE = float(setup.QE)
    dark_rate = dark_rate_from_setup(setup, e["s_cool"])
    cic = float(setup.cic0) * (0.5 + e["s_det"])
    optical_bg = float(getattr(setup, "optical_background_photons", 0.0) or 0.0)
    if optical_bg < 0.0:
        raise ValueError(
            f"optical_background_photons must be >= 0, got {optical_bg}"
        )
    lam = (
        QE * prnu_arr * (np.clip(I, 0, None) + optical_bg)
        + dark_rate * state.exposure_s
        + cic
    )
    if not bool(getattr(setup, "use_physical_spdc_gain", False)):
        lam = lam + float(setup.accidental_e_at_s1) * e["s_gain"] ** 2
    return np.clip(lam, 0.0, None)


def sample_input_electrons(
    expected_electrons: np.ndarray,
    rng: np.random.Generator,
    *,
    hot: np.ndarray | None = None,
) -> np.ndarray:
    """Sample shot electrons and add independent pre-EM hot-pixel electrons."""
    lam = np.asarray(expected_electrons)
    if not np.all(np.isfinite(lam)) or np.any(lam < 0.0):
        raise ValueError("expected_electrons must be finite and non-negative")
    n_in = rng.poisson(lam).astype(np.float32)
    if hot is not None:
        hot_arr = np.asarray(hot)
        if hot_arr.shape != lam.shape:
            raise ValueError("hot must have the same shape as expected_electrons")
        if not np.all(np.isfinite(hot_arr)) or np.any(hot_arr < 0.0):
            raise ValueError("hot must be finite and non-negative")
        n_in = n_in + hot_arr
    return n_in


def sample_em_register(
    input_electrons: np.ndarray,
    gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the existing conditional Gamma EM-register distribution."""
    n_in = np.asarray(input_electrons)
    if not np.all(np.isfinite(n_in)) or np.any(n_in < 0.0):
        raise ValueError("input_electrons must be finite and non-negative")
    G = float(gain)
    if not np.isfinite(G) or G <= 0.0:
        raise ValueError("gain must be finite and > 0")
    n_em = np.zeros_like(n_in, dtype=np.float32)
    mask = n_in > 0
    if np.any(mask):
        n_em[mask] = rng.gamma(
            shape=np.maximum(n_in[mask], 1e-6), scale=G
        ).astype(np.float32)
    return n_em


# ==============================================================================
# FIXED PATTERNS + DETECTOR CHAIN
# Role: detector physics
# Models: PRNU/hot masks; QE->dark->Poisson->EM->read->ADU
# Provenance: arch
# ==============================================================================

def _fixed_patterns(
    shape: tuple[int, int],
    state: LatentState,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """PRNU + hot pixels — fixed across the sequence (M6 mask + optional T scale)."""
    e = state.effective()
    setup = state.get_setup()
    prnu = (
        1.0 + float(setup.prnu_sigma) * e["s_det"] * rng.normal(0.0, 1.0, size=shape)
    ).astype(np.float32)
    prnu = np.clip(prnu, 0.0, None).astype(np.float32, copy=False)
    hot = np.zeros(shape, dtype=np.float32)
    frac = float(setup.hot_frac_at_s1) * e["s_cool"]
    n_hot = int(round(frac * shape[0] * shape[1]))
    if n_hot > 0:
        idx = rng.choice(shape[0] * shape[1], size=n_hot, replace=False)
        amps = rng.uniform(
            float(setup.hot_amp_lo), float(setup.hot_amp_hi), size=n_hot
        )
        if setup.hot_pixel_model == "mask_T":
            T = sensor_temperature_C(setup, e["s_cool"])
            amps = amps * np.exp(
                float(setup.hot_T_coef_per_C) * (T - float(setup.T_sensor_min_C))
            )
        hot.flat[idx] = amps
    return prnu, hot


def _detector_pc_frame(
    I: np.ndarray,
    state: LatentState,
    rng: np.random.Generator,
    *,
    prnu: np.ndarray,
    hot: np.ndarray,
) -> np.ndarray:
    """
    Ordered detector chain (Block 8) — EMCCD path (M1/M2 / V-P0-06 / V-P1-07):

      QE·PRNU → dark (M5) + CIC (pre-EM) → Poisson → hot (M6)
      → EM → soft saturation → signed read noise → bias → ADC clip
      [or PC: EM → signed read → threshold]

    Path selection uses explicit ``setup.detector_mode`` only
    (``analog_emccd`` | ``photon_counting``). Never inferred from
    ``LatentState.photon_counting``, EM gain, or flux.

    ``emccd_model="legacy"``: PC uses deterministic n·G (Phase-0 fingerprints).
    ``emccd_model="register"``: stochastic Gamma EM for both PC and analog.
    """
    validate_detector_operating_state(state)
    e = state.effective()
    setup = state.get_setup()
    mode = resolve_detector_mode(setup)
    use_pc = detector_mode_is_photon_counting(mode)
    # Mirror flag for meta/forbidden consumers — not an independent control.
    state.photon_counting = bool(use_pc)

    fwd = resolve_forward(setup, int(I.shape[0]), s_focus=e["s_focus"])
    s_read = read_noise_e_from_setup(setup, e["s_det"])
    T_sens = sensor_temperature_C(setup, e["s_cool"])

    # (1)+(2) photoelectrons + dark·t + CIC (+ mild accidental) — CIC pre-EM (M2)
    lam = detector_expected_electron_mean(I, state, prnu=prnu)
    # Legacy detector-level surrogate only. Physical SPDC mode already carries
    # multi-pair/stimulated effects through optical N1, N2, C12. Do not add a
    # multipair/stimulated pedestal here (report §7.22). Analog EMCCD is the
    # paper detector path; photon-counting remains for old tests only.
    # (3) shot noise + hot electrons pre-EM
    n_in = sample_input_electrons(lam, rng, hot=hot)

    G = max(float(state.em_gain), 1.0)
    use_register = setup.emccd_model == "register"

    if use_pc:
        if use_register:
            # M2 register: stochastic EM then signed read then threshold
            n_em = sample_em_register(n_in, G, rng)
            n_out = n_em + rng.normal(0.0, s_read, size=I.shape).astype(np.float32)
        else:
            # Legacy PC (Phase-0): deterministic multiply + signed read
            n_out = n_in * G + rng.normal(0.0, s_read, size=I.shape).astype(np.float32)
        thr = float(fwd.pc_threshold_k) * s_read
        return (n_out > thr).astype(np.float32)

    # ---- Analog EMCCD (detector_mode=analog_emccd) ----
    # EM multiplication (non-negative gamma samples)
    n_em = sample_em_register(n_in, G, rng)

    # Soft saturation on EM charge (before readout electronics)
    full_well_e = float(fwd.full_well_e)
    x = n_em / max(full_well_e, 1e-12)
    n_sat = (full_well_e * x / (1.0 + x)).astype(np.float32)

    # V-P0-06: signed read noise after EM, before bias — do NOT clip to >=0 here.
    electrons = n_sat + rng.normal(0.0, s_read, size=I.shape).astype(np.float32)

    g_adu, bias, max_adu = float(fwd.g_adu), float(fwd.bias_adu), float(fwd.max_adu)
    # Bias after signed noise; ADC clipping is the last stage.
    adu = np.round(electrons * g_adu + bias)
    _ = T_sens
    return np.clip(adu, 0.0, max_adu).astype(np.float32)


# ---------------------------------------------------------------------------
# Full forward simulation
# ---------------------------------------------------------------------------
