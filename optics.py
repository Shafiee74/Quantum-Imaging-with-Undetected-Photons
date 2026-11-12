"""
QIUP Digital Twin -- SECTION: optics.py  (PHYSICS CORE, signal path steps 1-3)
object -> SLM -> SPDC correlation PSF -> induced-coherence interference (G1)
-> optical PSF -> mode-conditional collection envelope. Output: ideal intensity
fields I+/I-.

Single source of truth for the interference: _interfere(). Clean and noisy
twins both go through _optical_intensity via simulate_forward (V-P0-04).
SPDC PSF is kept STRICTLY separate from the optical PSF (different physics).
Optical PSF is applied BEFORE the detector Poisson step (enforced at the
simulate.py boundary).

CONTENTS
  - Constants: OPTICAL_STAGES, VISIBILITY_DEFAULTS, PSF_DEFAULTS, SLM_TAXONOMY
  - Object encoding (steps 1-2): load_object_M, embed_object_within_beam, build_idler_object
  - Interference core: _interfere (unified)
  - Legacy primitives: lemos_optical_mapping, simulate_signal_images
  - Twin optics: _slm_complex, _visibility_factors, _lens ring, _optical_intensity
  - Object reference: build_clean_reference (T/φ only; clean D from simulate_forward)
Provenance: upi_pipeline.py (steps 1-3) + qiup_system_architecture.py
"""
from __future__ import annotations
import hashlib
import os
import time
from typing import Any
import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform, gaussian_filter, zoom
from setup_params import (
    SetupParams,
    ResolvedForward,
    internal_camera_sampling_pixel_um,
    resolve_forward,
    resolve_c3_fan_delta_phi_rms,
    um_to_px,
    um_to_px_setup,
    object_pixel_size_um,
    beam_size_px,
    psf_size_px,
    resolve_imaging_magnification,
    pump_overlap_eta,
    eta_mode_from_setup,
    two_beam_visibility,
    idler_displacement_um,
    resolved_idler_waist_um,
    field_overlap_visibility_map,
    mode_tilt_factor,
    v_coh_gaussian,
    effective_delta_L_mm,
    optical_path_difference_mm,
    opd_phase_rad,
    thermal_phase_offset_rad,
    thermal_beam_shift_um,
    THETA0_RAD,
    crystal_length_m_from_setup,
    delta_k_qpm_per_m_from_setup,
    pump_power_W_from_mW,
    resolved_k_spdc_per_sqrt_W,
    spdc_gain_r_from_pump_mW,
    idler_path_transmission_from_setup,
    signal_path_transmissions_from_setup,
    normalize_spdc_architecture,
    resolve_architecture_profile,
    validate_forward_profile_supported,
    normalize_imaging_mode,
    normalize_blur_model,
    normalize_object_preprocessing,
    normalize_spdc_source_mode,
    resolve_blur_sigmas,
    resolved_spatial_mapping,
    final_camera_collection_envelope,
    resolved_final_camera_collection_policy,
    imaging_correlation_type,
    IMAGING_MODE_NEAR,
    IMAGING_MODE_FAR,
    BLUR_MODEL_SINGLE,
    BLUR_MODEL_DOUBLE,
    OBJECT_PREPROCESS_CALIBRATED,
    OBJECT_PREPROCESS_VIS,
    OBJECT_PREPROCESS_DEFAULT,
    SPDC_SOURCE_CALIBRATED,
    SPDC_SOURCE_FIXED,
    FIXED_SOURCE_INPUT_NBAR,
    FIXED_SOURCE_INPUT_MEASURED_POWER,
    CAMERA_FILTER_MODEL_GAUSSIAN,
    CAMERA_FILTER_MODEL_MEASURED,
    photon_rate_from_power_W,
    spdc_crystal_state,
    qpm_poling_period_um,
    qpm_configuration_evidence,
    effective_nonlinearity_metadata,
    spdc_geometry_overlap_from_setup,
    sinh_squared_stable,
    scaled_sinh_squared_stable,
    SPDCGainOverflowError,
    pump_powers_from_setup,
    spdc_gain_mode_nbar,
    is_two_crystal_architecture,
    resolve_object_intensity_and_phase,
    ARCHITECTURE_PROPAGATION_COMPLEX_V1,
    OBJECT_TYPE_TRANSMISSIVE,
    OBJECT_TYPE_REFLECTIVE,
    OBJECT_TRANSFORMS,
    pump_generation_amplitude_weight_from_setup,
    resolved_pump_waist_um,
    near_field_joint_gaussian_coefficients,
    near_field_gain_1e_um,
    near_field_visibility_1e_um,
    equivalent_wavelength_nm,
    idler_lens_equivalent_propagation_um,
    resolved_far_field_focal_mm,
)
from latents import LatentState

# ==============================================================================
# CONSTANTS
# Role: defaults / stage order
# Models: fallback visibility, PSF, SLM taxonomy
# Provenance: upi+arch
# ==============================================================================

EPS = 1e-8
KHAKI_VISIBILITY_VARIATION_MAX = 0.15


OPTICAL_STAGES = (
    "object_slm_complex",          # C8: hardware + object + calibration error
    "slm_misregistration",         # M9 (no-op when misreg=0)
    "qiup_spatial_transfer",       # near_field (position) / far_field (momentum) kernel select
    "spdc_correlation_psf",        # C2: σ_SPDC on full complex object (gated by blur_model)
    "induced_coherence_interference",
    "lens_misalignment_rings",     # L06: idler-lens φ_I on seed before C12
    "optical_imaging_psf",         # C7: diffraction + defocus (NOT SPDC)
    "image_quality_factors",       # M3/M4: extra blur / scale / rotation
    "etalon_fabry_perot_fringes",  # unwanted filter/window etalon (optional)
    "vignetting",
)

# Ordered analog EMCCD stages (Day-2 roadmap) — documented in meta["detector_stages"]


VISIBILITY_DEFAULTS: dict[str, float] = {
    "V0": 1.0,                  # residual scale; 1 = physical factors only
    "overlap_delta_scale": 1.15,
    "delta_L_mm": 0.01,         # PDF §4.1 lab path mismatch [mm]
    "L_coh_mm": 0.20,           # PDF §4.1; editable [0.02, 0.50] mm
    "path1_intensity": 1.0,     # PDF §4.2 I1 (relative)
    "path2_intensity": 1.0,     # PDF §4.2 I2 (relative)
    "V_gain_drop": 0.40,
    "V_map_mottle": 0.25,
    "V_lemos_ref": 0.77,        # experimental reference (not used in forward math)
}

# S8 — PSF reference widths (px); physical derivation comes in Phase 2/3


PSF_DEFAULTS: dict[str, float] = {
    # σ_SPDC is not a constant: derived in resolve_forward (PDF §3.1 + eq. 50)
    "sigma_diffraction_px": None,  # derived via eq. 144 when physical
    "defocus_span_px": 3.6,         # σ_defocus,max (legacy); physical = K·Δz_max
    "K_defocus_formula": "eta_def*M*NA/(n*p)",
}

# S9 — SLM three-layer taxonomy (hardware / object / calibration error)


SLM_TAXONOMY = {
    "hardware": {
        "description": "Fixed device specs; not driven by s_slm",
        "params": {
            "levels0": 256.0,          # 8-bit ideal
            "bit_depth": 8,
            "phi_max_rad": float(2.0 * np.pi),
        },
    },
    "object_function": {
        "description": "Displayed object field O=A·exp(iφ); provided by ref['phi'], ref['T']",
        "params": ("phi", "T"),
    },
    "calibration_error": {
        "description": "Controlled by latent s_slm ∈ [0,1]",
        "params": {
            "quant_coef": 0.985,       # levels collapse
            "levels_floor": 3,
            "gamma_span": 0.9,         # γ = 1 + span * s_slm
            "leak_eta": 0.40,          # phase–amplitude coupling at s_slm=1
        },
    },
}

# Optical / thermal legacy scales still hard-wired numerically elsewhere;
# listed here for Phase-1 vocabulary (values unchanged).


# ==============================================================================
# HELPERS + OBJECT ENCODING (steps 1-2)
# Role: object -> T, phi, complex
# Models: phase-only SLM; calibrated vs visualization preprocessing (V-P0-08/09)
# Provenance: upi
# ==============================================================================

def normalize(img: np.ndarray) -> np.ndarray:
    """Content min-max normalize to [0, 1] — visualization / demo only.

    Must not be used on calibrated physics maps (V-P0-09).
    """
    img = np.asarray(img, dtype=np.float32)
    return (img - img.min()) / (img.max() - img.min() + EPS)


def offline_object_fixture(
    index: int = 2,
    size: int = 28,
) -> tuple[np.ndarray, str]:
    """Deterministic offline object when MNIST/torchvision is unavailable (V-P0-08)."""
    size = max(8, int(size))
    rng = np.random.default_rng(1000 + int(index))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (size / 6.0) ** 2))
    # Fixed-scale 8-bit-like map [0, 255] — not content-normalized.
    img = np.clip(200.0 * blob + 20.0 * rng.random((size, size)), 0.0, 255.0).astype(
        np.float32
    )
    return img, f"offline_fixture_{int(index)}"


def prepare_object_map(
    raw: np.ndarray,
    *,
    object_preprocessing: str | None = None,
    phi_max_rad: float | None = None,
    source_kind: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a raw object array under ``object_preprocessing`` (V-P0-09).

    calibrated:
      - no content min-max
      - ``source_kind='image8'``: fixed scale /255 (full digital scale)
      - ``source_kind='auto'``: [0,1] → fraction; otherwise physical passthrough
        (e.g. phase already in rad — including values ≤ 2π)
      - optional ``phi_max_rad`` maps a unit fraction to radians
    visualization_only:
      - content min-max to [0,1] (display/demo only)
    """
    mode = normalize_object_preprocessing(object_preprocessing)
    arr = np.asarray(raw, dtype=np.float32)
    kind = str(source_kind or "auto").strip().lower()
    meta: dict[str, Any] = {
        "object_preprocessing": mode,
        "source_kind": kind,
        "input_min": float(arr.min()) if arr.size else 0.0,
        "input_max": float(arr.max()) if arr.size else 0.0,
        "minmax_applied": False,
        "physical_units": mode == OBJECT_PREPROCESS_CALIBRATED,
    }
    if mode == OBJECT_PREPROCESS_VIS:
        out = normalize(arr)
        meta["minmax_applied"] = True
        meta["output_units"] = "normalized_0_1"
        return out.astype(np.float32), meta

    # calibrated — never content min-max
    mn = float(arr.min()) if arr.size else 0.0
    mx = float(arr.max()) if arr.size else 0.0
    if kind in ("image8", "uint8", "pil", "image"):
        unit = np.clip(arr, 0.0, 255.0) / 255.0
        meta["bit_scale"] = "uint8_fullscale_255"
        if phi_max_rad is not None:
            out = (unit * float(phi_max_rad)).astype(np.float32)
            meta["output_units"] = "rad"
            meta["phi_max_rad"] = float(phi_max_rad)
        else:
            out = unit.astype(np.float32)
            meta["output_units"] = "fraction_0_1_fixed_scale"
    elif kind in ("fraction", "unit") or (
        kind == "auto" and mx <= 1.0 + 1e-6 and mn >= -1e-6
    ):
        unit = np.clip(arr, 0.0, 1.0)
        meta["bit_scale"] = "unit_interval"
        if phi_max_rad is not None:
            out = (unit * float(phi_max_rad)).astype(np.float32)
            meta["output_units"] = "rad"
            meta["phi_max_rad"] = float(phi_max_rad)
        else:
            out = unit.astype(np.float32)
            meta["output_units"] = "fraction_0_1"
    else:
        # Physical passthrough (phase [rad], calibrated T, etc.)
        out = arr.astype(np.float32, copy=True)
        meta["bit_scale"] = "passthrough_physical"
        meta["output_units"] = "physical_as_loaded"
    return out, meta


def embed_object_within_beam(
    M: np.ndarray,
    *,
    N: int,
    beam_waist_px: float,
    fill: float = 0.85,
    object_preprocessing: str | None = None,
) -> np.ndarray:
    """Center-scale object so its extent stays inside the pump beam profile.

    Beam intensity uses ``exp(-2 r²/w²)`` with waist ``w = beam_waist_px``
    (1/e² radius). The object is inscribed in a square of side
    ``fill·√2·w`` and hard-clipped to disk radius ``fill·w``, then
    zero-padded to ``N×N``.

    Calibrated mode preserves physical amplitudes (no min-max). Visualization
    mode may normalize for display embedding.
    """
    mode = normalize_object_preprocessing(object_preprocessing)
    M = np.asarray(M, dtype=np.float32)
    if mode == OBJECT_PREPROCESS_VIS:
        M = normalize(M)
    w = max(float(beam_waist_px), 1.0)
    fill = float(np.clip(fill, 0.05, 1.0))
    # Inscribe object in the 1/e² beam disk (radius w): square side ≤ fill·√2·w
    max_side = int(max(8, min(int(N), round(fill * np.sqrt(2.0) * w))))
    # Tight crop on non-near-zero content when possible
    peak = float(np.max(np.abs(M))) if M.size else 0.0
    mask = np.abs(M) > 0.05 * max(peak, EPS)
    if np.any(mask):
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        M = M[y0:y1, x0:x1]
    h, w_img = M.shape
    scale = max_side / float(max(h, w_img))
    if abs(scale - 1.0) > 1e-6:
        M = zoom(M, scale, order=1).astype(np.float32)
    if mode == OBJECT_PREPROCESS_VIS:
        M = normalize(M)
    out = np.zeros((int(N), int(N)), dtype=np.float32)
    hh, ww = M.shape
    ys = max(0, (int(N) - hh) // 2)
    xs = max(0, (int(N) - ww) // 2)
    out[ys : ys + hh, xs : xs + ww] = M[: min(hh, int(N) - ys), : min(ww, int(N) - xs)]
    # Hard clip support to beam disk radius fill·w
    yy, xx = np.mgrid[0 : int(N), 0 : int(N)]
    cy = cx = (int(N) - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    out = np.where(r2 <= (fill * w) ** 2, out, 0.0).astype(np.float32)
    if mode == OBJECT_PREPROCESS_VIS and float(out.max()) > 0:
        return normalize(out)
    return out


def resize_to_original(img: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    zy = original_shape[0] / img.shape[0]
    zx = original_shape[1] / img.shape[1]
    return zoom(img, (zy, zx), order=1)


def spatial_overlap_map(
    shape: tuple[int, ...],
    visibility_variation: float,
    rng: np.random.Generator,
    smooth_sigma: float = 8.0,
    unit_pattern: np.ndarray | None = None,
) -> np.ndarray:
    """Signal–idler spatial overlap O(x,y) in [0, 1] (smooth, fixed pattern optional)."""
    if visibility_variation <= 0:
        return np.ones(shape, dtype=np.float32)
    if unit_pattern is not None:
        p = unit_pattern.astype(np.float64)
    else:
        raw = rng.normal(0.0, 1.0, size=shape)
        p = gaussian_filter(raw, sigma=smooth_sigma).astype(np.float64)
    p_norm = (p - p.min()) / (p.max() - p.min() + EPS)
    null_frac = min(float(visibility_variation) / KHAKI_VISIBILITY_VARIATION_MAX, 1.0) * 0.15
    null_frac = min(max(null_frac, 0.0), 0.95)
    cutoff = float(np.quantile(p_norm, null_frac))
    hi = float(p_norm.max())
    overlap = np.where(
        p_norm <= cutoff,
        0.0,
        (p_norm - cutoff) / max(hi - cutoff, EPS),
    )
    return np.clip(overlap, 0.0, 1.0).astype(np.float32)


def load_object_M(
    image_path: str | None = None,
    target_size: int = 128,
    use_mnist_fallback: bool = True,
    mnist_index: int = 2,
    output_dir: str = "UPI_Synthetic/output",
    *,
    beam_waist_px: float | None = None,
    object_beam_fill: float = 0.85,
    object_preprocessing: str | None = None,
    phi_max_rad: float | None = None,
) -> tuple[np.ndarray, str]:
    """Load object map for the twin (V-P0-08 / V-P0-09).

    ``object_preprocessing='calibrated'`` (default): fixed-scale / passthrough
    physical units — **no** content min-max. Phase maps stay in radians when
    ``phi_max_rad`` is set (or already physical).

    ``object_preprocessing='visualization_only'``: opt-in min-max to [0, 1]
    for display/demo only.

    Offline fixture is used when MNIST/torchvision is unavailable (no network
    required).
    """
    os.makedirs(output_dir, exist_ok=True)
    mode = normalize_object_preprocessing(object_preprocessing)
    source = "unknown"
    source_kind = "auto"

    if image_path and os.path.isfile(image_path):
        if image_path.lower().endswith(".npy"):
            raw = np.load(image_path)
            label = os.path.splitext(os.path.basename(image_path))[0]
            source = "npy"
            source_kind = "auto"
        else:
            img = Image.open(image_path).convert("L")
            label = os.path.splitext(os.path.basename(image_path))[0]
            native0 = max(28, int(target_size))
            img = img.resize((native0, native0), Image.BICUBIC)
            raw = np.array(img, dtype=np.float32)
            source = "image"
            source_kind = "image8"
    elif use_mnist_fallback:
        try:
            from torchvision import datasets

            ds = datasets.MNIST(root="./data", train=True, download=False)
            sample = ds[mnist_index][0]
            if hasattr(sample, "numpy"):
                img = Image.fromarray(sample.numpy())
            else:
                img = (
                    sample.convert("L")
                    if hasattr(sample, "convert")
                    else Image.fromarray(np.array(sample))
                )
            label = str(ds[mnist_index][1])
            native0 = max(28, int(target_size))
            img = img.resize((native0, native0), Image.BICUBIC)
            raw = np.array(img, dtype=np.float32)
            source = "mnist_local"
            source_kind = "image8"
        except Exception:
            raw, label = offline_object_fixture(index=mnist_index, size=28)
            source = "offline_fixture"
            source_kind = "image8"
    else:
        raise FileNotFoundError(
            "Set image_path or enable use_mnist_fallback / offline fixture"
        )

    # Geometry to target grid before unit conversion.
    if beam_waist_px is not None:
        M0, prep_meta = prepare_object_map(
            raw,
            object_preprocessing=mode,
            phi_max_rad=phi_max_rad,
            source_kind=source_kind,
        )
        M = embed_object_within_beam(
            M0,
            N=int(target_size),
            beam_waist_px=float(beam_waist_px),
            fill=object_beam_fill,
            object_preprocessing=mode,
        )
    else:
        native = max(28, int(target_size))
        if raw.shape[0] != native or raw.shape[1] != native:
            raw = resize_to_original(np.asarray(raw, dtype=np.float32), (native, native))
        if raw.shape != (int(target_size), int(target_size)):
            raw = resize_to_original(
                np.asarray(raw, dtype=np.float32),
                (int(target_size), int(target_size)),
            )
        M, prep_meta = prepare_object_map(
            raw,
            object_preprocessing=mode,
            phi_max_rad=phi_max_rad,
            source_kind=source_kind,
        )

    out_name = (
        "M_visualization.npy"
        if mode == OBJECT_PREPROCESS_VIS
        else "M_calibrated.npy"
    )
    np.save(os.path.join(output_dir, out_name), M)
    # Keep legacy filename for older notebooks; content follows preprocessing mode.
    np.save(os.path.join(output_dir, "M_normalized.npy"), M)
    prep_meta["source"] = source
    import json

    with open(os.path.join(output_dir, "M_preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)
    return M.astype(np.float32), label


def _load_user_object_array(path: str, *, role: str) -> np.ndarray:
    """Load one user map without assigning it QIUP physics."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{role} path must be a non-empty string")
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{role} file does not exist: {resolved}")
    if resolved.lower().endswith(".npy"):
        array = np.asarray(np.load(resolved), dtype=np.float64)
    else:
        array = np.asarray(Image.open(resolved).convert("F"), dtype=np.float64)
        if role in {"amplitude", "intensity"} and float(np.max(array)) > 1.0:
            array = array / 255.0
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{role} map must be a non-empty 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{role} map contains NaN/Inf")
    return array


def load_user_object_response(
    *,
    phase_map_path: str | None,
    amplitude_map_path: str | None,
    intensity_map_path: str | None,
    target_shape: tuple[int, int],
) -> dict[str, Any]:
    """SLM adapter over the generic complex optical response.

    Phase is in radians. Amplitude and intensity conventions are mutually
    exclusive; both are bounded to [0,1]. Input dimensions may differ and are
    resampled onto the declared object grid before the correlation model.
    """
    if amplitude_map_path is not None and intensity_map_path is not None:
        raise ValueError(
            "provide either object_amplitude_map_path or object_intensity_map_path, not both"
        )
    paths = {
        "phase": phase_map_path,
        "amplitude": amplitude_map_path,
        "intensity": intensity_map_path,
    }
    if not any(paths.values()):
        raise ValueError("at least one user object map path is required")
    target = (int(target_shape[0]), int(target_shape[1]))
    if min(target) < 1:
        raise ValueError("target_shape must contain positive dimensions")

    def _resample(array: np.ndarray) -> np.ndarray:
        if array.shape == target:
            return array
        return zoom(
            array,
            (target[0] / array.shape[0], target[1] / array.shape[1]),
            order=1,
        )

    loaded: dict[str, np.ndarray] = {}
    file_evidence: dict[str, Any] = {}
    for role, path in paths.items():
        if path is None:
            continue
        array = _resample(_load_user_object_array(path, role=role))
        if not np.all(np.isfinite(array)):
            raise ValueError(f"resampled {role} map contains NaN/Inf")
        loaded[role] = array
        resolved = os.path.abspath(path)
        with open(resolved, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        file_evidence[role] = {
            "path": resolved,
            "sha256": digest,
            "native_shape": list(_load_user_object_array(path, role=role).shape),
            "resolved_shape": list(target),
        }
    phase = loaded.get("phase", np.zeros(target, dtype=np.float64))
    if "amplitude" in loaded:
        amplitude = loaded["amplitude"]
        if np.any(amplitude < 0.0) or np.any(amplitude > 1.0):
            raise ValueError("object amplitude map must be in [0,1]")
        intensity = amplitude ** 2
        convention = "field_amplitude_squared_to_intensity"
    else:
        intensity = loaded.get("intensity", np.ones(target, dtype=np.float64))
        if np.any(intensity < 0.0) or np.any(intensity > 1.0):
            raise ValueError("object transmission/reflectance map must be in [0,1]")
        convention = "intensity_transmission_or_reflectance"
    field = np.sqrt(intensity) * np.exp(1j * phase)
    return {
        "T": intensity.astype(np.float32),
        "phi": phase.astype(np.float32),
        "O_i": field.astype(np.complex64),
        "file_evidence": file_evidence,
        "amplitude_intensity_convention": convention,
        "phase_convention": "radians",
        "coordinate_orientation": "object_xy",
    }


def build_idler_object(
    M: np.ndarray,
    phase_only_slm: bool = True,
    phi_max: float = np.pi,
    T_min: float = 0.15,
    T_max: float = 1.0,
    output_dir: str = "UPI_Synthetic/output",
    *,
    object_preprocessing: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode object into transmission T and phase φ [rad] (V-P0-09).

    calibrated: if ``M`` is already in radians (or prepared with phi_max), use
    it as φ directly (clipped to hardware span). Do not re-normalize.
    visualization_only: treat ``M`` as [0,1] and map φ = phi_max · M.
    """
    os.makedirs(output_dir, exist_ok=True)
    mode = normalize_object_preprocessing(object_preprocessing)
    M = np.asarray(M, dtype=np.float32)
    phi_span = float(phi_max)

    if mode == OBJECT_PREPROCESS_CALIBRATED:
        mx = float(np.max(np.abs(M))) if M.size else 0.0
        # Already physical radians if values exceed a unit interval meaningfully.
        if mx > 1.0 + 1e-3:
            phi = np.clip(M, -phi_span, phi_span).astype(np.float32)
            unit = np.clip(np.abs(phi) / max(phi_span, EPS), 0.0, 1.0)
        else:
            # Fixed-scale fraction [0,1] → radians (no content min-max).
            unit = np.clip(M, 0.0, 1.0)
            phi = (phi_span * unit).astype(np.float32)
        if phase_only_slm:
            T = np.ones_like(M, dtype=np.float32)
        else:
            T = np.clip(T_min + (T_max - T_min) * unit, 0.0, 1.0).astype(np.float32)
    else:
        # visualization_only: expect / force [0,1] map
        unit = normalize(M) if (float(M.max()) - float(M.min())) > EPS else M
        unit = np.clip(unit, 0.0, 1.0).astype(np.float32)
        if phase_only_slm:
            T = np.ones_like(M, dtype=np.float32)
        else:
            T = np.clip(T_min + (T_max - T_min) * unit, 0.0, 1.0).astype(np.float32)
        phi = (phi_span * unit).astype(np.float32)

    O_i = np.sqrt(T) * np.exp(1j * phi)

    np.save(os.path.join(output_dir, "T_transmission.npy"), T)
    np.save(os.path.join(output_dir, "phi_phase.npy"), phi)
    np.save(os.path.join(output_dir, "idler_complex_object.npy"), O_i)
    return T, phi, O_i


# ==============================================================================
# INTERFERENCE CORE (single source of truth)
# Role: physics core
# Models: PDF §4.2 eq. 69  I = I1 + I2 ± 2√(I1 I2) Γ √T cos(φ)
# Provenance: unified from upi + arch; V from eq. 70 not from r(s_align)
# ==============================================================================

def _interfere(I1, I2, Gamma, T, phi, theta=0.0, delta_phi=0.0):
    """Two-beam induced-coherence intensities (PDF §4.2 eq. 69).

    I± = I1 + I2 ± 2 √(I1 I2) Γ √T cos(φ + θ + Δφ)

    I1, I2 : path intensities (scalar or map)
    Gamma  : remaining coherence/overlap factors in [0, 1] (not a port ratio)
    T      : idler object intensity transmission
    phi    : object phase; theta / delta_phi : interferometer + wavefront phase
    """
    a = np.clip(np.asarray(I1, dtype=np.float32), 0.0, None)
    b = np.clip(np.asarray(I2, dtype=np.float32), 0.0, None)
    G = np.clip(np.asarray(Gamma, dtype=np.float32), 0.0, 1.0)
    T_c = np.clip(np.asarray(T, dtype=np.float32), 0.0, None)
    phi_tot = np.asarray(phi, dtype=np.float32) + float(theta) + np.asarray(delta_phi, dtype=np.float32)
    cross = (2.0 * np.sqrt(a * b) * G * np.sqrt(T_c) * np.cos(phi_tot)).astype(np.float32)
    I_plus = np.clip(a + b + cross, 0.0, None).astype(np.float32)
    I_minus = np.clip(a + b - cross, 0.0, None).astype(np.float32)
    return I_plus, I_minus


def spdc_nbar_from_r0_L_dk(
    *,
    r0: float,
    crystal_length_m: float,
    delta_k_qpm_per_m: float,
) -> dict[str, Any]:
    """Mean photons per mode at finite QPM mismatch (report §5.8 / §7.9).

    Phase-matched r0 = k_SPDC √P_W, κ0 = r0 / L (eq. 46). Then
        g_Δk² = κ0² − (Δk_QPM / 2)²
    hyperbolic (g_Δk² ≥ 0, eq. 48):
        nbar = (κ0² / g_Δk²) sinh²(g_Δk L)
    oscillatory (g_Δk² < 0, eq. 50–51):
        β = √((Δk_QPM / 2)² − κ0²),  nbar = (κ0² / β²) sin²(β L)
    Δk_QPM = 0 (eq. 49): nbar = sinh²(r0).

    High gain (V-P1-02): ``sinh²`` uses log-domain
    ``log_nbar = 2*(x - log(2))`` with overflow → ``SPDCGainOverflowError``
    (never silent zero).
    """
    r = float(r0)
    L = float(crystal_length_m)
    dk = float(delta_k_qpm_per_m)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("r0 must be finite and >= 0")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(f"crystal length must be > 0 m, got {crystal_length_m}")
    if not np.isfinite(dk):
        raise ValueError("delta_k_qpm_per_m must be finite")
    kappa0 = r / L
    half_dk = 0.5 * dk
    g2 = kappa0 * kappa0 - half_dk * half_dk
    scale = max(abs(kappa0), abs(half_dk), 1.0)
    crit = (1.0e-12 * scale) ** 2
    if abs(dk) * L < 1.0e-12:
        nbar = sinh_squared_stable(r)
        branch = "phase_matched"
        g_dk = kappa0
        beta = 0.0
    elif g2 > crit:
        g_dk = float(np.sqrt(g2))
        # (κ0² / g²) sinh²(g L) — log-stable at high gain
        nbar = scaled_sinh_squared_stable(kappa0 * kappa0 / g2, g_dk * L)
        branch = "hyperbolic"
        beta = 0.0
    elif g2 < -crit:
        beta = float(np.sqrt(-g2))
        nbar = float((kappa0 * kappa0 / (beta * beta)) * np.sin(beta * L) ** 2)
        branch = "oscillatory"
        g_dk = 0.0
    else:
        nbar = float(r * r)
        branch = "critical"
        g_dk = 0.0
        beta = 0.0
    if not np.isfinite(nbar) or nbar < 0.0:
        raise SPDCGainOverflowError(
            f"SPDC nbar overflow/non-finite ({nbar}) on branch={branch}, r0={r}"
        )
    return {
        "mean_photons_per_mode": nbar,
        "r0": r,
        "kappa0_per_m": float(kappa0),
        "delta_k_qpm_per_m": dk,
        "g_delta_k_per_m": float(g_dk) if branch == "hyperbolic" else None,
        "beta_per_m": float(beta) if branch == "oscillatory" else None,
        "branch": branch,
        "crystal_length_m": L,
    }


def spdc_nbar_from_k_pump_L_dk(
    *,
    k_spdc_per_sqrt_W: float,
    pump_power_W: float,
    crystal_length_m: float,
    delta_k_qpm_per_m: float,
) -> dict[str, Any]:
    """nbar from k_SPDC, P_W, L, Δk_QPM (report §7.9)."""
    k = float(k_spdc_per_sqrt_W)
    P = float(pump_power_W)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k_SPDC must be finite and > 0")
    if not np.isfinite(P) or P < 0.0:
        raise ValueError("pump_power_W must be finite and >= 0")
    out = spdc_nbar_from_r0_L_dk(
        r0=k * np.sqrt(P),
        crystal_length_m=crystal_length_m,
        delta_k_qpm_per_m=delta_k_qpm_per_m,
    )
    out["k_spdc_per_sqrt_W"] = k
    out["pump_power_W"] = P
    return out


def spdc_nbar_from_setup_pump(
    setup: SetupParams,
    pump_mW: float,
    *,
    interaction: int = 1,
) -> dict[str, Any]:
    """nbar for one pass from pump power on that pass (report §7.9).

    ``pump_mW`` is the power on the crystal for this pass (P_p1 or P_p2),
    not P_p,laser. Report §7.12 derives those from ηp,pre / ηp,return.
    r0 = k_SPDC √P_W. Δk_QPM from collection wavelengths and Λ(T).
    O_geom is already inside k_SPDC; it is not applied again.

    This is the calibrated_source generation chain (V-P1-01)::

        pump_power → SPDC model (k, L, Δk_QPM) → nbar
    """
    r0 = spdc_gain_r_from_pump_mW(setup, pump_mW, interaction=interaction)
    out = spdc_nbar_from_r0_L_dk(
        r0=r0,
        crystal_length_m=crystal_length_m_from_setup(
            setup, interaction=interaction
        ),
        delta_k_qpm_per_m=delta_k_qpm_per_m_from_setup(
            setup, interaction=interaction
        ),
    )
    k_W = resolved_k_spdc_per_sqrt_W(setup, interaction=interaction)
    out["k_spdc_per_sqrt_W"] = None if k_W is None else float(k_W)
    out["pump_power_W"] = pump_power_W_from_mW(pump_mW)
    out["pump_power_mW"] = float(pump_mW)
    out["spdc_source_mode"] = "calibrated_source"
    out["predictive"] = True
    return out


def resolve_spdc_source_nbar(
    setup: SetupParams,
    *,
    pump_mW: float | None = None,
    mean_photons_per_mode: float | None = None,
    spdc_gain_r: float | None = None,
    interaction: int = 1,
    relative_power_fluctuation: float = 0.0,
) -> dict[str, Any]:
    """Resolve nbar under explicit ``spdc_source_mode`` (V-P1-01).

    calibrated_source
        ``pump → SPDC generation model → nbar`` (responds to P, L, T/QPM).
        Requires pump power. Fixed nbar/r overrides are rejected.
    fixed_nbar
        Use requested occupancy (latent nbar, r, or ``spdc_gain_mode`` shortcut).
        Pump/T/QPM do not change nbar (explicit non-predictive dataset path).
    """
    mode = normalize_spdc_source_mode(getattr(setup, "spdc_source_mode", None))
    if mode == SPDC_SOURCE_CALIBRATED:
        if mean_photons_per_mode is not None or spdc_gain_r is not None:
            raise ValueError(
                "spdc_source_mode='calibrated_source' computes nbar from pump "
                "(V-P1-01). Clear mean_photons_per_mode/spdc_gain_r or set "
                "spdc_source_mode='fixed_nbar' for a requested occupancy."
            )
        if pump_mW is None:
            pump_mW = getattr(setup, "pump_power_laser_mW", None)
        if pump_mW is None:
            raise ValueError(
                "spdc_source_mode='calibrated_source' requires pump_power_mW "
                "(latent) or setup.pump_power_laser_mW; refusing constant "
                "nbar surrogate (V-P1-01)."
            )
        calibration_id = (
            getattr(setup, "spdc_gain_calibration_id_NL2", None)
            if int(interaction) == 2 else None
        )
        # The two-crystal architecture has independent nonlinear sources.
        # A calibration for crystal 1 is not silently reused for crystal 2.
        if calibration_id is None and not (
            int(interaction) == 2 and is_two_crystal_architecture(setup)
        ):
            calibration_id = getattr(setup, "spdc_gain_calibration_id", None)
        if not str(calibration_id or "").strip():
            raise ValueError(
                "spdc_source_mode='calibrated_source' requires a documented "
                "spdc_gain_calibration_id (or interaction-2 identifier); "
                "refusing uncalibrated predictive brightness"
            )
        powers = pump_powers_from_setup(
            setup,
            pump_mW,
            relative_power_fluctuation=float(relative_power_fluctuation),
        )
        key = (
            "pump_power_NL1_mW" if int(interaction) == 1 else "pump_power_NL2_mW"
        )
        p_pass = powers[key]
        if p_pass is None:
            raise ValueError(f"{key} could not be derived from P_p,laser")
        out = spdc_nbar_from_setup_pump(
            setup, float(p_pass), interaction=interaction
        )
        out["spdc_source_mode"] = mode
        out["source"] = "calibrated_source_pump"
        out["predictive"] = True
        out["spdc_gain_calibration_id"] = str(calibration_id)
        out["pump_powers"] = powers
        out["source_input_type"] = "pump_power"
        out["nbar_requested"] = None
        out["nbar_resolved"] = out["mean_photons_per_mode"]
        out["measured_power_W"] = None
        out["photon_rate_per_s"] = None
        out["conversion_method"] = "calibrated_spdc_gain_from_local_pump_power"
        out["calibration_id"] = str(calibration_id)
        return out

    # fixed_nbar — requested occupancy / gain_mode shortcut
    fixed_input = getattr(setup, "fixed_source_input", None) or FIXED_SOURCE_INPUT_NBAR
    if fixed_input == FIXED_SOURCE_INPUT_MEASURED_POWER:
        if mean_photons_per_mode is not None or spdc_gain_r is not None:
            raise ValueError(
                "fixed_source_input='measured_power' cannot be combined with "
                "a direct nbar or r request"
            )
        power_W = float(setup.measured_power_W)
        wavelength_nm = float(
            setup.measured_power_wavelength_nm or setup.lambda_s_nm
        )
        calibration = (
            setup.measured_power_to_nbar_per_W_NL2
            if int(interaction) == 2
            and setup.measured_power_to_nbar_per_W_NL2 is not None
            else setup.measured_power_to_nbar_per_W
        )
        if calibration is None or not str(
            setup.measured_power_to_nbar_calibration_id or ""
        ).strip():
            raise ValueError(
                "power alone provides photon rate but cannot determine modal nbar; "
                "a calibrated power-to-nbar mapping and calibration ID are required"
            )
        nbar = float(power_W * float(calibration))
        if not np.isfinite(nbar) or nbar < 0.0:
            raise ValueError("calibrated measured-power mapping produced invalid nbar")
        r0 = float(np.arcsinh(np.sqrt(nbar)))
        return {
            "mean_photons_per_mode": nbar,
            "r0": r0,
            "spdc_source_mode": mode,
            "source": "measured_power_calibrated_to_nbar",
            "predictive": False,
            "branch": "fixed_nbar_measured_power_calibration",
            "source_input_type": FIXED_SOURCE_INPUT_MEASURED_POWER,
            "nbar_requested": None,
            "nbar_resolved": nbar,
            "measured_power_W": power_W,
            "photon_rate_per_s": photon_rate_from_power_W(power_W, wavelength_nm),
            "measured_power_wavelength_nm": wavelength_nm,
            "conversion_method": "documented_linear_power_to_modal_nbar_calibration",
            "calibration_id": str(setup.measured_power_to_nbar_calibration_id),
            "pump_power_mW": None if pump_mW is None else float(pump_mW),
        }

    if mean_photons_per_mode is not None:
        nbar = float(mean_photons_per_mode)
        if not np.isfinite(nbar) or nbar < 0.0:
            raise ValueError("mean_photons_per_mode must be finite and >= 0")
        r0 = float(np.arcsinh(np.sqrt(nbar)))
        source = "mean_photons_per_mode"
    elif spdc_gain_r is not None:
        r0 = float(spdc_gain_r)
        if not np.isfinite(r0) or r0 < 0.0:
            raise ValueError("spdc_gain_r must be finite and >= 0")
        nbar = float(sinh_squared_stable(r0))
        source = "spdc_gain_r"
    else:
        nbar = float(
            getattr(
                setup,
                "spdc_default_mean_photons_per_mode",
                spdc_gain_mode_nbar(getattr(setup, "spdc_gain_mode", None)),
            )
        )
        r0 = float(np.arcsinh(np.sqrt(max(nbar, 0.0))))
        source = "spdc_gain_mode_default"
    return {
        "mean_photons_per_mode": nbar,
        "r0": r0,
        "spdc_source_mode": mode,
        "source": source,
        "predictive": False,
        "branch": "fixed_nbar",
        "source_input_type": FIXED_SOURCE_INPUT_NBAR,
        "nbar_requested": nbar,
        "nbar_resolved": nbar,
        "measured_power_W": None,
        "photon_rate_per_s": None,
        "conversion_method": "direct_mean_photons_per_spatiotemporal_mode",
        "calibration_id": None,
        "pump_power_mW": None if pump_mW is None else float(pump_mW),
    }


def physical_mode_flux(flux0: float, nbar: float, nref: float) -> float:
    """Physical Mode optical flux scale (report §7.10 / §7.14).

    Φ = flux0 · nbar / nref. nbar already includes P_p(t); do not multiply
    by (1+dP) again. After §7.17 the occupancy also lives in N1, N2, C12;
    the forward then uses flux0/nref as the per-mode scale instead of
    multiplying this Φ onto relative intensities.
    """
    ref = float(nref)
    if ref <= 0.0:
        raise ValueError("spdc_reference_mean_photons_per_mode must be > 0")
    return float(flux0) * (float(nbar) / ref)


def idler_object_amplitude(T_object, phi_object=0.0):
    """to = √To exp(i φo) (report eq. 10)."""
    To = np.clip(np.asarray(T_object, dtype=np.float64), 0.0, None)
    ph = np.asarray(phi_object, dtype=np.float64)
    return np.sqrt(To) * np.exp(1j * ph)


def idler_seed_from_parts(eta_mode, eta_i_path, t_object):
    """Unified idler seed (report §4.4 / §7.15).

    Tseed = ηmode · ηi,path · To
    tseed = √(ηmode ηi,path) · to
    ηmode is **relay** intensity overlap (V-P1-04); generation O_geom is
    already inside k_SPDC / nbar and must not also multiply visibility.
    """
    eta_m = float(np.clip(float(eta_mode), 0.0, 1.0))
    eta_i = float(np.clip(float(eta_i_path), 0.0, 1.0))
    t_o = np.asarray(t_object, dtype=np.complex128)
    t_seed = np.sqrt(eta_m * eta_i) * t_o
    T_seed = np.real(t_seed * np.conjugate(t_seed))
    T_seed = np.clip(T_seed, 0.0, None)
    return {
        "t_seed": t_seed,
        "T_seed": T_seed,
        "eta_mode": eta_m,
        "eta_i_path": eta_i,
    }


def spdc_double_pass_moments(nbar1, nbar2, t_seed):
    """Source moments before signal-path loss (report §4.3–4.6 / §7.17).

    N1,src = nbar1
    N2,src = nbar2 (1 + Tseed nbar1)
    C12,src = tseed √(nbar1 nbar2 (1+nbar1))
    """
    n1 = float(nbar1)
    n2 = float(nbar2)
    if not np.isfinite(n1) or n1 < 0.0:
        raise ValueError("nbar1 must be finite and >= 0")
    if not np.isfinite(n2) or n2 < 0.0:
        raise ValueError("nbar2 must be finite and >= 0")
    t = np.asarray(t_seed, dtype=np.complex128)
    T_seed = np.clip(np.real(t * np.conjugate(t)), 0.0, None)
    N1_src = np.full_like(T_seed, n1, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            N2_src = n2 * (1.0 + T_seed * n1)
        except FloatingPointError as exc:
            raise SPDCGainOverflowError(
                "SPDC second-interaction moment exceeds float64 domain"
            ) from exc
    amp = _stable_sqrt_positive_product(n1, n2, 1.0 + n1)
    C12_src = t * amp
    if not np.all(np.isfinite(N2_src)) or not np.all(np.isfinite(C12_src)):
        raise SPDCGainOverflowError("SPDC source moments are non-finite")
    return {
        "N1_source": N1_src,
        "N2_source": N2_src,
        "C12_source": C12_src,
        "T_seed": T_seed,
        "t_seed": t,
        "moment_model": "independent_source_idler_seeding",
        "assumptions": (
            "independent SPDC interactions; undepleted pump; one matched "
            "spatiotemporal mode per pixel"
        ),
    }


def _stable_sqrt_positive_product(*values):
    """sqrt(product(values)) without overflowing an intermediate product."""
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in values])
    if any(np.any(~np.isfinite(v)) or np.any(v < 0.0) for v in arrays):
        raise ValueError("square-root product inputs must be finite and nonnegative")
    zero = np.zeros_like(arrays[0], dtype=bool)
    log_sum = np.zeros_like(arrays[0], dtype=np.float64)
    for value in arrays:
        zero |= value == 0.0
        with np.errstate(divide="ignore"):
            log_sum += np.where(value > 0.0, np.log(value), 0.0)
    half_log = 0.5 * log_sum
    if np.any((~zero) & (half_log > np.log(np.finfo(np.float64).max))):
        raise SPDCGainOverflowError("SPDC coherence moment exceeds float64 domain")
    with np.errstate(over="raise"):
        out = np.where(zero, 0.0, np.exp(half_log))
    return float(out) if out.ndim == 0 else out


def spdc_michelson_return_moments(
    nbar1,
    nbar2,
    idler_return_amplitude,
    signal_return_amplitude,
    *,
    signal_seeds_second_interaction: bool = True,
):
    """Two-squeezer single-crystal Michelson component moments.

    With ``u_j=sqrt(1+n_j)`` and ``v_j=sqrt(n_j)``, the first-pass fields
    propagate as ``s_r=a_s s_1+loss`` and ``i_r=a_i i_1+loss`` before the
    backward squeezer.  The detected backward signal is decomposed into its
    returned first-pass and second-interaction contributions.  In the phase
    convention used by the shared combiner, ``C12=<S2^dagger S1_return>``.

    When the returned signal enters interaction 2:

    ``N1=(1+n2)|a_s|^2 n1``
    ``N2=n2(1+|a_i|^2 n1)``
    ``C12=a_s a_i sqrt[n1(1+n1)n2(1+n2)]``.

    If the selected topology bypasses signal seeding, the ``(1+n2)`` factors
    are absent from N1 and C12.  Loss vacua preserve commutators and contribute
    no normally ordered photons.
    """
    n1 = float(nbar1)
    n2 = float(nbar2)
    if not np.isfinite(n1) or n1 < 0.0:
        raise ValueError("nbar1 must be finite and >= 0")
    if not np.isfinite(n2) or n2 < 0.0:
        raise ValueError("nbar2 must be finite and >= 0")
    if type(signal_seeds_second_interaction) is not bool:
        raise ValueError("signal_seeds_second_interaction must be a boolean")
    ai = np.asarray(idler_return_amplitude, dtype=np.complex128)
    a_s = complex(signal_return_amplitude)
    if not np.all(np.isfinite(ai)) or not all(
        np.isfinite(v) for v in (a_s.real, a_s.imag)
    ):
        raise ValueError("Michelson return amplitudes must be finite")
    T_i = np.real(ai * np.conjugate(ai))
    if np.any(T_i < -1.0e-12) or np.any(T_i > 1.0 + 1.0e-9):
        raise ValueError("idler return intensity must be in [0, 1]")
    if abs(a_s) > 1.0 + 1.0e-9:
        raise ValueError("signal return amplitude magnitude must be <= 1")
    T_i = np.clip(T_i, 0.0, 1.0)
    signal_gain = 1.0 + n2 if signal_seeds_second_interaction else 1.0
    with np.errstate(over="raise", invalid="raise"):
        try:
            N1_src = np.full_like(T_i, signal_gain * (abs(a_s) ** 2) * n1)
            N2_src = n2 * (1.0 + T_i * n1)
        except FloatingPointError as exc:
            raise SPDCGainOverflowError(
                "Michelson intensity moment exceeds float64 domain"
            ) from exc
    factors = [n1, 1.0 + n1, n2]
    if signal_seeds_second_interaction:
        factors.append(1.0 + n2)
    coherence_scale = _stable_sqrt_positive_product(*factors)
    C12_src = a_s * ai * coherence_scale
    if not (
        np.all(np.isfinite(N1_src))
        and np.all(np.isfinite(N2_src))
        and np.all(np.isfinite(C12_src))
    ):
        raise SPDCGainOverflowError("Michelson source moments are non-finite")
    return {
        "N1_source": N1_src,
        "N2_source": N2_src,
        "C12_source": C12_src,
        "T_seed": T_i,
        "t_seed": ai,
        "signal_return_amplitude": a_s,
        "signal_return_intensity": float(abs(a_s) ** 2),
        "signal_seeds_second_interaction": signal_seeds_second_interaction,
        "moment_model": "single_crystal_two_squeezer_bogoliubov",
        "assumptions": (
            "undepleted pump; one matched spatiotemporal mode per pixel; "
            "passive loss represented by vacuum ports"
        ),
    }


def apply_signal_path_losses(
    N1_source,
    N2_source,
    C12_source,
    eta_s1: float = 1.0,
    eta_s2: float = 1.0,
    gamma_path: float = 1.0,
):
    """N1, N2, C12 after signal transmissions (report §4.7 / §7.18 / §7.21).

    N1 = ηs1 N1,src
    N2 = ηs2 N2,src
    C12 = √(ηs1 ηs2) γpath C12,src
    γpath is applied only here, not again on visibility.
    """
    e1 = max(float(eta_s1), 0.0)
    e2 = max(float(eta_s2), 0.0)
    g = float(np.clip(float(gamma_path), 0.0, 1.0))
    N1 = e1 * np.asarray(N1_source, dtype=np.float64)
    N2 = e2 * np.asarray(N2_source, dtype=np.float64)
    C12 = np.sqrt(e1 * e2) * g * np.asarray(C12_source, dtype=np.complex128)
    return {"N1": N1, "N2": N2, "C12": C12, "gamma_path": g}


def apply_signal_path_amplitudes(
    N1_source,
    N2_source,
    C12_source,
    signal1_amplitude,
    signal2_amplitude,
    gamma_path: float = 1.0,
):
    """Propagate source moments with complex signal-path field amplitudes.

    For detected fields ``E1=a1*S1`` and ``E2=a2*S2``:

    ``N1=|a1|^2 N1_src``, ``N2=|a2|^2 N2_src``, and
    ``C12=conj(a1)*a2*gamma_path*C12_src``.

    This is the shared detector-side propagation used for an explicit
    single-crystal signal return.  It does not claim to model stimulated
    high-gain feedback into the second nonlinear interaction.
    """
    a1 = complex(signal1_amplitude)
    a2 = complex(signal2_amplitude)
    if not all(np.isfinite(v) for v in (a1.real, a1.imag, a2.real, a2.imag)):
        raise ValueError("signal path amplitudes must be finite")
    g = float(np.clip(float(gamma_path), 0.0, 1.0))
    N1 = (abs(a1) ** 2) * np.asarray(N1_source, dtype=np.float64)
    N2 = (abs(a2) ** 2) * np.asarray(N2_source, dtype=np.float64)
    C12 = (
        np.conjugate(a1)
        * a2
        * g
        * np.asarray(C12_source, dtype=np.complex128)
    )
    return {
        "N1": N1,
        "N2": N2,
        "C12": C12,
        "gamma_path": g,
        "signal1_amplitude": a1,
        "signal2_amplitude": a2,
    }


def spdc_observables_from_moments(N1, N2, C12):
    """g^{(1)} and V_50:50 from final moments (report eq. 19–20 / §7.20)."""
    n1 = np.asarray(N1, dtype=np.float64)
    n2 = np.asarray(N2, dtype=np.float64)
    abs_c = np.abs(np.asarray(C12, dtype=np.complex128))
    if np.any(~np.isfinite(n1)) or np.any(~np.isfinite(n2)) or np.any(~np.isfinite(abs_c)):
        raise SPDCGainOverflowError("SPDC observables require finite moments")
    if np.any(n1 < 0.0) or np.any(n2 < 0.0):
        raise ValueError("SPDC intensity moments must be nonnegative")
    scale = np.maximum(np.maximum(n1, n2), abs_c)
    n1_scaled = np.divide(n1, scale, out=np.zeros_like(n1), where=scale > 0.0)
    n2_scaled = np.divide(n2, scale, out=np.zeros_like(n2), where=scale > 0.0)
    c_scaled = np.divide(abs_c, scale, out=np.zeros_like(abs_c), where=scale > 0.0)
    scaled_product = n1_scaled * n2_scaled
    g1 = np.divide(
        c_scaled,
        np.sqrt(np.maximum(scaled_product, 0.0)),
        out=np.zeros_like(abs_c, dtype=np.float64),
        where=scaled_product > 0.0,
    )
    v50 = np.divide(
        2.0 * c_scaled,
        n1_scaled + n2_scaled,
        out=np.zeros_like(abs_c, dtype=np.float64),
        where=(n1_scaled + n2_scaled) > 0.0,
    )
    if np.any(g1 > 1.0 + 1.0e-9) or np.any(v50 > 1.0 + 1.0e-9):
        raise ValueError("unphysical SPDC moments violate first-order coherence bounds")
    return {
        "intrinsic_g1": g1,
        "visibility_50_50": v50,
        "N1": n1,
        "N2": n2,
        "C12": np.asarray(C12, dtype=np.complex128),
        "C12_abs": abs_c,
    }


def _interfere_from_moments(N1, N2, C12, theta=0.0, delta_phi=0.0):
    """Combiner intensities from final N1, N2, C12 (report eq. 21 / §7.19).

    I± = (N1+N2)/2 ± Re[ C12 exp(i(θ+Δφ)) ]
    Object phase already lives in tseed/C12. Stimulated emission is not
    rebuilt here.
    """
    n1 = np.asarray(N1, dtype=np.float64)
    n2 = np.asarray(N2, dtype=np.float64)
    c = np.asarray(C12, dtype=np.complex128)
    phase = float(theta) + np.asarray(delta_phi, dtype=np.float64)
    cross = np.real(c * np.exp(1j * phase))
    mid = 0.5 * n1 + 0.5 * n2
    plus = mid + cross
    minus = mid - cross
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise SPDCGainOverflowError("interference intensity exceeds float64 domain")
    tolerance = 1.0e-10 * np.maximum(mid, 1.0)
    if np.any(plus < -tolerance) or np.any(minus < -tolerance):
        raise ValueError("unphysical moments produced a negative combiner intensity")
    if np.any(plus > np.finfo(np.float32).max) or np.any(minus > np.finfo(np.float32).max):
        raise SPDCGainOverflowError(
            "interference intensity exceeds the simulator float32 image domain"
        )
    return (
        np.maximum(plus, 0.0).astype(np.float32),
        np.maximum(minus, 0.0).astype(np.float32),
    )


def spdc_gain_observables(
    mean_photons_per_mode,
    T,
    path1_intensity: float = 1.0,
    path2_intensity: float = 1.0,
    nbar2=None,
    t_seed=None,
    gamma_path: float = 1.0,
):
    """Equal-nbar convenience wrapper around double-pass moments.

    Default T is idler *intensity* transmission with t=√T, nbar1=nbar2.
    Returned signal intensities are relative to nbar1 so existing tests
    keep I1_rel=ηs1 and I2_rel=ηs2(1+T nbar).
    """
    nbar = float(mean_photons_per_mode)
    if not np.isfinite(nbar) or nbar < 0.0:
        raise ValueError("mean_photons_per_mode must be finite and >= 0")
    n2 = nbar if nbar2 is None else float(nbar2)
    T_c = np.clip(np.asarray(T, dtype=np.float64), 0.0, 1.0)
    if t_seed is None:
        t_use = np.sqrt(T_c)
    else:
        t_use = t_seed
    src = spdc_double_pass_moments(nbar, n2, t_use)
    det = apply_signal_path_losses(
        src["N1_source"],
        src["N2_source"],
        src["C12_source"],
        eta_s1=path1_intensity,
        eta_s2=path2_intensity,
        gamma_path=gamma_path,
    )
    obs = spdc_observables_from_moments(det["N1"], det["N2"], det["C12"])
    p1 = max(float(path1_intensity), 0.0)
    p2 = max(float(path2_intensity), 0.0)
    if nbar > 0.0:
        signal1 = obs["N1"] / nbar
        signal2 = obs["N2"] / nbar
        coherence = obs["C12_abs"] / nbar
        Tseed = src["T_seed"]
        g1 = np.sqrt(
            np.divide(
                Tseed * (1.0 + nbar),
                1.0 + Tseed * nbar,
                out=np.zeros_like(Tseed, dtype=np.float64),
                where=(1.0 + Tseed * nbar) > 0.0,
            )
        ) * float(np.clip(gamma_path, 0.0, 1.0))
        if abs(n2 - nbar) > 1.0e-15:
            g1 = obs["intrinsic_g1"]
    else:
        Tseed = src["T_seed"]
        signal1 = np.full_like(Tseed, p1, dtype=np.float64)
        signal2 = np.full_like(Tseed, p2, dtype=np.float64)
        coherence = np.sqrt(p1 * p2) * np.sqrt(Tseed) * float(
            np.clip(gamma_path, 0.0, 1.0)
        )
        g1 = np.sqrt(Tseed) * float(np.clip(gamma_path, 0.0, 1.0))
    return {
        "signal1_relative": signal1,
        "signal2_relative": signal2,
        "coherence_relative": coherence,
        "intrinsic_g1": g1,
        "visibility_50_50": obs["visibility_50_50"],
        "N1": obs["N1"],
        "N2": obs["N2"],
        "C12": obs["C12"],
        "T_seed": src["T_seed"],
        "t_seed": src["t_seed"],
    }


def _interfere_spdc_gain(
    I1_low_gain,
    I2_low_gain,
    Gamma,
    T,
    mean_photons_per_mode,
    phi,
    theta=0.0,
    delta_phi=0.0,
):
    """Interfere gain-dependent SPDC signal fields at a fixed combiner.

    I2 acquires stimulated intensity (1+T*nbar), while the cross moment is
    proportional to sqrt(T*(1+nbar)). This reduces to _interfere at nbar=0.
    """
    nbar = float(mean_photons_per_mode)
    a = np.clip(np.asarray(I1_low_gain, dtype=np.float32), 0.0, None)
    b0 = np.clip(np.asarray(I2_low_gain, dtype=np.float32), 0.0, None)
    G = np.clip(np.asarray(Gamma, dtype=np.float32), 0.0, 1.0)
    T_c = np.clip(np.asarray(T, dtype=np.float32), 0.0, 1.0)
    b = b0 * (1.0 + T_c * nbar)
    phi_tot = (
        np.asarray(phi, dtype=np.float32)
        + float(theta)
        + np.asarray(delta_phi, dtype=np.float32)
    )
    cross = (
        2.0
        * np.sqrt(a * b0)
        * G
        * np.sqrt(T_c * (1.0 + nbar))
        * np.cos(phi_tot)
    ).astype(np.float32)
    return (
        np.clip(a + b + cross, 0.0, None).astype(np.float32),
        np.clip(a + b - cross, 0.0, None).astype(np.float32),
    )

# ==============================================================================
# LEGACY PRIMITIVES (steps 3) -- optional / regression helpers
# Role: forward mapping (not used for twin clean_D; see simulate_forward)
# Models: 4f map, SPDC PSF, spatial overlap -> V(x,y), Gamma(dL)
# Provenance: upi
# ==============================================================================

def apply_spdc_psf(complex_field: np.ndarray, sigma_spdc: float) -> np.ndarray:
    """Apply SPDC / QIUP correlation PSF to a complex object field (V-P0-01).

    Filters real and imaginary parts with the same Gaussian kernel so amplitude
    and phase objects share the correlation resolution limit.
    """
    field = np.asarray(complex_field)
    sig = float(sigma_spdc)
    if not np.isfinite(sig) or sig <= 0.0:
        return field.astype(np.complex128, copy=True)
    re = gaussian_filter(np.real(field).astype(np.float64), sigma=sig)
    im = gaussian_filter(np.imag(field).astype(np.float64), sigma=sig)
    return (re + 1j * im).astype(np.complex128)


def lemos_optical_mapping(
    T: np.ndarray,
    phi: np.ndarray,
    magnification: float = 1.0,
    spdc_psf_sigma: float = 1.0,
    V_mean: float = 0.77,
    visibility_variation: float = 0.08,
    delta_L: float = 0.01,
    coherence_length: float = 0.20,
    rng: np.random.Generator | None = None,
    unit_pattern: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """4f mapping, SPDC PSF, spatial signal–idler overlap O(x,y) → V(x,y), Γ(ΔL)."""
    rng = rng or np.random.default_rng()
    original_shape = T.shape

    if magnification != 1.0:
        T_mapped = zoom(T, magnification, order=1)
        phi_mapped = zoom(phi, magnification, order=1)
        T_mapped = resize_to_original(T_mapped, original_shape)
        phi_mapped = resize_to_original(phi_mapped, original_shape)
    else:
        T_mapped = T.copy()
        phi_mapped = phi.copy()

    # V-P0-01: blur the complex object √T·exp(iφ), not T and φ separately.
    complex_object = np.sqrt(np.clip(T_mapped, 0.0, None)).astype(np.float64) * np.exp(
        1j * np.asarray(phi_mapped, dtype=np.float64)
    )
    filtered = apply_spdc_psf(complex_object, float(spdc_psf_sigma))
    T_eff = np.clip(np.abs(filtered) ** 2, 0.0, 1.0).astype(np.float32)
    phi_eff = np.angle(filtered).astype(np.float32)

    if visibility_variation <= 0:
        V_map = np.full(T.shape, V_mean, dtype=np.float32)
    else:
        overlap = spatial_overlap_map(
            T.shape, visibility_variation, rng, unit_pattern=unit_pattern,
        )
        V_map = np.clip(V_mean * overlap, 0.0, 1.0).astype(np.float32)

    Gamma = np.exp(-(delta_L / coherence_length) ** 2)
    V_eff = V_map * Gamma

    return {
        "T_mapped": T_mapped.astype(np.float32),
        "phi_mapped": phi_mapped.astype(np.float32),
        "T_eff": T_eff.astype(np.float32),
        "phi_eff": phi_eff.astype(np.float32),
        "V_map": V_map.astype(np.float32),
        "Gamma": np.float32(Gamma),
        "V_eff": V_eff.astype(np.float32),
    }


def simulate_signal_images(
    optical: dict[str, np.ndarray],
    theta: float = np.pi / 4,
    I_max: float = 1.0,
    sigma_beam: float | None = None,
    output_dir: str = "UPI_Synthetic/output",
) -> dict[str, np.ndarray]:
    T_eff = optical["T_eff"]
    phi_eff = optical["phi_eff"]
    V_eff = optical["V_eff"]
    N = T_eff.shape[0]
    if sigma_beam is None:
        sigma_beam = N / 4

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    x0, y0 = N // 2, N // 2
    I0 = I_max * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_beam ** 2))

    I_plus, I_minus = _interfere(
        0.5 * I0, 0.5 * I0, V_eff, T_eff, phi_eff, theta=theta, delta_phi=0.0
    )
    D = (I_plus - I_minus).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    for name, arr in [
        ("T_mapped.npy", optical["T_mapped"]),
        ("phi_mapped.npy", optical["phi_mapped"]),
        ("T_effective.npy", T_eff),
        ("phi_effective.npy", phi_eff),
        ("visibility_map.npy", optical["V_map"]),
        ("effective_visibility.npy", V_eff),
        ("gaussian_envelope.npy", I0),
        ("signal_constructive.npy", I_plus),
        ("signal_destructive.npy", I_minus),
        ("signal_difference.npy", D),
    ]:
        np.save(os.path.join(output_dir, name), arr)

    return {"I0": I0, "I_plus": I_plus, "I_minus": I_minus, "D": D}


# ==============================================================================
# TWIN OPTICS (noisy forward) -- SLM, visibility, PSF, intensity
# Role: twin forward
# Models: calibratable SLM; multiplicative V; diffraction+defocus PSF
# Provenance: arch
# ==============================================================================

def _slm_complex(
    phi: np.ndarray,
    s_slm: float,
    fwd: ResolvedForward | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase → complex amplitude with calibration leakage (C8).

    Layering (S9 / L9):
      - hardware: levels0 / bit depth (from setup when ``fwd`` given)
      - object_function: input ``phi`` from the displayed object
      - calibration_error: driven only by ``s_slm``
    """
    if fwd is not None:
        levels0 = float(fwd.slm_levels0)
        quant_coef = float(fwd.slm_quant_coef)
        levels_floor = int(fwd.slm_levels_floor)
        gamma_span = float(fwd.slm_gamma_span)
        leak_eta = float(fwd.slm_leak_eta)
        phi_max = float(fwd.derived.get("slm_phi_max_rad", 2.0 * np.pi))
    else:
        hw = SLM_TAXONOMY["hardware"]["params"]
        cal = SLM_TAXONOMY["calibration_error"]["params"]
        levels0 = float(hw["levels0"])
        quant_coef = float(cal["quant_coef"])
        levels_floor = int(cal["levels_floor"])
        gamma_span = float(cal["gamma_span"])
        leak_eta = float(cal["leak_eta"])
        phi_max = float(hw["phi_max_rad"])
    levels = max(
        levels_floor,
        int(round(levels0 * (1.0 - quant_coef * s_slm) ** 2)),
    )
    gamma = 1.0 + gamma_span * s_slm
    leak = leak_eta * s_slm
    p = np.clip(phi / phi_max, 0.0, 1.0)
    p = np.power(p, gamma)
    p = np.round(p * levels) / max(levels, 1)
    ph = p * phi_max
    amp = np.sqrt(np.maximum(0.0, 1.0 - leak * np.sin(ph / 2.0) ** 2))
    return (amp * np.cos(ph)).astype(np.float32), (amp * np.sin(ph)).astype(np.float32)


def gaussian_filter_transmission(
    wavelength_nm: np.ndarray | float,
    *,
    center_nm: float,
    fwhm_nm: float,
) -> np.ndarray:
    """Peak-normalized Gaussian intensity transmission."""
    center = float(center_nm)
    fwhm = float(fwhm_nm)
    if not np.isfinite(center) or center <= 0.0:
        raise ValueError("camera_filter_center_nm must be finite and > 0")
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("camera_filter_fwhm_nm must be finite and > 0")
    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("filter wavelengths must be finite")
    return np.exp(
        -4.0 * np.log(2.0) * ((wavelength - center) / fwhm) ** 2
    )


def signal_filter_spectral_response(
    setup: SetupParams | None,
    *,
    opd_mm: float,
    samples: int = 2049,
) -> dict[str, Any]:
    """Reduced-order signal-filter throughput and complex coherence.

    The spatial image is separable from wavelength over the selected narrow
    band. A normalized Gaussian source spectrum is multiplied by the filter
    and integrated. The carrier OPD phase remains owned by ``opd_phase_rad``.
    """
    if setup is None or getattr(setup, "camera_filter_preset", None) is None:
        return {
            "enabled": False,
            "throughput": 1.0,
            "coherence_magnitude": 1.0,
            "coherence_phase_rad": 0.0,
            "model": None,
            "spectral_filter_model": "legacy_not_configured",
        }
    center = float(setup.camera_filter_center_nm)
    fwhm = float(setup.camera_filter_fwhm_nm)
    source_center = float(setup.lambda_s_nm)
    source_fwhm = float(setup.signal_spectrum_fwhm_nm)
    n = max(int(samples), 257)
    sigma_source = source_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_filter = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half_span = 7.0 * max(
        sigma_source, sigma_filter, abs(center - source_center), 0.1
    )
    lo = max(1.0e-6, min(center, source_center) - half_span)
    hi = max(center, source_center) + half_span
    wavelength = np.linspace(lo, hi, n, dtype=np.float64)
    source = np.exp(-0.5 * ((wavelength - source_center) / sigma_source) ** 2)
    model = str(setup.camera_filter_model)
    if model == CAMERA_FILTER_MODEL_GAUSSIAN:
        transmission = gaussian_filter_transmission(
            wavelength, center_nm=center, fwhm_nm=fwhm
        )
    elif model == CAMERA_FILTER_MODEL_MEASURED:
        transmission = np.interp(
            wavelength,
            np.asarray(setup.camera_filter_wavelengths_nm, dtype=np.float64),
            np.asarray(setup.camera_filter_transmissions, dtype=np.float64),
            left=0.0,
            right=0.0,
        )
    else:
        raise ValueError(f"unsupported camera_filter_model={model!r}")
    source_area = float(np.trapz(source, wavelength))
    weighted = source * transmission
    detected_area = float(np.trapz(weighted, wavelength))
    if source_area <= 0.0 or detected_area <= 0.0:
        raise ValueError("signal/filter spectral overlap must be positive")
    throughput = float(np.clip(detected_area / source_area, 0.0, 1.0))
    opd_nm = float(opd_mm) * 1.0e6
    relative_phase = 2.0 * np.pi * opd_nm * (
        1.0 / wavelength - 1.0 / source_center
    )
    coherence = (
        np.trapz(weighted * np.exp(1j * relative_phase), wavelength)
        / detected_area
    )
    return {
        "enabled": True,
        "preset": setup.camera_filter_preset,
        "center_nm": center,
        "fwhm_nm": fwhm,
        "model": model,
        "peak_transmission": float(np.max(transmission)),
        "throughput": throughput,
        "coherence_magnitude": float(np.clip(abs(coherence), 0.0, 1.0)),
        "coherence_phase_rad": float(np.angle(coherence)),
        "opd_mm": float(opd_mm),
        "signal_spectrum_center_nm": source_center,
        "signal_spectrum_fwhm_nm": source_fwhm,
        "spectral_filter_model": "reduced_order_separable_spectral_quadrature",
        "spectral_samples": n,
        "double_counting_control": (
            "replaces legacy L_coh envelope; throughput scales N1,N2,C12 once"
        ),
    }


def _visibility_factors(
    e: dict[str, float],
    fwd: ResolvedForward | None = None,
    *,
    pump_delta_x_px: float = 0.0,
    pump_delta_y_px: float = 0.0,
    setup: SetupParams | None = None,
) -> dict[str, float]:
    """Visibility factors (PDF §4).

    Path imbalance uses I1, I2 and eq. 70:
        V_port = 2 √(I1 I2) / (I1 + I2)
    No ratio r and no r = 1 − 0.45 s_align.

    ``V0`` is a residual calibration scale (default 1 = unused).  ``V_gamma`` is
    the product of the other factors (overlap, coherence, gain) that multiply
    the interference term.  ``V_eff = V_gamma * V_port`` is the reported fringe
    contrast.
    """
    if fwd is not None:
        V0 = float(fwd.V0)
        ov_scale = float(fwd.overlap_delta_scale)
        delta_L = float(fwd.delta_L_eff_mm)
        L_coh = float(fwd.L_coh_mm)
        I1 = float(fwd.path1_intensity)
        I2 = float(fwd.path2_intensity)
        V_gain_drop = float(fwd.V_gain_drop)
    else:
        vd = VISIBILITY_DEFAULTS
        V0 = float(vd["V0"])
        ov_scale = float(vd["overlap_delta_scale"])
        if setup is not None:
            delta_L = float(effective_delta_L_mm(setup))
            L_coh = float(setup.L_coh_mm)
            I1 = float(setup.path1_intensity)
            I2 = float(setup.path2_intensity)
        else:
            delta_L = float(vd["delta_L_mm"])
            L_coh = float(vd["L_coh_mm"])
            I1 = float(vd["path1_intensity"])
            I2 = float(vd["path2_intensity"])
        V_gain_drop = float(vd["V_gain_drop"])
    s_align, s_gain = e["s_align"], e["s_gain"]
    if fwd is not None and bool(fwd.use_analytic_mode_overlap) and setup is not None:
        V_ov = float(
            np.clip(
                eta_mode_from_setup(
                    setup,
                    pump_delta_x_px=pump_delta_x_px,
                    pump_delta_y_px=pump_delta_y_px,
                ),
                0.0,
                1.0,
            )
        )
    elif fwd is not None and bool(fwd.use_analytic_mode_overlap):
        V_ov = float(np.clip(fwd.eta_mode, 0.0, 1.0))
    else:
        d_ov = ov_scale * s_align
        V_ov = float(np.exp(-(d_ov ** 2) / 2.0))
    # PDF §4.1 eq. 65 — ΔL_eff includes thermal stretch when use_physical_thermal.
    filter_response = signal_filter_spectral_response(setup, opd_mm=delta_L)
    V_coh = (
        float(filter_response["coherence_magnitude"])
        if filter_response["enabled"]
        else v_coh_gaussian(delta_L, L_coh)
    )
    V_port = two_beam_visibility(I1, I2)
    use_physical_spdc = bool(
        setup is not None and getattr(setup, "use_physical_spdc_gain", False)
    )
    # Physical gain is applied through N1, N2, C12 (§7.17–7.20), not V_gain.
    V_gain = 1.0 if use_physical_spdc else float(1.0 - V_gain_drop * s_gain)
    # When spatial V(x,y) comes from |E1 E2*| (eq. 74), do not also multiply
    # the scalar V_ov into Γ (avoids double-counting the same mismatch).
    use_field_map = bool(
        fwd is not None and getattr(fwd, "use_field_overlap_visibility_map", False)
    )
    if use_physical_spdc:
        # §7.15 / §7.20 / §7.21: ηmode → tseed, V_coh → γpath on C12,
        # V_port/V0 diagnostic only. Do not multiply them into intensity.
        V_gamma = 1.0
        V_eff = float(np.clip(V_port, 0.0, 1.0))
    elif use_field_map:
        V_gamma = float(np.clip(V0 * V_coh * V_gain, 0.0, 1.0))
        # Audit proxy until the spatial map mean is folded in downstream.
        V_eff = float(np.clip(V_gamma * V_port * V_ov, 0.0, 1.0))
    else:
        V_gamma = float(np.clip(V0 * V_ov * V_coh * V_gain, 0.0, 1.0))
        V_eff = float(np.clip(V_gamma * V_port, 0.0, 1.0))
    return {
        "V_overlap": V_ov,
        "V_coh": V_coh,
        "spectral_filter": filter_response,
        "spectral_filter_throughput": float(filter_response["throughput"]),
        "spectral_phase_offset_rad": float(filter_response["coherence_phase_rad"]),
        "coherence_owner": (
            "signal_filter_spectral_quadrature"
            if filter_response["enabled"] else "legacy_L_coh_gaussian"
        ),
        "V_port": V_port,
        "V_gain": V_gain,
        "spdc_gain_model": (
            "wiseman_molmer_moments" if use_physical_spdc else "legacy_linear_penalty"
        ),
        "visibility_factors_are_diagnostic": bool(use_physical_spdc),
        "gamma_path": V_coh if use_physical_spdc else None,
        "V_gamma": V_gamma,
        "V_eff": V_eff,
        "V0": V0,
        "I1": float(I1),
        "I2": float(I2),
        "delta_L": delta_L,
        "L_coh": L_coh,
        "use_field_overlap_visibility_map": use_field_map and (not use_physical_spdc),
        "eta_mode": V_ov if (
            use_physical_spdc or (fwd is not None and fwd.use_analytic_mode_overlap)
        ) else None,
        "dT_eff_C": float(fwd.dT_eff_C) if fwd is not None else None,
        "use_physical_thermal": bool(fwd.use_physical_thermal) if fwd is not None else (
            bool(setup.use_physical_thermal) if setup is not None else False
        ),
    }


def idler_lens_misalignment_phase_map(
    N: int,
    setup: SetupParams,
    *,
    delta_z_um: float,
) -> dict[str, Any]:
    """Idler-lens phase on the camera grid (L06).

    Far-field axial term (P11): quadratic Haidinger phase with
    ``λ_eq = λ_s²/λ_i`` and ``d = f_u² δ / (f_u² + δ²)``. Single-crystal
    return paths multiply that axial term by ``pass_factor = 2``.

    Far-field single-crystal transverse term: the Michelson round trip
    translates the returning idler image by ``2Δ``, so
    ``Φ_tr = 4π (Δx x + Δy y) / (λ_s f_c)``. ``pass_factor`` is not applied
    again. Two-crystal far field keeps the previous single-pass prism
    ``2π (Δx x + Δy y) / (λ_eq f_c)``.

    Near field: the same lens move, but a position-plane Fresnel phase
    governed by ``λ_i`` and ``M_NF = M_s/M_u``, still multiplied by
    ``pass_factor``. The map is a phase, not an intensity texture.
    """
    n = int(N)
    if n <= 1:
        raise ValueError("phase map size must be > 1")
    pixel_um = float(internal_camera_sampling_pixel_um(setup))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    x_um = (xx - (n - 1) * 0.5) * pixel_um
    y_um = (yy - (n - 1) * 0.5) * pixel_um
    rho2_um2 = x_um * x_um + y_um * y_um
    sx_um = float(setup.lens_shift_x_um)
    sy_um = float(setup.lens_shift_y_um)
    mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    f_u_mm = float(resolved_far_field_focal_mm(setup))
    d_um = idler_lens_equivalent_propagation_um(delta_z_um, f_u_mm)
    lam_s_nm = float(setup.lambda_s_nm)
    lam_i_nm = float(setup.lambda_i_nm)
    lam_eq_nm = equivalent_wavelength_nm(lam_s_nm, lam_i_nm)
    profile = resolve_architecture_profile(setup)
    pass_factor = 1.0 if is_two_crystal_architecture(setup) else 2.0

    if mode == IMAGING_MODE_FAR:
        f_c_um = float(setup.F2_mm) * 1.0e3
        lam_eq_um = lam_eq_nm * 1.0e-3
        lam_s_um = lam_s_nm * 1.0e-3
        if f_c_um <= 0.0 or lam_eq_um <= 0.0 or lam_s_um <= 0.0:
            raise ValueError("far-field idler phase requires f_c > 0, λ_eq > 0, and λ_s > 0")
        phi_axial = (np.pi * d_um * rho2_um2) / (f_c_um * f_c_um * lam_eq_um)
        ramp = sx_um * x_um + sy_um * y_um
        if is_two_crystal_architecture(setup):
            phi_transverse = 2.0 * np.pi * ramp / (lam_eq_um * f_c_um)
            phi = pass_factor * (phi_axial + phi_transverse)
            law = "P11_lambda_eq"
            governing_nm = lam_eq_nm
            transverse_model = "two_crystal_single_pass_lambda_eq"
            transverse_pass_factor_applied = True
        else:
            # 4π already contains the round-trip image shift 2Δ. Do not multiply by pass_factor.
            phi_transverse = 4.0 * np.pi * ramp / (lam_s_um * f_c_um)
            phi = pass_factor * phi_axial + phi_transverse
            law = "single_crystal_far_field_image_shift_lambda_s"
            governing_nm = lam_s_nm
            transverse_model = "4*pi*(dx*x+dy*y)/(lambda_s*f_c)"
            transverse_pass_factor_applied = False
    else:
        m_u = float(setup.near_field_idler_magnification)
        m_s = float(setup.near_field_signal_magnification)
        if abs(m_u) <= np.finfo(np.float64).eps:
            raise ValueError("near-field idler magnification must be nonzero")
        m_nf = m_s / m_u
        f_u_um = f_u_mm * 1.0e3
        lam_i_um = lam_i_nm * 1.0e-3
        if f_u_um <= 0.0 or lam_i_um <= 0.0:
            raise ValueError("near-field idler phase requires f_u > 0 and λ_i > 0")
        phi = (np.pi * d_um * rho2_um2) / (
            lam_i_um * f_u_um * f_u_um * m_nf * m_nf
        )
        phi = phi + (
            2.0 * np.pi * (sx_um * x_um + sy_um * y_um)
            / (lam_i_um * f_u_um * m_nf)
        )
        law = "near_field_object_relay_lambda_i"
        governing_nm = lam_i_nm
        transverse_model = "near_field_lambda_i_with_pass_factor"
        transverse_pass_factor_applied = True
        phi = pass_factor * phi
    return {
        "phase_rad": phi.astype(np.float64),
        "enabled": True,
        "applied_to": "idler_seed_phase_before_C12",
        "applied_to_intensity": False,
        "imaging_mode": mode,
        "phase_law": law,
        "transverse_phase_model": transverse_model,
        "pass_factor_multiplies_transverse": transverse_pass_factor_applied,
        "pass_factor": float(pass_factor),
        "delta_z_um": float(delta_z_um),
        "equivalent_propagation_um": float(d_um),
        "idler_lens_focal_mm": float(f_u_mm),
        "lambda_s_nm": lam_s_nm,
        "lambda_i_nm": lam_i_nm,
        "lambda_equivalent_nm": float(lam_eq_nm),
        "governing_wavelength_nm": float(governing_nm),
        "lens_shift_x_um": sx_um,
        "lens_shift_y_um": sy_um,
    }


def _lens_misalignment_ring_map(
    N: int,
    setup: SetupParams,
    *,
    delta_z_um: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagnostic cos(φ_I) of the physical idler-lens phase (L06).

    Kept so older callers still receive a 2-D map. The forward path applies
    ``phase_rad`` to the idler seed, not this cosine to I±.
    """
    state = idler_lens_misalignment_phase_map(N, setup, delta_z_um=delta_z_um)
    diagnostic = np.cos(state["phase_rad"]).astype(np.float32)
    meta = dict(state)
    meta.pop("phase_rad", None)
    meta["n_rings"] = float(
        np.ptp(state["phase_rad"]) / (2.0 * np.pi)
    )
    return diagnostic, meta


def build_etalon_fringe_map(
    shape: tuple[int, int],
    *,
    contrast: float,
    period_px: float,
    angle_deg: float,
    phase_rad: float,
) -> np.ndarray:
    """Multiplicative Fabry–Pérot / etalon fringe map (intensity level).

    Fringe *bands* make angle ``angle_deg`` with the horizontal axis
    (0° → horizontal bands; 90° → vertical bands; lab often ~10–20°).

    map = 1 + c · cos(2π u / Λ + φ),  u = x sinα − y cosα
    """
    H, W = int(shape[0]), int(shape[1])
    c = float(contrast)
    period = max(float(period_px), 1e-6)
    alpha = np.deg2rad(float(angle_deg))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # Band orientation α to horizontal → phase varies along the band normal.
    u = xx * np.sin(alpha) - yy * np.cos(alpha)
    return (1.0 + c * np.cos(2.0 * np.pi * u / period + float(phase_rad))).astype(
        np.float32
    )


def apply_etalon_fringes(
    I_plus: np.ndarray,
    I_minus: np.ndarray,
    setup: SetupParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the same etalon map to both ports (common detected-path cavities)."""
    enabled = bool(getattr(setup, "enable_etalon_fringes", False))
    meta: dict[str, Any] = {
        "enabled": enabled,
        "model": "multiplicative_cosine_intensity",
        "full_wave_optics": False,
        "contrast": float(getattr(setup, "etalon_contrast", 0.0)),
        "period_px": float(getattr(setup, "etalon_period_px", 34.0)),
        "angle_deg": float(getattr(setup, "etalon_angle_deg", -40.0)),
        "phase_rad": float(getattr(setup, "etalon_phase_rad", 0.0)),
        "angle_convention": (
            "fringe bands relative to horizontal "
            "(0=horizontal, 90=vertical; lab typically tilted ~10-20 deg)"
        ),
    }
    if not enabled:
        return I_plus, I_minus, meta
    fringe = build_etalon_fringe_map(
        I_plus.shape,
        contrast=meta["contrast"],
        period_px=meta["period_px"],
        angle_deg=meta["angle_deg"],
        phase_rad=meta["phase_rad"],
    )
    meta["fringe_min"] = float(fringe.min())
    meta["fringe_max"] = float(fringe.max())
    return (
        (I_plus * fringe).astype(np.float32),
        (I_minus * fringe).astype(np.float32),
        meta,
    )


def apply_object_coordinate_transform(field, transform: str) -> np.ndarray:
    """Map a complex object field between declared optical coordinates."""
    key = str(transform).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in OBJECT_TRANSFORMS:
        raise ValueError(
            f"object transform must be one of {sorted(OBJECT_TRANSFORMS)}, got {transform!r}"
        )
    value = np.asarray(field, dtype=np.complex128)
    if value.ndim != 2:
        raise ValueError("object field must be a two-dimensional array")
    if key == "identity":
        return value.copy()
    if key == "flip_x":
        return np.flip(value, axis=1).copy()
    if key == "flip_y":
        return np.flip(value, axis=0).copy()
    return np.flip(value, axis=(0, 1)).copy()


def propagate_correlated_object_to_camera(
    field,
    setup: SetupParams,
    *,
    sigma_camera_px: float,
    apply_correlation_psf: bool,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the reduced-order conditional spatial-correlation integral.

    The zero-width part evaluates the object field at the physical coordinate
    selected by the position or momentum correlation. A normalized Gaussian
    convolution then represents the finite conditional correlation width.
    This SPDC kernel is independent of the ordinary camera optical PSF.
    """
    value = np.asarray(field, dtype=np.complex128)
    if value.ndim != 2:
        raise ValueError("correlated object field must be a two-dimensional array")
    if not np.all(np.isfinite(value)):
        raise ValueError("correlated object field must be finite")
    mapping = (
        dict(mapping)
        if mapping is not None
        else resolved_spatial_mapping(setup)
    )
    scale = float(mapping["object_to_camera_index_scale"])
    if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float64).eps:
        raise ValueError("object-to-camera index scale must be finite and nonzero")

    inverse_scale = 1.0 / scale
    matrix = np.diag([inverse_scale, inverse_scale])
    center = (np.asarray(value.shape, dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center

    def _resample(component: np.ndarray) -> np.ndarray:
        return affine_transform(
            component,
            matrix=matrix,
            offset=offset,
            output_shape=value.shape,
            order=1,
            mode="nearest",
            prefilter=False,
        )

    mapped = _resample(value.real) + 1j * _resample(value.imag)
    sigma = float(sigma_camera_px)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma_camera_px must be finite and nonnegative")
    filtered = (
        apply_spdc_psf(mapped, sigma)
        if apply_correlation_psf and sigma > 0.0
        else mapped
    )
    return {
        "field_before_correlation_psf": mapped,
        "field": filtered,
        "mapping": mapping,
        "spdc_psf_applied": bool(apply_correlation_psf and sigma > 0.0),
        "sigma_spdc_applied_px": sigma if apply_correlation_psf else 0.0,
        "integral_model": (
            "C12(rho_c) proportional to integral K(rho_c|rho_o) "
            "t_eff(rho_o) d2rho_o"
        ),
    }


def _resample_complex_affine(
    value: np.ndarray,
    *,
    scale: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Read input at center + scale*(out-center)."""
    field = np.asarray(value, dtype=np.complex128)
    if field.ndim != 2:
        raise ValueError("affine resample requires a 2-D field")
    k = float(scale)
    if not np.isfinite(k):
        raise ValueError("affine scale must be finite")
    if center is None:
        center = (np.asarray(field.shape, dtype=np.float64) - 1.0) / 2.0
    else:
        center = np.asarray(center, dtype=np.float64)
    if abs(k) <= np.finfo(np.float64).eps:
        iy = int(np.clip(round(center[0]), 0, field.shape[0] - 1))
        ix = int(np.clip(round(center[1]), 0, field.shape[1] - 1))
        return np.full(field.shape, field[iy, ix], dtype=np.complex128)
    matrix = np.diag([k, k])
    offset = center - matrix @ center

    def _one(component: np.ndarray) -> np.ndarray:
        return affine_transform(
            component,
            matrix=matrix,
            offset=offset,
            output_shape=field.shape,
            order=1,
            mode="nearest",
            prefilter=False,
        )

    return _one(field.real) + 1j * _one(field.imag)


def apply_near_field_joint_gaussian(
    field,
    setup: SetupParams,
    *,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """P10 Eqs. 7–9: joint Gaussian image G and visibility V.

    ``t_V = G / ∫P`` is the conditional object response used for C12.
    ``w = (∫P) / max(∫P)`` is the detected-beam marginal that scales I±
    so G = Imax−Imin follows Δ_G while V follows Δ_V. A unit object keeps
    t_V = 1 (L02). This is the paper Gaussian, not a sinc Φ.
    """
    value = np.asarray(field, dtype=np.complex128)
    if value.ndim != 2:
        raise ValueError("near-field joint field must be two-dimensional")
    if not np.all(np.isfinite(value)):
        raise ValueError("near-field joint field must be finite")
    mapping = (
        dict(mapping)
        if mapping is not None
        else resolved_spatial_mapping(setup)
    )
    mapped_state = propagate_correlated_object_to_camera(
        value,
        setup,
        sigma_camera_px=0.0,
        apply_correlation_psf=False,
        mapping=mapping,
    )
    mapped = np.asarray(mapped_state["field"], dtype=np.complex128)
    coeff = near_field_joint_gaussian_coefficients(
        setup,
        lambda_s_nm=mapping.get("lambda_s_nm"),
        lambda_i_nm=mapping.get("lambda_i_nm"),
    )
    A = float(coeff["A_per_um2"])
    B = float(coeff["B_per_um2"])
    lam_d = float(coeff["lambda_d_um"])
    lam_u = float(coeff["lambda_u_um"])
    alpha = A * lam_d * lam_d + B
    beta = A * lam_u * lam_u + B
    gamma = A * lam_d * lam_u - B
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("near-field joint Gaussian α must be finite and > 0")
    k = float(-gamma / alpha)
    env_coeff = float(beta - (gamma * gamma) / alpha)
    p_camera = float(mapping["camera_pixel_um"])
    m_s = float(mapping["near_field_signal_magnification_Ms"])
    if abs(m_s) <= np.finfo(np.float64).eps:
        raise ValueError("near-field signal magnification must be nonzero")
    p_xtal = p_camera / abs(m_s)
    sigma_um = 1.0 / np.sqrt(2.0 * alpha)
    sigma_px = float(sigma_um / p_xtal)

    height, width = mapped.shape
    x0 = float(getattr(setup, "pump_center_x_um", 0.0) or 0.0)
    y0 = float(getattr(setup, "pump_center_y_um", 0.0) or 0.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    yd = (yy - (height - 1) * 0.5) * p_xtal - y0
    xd = (xx - (width - 1) * 0.5) * p_xtal - x0
    r2 = xd * xd + yd * yd
    envelope = np.exp(-env_coeff * r2)
    pump_center_px = np.array(
        [
            (height - 1) * 0.5 + y0 / p_xtal,
            (width - 1) * 0.5 + x0 / p_xtal,
        ],
        dtype=np.float64,
    )

    def _joint_unnormalized(src: np.ndarray) -> np.ndarray:
        blurred = apply_spdc_psf(src, sigma_px)
        sampled = _resample_complex_affine(
            blurred, scale=k, center=pump_center_px
        )
        return sampled * envelope

    num = _joint_unnormalized(mapped)
    den = np.clip(np.real(_joint_unnormalized(np.ones_like(mapped))), 0.0, None)
    peak = float(np.max(den)) if den.size else 0.0
    t_v = np.zeros_like(num)
    weight = np.zeros_like(den)
    if peak > 0.0 and np.isfinite(peak):
        live = den > (1.0e-15 * peak)
        t_v = np.divide(num, den, out=np.zeros_like(num), where=live)
        weight = den / peak
    return {
        "field": t_v,
        "image_function": num,
        "marginal_weight": weight,
        "marginal": den,
        "field_before_correlation_psf": mapped,
        "mapping": mapping,
        "spdc_psf_applied": True,
        "sigma_spdc_applied_px": sigma_px,
        "conditional_center_scale": k,
        "joint_gaussian_coefficients": coeff,
        "image_1e_width_um": float(near_field_gain_1e_um(setup)),
        "visibility_1e_width_um": float(near_field_visibility_1e_um(setup)),
        "integral_model": (
            "G=∫ P10(ρd,ρu) t(ρo) dρo; "
            "t_V=G/∫P; I± *= (∫P)/max(∫P)"
        ),
        "sinc_replaced_by_gaussian": True,
        "biphoton_wavefunction": False,
    }


_FAR_FIELD_SIGNAL_MARGINAL_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def far_field_signal_marginal_from_spdc(
    setup: SetupParams,
    shape: tuple[int, int],
    *,
    radial_samples: int | None = None,
    pump_radial_samples: int = 72,
    angular_samples: int = 96,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Physical far-field signal marginal on the detector grid.

    The collinear, paraxial biphoton intensity used here is

    ``|Phi(q_s,q_i)|^2 = exp[-w_p^2 |q_s+q_i|^2/2] sinc^2(L Delta-k/2)``.

    ``Delta-k`` reuses the configured QPM residual, ppKTP refractive indices,
    wavelengths, and crystal length.  The idler integral is evaluated after
    the change of variable ``q_p=q_s+q_i``; this keeps the narrow pump angular
    spectrum well sampled even when ``|q_s|`` is large.  Rotational symmetry
    reduces the result to a one-dimensional physical radial marginal, which
    is then sampled at every detector pixel's momentum coordinate.  This is
    detector-grid sampling, not image resizing or fitted magnification.
    """
    mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    if mode != IMAGING_MODE_FAR:
        raise ValueError("far_field_signal_marginal_from_spdc requires far_field mode")
    if len(shape) != 2 or int(shape[0]) < 2 or int(shape[1]) < 2:
        raise ValueError(f"camera shape must be a two-dimensional grid, got {shape!r}")
    h, w = int(shape[0]), int(shape[1])
    n_r = int(radial_samples or max(512, max(h, w)))
    n_p = int(pump_radial_samples)
    n_theta = int(angular_samples)
    if n_r < 64 or n_p < 16 or n_theta < 24:
        raise ValueError("signal-marginal quadrature is undersampled")

    waist_um = resolved_pump_waist_um(setup, interaction=1)
    if waist_um is None or not np.isfinite(waist_um) or float(waist_um) <= 0.0:
        raise ValueError("far-field signal marginal requires a finite positive pump waist")
    waist_m = float(waist_um) * 1.0e-6
    crystal = spdc_crystal_state(setup, interaction=1)
    length_m = float(crystal["crystal_length_m"])
    lambda_p_m = float(setup.lambda_p_nm) * 1.0e-9
    lambda_s_m = float(setup.lambda_s_nm) * 1.0e-9
    lambda_i_m = float(setup.lambda_i_nm) * 1.0e-9
    k_p = 2.0 * np.pi * float(crystal["n_pump"]) / lambda_p_m
    k_s = 2.0 * np.pi * float(crystal["n_signal"]) / lambda_s_m
    k_i = 2.0 * np.pi * float(crystal["n_idler"]) / lambda_i_m
    k_s_air = 2.0 * np.pi / lambda_s_m
    delta_k0 = float(delta_k_qpm_per_m_from_setup(setup, interaction=1))
    mapping = resolved_spatial_mapping(setup)
    f_s_m = float(mapping["far_field_signal_focal_mm"]) * 1.0e-3
    m_s = abs(float(mapping["far_field_signal_relay_magnification"]))
    pixel_m = float(internal_camera_sampling_pixel_um(setup)) * 1.0e-6
    if f_s_m <= 0.0 or m_s <= 0.0 or pixel_m <= 0.0:
        raise ValueError("far-field camera mapping requires positive f_s, relay scale, and pitch")

    cx = 0.5 * (w - 1) + float(getattr(setup, "camera_center_x_px", 0.0) or 0.0)
    cy = 0.5 * (h - 1) + float(getattr(setup, "camera_center_y_px", 0.0) or 0.0)
    corner_radius_px = max(
        np.hypot(x - cx, y - cy)
        for x in (0.0, float(w - 1))
        for y in (0.0, float(h - 1))
    )
    q_per_pixel = k_s_air * pixel_m / (m_s * f_s_m)
    q_s_max = float(corner_radius_px * q_per_pixel)
    cache_key = (
        h, w, n_r, n_p, n_theta,
        float(internal_camera_sampling_pixel_um(setup)), float(waist_um), length_m,
        lambda_p_m, lambda_s_m, lambda_i_m,
        float(crystal["n_pump"]), float(crystal["n_signal"]),
        float(crystal["n_idler"]), delta_k0, f_s_m, m_s, cx, cy,
    )
    if use_cache and cache_key in _FAR_FIELD_SIGNAL_MARGINAL_CACHE:
        cached = _FAR_FIELD_SIGNAL_MARGINAL_CACHE[cache_key]
        out = dict(cached)
        out["metadata"] = dict(cached["metadata"])
        out["metadata"]["cache_hit"] = True
        out["metadata"]["cache_load_time_s"] = 0.0
        return out

    started = time.perf_counter()
    q_s_radial = np.linspace(0.0, q_s_max, n_r, dtype=np.float64)
    # Six inverse-waist units leave exp[-(w_p q_p)^2/2] below 2e-8.
    q_p_radial = np.linspace(0.0, 6.0 / waist_m, n_p, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)
    cos_theta = np.cos(theta)[None, None, :]
    qp = q_p_radial[None, :, None]
    qp2 = qp * qp
    pump_angular = np.exp(-0.5 * (waist_m * q_p_radial) ** 2)
    pump_3d = pump_angular[None, :, None]
    profile = np.empty_like(q_s_radial)
    batch = 64
    for start in range(0, n_r, batch):
        qs = q_s_radial[start : start + batch, None, None]
        qs2 = qs * qs
        # q_i=q_p-q_s, with q_s chosen along +x by rotational symmetry.
        qi2 = qp2 + qs2 - 2.0 * qp * qs * cos_theta
        delta_k = (
            delta_k0
            - qp2 / (2.0 * k_p)
            + qs2 / (2.0 * k_s)
            + qi2 / (2.0 * k_i)
        )
        phase_matching = np.sinc((0.5 * length_m * delta_k) / np.pi) ** 2
        angular_integral = 2.0 * np.pi * np.mean(
            pump_3d * phase_matching, axis=2
        )
        profile[start : start + batch] = np.trapz(
            angular_integral * q_p_radial[None, :], q_p_radial, axis=1
        )
    peak = float(np.max(profile))
    if not np.isfinite(peak) or peak <= 0.0 or not np.all(np.isfinite(profile)):
        raise ValueError("physical far-field signal marginal is non-finite or zero")
    profile = np.clip(profile / peak, 0.0, None)

    yy, xx = np.indices((h, w), dtype=np.float64)
    q_camera = np.hypot(xx - cx, yy - cy) * q_per_pixel
    camera = np.interp(q_camera.ravel(), q_s_radial, profile).reshape(h, w)
    camera = np.clip(camera, 0.0, None)
    camera /= float(np.max(camera))

    anti_correlated_delta_k = (
        delta_k0
        + q_s_radial ** 2 / (2.0 * k_s)
        + q_s_radial ** 2 / (2.0 * k_i)
    )
    phase_matching_slice = np.sinc(
        (0.5 * length_m * anti_correlated_delta_k) / np.pi
    ) ** 2
    elapsed = float(time.perf_counter() - started)
    key_hash = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    metadata = {
        "model": "physical_spdc_signal_marginal_v1",
        "equation": (
            "P_s(q_s)=integral exp[-w_p^2|q_s+q_i|^2/2] "
            "sinc^2[L*Delta_k(q_s,q_i)/2] d^2q_i"
        ),
        "pump_angular_spectrum_included": True,
        "pump_waist_active_in_sigma_q": True,
        "phase_matching_included": True,
        "idler_marginalization_implemented": True,
        "far_field_camera_mapping": "q_s=k_s_air*rho_camera/(M_s*f_s)",
        "legacy_final_gaussian": False,
        "normalization": "peak_one_after_idler_marginalization",
        "integration_method": (
            "rotationally_symmetric radial-angular quadrature in q_p=q_s+q_i; "
            "linear physical sampling onto detector radii"
        ),
        "shape": [h, w],
        "pixel_pitch_um": float(internal_camera_sampling_pixel_um(setup)),
        "pump_waist_um": float(waist_um),
        "crystal_length_mm": 1.0e3 * length_m,
        "lambda_p_nm": float(setup.lambda_p_nm),
        "lambda_s_nm": float(setup.lambda_s_nm),
        "lambda_i_nm": float(setup.lambda_i_nm),
        "n_pump": float(crystal["n_pump"]),
        "n_signal": float(crystal["n_signal"]),
        "n_idler": float(crystal["n_idler"]),
        "delta_k0_per_m": delta_k0,
        "signal_focal_length_mm": 1.0e3 * f_s_m,
        "signal_relay_magnification": m_s,
        "q_per_pixel_per_m": float(q_per_pixel),
        "q_s_max_per_m": q_s_max,
        "pump_q_max_per_m": float(q_p_radial[-1]),
        "radial_samples": n_r,
        "pump_radial_samples": n_p,
        "angular_samples": n_theta,
        "computation_time_s": elapsed,
        "estimated_working_memory_bytes": int(batch * n_p * n_theta * 8 * 3),
        "cache": "process_memory_by_physical_configuration",
        "cache_key_sha256": key_hash,
        "cache_hit": False,
        "cache_load_time_s": None,
        "full_camera_direct_sampling": True,
        "resize_used": False,
    }
    result = {
        "intensity": camera.astype(np.float64),
        "q_s_radial_per_m": q_s_radial,
        "signal_marginal_radial": profile,
        "pump_q_radial_per_m": q_p_radial,
        "pump_angular_intensity": pump_angular,
        "phase_matching_slice": phase_matching_slice,
        "metadata": metadata,
    }
    if use_cache:
        _FAR_FIELD_SIGNAL_MARGINAL_CACHE[cache_key] = result
    return result


def _physical_far_field_signal_marginal_enabled(setup: SetupParams) -> bool:
    """Scope the physical marginal to audited C_lab far-field profiles."""
    sid = str(getattr(setup, "setup_id", ""))
    return (
        normalize_imaging_mode(getattr(setup, "imaging_mode", None))
        == IMAGING_MODE_FAR
        and sid
        in {
            "config_c_lab_v1",
            "config_c_lab_joint_spdc_rabbit70_v1",
        }
        and getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
    )


def camera_generation_intensity(
    setup: SetupParams,
    shape: tuple[int, int],
    *,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Peak-normalized pair-generation intensity on the camera grid.

    Near field maps I_p(ρ) from the crystal/object plane (pairs are born
    where the pump is).  For the audited C_lab far-field profile, the pump
    angular spectrum and sinc-squared phase matching form a single-photon
    signal marginal after integrating over idler momentum.  This envelope
    multiplies physical signal moments, never t_object.
    """
    weight = pump_generation_amplitude_weight_from_setup(setup, shape)
    intensity = np.asarray(weight["intensity_shape_peak_one"], dtype=np.float64)
    mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    marginal: dict[str, Any] | None = None
    if (
        mode == IMAGING_MODE_NEAR
        and getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        mapped = propagate_correlated_object_to_camera(
            intensity.astype(np.complex128),
            setup,
            sigma_camera_px=0.0,
            apply_correlation_psf=False,
            mapping=mapping,
        )
        camera = np.clip(np.real(mapped["field"]), 0.0, None)
        peak = float(np.max(camera))
        if peak > 0.0 and np.isfinite(peak):
            camera = camera / peak
        applied_to_intensity = True
        equation = (
            "S_cam=map_NF(I_p/max); I_pm *= S_cam; t_object unchanged"
        )
    elif mode == IMAGING_MODE_NEAR:
        camera = intensity
        applied_to_intensity = True
        equation = (
            "S_cam=I_p/max on legacy unit-magnification NF grid; "
            "I_pm *= S_cam; t_object unchanged"
        )
    elif _physical_far_field_signal_marginal_enabled(setup):
        marginal = far_field_signal_marginal_from_spdc(setup, shape)
        camera = np.asarray(marginal["intensity"], dtype=np.float64)
        applied_to_intensity = True
        equation = str(marginal["metadata"]["equation"])
    else:
        camera = np.ones(shape, dtype=np.float64)
        applied_to_intensity = False
        equation = (
            "S_cam=1 (far-field I_p(ρ) is not an object-plane mask; "
            "pump enters σ_q=1/w_p)"
        )
    meta = {
        "applied": True,
        "applied_to_object_amplitude": False,
        "applied_to_detected_intensity": applied_to_intensity,
        "profile_type": weight["profile_type"],
        "waist_um": weight["waist_um"],
        "center_x_um": weight["center_x_um"],
        "center_y_um": weight["center_y_um"],
        "interaction": weight["interaction"],
        "equation": equation,
        "max_amplitude": weight["max_amplitude"],
        "peak_normalized": True,
        "total_power_owner": weight["total_power_owner"],
        "spatial_distribution_owner": weight["spatial_distribution_owner"],
        "not_reapplied_as": "idler_object_amplitude_or_t_seed",
        "architecture_profile": weight["architecture_profile"],
        "model_status": "generation_envelope_not_object_loss",
        "imaging_mode": mode,
        "signal_marginal_model": (
            None if marginal is None else marginal["metadata"]
        ),
    }
    return {
        "intensity": camera.astype(np.float64),
        "object_plane_intensity": intensity,
        "metadata": meta,
    }


def architecture_object_transfer(
    T: np.ndarray,
    phi: np.ndarray,
    setup: SetupParams,
    fwd: ResolvedForward,
) -> dict[str, Any]:
    """Build the profile-specific complex idler object/return field.

    A uses one transmitted field.  B/C multiply separately mapped outbound and
    return transmissions.  A reflective B/C object supplies one reflection
    field and replaces the end mirror.  The result is mapped from the declared
    object plane to the correlated signal coordinates, then passed through the
    existing reduced-order complex correlation PSF.
    """
    if getattr(setup, "architecture_propagation_model", None) != (
        ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        raise ValueError("architecture_object_transfer requires the opt-in propagation model")
    profile = validate_forward_profile_supported(setup)
    resolved_object = resolve_object_intensity_and_phase(T, phi, setup)
    T_in = np.asarray(resolved_object["T"], dtype=np.float64)
    phi_in = np.asarray(resolved_object["phi"], dtype=np.float64)
    if T_in.ndim != 2 or phi_in.shape != T_in.shape:
        raise ValueError("object transmission/reflectance and phase maps must be matching 2-D arrays")
    if not np.all(np.isfinite(T_in)) or np.any(T_in < 0.0) or np.any(T_in > 1.0):
        raise ValueError("object intensity transmission/reflectance map must be finite and in [0, 1]")
    if not np.all(np.isfinite(phi_in)):
        raise ValueError("object phase map must be finite")

    object_type = str(setup.object_type)
    raw_field = np.sqrt(T_in) * np.exp(1j * phi_in)
    forward_transform = str(setup.object_forward_transform)
    backward_transform = str(setup.object_backward_transform)
    backward_field: np.ndarray | None = None

    if is_two_crystal_architecture(setup):
        # t_eff=t_forward: one object interaction between NL1 and NL2.
        forward_field = apply_object_coordinate_transform(raw_field, forward_transform)
        effective_object_field = forward_field
        equation = "t_eff=t_forward"
        mirror_model = "not_applicable"
    elif object_type == OBJECT_TYPE_TRANSMISSIVE:
        # The maps are sampled independently so a reversed return is not a
        # blind square of the outbound image.
        forward_field = apply_object_coordinate_transform(raw_field, forward_transform)
        backward_field = apply_object_coordinate_transform(raw_field, backward_transform)
        effective_object_field = forward_field * backward_field
        equation = "t_eff=t_forward*t_backward"
        mirror_model = "ideal_end_mirror_external_losses_separate"
    elif object_type == OBJECT_TYPE_REFLECTIVE:
        reflection = (
            float(setup.object_reflection_amplitude)
            * np.sqrt(T_in)
            * np.exp(1j * (phi_in + float(setup.object_reflection_phase)))
        )
        forward_field = apply_object_coordinate_transform(reflection, forward_transform)
        # A reflection is one physical interaction.  The backward transform
        # only maps that reflected field into return coordinates.
        backward_field = apply_object_coordinate_transform(
            forward_field, backward_transform
        )
        effective_object_field = backward_field
        equation = "r_eff=backward_map(forward_map(r_obj))"
        mirror_model = "reflective_object_replaces_end_mirror"
    else:  # validation should make this unreachable
        raise ValueError(f"unsupported object_type={object_type!r}")

    # L02: pump shape is generation, not object / idler attenuation.
    # Conditional kernel acts on t_o only so a unit object stays unit.
    generation = camera_generation_intensity(
        setup,
        effective_object_field.shape,
        mapping=getattr(fwd, "derived", {}).get("spatial_mapping"),
    )
    pump_weight_meta = generation["metadata"]

    blur_model = normalize_blur_model(
        getattr(fwd, "blur_model", None) or getattr(setup, "blur_model", None)
    )
    sig_spdc = float(fwd.sigma_spdc_px)
    apply_spdc = blur_model == BLUR_MODEL_DOUBLE and sig_spdc > 0.0
    mapping = getattr(fwd, "derived", {}).get("spatial_mapping")
    use_joint_nf = (
        apply_spdc
        and normalize_imaging_mode(profile.imaging_mode) == IMAGING_MODE_NEAR
    )
    if use_joint_nf:
        spatial = apply_near_field_joint_gaussian(
            effective_object_field,
            setup,
            mapping=mapping,
        )
        generation = {
            "intensity": np.asarray(spatial["marginal_weight"], dtype=np.float64),
            "object_plane_intensity": np.asarray(
                spatial["marginal_weight"], dtype=np.float64
            ),
            "metadata": {
                **pump_weight_meta,
                "applied_to_object_amplitude": False,
                "applied_to_intensity": True,
                "equation": spatial["integral_model"],
                "source": "P10_Eq9_marginal",
            },
        }
        pump_weight_meta = generation["metadata"]
    else:
        spatial = propagate_correlated_object_to_camera(
            effective_object_field,
            setup,
            sigma_camera_px=sig_spdc,
            apply_correlation_psf=apply_spdc,
            mapping=mapping,
        )
    correlated_field = spatial["field_before_correlation_psf"]
    filtered = spatial["field"]
    T_out = np.clip(np.abs(filtered) ** 2, 0.0, None).astype(np.float32)
    phi_out = np.angle(filtered).astype(np.float32)
    coordinate_mapping = spatial["mapping"]
    signal_coordinate_transform = (
        "positive_scaled_position_correlation"
        if coordinate_mapping["orientation_sign"] > 0
        else "negative_scaled_momentum_anticorrelation"
    )
    metadata = {
        "imaging_mode": profile.imaging_mode,
        "architecture_profile": profile.to_dict(),
        "propagation_model": ARCHITECTURE_PROPAGATION_COMPLEX_V1,
        "model_status": (
            "p10_joint_gaussian_near_field"
            if use_joint_nf
            else "reduced_order_conditional_gaussian_correlation"
        ),
        "biphoton_wavefunction": False,
        "image_1e_width_um": spatial.get("image_1e_width_um"),
        "visibility_1e_width_um": spatial.get("visibility_1e_width_um"),
        "sinc_replaced_by_gaussian": bool(
            spatial.get("sinc_replaced_by_gaussian", True)
        ),
        "object_type": object_type,
        "object_device": getattr(setup, "object_device", None),
        "object_mode": getattr(setup, "object_mode", None),
        "object_mask_id": getattr(setup, "object_mask_id", None),
        "object_intensity_coefficient": resolved_object["intensity_coefficient"],
        "object_intensity_source": resolved_object["source"],
        "object_plane": setup.object_plane,
        "object_transmission_map": setup.object_transmission_map,
        "object_phase_map": setup.object_phase_map,
        "object_forward_transform": forward_transform,
        "object_backward_transform": backward_transform,
        "object_interaction_count": int(setup.object_interaction_count),
        "object_phase_map_path": getattr(setup, "object_phase_map_path", None),
        "object_amplitude_map_path": getattr(setup, "object_amplitude_map_path", None),
        "object_intensity_map_path": getattr(setup, "object_intensity_map_path", None),
        "object_phase_units": getattr(setup, "object_phase_units", None),
        "object_intensity_convention": getattr(
            setup, "object_intensity_convention", None
        ),
        "object_coordinate_orientation": getattr(
            setup, "object_coordinate_orientation", None
        ),
        "object_transfer_equation": equation,
        "end_mirror_model": mirror_model,
        "signal_coordinate_transform": signal_coordinate_transform,
        "coordinate_mapping": coordinate_mapping,
        "spatial_integral": spatial["integral_model"],
        "correlation_type": imaging_correlation_type(profile.imaging_mode),
        "plane": "image_plane" if profile.imaging_mode == IMAGING_MODE_NEAR else "fourier_plane",
        "blur_model": blur_model,
        "sigma_spdc_px": sig_spdc,
        "sigma_spdc_applied_px": spatial["sigma_spdc_applied_px"],
        "spdc_psf_applied": spatial["spdc_psf_applied"],
        "pump_generation_weight": pump_weight_meta,
        "model": "architecture_specific_complex_field_transfer",
        "detected_field_content": "signal_photons_only",
        "note": (
            "Complex outbound/return object fields are explicit. Near-field "
            "uses the P10 Eq. 9 joint Gaussian (not sinc Φ): t_V is the "
            "conditional object response and the P-marginal scales I± so "
            "G=Imax−Imin and V have distinct widths. Far-field remains a "
            "conditional momentum Gaussian. Pump power is separate from t_seed."
        ),
    }
    return {
        "T_eff": T_out,
        "phi_eff": phi_out,
        "object_field": raw_field,
        "forward_object_field": forward_field,
        "backward_object_field": backward_field,
        "effective_object_field_before_correlation": effective_object_field,
        "generation_field_after_pump_weight": effective_object_field,
        "generation_intensity_camera": generation["intensity"],
        "generation_intensity_object": generation["object_plane_intensity"],
        "correlated_field_before_spdc_psf": correlated_field,
        "idler_return_field": filtered,
        "image_function": spatial.get("image_function"),
        "joint_marginal_weight": spatial.get("marginal_weight"),
        "metadata": metadata,
    }


def michelson_signal_return_amplitude(setup: SetupParams) -> complex:
    """Signal field returned between the two crystal interactions.

    Detector path transmissions ``path1_intensity``/``path2_intensity`` are
    deliberately excluded and are applied later to the component moments.
    """
    return complex(
        np.sqrt(
            float(setup.signal_return_path_transmission)
            * float(setup.signal_return_mode_overlap)
        )
        * float(setup.signal_return_reflection_amplitude)
        * np.exp(1j * float(setup.signal_return_phase))
    )


def signal_return_amplitudes(setup: SetupParams) -> dict[str, Any]:
    """Return detector-side signal amplitudes for A or a B/C Michelson path."""
    profile = resolve_architecture_profile(setup)
    eta_s1, eta_s2 = signal_path_transmissions_from_setup(setup)
    if (
        getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
        and not is_two_crystal_architecture(setup)
    ):
        a_return = np.sqrt(eta_s1) * michelson_signal_return_amplitude(setup)
        a_second = np.sqrt(eta_s2) + 0.0j
        return {
            "signal1_amplitude": complex(a_return),
            "signal2_amplitude": complex(a_second),
            "return_active": True,
            "return_seeds_second_interaction": bool(
                setup.signal_return_seeds_second_interaction
            ),
            "model": "explicit_michelson_return_to_detector",
        }
    return {
        "signal1_amplitude": complex(np.sqrt(eta_s1)),
        "signal2_amplitude": complex(np.sqrt(eta_s2)),
        "return_active": False,
        "return_seeds_second_interaction": False,
        "model": "independent_signal_paths",
    }


def apply_qiup_spatial_transfer(
    T: np.ndarray,
    phi: np.ndarray,
    setup: SetupParams,
    fwd: ResolvedForward,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """QIUP spatial transfer on the *complex* object field (V-P0-01).

    Builds ``complex_object = √T · exp(i φ)`` then, when ``blur_model`` is
    ``double_blur``, applies the SPDC/correlation PSF to that complex field.
    ``single_blur`` skips the SPDC stage (optical PSF alone is applied later).

    Near-field / far-field only select which σ_SPDC law is in ``fwd.sigma_spdc_px``;
    both modes use the same complex-field transfer.
    """
    mode = normalize_imaging_mode(
        getattr(fwd, "imaging_mode", None) or getattr(setup, "imaging_mode", None)
    )
    corr = imaging_correlation_type(mode)
    blur_model = normalize_blur_model(
        getattr(fwd, "blur_model", None) or getattr(setup, "blur_model", None)
    )
    T_in = np.asarray(T, dtype=np.float64)
    phi_in = np.asarray(phi, dtype=np.float64)
    complex_object = np.sqrt(np.clip(T_in, 0.0, None)) * np.exp(1j * phi_in)
    generation = camera_generation_intensity(setup, complex_object.shape)
    pump_weight_meta = generation["metadata"]

    sig_spdc = float(fwd.sigma_spdc_px)
    apply_spdc = blur_model == BLUR_MODEL_DOUBLE and sig_spdc > 0.0
    if apply_spdc:
        filtered = apply_spdc_psf(complex_object, sig_spdc)
        sig_applied = sig_spdc
    else:
        filtered = complex_object
        sig_applied = 0.0

    T_out = np.clip(np.abs(filtered) ** 2, 0.0, None).astype(np.float32)
    phi_out = np.angle(filtered).astype(np.float32)
    meta = {
        "imaging_mode": mode,
        "architecture_profile": resolve_architecture_profile(setup).to_dict(),
        "correlation_type": corr,
        "blur_model": blur_model,
        "sigma_spdc_px": sig_spdc,
        "sigma_spdc_applied_px": float(sig_applied),
        "spdc_psf_applied": bool(apply_spdc),
        "pump_generation_weight": pump_weight_meta,
        "generation_intensity_camera": generation["intensity"],
        "plane": "image_plane" if mode == IMAGING_MODE_NEAR else "fourier_plane",
        "model": "complex_field_correlation_psf",
        "model_status": "intensity_level_approximation",
        "biphoton_wavefunction": False,
        "note": (
            "V-P0-01: SPDC/correlation PSF on sqrt(T)*exp(i phi). Pump shape "
            "is not multiplied into the object amplitude (L02). "
            "single_blur skips SPDC stage; double_blur applies it before optical PSF. "
            "NF: position correlations (P07/P10 Δ_V/√2). "
            "FF: momentum correlations (σ=m_i f_u λ_i/(2π w_p)). "
            "No post-image Fraunhofer FFT."
        ),
    }
    return T_out, phi_out, meta


def reconstruct_phase_from_phase_steps(
    frames: np.ndarray,
    phases_rad: Any,
) -> dict[str, Any]:
    """Independent N-step phase retrieval (L07 / P13), not a forward-model copy.

    Fits ``I(θ) = a + b cos θ + c sin θ`` per pixel. The recovered phase is
    ``φ = atan2(-c, b)`` so that ``I ≈ a + B cos(θ + φ)``. This is analysis
    of acquired frames; it does not read ``phi_eff`` or ``clean_reference``.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError("phase-step frames must have shape (K, H, W)")
    phases = np.asarray(list(phases_rad), dtype=np.float64).reshape(-1)
    if phases.size != stack.shape[0]:
        raise ValueError(
            f"phase-step count {phases.size} does not match frames {stack.shape[0]}"
        )
    if phases.size < 3:
        raise ValueError("phase retrieval requires at least 3 steps")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phase-step schedule must be finite")
    if not np.all(np.isfinite(stack)):
        raise ValueError("phase-step frames contain NaN/Inf")
    design = np.column_stack(
        (np.ones(phases.size), np.cos(phases), np.sin(phases))
    )
    pixels = stack.reshape(phases.size, -1)
    coeffs, residuals, *_ = np.linalg.lstsq(design, pixels, rcond=None)
    dc = coeffs[0].reshape(stack.shape[1], stack.shape[2])
    cos_amp = coeffs[1].reshape(stack.shape[1], stack.shape[2])
    sin_amp = coeffs[2].reshape(stack.shape[1], stack.shape[2])
    phase_rad = np.arctan2(-sin_amp, cos_amp)
    modulation = np.hypot(cos_amp, sin_amp)
    fitted = (design @ coeffs).reshape(stack.shape)
    residual = stack - fitted
    residual_rms = np.sqrt(np.mean(residual * residual, axis=0))
    snr = modulation / np.maximum(residual_rms, 1.0e-12)
    return {
        "phase_rad": phase_rad.astype(np.float64),
        "modulation": modulation.astype(np.float64),
        "dc": dc.astype(np.float64),
        "snr": snr.astype(np.float64),
        "residual_rms": residual_rms.astype(np.float64),
        "n_steps": int(phases.size),
        "phases_rad": [float(p) for p in phases],
        "method": "least_squares_n_step_psi",
        "reference": "P13 Fourier/phase-step analysis; P07 PDF pp. 65-66",
        "uses_clean_reference": False,
        "uses_forward_phi_eff": False,
    }


def wrapped_phase_difference(estimate_rad: np.ndarray, reference_rad: np.ndarray) -> np.ndarray:
    """Principal-value phase difference in (-π, π]."""
    return np.angle(
        np.exp(1j * (np.asarray(estimate_rad) - np.asarray(reference_rad)))
    )


def _optical_intensity(
    ref: dict[str, Any],
    state: LatentState,
    frame_idx: int,
    rng: np.random.Generator,
    *,
    interferometric_phase_override: float | None = None,
) -> dict[str, Any]:
    """Blocks 4–7: SLM → SPDC PSF (inside cosine) → interference → optical PSF.

    Ordering (S4): optical stages complete before detector noise.
    Separation (S3): ``sig_spdc`` is never folded into ``sig_opt`` (see F11).
    Phase-2: scales come from ``resolve_forward(state.get_setup(), N)``.
    """
    e = state.effective()
    phi = np.asarray(ref["phi"], dtype=np.float32)
    T = np.asarray(ref["T"], dtype=np.float32)
    H, W = reference_target_hw(ref)
    if H != W:
        raise ReferenceSchemaError(
            f"optical forward currently requires square grid, got target_size={(H, W)}"
        )
    N = int(H)
    setup = state.get_setup()
    architecture_profile = validate_forward_profile_supported(setup)
    fwd = resolve_forward(setup, N, s_focus=e["s_focus"])
    dP = e["s_pump"] * float(fwd.pump_sigma_rel) * float(rng.normal())
    spdc = state.spdc_operating_point(
        relative_power_fluctuation=(
            dP if bool(getattr(setup, "use_physical_spdc_gain", False)) else 0.0
        )
    )

    # M9: object misregistration on SLM (circular shift; no-op at 0)
    dx = int(round(float(setup.slm_misreg_x_px)))
    dy = int(round(float(setup.slm_misreg_y_px)))
    if dx != 0 or dy != 0:
        phi = np.roll(np.roll(phi, dy, axis=0), dx, axis=1)
        T = np.roll(np.roll(T, dy, axis=0), dx, axis=1)

    # C8 SLM phase calibration first, then form full complex object √T·E_SLM.
    re, im = _slm_complex(phi, e["s_slm"], fwd)
    complex_object = np.sqrt(np.clip(T, 0.0, None)).astype(np.float64) * (
        re.astype(np.float64) + 1j * im.astype(np.float64)
    )
    # The object intensity is physically bounded by [0, 1].  Float32
    # sin/cos roundoff can otherwise make a unit-transmission phase object
    # exceed 1 by a few ulps and fail the downstream strict domain check.
    T_pre = np.clip(np.abs(complex_object) ** 2, 0.0, 1.0).astype(np.float32)
    phi_pre = np.angle(complex_object).astype(np.float32)

    # The registered A/B/C/D profiles opt into architecture-specific object
    # propagation. Historical configurations retain the reviewed transfer.
    object_transfer_state: dict[str, Any] | None = None
    if (
        getattr(setup, "architecture_propagation_model", None)
        == ARCHITECTURE_PROPAGATION_COMPLEX_V1
    ):
        object_transfer_state = architecture_object_transfer(
            T_pre, phi_pre, setup, fwd
        )
        T_eff = object_transfer_state["T_eff"]
        phi_eff = object_transfer_state["phi_eff"]
        spatial_meta = object_transfer_state["metadata"]
    else:
        # V-P0-01 + blur_model: SPDC/correlation PSF on complex field
        # (double_blur), or skip it (single_blur).
        T_eff, phi_eff, spatial_meta = apply_qiup_spatial_transfer(
            T_pre, phi_pre, setup, fwd
        )
    if not spdc["physical"]:
        # Legacy synthetic floor retained only for numerical regression.
        T_eff = np.clip(0.15 + 0.85 * T_eff, 0.0, 1.0).astype(np.float32)
    else:
        T_eff = np.clip(T_eff, 0.0, 1.0).astype(np.float32)
    sig_spdc = float(spatial_meta.get("sigma_spdc_applied_px", 0.0))
    generation_camera = None
    if object_transfer_state is not None:
        generation_camera = object_transfer_state.get(
            "generation_intensity_camera"
        )
    if generation_camera is None:
        generation_camera = spatial_meta.get("generation_intensity_camera")
    signal_marginal_applied_to_moments = False

    # L06: idler-lens phase belongs on the seed that enters C12, not on I±.
    delta_z_used = float(fwd.delta_z_um)
    ring_meta: dict[str, Any] = {"enabled": False}
    lens_phase_rad: np.ndarray | None = None
    rings_on = bool(setup.enable_lens_rings) and (
        abs(delta_z_used) > 1e-9
        or abs(float(setup.lens_shift_x_um)) > 1e-9
        or abs(float(setup.lens_shift_y_um)) > 1e-9
    )
    if rings_on:
        lens_state = idler_lens_misalignment_phase_map(
            N, setup, delta_z_um=delta_z_used
        )
        lens_phase_rad = np.asarray(lens_state["phase_rad"], dtype=np.float64)
        ring_meta = {
            key: value for key, value in lens_state.items() if key != "phase_rad"
        }

    # C1 pointing jitter first so idler η_mode can inherit the same Δx, Δy.
    # Thermal beam drift (C4) is applied later to cx,cy only — not to η_mode.
    cx0 = N / 2.0
    cy0 = N / 2.0
    sigma_beam = float(fwd.pointing_rms_x)
    delta_x = e["s_pump"] * sigma_beam * float(rng.normal())
    delta_y = e["s_pump"] * sigma_beam * float(rng.normal())
    eta_overlap = pump_overlap_eta(delta_x, delta_y, fwd.w_px)

    vis = _visibility_factors(
        e,
        fwd,
        pump_delta_x_px=delta_x,
        pump_delta_y_px=delta_y,
        setup=setup,
    )
    V_gamma = float(vis["V_gamma"])
    V_eff = vis["V_eff"]
    use_two_beam = bool(fwd.use_physical_two_beam)
    # §7.15: ηmode is in tseed; do not also apply a spatial overlap visibility map.
    use_field_map = bool(fwd.use_field_overlap_visibility_map) and (
        not bool(spdc["physical"])
    )
    # Spatial mottling / walk / pump overlap act on Γ, not on a fake r(s_align).
    V_base = 1.0 if spdc["physical"] else (V_gamma if use_two_beam else V_eff)

    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)

    if spdc["physical"]:
        V_map = np.ones((N, N), dtype=np.float32)
    elif use_field_map and setup is not None:
        # PDF §4.3 eq. 74: V(x,y) from |E1 E2*|; centers follow pump + idler Δ.
        dx_um, dy_um = idler_displacement_um(
            setup,
            pump_delta_x_px=delta_x,
            pump_delta_y_px=delta_y,
        )
        dx_rel = um_to_px_setup(dx_um, setup)
        dy_rel = um_to_px_setup(dy_um, setup)
        w1_um, w2_um = resolved_idler_waist_um(setup)
        w1_px = um_to_px_setup(w1_um, setup)
        w2_px = um_to_px_setup(w2_um, setup)
        # Midpoint tracks pump/signal pointing; separation = idler relative Δ.
        cx_mid = cx0 + delta_x
        cy_mid = cy0 + delta_y
        cx1 = cx_mid - 0.5 * dx_rel
        cy1 = cy_mid - 0.5 * dy_rel
        cx2 = cx_mid + 0.5 * dx_rel
        cy2 = cy_mid + 0.5 * dy_rel
        tilt = mode_tilt_factor(
            w1_um,
            w2_um,
            dtheta_mrad=float(setup.idler_dtheta_mrad),
            lambda_i_nm=float(setup.lambda_i_nm),
        )
        V_spatial = field_overlap_visibility_map(
            xx,
            yy,
            cx1=cx1,
            cy1=cy1,
            w1_px=w1_px,
            cx2=cx2,
            cy2=cy2,
            w2_px=w2_px,
            tilt_factor=tilt,
        )
        V_map = np.clip(V_base * V_spatial, 0.0, 1.0).astype(np.float32)
        vis = dict(vis)
        vis["V_spatial_mean"] = float(np.mean(V_spatial))
        vis["field_dx_rel_px"] = float(dx_rel)
        vis["field_dy_rel_px"] = float(dy_rel)
        vis["field_w1_px"] = float(w1_px)
        vis["field_w2_px"] = float(w2_px)
        vis["field_tilt_factor"] = float(tilt)
        vis["V_eff"] = float(
            np.clip(V_base * float(vis["V_port"]) * vis["V_spatial_mean"], 0.0, 1.0)
        )
        V_eff = vis["V_eff"]
    elif e["s_align"] > 0.05:
        # Legacy PDF eq. 73: V_eff (1 - 0.25 s_align m̃)
        mott = rng.normal(0.0, 1.0, size=(N, N)).astype(np.float32)
        mott = gaussian_filter(mott, sigma=float(fwd.mottle_smooth_sigma_px))
        mott = mott / (float(mott.std()) + EPS)
        V_map = np.clip(
            V_base * (1.0 - fwd.V_map_mottle * e["s_align"] * mott), 0.0, 1.0
        )
    else:
        V_map = np.full((N, N), V_base, dtype=np.float32)

    # C4 / PDF §5.2–§5.3 / V-P1-05: θ = θ0 + k·OPD with OPD = ΔL0 + ΔL_thermal
    # (signed). Coherence uses |OPD| only — not this phase path.
    use_phys_T = bool(fwd.use_physical_thermal)
    t_s = float(frame_idx) * float(state.exposure_s)
    if use_phys_T:
        opd_mm = float(optical_path_difference_mm(setup, t_s=t_s))
        dphi_opd = float(opd_phase_rad(setup, t_s=t_s))
        theta0 = (
            float(THETA0_RAD)
            + dphi_opd
            + float(vis.get("spectral_phase_offset_rad", 0.0))
        )
        phase_walk = 0.0
        vis_walk = 0.0
        vis = dict(vis)
        vis["optical_path_difference_mm"] = opd_mm
        vis["opd_phase_rad"] = dphi_opd
    else:
        theta0 = np.pi / 4 + float(fwd.thermal_phase_offset_rad)
        phase_walk = float(fwd.phase_drift_rad_per_frame)
        vis_walk = float(fwd.vis_walk_per_frame)
    theta = (
        theta0
        + phase_walk * e["s_thermal"] * frame_idx
        + rng.normal(0.0, fwd.phase_jitter_align * e["s_align"])
    )
    if interferometric_phase_override is not None and not bool(spdc["physical"]):
        theta = theta + float(interferometric_phase_override)
    V_frame = float(
        np.clip(
            V_base * (1.0 - vis_walk * e["s_thermal"] * frame_idx),
            0.0,
            1.0,
        )
    )
    if not spdc["physical"]:
        V_map = np.clip(V_map * (V_frame / max(V_base, EPS)), 0.0, 1.0).astype(
            np.float32
        )

    # PDF §2.6: pump–crystal η_overlap on visibility. Physical Mode: pump
    # pointing enters generation O_geom (static) and relay ηmode/tseed
    # separately (V-P1-04) — never multiply the same offset twice onto V.
    apply_pump_overlap_vis = bool(setup.couple_pointing_overlap_visibility) and (
        not bool(spdc["physical"])
    )
    if apply_pump_overlap_vis:
        V_map = np.clip(V_map * eta_overlap, 0.0, 1.0).astype(np.float32)
        vis = dict(vis)
        vis["eta_overlap"] = float(eta_overlap)
        vis["V_pump_overlap"] = float(eta_overlap)
        vis["V_eff_after_pump"] = float(np.clip(V_eff * eta_overlap, 0.0, 1.0))
        vis["pump_delta_x_px"] = float(delta_x)
        vis["pump_delta_y_px"] = float(delta_y)
    elif spdc["physical"]:
        vis = dict(vis)
        vis["eta_overlap"] = float(eta_overlap)
        vis["V_pump_overlap"] = 1.0
        vis["pump_overlap_applied_to_visibility"] = False
        vis["pump_delta_x_px"] = float(delta_x)
        vis["pump_delta_y_px"] = float(delta_y)

    path1_report = float(vis["I1"])
    path2_report = float(vis["I2"])
    moments_det: dict[str, Any] | None = None
    if spdc["physical"]:
        nbar1 = float(spdc["mean_photons_per_mode_NL1"])
        nbar2 = float(spdc["mean_photons_per_mode_NL2"])
        architecture_specific = (
            getattr(setup, "architecture_propagation_model", None)
            == ARCHITECTURE_PROPAGATION_COMPLEX_V1
        )
        eta_mode_relay = vis.get("eta_mode")
        if eta_mode_relay is None:
            eta_mode_relay = eta_mode_from_setup(
                setup,
                pump_delta_x_px=delta_x,
                pump_delta_y_px=delta_y,
            )
        eta_mode_configured = (
            float(setup.idler_mode_matching) if architecture_specific else 1.0
        )
        eta_mode = float(eta_mode_relay) * eta_mode_configured
        eta_i_base = idler_path_transmission_from_setup(setup)
        idler_round_trip_transmission = 1.0
        idler_return_phase = 0.0
        if interferometric_phase_override is not None:
            relative_interferometric_phase = float(
                interferometric_phase_override
            )
        elif architecture_specific:
            relative_interferometric_phase = float(
                setup.relative_interferometric_phase
            )
        else:
            relative_interferometric_phase = 0.0
        if architecture_specific:
            if not is_two_crystal_architecture(setup):
                idler_round_trip_transmission = 1.0 - float(
                    setup.idler_round_trip_loss
                )
                idler_return_phase = float(setup.idler_return_phase)
        eta_i = eta_i_base * idler_round_trip_transmission
        if object_transfer_state is not None:
            t_object_for_seed = np.asarray(
                object_transfer_state["idler_return_field"], dtype=np.complex128
            )
        else:
            t_object_for_seed = idler_object_amplitude(T_eff, phi_eff)
        t_object_for_seed = t_object_for_seed * np.exp(
            1j * (idler_return_phase + relative_interferometric_phase)
        )
        if lens_phase_rad is not None:
            t_object_for_seed = t_object_for_seed * np.exp(1j * lens_phase_rad)
        seed = idler_seed_from_parts(
            eta_mode,
            eta_i,
            t_object_for_seed,
        )
        use_michelson_moments = (
            architecture_specific
            and not is_two_crystal_architecture(setup)
        )
        if use_michelson_moments:
            src = spdc_michelson_return_moments(
                nbar1,
                nbar2,
                seed["t_seed"],
                michelson_signal_return_amplitude(setup),
                signal_seeds_second_interaction=bool(
                    setup.signal_return_seeds_second_interaction
                ),
            )
        else:
            src = spdc_double_pass_moments(nbar1, nbar2, seed["t_seed"])
        eta_s1, eta_s2 = signal_path_transmissions_from_setup(setup)
        gamma_path = float(vis.get("gamma_path") or vis["V_coh"])
        signal_state = signal_return_amplitudes(setup)
        moments_det = apply_signal_path_losses(
            src["N1_source"],
            src["N2_source"],
            src["C12_source"],
            eta_s1=eta_s1,
            eta_s2=eta_s2,
            gamma_path=gamma_path,
        )
        filter_throughput = float(vis.get("spectral_filter_throughput", 1.0))
        if filter_throughput != 1.0:
            moments_det = {
                "N1": np.asarray(moments_det["N1"]) * filter_throughput,
                "N2": np.asarray(moments_det["N2"]) * filter_throughput,
                "C12": np.asarray(moments_det["C12"]) * filter_throughput,
                **{
                    key: value for key, value in moments_det.items()
                    if key not in {"N1", "N2", "C12"}
                },
            }
        if generation_camera is not None:
            gen = np.asarray(generation_camera, dtype=np.float64)
            if gen.shape == np.asarray(moments_det["N1"]).shape:
                # The single-photon marginal belongs to the physical moment
                # fields before the combiner.  Applying the same nonnegative
                # marginal to N1, N2, and C12 preserves local visibility while
                # giving both no-object arms finite support independent of the
                # idler object.  This is not a post-image pump Gaussian.
                moments_det = {
                    "N1": np.asarray(moments_det["N1"]) * gen,
                    "N2": np.asarray(moments_det["N2"]) * gen,
                    "C12": np.asarray(moments_det["C12"]) * gen,
                    **{
                        key: value for key, value in moments_det.items()
                        if key not in {"N1", "N2", "C12"}
                    },
                }
                signal_marginal_applied_to_moments = True
        obs = spdc_observables_from_moments(
            moments_det["N1"], moments_det["N2"], moments_det["C12"]
        )
        g1_map = obs["intrinsic_g1"]
        vis = dict(vis)
        vis.update(
            {
                "gain_regime": spdc["regime"],
                "gain_regime_NL1": spdc.get("regime_NL1", spdc["regime"]),
                "gain_regime_NL2": spdc.get("regime_NL2", spdc["regime"]),
                "gain_r": float(spdc["gain_r"]),
                "spdc_gain_r_NL1": float(spdc["spdc_gain_r_NL1"]),
                "spdc_gain_r_NL2": float(spdc["spdc_gain_r_NL2"]),
                "mean_photons_per_mode": nbar1,
                "mean_photons_per_mode_NL1": nbar1,
                "mean_photons_per_mode_NL2": nbar2,
                "gain_source": spdc["source"],
                "source_moment_model": src.get(
                    "moment_model", "independent_source_idler_seeding"
                ),
                "source_moment_assumptions": src.get(
                    "assumptions",
                    "independent interactions; matched idler seed mode",
                ),
                "pump_power_mW": spdc["pump_power_mW"],
                "pump_relative_power_fluctuation": float(
                    spdc.get("pump_relative_power_fluctuation", dP)
                ),
                "pump_power_NL1_mW": spdc.get("pump_power_NL1_mW"),
                "pump_power_NL2_mW": spdc.get("pump_power_NL2_mW"),
                "flux_includes_pump_dP": False,
                "eta_mode": float(eta_mode),
                "eta_mode_relay": float(eta_mode_relay),
                "idler_mode_matching": float(eta_mode_configured),
                "eta_i_path": float(eta_i),
                "eta_i_path_base": float(eta_i_base),
                "idler_round_trip_loss": float(
                    1.0 - idler_round_trip_transmission
                ),
                "idler_round_trip_transmission": float(
                    idler_round_trip_transmission
                ),
                "idler_return_phase": float(idler_return_phase),
                "relative_interferometric_phase": float(
                    relative_interferometric_phase
                ),
                "T_seed_mean": float(np.mean(seed["T_seed"])),
                "N1_mean": float(np.mean(obs["N1"])),
                "N2_mean": float(np.mean(obs["N2"])),
                "C12_abs_mean": float(np.mean(obs["C12_abs"])),
                "gamma_path": float(gamma_path),
                "spectral_filter_throughput": filter_throughput,
                "spectral_filter_applied_to_moments_once": True,
                "intrinsic_g1_mean": float(np.mean(g1_map)),
                "visibility_fixed_combiner_mean": float(
                    np.mean(obs["visibility_50_50"])
                ),
                "V_eff": float(np.mean(obs["visibility_50_50"])),
                "signal1_relative_mean": float(
                    np.mean(obs["N1"] / nbar1) if nbar1 > 0.0 else eta_s1
                ),
                "signal2_relative_mean": float(
                    np.mean(obs["N2"] / nbar1) if nbar1 > 0.0 else eta_s2
                ),
                "coherence_relative_mean": float(
                    np.mean(obs["C12_abs"] / nbar1) if nbar1 > 0.0 else 0.0
                ),
                "eta_mode_applied_to_visibility": False,
                "object_applied_to_visibility": False,
                "V_gain_applied_to_intensity": False,
                "V_port_applied_to_intensity": False,
                "signal_return_active": bool(signal_state["return_active"]),
                "signal_return_model": signal_state["model"],
                "signal_return_seeds_second_interaction": bool(
                    signal_state["return_seeds_second_interaction"]
                ),
                "signal_return_amplitude_abs": float(
                    abs(signal_state["signal1_amplitude"])
                ),
                "signal_return_amplitude_phase": float(
                    np.angle(signal_state["signal1_amplitude"])
                ),
                "signal_return_between_interactions_abs": (
                    float(abs(src["signal_return_amplitude"]))
                    if "signal_return_amplitude" in src else None
                ),
                "signal_return_stimulates_interaction_2": bool(
                    src.get("signal_seeds_second_interaction", False)
                ),
                "signal2_path_amplitude_abs": float(
                    abs(signal_state["signal2_amplitude"])
                ),
            }
        )
        path1_report = vis["signal1_relative_mean"]
        path2_report = vis["signal2_relative_mean"]

    # C1 pump: Physical Mode applies dP only to P_p1→P_p2 (§7.14). Occupancy
    # is in N1, N2, C12 (§7.17); scale is flux0/nref, not flux0·nbar/nref
    # times relative intensities.
    flux0 = float(fwd.flux0_e_per_s) * state.exposure_s
    if spdc["physical"]:
        nref = float(setup.spdc_reference_mean_photons_per_mode)
        flux_scale = flux0 / nref
        flux = physical_mode_flux(
            flux0,
            float(spdc["mean_photons_per_mode"]),
            nref,
        )
    else:
        super_p = 1.0 + float(fwd.flux_boost) * e["s_gain"]
        flux = flux0 * (1.0 + dP) * super_p
        flux_scale = flux

    # Signal / collection envelope follows the same pump pointing as the fields.
    cx = cx0 + delta_x
    cy = cy0 + delta_y
    # PDF §5.4/§5.9: physical thermal beam shift (not engineered px/frame walk)
    if use_phys_T:
        bx_um, by_um = thermal_beam_shift_um(setup, t_s=t_s)
        cx += um_to_px_setup(bx_um, setup)
        cy += um_to_px_setup(by_um, setup)
    else:
        cx += fwd.beam_drift_x_px_per_frame * e["s_thermal"] * frame_idx
        cy += fwd.beam_drift_y_px_per_frame * e["s_thermal"] * frame_idx

    camera_mode = normalize_imaging_mode(getattr(setup, "imaging_mode", None))
    env, final_camera_pump_envelope_applied, final_camera_pump_envelope_equation = (
        final_camera_collection_envelope(
            setup,
            camera_mode=camera_mode,
            N=N,
            cx=float(cx),
            cy=float(cy),
            xx=xx,
            yy=yy,
            w_px=float(fwd.w_px),
        )
    )

    try:
        from diagnostics.collection_envelope_policy import apply_policy_to_env

        _env_override = apply_policy_to_env(
            env=env,
            camera_mode=camera_mode,
            final_camera_pump_envelope_applied=final_camera_pump_envelope_applied,
            final_camera_pump_envelope_equation=final_camera_pump_envelope_equation,
            N=N,
            cx=float(cx),
            cy=float(cy),
            xx=xx,
            yy=yy,
            w_px=float(fwd.w_px),
            imaging_mode_far=IMAGING_MODE_FAR,
            imaging_mode_near=IMAGING_MODE_NEAR,
        )
        if _env_override is not None:
            env = _env_override["env"]
            final_camera_pump_envelope_applied = bool(_env_override["applied"])
            final_camera_pump_envelope_equation = str(_env_override["equation"])
    except ImportError:
        pass

    # Fan Δφ(x,y): C3 path-interference wavefront (PDF) — NOT imaging Zernike.
    # δφ_rms = 2π·W_waves + δφ_align_span·s_align + optional LatentState extra.
    dphi_rms, corr, fan_mode = resolve_c3_fan_delta_phi_rms(
        setup,
        s_align=float(e["s_align"]),
        delta_phi_rms_extra=float(getattr(state, "delta_phi_rms", 0.0) or 0.0),
    )
    # Allow LatentState.delta_phi_corr_px to override setup correlation length.
    corr_state = float(getattr(state, "delta_phi_corr_px", 0.0) or 0.0)
    if corr_state > 0.0:
        corr = corr_state
    if dphi_rms > 0.0:
        noise = rng.normal(0.0, 1.0, size=(N, N)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=max(float(corr), 1.0))
        noise = noise / (float(noise.std()) + EPS)
        delta_phi = (dphi_rms * noise).astype(np.float32)
    else:
        delta_phi = np.zeros((N, N), dtype=np.float32)

    # Induced coherence. Physical Mode: I± from N1, N2, C12 only (§7.19).
    if spdc["physical"] and moments_det is not None:
        I_plus_mode, I_minus_mode = _interfere_from_moments(
            moments_det["N1"],
            moments_det["N2"],
            moments_det["C12"],
            theta=theta,
            delta_phi=delta_phi,
        )
        scale_map = (float(flux_scale) * env).astype(np.float32)
        I_plus = (scale_map * I_plus_mode).astype(np.float32)
        I_minus = (scale_map * I_minus_mode).astype(np.float32)
        try:
            from diagnostics.collection_envelope_policy import record_pre_psf_intensity

            record_pre_psf_intensity(
                flux_scale=float(flux_scale),
                env=env,
                I_plus_mode=I_plus_mode,
                I_plus_pre_psf=I_plus,
                I_minus_pre_psf=I_minus,
            )
        except ImportError:
            pass
    else:
        base = (flux * env).astype(np.float32)
        if use_two_beam:
            I1_field = (0.5 * float(vis["I1"]) * base).astype(np.float32)
            I2_field = (0.5 * float(vis["I2"]) * base).astype(np.float32)
        else:
            I1_field = (0.5 * base).astype(np.float32)
            I2_field = (0.5 * base).astype(np.float32)
        phi_legacy = phi_eff
        if lens_phase_rad is not None:
            phi_legacy = (
                np.asarray(phi_eff, dtype=np.float64) + lens_phase_rad
            ).astype(np.float32)
        I_plus, I_minus = _interfere(
            I1_field,
            I2_field,
            V_map,
            T_eff,
            phi_legacy,
            theta=theta,
            delta_phi=delta_phi,
        )

    if generation_camera is not None and not signal_marginal_applied_to_moments:
        gen = np.asarray(generation_camera, dtype=np.float32)
        if gen.shape == I_plus.shape:
            I_plus = (I_plus * gen).astype(np.float32)
            I_minus = (I_minus * gen).astype(np.float32)

    # C7 optical PSF *after* interference, before Poisson (ordering constraint)
    # S3: do NOT add sig_spdc into this sum.
    # §6.5–6.10: σ_defocus = K·|Δz| (eq.164); Δz from C4 eq.160 (Δf) or mech;
    #           s_focus = |Δz|/Δz_max (eq.166). Eq.162 unused.
    K_used = float(fwd.K_defocus_px_per_um)
    sigma_defocus = abs(K_used * delta_z_used)

    sig_opt = float(
        np.sqrt(
            float(fwd.sigma_diffraction_px) ** 2
            + float(sigma_defocus) ** 2
            + float(fwd.sigma_aberration_px) ** 2
        )
    )
    blur_info = resolve_blur_sigmas(
        sigma_spdc_px=float(fwd.sigma_spdc_px),
        sigma_optical_px=sig_opt,
        blur_model=getattr(fwd, "blur_model", None)
        or getattr(setup, "blur_model", None),
    )
    spatial_meta = dict(spatial_meta)
    spatial_meta.update(
        {
            "sigma_optical_px": float(blur_info["sigma_optical_px"]),
            "sigma_total_px": float(blur_info["sigma_total_px"]),
            "blur_chain": (
                "object->optical_psf->detector"
                if blur_info["blur_model"] == BLUR_MODEL_SINGLE
                else "object->spdc_psf->optical_psf->detector"
            ),
            "return_field_state": {
                "idler_return_explicit": bool(object_transfer_state is not None),
                "signal_return_explicit": bool(
                    vis.get("signal_return_active", False)
                ),
                "idler_round_trip_loss": vis.get("idler_round_trip_loss"),
                "idler_mode_matching": vis.get("idler_mode_matching"),
                "idler_return_phase": vis.get("idler_return_phase"),
                "relative_interferometric_phase": vis.get(
                    "relative_interferometric_phase"
                ),
                "signal_return_amplitude_abs": vis.get(
                    "signal_return_amplitude_abs"
                ),
                "signal_return_amplitude_phase": vis.get(
                    "signal_return_amplitude_phase"
                ),
                "signal_return_seeds_second_interaction": vis.get(
                    "signal_return_seeds_second_interaction"
                ),
                "high_gain_michelson_status": (
                    "bogoliubov_single_mode_supported_spatial_model_provisional"
                    if object_transfer_state is not None
                    and not is_two_crystal_architecture(setup)
                    else "not_applicable"
                ),
            },
        }
    )
    I_plus = gaussian_filter(I_plus, sigma=sig_opt)
    I_minus = gaussian_filter(I_minus, sigma=sig_opt)
    try:
        from diagnostics.collection_envelope_policy import record_post_psf_intensity

        record_post_psf_intensity(I_plus=I_plus, sig_opt=float(sig_opt))
    except ImportError:
        pass

    # Geometric IQ (PDF §6.17–6.18): Magnification/Rotation ≠ Optical Blur — keep separate.
    # ε_M = delta_M_rel → scale = 1 + ε_M; θ = image_rotation_deg. Applied after PSF.
    extra_blur = float(setup.sigma_blur_extra_px)
    if extra_blur > 0.0:
        I_plus = gaussian_filter(I_plus, sigma=extra_blur)
        I_minus = gaussian_filter(I_minus, sigma=extra_blur)
    scale = float(fwd.image_scale)
    rot = float(fwd.image_rotation_deg)
    eps_M = float(fwd.delta_M_rel)
    if abs(scale - 1.0) > 1e-9 or abs(rot) > 1e-9:
        from scipy.ndimage import rotate, zoom

        def _iq(img: np.ndarray) -> np.ndarray:
            out = img
            if abs(rot) > 1e-9:
                out = rotate(out, rot, reshape=False, order=1, mode="nearest")
            if abs(scale - 1.0) > 1e-9:
                z = zoom(out, scale, order=1)
                # center-crop / pad back to N×N
                out = np.zeros((N, N), dtype=np.float32)
                h, w = z.shape
                y0 = max(0, (h - N) // 2)
                x0 = max(0, (w - N) // 2)
                ys = max(0, (N - h) // 2)
                xs = max(0, (N - w) // 2)
                hh = min(N, h)
                ww = min(N, w)
                out[ys : ys + hh, xs : xs + ww] = z[y0 : y0 + hh, x0 : x0 + ww]
            return out.astype(np.float32)

        I_plus = _iq(I_plus)
        I_minus = _iq(I_minus)

    # Unwanted Fabry–Pérot / etalon fringes (common path; optional)
    I_plus, I_minus, etalon_meta = apply_etalon_fringes(I_plus, I_minus, setup)

    # Vignetting disabled: no V_vig / α_vig / s_vig intensity falloff in the model.

    return {
        "I_plus": I_plus,
        "I_minus": I_minus,
        "D_ideal": (I_plus - I_minus).astype(np.float32),
        "V_map": V_map,
        "vis": vis,
        "sig_spdc": sig_spdc,
        "sig_opt": sig_opt,
        "sigma_total_px": float(blur_info["sigma_total_px"]),
        "blur_model": blur_info["blur_model"],
        "imaging_mode": spatial_meta.get("imaging_mode"),
        "correlation_type": spatial_meta.get("correlation_type"),
        "qiup_spatial_transfer": spatial_meta,
        "signal_filter": vis.get("spectral_filter", {}),
        "etalon_fringes": etalon_meta,
        "sigma_diffraction_px": float(fwd.sigma_diffraction_px),
        "sigma_defocus_px": float(sigma_defocus),
        "sigma_aberration_px": float(fwd.sigma_aberration_px),
        "optical_psf_model": "effective_gaussian",
        "delta_z_um_used": float(delta_z_used),
        "K_defocus_px_per_um": float(K_used),
        "coc_um": float(fwd.coc_um),
        "s_focus_from_dz": float(fwd.s_focus_from_dz),
        "lens_rings": ring_meta,
        "sigma_blur_extra_px": extra_blur,
        "delta_M_rel": eps_M,
        "image_scale": scale,
        "image_rotation_deg": rot,
        "M_QIUP": float(getattr(fwd, "M_QIUP", resolve_imaging_magnification(setup))),
        "object_pixel_size_um": float(object_pixel_size_um(setup)),
        "beam_size_px": float(fwd.w_px),
        "final_camera_pump_envelope_applied": bool(
            final_camera_pump_envelope_applied
        ),
        "final_camera_pump_envelope_equation": (
            final_camera_pump_envelope_equation
        ),
        "pump_width_correlation_contribution": (
            "active_in_physical_joint_momentum_pump_angular_spectrum"
            if camera_mode == IMAGING_MODE_FAR
            and _physical_far_field_signal_marginal_enabled(setup)
            else "active_sigma_q_equals_1_over_w_p"
            if camera_mode == IMAGING_MODE_FAR
            else "active_near_field_generation_and_collection"
        ),
        "PSF_size_px": float(fwd.sigma_spdc_px),
        "slm_misreg_x_px": float(setup.slm_misreg_x_px),
        "slm_misreg_y_px": float(setup.slm_misreg_y_px),
        "theta": float(theta),
        "delta_phi": delta_phi,
        "delta_phi_rms": float(dphi_rms),
        "delta_phi_corr_px": float(corr),
        "fan_phase_mode": fan_mode,
        "wavefront_rms_waves": float(setup.wavefront_rms_waves),
        "flux": float(flux),
        "optical_scale_per_mode": float(flux_scale) if spdc["physical"] else float(flux),
        "eta_overlap": float(eta_overlap),
        "delta_x_px": float(delta_x),
        "delta_y_px": float(delta_y),
        "couple_pointing_overlap_visibility": bool(
            setup.couple_pointing_overlap_visibility
        ),
        "pump_overlap_applied_to_visibility": bool(apply_pump_overlap_vis),
        "use_physical_two_beam": bool(use_two_beam),
        "use_field_overlap_visibility_map": bool(use_field_map),
        "path1_intensity": float(path1_report),
        "path2_intensity": float(path2_report),
        "signal_path1_transmission": float(setup.path1_intensity),
        "signal_path2_transmission": float(setup.path2_intensity),
        "spdc_operating_point": spdc,
        "spdc_physics": {
            "physics_policy_id": getattr(setup, "physics_policy_id", None),
            "architecture_profile": architecture_profile.to_dict(),
            "architecture": normalize_spdc_architecture(
                getattr(setup, "spdc_architecture", None)
            ),
            "d_eff_pm_per_V": setup.d_eff_pm_per_V,
            **effective_nonlinearity_metadata(setup),
            "d_eff_precedence": getattr(setup, "d_eff_precedence", None),
            "d33_pm_per_V": getattr(setup, "d33_pm_per_V", None),
            "poling_period_um": float(setup.poling_period_um),
            "poling_period_used_um": float(qpm_poling_period_um(setup, interaction=1)),
            "poling_period_used_NL1_um": float(
                qpm_poling_period_um(setup, interaction=1)
            ),
            "poling_period_used_NL2_um": float(
                qpm_poling_period_um(setup, interaction=2)
            ),
            "qpm_interaction_1": qpm_configuration_evidence(setup, interaction=1),
            "qpm_interaction_2": qpm_configuration_evidence(setup, interaction=2),
            "spdc_type": getattr(setup, "spdc_type", None) or setup.qpm_type,
            "crystal_matching_mode": getattr(setup, "crystal_matching_mode", None),
            "T_crystal_C": float(setup.T_crystal_NL1_C),
            "T_crystal_NL1_C": float(
                spdc_crystal_state(setup, interaction=1)["T_crystal_C"]
            ),
            "T_crystal_NL2_C": float(
                spdc_crystal_state(setup, interaction=2)["T_crystal_C"]
            ),
            "crystal_length_NL1_mm": float(
                spdc_crystal_state(setup, interaction=1)["crystal_length_mm"]
            ),
            "crystal_length_NL2_mm": float(
                spdc_crystal_state(setup, interaction=2)["crystal_length_mm"]
            ),
            "n_pump": float(
                spdc_crystal_state(setup, interaction=1)["n_pump"]
            ),
            "n_signal": float(
                spdc_crystal_state(setup, interaction=1)["n_signal"]
            ),
            "n_idler": float(
                spdc_crystal_state(setup, interaction=1)["n_idler"]
            ),
            "delta_k_qpm_per_m": float(
                delta_k_qpm_per_m_from_setup(setup, interaction=1)
            ),
            "delta_k_qpm_per_m_NL1": float(
                delta_k_qpm_per_m_from_setup(setup, interaction=1)
            ),
            "delta_k_qpm_per_m_NL2": float(
                delta_k_qpm_per_m_from_setup(setup, interaction=2)
            ),
            "O_geom_NL1": float(
                spdc_geometry_overlap_from_setup(setup, interaction=1)
            ),
            "O_geom_NL2": float(
                spdc_geometry_overlap_from_setup(setup, interaction=2)
            ),
            "k_spdc_per_sqrt_W_NL1": spdc.get("k_spdc_per_sqrt_W_NL1"),
            "k_spdc_per_sqrt_W_NL2": spdc.get("k_spdc_per_sqrt_W_NL2"),
            "pump_power_NL1_mW": spdc.get("pump_power_NL1_mW"),
            "pump_power_NL2_mW": spdc.get("pump_power_NL2_mW"),
            "pump_balance_mode": spdc.get("pump_balance_mode"),
            "P_target_1_mW": spdc.get("P_target_1_mW"),
            "P_target_2_mW": spdc.get("P_target_2_mW"),
            "P_actual_1_mW": spdc.get("P_actual_1_mW"),
            "P_actual_2_mW": spdc.get("P_actual_2_mW"),
            "delta_P": spdc.get("delta_P"),
            "abs_delta_P": spdc.get("abs_delta_P"),
            "spdc_gain_r_NL1": spdc.get("spdc_gain_r_NL1"),
            "spdc_gain_r_NL2": spdc.get("spdc_gain_r_NL2"),
            "mean_photons_per_mode_NL1": spdc.get("mean_photons_per_mode_NL1"),
            "mean_photons_per_mode_NL2": spdc.get("mean_photons_per_mode_NL2"),
            "T_seed": vis.get("T_seed_mean"),
            "N1": vis.get("N1_mean"),
            "N2": vis.get("N2_mean"),
            "C12_abs": vis.get("C12_abs_mean"),
            "regime_NL1": vis.get("gain_regime_NL1"),
            "regime_NL2": vis.get("gain_regime_NL2"),
            "pol_pump": setup.pol_pump,
            "pol_signal": setup.pol_signal,
            "pol_idler": setup.pol_idler,
            "paper_detector_path": "analog_emccd",
        },
        "S_cam": (
            None if generation_camera is None
            else np.asarray(generation_camera, dtype=np.float32)
        ),
        "N1": (
            None if moments_det is None
            else np.asarray(moments_det["N1"], dtype=np.float32)
        ),
        "N2": (
            None if moments_det is None
            else np.asarray(moments_det["N2"], dtype=np.float32)
        ),
        "abs_C12": (
            None if moments_det is None
            else np.abs(np.asarray(moments_det["C12"])).astype(np.float32)
        ),
        "signal_marginal_applied_to_moments": bool(
            signal_marginal_applied_to_moments
        ),
        "T_eff": T_eff,
        "phi_eff": phi_eff,
        "resolved_forward": fwd,
        "w_px": float(fwd.w_px),
    }


# ==============================================================================
# OBJECT REFERENCE (T/φ for the twin; clean D comes from simulate_forward)
# Role: object loader for the forward model
# Models: M → T, φ, O_i only (V-P0-04: no independent interference model)
# Provenance: arch
# ==============================================================================

class ReferenceSchemaError(ValueError):
    """Reference dict failed schema / shape / dtype checks (V-P0-11)."""


REFERENCE_ARRAY_KEYS: tuple[str, ...] = ("M", "T", "phi", "O_i", "D")
REFERENCE_REQUIRED_KEYS: tuple[str, ...] = (
    "M",
    "T",
    "phi",
    "O_i",
    "D",
    "target_size",
)
# Real-valued maps must be floating; idler object field must be complex.
REFERENCE_REAL_KEYS: tuple[str, ...] = ("M", "T", "phi", "D")
REFERENCE_COMPLEX_KEYS: tuple[str, ...] = ("O_i",)


def reference_target_hw(ref: dict[str, Any]) -> tuple[int, int]:
    """Resolve ``target_size`` to ``(H, W)`` (int or length-2 shape)."""
    if "target_size" not in ref:
        raise ReferenceSchemaError("missing required key 'target_size'")
    ts = ref["target_size"]
    if isinstance(ts, (int, np.integer)):
        n = int(ts)
        if n < 1:
            raise ReferenceSchemaError(f"target_size must be >= 1, got {ts!r}")
        return n, n
    try:
        hw = tuple(int(x) for x in ts)
    except TypeError as exc:
        raise ReferenceSchemaError(
            f"target_size must be int or (H, W), got {type(ts).__name__}"
        ) from exc
    if len(hw) != 2 or hw[0] < 1 or hw[1] < 1:
        raise ReferenceSchemaError(f"target_size must be (H, W) with H,W >= 1, got {ts!r}")
    return int(hw[0]), int(hw[1])


def validate_reference_schema(ref: dict[str, Any]) -> tuple[int, int]:
    """Fail-closed reference schema before simulate / dataset write (V-P0-11).

    Requires ``M``, ``T``, ``phi``, ``O_i``, ``D``, ``target_size``. Each map
    must match ``target_size`` (no silent reshape). Real maps must be floating
    dtype; ``O_i`` must be complex or floating (not integer/bool). Arrays must
    be finite (no NaN/Inf) so metrics are not poisoned before physics evaluation.
    """
    if not isinstance(ref, dict):
        raise ReferenceSchemaError(f"reference must be a dict, got {type(ref).__name__}")

    for key in REFERENCE_REQUIRED_KEYS:
        if key not in ref or ref[key] is None:
            raise ReferenceSchemaError(f"missing required key {key!r}")

    expected = reference_target_hw(ref)
    for key in REFERENCE_ARRAY_KEYS:
        arr = np.asarray(ref[key])
        if arr.ndim != 2:
            raise ReferenceSchemaError(
                f"{key} must be 2-D, got shape {getattr(arr, 'shape', None)}"
            )
        if tuple(int(s) for s in arr.shape) != expected:
            raise ReferenceSchemaError(
                f"{key} shape mismatch: got {arr.shape}, expected {expected} "
                f"(target_size={ref['target_size']!r})"
            )
        if key in REFERENCE_REAL_KEYS:
            if not np.issubdtype(arr.dtype, np.floating):
                raise ReferenceSchemaError(
                    f"{key} dtype must be floating, got {arr.dtype}"
                )
        elif key in REFERENCE_COMPLEX_KEYS:
            # Physical O_i is complex; real floating is accepted only as a
            # degenerate amplitude-only stand-in used by some fixtures.
            if not (
                np.issubdtype(arr.dtype, np.complexfloating)
                or np.issubdtype(arr.dtype, np.floating)
            ):
                raise ReferenceSchemaError(
                    f"{key} dtype must be complex or floating, got {arr.dtype}"
                )
        if not np.isfinite(arr).all():
            raise ReferenceSchemaError(f"{key} contains NaN/Inf")

    # Optional companions must match if present.
    for key in ("I0", "I_plus", "I_minus"):
        if key not in ref or ref[key] is None:
            continue
        arr = np.asarray(ref[key])
        if tuple(int(s) for s in arr.shape) != expected:
            raise ReferenceSchemaError(
                f"{key} shape mismatch: got {arr.shape}, expected {expected}"
            )
        if not np.issubdtype(arr.dtype, np.floating):
            raise ReferenceSchemaError(
                f"{key} dtype must be floating, got {arr.dtype}"
            )
        if not np.isfinite(arr).all():
            raise ReferenceSchemaError(f"{key} contains NaN/Inf")

    return expected


def build_clean_reference(
    output_dir: str = "UPI_Synthetic/qiup_arch",
    target_size: int = 128,
    seed: int = 42,
    mnist_index: int = 2,
    *,
    beam_waist_px: float | None = None,
    object_beam_fill: float = 0.85,
    image_path: str | None = None,
    phase_only_slm: bool = True,
    object_preprocessing: str | None = None,
    phi_max_rad: float | None = None,
    setup: SetupParams | None = None,
) -> dict[str, Any]:
    """Load object maps for the twin forward model (V-P0-04 / V-P0-08 / V-P0-09).

    This is **not** an independent physics path. It only builds ``M``, ``T``,
    ``φ``, ``O_i`` (and a geometric envelope ``I0``). Ideal ``I±`` / ``D`` are
    produced by ``simulate.simulate_forward(..., detector_noise=False)`` using
    the same SLM / blur / gain / magnification as the noisy run.

    Default preprocessing is ``calibrated`` (physical units, no min-max).
    Always returns a schema-valid reference (V-P0-11): maps share ``M.shape``
    and ``target_size`` is that shape.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if setup is not None:
        if object_preprocessing is None:
            object_preprocessing = getattr(setup, "object_preprocessing", None)
        if phi_max_rad is None:
            phi_max_rad = float(getattr(setup, "slm_phi_max_rad", np.pi))
    mode = normalize_object_preprocessing(object_preprocessing)
    phi_hw = float(np.pi if phi_max_rad is None else phi_max_rad)
    setup_paths = {
        "phase": None if setup is None else setup.object_phase_map_path,
        "amplitude": None if setup is None else setup.object_amplitude_map_path,
        "intensity": None if setup is None else setup.object_intensity_map_path,
    }
    explicit_setup_maps = any(setup_paths.values())
    object_file_evidence: dict[str, Any] = {}
    if explicit_setup_maps:
        if image_path is not None:
            raise ValueError(
                "image_path cannot be combined with explicit setup object-map paths"
            )
        response = load_user_object_response(
            phase_map_path=setup_paths["phase"],
            amplitude_map_path=setup_paths["amplitude"],
            intensity_map_path=setup_paths["intensity"],
            target_shape=(int(target_size), int(target_size)),
        )
        T = response["T"]
        phi = response["phi"]
        O_i = response["O_i"]
        M = phi.copy()
        label = "user_defined_object"
        object_file_evidence = dict(response["file_evidence"])
    else:
        M, label = load_object_M(
            image_path=image_path,
            target_size=target_size,
            use_mnist_fallback=image_path is None,
            mnist_index=mnist_index,
            output_dir=output_dir,
            beam_waist_px=beam_waist_px,
            object_beam_fill=object_beam_fill,
            object_preprocessing=mode,
            phi_max_rad=phi_hw if mode == OBJECT_PREPROCESS_CALIBRATED else None,
        )
        T, phi, O_i = build_idler_object(
            M,
            phase_only_slm=phase_only_slm,
            phi_max=phi_hw,
            output_dir=output_dir,
            object_preprocessing=mode,
        )
    M = np.asarray(M, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    O_i = np.asarray(O_i, dtype=np.complex64 if np.iscomplexobj(O_i) else np.float32)
    if M.shape != T.shape or M.shape != phi.shape or M.shape != O_i.shape:
        raise ReferenceSchemaError(
            f"build_clean_reference map shapes diverge: "
            f"M{M.shape} T{T.shape} phi{phi.shape} O_i{O_i.shape}"
        )
    H, W = int(M.shape[0]), int(M.shape[1])
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    sig_env = float(beam_waist_px) / np.sqrt(2.0) if beam_waist_px else (min(H, W) / 4.0)
    I0 = np.exp(-((x - W / 2) ** 2 + (y - H / 2) ** 2) / (2 * sig_env ** 2)).astype(
        np.float32
    )
    # Placeholder arrays — real clean I±/D come from simulate_forward (same optics).
    z = np.zeros((H, W), dtype=np.float32)
    ref = {
        "M": M,
        "T": T,
        "phi": phi,
        "O_i": O_i,
        "optical": None,
        "I0": I0,
        "I_plus": z.copy(),
        "I_minus": z.copy(),
        "D": z.copy(),
        "label": label,
        "rng": rng,
        "seed": seed,
        # V-P0-11: target_size is the map shape (H, W), not a free int.
        "target_size": (H, W),
        "beam_waist_px": None if beam_waist_px is None else float(beam_waist_px),
        "object_beam_fill": float(object_beam_fill),
        "phase_only_slm": bool(phase_only_slm),
        "object_preprocessing": mode,
        "object_input_file_evidence": object_file_evidence,
        "object_path": (
            None
            if not object_file_evidence
            else [entry["path"] for entry in object_file_evidence.values()]
        ),
        "durable_object_reference": (
            None
            if not object_file_evidence
            else {"input_files": object_file_evidence}
        ),
        "phi_max_rad": phi_hw,
        "output_dir": output_dir,
        "clean_from_forward": True,
        "note": (
            "V-P0-04/08/09/11: object maps only; target_size=M.shape; "
            "calibrated keeps physical phase [rad]. "
            "Use simulate_forward(..., detector_noise=False) for clean I±/D."
        ),
    }
    validate_reference_schema(ref)
    return ref


# ---------------------------------------------------------------------------
# Physics helpers derived from latents
# ---------------------------------------------------------------------------
