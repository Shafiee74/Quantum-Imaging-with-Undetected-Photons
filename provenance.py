"""
QIUP provenance — central SHA-256 fingerprints (Comment 7.1 / V-P0-13).

Package label (Comment 8.2): Physics-Informed Reduced-Order SPDC-QIUP
Forward Simulator Prototype. Not a Verified/Validated Digital Twin.

This module is the **single source of truth** for SHA-256 fingerprints used by
replay, dataset manifests, and delivery evidence packs.

Hash surfaces (all SHA-256)
---------------------------
- configuration     → ``hash_config`` / ``hash_configuration``
- object            → ``hash_object`` / ``reference_object_hashes``
- source modules    → ``hash_source_modules``
- build script      → ``hash_build_script``
- single-file dist  → ``hash_single_file_distribution``
- saved arrays      → ``hash_array`` / ``hash_saved_arrays`` / ``hash_sim_arrays``

Comment 2.4 build-hash fixtures remain here
(``compute_single_file_build_hashes`` / ``verify_single_file_build_hashes``).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Release identity (Comment 7.2 — unambiguous version, not phase5_v1)
# Package/working-tree designation is v3.4.1; the reduced-order physics model
# lineage remains 3.3.1-remediated. Do not collapse these into one label.
# ---------------------------------------------------------------------------

HASH_ALGORITHM = "sha256"
PACKAGE_VERSION = "3.4.1"
RELEASE_VERSION = PACKAGE_VERSION
MODEL_FAMILY = "qiup_spdc_simulator"
PHYSICS_MODEL_VERSION = "3.3.1-remediated"
MODEL_VERSION = f"{MODEL_FAMILY}_v{PHYSICS_MODEL_VERSION}"
PHYSICS_POLICY_VERSION = "three_configuration_physics_team_v2"
# Comment 8.2 — approved package label until supervisor sign-off.
# Do NOT claim "Verified Digital Twin" or "Validated Digital Twin".
PACKAGE_DISPLAY_NAME = (
    "Physics-Informed Reduced-Order SPDC-QIUP Forward Simulator Prototype"
)
PACKAGE_LABEL = PACKAGE_DISPLAY_NAME  # Comment 8.2 alias
MODEL_STATUS = PACKAGE_DISPLAY_NAME
RELEASE_STATUS = "prototype"
RELEASE_CLASSIFICATION = "Prototype / Pre-validation QIUP Simulator"
CALIBRATION_STATUS = "NOT YET EXPERIMENTALLY CALIBRATED"
CALIBRATION_STATEMENT = (
    "Version 3.4.1 is a prototype/pre-validation physics-based QIUP simulator. "
    "The implemented physical model and software verification tests do not "
    "constitute experimental calibration or validation. Calibration against "
    "the actual laboratory pump, nonlinear crystals, optical relays, SLM, and "
    "detector is required before the system can be classified as a validated "
    "digital twin."
)
PROHIBITED_CLAIM_PHRASES: tuple[str, ...] = (
    "Verified Digital Twin",
    "Validated Digital Twin",
    "verified digital twin",
    "validated digital twin",
)
PROVENANCE_SCHEMA = "qiup_v_p0_13_c7"

# Comment 7.3 — mandatory fields on every run / sample metadata (top-level).
REQUIRED_EXECUTION_EVIDENCE_KEYS: tuple[str, ...] = (
    "version",
    "release_status",
    "seed",
    "rng_scheme",
    "config_hash",
    "object_hash",
    "gain_mode",
    "spdc_source_mode",
    "blur_model",
    "detector_mode",
    "timestamp",
    "source_hashes",
    "build_hash",
    "environment",
    "array_hashes",
)

# Canonical saved-array names for Comment 7.3 / V-P0-13 replay.
SAVED_ARRAY_HASH_KEYS: tuple[str, ...] = (
    "frames_plus",
    "frames_minus",
    "frames_D",
    "clean_D",
)
_ARRAY_HASH_ALIASES: dict[str, str] = {
    "Yp": "frames_plus",
    "Ym": "frames_minus",
    "D": "frames_D",
    "frames_plus": "frames_plus",
    "frames_minus": "frames_minus",
    "frames_D": "frames_D",
    "clean_D": "clean_D",
    "image": "frames_plus",
}

# Modular physics / orchestration sources (Comment 7.1).
SOURCE_MODULES: tuple[str, ...] = (
    "setup_params.py",
    "latents.py",
    "optics.py",
    "detector.py",
    "provenance.py",
    "simulate.py",
)
BUILD_SCRIPT_NAME = "build_single_file.py"
SINGLE_FILE_DISTRIBUTION_NAME = "qiup_digital_twin_full.py"

# ---------------------------------------------------------------------------
# Comment 2.4 — single-file / modular source build hashes
# ---------------------------------------------------------------------------

SINGLE_FILE_BUILD_HASH_RELATIVE = "configs/single_file_build_hashes.json"
SINGLE_FILE_BUILD_ARTIFACTS: tuple[str, ...] = (
    *SOURCE_MODULES,
    BUILD_SCRIPT_NAME,
    SINGLE_FILE_DISTRIBUTION_NAME,
)
_SINGLE_FILE_MODULE_ORDER: tuple[str, ...] = tuple(
    name[: -len(".py")] for name in SOURCE_MODULES
)


def package_root() -> Path:
    """Directory that holds the modular sources and generated single file."""
    return Path(__file__).resolve().parent


def hash_bytes(data: bytes, *, algo: str = HASH_ALGORITHM) -> str:
    """Hex digest of raw bytes (default SHA-256)."""
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def hash_text_normalized(text: str, *, algo: str = HASH_ALGORITHM) -> str:
    """SHA-256 of UTF-8 text with newlines normalized to ``\\n``."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hash_bytes(normalized.encode("utf-8"), algo=algo)


def hash_source_file(path: str | Path, *, algo: str = HASH_ALGORITHM) -> str:
    """SHA-256 of a source / generated text file (LF-normalized)."""
    return hash_text_normalized(Path(path).read_text(encoding="utf-8"), algo=algo)


def single_file_build_hash_path(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else package_root()
    return base / SINGLE_FILE_BUILD_HASH_RELATIVE


def compute_single_file_build_hashes(
    root: str | Path | None = None,
    *,
    generated_text: str | None = None,
) -> dict[str, Any]:
    """Calculate SHA-256 for every Comment-2.4 build artifact.

    Always includes::

        setup_params.py, latents.py, optics.py, detector.py,
        provenance.py, simulate.py, build_single_file.py,
        qiup_digital_twin_full.py

    ``generated_text`` may be supplied when the single-file content is known
    in-memory (build script) before / instead of reading the on-disk file.
    """
    base = Path(root) if root is not None else package_root()
    files: dict[str, str] = {}
    for name in SINGLE_FILE_BUILD_ARTIFACTS:
        if name == SINGLE_FILE_DISTRIBUTION_NAME and generated_text is not None:
            files[name] = hash_text_normalized(generated_text)
            continue
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Comment 2.4: missing build artifact for hashing: {path}"
            )
        files[name] = hash_source_file(path)

    module_files = {
        f"{m}.py": files[f"{m}.py"] for m in _SINGLE_FILE_MODULE_ORDER
    }
    generated = files[SINGLE_FILE_DISTRIBUTION_NAME]
    if generated_text is not None:
        line_count = len(generated_text.replace("\r\n", "\n").splitlines())
    else:
        line_count = len(
            (base / SINGLE_FILE_DISTRIBUTION_NAME)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .splitlines()
        )

    return {
        "finding": "single_file_distribution_P0",
        "algorithm": HASH_ALGORITHM,
        "release_version": RELEASE_VERSION,
        "newline_normalization": "lf",
        "modules": list(_SINGLE_FILE_MODULE_ORDER),
        "files": files,
        "module_files": module_files,
        "build_script": BUILD_SCRIPT_NAME,
        "build_script_sha256": files[BUILD_SCRIPT_NAME],
        "generated_file": SINGLE_FILE_DISTRIBUTION_NAME,
        "generated_file_sha256": generated,
        "generated_line_count": int(line_count),
        "note": (
            f"Hand-editing {SINGLE_FILE_DISTRIBUTION_NAME} is prohibited. "
            f"Edit modular sources, then run: python {BUILD_SCRIPT_NAME}. "
            "Hashes are calculated/stored via provenance.compute_single_file_build_hashes."
        ),
    }


def write_single_file_build_hashes(
    manifest: dict[str, Any] | None = None,
    root: str | Path | None = None,
    *,
    generated_text: str | None = None,
) -> Path:
    """Persist the Comment-2.4 hash fixture under ``configs/``."""
    base = Path(root) if root is not None else package_root()
    payload = (
        manifest
        if manifest is not None
        else compute_single_file_build_hashes(base, generated_text=generated_text)
    )
    path = single_file_build_hash_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def load_single_file_build_hashes(
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the recorded Comment-2.4 hash fixture (fail-closed if missing)."""
    path = single_file_build_hash_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Comment 2.4: missing single-file build hash fixture {path}. "
            f"Run: python {BUILD_SCRIPT_NAME}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Comment 2.4: hash fixture must be a mapping: {path}")
    files = data.get("files")
    if not isinstance(files, dict):
        files = {}
        for name, digest in (data.get("module_files") or {}).items():
            files[str(name)] = str(digest)
        if data.get("build_script_sha256"):
            files[BUILD_SCRIPT_NAME] = str(data["build_script_sha256"])
        if data.get("generated_file_sha256"):
            files[SINGLE_FILE_DISTRIBUTION_NAME] = str(data["generated_file_sha256"])
        data = dict(data)
        data["files"] = files
    missing = [n for n in SINGLE_FILE_BUILD_ARTIFACTS if n not in files]
    if missing:
        raise ValueError(
            "Comment 2.4: hash fixture missing artifacts: " + ", ".join(missing)
        )
    return data


def verify_single_file_build_hashes(
    root: str | Path | None = None,
    *,
    generated_text: str | None = None,
) -> dict[str, Any]:
    """Compare live artifact hashes to the recorded fixture; raise on drift."""
    recorded = load_single_file_build_hashes(root)
    live = compute_single_file_build_hashes(root, generated_text=generated_text)
    bad: list[str] = []
    for name in SINGLE_FILE_BUILD_ARTIFACTS:
        want = str(recorded["files"][name]).lower()
        got = str(live["files"][name]).lower()
        if want != got:
            bad.append(f"{name}: live={got} recorded={want}")
    if bad:
        raise ValueError(
            "Comment 2.4: single-file build hash mismatch:\n  " + "\n  ".join(bad)
        )
    return live


# ---------------------------------------------------------------------------
# Comment 7.1 — named SHA-256 surfaces
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha256": hash_array(obj),
        }
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing (sorted keys, compact separators)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=True
    )


def hash_config(setup: Any) -> str:
    """SHA-256 of a SetupParams (or mapping) canonical serialization."""
    if hasattr(setup, "to_dict") and callable(setup.to_dict):
        payload = setup.to_dict()
    elif isinstance(setup, dict):
        payload = dict(setup)
    else:
        raise TypeError(
            f"hash_config expects SetupParams or dict, got {type(setup).__name__}"
        )
    return hash_bytes(canonical_json(payload).encode("utf-8"))


def hash_configuration(setup: Any) -> str:
    """Comment 7.1 alias — configuration hash (SHA-256)."""
    return hash_config(setup)


def hash_array(arr: Any) -> str:
    """SHA-256 of array dtype + shape + contiguous raw bytes (object identity)."""
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def hash_object(ref_or_array: Any) -> str:
    """Comment 7.1 — object hash.

    Accepts a reference dict (hashes ``M`` when present) or a raw array/map.
    """
    if isinstance(ref_or_array, dict):
        if ref_or_array.get("M") is None:
            raise ValueError("hash_object(ref) requires ref['M']")
        return hash_array(ref_or_array["M"])
    return hash_array(ref_or_array)


def hash_source_modules(root: str | Path | None = None) -> dict[str, str]:
    """Comment 7.1 — SHA-256 for each modular source file."""
    base = Path(root) if root is not None else package_root()
    out: dict[str, str] = {}
    for name in SOURCE_MODULES:
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"Comment 7.1: missing source module {path}")
        out[name] = hash_source_file(path)
    return out


def hash_build_script(root: str | Path | None = None) -> str:
    """Comment 7.1 — SHA-256 of ``build_single_file.py``."""
    base = Path(root) if root is not None else package_root()
    path = base / BUILD_SCRIPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Comment 7.1: missing build script {path}")
    return hash_source_file(path)


def hash_single_file_distribution(
    root: str | Path | None = None,
    *,
    generated_text: str | None = None,
) -> str:
    """Comment 7.1 — SHA-256 of ``qiup_digital_twin_full.py``."""
    if generated_text is not None:
        return hash_text_normalized(generated_text)
    base = Path(root) if root is not None else package_root()
    path = base / SINGLE_FILE_DISTRIBUTION_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Comment 7.1: missing single-file distribution {path}"
        )
    return hash_source_file(path)


def hash_sim_arrays(
    *,
    frames_plus: Any,
    frames_minus: Any,
    frames_D: Any,
    clean_D: Any,
) -> dict[str, str]:
    """SHA-256 map for the primary twin arrays (V-P0-13 replay compare)."""
    return {
        "frames_plus": hash_array(frames_plus),
        "frames_minus": hash_array(frames_minus),
        "frames_D": hash_array(frames_D),
        "clean_D": hash_array(clean_D),
    }


def hash_saved_arrays(arrays: dict[str, Any]) -> dict[str, str]:
    """Comment 7.1 / 7.3 — SHA-256 for an arbitrary mapping of saved arrays.

    Recognized aliases (``Yp``/``Ym``/``D``/``image``) are normalized to the
    Comment-7.3 names ``frames_plus`` / ``frames_minus`` / ``frames_D`` /
    ``clean_D``.
    """
    out: dict[str, str] = {}
    for key, value in arrays.items():
        if value is None:
            continue
        name = _ARRAY_HASH_ALIASES.get(str(key), str(key))
        out[name] = hash_array(value)
    return out


def default_rng_scheme(
    *,
    streams: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Comment 7.3 — documented SeedSequence RNG contract (V-P0-12)."""
    stream_names = list(streams) if streams is not None else [
        "rng_optics",
        "rng_detector",
        "rng_plus",
        "rng_minus",
    ]
    return {
        "method": "numpy.random.SeedSequence.spawn",
        "mixing": "SeedSequence.spawn(optics,detector,plus,minus)",
        "streams": stream_names,
        "stream_roles": {
            "rng_optics": "optical path (pump jitter, phase noise, …)",
            "rng_detector": "fixed detector patterns (PRNU / hot mask)",
            "rng_plus": "plus-arm shot/EM/read noise",
            "rng_minus": "minus-arm shot/EM/read noise",
        },
    }


def flatten_execution_evidence(
    meta: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy Comment 7.3 fields from ``provenance`` onto top-level ``meta``.

    Mutates and returns ``meta``. Missing keys are left unset so callers can
    detect incomplete provenance.
    """
    prov = provenance if provenance is not None else dict(meta.get("provenance") or {})
    for key in REQUIRED_EXECUTION_EVIDENCE_KEYS:
        if key in prov:
            meta[key] = prov[key]
    # Keep compatibility aliases used by older consumers.
    if "version" in prov:
        meta["model_version"] = meta.get("model_version") or prov["version"]
    if "release_status" in prov:
        meta["release_status"] = prov["release_status"]
    return meta


def assert_execution_evidence(meta_or_prov: dict[str, Any]) -> list[str]:
    """Return missing Comment-7.3 keys (empty list ⇒ complete)."""
    return [k for k in REQUIRED_EXECUTION_EVIDENCE_KEYS if k not in meta_or_prov]


def build_configuration_evidence(
    setup: Any,
    *,
    state: Any | None = None,
    operating_point: dict[str, Any] | None = None,
    resolved_forward: dict[str, Any] | None = None,
    last_opt: dict[str, Any] | None = None,
    physical_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Comment 7.4 — requested vs resolved configuration + derived quantities.

    Reviewers must see what was *asked for*, what the forward path *actually
    used*, and the derived physical numbers (nbar, r, σ, …).
    """
    from setup_params import (
        apparatus_records_from_setup,
        camera_plane_geometry_from_setup,
        normalize_detector_mode,
        normalize_spdc_source_mode,
        pump_powers_from_setup,
        pump_transverse_profile_evidence,
        qpm_configuration_evidence,
        resolve_architecture_profile,
        setup_selection_info,
        spdc_correlation_psf_status,
    )

    opt = dict(last_opt or {})
    fwd = dict(resolved_forward or {})
    report = dict(physical_report or {})
    if not report and state is not None:
        report = dict(getattr(state, "physical_report", None) or {})

    pump_override = None
    if state is not None and getattr(state, "pump_power_mW", None) is not None:
        pump_override = float(state.pump_power_mW)

    try:
        powers = pump_powers_from_setup(setup, pump_override)
    except Exception:
        powers = {
            "pump_power_laser_mW": getattr(setup, "pump_power_laser_mW", None),
            "pump_power_NL1_mW": None,
            "pump_power_NL2_mW": None,
        }

    op = dict(operating_point or {})
    if not op and state is not None and hasattr(state, "spdc_operating_point"):
        try:
            op = dict(state.spdc_operating_point())
        except Exception:
            op = {}

    requested: dict[str, Any] = {
        "physics_policy_id": getattr(setup, "physics_policy_id", None),
        "architecture_profile": resolve_architecture_profile(setup).to_dict(),
        "setup_selection": setup_selection_info(setup),
        "setup_id": getattr(setup, "setup_id", None),
        "model_version": getattr(setup, "model_version", None),
        "pump_power_laser_mW": getattr(setup, "pump_power_laser_mW", None),
        "pump_power_mW_state": pump_override,
        "pump_pre_crystal_transmission": getattr(
            setup, "pump_pre_crystal_transmission", None
        ),
        "pump_return_transmission": getattr(setup, "pump_return_transmission", None),
        "pump_to_NL2_transmission": getattr(setup, "pump_to_NL2_transmission", None),
        "pump_bs_reflectance": getattr(setup, "pump_bs_reflectance", None),
        "use_independent_pump_power_NL2": bool(
            getattr(setup, "use_independent_pump_power_NL2", False)
        ),
        "pump_power_NL2_override_mW": getattr(
            setup, "pump_power_NL2_override_mW", None
        ),
        "spdc_gain_mode": getattr(setup, "spdc_gain_mode", None),
        "spdc_source_mode": getattr(setup, "spdc_source_mode", None),
        "fixed_source_input": getattr(setup, "fixed_source_input", None),
        "measured_power_W": getattr(setup, "measured_power_W", None),
        "measured_power_wavelength_nm": getattr(
            setup, "measured_power_wavelength_nm", None
        ),
        "measured_power_to_nbar_calibration_id": getattr(
            setup, "measured_power_to_nbar_calibration_id", None
        ),
        "mean_photons_per_mode_requested": (
            None
            if state is None
            else getattr(state, "mean_photons_per_mode", None)
        ),
        "spdc_gain_r_requested": (
            None if state is None else getattr(state, "spdc_gain_r", None)
        ),
        "crystal_length_mm": getattr(setup, "crystal_length_mm", None),
        "crystal_length_NL2_mm": getattr(setup, "crystal_length_NL2_mm", None),
        "crystal_matching_mode": getattr(setup, "crystal_matching_mode", None),
        "spdc_type": getattr(setup, "spdc_type", None)
        or getattr(setup, "qpm_type", None),
        "wavelengths_nm": {
            "pump": getattr(setup, "lambda_p_nm", None),
            "signal": getattr(setup, "lambda_s_nm", None),
            "idler": getattr(setup, "lambda_i_nm", None),
        },
        "object_device": getattr(setup, "object_device", None),
        "object_mode": getattr(setup, "object_mode", None),
        "object_input_paths": {
            "phase": getattr(setup, "object_phase_map_path", None),
            "amplitude": getattr(setup, "object_amplitude_map_path", None),
            "intensity": getattr(setup, "object_intensity_map_path", None),
        },
        "camera_filter": {
            "preset": getattr(setup, "camera_filter_preset", None),
            "center_nm": getattr(setup, "camera_filter_center_nm", None),
            "fwhm_nm": getattr(setup, "camera_filter_fwhm_nm", None),
            "model": getattr(setup, "camera_filter_model", None),
        },
        "blur_model": getattr(setup, "blur_model", None),
        "detector_mode": getattr(setup, "detector_mode", None),
        "imaging_mode": getattr(setup, "imaging_mode", None),
        "magnification_requested": getattr(
            setup, "magnification_requested", None
        ),
        "magnification": getattr(setup, "magnification", None),
        "use_physical_spdc_gain": bool(
            getattr(setup, "use_physical_spdc_gain", False)
        ),
    }

    source_mode = getattr(setup, "spdc_source_mode", None)
    try:
        source_mode = normalize_spdc_source_mode(source_mode)
    except Exception:
        pass
    det_mode = getattr(setup, "detector_mode", None)
    try:
        det_mode = normalize_detector_mode(det_mode)
    except Exception:
        pass

    resolved: dict[str, Any] = {
        "physics_policy_id": getattr(setup, "physics_policy_id", None),
        "pump_power_laser_mW": powers.get("pump_power_laser_mW"),
        "pump_power_laser_mean_mW": powers.get("pump_power_laser_mean_mW"),
        "pump_power_NL1_mW": powers.get("pump_power_NL1_mW"),
        "pump_power_NL2_mW": powers.get("pump_power_NL2_mW"),
        "pump_fluctuation_scale": powers.get("pump_fluctuation_scale"),
        "pump_balance": {
            key: powers.get(key) for key in (
                "pump_balance_mode", "P_target_1_mW", "P_target_2_mW",
                "P_actual_1_mW", "P_actual_2_mW", "delta_P", "abs_delta_P",
                "hidden_equalization_applied",
            )
        },
        "spdc_gain_mode": getattr(setup, "spdc_gain_mode", None),
        "spdc_source_mode": source_mode,
        "blur_model": opt.get("blur_model", getattr(setup, "blur_model", None)),
        "detector_mode": det_mode,
        "imaging_mode": opt.get("imaging_mode", getattr(setup, "imaging_mode", None)),
        "correlation_type": opt.get("correlation_type"),
        "w_px": fwd.get("w_px"),
        "sigma_spdc_px": fwd.get("sigma_spdc_px"),
        "sigma_diffraction_px": fwd.get(
            "sigma_diffraction_px", opt.get("sigma_diffraction_px")
        ),
        "defocus_span_px": fwd.get("defocus_span_px"),
        "V0": fwd.get("V0"),
        "crystal_length_mm": getattr(setup, "crystal_length_mm", None),
        "qpm_interaction_1": qpm_configuration_evidence(setup, interaction=1),
        "qpm_interaction_2": qpm_configuration_evidence(setup, interaction=2),
        "signal_filter": opt.get("signal_filter"),
        "resolved_forward": fwd or None,
        "magnification": getattr(setup, "magnification", None),
        "M_QIUP": getattr(setup, "magnification", None),
        "M_QIUP_formula": "F2*lambda_s/(F1*lambda_i)",
        "magnification_synced_from_M_QIUP": True,
        "magnification_resolution_reason": (
            "Legacy YAML/API magnification is a mirror of M_QIUP; "
            "requested magnification_requested is preserved for audit while "
            "the forward path uses M_QIUP = F2*lambda_s/(F1*lambda_i)."
        ),
        "two_crystal_pump_topology": getattr(
            setup, "two_crystal_pump_topology", None
        ),
        "pump_transverse_profile": pump_transverse_profile_evidence(setup),
        "camera_plane_geometry": camera_plane_geometry_from_setup(setup),
        "apparatus_gpq": apparatus_records_from_setup(setup),
        "spdc_correlation_psf": spdc_correlation_psf_status(),
    }

    nbar1 = op.get("mean_photons_per_mode_NL1", op.get("mean_photons_per_mode"))
    nbar2 = op.get("mean_photons_per_mode_NL2", nbar1)
    derived: dict[str, Any] = {
        "nbar_NL1": None if nbar1 is None else float(nbar1),
        "nbar_NL2": None if nbar2 is None else float(nbar2),
        "spdc_gain_r_NL1": op.get("spdc_gain_r_NL1", op.get("gain_r")),
        "spdc_gain_r_NL2": op.get("spdc_gain_r_NL2"),
        "k_spdc_per_sqrt_W_NL1": op.get("k_spdc_per_sqrt_W_NL1"),
        "k_spdc_per_sqrt_W_NL2": op.get("k_spdc_per_sqrt_W_NL2"),
        "regime_NL1": op.get("regime_NL1", op.get("regime")),
        "regime_NL2": op.get("regime_NL2"),
        "sigma_total_px": opt.get("sigma_total_px"),
        "sig_spdc": opt.get("sig_spdc"),
        "sig_opt": opt.get("sig_opt"),
        "eta_overlap": opt.get("eta_overlap"),
        "delta_x_px": opt.get("delta_x_px"),
        "delta_y_px": opt.get("delta_y_px"),
        "spdc_operating_point": op or None,
        "source_input_type": op.get("source_input_type"),
        "photon_rate_per_s": op.get("photon_rate_per_s"),
        "source_conversion_method": op.get("source_conversion_method"),
        "source_input_calibration_id": op.get("source_input_calibration_id"),
        "physical_report_keys": sorted(report.keys()) if report else [],
    }
    # Keep a compact forward-derived snapshot when present.
    if fwd.get("derived"):
        derived["resolved_forward_derived"] = dict(fwd["derived"])

    return {
        "finding": "comment_7_4_resolved_configuration",
        "requested_configuration": requested,
        "resolved_configuration": resolved,
        "derived_physical_quantities": derived,
        # Compact reviewer-facing summary (Comment 7.4 example shape).
        "pump_power_laser_mW": resolved.get("pump_power_laser_mW"),
        "pump_power_NL1_mW": resolved.get("pump_power_NL1_mW"),
        "pump_power_NL2_mW": resolved.get("pump_power_NL2_mW"),
        "nbar_NL1": derived.get("nbar_NL1"),
        "nbar_NL2": derived.get("nbar_NL2"),
    }


def build_run_provenance(
    *,
    seed: int,
    setup: Any,
    ref: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    include_package_hashes: bool = False,
    arrays: dict[str, Any] | None = None,
    rng_streams: tuple[str, ...] | list[str] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the mandatory provenance block (V-P0-13 / Comment 7.3).

    Fields required for AI-dataset audit / Comment 7.3:
      version, release_status, seed, rng_scheme, config_hash, object_hash,
      gain_mode, spdc_source_mode, blur_model, detector_mode, timestamp,
      source_hashes, build_hash, environment, array_hashes.

    Set ``include_package_hashes=True`` for evidence packs so the full
    package hash set (incl. single-file) is also embedded.
    """
    from setup_params import resolve_architecture_profile, setup_selection_info
    architecture_profile = resolve_architecture_profile(setup)
    blur = getattr(setup, "blur_model", None)
    gain = getattr(setup, "spdc_gain_mode", None)
    det = getattr(setup, "detector_mode", None)
    source_mode = getattr(setup, "spdc_source_mode", None)
    obj_hash = None
    if ref is not None and ref.get("M") is not None:
        obj_hash = hash_object(ref)

    model_ver = getattr(setup, "model_version", None) or MODEL_VERSION
    base = Path(root) if root is not None else package_root()
    source_hashes = hash_source_modules(base)
    build_hash = hash_build_script(base)
    array_hashes = hash_saved_arrays(arrays) if arrays else {}

    meta: dict[str, Any] = {
        # Comment 7.3 required surface ---------------------------------
        "version": model_ver,
        "release_status": RELEASE_STATUS,
        "seed": int(seed),
        "rng_scheme": default_rng_scheme(streams=rng_streams),
        "config_hash": hash_configuration(setup),
        "object_hash": obj_hash,
        "gain_mode": gain,
        "spdc_source_mode": source_mode,
        "blur_model": blur,
        "detector_mode": det,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes,
        "build_hash": build_hash,
        "environment": get_environment(),
        "array_hashes": array_hashes,
        # Extended / compatibility fields ------------------------------
        "spdc_gain_mode": gain,
        "physics_policy_id": getattr(setup, "physics_policy_id", None),
        "spdc_type": getattr(setup, "spdc_type", None)
        or getattr(setup, "qpm_type", None),
        "crystal_matching_mode": getattr(setup, "crystal_matching_mode", None),
        "object_device": getattr(setup, "object_device", None),
        "object_mode": getattr(setup, "object_mode", None),
        "camera_filter": {
            "preset": getattr(setup, "camera_filter_preset", None),
            "center_nm": getattr(setup, "camera_filter_center_nm", None),
            "fwhm_nm": getattr(setup, "camera_filter_fwhm_nm", None),
            "model": getattr(setup, "camera_filter_model", None),
        },
        "object_preprocessing": getattr(setup, "object_preprocessing", None),
        "imaging_mode": getattr(setup, "imaging_mode", None),
        "architecture_profile": architecture_profile.to_dict(),
        "setup_selection": setup_selection_info(setup),
        "reference_hashes": reference_object_hashes(ref),
        "setup_id": getattr(setup, "setup_id", None),
        "model_version": model_ver,
        "model_status": MODEL_STATUS,
        "package_version": PACKAGE_VERSION,
        "release_version": RELEASE_VERSION,
        "release_classification": RELEASE_CLASSIFICATION,
        "calibration_status": CALIBRATION_STATUS,
        "calibration_statement": CALIBRATION_STATEMENT,
        "model_family": MODEL_FAMILY,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "physics_policy_version": PHYSICS_POLICY_VERSION,
        "git_commit": None,
        "git_commit_status": "unavailable",
        "physics_tier": getattr(setup, "physics_tier", None),
        "hash_algorithm": HASH_ALGORITHM,
        "provenance_schema": PROVENANCE_SCHEMA,
        "verified_digital_twin": False,
    }
    if include_package_hashes:
        meta["package_hashes"] = compute_package_hash_set(
            base, setup=setup, ref=ref, arrays=arrays
        )
    if extra:
        meta.update(extra)
        # Re-assert Comment 7.3 keys if extra overwrote aliases.
        if "config_hash" in extra:
            meta["config_hash"] = extra["config_hash"]
    return meta


def compare_array_hashes(
    expected: dict[str, str],
    actual: dict[str, str],
) -> list[str]:
    """Return mismatch descriptions; empty list means bit-exact hash match."""
    bad: list[str] = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        if key not in expected:
            bad.append(f"{key}: missing in expected")
            continue
        if key not in actual:
            bad.append(f"{key}: missing in actual")
            continue
        if str(expected[key]).lower() != str(actual[key]).lower():
            bad.append(f"{key}: expected={expected[key]} actual={actual[key]}")
    return bad


def get_environment() -> dict[str, Any]:
    """Runtime environment fingerprint for replay / evidence packs."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        env["numpy"] = np.__version__
    except Exception:
        env["numpy"] = None
    try:
        import scipy

        env["scipy"] = scipy.__version__
    except Exception:
        env["scipy"] = None
    try:
        from PIL import Image

        env["pillow"] = getattr(Image, "__version__", None) or getattr(
            sys.modules.get("PIL"), "__version__", None
        )
    except Exception:
        env["pillow"] = None
    return env


def reference_object_hashes(ref: dict[str, Any] | None) -> dict[str, str]:
    """Per-map hashes for the SLM / object reference."""
    if not ref:
        return {}
    out: dict[str, str] = {}
    for key in ("M", "T", "phi", "O_i", "D", "I0"):
        if key not in ref or ref[key] is None:
            continue
        try:
            out[key] = hash_array(ref[key])
        except Exception:
            continue
    return out


# Comment 7.5 — official evidence must keep exact object maps (or a durable ref).
OBJECT_EVIDENCE_REQUIRED_MAPS: tuple[str, ...] = ("T", "phi")
OBJECT_EVIDENCE_OFFICIAL_MAPS: tuple[str, ...] = ("T", "phi", "M")
OBJECT_EVIDENCE_OPTIONAL_MAPS: tuple[str, ...] = ("O_i", "D", "I0")
OBJECT_EVIDENCE_SUBDIR = "object"
OBJECT_EVIDENCE_MANIFEST_NAME = "object_evidence.json"


def build_object_evidence(
    ref: dict[str, Any] | None,
    *,
    durable_object_reference: dict[str, Any] | str | None = None,
    relative_dir: str = OBJECT_EVIDENCE_SUBDIR,
    include_optional: bool = True,
) -> dict[str, Any]:
    """Comment 7.5 — object evidence block (hashes + file map / durable ref).

    Official evidence prefers on-disk arrays (``T.npy``, ``phi.npy``, …) plus
    SHA-256. A durable reference may be supplied instead / in addition.
    """
    if not ref:
        raise ValueError("Comment 7.5: ref is required for object evidence")
    missing = [k for k in OBJECT_EVIDENCE_REQUIRED_MAPS if ref.get(k) is None]
    if missing and durable_object_reference is None:
        raise ValueError(
            "Comment 7.5: missing object maps "
            f"{missing} and no durable_object_reference supplied"
        )

    hashes = reference_object_hashes(ref)
    files: dict[str, str] = {}
    for key in OBJECT_EVIDENCE_OFFICIAL_MAPS:
        if ref.get(key) is not None:
            files[key] = f"{relative_dir}/{key}.npy".replace("\\", "/")
    if include_optional:
        for key in OBJECT_EVIDENCE_OPTIONAL_MAPS:
            if ref.get(key) is not None:
                files[key] = f"{relative_dir}/{key}.npy".replace("\\", "/")

    obj_hash = None
    if ref.get("M") is not None:
        obj_hash = hash_object(ref)

    durable = durable_object_reference
    if durable is None:
        # Prefer explicit ref metadata when present.
        durable = ref.get("durable_object_reference") or ref.get("object_id")
        if durable is None and ref.get("object_path"):
            durable = {"path": ref.get("object_path")}

    storage = "arrays" if files else "durable_reference"
    if files and durable is not None:
        storage = "arrays_and_durable_reference"

    return {
        "finding": "comment_7_5_object_evidence",
        "algorithm": HASH_ALGORITHM,
        "storage": storage,
        "object_hash": obj_hash,
        "hashes": hashes,
        "files": files,
        "required_maps": list(OBJECT_EVIDENCE_REQUIRED_MAPS),
        "durable_object_reference": durable,
        "target_size": ref.get("target_size"),
        "ref_seed": ref.get("seed"),
    }


def save_object_evidence(
    ref: dict[str, Any],
    out_dir: str | Path,
    *,
    subdir: str = OBJECT_EVIDENCE_SUBDIR,
    include_optional: bool = True,
    durable_object_reference: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Comment 7.5 — write object maps under ``out_dir/object/`` and a manifest.

    Always writes ``T.npy`` and ``phi.npy`` when present (official path). Also
    writes ``M.npy`` and optional maps when available. Returns the evidence
    block after verifying on-disk SHA-256 digests.
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    obj_dir = base / subdir
    obj_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_object_evidence(
        ref,
        durable_object_reference=durable_object_reference,
        relative_dir=subdir,
        include_optional=include_optional,
    )

    written: dict[str, str] = {}
    for key, rel in evidence["files"].items():
        arr = ref.get(key)
        if arr is None:
            continue
        path = base / Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(arr))
        # Verify bit-identity of the saved bytes via hash_array reload.
        loaded = np.load(path)
        digest = hash_array(loaded)
        expected = evidence["hashes"].get(key)
        if expected is not None and digest != expected:
            raise ValueError(
                f"Comment 7.5: saved {rel} hash mismatch "
                f"(file={digest}, expected={expected})"
            )
        written[key] = str(Path(rel).as_posix())

    evidence["files"] = written
    evidence["object_dir"] = str(Path(subdir).as_posix())
    manifest_path = obj_dir / OBJECT_EVIDENCE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    evidence["manifest_file"] = str(
        (Path(subdir) / OBJECT_EVIDENCE_MANIFEST_NAME).as_posix()
    )
    return evidence


def verify_object_evidence(
    out_dir: str | Path,
    *,
    expected: dict[str, Any] | None = None,
    subdir: str = OBJECT_EVIDENCE_SUBDIR,
) -> list[str]:
    """Return mismatch descriptions for saved object evidence (empty ⇒ OK)."""
    base = Path(out_dir)
    obj_dir = base / subdir
    manifest_path = obj_dir / OBJECT_EVIDENCE_MANIFEST_NAME
    bad: list[str] = []
    if not manifest_path.is_file():
        return [f"missing {manifest_path.as_posix()}"]
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected is not None:
        for key in ("object_hash",):
            if expected.get(key) and recorded.get(key) != expected.get(key):
                bad.append(
                    f"{key}: expected={expected.get(key)} actual={recorded.get(key)}"
                )
    files = dict(recorded.get("files") or {})
    hashes = dict(recorded.get("hashes") or {})
    for key in OBJECT_EVIDENCE_REQUIRED_MAPS:
        rel = files.get(key)
        if not rel:
            bad.append(f"{key}: missing file entry")
            continue
        path = base / rel
        if not path.is_file():
            bad.append(f"{key}: missing file {path.as_posix()}")
            continue
        digest = hash_array(np.load(path))
        expected_h = hashes.get(key)
        if expected_h is not None and digest != expected_h:
            bad.append(f"{key}: hash mismatch file={digest} expected={expected_h}")
    return bad


def load_object_ref_from_evidence(
    base_dir: str | Path,
    object_evidence: dict[str, Any] | None = None,
    *,
    subdir: str = OBJECT_EVIDENCE_SUBDIR,
) -> dict[str, Any]:
    """Comment 7.6 — rebuild the object ``ref`` dict from saved evidence maps.

    Loads ``T.npy`` / ``phi.npy`` / ``M.npy`` (and optional maps) from
    ``base_dir/object/`` (or paths listed in ``object_evidence['files']``).
    Verifies SHA-256 against the recorded hashes when present.
    """
    base = Path(base_dir)
    evidence = dict(object_evidence or {})
    if not evidence:
        manifest_path = base / subdir / OBJECT_EVIDENCE_MANIFEST_NAME
        if manifest_path.is_file():
            evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = dict(evidence.get("files") or {})
    if not files:
        # Fall back to conventional official filenames.
        for key in (*OBJECT_EVIDENCE_OFFICIAL_MAPS, *OBJECT_EVIDENCE_OPTIONAL_MAPS):
            candidate = base / subdir / f"{key}.npy"
            if candidate.is_file():
                files[key] = str((Path(subdir) / f"{key}.npy").as_posix())
    if not files:
        durable = evidence.get("durable_object_reference")
        raise FileNotFoundError(
            "Comment 7.6: no object map files under "
            f"{(base / subdir).as_posix()} and no loadable durable reference "
            f"(durable={durable!r})"
        )

    ref: dict[str, Any] = {}
    hashes = dict(evidence.get("hashes") or {})
    for key, rel in files.items():
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"Comment 7.6: object map missing: {path.as_posix()}"
            )
        arr = np.load(path)
        expected = hashes.get(key)
        if expected is not None:
            digest = hash_array(arr)
            if digest != expected:
                raise ValueError(
                    f"Comment 7.6: {key} hash mismatch "
                    f"(file={digest}, expected={expected})"
                )
        ref[key] = arr
    if evidence.get("target_size") is not None:
        ref["target_size"] = evidence["target_size"]
    elif ref.get("M") is not None:
        ref["target_size"] = int(np.asarray(ref["M"]).shape[0])
    if evidence.get("ref_seed") is not None:
        ref["seed"] = evidence["ref_seed"]
    if evidence.get("durable_object_reference") is not None:
        ref["durable_object_reference"] = evidence["durable_object_reference"]
    return ref


def compute_package_hash_set(
    root: str | Path | None = None,
    *,
    generated_text: str | None = None,
    setup: Any | None = None,
    ref: dict[str, Any] | None = None,
    arrays: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Comment 7.1 — complete package / evidence hash set (SHA-256).

    Always includes source modules, build script, and single-file distribution.
    Optionally includes configuration, object, and saved-array hashes.
    """
    base = Path(root) if root is not None else package_root()
    source = hash_source_modules(base)
    build = hash_build_script(base)
    single = hash_single_file_distribution(base, generated_text=generated_text)
    package: dict[str, Any] = {
        "finding": "comment_7_1_central_provenance",
        "algorithm": HASH_ALGORITHM,
        "package_version": PACKAGE_VERSION,
        "release_version": RELEASE_VERSION,
        "model_family": MODEL_FAMILY,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "physics_policy_version": PHYSICS_POLICY_VERSION,
        "model_status": MODEL_STATUS,
        "release_classification": RELEASE_CLASSIFICATION,
        "calibration_status": CALIBRATION_STATUS,
        "git_commit": None,
        "git_commit_status": "unavailable",
        "source_modules": source,
        "build_script": BUILD_SCRIPT_NAME,
        "build_script_sha256": build,
        "single_file_distribution": SINGLE_FILE_DISTRIBUTION_NAME,
        "single_file_sha256": single,
        "package_hash": hash_bytes(
            canonical_json(
                {
                    "source_modules": source,
                    "build_script_sha256": build,
                    "single_file_sha256": single,
                    "package_version": PACKAGE_VERSION,
                    "physics_model_version": PHYSICS_MODEL_VERSION,
                }
            ).encode("utf-8")
        ),
    }
    if setup is not None:
        package["config_hash"] = hash_configuration(setup)
    if ref is not None and ref.get("M") is not None:
        package["object_hash"] = hash_object(ref)
        package["reference_hashes"] = reference_object_hashes(ref)
    if arrays:
        package["array_hashes"] = hash_saved_arrays(arrays)
    return package
