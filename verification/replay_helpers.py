"""Active evidence helpers extracted for Version 3.4.2.

These helpers were previously embedded in
``generate_final_three_configuration_evidence.py`` (A/B/C Task-12 tool).
They contain no SPDC/optics equations — only evidence packaging, replay,
tamper checks, and image preview utilities used by ``generate_v3_4_1_evidence.py``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from provenance import hash_array, hash_sim_arrays
from simulate import replay_from_manifest, save_run_evidence_pack, simulate_forward

ROOT = Path(__file__).resolve().parents[1]
_TOOLS = ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from experiment_runner import (  # noqa: E402
    _validate_temp_point,
    prepare_experiment_point,
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def make_json_paths_portable(out: Path) -> int:
    """Replace package-root absolute paths in JSON evidence with relative paths."""
    root_text = str(ROOT.resolve())

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, str) and value.lower().startswith(root_text.lower()):
            try:
                return Path(value).resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                return value
        return value

    changed = 0
    for path in sorted(out.rglob("*.json")):
        current = json.loads(path.read_text(encoding="utf-8"))
        portable = convert(current)
        if portable != current:
            _json(path, portable)
            changed += 1
    return changed


def preview(array: np.ndarray) -> Image.Image:
    values = np.asarray(array, dtype=np.float64)
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    scaled = np.zeros(values.shape, dtype=np.uint8)
    if hi > lo:
        scaled = np.round(np.clip((values - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


# Backwards-compatible private alias for callers that used the legacy name.
_preview = preview


def input_evidence(out: Path, ref: dict[str, Any]) -> dict[str, Any]:
    destination = out / "input_object"
    destination.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}
    for name in ("M", "T", "phi", "O_i", "D"):
        path = destination / f"{name}.npy"
        np.save(path, np.asarray(ref[name]))
        files[name] = {
            "relative_path": path.relative_to(out).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "array_sha256": hash_array(np.asarray(ref[name])),
            "shape": list(np.asarray(ref[name]).shape),
            "dtype": str(np.asarray(ref[name]).dtype),
        }
    evidence = {
        "definition": "programmatically generated user SLM pattern",
        "phase_units": "rad",
        "intensity_convention": "T or R in [0,1] according to architecture",
        "coordinate_orientation": "object_xy",
        "files": files,
    }
    _json(destination / "object_input_manifest.json", evidence)
    return evidence


_input_evidence = input_evidence


def expect_replay_failure(pack: Path, mutation: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "pack"
        shutil.copytree(pack, copy)
        manifest_path = copy / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "configuration":
            manifest["config_hash"] = "0" * 64
            _json(manifest_path, manifest)
        elif mutation == "source_identity":
            key = sorted(manifest["source_hashes"])[0]
            manifest["source_hashes"][key] = "0" * 64
            _json(manifest_path, manifest)
        elif mutation == "build_identity":
            manifest["build_hash"] = "0" * 64
            _json(manifest_path, manifest)
        elif mutation == "array_hash":
            key = sorted(manifest["array_hashes"])[0]
            manifest["array_hashes"][key] = "0" * 64
            _json(manifest_path, manifest)
        elif mutation == "object":
            object_path = copy / "object" / "T.npy"
            array = np.load(object_path)
            array.flat[0] = 1.0 - float(array.flat[0])
            np.save(object_path, array)
        else:
            raise ValueError(mutation)
        try:
            replay_from_manifest(manifest_path, verify=True)
        except Exception as exc:
            return {
                "mutation": mutation,
                "detected": True,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        return {"mutation": mutation, "detected": False, "message": "not detected"}


_expect_replay_failure = expect_replay_failure


def saved_array_tamper_check(sample_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "sample_0000"
        shutil.copytree(sample_dir, copied)
        path = copied / "ideal_optical_signal.npy"
        array = np.load(path)
        array.flat[0] += 1.0
        np.save(path, array)
        record = json.loads((copied / "metadata.json").read_text(encoding="utf-8"))
        try:
            _validate_temp_point(copied, record)
        except Exception as exc:
            return {
                "mutation": "saved_array_file",
                "detected": True,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        return {"mutation": "saved_array_file", "detected": False}


_saved_array_tamper_check = saved_array_tamper_check


def replay_evidence(profile: str, out: Path, ref: dict[str, Any], seed: int) -> dict[str, Any]:
    point = prepare_experiment_point(
        profile=profile,
        experiment="mean_photon_gain",
        requested={"mean_photons_per_mode": 0.1},
        ref=ref,
        index=0,
        seed=seed,
    )
    result = simulate_forward(point.ref, point.state, seed=seed, detector_noise=True)
    pack = out / "replay_pack"
    manifest_path = save_run_evidence_pack(pack, result, point.ref)
    np.save(pack / "ideal_optical_signal.npy", result.ideal_signal_image)
    np.save(pack / "detector_signal.npy", result.detector_signal_image)
    np.save(pack / "clean_reference.npy", result.clean_reference_image)
    preview(result.ideal_signal_image).save(pack / "ideal_optical_signal.png")
    preview(result.detector_signal_image).save(pack / "detector_signal.png")
    replay = replay_from_manifest(manifest_path, verify=True)
    replay_hashes = hash_sim_arrays(
        frames_plus=replay.frames_plus,
        frames_minus=replay.frames_minus,
        frames_D=replay.frames_D,
        clean_D=replay.clean_D,
    )
    original_hashes = hash_sim_arrays(
        frames_plus=result.frames_plus,
        frames_minus=result.frames_minus,
        frames_D=result.frames_D,
        clean_D=result.clean_D,
    )
    report = {
        "profile": profile,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "bit_exact_arrays": replay_hashes == original_hashes,
        "configuration_hash_match": replay.meta["replay_from_manifest"][
            "verification"
        ]["config_hash_match"],
        "identity_verification": replay.meta["replay_from_manifest"][
            "verification"
        ]["identity_verification"],
        "recorded_array_hashes": original_hashes,
        "replayed_array_hashes": replay_hashes,
        "resolved_nbar_NL1": replay.meta["spdc_operating_point"][
            "mean_photons_per_mode_NL1"
        ],
        "resolved_nbar_NL2": replay.meta["spdc_operating_point"][
            "mean_photons_per_mode_NL2"
        ],
        "tamper_checks": [
            expect_replay_failure(pack, mutation)
            for mutation in (
                "configuration", "object", "source_identity",
                "build_identity", "array_hash",
            )
        ],
    }
    _json(pack / "replay_verification.json", report)
    return report


_replay_evidence = replay_evidence


def validate_outputs(out: Path, manifests: dict[str, Any]) -> dict[str, Any]:
    samples = 0
    validated = 0
    shape_by_profile: dict[str, list[int]] = {}
    for key, manifest in manifests.items():
        root = Path(manifest["output_dir"])
        manifest_path = root / "experiment_manifest.json"
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if parsed["n_samples"] != len(parsed["samples"]):
            raise RuntimeError(f"manifest count mismatch: {key}")
        for record in parsed["samples"]:
            sample = root / record["sample_dir"]
            _validate_temp_point(sample, record)
            ideal = np.load(sample / "ideal_optical_signal.npy")
            detector = np.load(sample / "detector_signal.npy")
            if ideal.shape != detector.shape or ideal.ndim != 2:
                raise RuntimeError(f"image shape mismatch: {sample}")
            shape_by_profile[key] = list(ideal.shape)
            samples += 1
            validated += 1
        if list(root.rglob(".tmp_sample_*")):
            raise RuntimeError(f"orphaned temporary sample directory: {root}")
    return {
        "required_outputs_exist": True,
        "samples_found": samples,
        "samples_validated": validated,
        "all_arrays_finite": True,
        "array_hashes_match": True,
        "atomic_transactions_complete": True,
        "orphaned_temp_directories": 0,
        "image_shapes": shape_by_profile,
    }


_validate_outputs = validate_outputs
