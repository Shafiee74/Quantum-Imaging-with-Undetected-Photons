"""Comment 7.7 — regenerate ``reference_outputs/digital_twin_run`` (canonical evidence).

Do **not** reuse the pre-remediation evidence pack. This script:

1. Deletes the previous ``reference_outputs/digital_twin_run/`` tree
2. Runs a fresh forward with Comment 7.1–7.6 provenance
3. Writes meta, frames, object maps, run_manifest, and verifies replay

Usage::

    python tools/regenerate_digital_twin_evidence.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_paths import REFERENCE_RUN, ROOT

OUT = REFERENCE_RUN

# Evidence grid: large enough for reviewer inspection, small enough to ship.
TARGET_SIZE = 64
NUM_FRAMES = 4
SEED = 42
OBJECT_SEED = 42


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return {
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "sha256_note": "arrays stored separately; see array_hashes / object/",
        }
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return str(obj)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from latents import apply_preset
    from optics import build_clean_reference
    from provenance import (
        MODEL_STATUS,
        MODEL_VERSION,
        RELEASE_STATUS,
        RELEASE_VERSION,
        REQUIRED_EXECUTION_EVIDENCE_KEYS,
        assert_execution_evidence,
        verify_object_evidence,
    )
    from setup_params import (
        SetupParams,
        is_canonical_setup,
        load_canonical_setup,
        save_setup_yaml,
        verify_canonical_file_hashes,
    )
    from simulate import (
        replay_from_manifest,
        save_run_evidence_pack,
        simulate_forward,
    )

    print("Comment 7.7 — regenerating reference_outputs/digital_twin_run (fresh pack)")
    print(f"  ROOT = {ROOT}")
    print(f"  OUT  = {OUT}")

    # 1) Do not reuse old evidence.
    if OUT.exists():
        shutil.rmtree(OUT)
        print("  removed previous reference_outputs/digital_twin_run/")
    OUT.mkdir(parents=True, exist_ok=False)

    # 2) Canonical lab setup (CFG-P0-00).
    setup = load_canonical_setup()
    assert is_canonical_setup(setup)
    verify_canonical_file_hashes(setup)
    save_setup_yaml(setup, OUT / "setup_used.yaml")

    # 3) Object maps (saved under OUT/clean_ref then copied into object evidence).
    clean_dir = OUT / "clean_ref"
    ref = build_clean_reference(
        str(clean_dir),
        target_size=TARGET_SIZE,
        seed=OBJECT_SEED,
        setup=setup,
    )
    # Drop non-array helpers that confuse hashing / replay loaders.
    ref = {
        "M": np.asarray(ref["M"]),
        "T": np.asarray(ref["T"]),
        "phi": np.asarray(ref["phi"]),
        "O_i": np.asarray(ref["O_i"]),
        "D": np.asarray(ref["D"]),
        "I0": np.asarray(ref["I0"]) if ref.get("I0") is not None else None,
        "target_size": TARGET_SIZE,
        "seed": OBJECT_SEED,
    }
    if ref["I0"] is None:
        ref.pop("I0")

    state = apply_preset(
        "well_aligned",
        num_frames=NUM_FRAMES,
        photon_counting=False,
        setup=setup,
        em_gain=250.0,
    )

    # 4) Forward run (detector on — measured twin frames).
    result = simulate_forward(
        ref,
        state,
        seed=SEED,
        detector_noise=True,
        raise_on_forbidden=True,
        require_canonical=True,
    )

    missing = assert_execution_evidence(result.meta)
    if missing:
        raise SystemExit(f"Comment 7.7: meta missing execution evidence: {missing}")

    # 5) Arrays + legacy meta names expected by reviewers / notebook consumers.
    np.savez_compressed(
        OUT / "frames.npz",
        frames_plus=result.frames_plus,
        frames_minus=result.frames_minus,
        frames_D=result.frames_D,
        clean_D=result.clean_D,
        sample=result.sample_dots,
    )
    np.save(OUT / "frames_plus.npy", result.frames_plus)
    np.save(OUT / "frames_minus.npy", result.frames_minus)
    np.save(OUT / "frames_D.npy", result.frames_D)
    np.save(OUT / "clean_D.npy", result.clean_D)

    meta_vis = result.meta.get("vis") or {}
    summary = {
        "finding": "comment_7_7_regenerated_evidence",
        "release_version": RELEASE_VERSION,
        "version": MODEL_VERSION,
        "release_status": RELEASE_STATUS,
        "model_status": MODEL_STATUS,
        "verified_digital_twin": False,
        "setup_id": result.meta.get("setup_id"),
        "model_version": result.meta.get("model_version"),
        "physics_tier": result.meta.get("physics_tier"),
        "seed": result.meta.get("seed"),
        "rng_scheme": result.meta.get("rng_scheme"),
        "config_hash": result.meta.get("config_hash"),
        "object_hash": result.meta.get("object_hash"),
        "gain_mode": result.meta.get("gain_mode"),
        "spdc_source_mode": result.meta.get("spdc_source_mode"),
        "blur_model": result.meta.get("blur_model"),
        "detector_mode": result.meta.get("detector_mode"),
        "timestamp": result.meta.get("timestamp"),
        "environment": result.meta.get("environment"),
        "source_hashes": result.meta.get("source_hashes"),
        "build_hash": result.meta.get("build_hash"),
        "array_hashes": result.meta.get("array_hashes"),
        "num_frames": result.meta.get("num_frames"),
        "photon_counting": result.meta.get("photon_counting"),
        "require_canonical": result.meta.get("require_canonical"),
        "is_canonical_config": result.meta.get("is_canonical_config"),
        "em_gain_GEM": result.meta.get("em_gain_GEM"),
        "V_eff": float(meta_vis.get("V_eff", float("nan"))),
        "w_px": result.meta.get("w_px"),
        "sig_spdc": result.meta.get("sig_spdc"),
        "sig_opt": result.meta.get("sig_opt"),
        "mean_hits_plus": result.meta.get("mean_hits_plus"),
        "target_size": TARGET_SIZE,
        "object_seed": OBJECT_SEED,
        "evidence_seed": SEED,
        "pump_power_laser_mW": result.meta.get("pump_power_laser_mW"),
        "pump_power_NL1_mW": result.meta.get("pump_power_NL1_mW"),
        "pump_power_NL2_mW": result.meta.get("pump_power_NL2_mW"),
        "nbar_NL1": result.meta.get("nbar_NL1"),
        "nbar_NL2": result.meta.get("nbar_NL2"),
    }
    (OUT / "meta_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (OUT / "meta_full.json").write_text(
        json.dumps(result.meta, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )

    # 6) Comment 7.5–7.6 evidence pack (object/ + run_manifest.json) + replay check.
    manifest_path = save_run_evidence_pack(OUT, result, ref)
    obj_bad = verify_object_evidence(OUT)
    if obj_bad:
        raise SystemExit("Comment 7.7: object evidence verify failed: " + "; ".join(obj_bad))

    replayed = replay_from_manifest(manifest_path, verify=True)
    np.testing.assert_array_equal(result.frames_plus, replayed.frames_plus)
    np.testing.assert_array_equal(result.frames_minus, replayed.frames_minus)
    np.testing.assert_array_equal(result.frames_D, replayed.frames_D)
    np.testing.assert_array_equal(result.clean_D, replayed.clean_D)

    readme = OUT / "README_EVIDENCE.md"
    readme.write_text(
        "\n".join(
            [
                "# QIUP reference_outputs/digital_twin_run — Comment 7.7 evidence pack",
                "",
                "This directory was **regenerated after Comment 7.1–7.6 remediation**.",
                "Do not treat any pre-remediation `qiup_arch_phase5_v1` artifacts as current.",
                "",
                f"- **version**: `{MODEL_VERSION}`",
                f"- **release**: `{RELEASE_VERSION}` / status `{RELEASE_STATUS}`",
                f"- **status label**: {MODEL_STATUS}",
                f"- **seed**: `{SEED}` (object seed `{OBJECT_SEED}`)",
                f"- **grid**: `{TARGET_SIZE}×{TARGET_SIZE}`, frames=`{NUM_FRAMES}`",
                f"- **config_hash**: `{summary['config_hash']}`",
                f"- **object_hash**: `{summary['object_hash']}`",
                "",
                "## Replay",
                "",
                "```python",
                "from simulate import replay_from_manifest",
                "result = replay_from_manifest('reference_outputs/digital_twin_run/run_manifest.json')",
                "```",
                "",
                "Required Comment 7.3 keys on meta_full / provenance:",
                ", ".join(REQUIRED_EXECUTION_EVIDENCE_KEYS),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("  wrote evidence under", OUT.resolve())
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(OUT)}  ({p.stat().st_size} bytes)")
    print("  replay verification: OK (bit-identical)")
    print("Comment 7.7 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
