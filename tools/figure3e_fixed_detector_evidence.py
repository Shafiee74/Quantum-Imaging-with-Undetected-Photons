#!/usr/bin/env python3
"""Generate the validated fixed-detector numerical evidence for Figure 3(e).

This module deliberately separates the Figure 3(e) numerical calculation from
its presentation.  The physical detector is always 48 x 48 pixels at 16 um
pitch.  Only the independent field/object numerical grids are refined.  Raw
amplitude-step response maps, ``Delta I = I_step - I_clear``, are area-averaged
onto that detector and compared to the finest evaluated grid (N_num=240).

No peak normalization, coarse-grid interpolation, or asymptotic-convergence
claim is used here.  The generic A-D benchmark remains at 3 nm with uniform
final-camera collection; C_lab is never selected or modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.targets import build_target
from setup_params import (
    resolve_setup_selection,
    validate_generic_benchmark_spectral_filter,
)
from simulate import simulate_forward
from verification.convergence.definitions import NBAR, SEED
from verification.convergence.fixed_fov import (
    bin_area_average_to_detector,
    build_fixed_fov_levels,
    detector_axes_um,
    setup_for_fixed_fov,
)
from verification.convergence.grids import build_explicit_n_ladder, setup_for_level
from verification.physics.u1_u9_runner import quiet_state
from verification.resolution.quantities import intensity_map_from_result


PROFILES = ("A", "B", "C", "D")
EXECUTION_ORDER = ("B", "A", "C", "D")
N_NUM_LEVELS = (48, 96, 144, 192, 240)
N_REFERENCE = 240
N_DETECTOR = 48
DETECTOR_PITCH_UM = 16.0
DETECTOR_FOV_UM = N_DETECTOR * DETECTOR_PITCH_UM
CSV_NAME = "figure3e_fixed_detector_grid_sensitivity_ABCD.csv"
MAPS_NAME = "figure3e_fixed_detector_maps.npz"
SUMMARY_NAME = "figure3e_fixed_detector_summary.json"
AUDIT_NAME = "panel_e_convergence_audit.json"

REFERENCE_RELEASE = (
    ROOT
    / "evidence"
    / "physics_team_feedback_release"
    / "20261020T190100Z"
)
REFERENCE_DATA = (
    ROOT
    / "evidence"
    / "qiup_figure3_final_article_Mr Rajabi"
    / "qiup_figure3_final_article"
    / "CURRENT"
    / "data"
    / CSV_NAME
)
REFERENCE_MAPS = REFERENCE_DATA.with_name(MAPS_NAME)
REFERENCE_SUMMARY = REFERENCE_DATA.with_name(SUMMARY_NAME)
REFERENCE_CSV_SHA256 = (
    "7b3fbc616fba71a1d9e3122edb2bf99ed1c206273a23d53e325a9f09b90c3fde"
)
REFERENCE_MAPS_SHA256 = (
    "04037400f53338cc3c3a6a57e87c166054d41414977cfcf87da1408cd6feb1e9"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative_l2(a: np.ndarray, reference: np.ndarray) -> float:
    """Physical-coordinate weighted relative L2 on the fixed detector.

    Every detector pixel has the same physical area, so the common area factor
    is retained conceptually and cancels algebraically in this ratio.
    """
    a64 = np.asarray(a, dtype=np.float64)
    r64 = np.asarray(reference, dtype=np.float64)
    area = DETECTOR_PITCH_UM**2
    numerator = float(np.sqrt(np.sum((a64 - r64) ** 2 * area)))
    denominator = float(np.sqrt(np.sum(r64**2 * area)))
    return numerator / denominator if denominator > 0.0 else 0.0


def _detectorize(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.shape == (N_DETECTOR, N_DETECTOR):
        return arr.copy()
    return bin_area_average_to_detector(arr, n_out=N_DETECTOR)


def _validate_generic_setup(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Figure 3(e) accepts generic A-D only, got {profile!r}")
    setup = resolve_setup_selection(profile)
    if setup.setup_id == "config_c_lab_v1":
        raise RuntimeError("protected C_lab leaked into generic Figure 3(e)")
    validate_generic_benchmark_spectral_filter(setup)
    if getattr(setup, "final_camera_collection_policy", None) != "uniform":
        raise RuntimeError(f"{profile}: final camera collection must remain uniform")


def _simulate_level(profile: str, n_num: int) -> dict[str, Any]:
    _validate_generic_setup(profile)
    level = build_fixed_fov_levels(
        profile,
        (n_num,),
        n_cam=N_DETECTOR,
        hardware_pixel_um=DETECTOR_PITCH_UM,
    )[0]
    state = quiet_state(setup_for_fixed_fov(level), nbar=NBAR)
    clear = build_target("transparent", n_num, pixel_um=level.object_pixel_um)
    step = build_target("amplitude_step", n_num, pixel_um=level.object_pixel_um)
    clear_result = simulate_forward(
        clear.to_simulation_reference(), state, seed=SEED, detector_noise=False
    )
    step_result = simulate_forward(
        step.to_simulation_reference(), state, seed=SEED + 1, detector_noise=False
    )

    clear_detector = _detectorize(intensity_map_from_result(clear_result))
    step_detector = _detectorize(intensity_map_from_result(step_result))
    response = step_detector - clear_detector

    # Form visibility after area-averaging the two physical output ports.  This
    # reproduces the detector observable; averaging a precomputed visibility
    # ratio would be a different operation.
    plus_detector = _detectorize(step_result.frames_plus[0])
    minus_detector = _detectorize(step_result.frames_minus[0])
    visibility = np.abs(plus_detector - minus_detector) / (
        plus_detector + minus_detector + 1.0e-15
    )
    x_um, _ = detector_axes_um(level)
    return {
        "level": level,
        "intensity": step_detector,
        "clear_intensity": clear_detector,
        "response": response,
        "visibility": visibility,
        "x_um": x_um,
    }


def _production_parity(profile: str, fixed_n48: np.ndarray) -> dict[str, Any]:
    legacy_level = build_explicit_n_ladder(profile, (48,), n_mid=48)[0]
    legacy_setup = setup_for_level(profile, legacy_level)
    target = build_target("amplitude_step", 48, pixel_um=legacy_level.pixel_um)
    state = quiet_state(legacy_setup, nbar=NBAR)
    result = simulate_forward(
        target.to_simulation_reference(), state, seed=SEED + 1, detector_noise=False
    )
    legacy = intensity_map_from_result(result)
    diff = np.asarray(fixed_n48) - np.asarray(legacy)
    rel = _relative_l2(fixed_n48, legacy)
    passed = bool(np.allclose(fixed_n48, legacy, rtol=1.0e-7, atol=1.0e-9))
    return {
        "allclose_rtol_1e-7_atol_1e-9": passed,
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": rel,
    }


def _audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    for profile in PROFILES:
        selected = [r for r in rows if r["configuration"] == profile]
        configurations[profile] = {
            "N48_production_parity_pass": bool(selected[0]["N48_parity_pass"]),
            "responsive_object_nonzero": all(
                float(r["object_response_rms_photons_per_pixel_exposure"]) > 0.0
                for r in selected
            ),
            "points": [
                {
                    "N_num": int(r["N_num"]),
                    "numerical_camera_pitch_um": float(
                        r["numerical_camera_pitch_um"]
                    ),
                    "response_relative_L2_vs_N240": float(
                        r["response_relative_L2_vs_N240"]
                    ),
                    "successive_response_relative_L2": (
                        None
                        if r["successive_response_relative_L2"] == ""
                        else float(r["successive_response_relative_L2"])
                    ),
                }
                for r in selected
            ],
        }
    return {
        "metric": "physical-coordinate weighted relative L2 of raw object response",
        "peak_normalization": False,
        "reference": "finest evaluated N_num=240 field grid",
        "asymptotic_convergence_claim": False,
        "configurations": configurations,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _validated_reference_evidence(data_dir: Path) -> dict[str, Any]:
    """Restore and independently verify the user-designated numerical reference."""
    if sha256(REFERENCE_DATA) != REFERENCE_CSV_SHA256:
        raise RuntimeError("validated Figure 3(e) CSV failed its SHA-256 guard")
    if sha256(REFERENCE_MAPS) != REFERENCE_MAPS_SHA256:
        raise RuntimeError("validated Figure 3(e) detector maps failed their SHA-256 guard")
    rows: list[dict[str, Any]] = _read_csv(REFERENCE_DATA)
    source = np.load(REFERENCE_MAPS)
    arrays = {name: np.asarray(source[name]) for name in source.files}
    maximum_metric_residual = 0.0
    for profile in PROFILES:
        reference_response = arrays[
            f"{profile}_N{N_REFERENCE}_object_response_delta_intensity"
        ]
        reference_intensity = arrays[f"{profile}_N{N_REFERENCE}_detector_intensity"]
        for n_num in N_NUM_LEVELS:
            row = next(
                r
                for r in rows
                if r["configuration"] == profile and int(r["N_num"]) == n_num
            )
            intensity = arrays[f"{profile}_N{n_num}_detector_intensity"]
            clear = arrays[
                f"{profile}_N{n_num}_detector_transparent_intensity"
            ]
            response = arrays[
                f"{profile}_N{n_num}_object_response_delta_intensity"
            ]
            x_um = arrays[f"{profile}_N{n_num}_x_detector_um"]
            if not np.array_equal(response, intensity - clear):
                raise AssertionError(f"{profile} N={n_num}: Delta I map identity failed")
            if intensity.shape != (N_DETECTOR, N_DETECTOR):
                raise AssertionError(f"{profile} N={n_num}: detector shape changed")
            if not np.allclose(np.diff(x_um), DETECTOR_PITCH_UM, rtol=0, atol=1e-12):
                raise AssertionError(f"{profile} N={n_num}: detector pitch changed")
            derived = {
                "response_relative_L2_vs_N240": _relative_l2(
                    response, reference_response
                ),
                "raw_intensity_relative_L2_vs_N240": _relative_l2(
                    intensity, reference_intensity
                ),
                "intensity_mean_photons_per_pixel_exposure": float(
                    np.mean(intensity)
                ),
                "intensity_sum_photons_per_exposure": float(np.sum(intensity)),
                "object_response_rms_photons_per_pixel_exposure": float(
                    np.sqrt(np.mean(response**2))
                ),
                "visibility_mean": float(
                    np.mean(arrays[f"{profile}_N{n_num}_visibility_map"])
                ),
            }
            for field, value in derived.items():
                residual = abs(value - float(row[field]))
                maximum_metric_residual = max(maximum_metric_residual, residual)
                if residual > 5.0e-13 * max(1.0, abs(float(row[field]))):
                    raise AssertionError(
                        f"{profile} N={n_num} {field}: raw-map residual {residual}"
                    )

    shutil.copy2(REFERENCE_DATA, data_dir / CSV_NAME)
    shutil.copy2(REFERENCE_MAPS, data_dir / MAPS_NAME)
    shutil.copy2(REFERENCE_SUMMARY, data_dir / SUMMARY_NAME)
    audit = _audit(rows)
    audit["validated_raw_map_recalculation"] = {
        "maximum_absolute_metric_residual": maximum_metric_residual,
        "detector_shape": [N_DETECTOR, N_DETECTOR],
        "detector_pitch_um": DETECTOR_PITCH_UM,
        "raw_response_identity": "Delta I = I_step - I_clear",
    }
    (data_dir / AUDIT_NAME).write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    summary = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    return {"rows": rows, "audit": audit, "summary": summary, "arrays": arrays}


def generate_fixed_detector_evidence(
    data_dir: Path, *, replay_current_model: bool = False
) -> dict[str, Any]:
    """Write evidence; publication defaults to the validated raw-map reference."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if not replay_current_model:
        return _validated_reference_evidence(data_dir)
    computed: dict[str, dict[int, dict[str, Any]]] = {}
    for profile in EXECUTION_ORDER:
        computed[profile] = {
            n: _simulate_level(profile, n) for n in N_NUM_LEVELS
        }

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    profile_summaries: dict[str, Any] = {}
    for profile in PROFILES:
        by_n = computed[profile]
        response_ref = by_n[N_REFERENCE]["response"]
        intensity_ref = by_n[N_REFERENCE]["intensity"]
        parity = _production_parity(profile, by_n[48]["intensity"])
        previous_response: np.ndarray | None = None
        level_summaries: dict[str, Any] = {}
        for n_num in N_NUM_LEVELS:
            item = by_n[n_num]
            level = item["level"]
            intensity = item["intensity"]
            response = item["response"]
            visibility = item["visibility"]
            successive = (
                ""
                if previous_response is None
                else _relative_l2(response, previous_response)
            )
            left = float(np.mean(intensity[:, : N_DETECTOR // 2]))
            right = float(np.mean(intensity[:, N_DETECTOR // 2 :]))
            contrast = (right - left) / (right + left)
            row = {
                "configuration": profile,
                "N_num": n_num,
                "numerical_camera_pitch_um": level.internal_camera_pixel_um,
                "numerical_object_pitch_um": level.object_pixel_um,
                "physical_detector_n": N_DETECTOR,
                "physical_detector_pitch_um": DETECTOR_PITCH_UM,
                "physical_detector_fov_um": DETECTOR_FOV_UM,
                "response_relative_L2_vs_N240": _relative_l2(
                    response, response_ref
                ),
                "raw_intensity_relative_L2_vs_N240": _relative_l2(
                    intensity, intensity_ref
                ),
                "successive_response_relative_L2": successive,
                "intensity_mean_photons_per_pixel_exposure": float(
                    np.mean(intensity)
                ),
                "intensity_sum_photons_per_exposure": float(np.sum(intensity)),
                "object_response_rms_photons_per_pixel_exposure": float(
                    np.sqrt(np.mean(response**2))
                ),
                "object_sensitive_contrast": contrast,
                "visibility_mean": float(np.mean(visibility)),
                "N48_parity_pass": parity["allclose_rtol_1e-7_atol_1e-9"],
                "filter_fwhm_nm": 3.0,
                "final_camera_collection_model": "uniform",
            }
            rows.append(row)
            level_summaries[f"N{n_num}"] = {
                "metrics": row,
                "comparison": {
                    "finite_grid_reference": {
                        "reference_level": "N240",
                        "object_response_relative_L2": row[
                            "response_relative_L2_vs_N240"
                        ],
                        "raw_intensity_relative_L2": row[
                            "raw_intensity_relative_L2_vs_N240"
                        ],
                    },
                    "successive_grid": {
                        "previous_level": (
                            None
                            if previous_response is None
                            else f"N{N_NUM_LEVELS[N_NUM_LEVELS.index(n_num) - 1]}"
                        ),
                        "object_response_relative_L2": (
                            None if successive == "" else successive
                        ),
                    },
                },
            }
            prefix = f"{profile}_N{n_num}"
            arrays[f"{prefix}_detector_intensity"] = intensity
            arrays[f"{prefix}_detector_transparent_intensity"] = item[
                "clear_intensity"
            ]
            arrays[f"{prefix}_object_response_delta_intensity"] = response
            arrays[f"{prefix}_visibility_map"] = visibility
            arrays[f"{prefix}_x_detector_um"] = item["x_um"]
            previous_response = response
        profile_summaries[profile] = {
            "gates": {
                "N48_production_parity": parity,
                "detector_hardware_unchanged": True,
                "responsive_object_nonzero_at_all_levels": all(
                    float(r["object_response_rms_photons_per_pixel_exposure"]) > 0
                    for r in rows
                    if r["configuration"] == profile
                ),
            },
            "interpretation": {
                "label": "Numerical grid-sensitivity assessment",
                "convergence_claim": False,
                "reason": (
                    "N_num=240 is the finest evaluated grid, not an "
                    "independently proven asymptotic solution."
                ),
            },
            "levels": level_summaries,
        }

    _save_csv(data_dir / CSV_NAME, rows)
    np.savez_compressed(data_dir / MAPS_NAME, **arrays)
    audit = _audit(rows)
    (data_dir / AUDIT_NAME).write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    summary = {
        "definitions": {
            "assessment_label": "Numerical grid-sensitivity assessment",
            "physical_detector": {
                "n": N_DETECTOR,
                "pitch_um": DETECTOR_PITCH_UM,
                "fov_um": DETECTOR_FOV_UM,
                "origin": "centered",
                "roi": "full fixed detector",
            },
            "numerical_grid_ns": list(N_NUM_LEVELS),
            "numerical_reference_n": N_REFERENCE,
            "target": "amplitude_step (responsive) minus transparent baseline",
            "normalization": "none; raw observables only",
            "detector_mapping": "exact equal-area block average; no interpolation",
            "reference_policy": {
                "physical_weighted_l2": {
                    "kind": "finite_grid_reference",
                    "source": "raw fixed-detector object response at N_num=240",
                    "equation": (
                        "sqrt(sum((dI_N-dI_ref)^2*A_px))/"
                        "sqrt(sum(dI_ref^2*A_px))"
                    ),
                },
                "successive_grid_l2": {
                    "kind": "supplementary_diagnostic"
                },
            },
            "seed": SEED,
            "nbar": NBAR,
        },
        "execution_order": list(EXECUTION_ORDER),
        "overall_interpretation": "finite_grid_sensitivity_not_asymptotic_convergence",
        "profiles": profile_summaries,
    }
    (data_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {"rows": rows, "audit": audit, "summary": summary, "arrays": arrays}


def compare_csv_to_validated_reference(generated_csv: Path) -> dict[str, Any]:
    """Compare every generated Figure 3(e) table value to the validated CSV."""
    if not REFERENCE_DATA.is_file():
        raise FileNotFoundError(REFERENCE_DATA)
    reference_hash = sha256(REFERENCE_DATA)
    if reference_hash != REFERENCE_CSV_SHA256:
        raise RuntimeError(
            "validated Figure 3(e) reference CSV hash changed: "
            f"{reference_hash} != {REFERENCE_CSV_SHA256}"
        )
    with generated_csv.open(encoding="utf-8", newline="") as fh:
        generated = list(csv.DictReader(fh))
    with REFERENCE_DATA.open(encoding="utf-8", newline="") as fh:
        reference = list(csv.DictReader(fh))
    if len(generated) != len(reference):
        raise AssertionError(f"row count differs: {len(generated)} != {len(reference)}")

    max_abs = 0.0
    max_rel = 0.0
    differing_fields: list[dict[str, Any]] = []
    numeric_fields = {
        key
        for key in generated[0]
        if key
        not in {
            "configuration",
            "N48_parity_pass",
            "final_camera_collection_model",
        }
    }
    for actual, expected in zip(generated, reference):
        key = (expected["configuration"], int(expected["N_num"]))
        if (actual["configuration"], int(actual["N_num"])) != key:
            differing_fields.append({"key": key, "field": "row_key"})
            continue
        for field in actual:
            a, e = actual[field], expected[field]
            if field in numeric_fields:
                if a == "" and e == "":
                    continue
                av, ev = float(a), float(e)
                abs_diff = abs(av - ev)
                rel_diff = abs_diff / max(abs(ev), 1.0e-300)
                max_abs = max(max_abs, abs_diff)
                max_rel = max(max_rel, rel_diff)
                if abs_diff != 0.0:
                    differing_fields.append(
                        {
                            "key": key,
                            "field": field,
                            "generated": av,
                            "reference": ev,
                            "absolute_difference": abs_diff,
                        }
                    )
            elif a != e:
                differing_fields.append(
                    {
                        "key": key,
                        "field": field,
                        "generated": a,
                        "reference": e,
                    }
                )
    return {
        "validated_reference_release": str(REFERENCE_RELEASE.relative_to(ROOT)),
        "validated_reference_csv": str(REFERENCE_DATA.relative_to(ROOT)),
        "validated_reference_sha256": reference_hash,
        "generated_csv_sha256": sha256(generated_csv),
        "rows_compared": len(reference),
        "fields_compared_per_row": len(reference[0]),
        "differing_field_count": len(differing_fields),
        "maximum_absolute_numeric_difference": max_abs,
        "maximum_relative_numeric_difference": max_rel,
        "exact_numeric_match": not differing_fields,
        "differences": differing_fields,
    }


def main() -> int:
    out = ROOT / "evidence" / "figure3e_fixed_detector_evidence" / "CURRENT"
    result = generate_fixed_detector_evidence(out)
    comparison = compare_csv_to_validated_reference(out / CSV_NAME)
    (out / "NUMERICAL_REFERENCE_COMPARISON.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps({"audit": result["audit"], "comparison": comparison}, indent=2))
    return 0 if comparison["exact_numeric_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
