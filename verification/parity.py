"""Behavioral parity: modular sources vs the generated single-file build.

For each registered configuration (A, B, C, D) and each scenario, both
implementations run with identical configuration, parameters, seed, object
and precision. Compared outputs:

* final arrays: signal frames for both ports, difference frames, clean
  difference, ideal/detector signal images, clean reference image, sample dots,
  reconstructed phase;
* the complete run-metadata tree (source moments, pump powers, spatial
  mapping, operating point, validity, provenance) except wall-clock timestamps;
* intermediate physics functions evaluated directly (crystal state, QPM
  mismatch, geometric overlap, gain coefficient, pump split, spatial mapping,
  camera geometry, thermal state, magnification).

The tolerance is zero: the single file is generated from the modular sources,
so any difference is a divergence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILES = ("A", "B", "C", "D")
SEED = 20260926
GRID = 16
TOLERANCE = 0.0
VOLATILE_META_KEYS = frozenset({"timestamp"})
TEST_ID = "SOFTWARE_TEST_ONLY_NOT_A_LAB_CALIBRATION"
ARRAY_OUTPUTS = (
    "frames_plus", "frames_minus", "frames_D", "clean_D", "ideal_signal_image",
    "detector_signal_image", "clean_reference_image", "sample_dots", "reconstructed_phase_rad",
)
QUIET = {"s_pump": 0.0, "s_gain": 0.0, "s_align": 0.0, "s_thermal": 0.0,
         "s_det": 0.0, "s_cool": 0.0, "s_focus": 0.0, "s_slm": 0.0}
NOISY = {"s_pump": 0.3, "s_gain": 0.3, "s_align": 0.3, "s_thermal": 0.3,
         "s_det": 0.3, "s_cool": 0.3, "s_focus": 0.3, "s_slm": 0.3}
CALIBRATED = {
    "spdc_source_mode": "calibrated_source", "spdc_gain_per_sqrt_mW": 0.05,
    "spdc_gain_calibration_id": TEST_ID, "spdc_gain_calibration_id_NL2": TEST_ID,
    "pump_power_laser_mW": 40.0,
}
SCENARIOS: dict[str, dict[str, Any]] = {
    "low_gain_quiet": {"setup": {}, "state": {**QUIET, "mean_photons_per_mode": 0.001}},
    "transition_quiet": {"setup": {}, "state": {**QUIET, "mean_photons_per_mode": 0.1}},
    "high_gain_quiet": {"setup": {}, "state": {**QUIET, "mean_photons_per_mode": 5.0}},
    "unequal_sources": {"setup": {}, "state": {**QUIET, "mean_photons_per_mode_NL1": 0.2,
                                                "mean_photons_per_mode_NL2": 0.05}},
    "latent_noise": {"setup": {}, "state": {**NOISY, "mean_photons_per_mode": 0.1, "num_frames": 3}},
    "calibrated_source": {"setup": dict(CALIBRATED), "state": {**NOISY, "num_frames": 2}},
    "photon_counting": {"setup": {"detector_mode": "photon_counting"},
                        "state": {**NOISY, "mean_photons_per_mode": 0.1, "photon_counting": True,
                                  "num_frames": 2, "em_gain": 300.0}},
    "single_blur": {"setup": {"blur_model": "single_blur"}, "state": {**NOISY, "mean_photons_per_mode": 0.1}},
}
BASE_STATE = {"num_frames": 1, "exposure_s": 0.02, "em_gain": 100.0}


def load_single_file(root: Path = ROOT) -> ModuleType:
    path = root / "qiup_digital_twin_full.py"
    name = "qiup_single_file_parity_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def object_reference(n: int = GRID) -> dict[str, Any]:
    """Asymmetric transmission and phase so flips, offsets and scaling show."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    T = np.clip(0.15 + 0.75 * xx / (n - 1), 0.0, 1.0).astype(np.float32)
    T[2:5, n - 5:n - 2] = 0.0
    phi = (1.1 * yy / (n - 1) + 0.6 * (xx > n / 3)).astype(np.float32)
    one = np.ones((n, n), dtype=np.float32)
    return {"M": one, "T": T, "phi": phi, "O_i": one, "D": np.zeros_like(one), "target_size": n}


def _leaf_error(a: Any, b: Any) -> float:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        return 0.0 if (type(a) is type(b) and a == b) else math.inf
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return 0.0
        if math.isinf(fa) or math.isinf(fb):
            return 0.0 if fa == fb else math.inf
        return abs(fa - fb)
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(a) - complex(b))
    return 0.0 if repr(a) == repr(b) else math.inf


def compare_tree(a: Any, b: Any, path: str = "") -> tuple[float, int, list[str]]:
    """Max absolute error over all leaves, leaf count, and mismatch paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = {k for k in a if k not in VOLATILE_META_KEYS}
        keys_b = {k for k in b if k not in VOLATILE_META_KEYS}
        worst, count, bad = 0.0, 0, [f"{path}/{k} (missing)" for k in sorted(keys_a ^ keys_b, key=str)]
        if bad:
            worst = math.inf
        for k in sorted(keys_a & keys_b, key=str):
            e, c, m = compare_tree(a[k], b[k], f"{path}/{k}")
            worst, count, bad = max(worst, e), count + c, bad + m
        return worst, count, bad
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf, 1, [f"{path} (length {len(a)} vs {len(b)})"]
        worst, count, bad = 0.0, 0, []
        for i, (x, y) in enumerate(zip(a, b)):
            e, c, m = compare_tree(x, y, f"{path}[{i}]")
            worst, count, bad = max(worst, e), count + c, bad + m
        return worst, count, bad
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return compare_arrays(a, b, path)
    e = _leaf_error(a, b)
    return e, 1, ([] if e <= TOLERANCE else [f"{path}: {a!r} vs {b!r}"])


def compare_arrays(a: Any, b: Any, path: str) -> tuple[float, int, list[str]]:
    if a is None and b is None:
        return 0.0, 0, []
    if a is None or b is None:
        return math.inf, 1, [f"{path} (None vs array)"]
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return math.inf, 1, [f"{path} (shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype})"]
    if x.size == 0:
        return 0.0, 0, []
    if not np.issubdtype(x.dtype, np.number) and x.dtype != bool:
        same = bool(np.array_equal(x, y))
        return (0.0 if same else math.inf), int(x.size), ([] if same else [path])
    xf, yf = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64), \
        y.astype(np.complex128 if np.iscomplexobj(y) else np.float64)
    both_nan = np.isnan(xf) & np.isnan(yf)
    diff = np.abs(np.where(both_nan, 0.0, xf - yf))
    diff = np.where(np.isnan(diff), np.inf, diff)
    err = float(diff.max())
    return err, int(x.size), ([] if err <= TOLERANCE else [f"{path}: max abs error = {err:.3e}"])


def _intermediate(module: Any, setup: Any, n: int) -> dict[str, Any]:
    sp = module if hasattr(module, "spdc_crystal_state") else module.setup_params
    out: dict[str, Any] = {}
    for i in (1, 2):
        out[f"spdc_crystal_state[{i}]"] = sp.spdc_crystal_state(setup, interaction=i)
        out[f"delta_k_qpm_per_m[{i}]"] = sp.delta_k_qpm_per_m_from_setup(setup, interaction=i)
        out[f"geometry_overlap[{i}]"] = sp.spdc_geometry_overlap_from_setup(setup, interaction=i)
        out[f"k_spdc_per_sqrt_W[{i}]"] = sp.resolved_k_spdc_per_sqrt_W(setup, interaction=i)
    out["pump_powers"] = sp.pump_powers_from_setup(setup, relative_power_fluctuation=0.07)
    out["architecture_parameter_snapshot"] = sp.architecture_parameter_snapshot(setup)
    out["resolved_spatial_mapping"] = sp.resolved_spatial_mapping(setup)
    out["camera_plane_geometry"] = sp.camera_plane_geometry_from_setup(setup, n_pixels=n)
    out["crystal_thermal_state"] = sp.crystal_thermal_state(setup)
    out["imaging_magnification"] = sp.resolve_imaging_magnification(setup)
    return out


def run_pair(full: ModuleType, profile: str, scenario: str, n: int = GRID) -> dict[str, Any]:
    import latents
    import setup_params
    import simulate

    spec = SCENARIOS[scenario]
    state_kw = {**BASE_STATE, **spec["state"]}
    mod_setup = setup_params.resolve_setup_selection(profile)
    full_setup = full.resolve_setup_selection(profile)
    if spec["setup"]:
        mod_setup = mod_setup.overridden(**spec["setup"])
        full_setup = full_setup.overridden(**spec["setup"])
    ref = object_reference(n)
    mod_res = simulate.simulate_forward(ref, latents.LatentState(setup=mod_setup, **state_kw),
                                        seed=SEED, detector_noise=True)
    full_res = full.simulate_forward(object_reference(n), full.LatentState(setup=full_setup, **state_kw),
                                     seed=SEED, detector_noise=True)

    groups: dict[str, tuple[float, int, list[str]]] = {}
    worst, count, bad = 0.0, 0, []
    for name in ARRAY_OUTPUTS:
        e, c, m = compare_arrays(getattr(mod_res, name, None), getattr(full_res, name, None), name)
        worst, count, bad = max(worst, e), count + c, bad + m
    groups["final_arrays"] = (worst, count, bad)
    groups["run_metadata"] = compare_tree(mod_res.meta, full_res.meta, "meta")
    groups["intermediate_functions"] = compare_tree(
        _intermediate(setup_params, mod_setup, n), _intermediate(full, full_setup, n), "intermediate")
    groups["validity"] = compare_tree(
        {"ok": mod_res.validity.ok, "violations": list(mod_res.validity.violations)},
        {"ok": full_res.validity.ok, "violations": list(full_res.validity.violations)}, "validity")
    groups["selected_output_port"] = compare_tree(mod_res.selected_output_port, full_res.selected_output_port,
                                                  "selected_output_port")
    if mod_res.meta["architecture_profile"]["profile_id"] != profile:
        raise AssertionError(f"modular run resolved to wrong profile for {profile}")

    max_err = max(g[0] for g in groups.values())
    return {
        "configuration": profile,
        "scenario": scenario,
        "seed": SEED,
        "grid": n,
        "outputs_compared": {k: {"values": v[1], "max_abs_error": v[0]} for k, v in groups.items()},
        "values_compared": sum(g[1] for g in groups.values()),
        "max_abs_error": max_err,
        "tolerance": TOLERANCE,
        "status": "PASS" if max_err <= TOLERANCE else "FAIL",
        "mismatches": [m for g in groups.values() for m in g[2]][:20],
    }


def run_all(root: Path = ROOT) -> list[dict[str, Any]]:
    full = load_single_file(root)
    return [run_pair(full, p, s) for p in PROFILES for s in SCENARIOS]


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in PROFILES:
        rows = [r for r in records if r["configuration"] == p]
        out.append({
            "configuration": p,
            "scenarios": len(rows),
            "outputs_compared": sorted({k for r in rows for k in r["outputs_compared"]}),
            "values_compared": sum(r["values_compared"] for r in rows),
            "max_abs_error": max(r["max_abs_error"] for r in rows),
            "tolerance": TOLERANCE,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    records = run_all()
    summary = summarize(records)
    for row in summary:
        print(f"{row['configuration']}: {row['scenarios']} scenarios, {row['values_compared']} values, "
              f"max abs error = {row['max_abs_error']:.3e} -> {row['status']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, default=str) + "\n",
                            encoding="utf-8", newline="\n")
    return 0 if all(r["status"] == "PASS" for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
