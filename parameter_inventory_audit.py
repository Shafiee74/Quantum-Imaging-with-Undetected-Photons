"""Trace which parameters each registered configuration actually consumes.

For every ``SetupParams`` and ``LatentState`` field and every registered
profile (A, B, C, D), the field is perturbed in a set of scenarios and the
forward model is re-run with a fixed seed. The outcome per profile is:

* ``physics``          -- a perturbation changes the simulated arrays;
* ``validated_fixed``  -- every perturbation is rejected by profile validation
                          (the value is fixed by the configuration);
* ``metadata_only``    -- arrays unchanged, run metadata changes;
* ``not_consumed``     -- accepted perturbations change nothing;
* ``not_perturbable``  -- no admissible perturbation value was found.

The result is written to ``configs/parameter_consumption_audit_3.4.1.json``
together with the SHA-256 of the modular sources it was traced against, so
``tests/test_parameter_inventory_consistency.py`` can detect a stale audit.

Calibration identifiers and gain coefficients used by the calibrated-source
scenarios are synthetic ``SOFTWARE_TEST_ONLY`` values, not laboratory data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIT_RELATIVE = "configs/parameter_consumption_audit_3.4.1.json"
PROFILES = ("A", "B", "C", "D")
SEED = 20260926
SIZE = 12
DETECTOR_STRESS_SIZE = 48
TEST_ID = "SOFTWARE_TEST_ONLY_NOT_A_LAB_CALIBRATION"
ARRAY_KEYS = (
    "frames_plus", "frames_minus", "frames_D", "clean_D",
    "ideal_signal_image", "detector_signal_image", "clean_reference_image",
)
VOLATILE_META_KEYS = {
    "timestamp", "environment", "source_hashes", "build_hash", "created_utc",
    "generated_utc", "elapsed_s", "runtime_s", "wall_time_s",
}
LATENT_SKIP = {"setup", "physical_report"}
STATUS_ORDER = ("physics", "validated_fixed", "metadata_only", "not_consumed", "not_perturbable")

CALIBRATED = {
    "spdc_source_mode": "calibrated_source",
    "spdc_gain_per_sqrt_mW": 0.05,
    "spdc_gain_calibration_id": TEST_ID,
    "spdc_gain_calibration_id_NL2": TEST_ID,
    "pump_power_laser_mW": 40.0,
}
BASE_LATENTS = {
    "num_frames": 2, "exposure_s": 0.02, "em_gain": 100.0,
    "s_pump": 0.3, "s_gain": 0.3, "s_align": 0.3, "s_thermal": 0.3,
    "s_det": 0.3, "s_cool": 0.3, "s_focus": 0.3, "s_slm": 0.3,
}


def scenarios(profile: str) -> dict[str, dict[str, dict[str, Any]]]:
    two = profile in ("A", "D")
    out = {
        "default": {"setup": {}, "state": {"mean_photons_per_mode": 0.1}},
        "unequal_sources": {"setup": {}, "state": {"mean_photons_per_mode_NL1": 0.2,
                                                   "mean_photons_per_mode_NL2": 0.05}},
        "squeezing_input": {"setup": {}, "state": {"spdc_gain_r": 0.3}},
        "calibrated_source": {"setup": dict(CALIBRATED), "state": {}},
        "calibrated_pump_override": {
            "setup": {**CALIBRATED, "use_independent_pump_power_NL2": True,
                      "pump_power_NL2_override_mW": 15.0,
                      "pump_power_NL2_override_calibration_id": TEST_ID},
            "state": {"pump_power_mW": 30.0},
        },
        "single_blur": {"setup": {"blur_model": "single_blur"}, "state": {"mean_photons_per_mode": 0.1}},
        "object_scalar": {"setup": {"object_phase_rad": 1.0}, "state": {"mean_photons_per_mode": 0.1}},
        "object_mask": {"setup": {"object_mode": "mask", "object_mask_id": 3},
                        "state": {"mean_photons_per_mode": 0.1}},
        "photon_counting": {"setup": {"detector_mode": "photon_counting"},
                            "state": {"mean_photons_per_mode": 0.1, "photon_counting": True}},
        "physical_thermal_off": {"setup": {"use_physical_thermal": False},
                                 "state": {"mean_photons_per_mode": 0.1}},
        "calibrated_physical_coupling": {
            "setup": {k: v for k, v in CALIBRATED.items() if k != "spdc_gain_per_sqrt_mW"}
            | {"d_eff_pm_per_V": 9.5},
            "state": {},
        },
        "etalon_on": {"setup": {"enable_etalon_fringes": True}, "state": {"mean_photons_per_mode": 0.1}},
        "lens_rings_on": {"setup": {"enable_lens_rings": True, "lens_shift_x_um": 20.0, "delta_z_um": 50.0},
                          "state": {"mean_photons_per_mode": 0.1}},
        "hot_pixels_mask_T": {"setup": {"hot_pixel_model": "mask_T"}, "state": {"mean_photons_per_mode": 0.1}},
        "strong_latents": {"setup": {}, "state": {"mean_photons_per_mode": 0.1, "s_pump": 0.8, "s_gain": 0.8,
                                                  "s_align": 0.8, "s_thermal": 0.8, "s_det": 0.8, "s_cool": 0.8,
                                                  "s_focus": 0.8, "s_slm": 0.8}},
        # Rare detector events (dark electrons, hot pixels) need enough pixels
        # and exposure to occur at all in a fixed-seed run.
        "detector_stress": {"setup": {}, "size": DETECTOR_STRESS_SIZE,
                            "state": {"mean_photons_per_mode": 0.1, "exposure_s": 50.0, "em_gain": 1.0,
                                      "num_frames": 3, "s_det": 0.8, "s_cool": 0.8, "s_thermal": 0.8}},
        "detector_stress_photon_counting": {
            "setup": {"detector_mode": "photon_counting", "emccd_model": "register"},
            "size": DETECTOR_STRESS_SIZE,
            "state": {"mean_photons_per_mode": 0.1, "photon_counting": True, "exposure_s": 5.0,
                      "em_gain": 300.0, "num_frames": 3, "s_det": 0.8, "s_cool": 0.8, "s_thermal": 0.8},
        },
        # Full-well / ADC clipping only acts on a saturated detector.
        "detector_saturation": {"setup": {}, "size": DETECTOR_STRESS_SIZE,
                                "state": {"mean_photons_per_mode": 0.1, "exposure_s": 50.0, "em_gain": 1000.0,
                                          "num_frames": 3, "s_det": 0.8, "s_cool": 0.8, "s_thermal": 0.8}},
        "physical_dark": {"setup": {"use_physical_dark": True}, "size": DETECTOR_STRESS_SIZE,
                          "state": {"mean_photons_per_mode": 0.1, "exposure_s": 50.0, "em_gain": 1.0,
                                    "num_frames": 3, "s_det": 0.8, "s_cool": 0.8, "s_thermal": 0.8}},
        "datasheet_read_noise": {"setup": {"read_noise_model": "datasheet_fixed"},
                                 "state": {"mean_photons_per_mode": 0.1}},
        # Legacy explicit-sigma / overlap parameters act only when the derived
        # physics switches are off.
        "legacy_switches": {
            "setup": {"use_derived_diffraction": False, "derive_sigma_spdc": False,
                      "use_derived_aberration": False, "use_derived_K_defocus": False,
                      "use_analytic_mode_overlap": False, "use_field_overlap_visibility_map": False},
            "state": {"mean_photons_per_mode": 0.1, "s_focus": 0.8, "s_align": 0.8},
        },
        "legacy_two_beam": {"setup": {"use_physical_two_beam": False}, "state": {"mean_photons_per_mode": 0.1}},
    }
    if two:
        out["calibrated_sequential"] = {
            "setup": {**CALIBRATED, "two_crystal_pump_topology": "sequential",
                      "pump_between_crystals_transmission": 0.6},
            "state": {},
        }
        out["mismatch_study"] = {
            "setup": {"crystal_matching_mode": "mismatch_study", "crystal_length_NL2_mm": 2.5},
            "state": {"mean_photons_per_mode": 0.1},
        }
        out["calibrated_mismatch"] = {
            "setup": {**CALIBRATED, "crystal_matching_mode": "mismatch_study", "crystal_length_NL2_mm": 2.5},
            "state": {},
        }
    return out


# --------------------------------------------------------------------------
# Worker-side helpers (imported lazily so the pool can spawn on Windows)
# --------------------------------------------------------------------------
_CACHE: dict[str, Any] = {}


def _modules():
    if "mods" not in _CACHE:
        import latents
        import setup_params
        import simulate
        _CACHE["mods"] = (setup_params, latents, simulate)
    return _CACHE["mods"]


def _reference(size: int = SIZE) -> dict[str, Any]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    T = np.clip(0.2 + 0.7 * xx / (size - 1), 0.0, 1.0).astype(np.float32)
    phi = (0.8 * yy / (size - 1) + 0.3 * (xx > size / 2)).astype(np.float32)
    one = np.ones((size, size), dtype=np.float32)
    return {"M": one, "T": T, "phi": phi, "O_i": one, "D": np.zeros_like(one), "target_size": size}


def _strip(meta: Any) -> Any:
    if isinstance(meta, dict):
        return {k: _strip(v) for k, v in meta.items() if k not in VOLATILE_META_KEYS}
    if isinstance(meta, (list, tuple)):
        return [_strip(v) for v in meta]
    if isinstance(meta, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(meta).tobytes()).hexdigest()
    if isinstance(meta, float) and not math.isfinite(meta):
        return str(meta)
    return meta


def _digest(setup_over: dict[str, Any], state_kw: dict[str, Any], profile: str,
            size: int = SIZE) -> tuple[str, str]:
    setup_params, latents, simulate = _modules()
    base = setup_params.resolve_setup_selection(profile)
    setup = base.overridden(**setup_over) if setup_over else base
    kw = {**BASE_LATENTS, **state_kw}
    state = latents.LatentState(setup=setup, **kw)
    res = simulate.simulate_forward(_reference(size), state, seed=SEED, detector_noise=True,
                                    raise_on_forbidden=False)
    h = hashlib.sha256()
    for key in ARRAY_KEYS:
        arr = getattr(res, key, None)
        if arr is not None:
            h.update(key.encode())
            h.update(np.ascontiguousarray(arr).tobytes())
    meta = json.dumps(_strip(res.meta), sort_keys=True, default=str)
    return h.hexdigest(), hashlib.sha256(meta.encode()).hexdigest()


def _string_candidates(value: str) -> list[str]:
    setup_params, _, _ = _modules()
    found: list[str] = []
    for obj in vars(setup_params).values():
        members: list[Any] = []
        if isinstance(obj, (set, frozenset, tuple, list)):
            members = list(obj)
        elif isinstance(obj, dict):
            members = list(obj.keys())
        if value in members and all(isinstance(m, str) for m in members) and len(members) <= 40:
            found.extend(m for m in sorted(members) if m != value and m not in found)
    return found[:6] + [value + "_audit_variant"]


def _candidates(name: str, value: Any) -> list[Any]:
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1, value * 2] if value else [1, 2]
    if isinstance(value, float):
        if value == 0.0:
            # 1000 resolves length offsets that are tiny relative to the scale
            # they act on (e.g. a pump focus shift against a ~0.4 m Rayleigh range).
            return [1.0, 1000.0, -1.0, 0.1]
        return [value * 1.1, value * 0.9, value * 0.5, value + 0.05, value - 0.05]
    if isinstance(value, str):
        return _string_candidates(value)
    if value is None:
        return [1.0, 0.5, 5.0, 100.0, 250.0, 2, TEST_ID, True]
    return []


def _audit_one(task: tuple[str, str, str]) -> tuple[str, str, str, dict[str, Any]]:
    owner, name, profile = task
    setup_params, latents, _ = _modules()
    scen = scenarios(profile)
    base_setup = setup_params.resolve_setup_selection(profile)
    outcome: dict[str, Any] = {"status": "not_perturbable", "scenario": None, "value": None,
                               "tried": 0, "rejected": 0, "accepted": 0}
    meta_changed = None
    for sname, spec in scen.items():
        key = f"base::{profile}::{sname}"
        size = spec.get("size", SIZE)
        if key not in _CACHE:
            try:
                _CACHE[key] = _digest(spec["setup"], spec["state"], profile, size)
            except Exception as exc:  # scenario not admissible for this profile
                _CACHE[key] = ("ERROR", repr(exc))
        base = _CACHE[key]
        if base[0] == "ERROR":
            continue
        if owner == "SetupParams":
            current = getattr(base_setup.overridden(**spec["setup"]) if spec["setup"] else base_setup, name)
        else:
            current = {**BASE_LATENTS, **spec["state"]}.get(name, getattr(latents.LatentState(), name, None))
        accepted_here = 0
        for cand in _candidates(name, current):
            outcome["tried"] += 1
            try:
                if owner == "SetupParams":
                    dig = _digest({**spec["setup"], name: cand}, spec["state"], profile, size)
                else:
                    dig = _digest(spec["setup"], {**spec["state"], name: cand}, profile, size)
            except Exception:
                outcome["rejected"] += 1
                continue
            outcome["accepted"] += 1
            accepted_here += 1
            if dig[0] != base[0]:
                outcome.update(status="physics", scenario=sname, value=repr(cand))
                return owner, name, profile, outcome
            if dig[1] != base[1] and meta_changed is None:
                meta_changed = (sname, repr(cand))
            if accepted_here >= 3:
                break
    if meta_changed is not None:
        outcome.update(status="metadata_only", scenario=meta_changed[0], value=meta_changed[1])
    elif outcome["accepted"] > 0:
        outcome["status"] = "not_consumed"
    elif outcome["tried"] > 0 and outcome["rejected"] == outcome["tried"]:
        outcome["status"] = "validated_fixed"
    return owner, name, profile, outcome


def source_hashes() -> dict[str, str]:
    from provenance import hash_source_modules
    return hash_source_modules(ROOT)


def run_audit(workers: int) -> dict[str, Any]:
    from latents import LatentState
    from setup_params import SetupParams
    tasks = [("SetupParams", f.name, p) for f in fields(SetupParams) for p in PROFILES]
    tasks += [("LatentState", f.name, p) for f in fields(LatentState) if f.name not in LATENT_SKIP
              for p in PROFILES]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for owner, name, profile, outcome in pool.map(_audit_one, tasks, chunksize=4):
            result.setdefault(f"{owner}.{name}", {})[profile] = outcome
    return {
        "schema": "qiup_parameter_consumption_audit_v1",
        "package_version": __import__("provenance").PACKAGE_VERSION,
        "method": (
            "perturb one field at a time in fixed-seed forward runs across profile scenarios; "
            "compare SHA-256 of simulated arrays (physics) and of non-volatile run metadata"
        ),
        "status_meanings": {
            "physics": "a perturbation changes simulated arrays",
            "validated_fixed": "every perturbation is rejected by profile validation",
            "metadata_only": "arrays unchanged; run metadata changes",
            "not_consumed": "accepted perturbations change nothing",
            "not_perturbable": "no admissible perturbation value found",
        },
        "seed": SEED,
        "grid_size": SIZE,
        "detector_stress_grid_size": DETECTOR_STRESS_SIZE,
        "scenarios": {p: {name: {"setup": spec["setup"], "state": spec["state"],
                                 "grid_size": spec.get("size", SIZE)}
                          for name, spec in scenarios(p).items()} for p in PROFILES},
        "software_test_calibration_values": {k: v for k, v in CALIBRATED.items()},
        "source_sha256": source_hashes(),
        "fields": dict(sorted(result.items())),
    }


def profiles_with_status(entry: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [p for p in PROFILES if entry[p]["status"] == status]


def applicable_profiles(entry: dict[str, dict[str, Any]]) -> list[str]:
    """Profiles in which the field has a detected effect on simulated arrays."""
    return profiles_with_status(entry, "physics")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    audit = run_audit(args.workers)
    target = ROOT / AUDIT_RELATIVE
    target.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    counts: dict[str, dict[str, int]] = {}
    for entry in audit["fields"].values():
        for p, o in entry.items():
            counts.setdefault(p, {}).setdefault(o["status"], 0)
            counts[p][o["status"]] += 1
    print(json.dumps(counts, indent=1))
    print(f"wrote {AUDIT_RELATIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
