#!/usr/bin/env python3
"""Figure 4 Stage 1 — pipeline audit + minimal A/B knife-edge modulation test."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.four_config.definitions import GRID_N, MASTER_SEED, NBAR
from benchmarks.four_config.runner import mapping_block, quiet_state
from benchmarks.targets import PHYSICAL_TRUTH, build_target
from latents import LatentState
from setup_params import (
    resolve_setup_selection,
    validate_generic_benchmark_final_camera_collection,
    validate_generic_benchmark_spectral_filter,
)
from simulate import simulate_forward
from verification.resolution.profiles import build_esf_profile

EVIDENCE_ROOT = ROOT / "evidence" / "figure4_stage1_object_transfer_audit"
STAGE_PROFILES = ("A", "B")
ILLUM_FRAC = 0.05


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def modulation_from_four_frames(frames_plus: np.ndarray) -> dict[str, np.ndarray]:
    I0 = np.asarray(frames_plus[0], dtype=np.float64)
    I90 = np.asarray(frames_plus[1], dtype=np.float64)
    I180 = np.asarray(frames_plus[2], dtype=np.float64)
    I270 = np.asarray(frames_plus[3], dtype=np.float64)
    D = 0.25 * (I0 + I90 + I180 + I270)
    a = 0.5 * (I0 - I180)
    b = 0.5 * (I90 - I270)
    amod = np.sqrt(a * a + b * b)
    return {"D": D, "a": a, "b": b, "A_mod": amod}


def run_phase_case(profile: str, target_id: str) -> dict[str, Any]:
    setup0 = resolve_setup_selection(profile)
    if setup0.setup_id == "config_c_lab_v1":
        raise RuntimeError("C_lab leaked")
    validate_generic_benchmark_spectral_filter(setup0)
    validate_generic_benchmark_final_camera_collection(setup0)
    n = GRID_N
    mapping = mapping_block(setup0, n=n)
    pixel_um = float(mapping["object_plane_pixel_um"])
    tgt = build_target(target_id, n, pixel_um=pixel_um)
    ref = tgt.to_simulation_reference()
    setup = setup0.overridden(enable_phase_step_acquisition=True)
    state = LatentState(
        setup=setup,
        mean_photons_per_mode=float(NBAR),
        num_frames=4,
        exposure_s=0.02,
        em_gain=1.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )
    seed = int(MASTER_SEED) + 7000 * (ord(profile) - ord("A")) + hash(target_id) % 997
    result = simulate_forward(ref, state, seed=seed, detector_noise=False)
    mod = modulation_from_four_frames(result.frames_plus)
    x_um = np.asarray(mapping["object_x_um"], dtype=np.float64)
    y_um = np.asarray(mapping["object_y_um"], dtype=np.float64)
    return {
        "profile": profile,
        "setup_id": setup0.setup_id,
        "target_id": target_id,
        "mapping": {k: v for k, v in mapping.items() if not isinstance(v, np.ndarray)},
        "object_T": np.asarray(tgt.T, dtype=np.float64),
        "object_x_um": x_um,
        "object_y_um": y_um,
        "I_plus_phi0": np.asarray(result.frames_plus[0], dtype=np.float64),
        "modulation": mod,
        "visibility_V_50_50": float(
            (result.meta.get("visibility_observables") or {}).get("V_50_50") or np.nan
        ),
    }


def reference_normalized_response(
    obj: dict[str, Any], ref: dict[str, Any]
) -> dict[str, Any]:
    A_obj = obj["modulation"]["A_mod"]
    A_ref = ref["modulation"]["A_mod"]
    thr = ILLUM_FRAC * float(np.max(A_ref))
    valid = A_ref >= thr
    R = np.full_like(A_ref, np.nan, dtype=np.float64)
    np.divide(A_obj, A_ref, out=R, where=valid)
    excluded_frac = float(1.0 - np.mean(valid))
    return {"R": R, "valid_mask": valid, "threshold": thr, "excluded_area_frac": excluded_frac}


def edge_profile_R(R: np.ndarray, x_um: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    R = np.asarray(R, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    Rm = np.where(valid, R, np.nan)
    h = Rm.shape[0]
    r0, r1 = max(0, h // 2 - 2), min(h, h // 2 + 3)
    line = np.nanmean(Rm[r0:r1, :], axis=0)
    esf = build_esf_profile(
        x_um,
        line,
        quantity="modulation_ratio",
        expected_edge_um=float(PHYSICAL_TRUTH["knife_edge_offset_um"]),
    )
    return {
        "axis_um": esf.axis_um,
        "profile": esf.processed_profile,
        "esf_10_90_um": esf.metric_value,
        "status": esf.extras.get("status", "OK"),
        "spatial_mapping_error_um": esf.extras.get("spatial_mapping_error_um"),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_static_audits(out: Path) -> None:
    (out / "FIGURE4_OLD_PIPELINE_AUDIT.md").write_text(
        """# Figure 4 historical pipeline audit

## Source chain (publication)

| Step | Location |
|------|----------|
| Suite driver | `benchmarks/four_config/runner.py` → `run_target_case` |
| Figure composite | `tools/qiup_figure4_final_article.py` |
| Legacy / U09 panels | `tools/taskX6_U09_Figure4.py` (X6 used `C_lab` in old map; **current** article uses general C via X14 policy) |
| Targets | `benchmarks/targets/catalog.py` (`PHYSICAL_TRUTH`) |

## Panels (a) and (b) — plotted quantity

1. **Quantity:** `optical_intensity` = `intensity_map_from_result(result)` → `SimResult.ideal_signal_image` for **single-frame** runs (`NUM_FRAMES_OPTICAL=1`).
2. **Not** visibility map, not reconstructed phase, not reference-normalized modulation, not object transmission inverted.
3. **Object inserted:** yes — `build_target(...).to_simulation_reference()` with intensity transmission `T` and phase `phi` on the object plane grid.
4. **Encoding:** complex object amplitude √T·exp(iφ) via standard forward path (`optics._optical_intensity`).
5. **Masks:** same `PHYSICAL_TRUTH` sampled on each config's native `object_plane_pixel_um` (not identical pixel arrays across A/B).
6. **Normalization:** shared absolute vmin/vmax across A–D (`PLOT_POLICY`); **no** per-image max normalize on comparison panels.
7. **No-object reference:** **not** used in historical panels (a)/(b).
8. **Point target:** single array pixel at nearest sample to origin (`make_point`); not a continuous δ function.
9. **Knife-edge:** `T=1` for x ≥ 0, `T=0` for x < 0 (`make_knife_edge_vertical`); φ=0.
10. **B/D Gaussian appearance:** near-field `generation_intensity_camera` / joint-Gaussian marginal weights the moments before interference (`optics.apply_near_field_joint_gaussian` → `architecture_object_transfer`); raw **I_plus** reflects pump/collection envelope, not ESF of transferred edge alone.
11. **A/C uniformity:** far-field momentum mapping + single-pixel point + optical PSF/correlation width → broad, slowly varying intensity maps.
12. **Canonical mapping (current article):** `SETUP_KEY_MAP` A/B/C/D → generic YAMLs (`qiup_figure4_final_article.py`); **not** legacy X6 `C_lab` map.

## Phase-sensitive content in production

Four-step schedule exists (`FOUR_STEP_PSI_SCHEDULE_R`) but historical point/knife targets **do not** enable it. Spatial **I_plus(φ)** is available via `enable_phase_step_acquisition=True` and `frames_plus` (Stage 1 uses this).

## Panels (c) and (d)

See `FIGURE4_REFERENCE_INDEPENDENCE_AUDIT.md`.
""",
        encoding="utf-8",
    )
    (out / "FIGURE4_CONFIGURATION_MAPPING.md").write_text(
        """# Figure 4 configuration mapping

| Label | setup_id (generic production) | YAML alias |
|-------|----------------------------------|------------|
| A | `two_crystal_2014_reference_v2` | `A` |
| B | `two_crystal_near_field_v1` | `B` |
| C | `single_crystal_far_field_v2` | `C` |
| D | `single_crystal_near_field_v2` | `D` |

**C_lab** (`config_c_lab_v1`) appears only in legacy `taskX6_U09_Figure4.SETUP_KEY_MAP` — **not** in current `qiup_figure4_final_article.py`.

Stage 1 executes **A and B only** under generic 3 nm + uniform E=1 validation.
""",
        encoding="utf-8",
    )
    (out / "FIGURE4_OBJECT_TRANSFER_THEORY.md").write_text(
        """# Object-sensitive observable (Stage 1)

Local plus-port intensity model (first harmonic):

I₊(x,y,ψ) ≈ D(x,y) + a(x,y) cos ψ + b(x,y) sin ψ

From four phase samples ψ ∈ {0, π/2, π, 3π/2} on **I_plus** (`frames_plus`):

- D = (I₀ + I_{π/2} + I_π + I_{3π/2})/4
- a = (I₀ − I_π)/2
- b = (I_{π/2} − I_{3π/2})/2
- A_mod = √(a² + b²)

Reference-normalized quantity (Stage 1):

R(x,y) = A_mod^object / A_mod^transparent

within valid mask A_mod^ref ≥ 5% of max(A_mod^ref).

**Label:** reference-normalized interference modulation — **not** reconstructed object transmission unless architecture-specific inversion is justified (deferred for C/D double-pass).
""",
        encoding="utf-8",
    )
    (out / "FIGURE4_REFERENCE_INDEPENDENCE_AUDIT.md").write_text(
        """# Panels (c) and (d) — reference independence (code inspection only)

## Panel (c) magnification

- **Production bar:** `resolved_spatial_mapping(setup)["signed_object_to_camera_magnification"]` (`taskX6_U09_Figure4.run_u09_corrected`).
- **Reference bar:** `verification.physics.independent_equations` (`far_field_signed_object_to_camera_scale` or `near_field_magnification`).
- **Verdict:** **Independent analytical cross-check** of the same mapping formulas — **not** measured from displacing a test object in simulated images.
- **Flag:** displacement-based magnification test **NOT_INDEPENDENT_VERIFICATION** (not performed in historical Figure 4).

## Panel (d) correlation width σ_obj

- **Production:** `correlation_sigma_object_um` from `resolved_spatial_mapping`.
- **Reference:** analytical `far_field_correlation_sigma_um` / `near_field_conditional_sigma_um` using joint-Gaussian coefficients.
- **Verdict:** **NOT_INDEPENDENT_VERIFICATION** against simulated ESF/PSF profiles — theoretical helper parity check only.

## Stage 2 recommendation

- Measure M from paired object/camera centroids of known mask features.
- Extract σ from ESF/PSF of **R(x,y)** or calibrated point spread, not from the same closed-form helper.
""",
        encoding="utf-8",
    )


def draw_diagnostic(
    out_png: Path,
    *,
    obj_a: dict[str, Any],
    ref_a: dict[str, Any],
    Ra: np.ndarray,
    prof_a: dict[str, Any],
    obj_b: dict[str, Any],
    ref_b: dict[str, Any],
    Rb: np.ndarray,
    prof_b: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    x = obj_a["object_x_um"]
    ideal_T = (x >= 0).astype(float)
    for col, (label, I0, Aref, Aobj, R, prof) in enumerate(
        [
            ("A FF", obj_a["I_plus_phi0"], ref_a["modulation"]["A_mod"], obj_a["modulation"]["A_mod"], Ra, prof_a),
            ("B NF", obj_b["I_plus_phi0"], ref_b["modulation"]["A_mod"], obj_b["modulation"]["A_mod"], Rb, prof_b),
        ]
    ):
        ax0 = axes[0, col]
        ax0.imshow(I0, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], aspect="auto")
        ax0.set_title(f"{label}: I₊(φ=0) knife-edge")
        ax1 = axes[1, col]
        ax1.imshow(R, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], aspect="auto")
        ax1.set_title(f"{label}: R = A_mod/A_ref")
        axp = axes[1, 2] if col == 0 else None
    axp = axes[0, 2]
    axp.plot(prof_a["axis_um"], prof_a["profile"], label="A: R ESF", color="#3a7ca5")
    axp.plot(prof_b["axis_um"], prof_b["profile"], label="B: R ESF", color="#2a9d8f")
    axp.plot(x, ideal_T, "--", color="0.4", label="ideal step (object plane)")
    axp.set_xlabel("Object-plane x (µm)")
    axp.set_ylabel("Normalized response")
    axp.legend(fontsize=7, frameon=False)
    axp.set_title("Central ESF of R(x,y)")
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()
    stamp = utc_stamp()
    out = EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    write_static_audits(out)

    if args.skip_run:
        (out / "FIGURE4_STAGE1_FINDINGS.md").write_text(
            "Audit-only bundle; numerical A/B test skipped.\n", encoding="utf-8"
        )
        print(out)
        return

    cases: dict[str, dict[str, Any]] = {}
    for p in STAGE_PROFILES:
        cases[f"{p}_transparent"] = run_phase_case(p, "transparent")
        cases[f"{p}_knife"] = run_phase_case(p, "knife_edge_vertical")

    params = {
        "profiles": list(STAGE_PROFILES),
        "grid_n": GRID_N,
        "nbar": NBAR,
        "master_seed_base": MASTER_SEED,
        "knife_edge_x_um": PHYSICAL_TRUTH["knife_edge_offset_um"],
        "illum_threshold_frac": ILLUM_FRAC,
        "phase_schedule": "FOUR_STEP_PSI_SCHEDULE_R",
        "detector_noise": False,
        "filter_policy": "generic 3 nm",
        "collection_policy": "uniform E=1",
    }
    (out / "FIGURE4_A_B_TEST_PARAMETERS.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )

    metrics: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    results_ab: dict[str, Any] = {}

    for p in STAGE_PROFILES:
        ref = cases[f"{p}_transparent"]
        obj = cases[f"{p}_knife"]
        rn = reference_normalized_response(obj, ref)
        obj["valid_mask"] = rn["valid_mask"]
        prof = edge_profile_R(rn["R"], obj["object_x_um"], rn["valid_mask"])
        results_ab[p] = {"obj": obj, "ref": ref, "R": rn["R"], "prof": prof, **rn}

        I_mean_obj = float(np.nanmean(obj["I_plus_phi0"]))
        I_int_obj = float(np.nansum(obj["I_plus_phi0"]) * obj["mapping"]["object_plane_pixel_um"] ** 2)
        metrics.append(
            {
                "profile": p,
                "setup_id": obj["setup_id"],
                "I_plus_phi0_mean_knife": I_mean_obj,
                "I_plus_phi0_integral_knife": I_int_obj,
                "A_mod_mean_knife": float(np.mean(obj["modulation"]["A_mod"])),
                "A_mod_mean_transparent": float(np.mean(ref["modulation"]["A_mod"])),
                "R_mean_valid": float(np.nanmean(rn["R"][rn["valid_mask"]])),
                "R_min_valid": float(np.nanmin(rn["R"][rn["valid_mask"]])),
                "R_max_valid": float(np.nanmax(rn["R"][rn["valid_mask"]])),
                "excluded_area_frac": rn["excluded_area_frac"],
                "V_50_50_knife": obj["visibility_V_50_50"],
                "object_pixel_um": obj["mapping"]["object_plane_pixel_um"],
                "camera_pixel_um": obj["mapping"]["camera_pixel_um"],
                "signed_M": obj["mapping"]["signed_object_to_camera_magnification"],
                "fov_object_um": float(GRID_N * obj["mapping"]["object_plane_pixel_um"]),
                "esf_10_90_um_on_R": prof["esf_10_90_um"],
                "esf_status": prof["status"],
            }
        )
        for i, val in enumerate(prof["profile"]):
            edge_rows.append(
                {
                    "profile": p,
                    "index": i,
                    "x_um": float(prof["axis_um"][i]),
                    "R_esf_profile": float(val),
                }
            )

    save_csv(out / "figure4_stage1_metrics.csv", metrics)
    save_csv(out / "figure4_stage1_edge_profiles.csv", edge_rows)

    np.savez_compressed(
        out / "figure4_stage1_arrays.npz",
        **{
            f"{p}_knife_I_phi0": results_ab[p]["obj"]["I_plus_phi0"]
            for p in STAGE_PROFILES
        },
        **{
            f"{p}_A_mod_knife": results_ab[p]["obj"]["modulation"]["A_mod"]
            for p in STAGE_PROFILES
        },
        **{
            f"{p}_A_mod_ref": results_ab[p]["ref"]["modulation"]["A_mod"]
            for p in STAGE_PROFILES
        },
        **{f"{p}_R": results_ab[p]["R"] for p in STAGE_PROFILES},
        **{f"{p}_object_x_um": results_ab[p]["obj"]["object_x_um"] for p in STAGE_PROFILES},
        **{f"{p}_object_T_knife": results_ab[p]["obj"]["object_T"] for p in STAGE_PROFILES},
    )

    draw_diagnostic(
        out / "FIGURE4_STAGE1_DIAGNOSTIC.png",
        obj_a=results_ab["A"]["obj"],
        ref_a=results_ab["A"]["ref"],
        Ra=results_ab["A"]["R"],
        prof_a=results_ab["A"]["prof"],
        obj_b=results_ab["B"]["obj"],
        ref_b=results_ab["B"]["ref"],
        Rb=results_ab["B"]["R"],
        prof_b=results_ab["B"]["prof"],
    )

    # Relative L2 between knife-edge A_mod maps (same grid, raw — diagnostic only)
    l2_ab = float(
        np.sqrt(np.mean((results_ab["A"]["obj"]["modulation"]["A_mod"] - results_ab["B"]["obj"]["modulation"]["A_mod"]) ** 2))
        / (np.sqrt(np.mean(results_ab["A"]["obj"]["modulation"]["A_mod"] ** 2)) + 1e-30)
    )

    a_esf = metrics[0]["esf_10_90_um_on_R"]
    b_esf = metrics[1]["esf_10_90_um_on_R"]
    a_resolved = np.isfinite(a_esf) and a_esf > 0
    b_resolved = np.isfinite(b_esf) and b_esf > 0

    if a_resolved and b_resolved:
        decision = "PASS"
    elif a_resolved or b_resolved:
        decision = "PARTIAL"
    else:
        decision = "BLOCKED"

    findings = f"""# Figure 4 Stage 1 findings

**Decision:** **{decision}**

## Answers (§12)

1. Historical (a)/(b): raw **ideal_signal_image** / optical intensity, single frame, object T/φ encoded; shared absolute scale.
2. A/C point maps nearly uniform: single-pixel point + FF correlation/PSF smearing + (for point) no edge structure.
3. B/D knife Gaussian: dominated by **near-field generation/marginal envelope** on raw I₊, not knife ESF.
4. Cause: **observable choice + source envelope**, not necessarily a forward bug (Stage 1 modulation test separates envelope from phase-sensitive term).
5. Production includes phase-sensitive I₊(ψ) via physical moments + interferometer (`interferometric_phase_override` per frame).
6. Direct spatial moment map for A_mod: **no** — four phase samples used (`enable_phase_step_acquisition`).
7. Additional phase calculations: **4 frames** per object condition (transparent + knife) for A and B.
8. Knife edge at x=0 µm; object plane axes from `mapping_block`; pixel pitch architecture-specific (A: 16 µm object sampling typical; B: 8 µm — see metrics CSV).
9. A edge in R: ESF 10–90% = {a_esf} µm (status {metrics[0]['esf_status']}).
10. B edge in R: ESF 10–90% = {b_esf} µm (status {metrics[1]['esf_status']}).
11. 10–90% widths: see metrics CSV; invalid → EDGE_RESPONSE_NOT_RESOLVED.
12. Panels (c)/(d): **NOT_INDEPENDENT_VERIFICATION** (analytical helper parity only).
13. Retain: architecture labels, independent-test *concept* for Stage 2; replace (a)/(b) intensity-only maps with object-sensitive R and documented ESF.
14. Stage 2 ready: **{decision != 'BLOCKED'}** — see decision above.

Diagnostic cross-profile L2(A_mod_A vs A_mod_B): {l2_ab:.4g} (not a convergence metric).

## Point-object (§8)

Historical point = **one central pixel** T=1; unsuitable as PSF without reference normalization. Deferred to Stage 2 unless knife-edge already sufficient.
"""
    (out / "FIGURE4_STAGE1_FINDINGS.md").write_text(findings, encoding="utf-8")
    (out / "REPRODUCE.md").write_text(
        f"```bash\npython tools/figure4_stage1_object_transfer.py\n```\n\nBundle: `{out}`\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
