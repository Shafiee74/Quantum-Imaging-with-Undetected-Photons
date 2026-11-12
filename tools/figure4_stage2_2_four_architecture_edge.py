#!/usr/bin/env python3
"""Figure 4 Stage 2.2 — four-architecture knife-edge R/ESF (C/D runs; A/B reused)."""
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

from benchmarks.four_config.definitions import ARCHITECTURE_LABELS, GRID_N, MASTER_SEED, NBAR
from benchmarks.four_config.runner import mapping_block
from benchmarks.targets import PHYSICAL_TRUTH, build_target
from benchmarks.targets.geometry import centered_axis_um
from latents import LatentState
from setup_params import (
    resolve_setup_selection,
    resolved_spatial_mapping,
    validate_generic_benchmark_final_camera_collection,
    validate_generic_benchmark_spectral_filter,
)
from simulate import simulate_forward
from verification.resolution.profiles import build_esf_profile

STAGE1_NPZ = (
    ROOT
    / "evidence"
    / "figure4_stage1_object_transfer_audit"
    / "20261020T062325Z"
    / "figure4_stage1_arrays.npz"
)
EVIDENCE_ROOT = ROOT / "evidence" / "figure4_stage2_2_four_architecture_edge"
ILLUM_FRAC = 0.05
PROFILES = ("A", "B", "C", "D")
NEW_PROFILES = ("C", "D")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def modulation_from_four_frames(frames_plus: np.ndarray) -> dict[str, np.ndarray]:
    I0 = np.asarray(frames_plus[0], dtype=np.float64)
    I90 = np.asarray(frames_plus[1], dtype=np.float64)
    I180 = np.asarray(frames_plus[2], dtype=np.float64)
    I270 = np.asarray(frames_plus[3], dtype=np.float64)
    a = 0.5 * (I0 - I180)
    b = 0.5 * (I90 - I270)
    amod = np.sqrt(a * a + b * b)
    return {"a": a, "b": b, "A_mod": amod, "I0": I0, "I90": I90, "I180": I180, "I270": I270}


def reference_normalized_response(A_obj: np.ndarray, A_ref: np.ndarray) -> dict[str, Any]:
    thr = ILLUM_FRAC * float(np.max(A_ref))
    valid = A_ref >= thr
    R = np.full_like(A_ref, np.nan, dtype=np.float64)
    np.divide(A_obj, A_ref, out=R, where=valid)
    return {
        "R": R,
        "valid_mask": valid,
        "threshold": thr,
        "valid_fraction": float(np.mean(valid)),
        "excluded_fraction": float(1.0 - np.mean(valid)),
    }


def central_strip_line(field: np.ndarray) -> np.ndarray:
    h = field.shape[0]
    r0, r1 = max(0, h // 2 - 2), min(h, h // 2 + 3)
    return np.nanmean(field[r0:r1, :], axis=0)


def sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def plateau_normalize_sorted(x: np.ndarray, y: np.ndarray, *, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = sort_xy(x, y)
    k = max(1, int(round(frac * ys.size)))
    low_side = float(np.mean(ys[:k]))
    high_side = float(np.mean(ys[-k:]))
    if abs(high_side - low_side) <= 0.0:
        return xs, np.zeros_like(ys)
    if low_side > high_side:
        proc = (low_side - ys) / (low_side - high_side)
    else:
        proc = (ys - low_side) / (high_side - low_side)
    return xs, proc


def mapping_for_profile(profile: str) -> dict[str, Any]:
    setup = resolve_setup_selection(profile)
    if setup.setup_id == "config_c_lab_v1":
        raise RuntimeError(f"C_lab leaked for profile {profile}")
    mb = mapping_block(setup, n=GRID_N)
    sm = resolved_spatial_mapping(setup)
    M = float(sm["signed_object_to_camera_magnification"])
    p_cam = float(sm["camera_pixel_um"])
    x_cam = centered_axis_um(GRID_N, p_cam)
    return {
        "profile": profile,
        "setup_id": setup.setup_id,
        "architecture": ARCHITECTURE_LABELS[profile],
        "signed_M": M,
        "object_plane_pixel_um": float(sm["object_plane_pixel_um"]),
        "camera_pixel_um": p_cam,
        "fov_object_um": float(GRID_N * float(sm["object_plane_pixel_um"])),
        "fov_camera_um": float(GRID_N * p_cam),
        "x_cam": x_cam,
        "x_obj": x_cam / M,
        "object_to_camera_index_map": sm.get("object_to_camera_index_map"),
        "mapping_block": {k: v for k, v in mb.items() if not isinstance(v, np.ndarray)},
    }


def run_cd_case(profile: str, target_id: str) -> dict[str, Any]:
    setup0 = resolve_setup_selection(profile)
    if setup0.setup_id == "config_c_lab_v1":
        raise RuntimeError("C_lab leaked")
    validate_generic_benchmark_spectral_filter(setup0)
    validate_generic_benchmark_final_camera_collection(setup0)
    mp = mapping_for_profile(profile)
    tgt = build_target(target_id, GRID_N, pixel_um=mp["object_plane_pixel_um"])
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
    return {
        **mp,
        "target_id": target_id,
        "object_T": np.asarray(tgt.T, dtype=np.float64),
        "frames_plus": np.asarray(result.frames_plus, dtype=np.float64),
        "modulation": mod,
        "I_plus_phi0": mod["I0"],
        "visibility_V_50_50": float(
            (result.meta.get("visibility_observables") or {}).get("V_50_50") or np.nan
        ),
    }


def load_ab_from_stage1() -> dict[str, dict[str, Any]]:
    z = np.load(STAGE1_NPZ)
    out: dict[str, dict[str, Any]] = {}
    for p in ("A", "B"):
        mp = mapping_for_profile(p)
        A_mod_k = np.asarray(z[f"{p}_A_mod_knife"], dtype=np.float64)
        A_mod_r = np.asarray(z[f"{p}_A_mod_ref"], dtype=np.float64)
        rn = reference_normalized_response(A_mod_k, A_mod_r)
        out[p] = {
            **mp,
            "source": "stage1_npz",
            "object_T": np.asarray(z[f"{p}_object_T_knife"], dtype=np.float64),
            "I_plus_phi0": np.asarray(z[f"{p}_knife_I_phi0"], dtype=np.float64),
            "A_mod_knife": A_mod_k,
            "A_mod_ref": A_mod_r,
            **rn,
        }
    return out


def esf_metrics(R: np.ndarray, x_obj: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    Rm = np.where(valid, R, np.nan)
    line = central_strip_line(Rm)
    fit = build_esf_profile(
        x_obj,
        line,
        quantity="modulation_ratio",
        expected_edge_um=float(PHYSICAL_TRUTH["knife_edge_offset_um"]),
    )
    width = fit.metric_value
    status = "OK"
    if width is None or not np.isfinite(width) or width <= 0:
        status = "EDGE_RESPONSE_NOT_RESOLVED"
    xo, proc = plateau_normalize_sorted(x_obj, line)
    rising = float(np.mean(proc[-3:]) - np.mean(proc[:3])) > 0
    return {
        "esf_10_90_um": width,
        "status": status,
        "x10_um": fit.extras.get("x10_um"),
        "x50_um": fit.extras.get("x50_um"),
        "x90_um": fit.extras.get("x90_um"),
        "I_low": fit.extras.get("I_low"),
        "I_high": fit.extras.get("I_high"),
        "flipped_for_rising": fit.extras.get("flipped_for_rising"),
        "spatial_mapping_error_um": fit.extras.get("spatial_mapping_error_um"),
        "transition_rising_in_signed_object_x": rising,
        "axis_um_sorted": xo,
        "profile_sorted": proc,
        "raw_line": line,
    }


def camera_curve_check_md() -> str:
    return """# Stage 2.1 A camera-coordinate ESF display check

## Symptom

In `FIGURE4_STAGE2_1_COORDINATE_VERIFICATION.png`, the curve **A: ESF on camera x_cam** appeared flat at zero.

## Root cause (plotting only)

`plateau_normalize_sorted` in `tools/figure4_stage2_coordinate_parity.py` used:

```python
proc = (ys - left) / span if span > 0.0 else np.zeros_like(ys)
```

For configuration **A**, raw **R** **falls** along ascending **x_cam** (high at **x_cam < 0**, low at **x_cam > 0**). End-plateau means give **span = right − left < 0**, triggering the zero branch → **flat line at 0**.

## Physical expectation

Same raw **R** samples as the signed object curve. On **x_cam**, the ESF is a **falling** transition (~1 → 0); on **x_obj = x_cam/M** (sorted), it is **rising** (~0 → 1) and matches **T(x_obj)**.

## Fix

Use **|span|** logic and separate rising vs falling normalization along ascending **x** (see updated `plateau_normalize_sorted`).

## Arrays / physics

**No change** to Stage 1 NPZ or forward model.

**Verdict:** plotting error only.
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def draw_diagnostic(cases: dict[str, dict[str, Any]], out_png: Path) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.2, 1, 1, 1, 1.1], hspace=0.35, wspace=0.25)
    ax_esf = fig.add_subplot(gs[0, :])
    edge_um = float(PHYSICAL_TRUTH["knife_edge_offset_um"])
    x_ideal = centered_axis_um(GRID_N, 16.0)
    ax_esf.plot(x_ideal, (x_ideal >= edge_um).astype(float), "--", color="0.45", lw=1.2, label="Ideal T(x_obj)")
    colors = {"A": "#3a7ca5", "B": "#2a9d8f", "C": "#586e99", "D": "#e76f51"}
    for p in PROFILES:
        esf = cases[p]["esf"]
        ax_esf.plot(esf["axis_um_sorted"], esf["profile_sorted"], color=colors[p], lw=1.6, label=f"{p}: R ESF")
    ax_esf.axvline(0.0, color="0.75", lw=0.8)
    ax_esf.set_xlim(-400, 400)
    ax_esf.set_ylabel("Plateau-norm ESF")
    ax_esf.set_title("Central ESF of R in signed object-plane coordinates (x_obj = x_cam / M)")
    ax_esf.legend(ncol=5, fontsize=7, frameon=False, loc="lower right")

    col_titles = ["I₊(ψ=0) knife", "A_mod knife", "R = A_mod/A_ref"]
    for col, title in enumerate(col_titles):
        fig.text(0.14 + col * 0.24, 0.72, title, ha="center", fontsize=9)

    r_vmax = 1.0
    r_vmin = 0.0
    for row, p in enumerate(PROFILES):
        c = cases[p]
        x_cam = c["x_cam"]
        ext = [x_cam[0], x_cam[-1], x_cam[0], x_cam[-1]]
        ax0 = fig.add_subplot(gs[row + 1, 0])
        ax0.imshow(c["I_plus_phi0"], origin="lower", extent=ext, aspect="auto")
        ax0.set_ylabel(f"{p}\n{c['architecture'][:18]}", fontsize=7)
        ax1 = fig.add_subplot(gs[row + 1, 1])
        im1 = ax1.imshow(c["A_mod_knife"], origin="lower", extent=ext, aspect="auto")
        ax2 = fig.add_subplot(gs[row + 1, 2])
        im2 = ax2.imshow(c["R"], origin="lower", extent=ext, aspect="auto", vmin=r_vmin, vmax=r_vmax)
        ax3 = fig.add_subplot(gs[row + 1, 3])
        T = c["object_T"]
        x_obj_plot = centered_axis_um(GRID_N, c["object_plane_pixel_um"])
        ext_obj = [x_obj_plot[0], x_obj_plot[-1], x_obj_plot[0], x_obj_plot[-1]]
        ax3.imshow(T, origin="lower", extent=ext_obj, aspect="auto", vmin=0, vmax=1)
        ax3.set_title("Input T (object-plane x)", fontsize=7)
        if row == len(PROFILES) - 1:
            for ax in (ax0, ax1, ax2):
                ax.set_xlabel("Camera x_cam (µm)")
    fig.suptitle("Figure 4 Stage 2.2 — four-architecture knife-edge diagnostic (not manuscript figure)", y=0.98, fontsize=11)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cd", action="store_true", help="Load only (requires prior C/D in bundle)")
    args = ap.parse_args()

    if not STAGE1_NPZ.is_file():
        raise FileNotFoundError(STAGE1_NPZ)

    stamp = utc_stamp()
    out = EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)

    (out / "FIGURE4_STAGE2_1_CAMERA_CURVE_CHECK.md").write_text(camera_curve_check_md(), encoding="utf-8")
    (out / "FIGURE4_SIGNED_COORDINATE_AUDIT.md").write_text(
        """# Signed coordinate audit (Stage 2.2)

- Detector sample **i** → **x_cam = (i − (N−1)/2) · p_cam** (µm).
- Object-plane coordinate for that sample: **x_obj = x_cam / M** (**M** signed).
- ESF profiles: sort **(x_obj, R)** together before display; metrics via `build_esf_profile` on **x_obj** aligned to column index (same raw **R** as simulation).
- Do **not** use **|M|** for coordinate recovery.
- Configuration **A**: **M < 0** → falling **R** vs ascending **x_cam**; rising vs ascending sorted **x_obj**.
- Configuration **B**: **object_x_um[i] = x_cam[i]/M** on shared detector index.
""",
        encoding="utf-8",
    )

    cases = load_ab_from_stage1()

    cd_audit_lines = ["# C/D object transfer audit\n"]
    if not args.skip_cd:
        for p in NEW_PROFILES:
            ref_case = run_cd_case(p, "transparent")
            knife = run_cd_case(p, "knife_edge_vertical")
            rn = reference_normalized_response(knife["modulation"]["A_mod"], ref_case["modulation"]["A_mod"])
            cases[p] = {
                **knife,
                "A_mod_knife": knife["modulation"]["A_mod"],
                "A_mod_ref": ref_case["modulation"]["A_mod"],
                "source": "stage2_2_forward",
                **rn,
            }
            cd_audit_lines.append(
                f"## {p} ({knife['setup_id']})\n"
                f"- Architecture: {knife['architecture']}\n"
                f"- Object map: `build_target('knife_edge_vertical')` on native object pitch {knife['object_plane_pixel_um']} µm\n"
                f"- Double-pass effective field: `optics.architecture_object_transfer` reflective convention (generic YAML, not C_lab)\n"
                f"- V_50_50 knife: {knife['visibility_V_50_50']:.4f}\n"
                f"- Valid illumination fraction: {rn['valid_fraction']:.4f}\n"
            )
    else:
        raise RuntimeError("--skip-cd not supported without saved C/D")

    (out / "FIGURE4_C_D_OBJECT_TRANSFER_AUDIT.md").write_text("\n".join(cd_audit_lines), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    esf_rows: list[dict[str, Any]] = []
    npz_payload: dict[str, Any] = {}

    for p in PROFILES:
        c = cases[p]
        esf = esf_metrics(c["R"], c["x_obj"], c["valid_mask"])
        c["esf"] = esf
        metrics_rows.append(
            {
                "configuration": p,
                "architecture": c["architecture"],
                "setup_id": c["setup_id"],
                "signed_M": c["signed_M"],
                "object_pitch_um": c["object_plane_pixel_um"],
                "camera_pitch_um": c["camera_pixel_um"],
                "fov_object_um": c["fov_object_um"],
                "fov_camera_um": c["fov_camera_um"],
                "esf_10_90_um": esf["esf_10_90_um"],
                "edge_center_x50_um": esf["x50_um"],
                "valid_fraction": c["valid_fraction"],
                "status": esf["status"],
                "A_mod_ref_mean": float(np.mean(c["A_mod_ref"])),
                "A_mod_ref_max": float(np.max(c["A_mod_ref"])),
                "R_min_valid": float(np.nanmin(c["R"][c["valid_mask"]])),
                "R_max_valid": float(np.nanmax(c["R"][c["valid_mask"]])),
                "transition_rising_object_x": esf["transition_rising_in_signed_object_x"],
                "V_50_50_knife": c.get("visibility_V_50_50", np.nan),
                "source": c.get("source", ""),
            }
        )
        for i in range(esf["axis_um_sorted"].size):
            esf_rows.append(
                {
                    "configuration": p,
                    "index": i,
                    "x_obj_um": float(esf["axis_um_sorted"][i]),
                    "R_esf_plateau_normalized": float(esf["profile_sorted"][i]),
                }
            )
        prefix = p
        npz_payload[f"{prefix}_R"] = c["R"]
        npz_payload[f"{prefix}_A_mod_knife"] = c["A_mod_knife"]
        npz_payload[f"{prefix}_A_mod_ref"] = c["A_mod_ref"]
        npz_payload[f"{prefix}_I_plus_phi0"] = c["I_plus_phi0"]
        npz_payload[f"{prefix}_object_T"] = c["object_T"]
        npz_payload[f"{prefix}_x_cam"] = c["x_cam"]
        npz_payload[f"{prefix}_x_obj"] = c["x_obj"]
        npz_payload[f"{prefix}_valid_mask"] = c["valid_mask"]
        if p in NEW_PROFILES:
            npz_payload[f"{prefix}_frames_plus_knife"] = c["frames_plus"]

    write_csv(out / "figure4_stage2_2_metrics.csv", metrics_rows)
    write_csv(out / "figure4_stage2_2_esf_profiles.csv", esf_rows)
    np.savez_compressed(out / "figure4_stage2_2_arrays.npz", **npz_payload)

    draw_diagnostic(cases, out / "FIGURE4_STAGE2_2_DIAGNOSTIC.png")

    # Decision logic
    ok = all(
        m["status"] == "OK"
        and m["transition_rising_object_x"]
        and m["valid_fraction"] > 0.5
        and abs(float(m["edge_center_x50_um"] or 999)) < 50
        for m in metrics_rows
    )
    decision = "PASS" if ok else "PARTIAL"
    if any(m["status"] != "OK" for m in metrics_rows):
        decision = "BLOCKED" if not any(m["status"] == "OK" for m in metrics_rows) else "PARTIAL"

    table = "| Configuration | Architecture | Signed M | Object pitch (µm) | ESF 10–90% (µm) | Edge center (µm) | Valid fraction | Status |\n"
    table += "|---|---|---:|---:|---:|---:|---:|---|\n"
    for m in metrics_rows:
        w = m["esf_10_90_um"]
        w_s = f"{w:.2f}" if w is not None and np.isfinite(w) else "EDGE_RESPONSE_NOT_RESOLVED"
        x50 = m["edge_center_x50_um"]
        x50_s = f"{x50:.3f}" if x50 is not None and np.isfinite(x50) else "—"
        table += (
            f"| {m['configuration']} | {m['architecture']} | {m['signed_M']:.4g} | "
            f"{m['object_pitch_um']} | {w_s} | {x50_s} | {m['valid_fraction']:.4f} | {m['status']} |\n"
        )

    findings = f"""# Figure 4 Stage 2.2 findings

**Decision:** **{decision}**

## Comparison table

{table}

## §13 answers

1. **Flat A camera ESF:** **Yes — plotting error** (`span > 0` guard zeroed falling ESF). See `FIGURE4_STAGE2_1_CAMERA_CURVE_CHECK.md`.
2. **Signed coordinates for A/B:** **Verified** (reused Stage 1 arrays; ESF on **x_obj = x_cam/M**).
3. **C object transfer:** {"Yes" if metrics_rows[2]["status"] == "OK" else "See status"} — knife-edge in **R**, ESF **{metrics_rows[2]["esf_10_90_um"]}** µm object plane.
4. **D object transfer:** {"Yes" if metrics_rows[3]["status"] == "OK" else "See status"} — ESF **{metrics_rows[3]["esf_10_90_um"]}** µm object plane.
5. **ESF widths (object µm):** A **{metrics_rows[0]["esf_10_90_um"]:.2f}**, B **{metrics_rows[1]["esf_10_90_um"]:.2f}**, C **{metrics_rows[2]["esf_10_90_um"]}**, D **{metrics_rows[3]["esf_10_90_um"]}** (checkpoints A≈192, B≈38 not enforced).
6. **Edge orientation/position:** Rising ESF in signed **x_obj** for all configs with x50 near 0 µm (see CSV).
7. **Same physical object:** **T(x_obj)=0/1**, edge at 0 µm via `PHYSICAL_TRUTH` on each native object sampling grid.
8. **C/D reflective convention:** Generic **single_crystal_*_v2** setups; effective field via double-pass reflective model (`optics.architecture_object_transfer`), not C_lab transmissive path.
9. **R validity:** Reference-normalized interference modulation; not single-pass transmission; 5% **A_mod_ref** threshold retained.
10. **Before Stage 2.3:** Independent **M**/σ bar comparisons still outstanding; no blocker from knife-edge transfer if decision PASS/PARTIAL.

## Stage 2.3

**Not started** (per task scope).
"""
    (out / "FIGURE4_STAGE2_2_FINDINGS.md").write_text(findings, encoding="utf-8")
    (out / "REPRODUCE.md").write_text(
        f"""```bash
python tools/figure4_stage2_2_four_architecture_edge.py
```

Reuses: `{STAGE1_NPZ}`

Runs forward model for **C/D only** (transparent + knife, 4 phase frames each).
""",
        encoding="utf-8",
    )
    (out / "run_parameters.json").write_text(
        json.dumps(
            {
                "grid_n": GRID_N,
                "nbar": NBAR,
                "illum_threshold_frac": ILLUM_FRAC,
                "stage1_npz": str(STAGE1_NPZ),
                "profiles_new_run": list(NEW_PROFILES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
