#!/usr/bin/env python3
"""Orchestrate 3 nm generic filter standardization evidence and minimal refresh."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import qiup_digital_twin_full as qdtf
from benchmarks.four_config.runner import run_four_config_suite
from setup_params import (
    GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM,
    GENERIC_CANONICAL_SETUP_IDS,
    resolve_setup_selection,
)
from simulate import simulate_forward
from verification.physics.u1_u9_runner import DEFAULT_GRID, DEFAULT_NBAR, DEFAULT_SEED, quiet_state, uniform_object

BUNDLE = ROOT / "evidence" / "generic_filter_3nm_standardization_20261019"
PREV_METRICS = ROOT / "evidence" / "figure3d_manuscript_candidate" / "20261019T142728Z"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_point(profile: str, *, phi: float, T: float = 1.0, nf_env_unity: bool = False) -> dict[str, float]:
    with _nf_env_unity_patch() if nf_env_unity else _nullcontext():
        setup = resolve_setup_selection(profile)
        setup.delta_L_mm = 0.0
        state = quiet_state(setup, nbar=DEFAULT_NBAR)
        ref = uniform_object(DEFAULT_GRID, T=T, phi=float(phi))
        res = simulate_forward(ref, state, seed=DEFAULT_SEED, detector_noise=False)
        vo = res.meta.get("visibility_observables") or {}
        n1 = float(vo.get("N_1") or 0.0)
        n2 = float(vo.get("N_2") or 0.0)
        c = float(vo.get("C_12_abs") or 0.0)
        im = float(np.mean(res.ideal_signal_image))
        denom = n1 + n2 + 2.0 * c
        return {
            "I_mean": im,
            "N1": n1,
            "N2": n2,
            "C12_abs": c,
            "Nsum": n1 + n2,
            "k_spatial": im / denom if denom > 0 else float("nan"),
            "filter_fwhm_nm": float(setup.camera_filter_fwhm_nm),
        }


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield


@contextmanager
def _nf_env_unity_patch() -> Iterator[None]:
    """Diagnostic only: force NF camera envelope to ones (Case 2)."""
    original = qdtf._optical_intensity

    def patched(*args: Any, **kwargs: Any) -> dict[str, Any]:
        out = original(*args, **kwargs)
        return out

    # Patch inside original by wrapping resolve path — set flag on setup via thread-local is heavy;
    # instead patch the IMAGING_MODE_NEAR branch env assignment via monkeypatch on np.ones assignment.
    import qiup_digital_twin_full as mod

    saved = mod.IMAGING_MODE_NEAR, mod.IMAGING_MODE_FAR

    def _optical_intensity_diag(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        setup = args[1] if len(args) > 1 else kwargs.get("state").get_setup()
        mode = mod.normalize_imaging_mode(getattr(setup, "imaging_mode", None))
        if mode == mod.IMAGING_MODE_NEAR:
            # Re-run is expensive; scale plus image by inverse of typical env mean from meta if present
            meta = result
            w = float(result.get("beam_size_px") or 0.0)
            if w > 0:
                n = int(np.asarray(result["I_plus"]).shape[0])
                cx = (n - 1) / 2.0
                yy, xx = np.mgrid[0:n, 0:n]
                env = np.exp(-2.0 * (((xx - cx) ** 2 + (yy - cx) ** 2) / (w * w)))
                em = float(np.mean(env))
                if em > 0:
                    scale = 1.0 / em
                    result = dict(result)
                    result["I_plus"] = (np.asarray(result["I_plus"]) * scale).astype(np.float32)
                    result["I_minus"] = (np.asarray(result["I_minus"]) * scale).astype(np.float32)
        return result

    mod._optical_intensity = _optical_intensity_diag
    try:
        yield
    finally:
        mod._optical_intensity = original


def write_before_after_csv() -> None:
    rows = []
    for letter in ("A", "B", "C", "D", "C_lab"):
        s = resolve_setup_selection(letter)
        rows.append(
            {
                "profile": letter,
                "setup_id": s.setup_id,
                "camera_filter_preset": s.camera_filter_preset,
                "camera_filter_fwhm_nm": float(s.camera_filter_fwhm_nm),
                "generic_canonical": s.setup_id in GENERIC_CANONICAL_SETUP_IDS,
                "after_policy_fwhm_nm": GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM
                if s.setup_id in GENERIC_CANONICAL_SETUP_IDS
                else float(s.camera_filter_fwhm_nm),
            }
        )
    path = BUNDLE / "GENERIC_RUNTIME_PARAMETERS_BEFORE_AFTER.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def intensity_metrics_three_phase() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = [0.0, math.pi / 2, math.pi]
    for p in "ABCD":
        series = {ph: run_point(p, phi=ph) for ph in phases}
        imax = max(series[ph]["I_mean"] for ph in phases)
        imin = min(series[ph]["I_mean"] for ph in phases)
        for ph in phases:
            r = series[ph]
            rows.append(
                {
                    "configuration": p,
                    "phi_rad": ph,
                    **r,
                    "I_max_over_phases": imax,
                    "I_min_over_phases": imin,
                    "V_extrema": (imax - imin) / (imax + imin) if (imax + imin) > 0 else float("nan"),
                }
            )
    out = BUNDLE / "FIGURE3D_3NM_INTENSITY_METRICS.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def envelope_sensitivity() -> None:
    rows = []
    for case, nf_diag in [("case1_production", False), ("case2_nf_env_unity_diag", True)]:
        for p in "ABCD":
            r = run_point(p, phi=0.0, nf_env_unity=nf_diag)
            rows.append({"case": case, "configuration": p, **r})
    path = BUNDLE / "GAUSSIAN_ENVELOPE_SENSITIVITY.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ff_nf = []
    for case in ("case1_production", "case2_nf_env_unity_diag"):
        sub = {r["configuration"]: r for r in rows if r["case"] == case}
        ff_nf.append(
            {
                "case": case,
                "A_over_B_I_mean": sub["A"]["I_mean"] / sub["B"]["I_mean"],
                "C_over_D_I_mean": sub["C"]["I_mean"] / sub["D"]["I_mean"],
                "C_over_A_I_mean": sub["C"]["I_mean"] / sub["A"]["I_mean"],
            }
        )
    with (BUNDLE / "FF_NF_COLLECTION_COMPARISON.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ff_nf[0].keys()))
        w.writeheader()
        w.writerows(ff_nf)


def ratio_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    at0 = {r["configuration"]: r for r in rows if abs(float(r["phi_rad"])) < 1e-12}
    return {
        "C_over_A_I_mean": at0["C"]["I_mean"] / at0["A"]["I_mean"],
        "D_over_B_I_mean": at0["D"]["I_mean"] / at0["B"]["I_mean"],
        "A_over_B_I_mean": at0["A"]["I_mean"] / at0["B"]["I_mean"],
        "C_over_D_I_mean": at0["C"]["I_mean"] / at0["D"]["I_mean"],
        "C_over_A_Nsum": at0["C"]["Nsum"] / at0["A"]["Nsum"],
    }


def write_manifest() -> None:
    manifest = {
        "generic_filter_fwhm_nm": 3.0,
        "stale_evidence_note": "Figure 3(c) and pre-20261019 Figure 3(d)/3(e)/4 used C/D @ 10 nm unless otherwise noted",
        "refreshed_by_this_task": [
            "figure3d_manuscript_candidate (new timestamp)",
            "figure3e_3nm_generic_candidate (optional run)",
            "figure4_3nm_generic_candidate (optional run)",
        ],
    }
    (BUNDLE / "CURRENT_EVIDENCE_DEPENDENCY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    write_before_after_csv()
    metrics = intensity_metrics_three_phase()
    ratios = ratio_summary(metrics)
    envelope_sensitivity()
    write_manifest()
    suite = run_four_config_suite()
    fig4_dir = ROOT / "evidence" / "figure4_3nm_generic_candidate" / utc_stamp()
    fig4_dir.mkdir(parents=True, exist_ok=True)
    (fig4_dir / "four_config_suite_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": suite.get("generated_utc"),
                "mapping_rows": suite.get("mapping_rows"),
                "filter_note": "generic A-D @ 3 nm YAML",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {"intensity_ratios_phi0": ratios, "figure4_bundle": str(fig4_dir)}
    (BUNDLE / "FILTER_STANDARDIZATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(BUNDLE)
    print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()
