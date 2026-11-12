#!/usr/bin/env python3
"""Evidence bundle for approved generic uniform collection + runtime verification."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qiup_digital_twin_full as qdtf
from setup_params import (
    FINAL_CAMERA_COLLECTION_MODE_LEGACY,
    FINAL_CAMERA_COLLECTION_UNIFORM,
    GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM,
    resolve_setup_selection,
    resolved_final_camera_collection_policy,
)
from simulate import simulate_forward
from verification.physics.u1_u9_runner import DEFAULT_GRID, DEFAULT_NBAR, DEFAULT_SEED, quiet_state, uniform_object

BUNDLE = ROOT / "evidence" / "generic_uniform_collection_finalization_20261019"
PHASES = [0.0, 0.5 * math.pi, math.pi]


def run_point(profile: str, phi: float) -> dict:
    setup = resolve_setup_selection(profile)
    setup.delta_L_mm = 0.0
    state = quiet_state(setup, nbar=DEFAULT_NBAR)
    ref = uniform_object(DEFAULT_GRID, T=1.0, phi=float(phi))
    res = simulate_forward(ref, state, seed=DEFAULT_SEED, detector_noise=False)
    vo = res.meta.get("visibility_observables") or {}
    opt = res.meta.get("last_optical") or {}
    return {
        "profile": profile,
        "phi_rad": phi,
        "setup_id": setup.setup_id,
        "filter_fwhm_nm": float(setup.camera_filter_fwhm_nm),
        "collection_policy": resolved_final_camera_collection_policy(setup),
        "I_mean": float(np.mean(res.ideal_signal_image)),
        "N1": float(vo.get("N_1") or 0),
        "N2": float(vo.get("N_2") or 0),
        "C12_abs": float(vo.get("C_12_abs") or 0),
        "final_env_equation": opt.get("final_camera_pump_envelope_equation"),
        "final_env_applied": opt.get("final_camera_pump_envelope_applied"),
    }


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    rows = [run_point(p, ph) for p in "ABCD" for ph in PHASES]
    phi0 = {r["profile"]: r for r in rows if r["phi_rad"] == 0.0}
    ratios = {
        "A_over_B": phi0["A"]["I_mean"] / phi0["B"]["I_mean"],
        "C_over_D": phi0["C"]["I_mean"] / phi0["D"]["I_mean"],
        "C_over_A": phi0["C"]["I_mean"] / phi0["A"]["I_mean"],
        "D_over_B": phi0["D"]["I_mean"] / phi0["B"]["I_mean"],
    }
    with (BUNDLE / "GENERIC_RUNTIME_VERIFICATION.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    param_rows = []
    for p in "ABCD":
        s = resolve_setup_selection(p)
        param_rows.append(
            {
                "profile": p,
                "setup_id": s.setup_id,
                "imaging_mode": s.imaging_mode,
                "filter_fwhm_nm": s.camera_filter_fwhm_nm,
                "pixel_um": s.pixel_um,
                "collection_policy": resolved_final_camera_collection_policy(s),
            }
        )
    clab = resolve_setup_selection("C_lab")
    param_rows.append(
        {
            "profile": "C_lab",
            "setup_id": clab.setup_id,
            "imaging_mode": clab.imaging_mode,
            "filter_fwhm_nm": clab.camera_filter_fwhm_nm,
            "pixel_um": clab.pixel_um,
            "collection_policy": resolved_final_camera_collection_policy(clab),
        }
    )
    with (BUNDLE / "FINAL_GENERIC_PARAMETER_TABLE.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(param_rows[0].keys()))
        w.writeheader()
        w.writerows(param_rows)
    manifest = {
        "policy_change": "generic A-D default final_camera_collection_policy=uniform",
        "not_equivalent_to_prior": [
            "evidence/figure3d_manuscript_candidate/20261019T152520Z (mode_legacy NF envelope)",
            "evidence/generic_filter_3nm_standardization_20261019 (pre-uniform collection default)",
            "evidence/figure3d_collection_policy_comparison_20261019 (controlled sensitivity)",
        ],
        "figure3c_stale_note": "Figure 3(c) filter review deferred; not regenerated here",
        "figure3e_figure4_flag": "Regenerate when collection-policy refresh is scheduled",
        "intensity_ratio_checkpoints_phi0": ratios,
    }
    (BUNDLE / "AFFECTED_EVIDENCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (BUNDLE / "MODULAR_MONOLITHIC_PARITY.md").write_text(
        f"""# Modular / monolithic parity

- Generic B collection policy (modular): `{resolved_final_camera_collection_policy(resolve_setup_selection('B'))}`
- Generic B collection policy (monolith): `{qdtf.resolved_final_camera_collection_policy(resolve_setup_selection('B'))}`
- Filter FWHM benchmark: {GENERIC_BENCHMARK_CAMERA_FILTER_FWHM_NM} nm
""",
        encoding="utf-8",
    )
    (BUNDLE / "C_LAB_NONREGRESSION.md").write_text(
        f"""# C_lab non-regression

- filter: {clab.camera_filter_fwhm_nm} nm
- pixel_um: {clab.pixel_um}
- collection policy: {resolved_final_camera_collection_policy(clab)} (unchanged far-field legacy path)
""",
        encoding="utf-8",
    )
    (BUNDLE / "COLLECTION_POLICY_IMPLEMENTATION.md").write_text(
        """# Collection policy implementation

Production resolves `final_camera_collection_envelope()` in `setup_params.py`, called from `optics._optical_intensity`.

| Policy | Behavior |
|--------|----------|
| `uniform` | E=1 on all architectures (generic A–D default) |
| `mode_legacy` | FF E=1; NF Gaussian exp(-2r²/w_px²) |
| `matched_gaussian` | Common Gaussian on A–D (sensitivity) |

Generic YAML: `signal_filter.final_camera_collection_policy: uniform`.

Diagnostic `collection_policy_context` remains opt-in for sensitivity only.

Constants: `FINAL_CAMERA_COLLECTION_*`, `validate_generic_benchmark_final_camera_collection`.
""",
        encoding="utf-8",
    )
    (BUNDLE / "FWHM_METRIC_CORRECTION.md").write_text(
        """# FWHM metric correction

See `evidence/figure3d_collection_policy_comparison_20261019/fwhm_verification_20261019/`.

Shared helper: `diagnostics/fwhm_metrics.py` (crossing-based; EDGE_CLIPPED → NaN).

Historical 368 µm on N=24 was invalid (full above-half span).
""",
        encoding="utf-8",
    )
    (BUNDLE / "REPRODUCE.md").write_text(
        """```bash
python build_single_file.py
python tools/generic_uniform_collection_finalization.py
python tools/figure3d_manuscript_candidate.py
python -m pytest tests/test_generic_filter_3nm_standard.py tests/test_generic_uniform_collection.py -q
```
""",
        encoding="utf-8",
    )
    print("Wrote", BUNDLE)
    print("Ratios phi=0:", ratios)


if __name__ == "__main__":
    main()
