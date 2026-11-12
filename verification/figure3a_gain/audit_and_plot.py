#!/usr/bin/env python3
"""Audit Figure 3(a) gain curves and emit an independent verification figure.

Does **not** modify ``taskX8_final_Figure3.draw_panel_a`` or production physics.
Outputs live under ``evidence/figure3a_gain_verification/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setup_params import resolve_setup_selection
from verification.figure3a_gain.independent_reference import (
    low_gain_approximation,
    low_gain_relative_error_vs_exact,
    mean_photons_per_mode_independent,
)
from verification.figure3a_gain.production_probe import (
    DEFAULT_PROFILE,
    R_GRID_DEFAULT,
    legacy_u02_state_for_r,
    probe_production_nbar_nl1,
    production_state_for_r,
    sweep_production,
    _uniform_ref,
    DEFAULT_GRID,
    DEFAULT_SEED,
)
from simulate import simulate_forward

EVIDENCE_ROOT = ROOT / "evidence" / "figure3a_gain_verification"
REL_ERR_FLOOR = 1.0e-15  # below this, relative error vs reference is omitted
DPI = 300


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_table(r_grid: np.ndarray, prod_rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(r_grid):
        n_ref = mean_photons_per_mode_independent(float(r))
        n_low = low_gain_approximation(float(r))
        n_prod = float(prod_rows[i]["nbar_production_NL1"])
        abs_err = abs(n_prod - n_ref)
        if n_ref > REL_ERR_FLOOR:
            rel_err = abs_err / n_ref
        else:
            rel_err = float("nan")
        out.append(
            {
                "r": float(r),
                "n_reference_independent": n_ref,
                "n_low_gain_r2": n_low,
                "n_production_NL1": n_prod,
                "abs_error_prod_vs_reference": abs_err,
                "rel_error_prod_vs_reference": rel_err,
                "epsilon_low_gain_vs_exact": low_gain_relative_error_vs_exact(float(r)),
            }
        )
    return out


def summarize_errors(table: list[dict]) -> dict:
    valid = [t for t in table if np.isfinite(t["rel_error_prod_vs_reference"])]
    rels = [t["rel_error_prod_vs_reference"] for t in valid if t["r"] > 0]
    abses = [t["abs_error_prod_vs_reference"] for t in table]
    domain = f"r in [{table[0]['r']}, {table[-1]['r']}], excluding r=0 for relative stats"
    return {
        "valid_domain": domain,
        "n_points": len(table),
        "max_abs_error_prod_vs_reference": max(abses),
        "mean_abs_error_prod_vs_reference": float(np.mean(abses)),
        "max_rel_error_prod_vs_reference": max(rels) if rels else 0.0,
        "mean_rel_error_prod_vs_reference": float(np.mean(rels)) if rels else 0.0,
        "rel_err_floor": REL_ERR_FLOOR,
    }


def audit_legacy_u02_circularity(r_probe: list[float]) -> list[dict]:
    """Document that legacy U02 passes n̄ directly (trivial match)."""
    setup = resolve_setup_selection(DEFAULT_PROFILE)
    rows = []
    ref = _uniform_ref(setup, DEFAULT_GRID)
    for j, r in enumerate(r_probe):
        n_req = mean_photons_per_mode_independent(r)
        state = legacy_u02_state_for_r(setup, r, n_req)
        result = simulate_forward(ref, state, seed=DEFAULT_SEED + j, detector_noise=False)
        n_meta = float(result.meta.get("nbar_NL1", n_req))
        rows.append(
            {
                "r": r,
                "nbar_injected": n_req,
                "nbar_meta_NL1": n_meta,
                "rel_error": abs(n_meta - n_req) / n_req if n_req > 0 else 0.0,
                "note": "Legacy U02 path (mean_photons_per_mode preset); not independent of reference",
            }
        )
    return rows


def write_audit_md(
    path: Path,
    summary: dict,
    setup_snapshot: dict,
    legacy_rows: list[dict],
) -> None:
    legacy_note = (
        "Legacy U02 (`verification/physics/u1_u9_runner.run_u02`) sets "
        "`mean_photons_per_mode = sinh²(r)` from `independent_equations`, then "
        "reads `nbar_NL1`. That reproduces the same number and is **not** an "
        "independent check of the production `spdc_gain_r → n̄` mapping."
    )
    text = f"""# Figure 3(a) — SPDC gain audit

## Physical definition (panel scope)

- **Observable:** mean photon number per spatiotemporal mode, $\\bar{{n}}$ (dimensionless).
- **Swept parameter:** parametric gain (squeezing) parameter $r$ (dimensionless).
- **Exact relation (single-mode Bogoliubov from vacuum):** $\\bar{{n}} = \\sinh^2(r)$.
- **Low-gain approximation:** $\\bar{{n}} \\approx r^2$ for $|r| \\ll 1$.
- **Not verified here:** pump power $P_p \\to r$ calibration chain.

## Production implementation

| Item | Location |
|------|----------|
| Figure panel | `tools/taskX8_final_Figure3.py` → `draw_panel_a` |
| Legacy data / U02 | `verification/physics/u1_u9_runner.py` → `run_u02` |
| Evidence CSV | `evidence/taskX3_U01_U03_rerun/data/U02/` |
| $r \\to \\bar{{n}}$ in simulator | `optics.resolve_spdc_source_nbar` when `spdc_gain_r` is set → `setup_params.sinh_squared_stable(r)` |
| Documentation | physical SPDC gain notes (shipped with modular `optics` / `setup_params` provenance) |

## Independence finding

1. **Blue “Exact SPDC relation” curve** in the manuscript panel is `verification.physics.independent_equations.mean_photons_sinh_squared` (same $\\sinh^2$ law, separate helper from production `sinh_squared_stable`, but **same formula**).
2. **Green “Numerical implementation” markers** in legacy U02 use **direct** `mean_photons_per_mode` injection at $\\sinh^2(r)$, not `spdc_gain_r`. {legacy_note}
3. **This diagnostic** compares production `spdc_gain_r` sweeps against an **expm1-based** reference in `verification/figure3a_gain/independent_reference.py` (no import of production or `independent_equations`).

## Fixed parameters (production probe)

```json
{json.dumps(setup_snapshot, indent=2)}
```

- Object: uniform unit transmission, grid $N={DEFAULT_GRID}$.
- Detector noise: off; `num_frames=1`; latent noise scales $s_* = 0$.
- Profile **{DEFAULT_PROFILE}** (A–D give identical $\\bar{{n}}$ on this check per X3 report).

## Swept parameter

- $r \\in [0, 0.5]$, 51 uniform samples (matches frozen `U02_independent_gain_curves.csv` grid).

## Quantitative summary (production vs independent reference)

```json
{json.dumps(summary, indent=2)}
```

## Legacy U02 circular probe (injected n̄)

| r | rel error meta vs injected |
|---|---------------------------|
"""
    for row in legacy_rows:
        text += f"| {row['r']} | {row['rel_error']:.3e} |\n"
    text += """
## Low-gain validity (approximation error, not implementation)

The column `low_gain_rel_error_vs_exact` in `verification_table.csv` reports
$|\\bar n_{{\\mathrm{{exact}}}} - r^2|/\\bar n_{{\\mathrm{{exact}}}}$ — intrinsic
error of the **approximation**, not simulator error.

## Reproduce

```powershell
python verification/figure3a_gain/audit_and_plot.py
pytest tests/test_figure3a_gain_verification.py -q
```
"""
    path.write_text(text, encoding="utf-8")


def plot_figure(table: list[dict], out: Path) -> None:
    r = np.array([t["r"] for t in table])
    n_ref = np.array([t["n_reference_independent"] for t in table])
    n_low = np.array([t["n_low_gain_r2"] for t in table])
    n_prod = np.array([t["n_production_NL1"] for t in table])
    rel = np.array([t["rel_error_prod_vs_reference"] for t in table])

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.plot(
        r,
        n_ref,
        color="#1f4e79",
        lw=2.0,
        label=r"Independent reference, $\bar{n}=\sinh^{2}(r)$ (expm1 form)",
    )
    ax0.plot(
        r,
        n_low,
        color="#c45c26",
        ls="--",
        lw=1.6,
        label=r"Low-gain approximation, $\bar{n}\approx r^{2}$",
    )
    ax0.scatter(
        r,
        n_prod,
        s=22,
        zorder=5,
        color="#2a9d8f",
        edgecolors="k",
        linewidths=0.35,
        label="Production simulator (spdc_gain_r → n̄_NL1)",
    )
    ax0.set_xlabel(r"Parametric gain parameter, $r$")
    ax0.set_ylabel(r"Mean photon number per mode, $\bar{n}$")
    ax0.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax0.grid(True, alpha=0.28)
    ax0.set_title("Figure 3(a) diagnostic — gain law verification", fontsize=10)

    rel_pct = np.where(np.isfinite(rel), rel * 100.0, np.nan)
    ax1.plot(r, rel_pct, color="#264653", lw=1.4)
    ax1.axhline(0.0, color="k", lw=0.6)
    ax1.set_xlabel(r"Parametric gain parameter, $r$")
    ax1.set_ylabel(r"$|\bar{n}_{\mathrm{prod}}-\bar{n}_{\mathrm{ref}}|/\bar{n}_{\mathrm{ref}}$ (%)")
    ax1.set_yscale("symlog", linthresh=1e-16)
    ax1.grid(True, alpha=0.28)
    ax1.text(
        0.02,
        0.92,
        f"Relative error omitted at r=0; floor $\\bar{{n}}_{{\\mathrm{{ref}}}}>{REL_ERR_FLOOR:.0e}$",
        transform=ax1.transAxes,
        fontsize=7,
        va="top",
    )

    stem = out / "FIGURE3a_GAIN_VERIFICATION_DIAGNOSTIC"
    fig.savefig(f"{stem}.png", dpi=DPI)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


def setup_snapshot() -> dict:
    setup = resolve_setup_selection(DEFAULT_PROFILE)
    return {
        "profile": DEFAULT_PROFILE,
        "spdc_source_mode": getattr(setup, "spdc_source_mode", None),
        "lambda_p_nm": setup.lambda_p_nm,
        "lambda_s_nm": setup.lambda_s_nm,
        "lambda_i_nm": setup.lambda_i_nm,
        "grid_n": DEFAULT_GRID,
        "seed": DEFAULT_SEED,
        "detector_noise": False,
        "sweep": "spdc_gain_r",
        "r_min": float(R_GRID_DEFAULT[0]),
        "r_max": float(R_GRID_DEFAULT[-1]),
        "n_r_points": int(len(R_GRID_DEFAULT)),
    }


def run(output: Path | None = None) -> Path:
    stamp = utc_stamp()
    out = output or (EVIDENCE_ROOT / stamp)
    out.mkdir(parents=True, exist_ok=True)

    r_grid = R_GRID_DEFAULT
    prod_rows = sweep_production(r_grid)
    table = build_table(r_grid, prod_rows)
    summary = summarize_errors(table)
    legacy = audit_legacy_u02_circularity([0.01, 0.05, 0.1, 0.2])

    fields = list(table[0].keys())
    save_csv(out / "verification_table.csv", table, fields)
    (out / "error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(out / "legacy_u02_circularity.csv", legacy, list(legacy[0].keys()))
    write_audit_md(out / "AUDIT.md", summary, setup_snapshot(), legacy)
    plot_figure(table, out)

    caption = (
        "Diagnostic verification of the SPDC mean photon number per mode "
        r"$\bar{n}=\sinh^2(r)$ implemented in the digital twin. "
        "The continuous curve is an independent expm1 evaluation of the "
        "Bogoliubov relation; markers are production operating points "
        r"obtained by requesting parametric gain $r$ via \texttt{spdc\_gain\_r}. "
        r"The dashed curve shows the low-gain approximation $\bar{n}\approx r^2$. "
        "The lower panel reports relative deviation of production "
        r"$\bar{n}_{\mathrm{NL1}}$ from the independent reference over "
        r"$r\in[0,0.5]$. This panel does not validate $P_p\to r$."
    )
    (out / "SUGGESTED_CAPTION.md").write_text(caption + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": summary}, indent=2))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
