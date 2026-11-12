"""Task 19 runner — C2 longitudinal + transverse validation.

Usage:
    python -m validation.config_c.c2_validation.runner
"""
from __future__ import annotations

import csv
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from output_layout import ensure_task_layout, utc_timestamp
from validation.config_c.c2_validation.identify import identify_c2
from validation.config_c.c2_validation.longitudinal import (
    analyze_longitudinal_mean,
    load_mean_std,
)
from validation.config_c.c2_validation.simulate_c2 import (
    simulate_longitudinal_intensity,
    simulate_transverse_intensity,
)
from validation.config_c.c2_validation.transverse import analyze_transverse_mean

EXPERIMENTAL_ROOT = Path(
    r"D:\Quantum\Quantum\QIUP_Synthetic_Dataset\Ver1.3.4(Digital_Twin_Verification)"
    r"\Report\Received_Verification\for generate results\exp data"
)

TASK17_REPORT = Path(
    "evidence/task17_transverse_lens/report/transverse_lens_verification_and_correction.md"
)

ASSUMED_PIXEL_UM = 16.0  # stock C; experimental pitch UNKNOWN (Task 16)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task17_passed() -> dict[str, Any]:
    if not TASK17_REPORT.is_file():
        return {"passed": False, "reason": "Task 17 report missing"}
    text = TASK17_REPORT.read_text(encoding="utf-8", errors="replace")
    ok = (
        "Correction applied: **True**" in text
        and "Post-correction match: **True**" in text
    )
    return {
        "passed": ok,
        "report": str(TASK17_REPORT),
        "reason": "Task 17 correction verified" if ok else "Task 17 not fully passed",
    }


def run_longitudinal(layout, identification) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    base = layout.data / "longitudinal"
    for sub in ("experimental", "simulation", "theory", "residuals"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    fig = layout.figures / "longitudinal"
    for sub in (
        "images",
        "radial_profiles",
        "ring_detection",
        "rho2_vs_order",
        "quadratic_coefficient",
        "residuals",
    ):
        (fig / sub).mkdir(parents=True, exist_ok=True)

    # Prefer no-object quantitative validation
    groups = identification.longitudinal["groups"]
    a_exp_list, a_sim_list, a_th_list, dz_list = [], [], [], []

    for dlt_mm in identification.longitudinal["targets_mm"]:
        key = f"witout object::dz_{dlt_mm:.3f}mm"
        g = groups.get(key) or {}
        tag = f"dz_{dlt_mm:.3f}mm"
        dz_um = float(dlt_mm) * 1.0e3

        exp_row: dict[str, Any] = {
            "delta_z_mm": dlt_mm,
            "delta_z_um": dz_um,
            "object_token": "witout object",
            "exp_status": g.get("status", "NO_DATA"),
        }

        if g.get("status") == "OK" and g.get("files_primary"):
            loaded = load_mean_std(g["files_primary"])
            if loaded.get("status") == "OK":
                mean = loaded["mean"]
                np.save(base / "experimental" / f"mean__{tag}.npy", mean)
                np.save(base / "experimental" / f"std__{tag}.npy", loaded["std"])
                _write_json(
                    base / "experimental" / f"load__{tag}.json",
                    {k: v for k, v in loaded.items() if k not in {"mean", "std"}},
                )
                ana = analyze_longitudinal_mean(
                    mean,
                    delta_z_um=dz_um,
                    pixel_um_assumed=ASSUMED_PIXEL_UM,
                    f_u_mm=75.0,
                )
                # persist profiles
                np.save(
                    base / "experimental" / f"radial_r_px__{tag}.npy",
                    ana["radial"]["r_px"],
                )
                np.save(
                    base / "experimental" / f"radial_profile__{tag}.npy",
                    ana["radial"]["profile"],
                )
                exp_row.update(
                    {
                        "exp_status": "OK",
                        "n_repeats": loaded["n_repeats"],
                        "shape": loaded["shape"],
                        "center_y_px": ana["center_y_px"],
                        "center_x_px": ana["center_x_px"],
                        "n_rings": len(ana["rings"].get("radii_px") or []),
                        "a_exp_per_um2": ana.get("a_exp_per_um2"),
                        "V_roi_exp": ana["visibility"]["V_roi"],
                        "V_local_exp": ana["visibility"]["V_local_proxy"],
                        "rho2_fit_status": ana["rho2_fit"].get("status"),
                        "primary_shape": g.get("primary_shape"),
                        "resize_applied": False,
                    }
                )
                # figures
                plt.figure(figsize=(4.8, 4.2))
                plt.imshow(mean, origin="lower", cmap="gray")
                plt.plot(ana["center_x_px"], ana["center_y_px"], "r+")
                plt.title(f"Exp mean δz={dlt_mm} mm")
                plt.tight_layout()
                plt.savefig(fig / "images" / f"exp_mean__{tag}.png", dpi=120)
                plt.close()

                plt.figure(figsize=(5.5, 3.5))
                plt.plot(ana["radial"]["r_px"], ana["radial"]["profile"])
                for r in ana["rings"].get("radii_px") or []:
                    plt.axvline(r, color="C1", alpha=0.5, ls="--")
                plt.xlabel("ρ [px]")
                plt.ylabel("radial mean intensity")
                plt.title(f"Radial profile δz={dlt_mm} mm")
                plt.tight_layout()
                plt.savefig(
                    fig / "radial_profiles" / f"radial__{tag}.png", dpi=120
                )
                plt.close()

                fit = ana["rho2_fit"]
                if fit.get("status") == "OK":
                    plt.figure(figsize=(5.0, 3.8))
                    nord = np.asarray(fit["orders"], dtype=float)
                    rho2 = np.asarray(fit["rho2_um2"], dtype=float)
                    plt.plot(nord, rho2, "o", label="exp")
                    plt.plot(
                        nord,
                        fit["slope_um2_per_order"] * nord + fit["intercept_um2"],
                        "-",
                        label="fit",
                    )
                    plt.xlabel("fringe order n")
                    plt.ylabel("ρ² [µm²] (ASSUMED px=16µm)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        fig / "rho2_vs_order" / f"rho2__{tag}.png", dpi=120
                    )
                    plt.close()
            else:
                exp_row["exp_status"] = loaded.get("status")
                exp_row["load_error"] = loaded

        # Simulation + theory (always; independent of experiment availability)
        sim = simulate_longitudinal_intensity(delta_z_um=dz_um, n=257)
        np.save(base / "simulation" / f"intensity__{tag}.npy", sim["intensity"])
        np.save(
            base / "simulation" / f"phase_production__{tag}.npy",
            sim["phase_production"],
        )
        np.save(base / "theory" / f"phase_theory__{tag}.npy", sim["phase_theory"])
        resid = sim["phase_production"] - sim["phase_theory"]
        np.save(base / "residuals" / f"phase_resid__{tag}.npy", resid)
        _write_json(
            base / "simulation" / f"summary__{tag}.json",
            {
                "a_sim_from_phase_per_um2": sim["a_sim_from_phase_per_um2"],
                "a_theory_per_um2": sim["a_theory_per_um2"],
                "phase_max_abs_err_vs_theory": sim["phase_max_abs_err_vs_theory"],
                "f_u_mm": sim["f_u_mm"],
            },
        )

        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(sim["intensity"], origin="lower", cmap="gray")
        plt.title(f"Sim intensity δz={dlt_mm} mm")
        plt.tight_layout()
        plt.savefig(fig / "images" / f"sim_intensity__{tag}.png", dpi=120)
        plt.close()

        exp_row["a_sim_per_um2"] = sim["a_sim_from_phase_per_um2"]
        exp_row["a_theory_per_um2"] = sim["a_theory_per_um2"]
        exp_row["phase_max_abs_err_sim_vs_theory"] = sim[
            "phase_max_abs_err_vs_theory"
        ]
        rows.append(exp_row)

        dz_list.append(dlt_mm)
        a_exp_list.append(exp_row.get("a_exp_per_um2"))
        a_sim_list.append(sim["a_sim_from_phase_per_um2"])
        a_th_list.append(sim["a_theory_per_um2"])

        # with-object index only (held-out; no retuning)
        wkey = f"with object::dz_{dlt_mm:.3f}mm"
        wg = groups.get(wkey) or {}
        if wg.get("status") == "OK" and wg.get("files_primary"):
            wloaded = load_mean_std(wg["files_primary"])
            if wloaded.get("status") == "OK":
                np.save(
                    base / "experimental" / f"mean_withobj__{tag}.npy",
                    wloaded["mean"],
                )
                _write_json(
                    base / "experimental" / f"withobj_load__{tag}.json",
                    {
                        "n_repeats": wloaded["n_repeats"],
                        "shape": wloaded["shape"],
                        "note": "Held-out with-object; parameters not retuned.",
                    },
                )

    # a(δ) comparison figure
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(dz_list, a_th_list, "k-", label="theory a(δ)")
    plt.plot(dz_list, a_sim_list, "s--", label="sim a(δ) from phase")
    a_exp_plot = [a if a is not None else np.nan for a in a_exp_list]
    plt.plot(dz_list, a_exp_plot, "o", label="exp a(δ) from ρ² fit")
    plt.xlabel("δz [mm]")
    plt.ylabel("a [1/µm²]")
    plt.title("Quadratic coefficient a(δ)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig / "quadratic_coefficient" / "a_vs_delta.png", dpi=140)
    plt.close()

    return rows


def run_transverse(layout, identification) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    base = layout.data / "transverse"
    for sub in ("experimental", "simulation", "theory", "residuals"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    fig = layout.figures / "transverse"
    for sub in (
        "images",
        "fft",
        "fringe_profiles",
        "spatial_frequency",
        "orientation",
        "residuals",
    ):
        (fig / sub).mkdir(parents=True, exist_ok=True)

    groups = identification.transverse["groups"]
    dxs, k_exp, k_sim, k_th = [], [], [], []

    for dx in identification.transverse["targets_um"]:
        key = f"witout object::dx_{dx:.0f}um"
        g = groups.get(key) or {}
        tag = f"dx_{int(dx):03d}um"
        row: dict[str, Any] = {
            "delta_x_um": dx,
            "object_token": "witout object",
            "exp_status": g.get("status", "NO_DATA"),
        }

        if g.get("status") == "OK" and g.get("files_primary"):
            loaded = load_mean_std(g["files_primary"])
            if loaded.get("status") == "OK":
                mean = loaded["mean"]
                np.save(base / "experimental" / f"mean__{tag}.npy", mean)
                np.save(base / "experimental" / f"std__{tag}.npy", loaded["std"])
                ana = analyze_transverse_mean(
                    mean, delta_x_um=float(dx), pixel_um_assumed=ASSUMED_PIXEL_UM
                )
                np.save(
                    base / "experimental" / f"fft_mag__{tag}.npy",
                    ana["fft_magnitude"],
                )
                row.update(
                    {
                        "exp_status": "OK",
                        "n_repeats": loaded["n_repeats"],
                        "shape": loaded["shape"],
                        "kappa_x_exp": ana["fft"]["kappa_x_cycles_per_um"],
                        "kappa_y_exp": ana["fft"]["kappa_y_cycles_per_um"],
                        "kappa_exp": ana["kappa_exp"],
                        "period_exp_um": ana["period_exp_um"],
                        "orientation_exp_deg": ana["orientation_exp_deg"],
                        "V_roi_exp": ana["visibility"]["V_roi"],
                        "primary_shape": g.get("primary_shape"),
                        "resize_applied": False,
                    }
                )
                plt.figure(figsize=(4.8, 4.2))
                plt.imshow(mean, origin="lower", cmap="gray")
                plt.title(f"Exp mean Δx={dx} µm")
                plt.tight_layout()
                plt.savefig(fig / "images" / f"exp_mean__{tag}.png", dpi=120)
                plt.close()

                plt.figure(figsize=(4.8, 4.2))
                plt.imshow(
                    np.log10(ana["fft_magnitude"] + 1e-12),
                    origin="lower",
                    cmap="magma",
                )
                plt.title(f"FFT |F| Δx={dx} µm")
                plt.tight_layout()
                plt.savefig(fig / "fft" / f"fft__{tag}.png", dpi=120)
                plt.close()

                plt.figure(figsize=(5.5, 3.2))
                plt.plot(ana["row_profile"])
                plt.title(f"Central-row profile Δx={dx} µm")
                plt.tight_layout()
                plt.savefig(
                    fig / "fringe_profiles" / f"profile__{tag}.png", dpi=120
                )
                plt.close()
            else:
                row["exp_status"] = loaded.get("status")

        sim = simulate_transverse_intensity(delta_x_um=float(dx), n=257)
        np.save(base / "simulation" / f"intensity__{tag}.npy", sim["intensity"])
        np.save(
            base / "simulation" / f"phase_production__{tag}.npy",
            sim["phase_production"],
        )
        np.save(base / "theory" / f"phase_theory__{tag}.npy", sim["phase_theory"])
        np.save(
            base / "residuals" / f"phase_resid__{tag}.npy",
            sim["phase_production"] - sim["phase_theory"],
        )
        sim_ana = sim["metrics"]
        row["kappa_sim"] = sim_ana["kappa_exp"]
        row["kappa_theory"] = sim["kappa_theory"]
        row["period_theory_um"] = sim_ana["period_theory_um"]
        row["phase_max_abs_err_sim_vs_theory"] = sim[
            "phase_max_abs_err_vs_theory"
        ]
        rows.append(row)

        dxs.append(dx)
        k_exp.append(row.get("kappa_exp"))
        k_sim.append(row.get("kappa_sim"))
        k_th.append(row.get("kappa_theory"))

        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(sim["intensity"], origin="lower", cmap="gray")
        plt.title(f"Sim intensity Δx={dx} µm")
        plt.tight_layout()
        plt.savefig(fig / "images" / f"sim_intensity__{tag}.png", dpi=120)
        plt.close()

        # with-object held-out dump
        wkey = f"with object::dx_{dx:.0f}um"
        wg = groups.get(wkey) or {}
        if wg.get("status") == "OK" and wg.get("files_primary"):
            wloaded = load_mean_std(wg["files_primary"])
            if wloaded.get("status") == "OK":
                np.save(
                    base / "experimental" / f"mean_withobj__{tag}.npy",
                    wloaded["mean"],
                )

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(dxs, k_th, "k-", label="theory |κ|")
    plt.plot(dxs, k_sim, "s--", label="sim |κ| (FFT)")
    plt.plot(
        dxs,
        [k if k is not None else np.nan for k in k_exp],
        "o",
        label="exp |κ| (FFT)",
    )
    plt.xlabel("Δx [µm]")
    plt.ylabel("|κ| [cycles/µm]")
    plt.title("Spatial frequency vs transverse displacement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig / "spatial_frequency" / "kappa_vs_dx.png", dpi=140)
    plt.close()

    # orientation vs dx
    orients = [r.get("orientation_exp_deg") for r in rows]
    plt.figure(figsize=(6.0, 3.8))
    plt.plot(dxs, [o if o is not None else np.nan for o in orients], "o-")
    plt.xlabel("Δx [µm]")
    plt.ylabel("orientation [deg]")
    plt.title("Experimental fringe orientation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig / "orientation" / "orientation_vs_dx.png", dpi=140)
    plt.close()

    return rows


def write_tables(layout, long_rows, trans_rows) -> None:
    lp = layout.tables / "longitudinal_metrics.csv"
    fields = [
        "delta_z_mm",
        "delta_z_um",
        "exp_status",
        "n_repeats",
        "n_rings",
        "a_exp_per_um2",
        "a_sim_per_um2",
        "a_theory_per_um2",
        "V_roi_exp",
        "phase_max_abs_err_sim_vs_theory",
        "primary_shape",
    ]
    with lp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in long_rows:
            w.writerow(r)

    tp = layout.tables / "transverse_metrics.csv"
    tfields = [
        "delta_x_um",
        "exp_status",
        "n_repeats",
        "kappa_exp",
        "kappa_x_exp",
        "kappa_y_exp",
        "kappa_sim",
        "kappa_theory",
        "period_exp_um",
        "period_theory_um",
        "orientation_exp_deg",
        "V_roi_exp",
        "phase_max_abs_err_sim_vs_theory",
        "primary_shape",
    ]
    with tp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=tfields, extrasaction="ignore")
        w.writeheader()
        for r in trans_rows:
            w.writerow(r)

    # Combined theory/simulation/experiment
    cp = layout.tables / "theory_simulation_experiment.csv"
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "axis",
                "displacement",
                "metric",
                "experiment",
                "simulation",
                "theory",
            ],
        )
        w.writeheader()
        for r in long_rows:
            w.writerow(
                {
                    "axis": "longitudinal",
                    "displacement": r["delta_z_mm"],
                    "metric": "a_per_um2",
                    "experiment": r.get("a_exp_per_um2"),
                    "simulation": r.get("a_sim_per_um2"),
                    "theory": r.get("a_theory_per_um2"),
                }
            )
        for r in trans_rows:
            w.writerow(
                {
                    "axis": "transverse",
                    "displacement": r["delta_x_um"],
                    "metric": "kappa_cycles_per_um",
                    "experiment": r.get("kappa_exp"),
                    "simulation": r.get("kappa_sim"),
                    "theory": r.get("kappa_theory"),
                }
            )


def write_report(
    layout,
    *,
    identification,
    t17,
    long_rows,
    trans_rows,
) -> Path:
    lines = [
        "# C2 Longitudinal and Transverse Validation (Task 19)",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Policy",
        "",
        "- Experimental data: **READ-ONLY**",
        "- No model-parameter retuning from held-out C2 results",
        "- No experimental resize to force agreement with simulation",
        "- Pixel pitch for physical µm conversion: **ASSUMED** stock `16 µm` "
        "(experimental pitch UNKNOWN per Task 16)",
        "- Production `f_u = 75 mm` (stock/lab C). Freeze plan also mentions "
        "`Li=15 mm` — **not** substituted (would be a retune).",
        "",
        "## Identification (Task 16 inventory + freeze-plan coordinates)",
        "",
        f"- Root: `{identification.experimental_root}`",
        "- Longitudinal: `z0=6.910 mm` → δ∈{0, 2.400, 3.580, 5.812} mm",
        "- Transverse: `x0=4.340 mm` → Δx∈{0,20,30,40,60,180} µm",
        "",
        "### Longitudinal group status (no-object)",
        "",
        "| δz [mm] | status | n_files | primary shape |",
        "|---------|--------|---------|---------------|",
    ]
    for dlt in identification.longitudinal["targets_mm"]:
        g = identification.longitudinal["groups"][f"witout object::dz_{dlt:.3f}mm"]
        lines.append(
            f"| {dlt:.3f} | {g['status']} | {g['n_files']} | {g.get('primary_shape')} |"
        )
    lines.extend(
        [
            "",
            "### Transverse group status (no-object)",
            "",
            "| Δx [µm] | status | n_files | primary shape |",
            "|---------|--------|---------|---------------|",
        ]
    )
    for dx in identification.transverse["targets_um"]:
        g = identification.transverse["groups"][f"witout object::dx_{dx:.0f}um"]
        lines.append(
            f"| {dx:.0f} | {g['status']} | {g['n_files']} | {g.get('primary_shape')} |"
        )

    lines.extend(
        [
            "",
            "## Part A — Longitudinal",
            "",
            "| δz [mm] | a_exp | a_sim | a_theory | V_roi_exp | rings |",
            "|---------|-------|-------|----------|-----------|-------|",
        ]
    )
    for r in long_rows:
        lines.append(
            f"| {r['delta_z_mm']:.3f} | {r.get('a_exp_per_um2')} | "
            f"{r.get('a_sim_per_um2')} | {r.get('a_theory_per_um2')} | "
            f"{r.get('V_roi_exp')} | {r.get('n_rings')} |"
        )
    lines.append(
        "\nSim↔theory phase residual max|Δφ| is near machine zero when "
        "production matches independent P11/Config-C law."
    )

    lines.extend(
        [
            "",
            "## Part B — Transverse",
            "",
            f"- Task 17 gate: **{t17['passed']}** ({t17['reason']})",
            "",
        ]
    )
    if t17["passed"]:
        lines.extend(
            [
                "| Δx [µm] | κ_exp | κ_sim | κ_theory | Λ_theory [µm] | orient_exp [deg] |",
                "|---------|-------|-------|----------|---------------|------------------|",
            ]
        )
        for r in trans_rows:
            lines.append(
                f"| {r['delta_x_um']:.0f} | {r.get('kappa_exp')} | "
                f"{r.get('kappa_sim')} | {r.get('kappa_theory')} | "
                f"{r.get('period_theory_um')} | {r.get('orientation_exp_deg')} |"
            )
    else:
        lines.append("Quantitative transverse validation **STOPPED** (Task 17 gate).")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- With-object frames are archived under "
            "`data/*/experimental/*_withobj_*` without parameter retuning.",
            "- Experimental κ and a use ASSUMED pixel pitch; treat absolute "
            "µm-scale comparisons as provisional until camera pitch is measured.",
            "",
            "## STOP",
            "",
            "Task 19 complete. Do not start Task 20+ in this run.",
            "",
        ]
    )
    path = layout.report / "C2_longitudinal_transverse_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    layout = ensure_task_layout("task19", "c2_validation")
    log_path = layout.logs / "task19_runner.log"
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{utc_timestamp()}] {msg}"
        logs.append(line)
        print(line)

    try:
        inventory = (
            Path("evidence")
            / "task16_config_c_lab"
            / "data"
            / "experimental_file_inventory.csv"
        )
        if not inventory.is_file():
            raise FileNotFoundError(inventory)
        if not EXPERIMENTAL_ROOT.is_dir():
            raise FileNotFoundError(EXPERIMENTAL_ROOT)

        log("Identify C2 sources")
        identification = identify_c2(
            experimental_root=EXPERIMENTAL_ROOT, inventory_csv=inventory
        )
        _write_json(
            layout.configs / "c2_identification.json",
            {
                "notes": identification.notes,
                "longitudinal": identification.longitudinal,
                "transverse": identification.transverse,
            },
        )

        # provenance hashes for used primary files
        hash_path = layout.provenance / "source_hashes.csv"
        with hash_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "relative_path",
                    "sha256",
                    "object_token",
                    "axis",
                    "displacement",
                ],
            )
            w.writeheader()
            for g in identification.longitudinal["groups"].values():
                for f in g.get("files_primary") or []:
                    w.writerow(
                        {
                            "relative_path": f["relative_path"],
                            "sha256": f["sha256"],
                            "object_token": g["object_token"],
                            "axis": "longitudinal",
                            "displacement": g["delta_z_mm"],
                        }
                    )
            for g in identification.transverse["groups"].values():
                for f in g.get("files_primary") or []:
                    w.writerow(
                        {
                            "relative_path": f["relative_path"],
                            "sha256": f["sha256"],
                            "object_token": g["object_token"],
                            "axis": "transverse",
                            "displacement": g["delta_x_um"],
                        }
                    )

        log("Part A: longitudinal")
        long_rows = run_longitudinal(layout, identification)

        t17 = task17_passed()
        log(f"Task 17 gate: {t17}")
        if t17["passed"]:
            log("Part B: transverse")
            trans_rows = run_transverse(layout, identification)
        else:
            trans_rows = []
            log("Part B STOPPED — Task 17 did not pass")

        write_tables(layout, long_rows, trans_rows)
        report = write_report(
            layout,
            identification=identification,
            t17=t17,
            long_rows=long_rows,
            trans_rows=trans_rows,
        )
        _write_json(
            layout.provenance / "validation_manifest.json",
            {
                "task": "task19_c2_validation",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "task17_gate": t17,
                "assumed_pixel_um": ASSUMED_PIXEL_UM,
                "f_u_mm_production": 75.0,
                "retuning": False,
                "report": str(report),
                "n_longitudinal_rows": len(long_rows),
                "n_transverse_rows": len(trans_rows),
            },
        )
        log(f"report: {report}")
        log("DONE Task 19")
        return 0
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        return 1
    finally:
        log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
