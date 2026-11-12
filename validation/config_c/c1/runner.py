"""Task 18 runner — C1 mirror/phase calibration and evaluation.

Usage:
    python -m validation.config_c.c1.runner
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
from validation.config_c.c1.calibrate import run_no_object_calibration
from validation.config_c.c1.evaluate import (
    compare_exp_sim_no_object,
    process_with_object,
    run_no_object_simulation,
)
from validation.config_c.c1.identify import (
    PHASE_LABELS,
    PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN,
    TARGET_STAGE_UM,
    identify_c1_sources,
    sha256_file,
    write_source_index,
)

EXPERIMENTAL_ROOT = Path(
    r"D:\Quantum\Quantum\QIUP_Synthetic_Dataset\Ver1.3.4(Digital_Twin_Verification)"
    r"\Report\Received_Verification\for generate results\exp data"
)


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


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")


def freeze_calibration(
    calib: dict[str, Any],
    *,
    freeze_dir: Path,
    calib_dir: Path,
) -> dict[str, Any]:
    freeze_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    frozen_body = {
        "calibration_id": "c1_mirror_phase_v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "object_token_used": "witout object",
        "with_object_data_used": False,
        "primary_shape": "419x455",
        "acquisition_labels": list(PHASE_LABELS),
        "nominal_phase_rad": {
            lab: float(__import__("validation.config_c.c1.identify", fromlist=["NOMINAL_PHASE_RAD"]).NOMINAL_PHASE_RAD[lab])
            for lab in PHASE_LABELS
        },
        "provisional_stage_um": dict(PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN),
        "stage_um_file_metadata": "UNRESOLVED",
        "targets_stopped_um": calib.get("targets_stopped"),
        "dark": calib.get("dark"),
        "per_label_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "load"}
            for k, v in (calib.get("per_label") or {}).items()
        },
        "four_step_calibration": calib.get("four_step_calibration"),
        "stage_to_phase_fit": calib.get("stage_to_phase_fit"),
        "ideal_geometric_law_forced": False,
        "notes": [
            "Frozen from no-object (witout object) data only.",
            "Must not be modified using with-object evaluation data.",
            "Stage µm values are provisional from freeze-plan experimenter report.",
        ],
    }
    # Flatten per_step onto top for evaluate convenience
    ft = calib.get("four_step_calibration") or {}
    frozen_body["per_step"] = ft.get("per_step")
    frozen_body["A_median"] = ft.get("A_median")
    frozen_body["B_median"] = ft.get("B_median")
    frozen_body["V_median"] = ft.get("V_median")
    frozen_body["phi0_circular_mean_rad"] = ft.get("phi0_circular_mean_rad")

    raw = _canonical_json_bytes(frozen_body)
    digest = hashlib.sha256(raw).hexdigest()
    frozen_body["content_sha256"] = digest

    path = freeze_dir / "c1_frozen_phase_calibration.json"
    path.write_text(
        json.dumps(frozen_body, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    # Re-hash file bytes as stored
    file_hash = sha256_file(path)
    manifest = {
        "calibration_id": frozen_body["calibration_id"],
        "path": str(path),
        "content_sha256_declared": digest,
        "file_sha256": file_hash,
        "frozen_utc": frozen_body["frozen_utc"],
        "immutable_policy": (
            "Do not modify using with-object data. New calibration requires "
            "a new calibration_id."
        ),
    }
    _write_json(freeze_dir / "freeze_manifest.json", manifest)
    _write_json(calib_dir / "no_object_calibration_summary.json", frozen_body)
    return frozen_body


def write_tables(layout, frozen: dict[str, Any], obj_eval: dict[str, Any], cmp: dict[str, Any]) -> None:
    # phase_calibration.csv
    path = layout.tables / "phase_calibration.csv"
    fields = [
        "acquisition_label",
        "nominal_phase_rad",
        "provisional_stage_um",
        "median_intensity",
        "effective_phi_j_rad",
        "effective_phi_j_source",
        "intensity_residual",
        "in_task18_target_list",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in frozen.get("per_step") or []:
            z = row.get("provisional_stage_um")
            w.writerow(
                {
                    "acquisition_label": row.get("acquisition_label"),
                    "nominal_phase_rad": row.get("nominal_phase_rad"),
                    "provisional_stage_um": z,
                    "median_intensity": row.get("median_intensity"),
                    "effective_phi_j_rad": row.get("effective_phi_j_rad"),
                    "effective_phi_j_source": row.get("effective_phi_j_source"),
                    "intensity_residual": row.get("intensity_residual"),
                    "in_task18_target_list": z in set(TARGET_STAGE_UM) if z is not None else False,
                }
            )
        # stopped targets
        for z in TARGET_STAGE_UM:
            if z not in set(PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.values()):
                w.writerow(
                    {
                        "acquisition_label": "",
                        "nominal_phase_rad": "",
                        "provisional_stage_um": z,
                        "median_intensity": "",
                        "effective_phi_j_rad": "",
                        "effective_phi_j_source": "NO_DATA_IDENTIFIED",
                        "intensity_residual": "",
                        "in_task18_target_list": True,
                    }
                )

    # visibility_metrics.csv
    vpath = layout.tables / "visibility_metrics.csv"
    with vpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "schedule",
                "A_median",
                "B_median",
                "V_median",
                "residual_rms_median",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "dataset": "no_object_experiment",
                "schedule": "four_step_nominal_for_recon",
                "A_median": frozen.get("A_median"),
                "B_median": frozen.get("B_median"),
                "V_median": frozen.get("V_median"),
                "residual_rms_median": "",
            }
        )
        if obj_eval.get("status") == "OK":
            mf = obj_eval["metrics_frozen_schedule"]
            w.writerow(
                {
                    "dataset": "with_object_experiment",
                    "schedule": "frozen_effective",
                    "A_median": mf["A_median"],
                    "B_median": mf["B_median"],
                    "V_median": mf["V_median"],
                    "residual_rms_median": mf["residual_rms_median"],
                }
            )

    # validation_metrics.csv
    val = layout.tables / "validation_metrics.csv"
    with val.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["metric", "value", "note"],
        )
        w.writeheader()
        rows = [
            (
                "stage_um_file_metadata",
                "UNRESOLVED",
                "No µm readings in TIFF paths",
            ),
            (
                "targets_stopped_um",
                ",".join(str(x) for x in (frozen.get("targets_stopped") or [])),
                "No folder uniquely identifies these stages",
            ),
            (
                "frozen_V_median_noobj",
                frozen.get("V_median"),
                "No-object four-step reconstruction",
            ),
            (
                "with_object_V_median_frozen_schedule",
                (obj_eval.get("metrics_frozen_schedule") or {}).get("V_median"),
                "",
            ),
            (
                "rabbit_object_map",
                "NOT_FOUND",
                "With-object spatial simulation STOPPED",
            ),
            (
                "exp_sim_visibility_abs_diff_median",
                (cmp or {}).get("visibility_abs_diff_median"),
                "Scalar ROI comparison; no resize",
            ),
            (
                "exp_sim_phase_circular_diff_rad",
                (cmp or {}).get("phase_circular_diff_rad"),
                "",
            ),
            (
                "spatial_residual_status",
                ((cmp or {}).get("spatial_residual") or {}).get("status"),
                "",
            ),
        ]
        for m, v, n in rows:
            w.writerow({"metric": m, "value": v, "note": n})


def make_figures(
    layout,
    *,
    frozen: dict[str, Any],
    processed_noobj: Path,
    processed_obj: Path,
    residuals_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_cal = layout.figures / "calibration"
    fig_cmp = layout.figures / "experimental_vs_simulation"
    fig_res = layout.figures / "residuals"
    fig_prof = layout.figures / "profiles"
    for d in (fig_cal, fig_cmp, fig_res, fig_prof):
        d.mkdir(parents=True, exist_ok=True)

    # Phase calibration plot
    fit = frozen.get("stage_to_phase_fit") or {}
    steps = frozen.get("per_step") or []
    if fit.get("status") == "OK" and steps:
        xs = [r["provisional_stage_um"] for r in steps if r.get("provisional_stage_um") is not None]
        ys = [r["effective_phi_j_rad"] for r in steps if r.get("provisional_stage_um") is not None]
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(xs, ys, "o", label="fitted φ_j (data)")
        x_line = np.linspace(min(xs), max(xs), 100)
        y_fit = fit["p0_rad"] + fit["p1_rad_per_um"] * x_line
        plt.plot(x_line, y_fit, "-", label="empirical fit")
        y_ideal = fit["ideal_geometric_slope_rad_per_um"] * x_line
        plt.plot(
            x_line,
            y_ideal,
            "--",
            label="ideal geometric 4πΔz/λ_i (reference only)",
        )
        plt.xlabel("provisional stage Δz [µm]")
        plt.ylabel("effective φ_j [rad]")
        plt.title("C1 phase calibration (ideal law not forced)")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_cal / "phase_vs_stage.png", dpi=140)
        plt.close()

    # Intensity modulation vs label
    if steps:
        plt.figure(figsize=(6.0, 3.8))
        labs = [r["acquisition_label"] for r in steps]
        ints = [r["median_intensity"] for r in steps]
        plt.plot(labs, ints, "o-")
        plt.ylabel("ROI median intensity")
        plt.xlabel("acquisition label")
        plt.title("No-object intensity vs phase label")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_cal / "intensity_vs_label.png", dpi=140)
        plt.close()

    # Reconstruction maps
    for name, title in (
        ("noobj_recon__dc_A.npy", "No-object A"),
        ("noobj_recon__visibility_V.npy", "No-object V"),
        ("noobj_recon__phase_wrapped_rad.npy", "No-object φ"),
    ):
        p = processed_noobj / name
        if p.exists():
            arr = np.load(p)
            plt.figure(figsize=(5.0, 4.2))
            plt.imshow(arr, origin="lower", cmap="viridis")
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_cal / f"{name.replace('.npy', '.png')}", dpi=120)
            plt.close()

    for name, title in (
        ("obj_recon_frozen__visibility_V.npy", "With-object V (frozen schedule)"),
        ("obj_recon_frozen__phase_wrapped_rad.npy", "With-object φ (frozen schedule)"),
        ("obj_recon_frozen__dc_A.npy", "With-object A (frozen schedule)"),
        ("obj_recon_frozen__residual_rms.npy", "With-object residual RMS"),
    ):
        p = processed_obj / name
        if p.exists():
            arr = np.load(p)
            plt.figure(figsize=(5.0, 4.2))
            plt.imshow(arr, origin="lower", cmap="magma" if "residual" in name else "viridis")
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            stem = name.replace(".npy", ".png")
            plt.savefig(fig_cmp / stem, dpi=120)
            plt.close()
            if "residual" in name:
                plt.figure(figsize=(5.0, 4.2))
                plt.imshow(arr, origin="lower", cmap="magma")
                plt.colorbar()
                plt.title(title)
                plt.tight_layout()
                plt.savefig(fig_res / stem, dpi=120)
                plt.close()

    # Profiles
    for fname, ylabel in (
        ("exp_profile_row_A.npy", "A"),
        ("exp_profile_row_V.npy", "V"),
        ("exp_profile_row_phase.npy", "φ [rad]"),
    ):
        p = residuals_dir / fname
        if p.exists():
            arr = np.load(p)
            plt.figure(figsize=(6.0, 3.2))
            plt.plot(arr)
            plt.ylabel(ylabel)
            plt.xlabel("column index (native experimental grid)")
            plt.title(fname.replace(".npy", ""))
            plt.tight_layout()
            plt.savefig(fig_prof / f"{fname.replace('.npy', '.png')}", dpi=120)
            plt.close()


def write_report(
    layout,
    *,
    identification,
    frozen: dict[str, Any],
    obj_eval: dict[str, Any],
    sim: dict[str, Any],
    cmp: dict[str, Any] | None,
) -> Path:
    fit = frozen.get("stage_to_phase_fit") or {}
    lines = [
        "# C1 Mirror/Phase Calibration and Validation (Task 18)",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Policy",
        "",
        "- Experimental source data: **READ-ONLY**",
        "- No filename/acquisition-meaning guessing beyond explicit directory tokens",
        "- Ideal geometric mirror law **not** forced onto stage positions",
        "- Frozen calibration uses **no-object** data only",
        "- Experimental images **not** resized to match simulation",
        "",
        "## Stage 1 — Data identification",
        "",
        f"- Experimental root: `{identification.experimental_root}`",
        f"- Mirror-drift files (Task 16 inventory): **{identification.n_mirror_files}**",
        "- No-object token: `witout object` (on-disk spelling)",
        "- With-object token: `with object`",
        "- Phase-step labels: `0`, `pi2`, `pi`, `3pi2`, `2pi`",
        "- Dark / ref labels present under both object conditions",
        "- Primary analysis shape: **419×455** (other shapes indexed, not resized)",
        "",
        "### Target stage positions (µm)",
        "",
        "| Stage [µm] | Status | Acquisition label |",
        "|------------|--------|-------------------|",
    ]
    for z, info in identification.stage_um_status["per_target"].items():
        lines.append(
            f"| {z} | {info['status']} | {info.get('acquisition_label')} |"
        )
    lines.extend(
        [
            "",
            "**File-level stage µm metadata:** UNRESOLVED "
            "(folders are phase-labeled only).",
            "",
            "Provisional label→µm mapping uses only the freeze-plan experimenter "
            "report that 25 µm increments ≈ {0, π/2, π, 3π/2, 2π}. "
            "Targets **150 µm** and **200 µm**: analysis **STOPPED** (no unique folder).",
            "",
            "## Stage 2 — No-object calibration",
            "",
            f"- Dark primary-shape frames used: "
            f"**{(frozen.get('dark') or {}).get('available')}**",
            f"- A_median: `{frozen.get('A_median')}`",
            f"- B_median: `{frozen.get('B_median')}`",
            f"- V_median: `{frozen.get('V_median')}`",
            f"- φ0 circular mean [rad]: `{frozen.get('phi0_circular_mean_rad')}`",
            f"- Intensity model RMSE: "
            f"`{(frozen.get('four_step_calibration') or {}).get('intensity_model_rmse')}`",
            f"- φ_j source: folder acquisition labels (not geometric law)",
            "",
            "### Stage → effective phase fit (empirical)",
            "",
            f"- Status: `{fit.get('status')}`",
            f"- Model: `{fit.get('model')}`",
            f"- p0 [rad]: `{fit.get('p0_rad')}`",
            f"- p1 [rad/µm]: `{fit.get('p1_rad_per_um')}`",
            f"- RMSE [rad]: `{fit.get('rmse_rad')}`",
            f"- Note: `{fit.get('note')}`",
            f"- Ideal geometric slope (reference only) [rad/µm]: "
            f"`{fit.get('ideal_geometric_slope_rad_per_um')}`",
            f"- Ideal law forced: **False**",
            "",
            "## Stage 3 — Freeze",
            "",
            f"- Calibration id: `{frozen.get('calibration_id')}`",
            f"- Content SHA-256: `{frozen.get('content_sha256')}`",
            "- Path: `data/frozen_calibration/c1_frozen_phase_calibration.json`",
            "- With-object data **not** used to modify the freeze.",
            "",
            "## Stage 4 — With-object evaluation",
            "",
            f"- Experimental with-object status: `{obj_eval.get('status')}`",
        ]
    )
    if obj_eval.get("status") == "OK":
        mf = obj_eval["metrics_frozen_schedule"]
        lines.extend(
            [
                f"- V_median (frozen schedule): `{mf.get('V_median')}`",
                f"- residual_rms_median: `{mf.get('residual_rms_median')}`",
                f"- Reconstruction method: `{mf.get('method')}`",
            ]
        )
    rabbit = (obj_eval.get("rabbit_object_map") or {})
    lines.extend(
        [
            "",
            f"- Rabbit object map: **{rabbit.get('status')}**",
            f"  - {rabbit.get('note')}",
            "",
            "### Simulation (no-object, frozen schedule)",
            "",
            f"- Status: `{sim.get('status')}`",
            f"- Schedule [rad]: `{sim.get('schedule_rad')}`",
            f"- Pump note: {sim.get('pump_power_note')}",
        ]
    )
    if cmp:
        lines.extend(
            [
                "",
                "### Experiment vs simulation (scalar; no resize)",
                "",
                f"- |ΔV|_median: `{cmp.get('visibility_abs_diff_median')}`",
                f"- Δφ circular [rad]: `{cmp.get('phase_circular_diff_rad')}`",
                f"- Spatial residual: `{(cmp.get('spatial_residual') or {}).get('status')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `data/source_index/`",
            "- `data/calibration/`",
            "- `data/frozen_calibration/`",
            "- `data/experimental_processed/`",
            "- `data/simulation/`",
            "- `data/residuals/`",
            "- `tables/phase_calibration.csv`",
            "- `tables/visibility_metrics.csv`",
            "- `tables/validation_metrics.csv`",
            "- `provenance/calibration_manifest.json`",
            "- `provenance/source_hashes.csv`",
            "",
            "## STOP",
            "",
            "Task 18 complete. Do not start Task 19+ in this run.",
            "",
        ]
    )
    path = layout.report / "C1_calibration_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_source_hashes(layout, identification) -> None:
    path = layout.provenance / "source_hashes.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["relative_path", "sha256", "file_size_bytes", "object_token", "acquisition_label"]
        )
        w.writeheader()
        for g in identification.groups.values():
            for f in g["files_primary"]:
                w.writerow(
                    {
                        "relative_path": f["relative_path"],
                        "sha256": f["sha256"],
                        "file_size_bytes": f["file_size_bytes"],
                        "object_token": g["object_token"],
                        "acquisition_label": g["acquisition_label"],
                    }
                )


def main() -> int:
    layout = ensure_task_layout("task18", "c1_validation")
    # Extra subdirs required by the task brief
    for sub in (
        layout.data / "source_index",
        layout.data / "calibration",
        layout.data / "frozen_calibration",
        layout.data / "experimental_processed" / "no_object",
        layout.data / "experimental_processed" / "with_object",
        layout.data / "simulation",
        layout.data / "residuals",
    ):
        sub.mkdir(parents=True, exist_ok=True)

    log_path = layout.logs / "task18_runner.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{utc_timestamp()}] {msg}"
        log_lines.append(line)
        print(line)

    try:
        inventory = (
            Path("evidence")
            / "task16_config_c_lab"
            / "data"
            / "experimental_file_inventory.csv"
        )
        if not inventory.is_file():
            raise FileNotFoundError(f"Task 16 inventory missing: {inventory}")
        if not EXPERIMENTAL_ROOT.is_dir():
            raise FileNotFoundError(
                f"Experimental root missing: {EXPERIMENTAL_ROOT}"
            )

        log("Stage 1: identify C1 sources from Task 16 inventory")
        identification = identify_c1_sources(
            experimental_root=EXPERIMENTAL_ROOT,
            inventory_csv=inventory,
        )
        write_source_index(identification, layout.data / "source_index")
        write_source_hashes(layout, identification)
        _write_json(
            layout.configs / "c1_identification_notes.json",
            {
                "notes": identification.notes,
                "stage_um_status": identification.stage_um_status,
            },
        )

        log("Stage 2: no-object calibration")
        calib = run_no_object_calibration(
            identification,
            processed_dir=layout.data / "experimental_processed" / "no_object",
        )
        maps = calib.pop("_maps", None)

        log("Stage 3: freeze calibration")
        frozen = freeze_calibration(
            calib,
            freeze_dir=layout.data / "frozen_calibration",
            calib_dir=layout.data / "calibration",
        )
        _write_json(
            layout.provenance / "calibration_manifest.json",
            {
                "calibration_id": frozen["calibration_id"],
                "content_sha256": frozen["content_sha256"],
                "file_sha256": sha256_file(
                    layout.data
                    / "frozen_calibration"
                    / "c1_frozen_phase_calibration.json"
                ),
                "with_object_used": False,
                "ideal_geometric_law_forced": False,
            },
        )

        log("Stage 4: with-object evaluation + simulation")
        obj_eval = process_with_object(
            identification,
            processed_dir=layout.data / "experimental_processed" / "with_object",
            frozen=frozen,
        )
        # Strip non-JSON payloads before saving summary
        obj_save = {
            k: v
            for k, v in obj_eval.items()
            if not k.startswith("_")
        }
        _write_json(
            layout.data / "experimental_processed" / "with_object" / "summary.json",
            obj_save,
        )

        sim = run_no_object_simulation(
            frozen=frozen,
            sim_dir=layout.data / "simulation",
        )
        sim_save = {k: v for k, v in sim.items() if not k.startswith("_")}
        _write_json(layout.data / "simulation" / "summary.json", sim_save)

        cmp = None
        if maps is not None and sim.get("status") == "OK":
            cmp = compare_exp_sim_no_object(
                exp_maps=maps,
                sim_rec=sim["_rec"],
                residuals_dir=layout.data / "residuals",
            )
            _write_json(layout.data / "residuals" / "comparison_summary.json", cmp)

        write_tables(layout, frozen, obj_save, cmp or {})
        make_figures(
            layout,
            frozen=frozen,
            processed_noobj=layout.data / "experimental_processed" / "no_object",
            processed_obj=layout.data / "experimental_processed" / "with_object",
            residuals_dir=layout.data / "residuals",
        )
        report = write_report(
            layout,
            identification=identification,
            frozen=frozen,
            obj_eval=obj_save,
            sim=sim_save,
            cmp=cmp,
        )
        log(f"report: {report}")
        log("DONE Task 18")
        return 0
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        return 1
    finally:
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
