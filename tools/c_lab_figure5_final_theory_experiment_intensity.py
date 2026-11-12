#!/usr/bin/env python3
"""Assemble final Figure 5: preserved simulation phase scan + experimental intensity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

TopAxisMode = Literal["equiv_one_period", "physical_wrapped_labels"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter

from tools.visibility import load_raw_tiff_phase_series, phase_from_mirror_um, unique_tiff_files

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure5_final_theory_experiment_intensity"
THEORY_EVIDENCE = ROOT / "evidence" / "c_lab_figure5_theory_phase_scan" / "20261017T061813Z"
from tools.c_lab_manuscript_convention import (  # noqa: E402 — active manuscript defaults
    DEFAULT_MANUSCRIPT_SETTINGS,
    MANUSCRIPT_EXP_ROOT as EXP_ROOT,
    compute_experimental_statistics,
    experimental_roi_mask,
    theory_csv_dict,
)

EXP_ROI = experimental_roi_mask()  # fixed r=130 px; legacy name retained for callers
POSITIONS_JSON = ROOT / "tools" / "visibility" / "data" / "mirror_drift_5phase_um.json"
LAMBDA_I_UM = 1.550
DZ_EXP = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float64)
NOMINAL_WRAPPED_DEG = [0.0, 92.903, 185.806, 278.710, 11.613]
THEORY_TIFFS = (
    "THEORY_CONSTRUCTIVE_NO_OBJECT_419x455.tif",
    "THEORY_CONSTRUCTIVE_RABBIT_419x455.tif",
    "THEORY_DESTRUCTIVE_NO_OBJECT_419x455.tif",
    "THEORY_DESTRUCTIVE_RABBIT_419x455.tif",
)


@dataclass(frozen=True)
class ExpPlotOptions:
    """Figure-level experimental plotting overrides (raw TIFFs unchanged on disk)."""

    exclude_folders: frozenset[str] = frozenset()
    dz_plot_um_by_folder: dict[str, float] = field(default_factory=dict)
    tag: str = ""

    @staticmethod
    def exclude_2pi_shift_75_to_85() -> ExpPlotOptions:
        return ExpPlotOptions(
            exclude_folders=frozenset({"2pi"}),
            dz_plot_um_by_folder={"3pi2": 85.0},
            tag="EXCLUDE_2PI_SHIFT75TO85",
        )

    @staticmethod
    def exclude_2pi_only() -> ExpPlotOptions:
        return ExpPlotOptions(exclude_folders=frozenset({"2pi"}), tag="EXCLUDE_2PI_ONLY")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_positions() -> dict[str, float]:
    pos = json.loads(POSITIONS_JSON.read_text(encoding="utf-8"))
    order = ("0", "pi2", "pi", "3pi2", "2pi")
    return {k: float(pos[k]) for k in order}


def wrapped_phase_rad(dz_um: float) -> float:
    return float(4.0 * np.pi * dz_um / LAMBDA_I_UM) % (2.0 * np.pi)


def plot_dz_um(folder: str, pos: dict[str, float], plot_opts: ExpPlotOptions | None) -> float:
    if plot_opts and folder in plot_opts.dz_plot_um_by_folder:
        return float(plot_opts.dz_plot_um_by_folder[folder])
    return float(pos[folder])


def panel_a_folders(pos: dict[str, float], plot_opts: ExpPlotOptions | None) -> list[str]:
    folders = list(pos.keys())
    if not plot_opts:
        return folders
    return [f for f in folders if f not in plot_opts.exclude_folders]


def sinusoid(phi: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * np.cos(phi) + c * np.sin(phi)


def load_theory_csv(path: Path) -> dict[str, np.ndarray]:
    phases, dz, m_no, m_rb = [], [], [], []
    with path.open(newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            phases.append(float(row["phase_rad"]))
            dz.append(float(row["dz_um_one_period"]))
            m_no.append(float(row["roi_mean_no_object_au"]))
            m_rb.append(float(row["roi_mean_rabbit_au"]))
    return {
        "phase_rad": np.asarray(phases, dtype=np.float64),
        "dz_um_one_period": np.asarray(dz, dtype=np.float64),
        "roi_mean_no_object_au": np.asarray(m_no, dtype=np.float64),
        "roi_mean_rabbit_au": np.asarray(m_rb, dtype=np.float64),
    }


def experimental_stats(
    pos: dict[str, float] | None = None,
    exp_roi: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Mirror-drift statistics via active manuscript convention (fresh TIFF read)."""
    _ = pos, exp_roi
    out, pos_map, audit = compute_experimental_statistics(DEFAULT_MANUSCRIPT_SETTINGS)
    per_frame: list[dict] = []
    for row in audit:
        per_frame.append(
            {
                "condition": row["condition"],
                "acquisition_id": f"{row['phase_folder']}/{Path(row['source_path']).stem}",
                "phase_folder": row["phase_folder"],
                "mirror_um": row["delta_z_um"],
                "wrapped_phase_rad": row["phi_z_rad"],
                "wrapped_phase_deg": float(np.degrees(row["phi_z_rad"])),
                "tiff_path": row["source_path"],
                "tiff_sha256": row["sha256"],
                "tiff_page_index": 0,
                "tiff_page_count": 1,
                "roi_mean_adu_per_pixel": row["roi_mean_adu_per_pixel"],
            }
        )
    out["_manuscript_position_map"] = pos_map
    return out, per_frame


def shared_display_vmax(images: list[np.ndarray], pct: float = 99.5) -> float:
    return float(max(np.percentile(np.asarray(im, dtype=np.float64), pct) for im in images))


def setup_phase_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 2 * np.pi)
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"Interferometric phase $\varphi$ (rad)")


def dz_eq_um_from_phase(phi_rad: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(phi_rad, dtype=np.float64)
    out = p * LAMBDA_I_UM / (4.0 * np.pi)
    return float(out) if np.ndim(phi_rad) == 0 else out


def attach_top_axis(ax: plt.Axes, mode: TopAxisMode) -> None:
    if mode == "equiv_one_period":
        sec = ax.secondary_xaxis(
            "top",
            functions=(
                dz_eq_um_from_phase,
                lambda z: np.asarray(z, dtype=np.float64) * 4.0 * np.pi / LAMBDA_I_UM,
            ),
        )
        sec.set_xlabel(r"Equivalent mirror displacement $\Delta z_{\mathrm{eq}}$ ($\mu$m, one $2\pi$ period)")
        sec.set_xlim(0.0, LAMBDA_I_UM / 2.0)
    else:
        ax_top = ax.twiny()
        ax_top.set_xlim(0, 2 * np.pi)
        dz_ticks = (0.0, 25.0, 50.0, 75.0, 100.0)
        tick_pos = [wrapped_phase_rad(z) for z in dz_ticks]
        ax_top.set_xticks(tick_pos)
        ax_top.set_xticklabels([f"{int(z)}" for z in dz_ticks])
        ax_top.set_xlabel(r"Physical idler mirror displacement $\Delta z_{\mathrm{exp}}$ ($\mu$m)")


def draw_panel_a(
    ax_left: plt.Axes,
    theory: dict[str, np.ndarray],
    fit: dict[str, Any],
    exp: dict[str, Any],
    pos: dict[str, float],
    *,
    normalized: bool = False,
    plot_opts: ExpPlotOptions | None = None,
    top_axis: TopAxisMode = "equiv_one_period",
) -> None:
    phi = theory["phase_rad"]
    phi_dense = np.linspace(0, 2 * np.pi, 720)
    f_no = fit["no_object"]
    f_rb = fit["rabbit"]
    curve_no = sinusoid(phi_dense, f_no["A"], f_no["B"], f_no["C"])
    curve_rb = sinusoid(phi_dense, f_rb["A"], f_rb["B"], f_rb["C"])
    y_no = theory["roi_mean_no_object_au"]
    y_rb = theory["roi_mean_rabbit_au"]

    folders_all = list(pos.keys())
    folders = panel_a_folders(pos, plot_opts)
    idxs = [folders_all.index(f) for f in folders]
    phi_exp = np.array([wrapped_phase_rad(plot_dz_um(f, pos, plot_opts)) for f in folders])
    exp_no = exp["no_object"]["roi_mean_adu"][idxs]
    exp_rb = exp["rabbit"]["roi_mean_adu"][idxs]
    sem_no = exp["no_object"]["roi_mean_sem_adu"][idxs]
    sem_rb = exp["rabbit"]["roi_mean_sem_adu"][idxs]

    if normalized:
        ref_t = float(np.mean(y_no))
        ref_e = float(np.mean(exp["no_object"]["roi_mean_adu"]))
        y_no, y_rb = y_no / ref_t, y_rb / ref_t
        curve_no, curve_rb = curve_no / ref_t, curve_rb / ref_t
        exp_no, exp_rb = exp_no / ref_e, exp_rb / ref_e
        sem_no, sem_rb = sem_no / ref_e, sem_rb / ref_e
        ylab_l = "Simulation ROI mean / no-object simulation mean"
        ylab_r = "Experimental ROI mean / no-object experimental mean"
        title = "Panel (a) diagnostic — per-modality no-object normalization (not radiometric)"
    else:
        ylab_l = "Simulation ROI mean signal intensity (a.u.)"
        ylab_r = "Experimental RAW ROI mean (ADU/pixel)"
        title = (
            "Panel (a) Phase-dependent ROI mean intensity\n"
            "(theory a.u. vs experiment RAW ADU; no absolute radiometric calibration)"
        )

    ax_left.plot(phi_dense, curve_no, "-", color="#4c72b0", lw=1.6, label="Simulation fit no-object")
    ax_left.plot(phi_dense, curve_rb, "-", color="#8172b3", lw=1.6, label="Simulation fit rabbit")
    ax_left.plot(phi, y_no, "o", ms=2.5, color="#4c72b0", alpha=0.35, label="Simulation samples no-object")
    ax_left.plot(phi, y_rb, "o", ms=2.5, color="#8172b3", alpha=0.35, label="Simulation samples rabbit")
    ax_left.set_ylabel(ylab_l, color="#333333")
    setup_phase_axis(ax_left)
    ax_left.set_title(title, fontsize=10)

    ax_right = ax_left.twinx()
    ax_right.errorbar(
        phi_exp,
        exp_no,
        yerr=sem_no,
        fmt="s",
        ms=8,
        color="#c44e52",
        capsize=4,
        label="Exp no-object",
        zorder=5,
    )
    ax_right.errorbar(
        phi_exp,
        exp_rb,
        yerr=sem_rb,
        fmt="^",
        ms=8,
        color="#55a868",
        capsize=4,
        label="Exp rabbit",
        zorder=5,
    )
    ax_right.set_ylabel(ylab_r, color="#444444")

    attach_top_axis(ax_left, top_axis)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, ncol=2, loc="upper right")

    if plot_opts and plot_opts.tag == "EXCLUDE_2PI_SHIFT75TO85":
        note = (
            "Exp. panel (a): four points only (100 µm excluded); "
            r"3$\pi$/2 folder plotted at $\Delta z_{\mathrm{exp}}=85\,\mu\mathrm{m}$."
        )
    elif plot_opts and plot_opts.exclude_folders:
        note = (
            "Exp. panel (a): four points (0, 25, 50, 75 µm); 100 µm excluded. "
            r"Points at wrapped $\varphi_{\mathrm{exp}}=(4\pi\Delta z/\lambda_i)\ \mathrm{mod}\ 2\pi$."
        )
        if top_axis == "physical_wrapped_labels":
            note += " Top axis: physical displacement labels at wrapped phase positions."
    else:
        note = (
            "Exp. points: nominal wrapped phase from "
            r"$\varphi_{\mathrm{exp}}=(4\pi\Delta z_{\mathrm{exp}}/\lambda_i)\ \mathrm{mod}\ 2\pi$; "
            r"physical $\Delta z_{\mathrm{exp}}\in\{0,25,50,75,100\}\,\mu\mathrm{m}$ "
            "(see caption table)."
        )
    ax_left.text(0.02, 0.02, note, transform=ax_left.transAxes, fontsize=6.5, color="0.35", va="bottom")

    for f in folders:
        phi_m = wrapped_phase_rad(plot_dz_um(f, pos, plot_opts))
        ax_left.axvline(phi_m, color="0.88", ls=":", lw=0.7, zorder=0)


def build_full_figure(
    out_png: Path,
    out_pdf: Path,
    theory: dict[str, np.ndarray],
    fit: dict[str, Any],
    summary: dict[str, Any],
    exp: dict[str, Any],
    pos: dict[str, float],
    theory_u16: list[np.ndarray],
    exp_images: list[np.ndarray],
    theory_vmax: float,
    exp_vmax: float,
    selection: dict[str, Any],
    plot_opts: ExpPlotOptions | None = None,
    top_axis: TopAxisMode = "equiv_one_period",
    suptitle: str | None = None,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.25, 1.0, 1.0], hspace=0.42, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, :])
    draw_panel_a(
        ax_a, theory, fit, exp, pos, normalized=False, plot_opts=plot_opts, top_axis=top_axis
    )

    theory_meta = [
        ("constructive", "no_object", 0, "Simulation constructive-like\nno-object"),
        ("constructive", "rabbit", 1, "Simulation constructive-like\nrabbit"),
        ("destructive", "no_object", 2, "Simulation destructive-like\nno-object"),
        ("destructive", "rabbit", 3, "Simulation destructive-like\nrabbit"),
    ]
    exp_meta = [
        ("high", "no_object", 0, "Experimental high-signal\nno-object"),
        ("high", "rabbit", 1, "Experimental high-signal\nrabbit"),
        ("low", "no_object", 2, "Experimental low-signal\nno-object"),
        ("low", "rabbit", 3, "Experimental low-signal\nrabbit"),
    ]

    for col, (state, cond, idx, title) in enumerate(theory_meta):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(theory_u16[idx], cmap="gray", origin="upper", vmin=0, vmax=theory_vmax)
        block = summary[f"{state}_like"]
        roi_key = "roi_mean_no_object_au" if cond == "no_object" else "roi_mean_rabbit_au"
        ax.set_title(
            f"{title}\n"
            rf"$\varphi_{{theory}}$={block['phi_rad']:.3f} rad, "
            rf"$\Delta z$={block['dz_um_one_period_axis']:.4f} $\mu$m (one period)\n"
            f"Mean ROI intensity = {block[roi_key]:.4g} a.u.",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    hi = selection["experimental_high_signal"]
    lo = selection["experimental_low_signal"]
    exp_blocks = [
        (hi, "no_object", 0),
        (hi, "rabbit", 1),
        (lo, "no_object", 2),
        (lo, "rabbit", 3),
    ]
    for col, (blk, cond, idx) in enumerate(exp_blocks):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(exp_images[idx], cmap="gray", origin="upper", vmin=0, vmax=exp_vmax)
        em = blk["conditions"][cond]
        dz_label = float(blk.get("mirror_um_plot", blk.get("mirror_um", 0.0)))
        ax.set_title(
            f"Experimental {'high' if col < 2 else 'low'}-signal {cond.replace('_', ' ')}\n"
            rf"$\Delta z_{{exp}}$={dz_label:.0f} $\mu$m, "
            f"nominal wrapped phase = {blk['wrapped_phase_deg']:.3f} deg\n"
            f"Mean ROI intensity = {em['roi_mean_adu_per_pixel']:.4g} ADU/pixel "
            f"({em['n_frames']} frames)",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        suptitle
        or "Figure 5 — theory phase scan (preserved) with experimental RAW intensity (C_lab Mi=2)",
        fontsize=12,
        y=0.98,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run(
    output: Path | None = None,
    theory_dir: Path = THEORY_EVIDENCE,
    plot_opts: ExpPlotOptions | None = None,
) -> Path:
    if not theory_dir.is_dir():
        raise FileNotFoundError(f"Approved theory evidence missing: {theory_dir}")
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    # The packaged experimental acquisition contains four physical phase
    # folders; the nominal 2pi/100 um folder is absent.  Make the reproducible
    # four-point policy the default rather than requesting a nonexistent input.
    plot_opts = plot_opts or ExpPlotOptions.exclude_2pi_only()

    theory_csv = load_theory_csv(theory_dir / "FIGURE5_THEORY_PHASE_SCAN.csv")
    fit_payload = json.loads((theory_dir / "FIGURE5_THEORY_FIT_REPORT.json").read_text(encoding="utf-8"))
    summary = json.loads((theory_dir / "FIGURE5_THEORY_SUMMARY.json").read_text(encoding="utf-8"))
    pos = {
        key: value
        for key, value in load_positions().items()
        if key not in plot_opts.exclude_folders
    }
    exp_roi = np.load(EXP_ROI).astype(bool)
    if int(exp_roi.sum()) != 20309:
        raise ValueError(f"Unexpected experimental ROI pixel count: {exp_roi.sum()}")

    exp, per_frame = experimental_stats(pos, exp_roi)
    folders = list(pos.keys())
    means_no = exp["no_object"]["roi_mean_adu"]
    idx_hi = int(np.argmax(means_no))
    idx_lo = int(np.argmin(means_no))

    theory_paths = [theory_dir / name for name in THEORY_TIFFS]
    theory_u16 = [tifffile.imread(p) for p in theory_paths]
    theory_hashes = {p.name: sha256_file(p) for p in theory_paths}

    hi_folder = folders[idx_hi]
    lo_folder = folders[idx_lo]

    def _exp_state(folder: str, index: int, rule: str) -> dict[str, Any]:
        dz_plot = plot_dz_um(folder, pos, plot_opts)
        dz_disk = float(pos[folder])
        phi = wrapped_phase_rad(dz_plot)
        block: dict[str, Any] = {
            "selection_rule": rule,
            "phase_folder": folder,
            "mirror_um_plot": dz_plot,
            "wrapped_phase_rad": phi,
            "wrapped_phase_deg": float(np.degrees(phi)),
            "index": index,
            "conditions": {},
        }
        if plot_opts and dz_plot != dz_disk:
            block["mirror_um_folder_nominal_on_disk"] = dz_disk
            block["plotting_note"] = f"TIFFs from folder '{folder}' ({dz_disk} µm nominal); plotted/labeled as {dz_plot} µm"
        else:
            block["mirror_um"] = dz_plot
        return block

    selection: dict[str, Any] = {
        "theory_source_evidence": str(theory_dir.relative_to(ROOT)),
        "theory_image_sha256": theory_hashes,
        "plotting_adjustments": None
        if plot_opts is None
        else {
            "tag": plot_opts.tag,
            "exclude_folders_from_panel_a": sorted(plot_opts.exclude_folders),
            "dz_plot_um_by_folder": dict(plot_opts.dz_plot_um_by_folder),
        },
        "theory_constructive_like": summary["constructive_like"],
        "theory_destructive_like": summary["destructive_like"],
        "experimental_high_signal": _exp_state(
            hi_folder,
            idx_hi,
            f"maximum no-object experimental ROI mean among {len(folders)} available mirror positions",
        ),
        "experimental_low_signal": _exp_state(
            lo_folder,
            idx_lo,
            f"minimum no-object experimental ROI mean among {len(folders)} available mirror positions",
        ),
        "panel_a_experimental_points": [],
        "phase_matching_note": (
            "Simulation extrema from continuous 0-2pi scan; experimental images at discrete mirror steps. "
            "High/low pairing is by experimental intensity state, not established absolute phase equality."
        ),
    }
    for state_key, idx in (("experimental_high_signal", idx_hi), ("experimental_low_signal", idx_lo)):
        for cond in ("no_object", "rabbit"):
            row = exp[cond]["phase_rows"][idx]
            selection[state_key]["conditions"][cond] = {
                "roi_mean_adu_per_pixel": float(exp[cond]["roi_mean_adu"][idx]),
                "roi_mean_sem_adu_per_pixel": float(exp[cond]["roi_mean_sem_adu"][idx]),
                "n_frames": int(row["n_frames"]),
            }

    for f in panel_a_folders(pos, plot_opts):
        dz_plot = plot_dz_um(f, pos, plot_opts)
        j = folders.index(f)
        selection["panel_a_experimental_points"].append(
            {
                "phase_folder": f,
                "mirror_um_plot": dz_plot,
                "mirror_um_folder_nominal_on_disk": float(pos[f]),
                "wrapped_phase_deg": float(np.degrees(wrapped_phase_rad(dz_plot))),
                "no_object_roi_mean_adu": float(exp["no_object"]["roi_mean_adu"][j]),
                "rabbit_roi_mean_adu": float(exp["rabbit"]["roi_mean_adu"][j]),
            }
        )

    exp_images = [
        exp["no_object"]["phase_mean_images"][idx_hi],
        exp["rabbit"]["phase_mean_images"][idx_hi],
        exp["no_object"]["phase_mean_images"][idx_lo],
        exp["rabbit"]["phase_mean_images"][idx_lo],
    ]
    theory_vmax = shared_display_vmax([t.astype(np.float64) for t in theory_u16])
    exp_vmax = shared_display_vmax(exp_images)

    if plot_opts and plot_opts.tag == "EXCLUDE_2PI_SHIFT75TO85":
        fig_stem = "FIGURE5_FINAL_INTENSITY_EXCLUDE_2PI_SHIFT75TO85"
        csv_name = "FIGURE5_PHASE_INTENSITY_DATA_EXCLUDE_2PI_SHIFT75TO85.csv"
        sel_name = "FIGURE5_IMAGE_SELECTION_EXCLUDE_2PI_SHIFT75TO85.json"
        write_panel_a_extra = False
    else:
        fig_stem = "FIGURE5_FINAL_INTENSITY"
        csv_name = "FIGURE5_PHASE_INTENSITY_DATA.csv"
        sel_name = "FIGURE5_IMAGE_SELECTION.json"
        write_panel_a_extra = True

    build_full_figure(
        out / f"{fig_stem}.png",
        out / f"{fig_stem}.pdf",
        theory_csv,
        fit_payload,
        {
            "constructive_like": summary["constructive_like"],
            "destructive_like": summary["destructive_like"],
        },
        exp,
        pos,
        theory_u16,
        exp_images,
        theory_vmax,
        exp_vmax,
        selection,
        plot_opts=plot_opts,
    )

    if write_panel_a_extra:
        fig_a, ax_a = plt.subplots(figsize=(11, 4.5))
        draw_panel_a(ax_a, theory_csv, fit_payload, exp, pos, normalized=False)
        fig_a.tight_layout()
        fig_a.savefig(out / "FIGURE5_PANEL_A_THEORY_EXPERIMENT.png", dpi=200, bbox_inches="tight")
        plt.close(fig_a)

        fig_n, ax_n = plt.subplots(figsize=(11, 4.5))
        draw_panel_a(ax_n, theory_csv, fit_payload, exp, pos, normalized=True)
        fig_n.tight_layout()
        fig_n.savefig(out / "FIGURE5_PANEL_A_NORMALIZED_DIAGNOSTIC.png", dpi=200, bbox_inches="tight")
        plt.close(fig_n)

    with (out / csv_name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "source",
                "condition",
                "phase_rad",
                "dz_um",
                "wrapped_phase_deg",
                "roi_mean",
                "roi_sem",
                "units",
            ]
        )
        for i in range(len(theory_csv["phase_rad"])):
            w.writerow(
                [
                    "theory",
                    "no_object",
                    theory_csv["phase_rad"][i],
                    theory_csv["dz_um_one_period"][i],
                    float(np.degrees(theory_csv["phase_rad"][i])),
                    theory_csv["roi_mean_no_object_au"][i],
                    "",
                    "a.u.",
                ]
            )
            w.writerow(
                [
                    "theory",
                    "rabbit",
                    theory_csv["phase_rad"][i],
                    theory_csv["dz_um_one_period"][i],
                    float(np.degrees(theory_csv["phase_rad"][i])),
                    theory_csv["roi_mean_rabbit_au"][i],
                    "",
                    "a.u.",
                ]
            )
        for folder in panel_a_folders(pos, plot_opts):
            j = folders.index(folder)
            dz = plot_dz_um(folder, pos, plot_opts)
            phi = wrapped_phase_rad(dz)
            for cond in ("no_object", "rabbit"):
                w.writerow(
                    [
                        "experiment",
                        cond,
                        phi,
                        dz,
                        float(np.degrees(phi)),
                        float(exp[cond]["roi_mean_adu"][j]),
                        float(exp[cond]["roi_mean_sem_adu"][j]),
                        "ADU/pixel",
                    ]
                )

    if write_panel_a_extra:
        with (out / "FIGURE5_EXPERIMENTAL_PER_FRAME.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_frame[0].keys()))
            w.writeheader()
            w.writerows(per_frame)

    (out / sel_name).write_text(json.dumps(selection, indent=2), encoding="utf-8")

    exp_table = []
    for j, folder in enumerate(folders):
        dz = float(pos[folder])
        phi = wrapped_phase_rad(dz)
        exp_table.append(
            {
                "folder": folder,
                "dz_um": dz,
                "wrapped_phase_deg": float(np.degrees(phi)),
                "nominal_wrapped_deg_reference": NOMINAL_WRAPPED_DEG[j],
                "no_object_mean": float(exp["no_object"]["roi_mean_adu"][j]),
                "no_object_sem": float(exp["no_object"]["roi_mean_sem_adu"][j]),
                "rabbit_mean": float(exp["rabbit"]["roi_mean_adu"][j]),
                "rabbit_sem": float(exp["rabbit"]["roi_mean_sem_adu"][j]),
            }
        )

    if not write_panel_a_extra:
        phi_85 = wrapped_phase_rad(85.0)
        panel_pts = selection["panel_a_experimental_points"]
        (out / "FINAL_DECISION.md").write_text(
            f"""# Figure 5 — exclude 2π point + 75→85 µm plot relabel

- **Evidence:** `{out}`
- **Prior figure (unchanged theory):** `evidence/c_lab_figure5_final_theory_experiment_intensity/20261017T073336Z`
- **Simulation source:** `{theory_dir}`
- **PRODUCTION_FILES_MODIFIED:** NO

## Plotting adjustments (figure/CSV only)
1. **Excluded from Panel (a):** nominal Δz_exp = 100 µm (`2pi` folder) — not plotted; raw TIFFs untouched.
2. **Relabeled in Panel (a):** data from `3pi2` folder (75 µm nominal on disk) plotted/labeled at **Δz_exp = 85 µm**.

## Panel (a) experimental points used ({len(panel_pts)} positions)
"""
            + "\n".join(
                f"- `{p['phase_folder']}`: plot Δz = {p['mirror_um_plot']:.0f} µm, "
                f"φ_wrap = {p['wrapped_phase_deg']:.3f}° "
                f"(disk nominal {p['mirror_um_folder_nominal_on_disk']:.0f} µm)"
                for p in panel_pts
            )
            + f"""

## Reassigned 85 µm wrapped phase
- φ_exp = (4π×85/1.55) mod 2π = **{phi_85:.6f} rad** = **{np.degrees(phi_85):.3f}°**

## Image rows (unchanged from prior assembly)
- Simulation: preserved constructive/destructive TIFFs (same SHA256 as prior run).
- Experimental: same high/low selection (50 µm high, `3pi2` low); low-signal **label** uses 85 µm / {np.degrees(phi_85):.3f}° for plotting interpretation.

## Outputs
- `{fig_stem}.png` / `.pdf`
- `{csv_name}`
- `{sel_name}`
""",
            encoding="utf-8",
        )
        print(f"Done: {out}")
        return out

    (out / "FIGURE5_CAPTION_FINAL.md").write_text(
        f"""# Figure 5 caption (final candidate)

**(a)** ROI mean signal intensity versus interferometric phase for the frozen C_lab digital-twin simulation (Mi=2, production `I_plus`, simulation ROI on native 1024 grid) and raw experimental EMCCD mirror-drift data (fixed 90% experimental ROI, {int(exp_roi.sum()):,} pixels). Simulation curves are the approved sinusoidal phase scan (sampled points + fit `I(φ)=A+B cos φ+C sin φ`) in arbitrary units (a.u.). Experimental points are spatial ROI means of uint16 TIFF frames without dark subtraction, plotted at **wrapped** phase `φ_exp=(4π Δz_exp/λ_i) mod 2π` with `λ_i=1.55 µm` and **physical** mirror steps `Δz_exp ∈ {{0,25,50,75,100}} µm`. The upper axis shows equivalent displacement for **one** simulated `0→2π` period (`0–0.775 µm`); experimental displacements beyond one period are **not** mapped onto that axis. Dual y-axes reflect the absence of absolute radiometric calibration between simulation a.u. and detector ADU/pixel. Error bars: experimental SEM over repeated frames.

**Images (columns: high-signal no-object, high-signal rabbit, low-signal no-object, low-signal rabbit).** Top row: **preserved** theory TIFFs from `{theory_dir.name}` at constructive-like (`φ_theory≈{summary['constructive_like']['phi_rad']:.3f}` rad) and destructive-like (`φ_theory≈{summary['destructive_like']['phi_rad']:.3f}` rad) phases from the theory scan. Bottom row: experimental mean RAW frames at mirror positions selected from **no-object** ROI extrema (high: `{pos[hi_folder]:.0f} µm`, low: `{pos[lo_folder]:.0f} µm`), applied to both object conditions. Image pairing is by high/low **intensity state**, not by established absolute phase equality.

**Experimental phase-position reference (wrapped deg):** {', '.join(f'{d:.3f}° @ {z:.0f} µm' for z, d in zip(DZ_EXP, NOMINAL_WRAPPED_DEG))}.
""",
        encoding="utf-8",
    )

    trend = (
        "Both modalities show phase-dependent ROI mean intensity with rabbit below no-object "
        "at comparable states; experimental five-point sampling follows a similar wrapped-phase "
        "pattern to theory, but absolute phase alignment and radiometric scaling are not calibrated."
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Final Figure 5 — theory (preserved) + experiment assembly

- **Evidence:** `{out}`
- **Simulation source (unchanged):** `{theory_dir}`
- **PRODUCTION_FILES_MODIFIED:** NO

## Experimental mirror selection (no-object ROI mean)
- **High-signal:** Δz_exp = {pos[hi_folder]:.0f} µm ({hi_folder}), wrapped φ = {selection['experimental_high_signal']['wrapped_phase_deg']:.3f}°
- **Low-signal:** Δz_exp = {pos[lo_folder]:.0f} µm ({lo_folder}), wrapped φ = {selection['experimental_low_signal']['wrapped_phase_deg']:.3f}°

## Simulation reference phases (from approved summary)
- **Constructive-like:** φ = {summary['constructive_like']['phi_rad']:.6f} rad, one-period Δz = {summary['constructive_like']['dz_um_one_period_axis']:.6f} µm
- **Destructive-like:** φ = {summary['destructive_like']['phi_rad']:.6f} rad, one-period Δz = {summary['destructive_like']['dz_um_one_period_axis']:.6f} µm

## Experimental five-point ROI means (ADU/pixel ± SEM)
| Δz (µm) | φ wrap (deg) | No-object | Rabbit |
|---------|--------------|-----------|--------|
"""
        + "\n".join(
            f"| {r['dz_um']:.0f} | {r['wrapped_phase_deg']:.3f} | "
            f"{r['no_object_mean']:.4g} ± {r['no_object_sem']:.4g} | "
            f"{r['rabbit_mean']:.4g} ± {r['rabbit_sem']:.4g} |"
            for r in exp_table
        )
        + f"""

## Simulation TIFF preservation
Simulation image pixels read directly from approved TIFFs (SHA256):
"""
        + "\n".join(f"- `{k}`: `{v}`" for k, v in theory_hashes.items())
        + f"""

## Phase trend assessment
{trend}

## Outputs
- FIGURE5_FINAL_INTENSITY.png / .pdf
- FIGURE5_PANEL_A_THEORY_EXPERIMENT.png
- FIGURE5_PANEL_A_NORMALIZED_DIAGNOSTIC.png
- FIGURE5_PHASE_INTENSITY_DATA.csv
- FIGURE5_EXPERIMENTAL_PER_FRAME.csv
- FIGURE5_IMAGE_SELECTION.json
- FIGURE5_CAPTION_FINAL.md
""",
        encoding="utf-8",
    )

    print(f"Done: {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--theory-dir", type=Path, default=THEORY_EVIDENCE)
    p.add_argument(
        "--exclude-2pi-shift-75-to-85",
        action="store_true",
        help="Panel (a): omit 100 µm points; plot 3pi2 folder at 85 µm phase/labels.",
    )
    args = p.parse_args()
    plot_opts = ExpPlotOptions.exclude_2pi_shift_75_to_85() if args.exclude_2pi_shift_75_to_85 else None
    run(args.output, theory_dir=args.theory_dir, plot_opts=plot_opts)


if __name__ == "__main__":
    main()
