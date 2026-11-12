#!/usr/bin/env python3
"""Replot Figure 3(e) for Mr Rajabi bundle — MDPI panel style, fixed-detector CSV."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tools.qiup_figure3_final_article as q3
import tools.taskY5_figure3_final_corrections as y5
from verification.figure3a_gain.mdpi_figure3_panel_style import (
    FIG3_DPI_EXPORT,
    FIG3_PANEL_SIZE_IN,
    apply_mdpi_figure3_rcparams,
    style_axes_clean,
)

RAJABI = ROOT / "evidence" / "qiup_figure3_final_article_Mr Rajabi" / "qiup_figure3_final_article"
CURRENT = RAJABI / "CURRENT"
CSV_PATH = CURRENT / "data" / "figure3e_fixed_detector_grid_sensitivity_ABCD.csv"
PLOT_NS = (48, 96, 144, 192, 240)
N_REF = 240
PROFILES = y5.PROFILES


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def draw_panel_e_fixed_detector(ax, rows: list[dict[str, str]]) -> dict:
    ymax = 0.0
    series: dict[str, list[tuple[int, float]]] = {p: [] for p in PROFILES}
    for p in PROFILES:
        for n in PLOT_NS:
            match = next(
                r for r in rows if r["configuration"] == p and int(r["N_num"]) == n
            )
            y = float(match["response_relative_L2_vs_N240"])
            series[p].append((n, y))
            ymax = max(ymax, y)
    # A≈C and B≈D numerically; draw paired configs in order with distinct marker layering.
    draw_order = ("A", "C", "B", "D")
    marker_style = {
        "A": {"ms": 6.2, "mew": 1.25, "z": 2},
        "C": {"ms": 7.0, "mew": 1.85, "z": 4},
        "B": {"ms": 6.8, "mew": 1.45, "z": 3},
        "D": {"ms": 7.8, "mew": 2.0, "z": 5},
    }
    for p in draw_order:
        xs = [t[0] for t in series[p]]
        ys = [t[1] for t in series[p]]
        st = marker_style[p]
        color = q3.FIG3C_COLORS[p]
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=q3.FIG3C_DASH[p],
            lw=1.8,
            alpha=0.92,
            zorder=st["z"],
        )
        ax.plot(
            xs,
            ys,
            linestyle="none",
            marker=q3.CONFIGURATION_MARKERS[p],
            color=color,
            ms=st["ms"],
            mfc="white",
            mec=color,
            mew=st["mew"],
            zorder=10 + st["z"],
            label=f"Configuration {p}",
        )
    ax.set_xlabel(r"Internal numerical grid size, $N_\mathrm{num}$")
    ax.set_ylabel(
        rf"Relative spatial $L_2$ error vs $N_\mathrm{{ref}}$={N_REF}",
        fontsize=8.5,
    )
    ax.set_xticks(list(PLOT_NS))
    ax.set_xlim(min(PLOT_NS) - 12, max(PLOT_NS) + 12)
    ax.set_ylim(-0.00002, ymax * 1.18 + 1.0e-6)
    style_axes_clean(ax)
    note = (
        "48×48 detector; 16 µm pitch; 768 µm FOV.\n"
        rf"Area-weighted object-response $L_2$ error vs $N_\mathrm{{ref}}$={N_REF}"
    )
    note_artist = ax.text(
        0.5,
        0.985,
        note,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=6.7,
        color="#222",
        linespacing=1.05,
        bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="#cccccc", alpha=0.96),
        clip_on=False,
        zorder=30,
    )
    ax.figure.canvas.draw()
    note_bbox = note_artist.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    note_bbox_axes = note_bbox.transformed(ax.transAxes.inverted())
    leg = ax.legend(
        fontsize=6.5,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cccccc",
        loc="upper center",
        bbox_to_anchor=(0.5, float(note_bbox_axes.y0)),
        bbox_transform=ax.transAxes,
        ncol=4,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=0.85,
    )
    if leg is not None:
        leg.set_zorder(29)
    return {"plot_ns": list(PLOT_NS), "n_ref": N_REF, "y_max": ymax, "profiles": list(PROFILES)}


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=FIG3_DPI_EXPORT)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    try:
        fig.savefig(
            f"{stem}.tif",
            dpi=FIG3_DPI_EXPORT,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception:
        pass


def _legacy_standalone_main_disabled() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)
    rows = load_rows()
    apply_mdpi_figure3_rcparams()
    fig, ax = plt.subplots(figsize=FIG3_PANEL_SIZE_IN)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.18, top=0.96)
    meta = draw_panel_e_fixed_detector(ax, rows)
    panels = CURRENT / "figure3" / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    stem = panels / "FIGURE3e_FINAL_ARTICLE"
    export_figure(fig, stem)
    plt.close(fig)
    stamp = utc_stamp()
    (RAJABI / "LATEST.txt").write_text(f"{stamp}\n{CURRENT}\n", encoding="utf-8")
    (CURRENT / "SOURCE_STAMP.txt").write_text(
        f"figure3e_panel_replot={stamp}\ncsv={CSV_PATH.name}\n", encoding="utf-8"
    )
    (CURRENT / "data" / "panel_e_fixed_detector_replot.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (CURRENT / "REPRODUCE_FIGURE3e.md").write_text(
        f"```bash\npython tools/qiup_figure3e_rajabi_panel_replot.py\n```\n\nData: `{CSV_PATH}`\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))


def main() -> int:
    """Compatibility entry point; regenerate from the authoritative shared pipeline."""
    return q3.main()


if __name__ == "__main__":
    raise SystemExit(main())
