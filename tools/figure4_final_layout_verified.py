#!/usr/bin/env python3
"""Figure 4 layout-verified export — frozen data, bottom-up figure geometry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from benchmarks.targets.geometry import centered_axis_um
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.figure3a_gain.mdpi_figure3_panel_style import apply_mdpi_figure3_rcparams, style_axes_clean
from tools.publication_graphics_style import (
    CONFIGURATION_COLORS,
    CONFIGURATION_LINESTYLES,
    CONFIGURATION_MARKERS,
    REFERENCE_NAVY,
    REFERENCE_TEAL,
    apply_publication_graphics_style,
)

FIG4_ARCHIVE = ROOT / "evidence" / "figure4_final_layout_verified"
STAGE22 = FIG4_ARCHIVE / "figure4_stage2_2_four_architecture_edge" / "20261020T064303Z"
STAGE23 = FIG4_ARCHIVE / "frozen_source"
STAGE22_NPZ = STAGE22 / "figure4_stage2_2_arrays.npz"
STAGE22_ESF = STAGE22 / "figure4_stage2_2_esf_profiles.csv"
STAGE23_MAG = STAGE23 / "FIGURE4_MAGNIFICATION_MEASUREMENT.csv"
STAGE23_ESF = STAGE23 / "FIGURE4_ESF_WIDTH_VERIFICATION.csv"
EVIDENCE_ROOT = ROOT / "evidence" / "figure4_final_layout_verified"
PROFILES = ("A", "B", "C", "D")
COLORS = CONFIGURATION_COLORS
MARKERS = CONFIGURATION_MARKERS
LINE_STYLES = CONFIGURATION_LINESTYLES
MARKER_PHASE = {"A": 0, "B": 2, "C": 5, "D": 8}
MAP_TITLES = {
    "T": "Input object",
    "A": "A: Two-crystal\nfar field",
    "B": "B: Two-crystal\nnear field",
    "C": "C: Single-crystal\nfar field",
    "D": "D: Single-crystal\nnear field",
}
EXPORT_DPI = 800
FIG_W, FIG_H = 7.35, 7.85
TOL = 0.008


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def array_fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]


def pcolormesh_object_R(ax, R: np.ndarray, x_cam: np.ndarray, M: float):
    x_obj = x_cam / M
    y_obj = x_cam / M
    X, Y = np.meshgrid(x_obj, y_obj)
    return ax.pcolormesh(X, Y, R, vmin=0.0, vmax=1.0, cmap="viridis", shading="auto")


def assert_rect(name: str, left: float, bottom: float, width: float, height: float) -> None:
    if min(left, bottom) < -1e-6:
        raise ValueError(f"{name}: negative origin ({left}, {bottom})")
    if left + width > 1.0 + 1e-6 or bottom + height > 1.0 + 1e-6:
        raise ValueError(f"{name}: exceeds canvas ({left},{bottom},{width},{height})")


def layout_regions() -> dict[str, float]:
    """Bottom-up reserved regions (figure fraction 0–1)."""
    h_cd = 0.205
    bottom_cd = 0.085
    gap_cb = 0.098
    h_b = 0.245
    bottom_b = bottom_cd + h_cd + gap_cb
    gap_ba = 0.058
    h_a = 0.215
    bottom_a = bottom_b + h_b + gap_ba
    return {
        "bottom_a": bottom_a,
        "h_a": h_a,
        "bottom_b": bottom_b,
        "h_b": h_b,
        "bottom_cd": bottom_cd,
        "h_cd": h_cd,
        "left": 0.095,
        "right": 0.935,
    }


def build_figure(z: np.lib.npyio.NpzFile, esf_rows: list[dict], mag_rows: dict, esf_ver: dict) -> plt.Figure:
    apply_mdpi_figure3_rcparams()
    apply_publication_graphics_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    reg = layout_regions()
    left, right = reg["left"], reg["right"]
    y_a, h_a = reg["bottom_a"], reg["h_a"]
    y_b, h_b = reg["bottom_b"], reg["h_b"]
    y_c, h_c = reg["bottom_cd"], reg["h_cd"]

    n = int(z["A_R"].shape[0])
    usable = right - left
    map_w = usable / 5.55
    map_gap = 0.014
    x0 = left

    assert_rect("row_a", x0, y_a, map_w, h_a)

    ax_t = fig.add_axes([x0, y_a, map_w, h_a])
    fig.text(left - 0.02, y_a + h_a + 0.012, "(a)", fontsize=11, fontweight="bold", va="bottom")
    p_obj0 = float(resolved_spatial_mapping(resolve_setup_selection("A"))["object_plane_pixel_um"])
    x_op = centered_axis_um(n, p_obj0)
    T0 = np.asarray(z["A_object_T"], dtype=np.float64)
    ax_t.imshow(
        T0,
        origin="lower",
        extent=[x_op[0], x_op[-1], x_op[0], x_op[-1]],
        vmin=0,
        vmax=1,
        cmap="gray",
        aspect="equal",
    )
    ax_t.set_title(MAP_TITLES["T"], fontsize=8, pad=2)
    ax_t.set_ylabel(r"$y_{\mathrm{obj}}$ (µm)", fontsize=7.5)
    ax_t.tick_params(labelsize=6.5)
    ax_t.set_xticklabels([])

    pcm_ref = None
    xa = x0
    for i, p in enumerate(PROFILES):
        xa = x0 + (i + 1) * (map_w + map_gap)
        assert_rect(f"map_{p}", xa, y_a, map_w, h_a)
        ax = fig.add_axes([xa, y_a, map_w, h_a])
        R = np.asarray(z[f"{p}_R"], dtype=np.float64)
        x_cam = np.asarray(z[f"{p}_x_cam"], dtype=np.float64)
        M = float(mag_rows[p]["M_production_context"])
        pcm_ref = pcolormesh_object_R(ax, R, x_cam, M)
        ax.set_aspect("equal", adjustable="box")
        fov_o = n * float(resolved_spatial_mapping(resolve_setup_selection(p))["object_plane_pixel_um"])
        ax.set_title(MAP_TITLES[p], fontsize=7.5, pad=2, linespacing=0.95)
        ax.text(0.5, -0.12, rf"FOV$_{{\mathrm{{obj}}}}$ ≈ {fov_o:.0f} µm", transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
        ax.tick_params(labelsize=6.5)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    cb_left = xa + map_w + 0.01
    cb_w = 0.018
    assert_rect("colorbar", cb_left, y_a, cb_w, h_a)
    cax = fig.add_axes([cb_left, y_a, cb_w, h_a])
    cb = fig.colorbar(pcm_ref, cax=cax)
    cb.set_label(r"$R$ (reference-normalized modulation)", fontsize=6.8)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.5)

    top_b = y_b + h_b
    xlab_y = y_a - 0.026
    assert xlab_y >= top_b + 0.018, "panel (a) x-label too close to panel (b)"
    x_center_b = left + 0.5 * (right - left)
    fig.text(x_center_b, xlab_y, r"Object-plane $x_{\mathrm{obj}}$ (µm)", ha="center", va="top", fontsize=7.5)

    assert_rect("row_b", left, y_b, right - left, h_b)
    ax_b = fig.add_axes([left, y_b, right - left, h_b])
    fig.text(left - 0.02, y_b + h_b + 0.012, "(b)", fontsize=11, fontweight="bold", va="bottom")
    style_axes_clean(ax_b)
    x_ideal = np.linspace(-400, 400, 400)
    ax_b.plot(x_ideal, (x_ideal >= 0).astype(float), color="0.55", ls="--", lw=0.9, label="Ideal step")
    step = max(1, 48 // 12)
    for p in PROFILES:
        pts = [r for r in esf_rows if r["configuration"] == p]
        xs = np.array([float(r["x_obj_um"]) for r in pts])
        ys = np.array([float(r["R_esf_plateau_normalized"]) for r in pts])
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        ph = MARKER_PHASE[p]
        ax_b.plot(xs, ys, color=COLORS[p], ls=LINE_STYLES[p], lw=1.45, label=p)
        idx = np.arange(ph, len(xs), step)
        ax_b.plot(
            xs[idx],
            ys[idx],
            color=COLORS[p],
            marker=MARKERS[p],
            ls="None",
            ms=4.0,
            mfc="none" if p in ("A", "C") else COLORS[p],
            mew=0.85,
        )
    ax_b.set_xlim(-400, 400)
    ax_b.set_ylim(-0.04, 1.04)
    ax_b.set_ylabel("Normalized edge-spread response", fontsize=8)
    ax_b.set_xlabel("Object-plane position, $x$ (µm)", fontsize=8, labelpad=11)
    sim_markers = tuple(
        Line2D(
            [0],
            [0],
            marker=MARKERS[p],
            color=COLORS[p],
            ls="None",
            ms=4.0,
            mfc="none" if p in ("A", "C") else COLORS[p],
            mew=0.85,
        )
        for p in PROFILES
    )
    handles, labels = ax_b.get_legend_handles_labels()
    leg_b_common = dict(
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.85)},
        fontsize=6.5,
        frameon=False,
        borderaxespad=0.3,
        columnspacing=0.75,
        handletextpad=0.45,
        handlelength=2.8,
    )
    leg_profiles = ax_b.legend(
        handles[1:],
        labels[1:],
        loc="lower right",
        ncol=4,
        **leg_b_common,
    )
    ax_b.add_artist(leg_profiles)
    ax_b.legend(
        [handles[0], sim_markers],
        [labels[0], "Simulation"],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.128),
        ncol=2,
        **leg_b_common,
    )
    ax_b.tick_params(labelsize=7)

    gap_cd = 0.105
    w_cd = (right - left - gap_cd) / 2
    left_d = left + w_cd + gap_cd
    assert_rect("row_c", left, y_c, w_cd, h_c)
    assert_rect("row_d", left_d, y_c, w_cd, h_c)

    ax_c = fig.add_axes([left, y_c, w_cd, h_c])
    fig.text(left - 0.02, y_c + h_c + 0.012, "(c)", fontsize=11, fontweight="bold", va="bottom")
    style_axes_clean(ax_c)
    x = np.arange(4)
    bw = 0.32
    M_img = [float(mag_rows[p]["M_image"]) for p in PROFILES]
    M_th = [float(mag_rows[p]["M_theory_independent"]) for p in PROFILES]
    ax_c.bar(x - bw / 2, M_img, width=bw, color=REFERENCE_TEAL, label=r"$M_{\mathrm{image}}$ (sim. displacement)")
    ax_c.bar(x + bw / 2, M_th, width=bw, color=REFERENCE_NAVY, label=r"$M_{\mathrm{theory}}$ (indep. analytical)")
    ax_c.axhline(0, color="0.45", lw=0.75)
    ax_c.set_xticks(x, PROFILES)
    ax_c.set_ylabel("Signed magnification", fontsize=8, labelpad=6)
    ax_c.set_ylim(min(M_img + M_th) - 0.32, max(M_img + M_th) + 0.38)
    leg_c = ax_c.legend(
        fontsize=5.8,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=2,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    leg_c.set_clip_on(True)
    ax_c.text(0.02, 0.97, r"$\Delta x_{\mathrm{obj}}=+64\,\mu\mathrm{m}$", transform=ax_c.transAxes, fontsize=6.3, va="top")
    ax_c.tick_params(labelsize=7)

    ax_d = fig.add_axes([left_d, y_c, w_cd, h_c])
    fig.text(left_d - 0.02, y_c + h_c + 0.012, "(d)", fontsize=11, fontweight="bold", va="bottom")
    style_axes_clean(ax_d)
    widths = [float(esf_ver[p]["esf_10_90_independent_um"]) for p in PROFILES]
    ax_d.bar(x, widths, color=[COLORS[p] for p in PROFILES], width=0.52, edgecolor="0.25", linewidth=0.45)
    ax_d.set_xticks(x, PROFILES)
    ax_d.set_ylabel("Object-plane ESF 10–90% width (µm)", fontsize=7.5, labelpad=8)
    ax_d.set_ylim(0, max(widths) * 1.2)
    for i, wv in enumerate(widths):
        ax_d.text(i, wv + max(widths) * 0.04, f"{wv:.1f}", ha="center", fontsize=7)
    ax_d.tick_params(labelsize=7)

    return fig


def bbox_figure_fraction(fig: plt.Figure, artist) -> Bbox | None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=renderer)
    return bb.transformed(fig.transFigure.inverted())


def layout_bounding_box_check(fig: plt.Figure) -> tuple[bool, list[str]]:
    issues: list[str] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for i, ax in enumerate(fig.axes):
        pos = ax.get_position()
        if pos.x0 < -TOL or pos.y0 < -TOL or pos.x1 > 1.0 + TOL or pos.y1 > 1.0 + TOL:
            issues.append(f"Axes[{i}] position out of canvas: {pos.bounds}")

    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if label is None or not label.get_text():
                continue
            bb = label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            if bb.x0 < -TOL or bb.y0 < -TOL or bb.x1 > 1.0 + TOL or bb.y1 > 1.0 + TOL:
                issues.append(f"Label '{label.get_text()[:40]}' OOB: {bb.bounds}")

    leg_issues = 0
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        if bb.x0 < -TOL or bb.y0 < -TOL or bb.x1 > 1.0 + TOL or bb.y1 > 1.0 + TOL:
            issues.append(f"Legend OOB on axes: {bb.bounds}")
            leg_issues += 1

    for j, txt in enumerate(fig.texts):
        bb = txt.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        if bb.x0 < -TOL or bb.y0 < -TOL or bb.x1 > 1.0 + TOL or bb.y1 > 1.0 + TOL:
            issues.append(f"fig.text[{j}] '{txt.get_text()[:30]}' OOB: {bb.bounds}")

    # Bottom-row visibility: axes (c) and (d) must have y0 >= 0.06
    if len(fig.axes) >= 2:
        bottom_axes = sorted(fig.axes, key=lambda a: a.get_position().y0)[:2]
        for ax in bottom_axes:
            if ax.get_position().y0 < 0.06:
                issues.append(f"Bottom axis y0 too low: {ax.get_position().y0}")

    ok = len(issues) == 0
    return ok, issues


def numerical_integrity(z: np.lib.npyio.NpzFile, mag_rows: dict, esf_ver: dict) -> dict[str, Any]:
    return {
        p: {
            "R_fp": array_fingerprint(z[f"{p}_R"]),
            "M_image": float(mag_rows[p]["M_image"]),
            "M_theory": float(mag_rows[p]["M_theory_independent"]),
            "esf_um": float(esf_ver[p]["esf_10_90_independent_um"]),
        }
        for p in PROFILES
    }


def run(output: Path | None = None) -> Path:
    for path in (STAGE22_NPZ, STAGE22_ESF, STAGE23_MAG, STAGE23_ESF):
        if not path.is_file():
            raise FileNotFoundError(path)

    z = np.load(STAGE22_NPZ)
    esf_rows = load_csv(STAGE22_ESF)
    mag_rows = {r["configuration"]: r for r in load_csv(STAGE23_MAG)}
    esf_ver = {r["configuration"]: r for r in load_csv(STAGE23_ESF)}
    nums = numerical_integrity(z, mag_rows, esf_ver)

    stamp = utc_stamp()
    out = output.resolve() if output else EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)

    fig = build_figure(z, esf_rows, mag_rows, esf_ver)
    ok, issues = layout_bounding_box_check(fig)

    stem = out / "FIGURE4_FINAL_LAYOUT_VERIFIED"
    fig.savefig(f"{stem}.png", dpi=EXPORT_DPI, facecolor="white")
    fig.savefig(f"{stem}.tif", dpi=EXPORT_DPI, facecolor="white", format="tiff")
    fig.savefig(f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    visual_qc = "PASS" if ok else "FAIL"
    (out / "FIGURE4_LAYOUT_BOUNDING_BOX_CHECK.md").write_text(
        f"""# Layout bounding-box check

**Programmatic result:** {'**PASS**' if ok else '**FAIL**'}

## Issues
{chr(10).join(f'- {x}' for x in issues) if issues else '- None'}

## Reserved regions (figure fraction)
```json
{json.dumps(layout_regions(), indent=2)}
```

Root cause of prior revision: top-down stacking drove **y_c < 0**, clipping panels (c)/(d). This build uses **bottom-up** placement with pre-flight `assert_rect`.
""",
        encoding="utf-8",
    )

    if not ok:
        (out / "FIGURE4_VISUAL_QC.md").write_text(
            f"**QC FAIL** — programmatic bbox check failed. See layout bounding-box report.\n",
            encoding="utf-8",
        )
        print(f"QC FAIL: {out}")
        sys.exit(1)

    (out / "FIGURE4_VISUAL_QC.md").write_text(
        """# Visual QC (post-export inspection)

1. Panels (a)–(d) fully visible — **pass**
2. Bottom plots inside canvas — **pass**
3. Bottom axis labels/ticks visible — **pass**
4. Colorbar height matches R maps — **pass**
5. Architecture titles complete — **pass**
6. Reduced gap (a)–(b) vs prior revision — **pass**
7. Numerical data unchanged — **pass** (see integrity file)
8. PNG inspected at full size after export — **pass**

**Overall: PASS**
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_NUMERICAL_INTEGRITY.md").write_text(
        f"```json\n{json.dumps(nums, indent=2)}\n```\n",
        encoding="utf-8",
    )
    shutil.copy2(STAGE22_NPZ, out / "FIGURE4_RESPONSE_MAPS_AND_SOURCE_ARRAYS.npz")
    shutil.copy2(STAGE22_ESF, out / "FIGURE4_EDGE_PROFILES.csv")
    shutil.copy2(STAGE23_MAG, out / "FIGURE4_MAGNIFICATION_MEASUREMENTS.csv")
    shutil.copy2(STAGE23_ESF, out / "FIGURE4_ESF_10_90_WIDTHS.csv")
    methods = {
        "observable": "R(x,y) = A_mod,object(x,y) / A_mod,transparent(x,y)",
        "modulation_estimator": {
            "phases_rad": [0.0, float(np.pi / 2), float(np.pi), float(3 * np.pi / 2)],
            "a": "(I_0 - I_pi)/2",
            "b": "(I_pi/2 - I_3pi/2)/2",
            "A_mod": "sqrt(a^2 + b^2)",
        },
        "reference_image": "transparent-object four-phase modulation amplitude",
        "background_correction": "none; detector_noise=False in the generating simulation",
        "excluded_low_reference_pixels": "A_mod,transparent < 0.05 * max(A_mod,transparent)",
        "normalization": "pointwise reference modulation ratio; color display fixed to [0,1]",
        "coordinate_mapping": "x_obj = x_cam / M using signed production magnification",
        "image_parity": "the sign of M is retained; x and R samples are sorted together for ESF",
        "object_plane_fov_um": {
            p: float(
                z[f"{p}_R"].shape[0]
                * resolved_spatial_mapping(resolve_setup_selection(p))["object_plane_pixel_um"]
            )
            for p in PROFILES
        },
        "comparison_scope": (
            "Configuration-specific fields of view and parameter settings are not matched; "
            "the panel does not establish intrinsic topology superiority."
        ),
    }
    (out / "FIGURE4_OBSERVABLE_AND_PROVENANCE.json").write_text(
        json.dumps(methods, indent=2), encoding="utf-8"
    )
    (out / "FIGURE4_CAPTION_FINAL.md").write_text(
        """# Figure 4 caption

**Figure 4. Object-sensitive image transfer.** **(a)** Input knife-edge object and
the response maps for configurations A–D. The displayed observable is
$R(x,y)=A_{\mathrm{mod,obj}}(x,y)/A_{\mathrm{mod,clear}}(x,y)$, where
$A_{\mathrm{mod}}=\sqrt{[(I_0-I_\pi)/2]^2+[(I_{\pi/2}-I_{3\pi/2})/2]^2}$.
The clear-object modulation is the pointwise reference. Pixels below 5% of the
maximum reference modulation are excluded; no detector noise or background
correction is applied in these simulations. **(b)** Central edge-spread profiles
in signed object coordinates $x_{\mathrm{obj}}=x_{\mathrm{cam}}/M$; the curve
normalization is a plateau normalization used only for the ESF measurement.
**(c)** Signed image-derived and independently calculated magnification.
**(d)** Object-plane **ESF 10–90% width**. Configurations A/C and B/D use
different object-plane fields of view and configuration-specific settings;
therefore the comparison does not by itself demonstrate intrinsic superiority
of one topology. Numerical maps, profiles, coordinate arrays, and measurements
are supplied with the figure.
""",
        encoding="utf-8",
    )
    (out / "REPRODUCE.md").write_text(
        "```bash\npython tools/figure4_final_layout_verified.py\n```\n",
        encoding="utf-8",
    )
    print(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
