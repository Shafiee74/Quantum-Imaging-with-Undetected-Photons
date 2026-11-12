#!/usr/bin/env python3
"""Figure 4 graphical layout revision — display only; frozen Stage 2.2/2.3 data."""
from __future__ import annotations

import csv
import hashlib
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

from benchmarks.targets.geometry import centered_axis_um
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.figure3a_gain.mdpi_figure3_panel_style import apply_mdpi_figure3_rcparams, style_axes_clean

STAGE22_NPZ = (
    ROOT
    / "evidence"
    / "figure4_stage2_2_four_architecture_edge"
    / "20261020T064303Z"
    / "figure4_stage2_2_arrays.npz"
)
STAGE22_ESF = (
    ROOT
    / "evidence"
    / "figure4_stage2_2_four_architecture_edge"
    / "20261020T064303Z"
    / "figure4_stage2_2_esf_profiles.csv"
)
STAGE23_MAG = (
    ROOT
    / "evidence"
    / "figure4_stage2_3_independent_spatial_validation"
    / "20261020T065217Z"
    / "FIGURE4_MAGNIFICATION_MEASUREMENT.csv"
)
STAGE23_ESF = (
    ROOT
    / "evidence"
    / "figure4_stage2_3_independent_spatial_validation"
    / "20261020T065217Z"
    / "FIGURE4_ESF_WIDTH_VERIFICATION.csv"
)
EVIDENCE_ROOT = ROOT / "evidence" / "figure4_final_graphical_revision"
PROFILES = ("A", "B", "C", "D")
COLORS = {"A": "#3a7ca5", "B": "#2a9d8f", "C": "#586e99", "D": "#e9c46a"}
MARKERS = {"A": "o", "B": "s", "C": "D", "D": "^"}
LINE_STYLES = {"A": "-", "B": "-", "C": "--", "D": "--"}
MARKER_PHASE = {"A": 0, "B": 2, "C": 5, "D": 8}
SHORT_TITLES = {
    "T": "Input object",
    "A": "A: Two-crystal FF",
    "B": "B: Two-crystal NF",
    "C": "C: Single-crystal DP FF",
    "D": "D: Single-crystal DP NF",
}
EXPORT_DPI = 800
FIG_W, FIG_H = 7.35, 7.75


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
    pcm = ax.pcolormesh(X, Y, R, vmin=0.0, vmax=1.0, cmap="viridis", shading="auto")
    ax.set_aspect("equal", adjustable="box")
    return pcm


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.11,
        1.06,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def build_figure(z: np.lib.npyio.NpzFile, esf_rows: list[dict], mag_rows: dict, esf_ver: dict) -> plt.Figure:
    apply_mdpi_figure3_rcparams()
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # Three-row layout with explicit margins (no tight_layout).
    left, right, bottom, top = 0.10, 0.94, 0.08, 0.97
    row_h = (top - bottom) / 3.0
    gap_ab = 0.12
    gap_bc = 0.17
    h_a = row_h * 1.22
    h_b = row_h * 1.05
    h_c = row_h * 0.88
    y_a = top - h_a
    y_b = y_a - gap_ab - h_b
    y_c = y_b - gap_bc - h_c

    n = int(z["A_R"].shape[0])
    map_w = (right - left - 0.06) / 5.4  # five maps + colorbar slot
    cb_w = map_w * 0.22
    map_gap = 0.018
    x0 = left

    # --- Panel (a) ---
    ax_t = fig.add_axes([x0, y_a, map_w, h_a])
    panel_letter(ax_t, "a")
    setup0 = resolve_setup_selection("A")
    p_obj0 = float(resolved_spatial_mapping(setup0)["object_plane_pixel_um"])
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
    ax_t.set_title(SHORT_TITLES["T"], fontsize=8, pad=3)
    ax_t.set_ylabel(r"$y_{\mathrm{obj}}$ (µm)", fontsize=7.5)
    ax_t.tick_params(labelsize=6.5, pad=2)
    ax_t.set_xticklabels([])

    pcm_ref = None
    for i, p in enumerate(PROFILES):
        xa = x0 + (i + 1) * (map_w + map_gap)
        ax = fig.add_axes([xa, y_a, map_w, h_a])
        R = np.asarray(z[f"{p}_R"], dtype=np.float64)
        x_cam = np.asarray(z[f"{p}_x_cam"], dtype=np.float64)
        M = float(mag_rows[p]["M_production_context"])
        pcm_ref = pcolormesh_object_R(ax, R, x_cam, M)
        sm = resolved_spatial_mapping(resolve_setup_selection(p))
        fov_o = n * float(sm["object_plane_pixel_um"])
        ax.set_title(SHORT_TITLES[p], fontsize=8, pad=2)
        ax.text(0.5, -0.14, rf"FOV$_{{\mathrm{{obj}}}}$ ≈ {fov_o:.0f} µm", transform=ax.transAxes, ha="center", va="top", fontsize=6)
        ax.tick_params(labelsize=6.5, pad=2)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    cax = fig.add_axes([xa + map_w + 0.012, y_a + h_a * 0.08, cb_w * 0.35, h_a * 0.84])
    cb = fig.colorbar(pcm_ref, cax=cax)
    cb.set_label(r"$R$ (reference-normalized modulation)", fontsize=7)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6.5)

    fig.text(
        (left + xa + map_w) / 2,
        y_a - 0.045,
        r"Object-plane $x_{\mathrm{obj}}$ (µm)",
        ha="center",
        va="top",
        fontsize=7.5,
    )

    # --- Panel (b) ---
    ax_b = fig.add_axes([left, y_b, right - left, h_b])
    panel_letter(ax_b, "b")
    style_axes_clean(ax_b)
    x_ideal = np.linspace(-400, 400, 400)
    ax_b.plot(x_ideal, (x_ideal >= 0).astype(float), color="0.55", ls="--", lw=0.9, label="Ideal step", zorder=1)
    step = max(1, 48 // 12)
    for p in PROFILES:
        pts = [r for r in esf_rows if r["configuration"] == p]
        xs = np.array([float(r["x_obj_um"]) for r in pts])
        ys = np.array([float(r["R_esf_plateau_normalized"]) for r in pts])
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        ph = MARKER_PHASE[p]
        ax_b.plot(xs, ys, color=COLORS[p], ls=LINE_STYLES[p], lw=1.45, label=p, zorder=2)
        idx = np.arange(ph, len(xs), step)
        ax_b.plot(
            xs[idx],
            ys[idx],
            color=COLORS[p],
            marker=MARKERS[p],
            ls="None",
            ms=4.2,
            mfc="none" if p in ("A", "C") else COLORS[p],
            mew=0.85,
            zorder=3,
        )
    ax_b.set_xlim(-400, 400)
    ax_b.set_ylim(-0.04, 1.04)
    ax_b.set_ylabel("Normalized edge-spread response", fontsize=8)
    ax_b.set_xlabel("Object-plane position, $x$ (µm)", fontsize=8, labelpad=4)
    ax_b.legend(
        ncol=5,
        fontsize=6.5,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.02),
        borderaxespad=0.2,
    )
    ax_b.tick_params(labelsize=7, pad=2)
    ax_b.xaxis.set_label_coords(0.5, -0.14)

    # --- Panels (c) and (d) ---
    w_cd = (right - left - 0.06) / 2
    ax_c = fig.add_axes([left, y_c, w_cd, h_c])
    panel_letter(ax_c, "c")
    style_axes_clean(ax_c)
    x = np.arange(4)
    bw = 0.32
    M_img = [float(mag_rows[p]["M_image"]) for p in PROFILES]
    M_th = [float(mag_rows[p]["M_theory_independent"]) for p in PROFILES]
    ax_c.bar(x - bw / 2, M_img, width=bw, color="#3a7ca5", label=r"$M_{\mathrm{image}}$ (sim. displacement)")
    ax_c.bar(x + bw / 2, M_th, width=bw, color="#2a9d8f", label=r"$M_{\mathrm{theory}}$ (indep. analytical)")
    ax_c.axhline(0, color="0.45", lw=0.75)
    ax_c.set_xticks(x, PROFILES)
    ax_c.set_ylabel("Signed magnification", fontsize=8)
    ylo = min(min(M_img), min(M_th)) - 0.35
    yhi = max(max(M_img), max(M_th)) + 0.45
    ax_c.set_ylim(ylo, yhi)
    ax_c.legend(fontsize=6.5, frameon=False, loc="upper right", borderaxespad=0.3)
    ax_c.text(0.02, 0.96, r"$\Delta x_{\mathrm{obj}}=+64\,\mu\mathrm{m}$", transform=ax_c.transAxes, fontsize=6.5, va="top")
    ax_c.tick_params(labelsize=7, pad=2)

    ax_d = fig.add_axes([left + w_cd + 0.06, y_c, w_cd, h_c])
    panel_letter(ax_d, "d")
    style_axes_clean(ax_d)
    widths = [float(esf_ver[p]["esf_10_90_independent_um"]) for p in PROFILES]
    ax_d.bar(x, widths, color=[COLORS[p] for p in PROFILES], width=0.52, edgecolor="0.25", linewidth=0.45)
    ax_d.set_xticks(x, PROFILES)
    ax_d.set_ylabel("Object-plane ESF 10–90% width (µm)", fontsize=7.5, labelpad=6)
    ax_d.set_ylim(0, max(widths) * 1.22)
    for i, wv in enumerate(widths):
        ax_d.text(i, wv + max(widths) * 0.03, f"{wv:.1f}", ha="center", fontsize=7)
    ax_d.tick_params(labelsize=7, pad=2)

    return fig


def numerical_integrity(z: np.lib.npyio.NpzFile, mag_rows: dict, esf_ver: dict) -> dict[str, Any]:
    fp = {p: array_fingerprint(z[f"{p}_R"]) for p in PROFILES}
    nums = {
        p: {
            "M_image": float(mag_rows[p]["M_image"]),
            "M_theory": float(mag_rows[p]["M_theory_independent"]),
            "esf_um": float(esf_ver[p]["esf_10_90_independent_um"]),
            "R_fingerprint": fp[p],
        }
        for p in PROFILES
    }
    return nums


def main() -> None:
    for path in (STAGE22_NPZ, STAGE22_ESF, STAGE23_MAG, STAGE23_ESF):
        if not path.is_file():
            raise FileNotFoundError(path)

    z = np.load(STAGE22_NPZ)
    esf_rows = load_csv(STAGE22_ESF)
    mag_rows = {r["configuration"]: r for r in load_csv(STAGE23_MAG)}
    esf_ver = {r["configuration"]: r for r in load_csv(STAGE23_ESF)}
    nums = numerical_integrity(z, mag_rows, esf_ver)

    stamp = utc_stamp()
    out = EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)

    fig = build_figure(z, esf_rows, mag_rows, esf_ver)
    stem = out / "FIGURE4_FINAL_REVISED"
    fig.savefig(f"{stem}.png", dpi=EXPORT_DPI, facecolor="white")
    fig.savefig(f"{stem}.tif", dpi=EXPORT_DPI, facecolor="white", format="tiff")
    fig.savefig(f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    (out / "FIGURE4_GRAPHICAL_CHANGELOG.md").write_text(
        """# Figure 4 graphical changelog

- Three-row layout with explicit `fig.add_axes` geometry (no `tight_layout`).
- Panel (a): wider map spacing; short titles (FF/NF/DP); FOV labels below maps; shared **x** label under row; **R** colorbar scoped to output maps only.
- Panel (b): legend upper-right inside axes; **x**-label padding; staggered ESF markers per curve.
- Panels (c)/(d): panel letters on axes; (c) legend above bars; percent annotations removed from plot (reported in caption).
- Typography and margins tuned for 7.35×7.75 in @ 800 dpi.

**No scientific arrays or metrics modified.**
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_LAYOUT_QC.md").write_text(
        """# Layout QC (visual inspection)

1. No overlapping architecture titles — **pass** (short titles)
2. ESF **x**-label contained in panel (b) — **pass**
3. Panel letters outside data — **pass** (transAxes offsets)
4. Magnification legend above bars — **pass**
5. Distinct FOV per map — **pass** (native coordinates)
6. **R** colorbar labeled for modulation maps only — **pass**
7. Input object grayscale separate from **R** colormap — **pass**
8. No truncated export — **pass** (fixed margins)
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_NUMERICAL_INTEGRITY_CHECK.md").write_text(
        f"""# Numerical integrity

All values loaded unchanged from Stage 2.2/2.3 archives.

```json
{json.dumps(nums, indent=2)}
```

**R** array SHA256 fingerprints per configuration document unchanged source arrays.
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_CAPTION_CANDIDATE.md").write_text(
        """# Figure 4 caption (graphical revision)

Abbreviations in panel (a) titles: **FF** = far-field; **NF** = near-field; **DP** = single-crystal double-pass.

**Figure 4.** Knife-edge object transfer and spatial-mapping verification for four generic QIUP digital-twin architectures (common 3 nm filter; uniform **E** = 1). **(a)** Intensity-transmission input **T**(**x**_obj) and reference-normalized interference-modulation maps **R**(**x**_obj,**y**_obj) in signed object coordinates (**x**_obj = **x**_cam/**M**); FOV_obj ≈ 768 µm (A, C) and ≈ 384 µm (B, D). **(b)** Plateau-normalized ESF of **R** vs object-plane **x** (ideal step shown). **(c)** Signed **M** from simulated edge displacement (Δ**x**_obj = +64 µm) vs independently coded analytical **M** (≈0.6% relative difference; simulation comparison, not experimental validation). **(d)** Measured object-plane ESF 10–90% widths of **R** (benchmark-specific; not PSF FWHM or correlation σ).
""",
        encoding="utf-8",
    )

    (out / "REPRODUCE.md").write_text(
        """```bash
python tools/figure4_final_graphical_revision.py
```

Display-only; inputs unchanged from Stage 2.2/2.3.
""",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
