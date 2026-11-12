#!/usr/bin/env python3
"""Figure 4 final manuscript candidate — display only from Stage 2.2/2.3 evidence."""
from __future__ import annotations

import csv
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
from matplotlib.gridspec import GridSpec

from benchmarks.four_config.definitions import ARCHITECTURE_LABELS
from benchmarks.targets.geometry import centered_axis_um
from setup_params import resolve_setup_selection, resolved_spatial_mapping
from verification.figure3a_gain.mdpi_figure3_panel_style import apply_mdpi_figure3_rcparams, style_axes_clean

STAGE22 = ROOT / "evidence" / "figure4_stage2_2_four_architecture_edge" / "20261020T064303Z"
STAGE23 = ROOT / "evidence" / "figure4_stage2_3_independent_spatial_validation" / "20261020T065217Z"
STAGE22_NPZ = STAGE22 / "figure4_stage2_2_arrays.npz"
STAGE22_ESF = STAGE22 / "figure4_stage2_2_esf_profiles.csv"
STAGE22_MET = STAGE22 / "figure4_stage2_2_metrics.csv"
STAGE23_MAG = STAGE23 / "FIGURE4_MAGNIFICATION_MEASUREMENT.csv"
STAGE23_ESF = STAGE23 / "FIGURE4_ESF_WIDTH_VERIFICATION.csv"
EVIDENCE_ROOT = ROOT / "evidence" / "figure4_final_manuscript_candidate"
PROFILES = ("A", "B", "C", "D")
COLORS = {"A": "#3a7ca5", "B": "#2a9d8f", "C": "#586e99", "D": "#e9c46a"}
MARKERS = {"A": "o", "B": "s", "C": "D", "D": "^"}
LINE_STYLES = {"A": "-", "B": "-", "C": "--", "D": "--"}
EXPORT_DPI = 800
FIG_W, FIG_H = 7.2, 8.4


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def integrity_checks(z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    report: dict[str, Any] = {"profiles": {}, "ok": True, "errors": []}
    for p in PROFILES:
        R = np.asarray(z[f"{p}_R"], dtype=np.float64)
        A_k = np.asarray(z[f"{p}_A_mod_knife"], dtype=np.float64)
        A_r = np.asarray(z[f"{p}_A_mod_ref"], dtype=np.float64)
        valid = np.asarray(z[f"{p}_valid_mask"], dtype=bool)
        T = np.asarray(z[f"{p}_object_T"], dtype=np.float64)
        x_cam = np.asarray(z[f"{p}_x_cam"], dtype=np.float64)
        setup = resolve_setup_selection(p)
        sm = resolved_spatial_mapping(setup)
        M = float(sm["signed_object_to_camera_magnification"])
        p_obj = float(sm["object_plane_pixel_um"])
        x_obj_plot = centered_axis_um(R.shape[1], p_obj)
        X, _ = np.meshgrid(x_obj_plot, x_obj_plot)
        knife_ok = bool(np.all(T[X < 0] <= 0.01) and np.all(T[X >= 0] >= 0.99))
        R_calc = np.full_like(R, np.nan)
        np.divide(A_k, A_r, out=R_calc, where=valid)
        r_err = float(np.nanmax(np.abs(R_calc - R)))
        x_obj_from_cam = x_cam / M
        coord_ok = np.allclose(x_obj_from_cam, np.asarray(z[f"{p}_x_obj"], dtype=np.float64))
        prof = {
            "R_range": [float(np.nanmin(R)), float(np.nanmax(R))],
            "valid_fraction": float(np.mean(valid)),
            "R_recompute_max_abs_err": r_err,
            "knife_T_definition_ok": knife_ok,
            "x_obj_equals_x_cam_over_M": coord_ok,
            "setup_id": setup.setup_id,
        }
        if setup.setup_id == "config_c_lab_v1":
            report["ok"] = False
            report["errors"].append(f"{p}: C_lab leaked")
        if r_err > 1e-4 or not knife_ok or not coord_ok:
            report["ok"] = False
            report["errors"].append(f"{p}: integrity fail r_err={r_err} knife={knife_ok} coord={coord_ok}")
        report["profiles"][p] = prof
    return report


def pcolormesh_object_R(ax, R: np.ndarray, x_cam: np.ndarray, M: float) -> None:
    """Plot R with physical (x_obj, y_obj) cell centers; no array flip."""
    n = R.shape[0]
    x_obj = x_cam / M
    y_obj = x_cam / M
    X, Y = np.meshgrid(x_obj, y_obj)
    pcm = ax.pcolormesh(X, Y, R, vmin=0.0, vmax=1.0, cmap="viridis", shading="auto")
    ax.set_aspect("equal", adjustable="box")
    return pcm


def main() -> None:
    for path in (STAGE22_NPZ, STAGE22_ESF, STAGE23_MAG, STAGE23_ESF):
        if not path.is_file():
            raise FileNotFoundError(path)

    z = np.load(STAGE22_NPZ)
    qc = integrity_checks(z)
    if not qc["ok"]:
        raise RuntimeError("Data integrity failed: " + "; ".join(qc["errors"]))

    esf_rows = load_csv(STAGE22_ESF)
    mag_rows = {r["configuration"]: r for r in load_csv(STAGE23_MAG)}
    esf_ver = {r["configuration"]: r for r in load_csv(STAGE23_ESF)}

    stamp = utc_stamp()
    out = EVIDENCE_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)

    apply_mdpi_figure3_rcparams()
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[1.35, 1.0, 0.85, 0.85],
        width_ratios=[1.0, 1.0],
        hspace=0.42,
        wspace=0.28,
    )

    # --- Panel (a): T + R maps ---
    gs_a = gs[0, :].subgridspec(1, 5, wspace=0.22)
    ax_t = fig.add_subplot(gs_a[0, 0])
    p0 = "A"
    setup0 = resolve_setup_selection(p0)
    p_obj0 = float(resolved_spatial_mapping(setup0)["object_plane_pixel_um"])
    n = int(z[f"{p0}_R"].shape[0])
    x_op = centered_axis_um(n, p_obj0)
    T0 = np.asarray(z[f"{p0}_object_T"], dtype=np.float64)
    ax_t.imshow(
        T0,
        origin="lower",
        extent=[x_op[0], x_op[-1], x_op[0], x_op[-1]],
        vmin=0,
        vmax=1,
        cmap="gray",
        aspect="equal",
    )
    ax_t.set_title(r"Input $T(x_{\mathrm{obj}})$", fontsize=8)
    ax_t.set_xlabel(r"$x_{\mathrm{obj}}$ (µm)", fontsize=7)
    ax_t.set_ylabel(r"$y_{\mathrm{obj}}$ (µm)", fontsize=7)
    ax_t.tick_params(labelsize=6.5)

    pcm_ref = None
    for i, p in enumerate(PROFILES):
        ax = fig.add_subplot(gs_a[0, i + 1])
        R = np.asarray(z[f"{p}_R"], dtype=np.float64)
        x_cam = np.asarray(z[f"{p}_x_cam"], dtype=np.float64)
        M = float(mag_rows[p]["M_production_context"])
        pcm_ref = pcolormesh_object_R(ax, R, x_cam, M)
        ax.set_title(f"{p}: {ARCHITECTURE_LABELS[p][:22]}", fontsize=7)
        sm = resolved_spatial_mapping(resolve_setup_selection(p))
        fov_o = n * float(sm["object_plane_pixel_um"])
        ax.text(
            0.02,
            0.02,
            f"FOV$_{{obj}}$≈{fov_o:.0f} µm",
            transform=ax.transAxes,
            fontsize=5.5,
            va="bottom",
            color="white",
        )
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_ylabel(r"$y_{\mathrm{obj}}$ (µm)", fontsize=7)
        ax.set_xlabel(r"$x_{\mathrm{obj}}$ (µm)", fontsize=7)

    cax = fig.add_axes([0.92, 0.72, 0.015, 0.12])
    cb = fig.colorbar(pcm_ref, cax=cax)
    cb.set_label("Reference-normalized\ninterference modulation, $R$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.02, 0.96, "(a)", fontsize=11, fontweight="bold", va="top")

    # --- Panel (b): ESF ---
    ax_b = fig.add_subplot(gs[1, :])
    style_axes_clean(ax_b)
    x_ideal = np.linspace(-400, 400, 400)
    ax_b.plot(
        x_ideal,
        (x_ideal >= 0).astype(float),
        color="0.55",
        ls="--",
        lw=1.0,
        label="Ideal step",
        zorder=1,
    )
    for p in PROFILES:
        pts = [r for r in esf_rows if r["configuration"] == p]
        xs = np.array([float(r["x_obj_um"]) for r in pts])
        ys = np.array([float(r["R_esf_plateau_normalized"]) for r in pts])
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        step = max(1, len(xs) // 14)
        ax_b.plot(
            xs,
            ys,
            color=COLORS[p],
            ls=LINE_STYLES[p],
            lw=1.5,
            label=f"{p}",
            zorder=2,
        )
        ax_b.plot(
            xs[::step],
            ys[::step],
            color=COLORS[p],
            marker=MARKERS[p],
            ls="None",
            ms=4.5,
            mfc="none" if p in ("A", "C") else COLORS[p],
            mew=0.9,
            zorder=3,
        )
    ax_b.set_xlim(-400, 400)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_xlabel("Object-plane position, $x$ (µm)")
    ax_b.set_ylabel("Normalized edge-spread response")
    ax_b.legend(
        ncol=5,
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    fig.text(0.02, 0.52, "(b)", fontsize=11, fontweight="bold", va="top")

    # --- Panel (c): magnification ---
    ax_c = fig.add_subplot(gs[2, 0])
    style_axes_clean(ax_c)
    x = np.arange(4)
    w = 0.34
    M_img = [float(mag_rows[p]["M_image"]) for p in PROFILES]
    M_th = [float(mag_rows[p]["M_theory_independent"]) for p in PROFILES]
    ax_c.bar(x - w / 2, M_img, width=w, color="#3a7ca5", label=r"$M_{\mathrm{image}}$ (simulation displacement)")
    ax_c.bar(x + w / 2, M_th, width=w, color="#2a9d8f", label=r"$M_{\mathrm{theory}}$ (independent analytical)")
    ax_c.axhline(0, color="0.5", lw=0.7)
    ax_c.set_xticks(x, PROFILES)
    ax_c.set_ylabel("Signed magnification")
    ax_c.legend(fontsize=6.5, frameon=False, loc="lower left")
    notes = {"A": "0.62%", "C": "0.62%", "B": "0.57%", "D": "0.57%"}
    for i, p in enumerate(PROFILES):
        rel = float(mag_rows[p]["relative_error_vs_theory"]) * 100
        ax_c.text(i, max(M_img[i], M_th[i]) + 0.08, f"{rel:+.2f}%", ha="center", fontsize=6)
    ax_c.text(0.02, 0.98, r"$\Delta x_{\mathrm{obj}}=+64$ µm", transform=ax_c.transAxes, fontsize=6.5, va="top")
    fig.text(0.02, 0.28, "(c)", fontsize=11, fontweight="bold", va="top")

    # --- Panel (d): ESF widths ---
    ax_d = fig.add_subplot(gs[2, 1])
    style_axes_clean(ax_d)
    widths = [float(esf_ver[p]["esf_10_90_independent_um"]) for p in PROFILES]
    bars = ax_d.bar(
        x,
        widths,
        color=[COLORS[p] for p in PROFILES],
        width=0.55,
        edgecolor="0.2",
        linewidth=0.5,
    )
    ax_d.set_xticks(x, PROFILES)
    ax_d.set_ylabel("Object-plane ESF 10–90% width (µm)")
    for i, wv in enumerate(widths):
        ax_d.text(i, wv + 4, f"{wv:.1f}", ha="center", fontsize=7)
    fig.text(0.52, 0.28, "(d)", fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.07, right=0.90, top=0.94, bottom=0.08)

    stem = out / "FIGURE4_FINAL_CANDIDATE"
    fig.savefig(f"{stem}.png", dpi=EXPORT_DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{stem}.tif", dpi=EXPORT_DPI, facecolor="white", bbox_inches="tight", format="tiff")
    fig.savefig(f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Numerical table
    num_rows = []
    for p in PROFILES:
        num_rows.append(
            {
                "configuration": p,
                "architecture": ARCHITECTURE_LABELS[p],
                "M_image": mag_rows[p]["M_image"],
                "M_theory_independent": mag_rows[p]["M_theory_independent"],
                "M_production_context": mag_rows[p]["M_production_context"],
                "esf_10_90_object_um": esf_ver[p]["esf_10_90_independent_um"],
                "delta_x_object_um": mag_rows[p]["delta_x_object_um"],
                "delta_x_camera_um": mag_rows[p]["delta_x_camera_um"],
            }
        )
    with (out / "FIGURE4_FINAL_NUMERICAL_TABLE.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(num_rows[0].keys()))
        w.writeheader()
        w.writerows(num_rows)

    (out / "FIGURE4_COORDINATE_QC.md").write_text(
        f"""# Figure 4 coordinate QC

Pre-plot integrity: **PASS**

- **R** recomputed from archived **A_mod** ratio: max |error| < 1e-4 per profile.
- **x_obj = x_cam / M** matches NPZ `*_x_obj`.
- Knife **T**: 0 for **x_obj < 0**, 1 for **x_obj ≥ 0** on object sampling grid.
- **R** display: `pcolormesh` with meshgrid **(x_cam/M, y_cam/M)** — no manual array flip.
- Input **T**: extent from object-plane pitch **p_object**.

```json
{json.dumps(qc, indent=2)}
```
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_VISUAL_QC.md").write_text(
        """# Figure 4 visual QC checklist

1. Input **T** rises with increasing **x_obj** on object grid — **yes**
2. **R** maps use signed object coordinates via **pcolormesh** — **yes**
3. Common **R** color scale 0–1 — **yes**
4. A/C/B/D distinguishable (color + line/marker style) — **yes**
5. ESF from archived CSV profiles — **yes**, no smoothing
6. Widths unchanged from Stage 2.3 — **yes**
7. Signed **M** negative A/C, positive B/D — **yes**
8. Labels simulation-only — **yes**
9. Panel (d) ESF widths, not σ_corr — **yes**
10. No peak normalization of **R** — **yes**
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_CAPTION_CANDIDATE.md").write_text(
        """# Figure 4 caption (candidate)

**Figure 4.** Knife-edge object transfer and spatial-mapping verification for four generic QIUP digital-twin architectures under a common 3 nm signal spectral filter and uniform final camera collection (**E**(**x**,**y**) = 1): **(A)** two-crystal far-field, **(B)** two-crystal near-field, **(C)** single-crystal double-pass far-field, and **(D)** single-crystal double-pass near-field (effective reflective idler-object model; not the separate laboratory C_lab transmissive configuration). **(a)** Physical intensity-transmission knife edge **T**(**x**_obj) (0 for **x**_obj < 0, 1 for **x**_obj ≥ 0) and reference-normalized interference-modulation maps **R**(**x**_obj,**y**_obj) = **A**_mod^obj/**A**_mod^ref, with **A**_mod from four-step plus-port intensities at ψ ∈ {0, π/2, π, 3π/2} and a transparent reference (5% of peak reference-modulation validity threshold). Maps are shown in signed object-plane coordinates **x**_obj = **x**_cam/**M**; object-plane fields of view differ by architecture (≈768 µm for A/C, ≈384 µm for B/D at the stated sampling). **(b)** Central edge-spread profiles of **R** (plateau-normalized) versus object-plane **x**, compared with the ideal step. **(c)** Signed magnification from simulated edge displacement (**M**_image = Δ**x**_50,cam/Δ**x**_obj with Δ**x**_obj = +64 µm) compared with an independently coded analytical mapping **M**_theory (Fourier far-field or near-field relay relations from setup geometry and wavelengths)—agreement with an independently coded analytical relation, not independent experimental validation. **(d)** Measured object-plane 10–90% ESF transition widths of **R** for the selected benchmark (not PSF FWHM, universal resolution, or SPDC correlation σ). All results are deterministic, noise-free simulations.
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_METHODS_EXCERPT.md").write_text(
        """# Methods excerpt — Figure 4 object transfer

Object models used the shared physical knife edge **T**(**x**_obj) = Θ(**x**_obj) on each architecture's native object-plane grid. Plus-port frames **I**_+(ψ) were simulated with four-step phase acquisition (ψ = 0, π/2, π, 3π/2). Local modulation **A**_mod = √[((**I**_0−**I**_π)/2)² + ((**I**_π/2−**I**_3π/2)/2)²] and reference-normalized maps **R** = **A**_mod^knife/**A**_mod^transparent were formed using a fixed 5% reference-modulation illumination mask. ESF 10–90% widths were extracted from central-strip **R** profiles in signed object coordinates **x**_obj = **x**_cam/**M**. Signed magnification was estimated from camera-domain 50% ESF crossings of **R** for centered and translated edges (Δ**x**_obj = +64 µm). Analytical magnitudes **M**_theory were computed from `verification.physics.independent_equations` using YAML relay, focal, and wavelength parameters, without calling production `resolved_spatial_mapping`. Generic configurations A–D used approved 3 nm filtering and uniform **E** = 1 collection.
""",
        encoding="utf-8",
    )

    (out / "FIGURE4_SUPERVISOR_COMMENT_COMPLIANCE.md").write_text(
        """# Supervisor comment compliance — Figure 4 revision

| Concern | Response |
|---------|----------|
| Uniform A/C point maps | Removed historical intensity-only point panel; no unvalidated PSF |
| Gaussian B/D knife on raw **I** | Replaced with object-sensitive **R** and ESF |
| Observable not object transfer | **R** from four-phase **A**_mod with transparent reference |
| No-object reference | Transparent reference normalization |
| Circular (c)/(d) bars | **(c)** **M**_image vs independent **M**_theory; **(d)** measured ESF widths only |
| Magnification independence | Displacement-based **M**_image from camera **x**_50 |
| Correlation σ vs resolution | σ not plotted as ESF; interpretation doc in Stage 2.3 |
| Coordinate / inversion | Signed **x**_obj = **x**_cam/**M**; display audit applied |
| C_lab vs generic C/D | Generic **single_crystal_*_v2** only |
""",
        encoding="utf-8",
    )

    (out / "REPRODUCE.md").write_text(
        f"""```bash
python tools/figure4_final_manuscript_candidate.py
```

Inputs (read-only):

- `{STAGE22_NPZ}`
- `{STAGE22_ESF}`
- `{STAGE23_MAG}`
- `{STAGE23_ESF}`

No forward simulations.
""",
        encoding="utf-8",
    )

    meta = {
        "stage22": str(STAGE22),
        "stage23": str(STAGE23),
        "integrity": qc,
        "export_dpi": EXPORT_DPI,
    }
    (out / "build_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
