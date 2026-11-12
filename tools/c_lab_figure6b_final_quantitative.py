#!/usr/bin/env python3
"""Figure 6(b) final: corrected effective-defocus images + rho^2 vs bright-ring order."""
from __future__ import annotations

import argparse
import csv
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
from matplotlib.gridspec import GridSpec

from camera.c_lab_display_zoom import export_zoom_config_from_setup, resolve_fixed_crop_window
from setup_params import resolve_setup_selection
from tools.c_lab_figure6_lens_displacement_validation import (
    EXP_ROOT,
    N,
    Z0_MM,
    group_tiffs,
    mean_stack,
)
from tools.c_lab_figure6b_corrected_effective_defocus import (
    Z_EFF_MM,
    Z_STAGE_MM,
    array_sha256,
    max_radius_mm,
    normalized_profile,
    theory_effective_double_pass_display,
    theory_high_ring_release_display,
)
from tools.c_lab_figure6b_ring_extraction import (
    PIXEL_MM,
    detect_rings,
    radial_profile,
    reference_center,
)
from tools.c_lab_isolated_pilot import DeltaKCalibrationState, run_forward_case
from tools.z2.physics import ref_from_T

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_figure6b_final_quantitative"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_panels(
    *, high_ring_release: bool = False
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[str, int], float, float]:
    setup = resolve_setup_selection("C_lab")
    dk = DeltaKCalibrationState(mode="theory")
    ref = ref_from_T(np.ones((N, N), dtype=np.float32))
    I_ref = run_forward_case(setup.overridden(enable_lens_rings=True), ref, dk, grid_n=N)["I_plus"]
    _, bounds = resolve_fixed_crop_window(np.asarray(I_ref, dtype=np.float64), export_zoom_config_from_setup(setup))

    renderer = (
        theory_high_ring_release_display
        if high_ring_release
        else theory_effective_double_pass_display
    )
    theory: dict[float, np.ndarray] = {}
    for z_stage in Z_STAGE_MM:
        theory[z_stage] = renderer(
            setup, dk, bounds, dz_um=z_stage * 1000.0
        )

    groups = group_tiffs(EXP_ROOT / "L drift", ref_mm=Z0_MM, rel_scale=1.0)
    exp_imgs = {z: mean_stack(groups[round(z, 6)])["mean"] for z in Z_STAGE_MM}

    t_v = max(np.percentile(theory[z], 99.5) for z in Z_STAGE_MM)
    e_v = max(np.percentile(exp_imgs[z], 99.5) for z in Z_STAGE_MM)
    return theory, exp_imgs, bounds, t_v, e_v


def extract_bright_table(
    img: np.ndarray,
    center,
    *,
    source: str,
    z_stage: float,
    z_eff: float,
    r_lim_mm: float,
    max_rings: int | None = 8,
) -> list[dict[str, Any]]:
    ana = detect_rings(img, center, pixel_mm=PIXEL_MM, use_sector_fallback=False)
    rows: list[dict[str, Any]] = []
    for ext in sorted(ana.bright, key=lambda e: e.fringe_order):
        if ext.radius_mm > r_lim_mm:
            continue
        if max_rings is not None and ext.fringe_order > max_rings:
            break
        rows.append(
            {
                "source": source,
                "dz_stage_mm": z_stage,
                "dz_eff_mm": z_eff,
                "ring_type": "bright",
                "ring_order": ext.fringe_order,
                "radius_px": round(ext.index_px, 3),
                "radius_mm": round(ext.radius_mm, 4),
                "radius_sq_mm2": round(ext.radius_sq_mm2, 6),
                "center_row": round(center.cy, 3),
                "center_col": round(center.cx, 3),
                "prominence": round(ext.prominence, 6),
                "detection_quality": ext.quality,
                "center_method": center.method,
            }
        )
    return rows


def run(output: Path | None = None) -> Path:
    out = output or EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    theory, exp_imgs, bounds, t_v, e_v = load_panels()
    th_center = reference_center(theory[0.0], modality="theory")
    ex_center = reference_center(exp_imgs[0.0], modality="experiment")

    simulation_sha256 = {
        f"z_stage_{z}": array_sha256(theory[z]) for z in Z_STAGE_MM
    }

    table: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    analyses: dict[tuple[str, float], Any] = {}

    for z_stage, z_eff in zip(Z_STAGE_MM, Z_EFF_MM):
        if z_stage == 0:
            continue
        rl_th = max_radius_mm(theory[z_stage], th_center.cy, th_center.cx)
        rl_ex = max_radius_mm(exp_imgs[z_stage], ex_center.cy, ex_center.cx)
        # Experiment: keep fewer rings when annular contrast is weak.
        th_rows = extract_bright_table(
            theory[z_stage], th_center, source="simulation", z_stage=z_stage, z_eff=z_eff, r_lim_mm=rl_th
        )
        ex_rows = extract_bright_table(
            exp_imgs[z_stage],
            ex_center,
            source="experimental",
            z_stage=z_stage,
            z_eff=z_eff,
            r_lim_mm=rl_ex,
            max_rings=4,
        )
        if ex_rows:
            pmax = max(r["prominence"] for r in ex_rows)
            ex_rows = [r for r in ex_rows if r["prominence"] >= 0.15 * pmax]
            for k, r in enumerate(ex_rows, start=1):
                r["ring_order"] = k
        table.extend(th_rows)
        table.extend(ex_rows)
        counts[f"simulation_z_eff_{z_eff}"] = len(th_rows)
        counts[f"experimental_z_eff_{z_eff}"] = len(ex_rows)
        analyses[("simulation", z_eff)] = detect_rings(
            theory[z_stage], th_center, pixel_mm=PIXEL_MM, use_sector_fallback=False
        )
        analyses[("experimental", z_eff)] = detect_rings(
            exp_imgs[z_stage], ex_center, pixel_mm=PIXEL_MM, use_sector_fallback=False
        )

    with (out / "RING_RADII_TABLE.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(table[0].keys()) if table else ["note"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    # Diagnostic: radial profiles (not in publication figure).
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (z_stage, z_eff) in zip(axes.ravel(), zip(Z_STAGE_MM, Z_EFF_MM)):
        r_th, raw_th, _ = radial_profile(theory[z_stage], th_center, pixel_mm=PIXEL_MM)
        r_ex, raw_ex, _ = radial_profile(exp_imgs[z_stage], ex_center, pixel_mm=PIXEL_MM)
        rl_th = max_radius_mm(theory[z_stage], th_center.cy, th_center.cx)
        rl_ex = max_radius_mm(exp_imgs[z_stage], ex_center.cy, ex_center.cx)
        m_th = np.isfinite(raw_th) & (r_th <= rl_th)
        m_ex = np.isfinite(raw_ex) & (r_ex <= rl_ex)
        ax.plot(r_th[m_th], normalized_profile(raw_th)[m_th], "-", label="Simulation", linewidth=1.1)
        ax.plot(r_ex[m_ex], normalized_profile(raw_ex)[m_ex], "-", label="Experimental", linewidth=1.0, alpha=0.85)
        if z_eff in [e for s, e in zip(Z_STAGE_MM, Z_EFF_MM) if s > 0]:
            for ext in analyses[("simulation", z_eff)].bright:
                if ext.radius_mm <= rl_th:
                    ax.axvline(ext.radius_mm, color="C0", alpha=0.25, linestyle="--")
        ax.set_title(rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm")
        ax.set_xlabel("ρ (mm)")
        ax.set_ylabel("Normalized intensity")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 2.8)
    fig.suptitle("Diagnostic: radial profiles and simulated bright-ring radii", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "RADIAL_PROFILES_DIAGNOSTIC.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (z_stage, z_eff) in zip(axes, [(s, e) for s, e in zip(Z_STAGE_MM, Z_EFF_MM) if s > 0]):
        ana_th = analyses[("simulation", z_eff)]
        ana_ex = analyses[("experimental", z_eff)]
        ax.plot(ana_th.r_mm, ana_th.intensity_smooth, "-", color="C0", label="Simulation smooth")
        ax.plot(ana_ex.r_mm, ana_ex.intensity_smooth, "-", color="C3", label="Experimental smooth", alpha=0.8)
        for ext in ana_th.bright:
            ax.axvline(ext.radius_mm, color="C0", linestyle="--", alpha=0.5)
        for ext in ana_ex.bright[:4]:
            ax.axvline(ext.radius_mm, color="C3", linestyle=":", alpha=0.5)
        ax.set_title(rf"$\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm")
        ax.set_xlabel("ρ (mm)")
        ax.set_ylabel("Annular mean")
    fig.tight_layout()
    fig.savefig(out / "RING_EXTRACTION_DIAGNOSTICS.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Publication figure
    fig = plt.figure(figsize=(12, 8.5))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 0.95], hspace=0.35, wspace=0.12)

    for j, (z_stage, z_eff) in enumerate(zip(Z_STAGE_MM, Z_EFF_MM)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(theory[z_stage], cmap="gray", origin="upper", vmin=0, vmax=t_v)
        ax.set_title(rf"Simulation — $\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(exp_imgs[z_stage], cmap="gray", origin="upper", vmin=0, vmax=e_v)
        ax.set_title(rf"Experimental — $\Delta z_{{\mathrm{{eff}}}}$ = {z_eff:.3f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    axp = fig.add_subplot(gs[2, :])
    styles = {
        (4.8, "simulation"): ("o-", "Simulation 4.800 mm", "C0"),
        (7.16, "simulation"): ("s-", "Simulation 7.160 mm", "C1"),
        (11.624, "simulation"): ("^-", "Simulation 11.624 mm", "C2"),
        (4.8, "experimental"): ("o--", "Experimental 4.800 mm", "C3"),
        (7.16, "experimental"): ("s--", "Experimental 7.160 mm", "C4"),
        (11.624, "experimental"): ("^--", "Experimental 11.624 mm", "C5"),
    }
    for z_eff in (4.8, 7.16, 11.624):
        for src in ("simulation", "experimental"):
            pts = sorted(
                [r for r in table if r["source"] == src and float(r["dz_eff_mm"]) == z_eff],
                key=lambda r: r["ring_order"],
            )
            if not pts:
                continue
            sty, lab, col = styles[(z_eff, src)]
            axp.plot(
                [p["ring_order"] for p in pts],
                [p["radius_sq_mm2"] for p in pts],
                sty,
                color=col,
                label=lab,
                markersize=7,
                linewidth=1.2,
            )

    axp.set_xlabel("Bright-ring fringe order")
    axp.set_ylabel(r"Squared ring radius $\rho^2$ (mm$^2$)")
    axp.set_xlim(0.5, None)
    axp.set_ylim(bottom=0)
    axp.legend(fontsize=7, ncol=3, loc="upper left")
    axp.grid(True, alpha=0.25)
    fig.suptitle(
        r"Figure 6(b). Axial idler-lens defocus: simulation and experimental results, $\Delta z_{\mathrm{eff}}$",
        fontsize=12,
    )
    fig.savefig(out / "FIGURE6B_FINAL_QUANTITATIVE.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "FIGURE6B_FINAL_QUANTITATIVE.pdf", bbox_inches="tight")
    plt.close(fig)

    exp_limit = (
        "Experimental bright-ring counts are limited (typically one robust peak per Δz_eff) "
        "because annular averaging suppresses contrast on the native EMCCD field with diagonal "
        "background structure; the simulation on the 419×455 crop resolves multiple bright orders at "
        "4.800 and 7.160 mm effective defocus."
    )

    (out / "FIGURE6B_CAPTION.md").write_text(
        """# Figure 6(b) caption

**Figure 6(b).** Axial idler-lens defocus: simulation and experimental results (no object). Top row: simulated signal `I_+` at effective defocus Δz_eff = 0, 4.800, 7.160, and 11.624 mm. The forward model receives the mechanical stage displacement Δz_stage = 0, 2.400, 3.580, and 5.812 mm once; its production Michelson `pass_factor = 2` supplies the double-pass phase. Second row: mean experimental frames at the corresponding Δz_eff labels. Lower panel: squared bright-ring radius ρ² versus fringe order from annular radial profiles (13 µm/px); separate intensity centroids at Δz_eff = 0 for simulation and experiment. Experimental ring orders are sparse where fringes are noncircular.
""",
        encoding="utf-8",
    )

    (out / "FINAL_DECISION.md").write_text(
        f"""# Figure 6(b) final quantitative

- **Evidence:** `{out}`
- **Simulation images:** production `I_plus` via `theory_effective_double_pass_display()`; mechanical Δz_stage supplied once and production pass factor 2 applied once.
- **Simulation array SHA256:** {simulation_sha256}
- **Bottom panel:** ρ² vs bright-ring order (not normalized radial profiles).
- **Bright-ring counts:** {json.dumps(counts, indent=2)}
- **Experimental limitation:** {exp_limit}
- **Double counting:** prevented; `d(Δz_stage)` is not pre-doubled.
- **PRODUCTION_PHYSICS_MODIFIED:** NO
""",
        encoding="utf-8",
    )

    print(f"Done: {out}")
    print(json.dumps(counts, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
