#!/usr/bin/env python3
"""Generate the two-panel, code-grounded Figure 2 architecture overview."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence" / "figure2_digital_twin_architecture"
DETAIL = ROOT / "evidence" / "digital_twin_system_architecture" / "DIGITAL_TWIN_ARCHITECTURE.svg"
STATUS = {
    "implemented": ("#d8eef7", "#1f4e79"),
    "partial": ("#fff0c7", "#a66a00"),
    "disabled": ("#eeeeee", "#777777"),
    "proposed": ("#eadcf3", "#6f3c8f"),
}


def box(ax, xy, wh, title, body, status="implemented"):
    face, edge = STATUS[status]
    p = FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.012,rounding_size=0.012",
                       facecolor=face, edgecolor=edge, linewidth=1.25, zorder=3)
    ax.add_patch(p)
    x, y = xy
    w, h = wh
    ax.text(x+w/2, y+h*0.70, title, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color="#17202a", zorder=4)
    ax.text(x+w/2, y+h*0.31, body, ha="center", va="center", fontsize=6.4,
            color="#263238", linespacing=1.18, zorder=4)


def arrow(ax, src, dst, label="", *, dashed=False, color="#4f5b62"):
    ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=10,
                 linewidth=1.05, color=color, linestyle="--" if dashed else "-",
                 shrinkA=1.5, shrinkB=1.5, zorder=5))
    if label:
        mx, my = (src[0]+dst[0])/2, (src[1]+dst[1])/2
        ax.text(mx, my+0.018, label, ha="center", va="bottom", fontsize=5.8,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=0.7), zorder=5)


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    y, w, h = 0.45, 0.120, 0.29
    xs = [0.025, 0.18, 0.335, 0.49, 0.645, 0.80]
    ax = axes[0]
    ax.text(0.01, 0.98, "(a) Physical digital twin", fontsize=12, fontweight="bold", va="top")
    blocks = [
        ("DT-02/03/04", "Configuration & state\nvalidated inputs", "implemented"),
        ("DT-11", "Crystal, dispersion\nQPM & thermal state", "implemented"),
        ("DT-10/15", "Pump transport, SPDC\nlow/high-gain moments", "implemented"),
        ("DT-12/13", "Object transfer &\npropagation/mapping", "implemented"),
        ("DT-16/17", "Induced coherence &\nsignal image formation", "implemented"),
        ("DT-18/19", "Signal detector &\npredicted observables", "implemented"),
    ]
    for x, (t, b, s) in zip(xs, blocks): box(ax, (x, y), (w, h), t, b, s)
    for i in range(5): arrow(ax, (xs[i]+w, y+h/2), (xs[i+1], y+h/2))
    ax.text(0.257, 0.79, "resolved Δk, sinc² response, L, d_eff", fontsize=6.1, ha="center", color="#1f4e79")
    box(ax, (0.34, 0.08), (0.13, 0.20), "Optional optics", "PSF/aberration toggles\n(vignetting disabled)", "disabled")
    arrow(ax, (0.405, 0.28), (0.555, y), "optional image effects", dashed=True)
    box(ax, (0.55, 0.08), (0.15, 0.20), "Not implemented", "full 4-D biphoton /\ndirect idler detection", "proposed")

    ax = axes[1]
    ax.text(0.01, 0.98, "(b) Laboratory evaluation and evidence", fontsize=12, fontweight="bold", va="top")
    blocks2 = [
        ("LAB-01", "TIFF acquisition\nexternal records", "implemented"),
        ("LAB-02", "Camera correction\nraw mode; bias optional", "partial"),
        ("LAB-03", "Calibration & parameter\nidentification", "partial"),
        ("LAB-04", "Frozen parameters &\nindependent evaluation", "partial"),
        ("LAB-05", "Residuals, SEM &\nuncertainty evidence", "partial"),
        ("LAB-06", "Manual review and\nrelease decision", "partial"),
    ]
    for x, (t, b, s) in zip(xs, blocks2): box(ax, (x, y), (w, h), t, b, s)
    for i in range(5): arrow(ax, (xs[i]+w, y+h/2), (xs[i+1], y+h/2))
    box(ax, (0.25, 0.08), (0.19, 0.20), "DT-41", "Independent equations,\nregression & convergence", "implemented")
    box(ax, (0.57, 0.08), (0.19, 0.20), "Held-out validation", "Proposed where no independent\nmatched acquisition exists", "proposed")
    arrow(ax, (0.345, 0.28), (0.565, y), "verification evidence", dashed=True, color="#a66a00")
    arrow(ax, (0.665, 0.28), (0.565, y), "future independent data", dashed=True, color="#6f3c8f")
    ax.text(0.50, 0.015, "No automated feedback from laboratory data into the production physics model",
            fontsize=6.8, ha="center", color="#6f3c8f")
    handles = [Patch(facecolor=f, edgecolor=e, label=k.capitalize()) for k, (f, e) in STATUS.items()]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.012), frameon=False, fontsize=8)
    fig.suptitle("Figure 2. QIUP digital twin and laboratory-evaluation workflow", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.07, hspace=0.22)
    stem = out / "FIGURE2_DIGITAL_TWIN_ARCHITECTURE"
    for ext, kw in (("png", {"dpi": 400}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix("."+ext), facecolor="white", **kw)
    plt.close(fig)
    if DETAIL.is_file(): shutil.copy2(DETAIL, out / "FIGURE2_DETAILED_ARCHITECTURE_SUPPLEMENT.svg")
    prov = {"qpm_to_spdc_dependency": "DT-11 -> DT-10/15", "automated_experimental_feedback": False,
            "global_experimental_validation_claim": False,
            "status_palette": {k: {"fill": v[0], "edge": v[1]} for k, v in STATUS.items()},
            "detailed_architecture_source": str(DETAIL.relative_to(ROOT))}
    (out / "FIGURE2_STATUS_AND_PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (out / "FIGURE2_CAPTION_FINAL.md").write_text("""# Figure 2 caption

**Figure 2. Implemented QIUP digital twin and laboratory-evaluation workflow.**
**(a)** Configuration and validation resolve the crystal, dispersion, thermal,
and QPM state; these quantities feed the pump/SPDC operating point and moments
(DT-11 → DT-10/15), followed by object transfer, propagation, induced coherence,
signal-only detection, and predicted observables. Optional, disabled, and
not-implemented components are explicit. **(b)** External TIFF acquisition,
camera handling, calibration, frozen-parameter evaluation, residual/uncertainty
analysis, and release decisions are separated by status. The workflow does not
claim completed end-to-end validation or automated experimental feedback into
production physics. The exhaustive architecture is retained as a supplement.
""", encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    print(build(p.parse_args().output.resolve()))


if __name__ == "__main__": main()
