"""Shared absolute-scale plotting helpers for the four-config paper figure."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.four_config.definitions import ARCHITECTURE_LABELS, PLOT_POLICY, PROFILES


def extent_from_axes(x_um: np.ndarray, y_um: np.ndarray) -> list[float]:
    """imshow extent [left, right, bottom, top] in µm (pixel-edge convention)."""
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(y_um, dtype=np.float64)
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    return [
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    ]


def shared_abs_limits(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    """Global vmin/vmax across configs (absolute; no per-image peak=1)."""
    vals = []
    for a in arrays:
        flat = np.asarray(a, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            vals.append(flat)
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    return float(np.min(allv)), float(np.max(allv))


def save_image_panel(
    path: Path,
    image: np.ndarray,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(
        np.asarray(image, dtype=np.float64),
        origin=PLOT_POLICY["origin"],
        extent=extent_from_axes(x_um, y_um),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("Idler/object-plane coordinate, $x_i$ ($\\mathrm{\\mu m}$)")
    ax.set_ylabel("Idler/object-plane coordinate, $y_i$ ($\\mathrm{\\mu m}$)")
    ax.set_title(title, fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    if cbar_label:
        cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_2x2(
    path: Path,
    images: dict[str, np.ndarray],
    *,
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    cmap: str,
    cbar_label: str,
    shared_scale: bool = True,
) -> dict[str, float]:
    """2×2 A/B/C/D panel with optional shared absolute color scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [p for p in PROFILES if p in images]
    arrs = [images[p] for p in order]
    if shared_scale:
        vmin, vmax = shared_abs_limits(arrs)
    else:
        vmin, vmax = None, None

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    for ax, profile in zip(axes.ravel(), order):
        x_um, y_um = coordinates[profile]
        im = ax.imshow(
            np.asarray(images[profile], dtype=np.float64),
            origin=PLOT_POLICY["origin"],
            extent=extent_from_axes(x_um, y_um),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(f"{profile}: {ARCHITECTURE_LABELS[profile]}", fontsize=8)
        ax.set_xlabel("Idler/object-plane coordinate, $x_i$ ($\\mathrm{\\mu m}$)")
        ax.set_ylabel("Idler/object-plane coordinate, $y_i$ ($\\mathrm{\\mu m}$)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title + (f"\nshared absolute scale [{cbar_label}]" if shared_scale else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "vmin": float(vmin) if vmin is not None else float("nan"),
        "vmax": float(vmax) if vmax is not None else float("nan"),
        "shared_scale": bool(shared_scale),
    }


def save_line_profile_overlay(
    path: Path,
    profiles: dict[str, dict[str, np.ndarray]],
    *,
    title: str,
    ylabel: str,
) -> None:
    """Overlay central line profiles for A/B/C/D (absolute y; no rescaling)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for profile in PROFILES:
        if profile not in profiles:
            continue
        blk = profiles[profile]
        ax.plot(
            blk["axis_um"],
            blk["values"],
            label=f"{profile} ({ARCHITECTURE_LABELS[profile]})",
            lw=1.6,
        )
    ax.set_xlabel("Idler/object-plane coordinate, $x_i$ ($\\mathrm{\\mu m}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
