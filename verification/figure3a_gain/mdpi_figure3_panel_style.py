"""Reusable MDPI-oriented matplotlib style for Figure 3 panels (a–e).

Apply via ``apply_mdpi_figure3_rcparams()`` before plotting. Panel (a) uses this
first; panels (b–e) can adopt the same rcParams in a later integration pass.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

# Match taskX8 / composite Figure 3 panel (a) physical size (inches).
FIG3_PANEL_SIZE_IN = (5.6, 3.8)
FIG3_DPI_EXPORT = 600

FIG3_COLORS = {
    "analytical_reference": "#1f4e79",
    "production_simulator": "#2a9d8f",
    "relative_error": "#264653",
}

MDPI_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": FIG3_DPI_EXPORT,
    "figure.dpi": 100,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Figure 3(a) inset: [x0, y0, width, height] in axes fraction (upper-left / slightly left of center).
FIG3A_INSET_BOUNDS = (0.18, 0.60, 0.34, 0.31)


def apply_mdpi_figure3_rcparams() -> None:
    plt.rcParams.update(MDPI_RC)


def style_axes_clean(ax) -> None:
    """MDPI-style axes: no grid, white background."""
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
