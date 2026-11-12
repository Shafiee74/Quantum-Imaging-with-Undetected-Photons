"""Shared graphics-only publication defaults for manuscript Figures 3--6.

The helper deliberately contains no numerical transforms, axis limits, data
selection, or physical parameters. Individual figure generators retain their
validated scientific content and figure-specific font sizes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


PUBLICATION_SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]

# Authoritative graphics tokens sampled from the approved Figure 3(a)--3(d)
# generators in evidence/physics_team_feedback_release/graphic_example.  These
# constants are presentation-only; they must never be used in numerical code.
REFERENCE_NAVY = "#1f4e79"
REFERENCE_TEAL = "#2a9d8f"
REFERENCE_DARK_TEAL = "#264653"
REFERENCE_GOLD = "#e9c46a"

# Four-series palette used by the approved Figure 3(d) reference panel.
CONFIGURATION_COLORS = {
    "A": "#3a7ca5",
    "B": REFERENCE_TEAL,
    "C": "#586e99",
    "D": REFERENCE_GOLD,
}
CONFIGURATION_MARKERS = {"A": "o", "B": "s", "C": "D", "D": "^"}
CONFIGURATION_LINESTYLES = {
    "A": "-",
    "B": (0, (6.0, 3.0)),
    "C": "-",
    "D": (0, (2.5, 2.0, 6.0, 2.0)),
}


def apply_publication_graphics_style() -> None:
    """Apply common typography and stroke/export settings only."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": PUBLICATION_SANS_SERIF,
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_reference_axes(ax) -> None:
    """Apply the clean, grid-free axes treatment of reference Figure 3(a--d)."""
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
