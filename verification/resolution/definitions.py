"""Predeclared resolution metric definitions (Task 08).

Definitions are fixed **before** interpreting results. Callers must not switch
to an alternate width convention because another number looks better.

Quantity families (never silently equated)
-----------------------------------------
* ``intensity`` — selected-port ideal intensity image ``I``
* ``visibility`` — local Michelson map ``V = |I+ − I−| / (I+ + I− + ε)``

Amplitude ``√I`` may be saved as a diagnostic array but is **not** used as a
silent substitute for either family.
"""
from __future__ import annotations

from typing import Any

# Canonical sampling for the Task 08 evidence suite.
DEFAULT_GRID_N: int = 48
DEFAULT_SEED: int = 20260930
DEFAULT_NBAR: float = 0.01

# Profile extraction
PSF_AXIS: str = "horizontal"  # 1-D cut through peak along +x
ESF_PLATEAU_FRACTION: float = 0.15  # outer fraction used for I_low / I_high
ESF_LOW_LEVEL: float = 0.10
ESF_HIGH_LEVEL: float = 0.90
# Do NOT change these levels retrospectively (e.g. to 20–80%).
ESF_TRANSITION_LEVELS: tuple[float, float] = (ESF_LOW_LEVEL, ESF_HIGH_LEVEL)

LSF_FROM: str = "gradient_of_ESF_nonnegative_rising_lobe"

FWHM_METHOD: str = "half_maximum_linear_interpolation"
CENTROID_METHOD: str = "intensity_weighted_1d"

QUANTITY_INTENSITY: str = "intensity"
QUANTITY_VISIBILITY: str = "visibility"
QUANTITIES: tuple[str, ...] = (QUANTITY_INTENSITY, QUANTITY_VISIBILITY)

TARGETS: tuple[str, ...] = (
    "point",
    "narrow_slit",
    "knife_edge_vertical",
    "knife_edge_horizontal",
)

PROFILES: tuple[str, ...] = ("A", "B", "C", "D")

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "psf_axis": PSF_AXIS,
    "psf_fwhm_method": FWHM_METHOD,
    "esf_plateau_fraction": ESF_PLATEAU_FRACTION,
    "esf_transition_levels": list(ESF_TRANSITION_LEVELS),
    "esf_transition_name": "10_to_90_percent",
    "lsf_definition": LSF_FROM,
    "lsf_fwhm_method": FWHM_METHOD,
    "centroid_method": CENTROID_METHOD,
    "quantities": list(QUANTITIES),
    "quantity_notes": {
        QUANTITY_INTENSITY: "ideal_signal_image (selected port intensity)",
        QUANTITY_VISIBILITY: "abs(I_plus-I_minus)/(I_plus+I_minus+eps); not intensity",
    },
    "amplitude_policy": (
        "sqrt(I) may be exported as diagnostic only; metrics use intensity or "
        "visibility channels explicitly tagged"
    ),
    "targets": list(TARGETS),
    "profiles": list(PROFILES),
    "selection_policy": (
        "Definitions fixed a priori; do not pick alternate width conventions "
        "after seeing which result looks best."
    ),
}
