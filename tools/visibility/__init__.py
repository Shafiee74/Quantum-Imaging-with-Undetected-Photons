"""C_lab phase-resolved visibility analysis (experimental TIFF + simulated stacks).

Public API — see ``tools/visibility/README.md`` for usage.

Figure 5 four-step pixel maps remain in ``tools.z1.estimator``; this package
generalizes sinusoidal LS to arbitrary mirror positions and separates metric labels.
"""

from tools.visibility.c_lab_shared import (
    DEFAULT_POSITION_UM,
    bootstrap_integrated,
    figure5_four_step_map,
    fit_integrated,
    fit_pixelwise,
    load_raw_tiff_phase_series,
    unique_tiff_files,
    moments_to_phase_stack,
    phase_from_mirror_um,
    reconciliation_report,
    run_experiment,
    run_simulated_stack,
    theory_integrated_moments,
    verify_integrated_matches_pixelwise_sums,
    visibility_from_abc,
)
from tools.visibility.intensity import BackgroundMode, prepare_intensity_frame

__all__ = [
    "BackgroundMode",
    "DEFAULT_POSITION_UM",
    "bootstrap_integrated",
    "figure5_four_step_map",
    "fit_integrated",
    "fit_pixelwise",
    "load_raw_tiff_phase_series",
    "unique_tiff_files",
    "moments_to_phase_stack",
    "phase_from_mirror_um",
    "prepare_intensity_frame",
    "reconciliation_report",
    "run_experiment",
    "run_simulated_stack",
    "theory_integrated_moments",
    "verify_integrated_matches_pixelwise_sums",
    "visibility_from_abc",
]
