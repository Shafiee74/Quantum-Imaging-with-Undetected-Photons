"""QIUP verification/phase_reconstruction — shared four-step retrieval.

PRODUCTION IMPLEMENTATION note
------------------------------
Production ``optics.reconstruct_phase_from_phase_steps`` remains the
in-simulator analysis path. This package is the **shared** algorithm
intended for simulation demos **and** experimental Configuration-C
intensity stacks. Do not fork a second experimental reconstructor.

Public API
----------
- ``reconstruct_four_step`` / ``reconstruct_phase_steps``
- ``circular_phase_error`` / ``phase_error_metrics``
- ``synthesize_four_step_frames`` / example runners
"""
from __future__ import annotations

from verification.phase_reconstruction.constants import (
    FOUR_STEP_LABELS,
    FOUR_STEP_PHASES_RAD,
)
from verification.phase_reconstruction.examples import (
    EXAMPLE_SPECS,
    run_all_examples,
    run_example,
)
from verification.phase_reconstruction.metrics import (
    circular_phase_error,
    phase_error_metrics,
    remove_global_piston,
    visibility_from_ab,
    wrap_phase,
)
from verification.phase_reconstruction.reconstruct import (
    PhaseReconstructionResult,
    reconstruct_four_step,
    reconstruct_phase_steps,
)
from verification.phase_reconstruction.synthetic import (
    amplitude_phase_maps,
    constant_phase_maps,
    phase_ramp_maps,
    phase_step_maps,
    synthesize_four_step_frames,
)
from verification.phase_reconstruction.unwrap import maybe_unwrap, unwrap_phase_2d

__all__ = [
    "EXAMPLE_SPECS",
    "FOUR_STEP_LABELS",
    "FOUR_STEP_PHASES_RAD",
    "PhaseReconstructionResult",
    "amplitude_phase_maps",
    "circular_phase_error",
    "constant_phase_maps",
    "maybe_unwrap",
    "phase_error_metrics",
    "phase_ramp_maps",
    "phase_step_maps",
    "reconstruct_four_step",
    "reconstruct_phase_steps",
    "remove_global_piston",
    "run_all_examples",
    "run_example",
    "synthesize_four_step_frames",
    "unwrap_phase_2d",
    "visibility_from_ab",
    "wrap_phase",
]
