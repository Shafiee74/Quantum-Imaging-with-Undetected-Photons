"""Matched four-configuration benchmark definitions (Task 15)."""
from __future__ import annotations

from typing import Any

from benchmarks.targets import PHYSICAL_TRUTH

PROFILES: tuple[str, ...] = ("A", "B", "C", "D")

# Paper-facing architecture labels (must match setup_params ArchitectureProfile).
ARCHITECTURE_LABELS: dict[str, str] = {
    "A": "two-crystal far-field",
    "B": "two-crystal near-field",
    "C": "single-crystal double-pass far-field",
    "D": "single-crystal double-pass near-field",
}

# Shared physical targets (PHYSICAL_TRUTH) sampled on each config's native
# object-plane pixel — same physical ground truth, correct local sampling.
BENCHMARK_TARGETS: tuple[str, ...] = (
    "transparent",
    "amplitude_step",
    "phase_step",
    "point",
    "knife_edge_vertical",
)

PHASE_TARGETS: tuple[str, ...] = ("phase_step",)
RESOLUTION_TARGETS: tuple[str, ...] = ("point", "knife_edge_vertical")

GRID_N: int = 48
MASTER_SEED: int = 20261515
NBAR: float = 0.01
NUM_FRAMES_OPTICAL: int = 1
NUM_FRAMES_PHASE: int = 4

# Plotting: never hide differences with per-panel max-normalization.
PLOT_POLICY: dict[str, Any] = {
    "comparison_intensity_scale": "shared_absolute_vmin_vmax_across_ABCD",
    "per_config_colorbar": "absolute_native_units",
    "forbidden": [
        "per_image_max_normalize_on_comparison_panels",
        "silent_sqrt_I_as_intensity",
        "equating_visibility_to_intensity",
    ],
    "cmap_intensity": "viridis",
    "cmap_visibility": "magma",
    "cmap_phase": "twilight",
    "cmap_difference": "RdBu_r",
    "origin": "lower",
    "extent_units": "um",
}

DEFINITIONS_MANIFEST: dict[str, Any] = {
    "task": "task15_four_config_benchmark",
    "profiles": list(PROFILES),
    "architecture_labels": dict(ARCHITECTURE_LABELS),
    "benchmark_targets": list(BENCHMARK_TARGETS),
    "physical_truth": {
        k: PHYSICAL_TRUTH[k]
        for k in PHYSICAL_TRUTH
        if k
        not in (
            "default_n",
            "default_pixel_um",
        )
    },
    "sampling_policy": (
        "Same PHYSICAL_TRUTH for all configs; each config uses its native "
        "object_plane_pixel_um from resolved_spatial_mapping (matched physical "
        "target, architecture-correct sampling)."
    ),
    "matched_controls": {
        "nbar": NBAR,
        "grid_n": GRID_N,
        "master_seed": MASTER_SEED,
        "detector_noise": False,
        "latents": "quiet (all s_*=0)",
    },
    "plot_policy": PLOT_POLICY,
    "metrics": [
        "object_truth",
        "optical_intensity",
        "N1",
        "N2",
        "C12_abs",
        "visibility",
        "reconstructed_phase",
        "detector_coordinates",
        "object_coordinates",
        "magnification_mapping",
        "resolution_metrics",
        "line_profiles",
    ],
}
