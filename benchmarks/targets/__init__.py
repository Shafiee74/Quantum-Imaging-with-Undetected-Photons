"""QIUP benchmarks/targets — deterministic physical ground-truth objects.

Config-agnostic: the same physical parameters apply for A/B/C/D. Only the
sampling grid ``(n, pixel_um)`` changes when a consumer needs a different
resolution; geometry is rebuilt from ``PHYSICAL_TRUTH``, not forked.
"""
from __future__ import annotations

from benchmarks.targets.catalog import (
    PHYSICAL_TRUTH,
    TARGET_BUILDERS,
    TARGET_IDS,
    build_all_targets,
    build_target,
    make_amplitude_step,
    make_blocked,
    make_checkerboard,
    make_knife_edge_horizontal,
    make_knife_edge_vertical,
    make_narrow_slit,
    make_phase_ramp,
    make_phase_step,
    make_point,
    make_transparent,
    target_definitions_dict,
)
from benchmarks.targets.geometry import (
    DEFAULT_N,
    DEFAULT_PIXEL_UM,
    centered_axis_um,
    coordinate_grids_um,
    geometry_metadata,
    physical_extent_um,
)
from benchmarks.targets.target import BenchmarkTarget

__all__ = [
    "DEFAULT_N",
    "DEFAULT_PIXEL_UM",
    "PHYSICAL_TRUTH",
    "TARGET_BUILDERS",
    "TARGET_IDS",
    "BenchmarkTarget",
    "build_all_targets",
    "build_target",
    "centered_axis_um",
    "coordinate_grids_um",
    "geometry_metadata",
    "make_amplitude_step",
    "make_blocked",
    "make_checkerboard",
    "make_knife_edge_horizontal",
    "make_knife_edge_vertical",
    "make_narrow_slit",
    "make_phase_ramp",
    "make_phase_step",
    "make_point",
    "make_transparent",
    "physical_extent_um",
    "target_definitions_dict",
]
