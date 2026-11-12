"""QIUP verification/convergence — U14 grid / sampling convergence.

Physical domain ``L = N·Δx`` is held fixed while ``N`` changes. Analytical
references and finest-grid **numerical-reference** comparisons are labeled
explicitly and never silently equated.
"""
from __future__ import annotations

from verification.convergence.definitions import DEFINITIONS_MANIFEST, PROFILES
from verification.convergence.grids import GridLevel, build_grid_levels, domain_um_for_profile
from verification.convergence.runner import run_convergence_suite, run_profile_convergence

__all__ = [
    "DEFINITIONS_MANIFEST",
    "PROFILES",
    "GridLevel",
    "build_grid_levels",
    "domain_um_for_profile",
    "run_convergence_suite",
    "run_profile_convergence",
]
