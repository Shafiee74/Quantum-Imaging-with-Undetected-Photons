"""QIUP verification/physics — independent reference equations (Task 04).

This package hosts **INDEPENDENT REFERENCE** implementations for U1–U9 style
checks. It must not import production physics modules.

Existing Configuration-D independent solvers remain in
``verification.independent_reference`` and are reused (not duplicated) via
``independent_equations``.
"""
from __future__ import annotations

from verification.physics.equation_meta import EQUATION_REGISTRY, EquationSpec, inventory_rows
from verification.physics.independent_equations import *  # noqa: F401,F403
from verification.physics.independent_equations import analytical_reference_casebook

__all__ = [
    "EQUATION_REGISTRY",
    "EquationSpec",
    "analytical_reference_casebook",
    "inventory_rows",
]
