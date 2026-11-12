"""Equation metadata for independent physics references (Task 04).

PRODUCTION IMPLEMENTATION lives in ``setup_params.py`` / ``optics.py`` / …
INDEPENDENT REFERENCE IMPLEMENTATION lives in ``verification.independent_reference``
and ``verification.physics.independent_equations``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EquationSpec:
    """Documentation block attached to every independent reference function."""

    function: str
    equation: str
    variables: dict[str, str]
    si_units: dict[str, str]
    assumptions: tuple[str, ...]
    configurations: tuple[str, ...]
    source: str
    domain: str
    u_suite: tuple[str, ...] = ()
    implementation_home: str = "verification.physics.independent_equations"
    reuses_existing_independent: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Registry filled by independent_equations at import time.
EQUATION_REGISTRY: dict[str, EquationSpec] = {}


def register(spec: EquationSpec) -> EquationSpec:
    if spec.function in EQUATION_REGISTRY:
        raise ValueError(f"duplicate equation registration: {spec.function}")
    EQUATION_REGISTRY[spec.function] = spec
    return spec


def inventory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(EQUATION_REGISTRY):
        spec = EQUATION_REGISTRY[name]
        rows.append(
            {
                "function": spec.function,
                "u_suite": ";".join(spec.u_suite),
                "configurations": ";".join(spec.configurations),
                "equation": spec.equation,
                "si_units": "; ".join(f"{k} [{v}]" for k, v in spec.si_units.items()),
                "assumptions": " | ".join(spec.assumptions),
                "source": spec.source,
                "domain": spec.domain,
                "implementation_home": spec.implementation_home,
                "reuses_existing_independent": spec.reuses_existing_independent or "",
                "notes": spec.notes,
            }
        )
    return rows
