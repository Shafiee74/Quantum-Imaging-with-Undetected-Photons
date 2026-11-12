"""Authoritative C_lab crystal length from production YAML (config_c_lab_v1).

All executable C_lab workflows must use these values, not obsolete 10 mm pilots.
"""
from __future__ import annotations

from setup_params import crystal_length_m_from_setup, resolve_setup_selection

_OBSOLETE_L_MM = 10.0
_OBSOLETE_L_M = 0.010

# Historical incorrect assumption — do not use in executable paths.
OBSOLETE_CRYSTAL_LENGTH_MM = _OBSOLETE_L_MM
OBSOLETE_CRYSTAL_LENGTH_M = _OBSOLETE_L_M


def c_lab_production_setup():
    return resolve_setup_selection("C_lab")


def c_lab_crystal_length_m(setup=None) -> float:
    s = setup if setup is not None else c_lab_production_setup()
    return float(crystal_length_m_from_setup(s, interaction=1))


def c_lab_crystal_length_mm(setup=None) -> float:
    return c_lab_crystal_length_m(setup) * 1000.0


# Module-level constants resolved once from YAML (L = 2 mm production freeze).
_C_LAB_L_M = c_lab_crystal_length_m()
if abs(_C_LAB_L_M - 0.002) > 1e-12:
    raise RuntimeError(f"config_c_lab_v1 crystal length must be 2 mm, got {_C_LAB_L_M * 1000} mm")

C_LAB_CRYSTAL_LENGTH_M: float = _C_LAB_L_M
C_LAB_CRYSTAL_LENGTH_MM: float = _C_LAB_L_M * 1000.0
