"""Executable C_lab workflows must consume L=2 mm from config_c_lab_v1."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from setup_params import crystal_length_m_from_setup
from tools.c_lab_constants import C_LAB_CRYSTAL_LENGTH_M, C_LAB_CRYSTAL_LENGTH_MM
from tools.c_lab_isolated_pilot import build_pilot_setup
from verification.c_lab_crystal_length_static_audit import scan_file

ROOT = Path(__file__).resolve().parents[1]
A6_MODULES = sorted((ROOT / "tools").glob("c_lab_a6_*.py"))
Z_MODULES = [
    ROOT / "tools" / "c_lab_z8_rabbit_mi2_dk0_L_sweep.py",
    ROOT / "tools" / "c_lab_z8_rabbit_original_mi2_dk0_L_sweep.py",
    ROOT / "tools" / "c_lab_z9_original_rabbit_ms1_mi1_L_dk_sweep.py",
]


def test_build_pilot_setup_consumes_2mm() -> None:
    bundle = build_pilot_setup()
    for ix in (1, 2):
        lm = crystal_length_m_from_setup(bundle.effective, interaction=ix)
        assert abs(lm - C_LAB_CRYSTAL_LENGTH_M) < 1e-12


def test_c_lab_constants_match_yaml() -> None:
    assert abs(C_LAB_CRYSTAL_LENGTH_MM - 2.0) < 1e-12
    assert abs(C_LAB_CRYSTAL_LENGTH_M - 0.002) < 1e-12


@pytest.mark.parametrize("path", A6_MODULES + Z_MODULES, ids=lambda p: p.name)
def test_no_obsolete_crystal_length_literals(path: Path) -> None:
    hits = scan_file(path)
    assert not hits, hits


@pytest.mark.parametrize(
    "mod_name,func_name",
    [
        ("tools.c_lab_z8_rabbit_mi2_dk0_L_sweep", "build_setup"),
        ("tools.c_lab_z9_original_rabbit_ms1_mi1_L_dk_sweep", "build_setup"),
    ],
)
def test_build_setup_rejects_10mm(mod_name: str, func_name: str) -> None:
    mod = importlib.import_module(mod_name)
    build_setup = getattr(mod, func_name)
    with pytest.raises(ValueError, match="2"):
        build_setup(10.0)
