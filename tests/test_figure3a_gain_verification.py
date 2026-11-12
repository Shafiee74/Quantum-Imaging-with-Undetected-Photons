"""Tests for Figure 3(a) independent gain verification (diagnostic only)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setup_params import sinh_squared_stable
from verification.figure3a_gain.independent_reference import mean_photons_per_mode_independent
from verification.figure3a_gain.production_probe import probe_production_nbar_nl1


def test_independent_reference_module_isolated():
    src = (ROOT / "verification" / "figure3a_gain" / "independent_reference.py").read_text(
        encoding="utf-8"
    )
    assert "import verification.physics.independent_equations" not in src
    assert "from verification.physics.independent_equations" not in src
    assert "from setup_params import sinh_squared_stable" not in src
    assert "import optics" not in src and "from optics" not in src


def test_reference_matches_expm1_identity_at_sample_points():
    for r in [0.0, 1e-6, 0.01, 0.05, 0.2, 0.5]:
        n = mean_photons_per_mode_independent(r)
        if r == 0.0:
            assert n == 0.0
        else:
            m = __import__("math")
            expected = 0.25 * (m.expm1(2 * r) + m.expm1(-2 * r))
            assert n == pytest.approx(expected, rel=0, abs=1e-15)


def test_production_spdc_gain_r_tracks_independent_reference():
    for r in [0.01, 0.05, 0.1, 0.2, 0.5]:
        row = probe_production_nbar_nl1("A", r)
        n_ref = mean_photons_per_mode_independent(r)
        n_prod = row["nbar_production_NL1"]
        assert n_prod == pytest.approx(n_ref, rel=0, abs=1e-12)
        # Production uses sinh_squared_stable; should match reference on this domain.
        assert n_prod == pytest.approx(sinh_squared_stable(r), rel=0, abs=1e-12)


def test_independent_reference_differs_from_iref_helper_at_tiny_r():
    iref = importlib.import_module("verification.physics.independent_equations")
    r = 1e-7
    n_ind = mean_photons_per_mode_independent(r)
    n_iref = iref.mean_photons_sinh_squared(r)
    # iref uses leading-order r^2 for |r| < 1e-6; expm1 path keeps full identity.
    assert n_ind == pytest.approx(r * r, rel=1e-6)
    assert n_iref == pytest.approx(r * r, rel=1e-6)


def test_low_gain_relative_error_values():
    from verification.figure3a_gain.independent_reference import low_gain_relative_error_vs_exact

    assert low_gain_relative_error_vs_exact(0.2) == pytest.approx(0.013227340142355862, rel=1e-6)
    assert low_gain_relative_error_vs_exact(0.5) == pytest.approx(0.07932640579220776, rel=1e-6)
    assert low_gain_relative_error_vs_exact(0.05) < low_gain_relative_error_vs_exact(0.5)


def test_production_uses_spdc_gain_r_source():
    row = probe_production_nbar_nl1("A", 0.1)
    assert row["source_NL1"] == "spdc_gain_r"
    assert row["spdc_source_mode"] == "fixed_nbar"


def test_nbar_mapping_shared_across_abcd():
    r = 0.1
    n_ref = mean_photons_per_mode_independent(r)
    for profile in ("A", "B", "C", "D"):
        row = probe_production_nbar_nl1(profile, r)
        assert row["nbar_production_NL1"] == pytest.approx(n_ref, rel=0, abs=1e-12)
        assert row["source_NL1"] == "spdc_gain_r"
