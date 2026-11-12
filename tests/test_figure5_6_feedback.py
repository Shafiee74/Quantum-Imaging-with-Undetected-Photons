"""Regression tests for the Figure 5/6 physics-team labeling corrections."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from optics import idler_lens_misalignment_phase_map
from setup_params import (
    equivalent_wavelength_nm,
    idler_lens_equivalent_propagation_um,
    internal_camera_sampling_pixel_um,
    resolve_setup_selection,
)
from tools.c_lab_figure5_final_article import (
    EXPERIMENTAL_LABEL as FIG5_EXPERIMENTAL_LABEL,
    SIMULATION_LABEL as FIG5_SIMULATION_LABEL,
)
from tools.c_lab_figure5_final_theory_experiment_intensity import (
    LAMBDA_I_UM,
    wrapped_phase_rad,
)
from tools.c_lab_figure6_final_article_revision import (
    EXPERIMENTAL_LABEL as FIG6_EXPERIMENTAL_LABEL,
    SIMULATION_LABEL as FIG6_SIMULATION_LABEL,
)
from tools.c_lab_figure6b_corrected_effective_defocus import Z_EFF_MM, Z_STAGE_MM
from verification.convergence.definitions import SEED
from verification.convergence.object_sensitive import _seed_for


ROOT = Path(__file__).resolve().parents[1]


def test_figure5_6_source_labels_are_explicit() -> None:
    assert FIG5_SIMULATION_LABEL == "Simulation"
    assert FIG5_EXPERIMENTAL_LABEL == "Experimental"
    assert FIG6_SIMULATION_LABEL == "Simulation"
    assert FIG6_EXPERIMENTAL_LABEL == "Experimental"


def test_figure5_double_pass_mirror_phase_law() -> None:
    # Delta phi = 4*pi*Delta z/lambda_i: lambda_i/4 produces pi.
    assert wrapped_phase_rad(LAMBDA_I_UM / 4.0) == pytest.approx(np.pi)


def test_figure6_effective_displacement_mapping_is_label_only() -> None:
    assert np.asarray(Z_EFF_MM) == pytest.approx(2.0 * np.asarray(Z_STAGE_MM))


def test_figure6b_phase_uses_production_double_pass_once() -> None:
    setup = resolve_setup_selection("C_lab").overridden(
        enable_lens_rings=True,
        lens_shift_x_um=0.0,
        lens_shift_y_um=0.0,
    )
    dz_stage_um = 2400.0
    state = idler_lens_misalignment_phase_map(48, setup, delta_z_um=dz_stage_um)

    d_um = idler_lens_equivalent_propagation_um(
        dz_stage_um, float(setup.far_field_focal_mm)
    )
    assert state["equivalent_propagation_um"] == pytest.approx(d_um)
    assert state["pass_factor"] == pytest.approx(2.0)

    p_um = internal_camera_sampling_pixel_um(setup)
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float64)
    x_um = (xx - 47.0 / 2.0) * p_um
    y_um = (yy - 47.0 / 2.0) * p_um
    rho2_um2 = x_um * x_um + y_um * y_um
    f_c_um = float(setup.F2_mm) * 1.0e3
    lambda_eq_um = equivalent_wavelength_nm(
        setup.lambda_s_nm, setup.lambda_i_nm
    ) * 1.0e-3
    single_pass = np.pi * d_um * rho2_um2 / (f_c_um**2 * lambda_eq_um)
    assert np.allclose(state["phase_rad"], 2.0 * single_pass, rtol=1e-13, atol=1e-13)
    assert not np.allclose(state["phase_rad"], 4.0 * single_pass, rtol=1e-6, atol=1e-12)


def test_figure6_evidence_generators_do_not_pre_double_equivalent_distance() -> None:
    for relative in (
        "tools/c_lab_figure6b_corrected_effective_defocus.py",
        "tools/c_lab_figure6b_theoretical_ring_evolution.py",
        "tools/c_lab_figure6b_final_quantitative.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "doubled_equivalent_distance" not in source
    validation_source = (
        ROOT / "tools/c_lab_figure6_lens_displacement_validation.py"
    ).read_text(encoding="utf-8")
    assert "equivalent_distance_pre_doubled\": False" in validation_source
    assert "Δz_eff = 2 Δz_stage" in validation_source


def test_object_sensitive_seed_is_process_stable() -> None:
    target = "knife_edge_vertical"
    offset = int.from_bytes(hashlib.sha256(target.encode("utf-8")).digest()[:4], "big") % 997
    assert _seed_for("C", target, extra=11) == SEED + 2 * 7000 + offset + 11
