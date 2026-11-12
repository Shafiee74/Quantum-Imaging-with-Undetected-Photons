"""Guards the frozen C_lab production YAML (Mi=2.0, L=2 mm, calculated Δk₀)."""
from __future__ import annotations

from setup_params import (
    delta_k_qpm_per_m_from_setup,
    resolve_setup_selection,
    resolved_spatial_mapping,
)

M_SIGNED_EXPECTED = -810.0 / 3100.0  # -(Ms/Mi)*λs*fs/(λi*fi), Ms=1, Mi=2, 150 mm focals


def test_c_lab_production_yaml_mi2_l2_calculated_dk0() -> None:
    setup = resolve_setup_selection("C_lab")
    assert setup.setup_id == "config_c_lab_v1"
    assert float(setup.crystal_length_mm) == 2.0
    assert float(setup.far_field_signal_relay_magnification) == 1.0
    assert float(setup.far_field_idler_relay_magnification) == 2.0
    assert float(setup.pixel_um) == 13.0
    assert float(setup.object_map_pixel_um) == 13.0
    assert setup.object_type == "transmissive"
    assert int(setup.object_interaction_count) == 2
    assert float(setup.y1_rabbit_aperture_width_mm) == 7.285
    assert float(setup.y1_rabbit_aperture_height_mm) == 9.850
    dk1 = float(delta_k_qpm_per_m_from_setup(setup, interaction=1))
    dk2 = float(delta_k_qpm_per_m_from_setup(setup, interaction=2))
    assert abs(dk1 - 1090.5510343926662) < 1e-6
    assert abs(dk2 + 1090.5510343926662) < 1e-6
    m = float(resolved_spatial_mapping(setup)["signed_object_to_camera_magnification"])
    assert abs(m - M_SIGNED_EXPECTED) < 1e-12
