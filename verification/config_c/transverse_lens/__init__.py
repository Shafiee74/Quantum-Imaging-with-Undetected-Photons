"""Independent Config-C transverse idler-lens verification (Task 17)."""

from verification.config_c.transverse_lens.derivation import (
    DIMENSIONAL_STATUS,
    LegacyFarFieldBug,
    config_c_pass_factor,
    equivalent_wavelength_nm,
    fringe_period_um,
    fringe_spatial_frequency_per_um,
    phase_slope_rad_per_um,
    thin_lens_linear_phase_rad,
    transverse_wavevector_rad_per_um,
    validate_dimensional_consistency,
)
from verification.config_c.transverse_lens.reference import (
    predicted_observables,
    transverse_phase_map,
)

__all__ = [
    "DIMENSIONAL_STATUS",
    "LegacyFarFieldBug",
    "config_c_pass_factor",
    "equivalent_wavelength_nm",
    "fringe_period_um",
    "fringe_spatial_frequency_per_um",
    "phase_slope_rad_per_um",
    "predicted_observables",
    "thin_lens_linear_phase_rad",
    "transverse_phase_map",
    "transverse_wavevector_rad_per_um",
    "validate_dimensional_consistency",
]
