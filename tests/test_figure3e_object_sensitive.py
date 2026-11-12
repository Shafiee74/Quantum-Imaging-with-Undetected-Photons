"""Object-sensitive knife-edge R on fixed-FoV detector (Figure 3e)."""
from __future__ import annotations

import pytest

from verification.convergence.fixed_fov import build_fixed_fov_levels
from verification.convergence.object_sensitive import (
    sensitivity_diagnostics,
    simulate_knife_R_on_detector,
)


@pytest.mark.parametrize("profile", ["A", "B", "C", "D"])
def test_knife_edge_spatially_sensitive(profile: str) -> None:
    lev = build_fixed_fov_levels(profile, (96,), n_cam=48)[0]
    run = simulate_knife_R_on_detector(lev, profile=profile)
    diag = sensitivity_diagnostics(run)
    assert diag["passes_spatial_sensitivity"], diag
