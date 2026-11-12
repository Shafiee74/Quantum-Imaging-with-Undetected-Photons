"""Numerical invariance after A/B/C/D letter swap (Figure 1 manuscript mapping).

Physical YAML registry names are the stable architecture identity.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from latents import LatentState
from setup_params import (
    IMAGING_MODE_FAR,
    IMAGING_MODE_NEAR,
    REGISTERED_SETUP_PROFILES,
    SPDC_ARCHITECTURE_SINGLE,
    SPDC_ARCHITECTURE_TWO,
    resolve_architecture_profile,
    resolve_setup_selection,
)
from simulate import simulate_forward
from verification.monte_carlo.channels import make_phase_reference

# Pre-migration letter → registered YAML (physical identity preserved).
LEGACY_LETTER_TO_REGISTRY: dict[str, str] = {
    "A": "two_crystal_2014",
    "B": "single_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "two_crystal_near_field",
}

NEW_LETTER_TO_REGISTRY: dict[str, str] = {
    "A": "two_crystal_2014",
    "B": "two_crystal_near_field",
    "C": "single_crystal_far_field",
    "D": "single_crystal_near_field",
}

MIGRATION: dict[str, str] = {"A": "A", "B": "D", "C": "C", "D": "B"}


def _ref(setup, n: int = 24) -> dict:
    from setup_params import resolved_spatial_mapping

    px = float(resolved_spatial_mapping(setup)["object_plane_pixel_um"])
    return make_phase_reference(n, px, phi_rad=0.0)


def _run(setup, seed: int = 20261019) -> np.ndarray:
    state = LatentState(
        setup=setup,
        mean_photons_per_mode=0.01,
        num_frames=1,
        exposure_s=0.02,
        em_gain=1.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )
    out = simulate_forward(_ref(setup), state, seed=seed, detector_noise=False)
    return np.asarray(out.ideal_signal_image, dtype=np.float64)


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


@pytest.mark.parametrize("legacy_letter", ["A", "B", "C", "D"])
def test_new_letter_resolves_same_yaml_as_legacy_physical(legacy_letter: str) -> None:
    new_letter = MIGRATION[legacy_letter]
    old_reg = LEGACY_LETTER_TO_REGISTRY[legacy_letter]
    new_reg = NEW_LETTER_TO_REGISTRY[new_letter]
    assert old_reg == new_reg
    old_setup = resolve_setup_selection(old_reg)
    new_setup = resolve_setup_selection(new_letter)
    assert old_setup.setup_id == new_setup.setup_id
    assert resolve_architecture_profile(old_setup).spatial_transfer_path == (
        resolve_architecture_profile(new_setup).spatial_transfer_path
    )


@pytest.mark.parametrize("legacy_letter", ["A", "B", "C", "D"])
def test_simulation_invariant_under_letter_migration(legacy_letter: str) -> None:
    new_letter = MIGRATION[legacy_letter]
    reg = LEGACY_LETTER_TO_REGISTRY[legacy_letter]
    img_legacy = _run(resolve_setup_selection(reg))
    img_new = _run(resolve_setup_selection(new_letter))
    assert _hash(img_legacy) == _hash(img_new)
    assert img_legacy == pytest.approx(img_new, rel=0, abs=0)


def test_manuscript_architecture_table() -> None:
    expected = {
        "A": (SPDC_ARCHITECTURE_TWO, IMAGING_MODE_FAR),
        "B": (SPDC_ARCHITECTURE_TWO, IMAGING_MODE_NEAR),
        "C": (SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_FAR),
        "D": (SPDC_ARCHITECTURE_SINGLE, IMAGING_MODE_NEAR),
    }
    for letter, (arch, img) in expected.items():
        setup = resolve_setup_selection(letter)
        prof = resolve_architecture_profile(setup)
        assert prof.profile_id == letter
        assert prof.spdc_architecture == arch
        assert prof.imaging_mode == img


def test_legacy_to_manuscript_relabel_helpers() -> None:
    from verification.configuration_identity import (
        LEGACY_TO_MANUSCRIPT_LETTER,
        manuscript_letter_from_legacy_evidence,
        registry_for_legacy_evidence_letter,
        registry_for_manuscript_letter,
    )

    assert LEGACY_TO_MANUSCRIPT_LETTER == {"A": "A", "B": "D", "C": "C", "D": "B"}
    assert manuscript_letter_from_legacy_evidence("B") == "D"
    assert manuscript_letter_from_legacy_evidence("D") == "B"
    assert (
        registry_for_legacy_evidence_letter("B")
        == registry_for_manuscript_letter("D")
        == "single_crystal_near_field"
    )
    assert (
        registry_for_legacy_evidence_letter("D")
        == registry_for_manuscript_letter("B")
        == "two_crystal_near_field"
    )


def _observables(setup, seed: int = 20261019) -> dict:
    state = LatentState(
        setup=setup,
        mean_photons_per_mode=0.01,
        num_frames=1,
        exposure_s=0.02,
        em_gain=1.0,
        s_pump=0.0,
        s_gain=0.0,
        s_align=0.0,
        s_thermal=0.0,
        s_det=0.0,
        s_cool=0.0,
        s_focus=0.0,
        s_slm=0.0,
    )
    out = simulate_forward(_ref(setup), state, seed=seed, detector_noise=False)
    vo = dict(out.meta.get("visibility_observables") or {})
    return {
        "ideal_mean": float(np.mean(out.ideal_signal_image)),
        "ideal_max": float(np.max(out.ideal_signal_image)),
        "V_50_50": float(vo.get("V_50_50", float("nan"))),
        "nbar_NL1": float(out.meta.get("nbar_NL1", float("nan"))),
    }


@pytest.mark.parametrize("legacy_letter", ["A", "B", "C", "D"])
def test_observables_invariant_under_letter_migration(legacy_letter: str) -> None:
    new_letter = MIGRATION[legacy_letter]
    reg = LEGACY_LETTER_TO_REGISTRY[legacy_letter]
    obs_legacy = _observables(resolve_setup_selection(reg))
    obs_new = _observables(resolve_setup_selection(new_letter))
    assert obs_legacy == obs_new


def test_c_lab_unchanged_and_distinct_from_c() -> None:
    c = resolve_setup_selection("C")
    c_lab = resolve_setup_selection("C_lab")
    assert c.setup_id == "single_crystal_far_field_v2"
    assert c_lab.setup_id == "config_c_lab_v1"
    assert c_lab.setup_id != c.setup_id
    assert REGISTERED_SETUP_PROFILES["config_c_lab_v1"].endswith("setup_config_c_lab_v1.yaml")
