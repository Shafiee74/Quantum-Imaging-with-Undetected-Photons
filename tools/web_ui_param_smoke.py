"""Smoke-test web-UI knobs: each should change the noisy A image vs baseline.

Mirrors qiup_ui_backend run merge + web bridges (pump waist sync, s_focus→Δz).
Exit code 0 if all expected movers change; 1 if any expected mover is flat.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import generate_mnist5_four_arrangement_images as gen  # noqa: E402
from latents import LatentState  # noqa: E402
from qiup_ui_backend import (  # noqa: E402
    _bridge_setup_for_web_latents,
    _build_reference,
    _latents,
    _sync_pump_waists,
    _thermal_overrides,
)
from setup_params import resolve_setup_selection  # noqa: E402
from simulate import simulate_forward  # noqa: E402

SIZE = 32
SEED = 7
PROFILE = "A"
EPS = 1e-4


def _simulate(noisy_setup: dict[str, Any], noisy_latents: dict[str, Any]) -> np.ndarray:
    ref, _ = _build_reference({"source": "mnist", "digit": 5}, SIZE)
    base = resolve_setup_selection(PROFILE)
    setup_raw = _sync_pump_waists({**gen.NOISY_SETUP, **noisy_setup})
    lat_raw = _latents({**gen.NOISY_LATENTS, **noisy_latents})
    overrides = {**gen._allowed_overrides(base, setup_raw), **_thermal_overrides(base, None)}
    setup = _bridge_setup_for_web_latents(base.overridden(**overrides), lat_raw)
    result = simulate_forward(
        ref,
        LatentState(setup=setup, **lat_raw),
        seed=SEED,
        detector_noise=True,
        raise_on_forbidden=False,
    )
    return np.asarray(result.detector_signal_image, dtype=np.float64)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


Patch = Callable[[dict[str, Any], dict[str, Any]], None]

CASES: list[tuple[str, Patch, bool]] = [
    ("mean_photons_per_mode", lambda s, l: l.__setitem__("mean_photons_per_mode", 8.0), True),
    ("em_gain", lambda s, l: l.__setitem__("em_gain", 120.0), True),
    ("exposure_s", lambda s, l: l.__setitem__("exposure_s", 0.2), True),
    ("num_frames", lambda s, l: l.__setitem__("num_frames", 8), True),
    ("s_pump", lambda s, l: l.__setitem__("s_pump", 0.9), True),
    ("s_align", lambda s, l: l.__setitem__("s_align", 0.9), True),
    ("s_det", lambda s, l: l.__setitem__("s_det", 0.9), True),
    ("s_cool", lambda s, l: l.__setitem__("s_cool", 0.9), True),
    # Amplitude-only MNIST (φ≡0): SLM phase-calibration latent is intentionally flat.
    ("s_slm", lambda s, l: l.__setitem__("s_slm", 0.9), False),
    ("delta_phi_rms", lambda s, l: l.__setitem__("delta_phi_rms", 0.4), True),
    ("s_gain", lambda s, l: l.__setitem__("s_gain", 0.9), False),
    ("s_focus", lambda s, l: l.__setitem__("s_focus", 1.0), True),
    (
        "s_thermal",
        lambda s, l: l.update({"s_thermal": 0.7, "num_frames": 64, "exposure_s": 0.5}),
        True,
    ),
    ("sigma_blur_extra_px", lambda s, l: s.__setitem__("sigma_blur_extra_px", 2.5), True),
    ("sigma_diffraction_px", lambda s, l: s.__setitem__("sigma_diffraction_px", 2.0), True),
    ("sigma_aberration_px", lambda s, l: s.__setitem__("sigma_aberration_px", 2.0), True),
    ("image_rotation_deg", lambda s, l: s.__setitem__("image_rotation_deg", 4.0), True),
    ("slm_misreg_x_px", lambda s, l: s.__setitem__("slm_misreg_x_px", 3.0), True),
    ("slm_misreg_y_px", lambda s, l: s.__setitem__("slm_misreg_y_px", 3.0), True),
    ("delta_phi_align_span_rad", lambda s, l: s.__setitem__("delta_phi_align_span_rad", 0.8), True),
    (
        "lens_shift_x_um+rings",
        lambda s, l: s.update({"enable_lens_rings": True, "lens_shift_x_um": 40.0}),
        True,
    ),
    (
        "lens_shift_y_um+rings",
        lambda s, l: s.update({"enable_lens_rings": True, "lens_shift_y_um": 40.0}),
        True,
    ),
    (
        "delta_z_um+rings",
        lambda s, l: s.update({"enable_lens_rings": True, "delta_z_um": 80.0}),
        True,
    ),
    ("enable_lens_rings_alone", lambda s, l: s.__setitem__("enable_lens_rings", True), False),
    ("enable_etalon_fringes", lambda s, l: s.__setitem__("enable_etalon_fringes", True), True),
    ("blur_model_double", lambda s, l: s.__setitem__("blur_model", "double_blur"), True),
    ("read_noise0_e", lambda s, l: s.__setitem__("read_noise0_e", 30.0), True),
    ("prnu_sigma", lambda s, l: s.__setitem__("prnu_sigma", 0.08), True),
    ("cic0", lambda s, l: s.__setitem__("cic0", 0.04), True),
    ("dark0", lambda s, l: s.__setitem__("dark0", 0.015), True),
    ("hot_frac_at_s1", lambda s, l: s.__setitem__("hot_frac_at_s1", 0.008), True),
    # Accidental pedestal exists in twin code but does not move the current
    # analog EMCCD web path image (verified flat even at s_gain=1).
    ("accidental_e_at_s1", lambda s, l: s.__setitem__("accidental_e_at_s1", 0.04), False),
    ("pump_waist_um+double_blur", lambda s, l: s.update({"pump_waist_um": 120.0, "blur_model": "double_blur", "derive_sigma_spdc": True}), True),
    ("pixel_um", lambda s, l: s.__setitem__("pixel_um", 16.0), True),
    ("object_map_pixel_um", lambda s, l: s.__setitem__("object_map_pixel_um", 4.0), True),
    (
        "far_field_idler_relay_magnification",
        lambda s, l: s.__setitem__("far_field_idler_relay_magnification", 2.0),
        True,
    ),
]


def main() -> int:
    print(f"Baseline: profile={PROFILE} size={SIZE} seed={SEED}")
    baseline = _simulate({}, {})
    print(f"baseline shape={baseline.shape} mean={baseline.mean():.4g} std={baseline.std():.4g}")

    unexpected_flat: list[str] = []
    unexpected_move: list[str] = []

    for name, patch, expect in CASES:
        setup: dict[str, Any] = {}
        latents: dict[str, Any] = {}
        patch(setup, latents)
        try:
            img = _simulate(setup, latents)
            rms = _rms(baseline, img)
            moved = rms > EPS
            if expect and not moved:
                status = "FAIL_FLAT"
                unexpected_flat.append(name)
            elif (not expect) and moved:
                status = "WARN_MOVED"
                unexpected_move.append(name)
            else:
                status = "OK"
            print(f"  {status:10s}  rms={rms:12.5g}  expect={expect}  {name}")
        except Exception as exc:  # noqa: BLE001
            unexpected_flat.append(name)
            print(f"  ERROR       {name}: {exc}")
            traceback.print_exc()

    print("\n=== SUMMARY ===")
    print(
        f"tested={len(CASES)}  flat={len(unexpected_flat)}  unexpected_move={len(unexpected_move)}"
    )
    if unexpected_flat:
        print("FLAT/ERROR (should change):", ", ".join(unexpected_flat))
    if unexpected_move:
        print("MOVED (expected flat):", ", ".join(unexpected_move))
    return 1 if unexpected_flat else 0


if __name__ == "__main__":
    raise SystemExit(main())
