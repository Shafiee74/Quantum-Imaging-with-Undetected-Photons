#!/usr/bin/env python3
"""Isolated C_lab physics-correction validation pilot (non-production).

Creates evidence under evidence/c_lab_isolated_pilot/<timestamp>/ only.
Does not modify production sources, configs, or frozen experimental data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optics
import setup_params
from benchmarks.four_config.runner import quiet_state
from optics import (
    _interfere_from_moments,
    far_field_signal_marginal_from_spdc,
    gaussian_filter,
    spdc_observables_from_moments,
)
from setup_params import (
    crystal_length_m_from_setup,
    delta_k_qpm_per_m_from_setup,
    resolve_setup_selection,
    validate_spdc_energy_conservation,
)
from tools.taskZ6_c_lab_mask_position_signal_field import (
    build_setup,
    camera_orientation_90ccw_no_flip,
    crop_without_resize,
    load_rabbit_transmission,
    roi_spec,
    signal_extent_metrics,
)

FULL_N = 1024
SMOKE_N = 128
PIXEL_UM = 13.0
SEED = 1705
from tools.c_lab_constants import C_LAB_CRYSTAL_LENGTH_MM, C_LAB_CRYSTAL_LENGTH_M

# Production C_lab authoritative length (configs/setup_config_c_lab_v1.yaml).
L_MM = C_LAB_CRYSTAL_LENGTH_MM
T_C = 84.3
LAMBDA_P = 532.0
LAMBDA_S = 810.0
LAMBDA_I = 1550.0
RING_MIN_R = 20
RING_MIN_H = 0.05

Z6_DATA = ROOT / "evidence" / "taskZ6_c_lab_mask_position_signal_field" / "data"
Y1_MASK = ROOT / "evidence" / "taskY1_rabbit_ground_truth" / "data" / "frozen" / "mask_rabbit.npy"
Y1_META = ROOT / "evidence" / "taskY1_rabbit_ground_truth" / "data" / "frozen" / "rabbit_mask_metadata.json"
ROI_SPEC_PATH = (
    ROOT
    / "evidence"
    / "taskZ4_camera_image_formation"
    / "roi"
    / "QIUP_Mon_Sep_28_2026_13.07.41_roi_spec.json"
)

MONITORED_SOURCES = [
    ROOT / "optics.py",
    ROOT / "setup_params.py",
    ROOT / "latents.py",
    ROOT / "configs" / "setup_config_c_lab_v1.yaml",
    ROOT / "configs" / "setup_config_a.yaml",
    ROOT / "configs" / "setup_config_b.yaml",
    ROOT / "configs" / "setup_config_c.yaml",
    ROOT / "configs" / "setup_config_d.yaml",
]

_INTERFERE_ORIG = optics._interfere_from_moments
_GAUSSIAN_ORIG = optics.gaussian_filter


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_monitored_files() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in MONITORED_SOURCES:
        key = str(path.relative_to(ROOT)) if path.is_file() else str(path)
        out[key] = sha256_file(path) if path.is_file() else "MISSING"
    return out


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return str(value)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


@dataclass
class PilotSetupBundle:
    original_resolved: Any
    effective: Any
    magnification: float
    object_pitch_um: float
    energy_conservation_relative_residual: float
    override_audit: dict[str, Any]


def build_pilot_setup() -> PilotSetupBundle:
    original = resolve_setup_selection("C_lab")
    with patch("setup_params.validate_crystal_length_mm", side_effect=lambda x: float(x)):
        base, magnification, object_pitch_um = build_setup()
    effective = base.overridden(
        crystal_length_mm=L_MM,
        lambda_p_nm=LAMBDA_P,
        lambda_s_nm=LAMBDA_S,
        lambda_i_nm=LAMBDA_I,
        T_crystal_C=T_C,
        T_crystal_NL1_C=T_C,
        T_crystal_NL2_C=T_C,
        enable_lens_rings=False,
    )
    for interaction in (1, 2):
        length_m = crystal_length_m_from_setup(effective, interaction=interaction)
        if abs(length_m - C_LAB_CRYSTAL_LENGTH_M) > 1e-12:
            raise AssertionError(
                f"crystal length must be {L_MM} mm, got {length_m * 1000} mm ix={interaction}"
            )
    energy = float(validate_spdc_energy_conservation(effective))
    audit = {
        "pilot_only_overrides": {
            "crystal_length_mm": L_MM,
            "lambda_p_nm": LAMBDA_P,
            "lambda_s_nm": LAMBDA_S,
            "lambda_i_nm": LAMBDA_I,
            "T_crystal_C": T_C,
            "enable_lens_rings": False,
        },
        "build_setup_overrides_from_Z6_pattern": [
            "pump_waist_um=1000",
            "pump_power_laser_mW=250",
            "pump_pre_crystal_transmission=0.28",
            "pump_target_power_NL1_mW=70",
            "pump_target_power_NL2_mW=70",
            "pixel_um=13",
        ],
        "wavelength_energy_conservation": {
            "nominal_wavelengths_nm": [LAMBDA_P, LAMBDA_S, LAMBDA_I],
            "relative_residual_metadata_only": energy,
            "note": "Nominal 532/810/1550 retained; not energy-exact retuning.",
        },
    }
    return PilotSetupBundle(
        original_resolved=original,
        effective=effective,
        magnification=magnification,
        object_pitch_um=object_pitch_um,
        energy_conservation_relative_residual=energy,
        override_audit=audit,
    )


@dataclass
class DeltaKCalibrationState:
    mode: Literal["theory", "calibrated"]
    dk0_theory_ix1: float = 0.0
    dk0_theory_ix2: float = 0.0
    dk0_consumed_marginal: list[float] = field(default_factory=list)
    dk0_consumed_operating_point: list[float] = field(default_factory=list)


@contextmanager
def delta_k_calibration(
    state: DeltaKCalibrationState,
    *,
    theory_fn: Callable[..., float] | None = None,
) -> Iterator[None]:
    """Patch all Δk₀ consumers used by marginals, optics metadata, and latents."""

    if theory_fn is None:
        theory_fn = delta_k_qpm_per_m_from_setup

    if state.mode == "theory":

        def _wrap(setup, *, interaction=1, T_C=None, **kwargs):
            val = float(theory_fn(setup, interaction=interaction, T_C=T_C, **kwargs))
            state.dk0_consumed_marginal.append(val)
            return val

    else:
        offsets = {
            1: float(state.dk0_theory_ix1),
            2: float(state.dk0_theory_ix2),
        }

        def _wrap(setup, *, interaction=1, T_C=None, **kwargs):
            raw = float(theory_fn(setup, interaction=interaction, T_C=T_C, **kwargs))
            calibrated = raw - offsets[int(interaction)]
            state.dk0_consumed_marginal.append(calibrated)
            return calibrated

    with patch("setup_params.delta_k_qpm_per_m_from_setup", side_effect=_wrap):
        with patch("optics.delta_k_qpm_per_m_from_setup", side_effect=_wrap):
            yield


@contextmanager
def optical_capture(*, grid_n: int) -> Iterator[dict[str, Any]]:
    cap: dict[str, Any] = {"gaussian_inputs": []}

    def _interfere_hook(N1, N2, C12, theta=0.0, delta_phi=0.0):
        cap["N1"] = np.asarray(N1, dtype=np.float64).copy()
        cap["N2"] = np.asarray(N2, dtype=np.float64).copy()
        cap["C12"] = np.asarray(C12, dtype=np.complex128).copy()
        cap["theta"] = float(theta)
        cap["delta_phi"] = np.asarray(delta_phi, dtype=np.float64).copy()
        ip, im = _INTERFERE_ORIG(N1, N2, C12, theta=theta, delta_phi=delta_phi)
        cap["I_plus_pre_psf"] = np.asarray(ip, dtype=np.float64).copy()
        cap["I_minus_pre_psf"] = np.asarray(im, dtype=np.float64).copy()
        return ip, im

    def _gaussian_hook(input, sigma, *args, **kwargs):
        arr = np.asarray(input)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] == grid_n:
            cap["gaussian_inputs"].append(
                {"array": arr.astype(np.float64, copy=True), "sigma": float(sigma)}
            )
        return _GAUSSIAN_ORIG(input, sigma, *args, **kwargs)

    optics._interfere_from_moments = _interfere_hook
    optics.gaussian_filter = _gaussian_hook
    try:
        yield cap
    finally:
        optics._interfere_from_moments = _INTERFERE_ORIG
        optics.gaussian_filter = _GAUSSIAN_ORIG


def _radial_profile(img: np.ndarray, cx: float, cy: float) -> np.ndarray:
    yy, xx = np.indices(img.shape, dtype=np.float64)
    r = np.hypot(xx - cx, yy - cy).astype(np.int32)
    prof = np.zeros(int(r.max()) + 1, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    for ri in range(prof.size):
        m = r == ri
        if np.any(m):
            prof[ri] = float(a[m].mean())
    return prof


def _ring_metrics(prof: np.ndarray) -> dict[str, Any]:
    p = np.asarray(prof, dtype=np.float64)
    pk = float(p.max()) if p.size else 0.0
    rings: list[dict[str, float]] = []
    if pk > 0:
        n = p / pk
        for i in range(2, n.size - 1):
            if (
                n[i] >= n[i - 1]
                and n[i] >= n[i + 1]
                and i >= RING_MIN_R
                and n[i] >= RING_MIN_H
            ):
                rings.append({"radius_px": float(i), "height_norm": float(n[i])})
    return {
        "n_rings_ge_5pct": len(rings),
        "first_three_rings": rings[:3],
        "strongest_side_ring_height_norm": max((r["height_norm"] for r in rings), default=0.0),
    }


def _shape_normalize(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    bg = float(np.percentile(a, 1.0))
    a = np.clip(a - bg, 0.0, None)
    peak = float(a.max())
    if peak <= 0:
        return a
    return a / peak


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x = _shape_normalize(a).ravel()
    y = _shape_normalize(b).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den <= 0:
        return float("nan")
    return float(np.dot(x, y) / den)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    x = _shape_normalize(a).ravel()
    y = _shape_normalize(b).ravel()
    return float(np.sqrt(np.mean((x - y) ** 2)))


def phase_audit(optical: dict[str, Any], cap: dict[str, Any]) -> dict[str, Any]:
    vis = optical.get("vis") or {}
    c12 = np.asarray(cap["C12"], dtype=np.complex128)
    n1 = np.asarray(cap["N1"], dtype=np.float64)
    n2 = np.asarray(cap["N2"], dtype=np.float64)
    i0 = n1 + n2 + 2.0 * np.abs(c12)
    flat = int(np.argmax(i0))
    iy, ix = divmod(flat, i0.shape[1])
    theta = float(cap.get("theta", optical.get("theta", 0.0)))
    delta_phi = cap.get("delta_phi")
    delta_phi_center = float(np.asarray(delta_phi)[iy, ix]) if delta_phi is not None else 0.0
    total_phase = theta + delta_phi_center
    destructive_target = math.pi
    mod_diff = abs((total_phase - destructive_target + math.pi) % (2 * math.pi) - math.pi)
    return {
        "object_phase_rad_at_peak": 0.0,
        "idler_return_phase_rad": float(vis.get("idler_return_phase", 0.0)),
        "relative_interferometric_phase_rad": float(vis.get("relative_interferometric_phase", 0.0)),
        "opd_phase_rad": float(vis.get("opd_phase_rad", 0.0)),
        "theta_opd_carrier_rad": theta,
        "delta_phi_spatial_at_peak_rad": delta_phi_center,
        "total_interference_phase_at_peak_rad": total_phase,
        "destructive_pi_mod_2pi_residual_rad": mod_diff,
        "C12_abs_at_peak": float(np.abs(c12[iy, ix])),
        "arg_C12_at_peak_rad": float(np.angle(c12[iy, ix])),
    }


def verify_i_plus_equivalence(cap: dict[str, Any], optical: dict[str, Any]) -> dict[str, float]:
    n1, n2, c12 = cap["N1"], cap["N2"], cap["C12"]
    theta = float(cap["theta"])
    delta_phi = np.asarray(cap["delta_phi"], dtype=np.float64)
    recon, _ = _interfere_from_moments(n1, n2, c12, theta=theta, delta_phi=delta_phi)
    pre = np.asarray(cap["I_plus_pre_psf"], dtype=np.float64)
    err = float(np.max(np.abs(recon - pre)))
    return {"max_abs_error_reconstructed_vs_pre_psf_I_plus": err}


def run_forward_case(
    setup,
    ref: dict[str, Any],
    dk_state: DeltaKCalibrationState,
    *,
    grid_n: int,
) -> dict[str, Any]:
    state = quiet_state(setup, nbar=0.01, num_frames=1, enable_phase_steps=False)
    with delta_k_calibration(dk_state):
        with optical_capture(grid_n=grid_n) as cap:
            optical = optics._optical_intensity(ref, state, 0, np.random.default_rng(SEED))
        spdc = state.spdc_operating_point(relative_power_fluctuation=0.0)
        for item in spdc.get("interactions") or []:
            dk_state.dk0_consumed_operating_point.append(float(item["delta_k_qpm_per_m"]))
    scam = np.asarray(optical.get("S_cam"), dtype=np.float64)
    obs = spdc_observables_from_moments(cap["N1"], cap["N2"], cap["C12"])
    visibility = np.asarray(obs["visibility_50_50"], dtype=np.float64)
    return {
        "optical": optical,
        "cap": cap,
        "S_cam": scam,
        "I_plus": np.asarray(optical["I_plus"], dtype=np.float64),
        "I_minus": np.asarray(optical["I_minus"], dtype=np.float64),
        "I_plus_pre_psf": cap["I_plus_pre_psf"],
        "spdc_operating_point": spdc,
        "visibility_map": visibility,
        "phase": phase_audit(optical, cap),
        "i_plus_equiv": verify_i_plus_equivalence(cap, optical),
    }


def camera_pipeline(img_pre_orientation: np.ndarray) -> dict[str, np.ndarray]:
    oriented = camera_orientation_90ccw_no_flip(img_pre_orientation)
    roi = crop_without_resize(oriented)
    return {"full_oriented": oriented, "roi": roi}


def image_metrics(img: np.ndarray) -> dict[str, Any]:
    sig = signal_extent_metrics(np.asarray(img, dtype=np.float64), pixel_pitch_um=PIXEL_UM)
    prof = _radial_profile(img, float(sig["x0_px"]), float(sig["y0_px_top"]))
    rings = _ring_metrics(prof)
    return {**sig, **rings, "radial_profile": prof}


def quadrature_convergence_check(setup, *, grid_n: int = 512) -> dict[str, Any]:
    specs = ((256, 48, 48), (512, 72, 96), (768, 72, 96))
    fwhms: list[float] = []
    for n_r, n_p, n_theta in specs:
        out = far_field_signal_marginal_from_spdc(
            setup,
            (grid_n, grid_n),
            radial_samples=n_r,
            pump_radial_samples=n_p,
            angular_samples=n_theta,
            use_cache=False,
        )
        m = signal_extent_metrics(np.asarray(out["intensity"], dtype=np.float64), pixel_pitch_um=PIXEL_UM)
        fwhms.append(0.5 * (float(m["FWHM_x_px"]) + float(m["FWHM_y_px"])))
    spread = max(fwhms) - min(fwhms)
    converged = spread / max(min(fwhms), 1e-9) < 0.05
    return {
        "grid_n": grid_n,
        "fwhm_px_by_quadrature": fwhms,
        "relative_spread": spread / max(min(fwhms), 1e-9),
        "demonstrably_converged": bool(converged),
        "note": "Calibrated Δk₀ not applied in this sweep; physical inputs otherwise unchanged.",
    }


def evaluate_stage_a_gates(
    case_a0: dict[str, Any],
    case_a1: dict[str, Any],
    exp_metrics: dict[str, Any],
    sim_a1_roi_metrics: dict[str, Any],
    comparison: dict[str, Any],
    calibration_audit: dict[str, Any],
) -> dict[str, str]:
    dk_a1_meta = float(case_a1["optical"].get("spdc_physics", {}).get("delta_k_qpm_per_m", float("nan")))
    cal_ok = calibration_audit.get("CALIBRATION_APPLIED_CONSISTENTLY") is True and abs(dk_a1_meta) < 1e-3
    ring_a0 = int(case_a0["full_metrics"]["n_rings_ge_5pct"])
    ring_a1 = int(case_a1["full_metrics"]["n_rings_ge_5pct"])
    ring_pass = ring_a1 < ring_a0 and ring_a1 <= 1
    w_rel = abs(sim_a1_roi_metrics["Wsignal_px"] - exp_metrics["Wsignal_px"]) / exp_metrics["Wsignal_px"]
    h_rel = abs(sim_a1_roi_metrics["Hsignal_px"] - exp_metrics["Hsignal_px"]) / exp_metrics["Hsignal_px"]
    size_pass = w_rel <= 0.2 and h_rel <= 0.2
    ncc = float(comparison.get("ncc_shape_normalized", float("nan")))
    nrmse = float(comparison.get("nrmse_shape_normalized", float("nan")))
    profile_pass = ncc >= 0.65 and nrmse <= 0.35
    reg = "PASS" if comparison.get("roi_registration") == "ACQUISITION_VERIFIED" else "INCONCLUSIVE"

    def gate(ok: bool, inconclusive: bool = False) -> str:
        if inconclusive:
            return "INCONCLUSIVE"
        return "PASS" if ok else "FAIL"

    gates = {
        "CALIBRATION_APPLIED_CONSISTENTLY": gate(cal_ok),
        "NO_OBJECT_RING_REDUCTION": gate(ring_pass),
        "NO_OBJECT_SIZE_AGREEMENT": gate(size_pass, inconclusive=math.isnan(w_rel)),
        "NO_OBJECT_PROFILE_AGREEMENT": gate(profile_pass, inconclusive=math.isnan(ncc)),
        "NO_OBJECT_REGISTRATION_CONFIDENCE": reg,
    }
    if any(v == "FAIL" for v in gates.values()):
        overall = "FAIL"
    elif any(v == "INCONCLUSIVE" for v in gates.values()):
        overall = "INCONCLUSIVE"
    elif all(v == "PASS" for v in gates.values()):
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"
    gates["NO_OBJECT_VALIDATION_STATUS"] = overall
    gates["_numeric"] = {
        "ring_counts": {"A0": ring_a0, "A1": ring_a1},
        "width_height_relative_error": {"w": w_rel, "h": h_rel},
        "ncc": ncc,
        "nrmse": nrmse,
    }
    return gates


def write_baseline_plot(out_path: Path, a0: np.ndarray, a1: np.ndarray, prof_a0: np.ndarray, prof_a1: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    vmax = max(float(a0.max()), float(a1.max()), 1e-30)
    axes[0].imshow(a0, cmap="magma", origin="upper", vmin=0, vmax=vmax)
    axes[0].set_title("A0 theory Δk₀ — I_plus ROI")
    axes[1].imshow(a1, cmap="magma", origin="upper", vmin=0, vmax=vmax)
    axes[1].set_title("A1 calibrated Δk₀ — I_plus ROI")
    rmax = min(400, prof_a0.size, prof_a1.size)
    p0 = prof_a0[:rmax] / max(prof_a0.max(), 1e-30)
    p1 = prof_a1[:rmax] / max(prof_a1.max(), 1e-30)
    axes[2].plot(p0, label="A0")
    axes[2].plot(p1, "--", label="A1")
    axes[2].set_xlabel("radius px")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_exp_comparison_plot(path: Path, exp: np.ndarray, sim: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    exp_n = _shape_normalize(exp)
    sim_n = _shape_normalize(sim)
    vmax = 1.0
    axes[0].imshow(exp_n, cmap="magma", origin="upper", vmin=0, vmax=vmax)
    axes[0].set_title("Experiment (shape norm)")
    axes[1].imshow(sim_n, cmap="magma", origin="upper", vmin=0, vmax=vmax)
    axes[1].set_title("Sim A1 calibrated (shape norm)")
    axes[2].imshow(sim_n - exp_n, cmap="coolwarm", origin="upper")
    axes[2].set_title("Sim − Exp")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_pilot(*, smoke: bool = False, output_root: Path | None = None) -> Path:
    hashes_before = hash_monitored_files()
    grid_n = SMOKE_N if smoke else FULL_N
    bundle = build_pilot_setup()
    setup = bundle.effective

    dk1 = float(delta_k_qpm_per_m_from_setup(setup, interaction=1))
    dk2 = float(delta_k_qpm_per_m_from_setup(setup, interaction=2))

    out_dir = output_root or (ROOT / "evidence" / "c_lab_isolated_pilot" / utc_stamp())
    out_dir.mkdir(parents=True, exist_ok=True)

    from tools.z2.physics import ref_from_T

    ref = ref_from_T(np.ones((grid_n, grid_n), dtype=np.float32))
    ref["phi"] = np.zeros_like(ref["T"], dtype=np.float32)

    state_a0 = DeltaKCalibrationState(mode="theory", dk0_theory_ix1=dk1, dk0_theory_ix2=dk2)
    state_a1 = DeltaKCalibrationState(mode="calibrated", dk0_theory_ix1=dk1, dk0_theory_ix2=dk2)

    case_a0 = run_forward_case(setup, ref, state_a0, grid_n=grid_n)
    case_a1 = run_forward_case(setup, ref, state_a1, grid_n=grid_n)

    if grid_n == FULL_N:
        cam_a0 = camera_pipeline(case_a0["I_plus"])
        cam_a1 = camera_pipeline(case_a1["I_plus"])
        case_a0["full_metrics"] = image_metrics(cam_a0["full_oriented"])
        case_a1["full_metrics"] = image_metrics(cam_a1["full_oriented"])
        case_a0["roi_metrics"] = image_metrics(cam_a0["roi"])
        case_a1["roi_metrics"] = image_metrics(cam_a1["roi"])
        case_a0["cam"] = cam_a0
        case_a1["cam"] = cam_a1
    else:
        case_a0["full_metrics"] = image_metrics(case_a0["I_plus"])
        case_a1["full_metrics"] = image_metrics(case_a1["I_plus"])
        case_a0["roi_metrics"] = case_a0["full_metrics"]
        case_a1["roi_metrics"] = case_a1["full_metrics"]

    calibration_audit = {
        "reference_temperature_C": T_C,
        "delta_k0_theory_rad_per_m_interaction_1": dk1,
        "delta_k0_theory_rad_per_m_interaction_2": dk2,
        "delta_k0_calibrated_reference_rad_per_m": 0.0,
        "offsets_subtracted_rad_per_m": {"interaction_1": dk1, "interaction_2": dk2},
        "consumed_during_marginal_generation": {
            "A0": state_a0.dk0_consumed_marginal[:6],
            "A1": state_a1.dk0_consumed_marginal[:6],
        },
        "consumed_during_spdc_operating_point": {
            "A0": state_a0.dk0_consumed_operating_point,
            "A1": state_a1.dk0_consumed_operating_point,
        },
        "patch_targets": [
            "setup_params.delta_k_qpm_per_m_from_setup",
            "optics.delta_k_qpm_per_m_from_setup",
        ],
        "cache_policy": "use_cache=False for marginal convergence checks; forward path uses live Δk",
    }
    calibration_audit["CALIBRATION_APPLIED_CONSISTENTLY"] = (
        len(state_a1.dk0_consumed_operating_point) >= 2
        and all(abs(v) < 1e-6 for v in state_a1.dk0_consumed_operating_point)
        and abs(float(case_a1["optical"].get("spdc_physics", {}).get("delta_k_qpm_per_m", 1.0))) < 1e-3
    )

    exp_path = Z6_DATA / "experimental_no_object_reference_mean.npy"
    exp_available = exp_path.is_file() and grid_n == FULL_N
    comparison: dict[str, Any] = {"experimental_available": exp_available}
    gates: dict[str, Any] = {"smoke_run": smoke}
    stage_b: dict[str, Any] = {"executed": False, "reason": "Stage A gate not PASS"}

    if exp_available:
        exp = np.load(exp_path, mmap_mode="r")
        exp_arr = np.asarray(exp, dtype=np.float64)
        sim_roi = case_a1["cam"]["roi"]
        comparison.update(
            {
                "ncc_shape_normalized": _ncc(sim_roi, exp_arr),
                "nrmse_shape_normalized": _nrmse(sim_roi, exp_arr),
                "experimental_path": str(exp_path),
                "normalization": "1st percentile background subtract, peak normalize; shape only",
                "roi_spec_source": str(ROI_SPEC_PATH),
                "roi_registration": "PROVISIONAL",
                "roi_registration_note": "Integer ROI from Z4 spec; origin previously inferred not acquisition-verified (Z6 audit).",
            }
        )
        exp_metrics = signal_extent_metrics(exp_arr, pixel_pitch_um=PIXEL_UM)
        comparison["experimental_metrics"] = {k: v for k, v in exp_metrics.items() if k != "radial_profile"}
        gates = evaluate_stage_a_gates(case_a0, case_a1, exp_metrics, case_a1["roi_metrics"], comparison, calibration_audit)
        write_exp_comparison_plot(out_dir / "no_object_experiment_comparison.png", exp_arr, sim_roi)
        sim_roi_metrics = {k: v for k, v in case_a1["roi_metrics"].items() if k != "radial_profile"}
        save_json(out_dir / "no_object_metrics.json", {**comparison, "gates": gates, "sim_A1_roi": sim_roi_metrics})
    elif not smoke:
        gates = {
            "NO_OBJECT_VALIDATION_STATUS": "BLOCKED",
            "reason": "experimental_no_object_reference_mean.npy missing or not full resolution",
        }

    if grid_n == FULL_N:
        write_baseline_plot(
            out_dir / "baseline_theory_vs_calibrated.png",
            case_a0["cam"]["roi"],
            case_a1["cam"]["roi"],
            case_a0["full_metrics"]["radial_profile"],
            case_a1["full_metrics"]["radial_profile"],
        )
        rows = []
        for tag, prof in (("A0_theory", case_a0["full_metrics"]["radial_profile"]), ("A1_calibrated", case_a1["full_metrics"]["radial_profile"])):
            for r, val in enumerate(prof[:450]):
                rows.append({"case": tag, "radius_px": r, "mean_intensity": float(val)})
        with (out_dir / "baseline_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "radius_px", "mean_intensity"])
            writer.writeheader()
            writer.writerows(rows)

    qconv = {"skipped": smoke}
    if not smoke:
        with delta_k_calibration(DeltaKCalibrationState(mode="calibrated", dk0_theory_ix1=dk1, dk0_theory_ix2=dk2)):
            qconv = quadrature_convergence_check(setup)

    if gates.get("NO_OBJECT_VALIDATION_STATUS") == "PASS" and grid_n == FULL_N:
        rabbit_T, rabbit_meta = load_rabbit_transmission(bundle.object_pitch_um)
        ref_rb = ref_from_T(np.asarray(rabbit_T, dtype=np.float32))
        ref_rb["phi"] = np.zeros_like(ref_rb["T"], dtype=np.float32)
        rb = run_forward_case(setup, ref_rb, state_a1, grid_n=grid_n)
        cam_rb = camera_pipeline(rb["I_plus"]) if grid_n == FULL_N else {"roi": rb["I_plus"]}
        exp_rb_path = Z6_DATA / "experimental_rabbit_paired_reference_mean.npy"
        stage_b = {
            "executed": True,
            "propagation_model": "production architecture_object_transfer (reduced-order); intermediate-mask Fresnel round-trip NOT implemented (missing verified BCs).",
            "object_interpretation_audit": "YAML reflective end-mirror replacement vs physics-team transmissive double-pass — pilot uses production path only.",
            "rabbit_mask": {"path": str(Y1_MASK), "metadata": str(Y1_META), "resample_meta": rabbit_meta},
        }
        if exp_rb_path.is_file():
            exp_rb = np.asarray(np.load(exp_rb_path), dtype=np.float64)
            write_exp_comparison_plot(out_dir / "rabbit_experiment_comparison.png", exp_rb, cam_rb["roi"])
            delta = cam_rb["roi"] - case_a1["cam"]["roi"]
            stage_b["metrics"] = {
                "ncc_shape_normalized": _ncc(cam_rb["roi"], exp_rb),
                "nrmse_shape_normalized": _nrmse(cam_rb["roi"], exp_rb),
                "delta_I_l1": float(np.mean(np.abs(delta))),
                "paired_no_object_subtracted": True,
            }
            save_json(out_dir / "rabbit_metrics.json", stage_b)

    effective_config = {
        "setup_id": setup.setup_id,
        "architecture": setup.spdc_architecture,
        "imaging_mode": setup.imaging_mode,
        "wavelengths_nm": [setup.lambda_p_nm, setup.lambda_s_nm, setup.lambda_i_nm],
        "crystal_length_mm": setup.crystal_length_mm,
        "T_crystal_C": setup.T_crystal_C,
        "poling_period_um": setup.poling_period_um,
        "pump_waist_um": setup.pump_waist_um,
        "pixel_um": setup.pixel_um,
        "camera_filter_fwhm_nm": getattr(setup, "camera_filter_fwhm_nm", None),
        "interactions": {
            "1": {"pump_target_mW": setup.pump_target_power_NL1_mW},
            "2": {"pump_target_mW": setup.pump_target_power_NL2_mW},
        },
        "original_resolved_snapshot": {
            "crystal_length_mm": bundle.original_resolved.crystal_length_mm,
            "lambda_p_nm": bundle.original_resolved.lambda_p_nm,
        },
    }
    save_json(out_dir / "effective_config.json", effective_config)
    save_json(out_dir / "parameter_override_audit.json", bundle.override_audit)
    save_json(out_dir / "qpm_calibration_audit.json", calibration_audit)

    vis_support = case_a1["S_cam"] > 0.05 * max(float(case_a1["S_cam"].max()), 1e-30)
    visibility_report = {
        "formula": "V=2|C12|/(N1+N2)",
        "illuminated_support_mean": float(np.mean(case_a1["visibility_map"][vis_support])),
        "full_camera_mean_labeled_not_fringe_contrast": float(np.mean(case_a1["visibility_map"])),
        "mask_rules": "support from S_cam>5% peak; rabbit interior/exterior require Stage B",
    }

    hashes_after = hash_monitored_files()
    unchanged = hashes_before == hashes_after
    save_json(
        out_dir / "source_file_hashes_before_after.json",
        {"before": hashes_before, "after": hashes_after, "all_monitored_unchanged": unchanged},
    )

    test_results = {
        "smoke": smoke,
        "grid_n": grid_n,
        "stage_a_gates": gates,
        "stage_b": stage_b,
        "quadrature_convergence": qconv,
        "visibility": visibility_report,
        "spectral_filter_note": "810 nm 10 nm FWHM unchanged; spatial marginal monochromatic at nominal lambda_s (no new spectral integration module).",
        "S_cam_single_application": case_a0["optical"].get("qiup_spatial_transfer", {}).get("generation_intensity_camera") is not None,
    }
    save_json(out_dir / "test_results.json", test_results)

    production_patch_plan = [
        "setup_params.delta_k_qpm_per_m_from_setup: optional laboratory axial reference offset at T=84.3°C",
        "optics.far_field_signal_marginal_from_spdc: ensure Δk calibration propagates consistently (already uses delta_k_qpm_per_m_from_setup)",
        "latents.spdc_operating_point metadata: same Δk reference",
        "architecture_object_transfer / propagate_correlated_object_to_camera: transmissive double-pass intermediate mask if approved",
        "configs/setup_config_c_lab_v1.yaml: production freeze L=2 mm, Ms=1, Mi=2.0 (no L=10 mm baseline)",
    ]

    decision_lines = [
        f"# C_lab isolated pilot — {utc_stamp()}",
        "",
        "## Answers",
        f"1. Baseline 532/810/1550 nm, L={L_MM} mm, T={T_C}°C, pitch=13 µm: **{'YES' if not smoke else 'SMOKE ONLY'}** (pilot overrides on resolved C_lab).",
        f"2. Δk₀=0 at reference applied consistently: **{calibration_audit.get('CALIBRATION_APPLIED_CONSISTENTLY')}** "
        f"(theory ix1={dk1:.4f} rad/m; patched setup_params + optics).",
        f"3. Strong rings in I_plus reduced A0→A1: A0 rings={case_a0['full_metrics'].get('n_rings_ge_5pct')} A1 rings={case_a1['full_metrics'].get('n_rings_ge_5pct')}.",
        f"4. Calibrated no-object spot vs experiment: **{gates.get('NO_OBJECT_SIZE_AGREEMENT', 'N/A')}** / **{gates.get('NO_OBJECT_PROFILE_AGREEMENT', 'N/A')}**.",
        "5. Remaining discrepancies: see no_object_metrics.json and gates._numeric.",
        f"6. Rabbit vs experiment: **{'executed' if stage_b.get('executed') else 'skipped'}**.",
        "7. Intermediate-mask propagation: **NOT implemented in production**; rabbit validation limited to reduced-order model.",
        "8. Production touch list:",
    ]
    for item in production_patch_plan:
        decision_lines.append(f"   - {item}")
    decision_lines.append(f"9. Monitored source files byte-identical: **{unchanged}**")
    decision_lines.append("")
    decision_lines.append("## Proposed production patch plan (NOT APPLIED)")
    for item in production_patch_plan:
        decision_lines.append(f"- {item}")
    (out_dir / "FINAL_DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="C_lab isolated validation pilot")
    parser.add_argument("--smoke", action="store_true", help="Run low-resolution smoke check only")
    parser.add_argument("--output", type=str, default="", help="Optional output directory")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    path = run_pilot(smoke=args.smoke, output_root=out)
    print(str(path))


if __name__ == "__main__":
    main()
