"""A6.2-CAL: calibrated 1024 signal marginal, equal-pitch rabbit map, pupil bound.

The signal quadrature is a local copy of the physical far-field marginal.
Delta-k0 = 0 and L = 2 mm (config_c_lab_v1) are arguments of that local function. Production
modules are not edited and are not monkey-patched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion

from object_maps.c_lab_y1_rabbit_physical import load_c_lab_y1_object_map, resolved_aperture_mm
from setup_params import resolve_setup_selection
from optics import spdc_crystal_state
from setup_params import (
    delta_k_qpm_per_m_from_setup,
    resolved_pump_waist_um,
    resolved_spatial_mapping,
    spatial_object_um_to_camera_px,
)
from tools.c_lab_constants import C_LAB_CRYSTAL_LENGTH_M, C_LAB_CRYSTAL_LENGTH_MM
from tools.c_lab_isolated_pilot import build_pilot_setup

Y1_DIR = ROOT / "evidence" / "taskY1_rabbit_ground_truth" / "data" / "frozen"
Y1_MASK = Y1_DIR / "mask_rabbit.npy"
Y1_META = Y1_DIR / "rabbit_mask_metadata.json"
INVALID_CACHE_META = (
    ROOT
    / "evidence"
    / "taskZ6_signal_marginal_rabbit_fix"
    / "provenance"
    / "S_cam_cache_metadata.json"
)

N = 1024
PITCH_UM = 13.0
SPAN_MM = N * PITCH_UM * 1e-3
P_NATIVE_UM = 39.72758229284903
P_OLD_OBJECT_UM = 24.87654321
_C_LAB_SETUP = resolve_setup_selection("C_lab")
APERTURE_W_MM, APERTURE_H_MM = resolved_aperture_mm(_C_LAB_SETUP)
D_CLEAR_MM = 22.86
LEGACY_PUPIL_MM = 7.1265
A3_FWHM_PX_HISTORICAL = 152.0

FORMULA = (
    "P_s(q_s)=integral exp[-w_p^2|q_s+q_i|^2/2] "
    "sinc^2[L*Delta_k(q_s,q_i)/2] d^2q_i; "
    "Delta_k=Delta_k0-qp^2/(2*k_p)+qs^2/(2*k_s)+qi^2/(2*k_i); "
    "q_s=k_s_air*x_camera/(M_s*f_s)"
)

PRODUCTION = [
    ROOT / "optics.py",
    ROOT / "setup_params.py",
    ROOT / "simulate.py",
    ROOT / "latents.py",
    ROOT / "configs" / "setup_config_c_lab_v1.yaml",
    ROOT / "configs" / "setup_config_a.yaml",
    ROOT / "configs" / "setup_config_b.yaml",
    ROOT / "configs" / "setup_config_c.yaml",
    ROOT / "configs" / "setup_config_d.yaml",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    paths = list(PRODUCTION) + [INVALID_CACHE_META, Y1_META]
    if Y1_DIR.is_dir():
        paths.extend(p for p in Y1_DIR.iterdir() if p.is_file())
    for pattern in (
        "tools/c_lab_a5*.py",
        "tools/c_lab_a6*.py",
        "tests/test_c_lab_a5*.py",
        "tests/test_c_lab_a6*.py",
    ):
        paths.extend(ROOT.glob(pattern))
    return sorted(set(paths), key=lambda p: str(p).lower())


def hash_protected() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in protected_paths()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"repository": False, "error": str(exc)}
    if status.returncode != 0:
        return {"repository": False, "error": (status.stderr or status.stdout or "").strip()}
    return {"repository": True, "error": "", "porcelain": status.stdout}


def axis_um(n: int, pitch_um: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * pitch_um


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("RGB image required")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + image[i].tobytes() for i in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    )


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def measures(mask: np.ndarray, pitch_um: float, x_um: np.ndarray, y_um: np.ndarray) -> dict[str, float]:
    active = np.asarray(mask) >= 0.5
    rows, cols = np.where(active)
    if cols.size == 0:
        raise RuntimeError("empty transmission")
    col_index = np.arange(active.shape[1])
    row_index = np.arange(active.shape[0])
    row_max = np.where(active, col_index, -1).max(axis=1)
    row_min = np.where(active, col_index, active.shape[1]).min(axis=1)
    row_span = (row_max - row_min + 1)[active.any(axis=1)]
    col_max = np.where(active, row_index[:, None], -1).max(axis=0)
    col_min = np.where(active, row_index[:, None], active.shape[0]).min(axis=0)
    col_span = (col_max - col_min + 1)[active.any(axis=0)]
    return {
        "aperture_width_px": int(row_span.max()),
        "aperture_height_px": int(col_span.max()),
        "aperture_width_mm": float(row_span.max()) * pitch_um * 1e-3,
        "aperture_height_mm": float(col_span.max()) * pitch_um * 1e-3,
        "bbox_width_px": int(cols.max() - cols.min() + 1),
        "bbox_height_px": int(rows.max() - rows.min() + 1),
        "bbox_width_mm": float(cols.max() - cols.min() + 1) * pitch_um * 1e-3,
        "bbox_height_mm": float(rows.max() - rows.min() + 1) * pitch_um * 1e-3,
        "centroid_x_mm": float(np.asarray(x_um)[cols].mean()) * 1e-3,
        "centroid_y_mm": float(np.asarray(y_um)[rows].mean()) * 1e-3,
        "edge_margin_px": int(
            min(rows.min(), cols.min(), active.shape[0] - 1 - rows.max(), active.shape[1] - 1 - cols.max())
        ),
    }


def project_once(mask: np.ndarray, magnification: float) -> np.ndarray:
    """Inverse map: each camera pixel samples x_object = x_camera / M once."""
    n = int(mask.shape[0])
    center = 0.5 * (n - 1)
    scale = 1.0 / float(magnification)
    matrix = np.array([[scale, 0.0], [0.0, scale]], dtype=np.float64)
    offset = np.array([center * (1.0 - scale), center * (1.0 - scale)], dtype=np.float64)
    sampled = affine_transform(
        np.asarray(mask, dtype=np.float64),
        matrix,
        offset=offset,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return sampled >= 0.5


def outline(mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask) >= 0.5
    return active & ~binary_erosion(active)


def square_mask(x_um: np.ndarray, y_um: np.ndarray, center_mm: float, side_mm: float) -> np.ndarray:
    half = 0.5 * side_mm * 1.0e3
    return (np.abs(x_um[None, :] - center_mm * 1.0e3) <= half) & (np.abs(y_um[:, None]) <= half)


def half_max_width(values: np.ndarray) -> float:
    peak = float(np.max(values))
    half = 0.5 * peak
    above = np.flatnonzero(values >= half)
    if above.size == 0:
        return float("nan")
    i0 = int(above[0])
    i1 = int(above[-1])

    def edge(inside: int, outside: int) -> float:
        v_out = float(values[outside])
        v_in = float(values[inside])
        if v_in == v_out:
            return float(inside)
        return outside + (half - v_out) / (v_in - v_out)

    left = edge(i0, i0 - 1) if i0 > 0 else float(i0)
    right = edge(i1, i1 + 1) if i1 < values.size - 1 else float(i1)
    return float(right - left)


def radial_profile(image: np.ndarray) -> np.ndarray:
    n = image.shape[0]
    center = 0.5 * (n - 1)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    radius = np.rint(np.hypot(xx - center, yy - center)).astype(np.int32)
    max_r = int(radius.max())
    sums = np.bincount(radius.ravel(), weights=image.ravel(), minlength=max_r + 1)
    counts = np.bincount(radius.ravel(), minlength=max_r + 1)
    return sums / np.maximum(counts, 1)


def signal_widths(image: np.ndarray, pitch_um: float) -> dict[str, float]:
    row = 0.5 * (image[N // 2 - 1] + image[N // 2])
    col = 0.5 * (image[:, N // 2 - 1] + image[:, N // 2])
    radial = radial_profile(image)
    fwhm_x = half_max_width(row)
    fwhm_y = half_max_width(col)
    fwhm_r = 2.0 * half_max_width(radial)
    radius_10 = half_max_width_level(radial, 0.10)
    radius_01 = half_max_width_level(radial, 0.01)
    return {
        "fwhm_x_px": fwhm_x,
        "fwhm_y_px": fwhm_y,
        "fwhm_radial_diameter_px": fwhm_r,
        "fwhm_x_mm": fwhm_x * pitch_um * 1e-3,
        "fwhm_y_mm": fwhm_y * pitch_um * 1e-3,
        "fwhm_radial_diameter_mm": fwhm_r * pitch_um * 1e-3,
        "contour_10pct_diameter_px": 2.0 * radius_10,
        "contour_1pct_diameter_px": 2.0 * radius_01,
        "contour_10pct_diameter_mm": 2.0 * radius_10 * pitch_um * 1e-3,
        "contour_1pct_diameter_mm": 2.0 * radius_01 * pitch_um * 1e-3,
        "a3_historical_fwhm_diameter_px_on_512": A3_FWHM_PX_HISTORICAL,
        "a3_is_a_resizable_input": False,
    }


def half_max_width_level(values: np.ndarray, level: float) -> float:
    peak = float(np.max(values))
    threshold = level * peak
    above = np.flatnonzero(values >= threshold)
    if above.size == 0:
        return float("nan")
    i1 = int(above[-1])
    if i1 >= values.size - 1:
        return float(i1)
    v_in = float(values[i1])
    v_out = float(values[i1 + 1])
    if v_in == v_out:
        return float(i1)
    return i1 + (threshold - v_in) / (v_out - v_in)


def calibrated_radial_marginal(
    *,
    n_radial: int,
    q_s_max: float,
    waist_m: float,
    length_m: float,
    delta_k0: float,
    k_p: float,
    k_s: float,
    k_i: float,
    n_pump: int,
    n_theta: int,
) -> np.ndarray:
    """Local quadrature. The axial mismatch argument is the value that is integrated."""
    if abs(float(delta_k0)) > 0.0 or abs(float(length_m) - C_LAB_CRYSTAL_LENGTH_M) > 1e-15:
        raise RuntimeError(
            f"calibrated marginal refused consumed values dk0={delta_k0}, L={length_m}"
        )
    q_s = np.linspace(0.0, q_s_max, int(n_radial), dtype=np.float64)
    q_p = np.linspace(0.0, 6.0 / waist_m, int(n_pump), dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    cos_theta = np.cos(theta)[None, None, :]
    qp = q_p[None, :, None]
    qp2 = qp * qp
    pump = np.exp(-0.5 * (waist_m * q_p) ** 2)[None, :, None]
    profile = np.empty_like(q_s)
    batch = 64
    for start in range(0, q_s.size, batch):
        qs = q_s[start : start + batch, None, None]
        qs2 = qs * qs
        qi2 = qp2 + qs2 - 2.0 * qp * qs * cos_theta
        delta_k = delta_k0 - qp2 / (2.0 * k_p) + qs2 / (2.0 * k_s) + qi2 / (2.0 * k_i)
        phase = np.sinc((0.5 * length_m * delta_k) / np.pi) ** 2
        angular = 2.0 * np.pi * np.mean(pump * phase, axis=2)
        profile[start : start + batch] = np.trapz(angular * q_p[None, :], q_p, axis=1)
    peak = float(np.max(profile))
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("calibrated signal marginal is non-finite or zero")
    return np.clip(profile / peak, 0.0, None)


def sample_camera(profile: np.ndarray, q_s_max: float, q_per_pixel: float) -> np.ndarray:
    q_axis = np.linspace(0.0, q_s_max, profile.size, dtype=np.float64)
    center = 0.5 * (N - 1)
    yy, xx = np.indices((N, N), dtype=np.float64)
    q_camera = np.hypot(xx - center, yy - center) * q_per_pixel
    image = np.interp(q_camera.ravel(), q_axis, profile, left=profile[0], right=0.0).reshape(N, N)
    peak = float(np.max(image))
    return np.clip(image / peak, 0.0, None)


def run_a62_cal(output: Path | None = None) -> Path:
    t0 = time.perf_counter()
    hashes_before = hash_protected()
    git_before = git_snapshot()
    out = output if output is not None else ROOT / "evidence" / "c_lab_a6_2_calibrated_1024" / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)

    setup = build_pilot_setup().effective.overridden(object_map_pixel_um=PITCH_UM)
    crystal = spdc_crystal_state(setup, interaction=1)
    mapping = resolved_spatial_mapping(setup)
    waist_um = float(resolved_pump_waist_um(setup, interaction=1))
    theory_dk0 = float(delta_k_qpm_per_m_from_setup(setup, interaction=1))
    length_m = float(crystal["crystal_length_m"])
    delta_k0 = 0.0
    if abs(length_m - C_LAB_CRYSTAL_LENGTH_M) > 1e-12:
        raise RuntimeError(f"pilot crystal length consumed {crystal['crystal_length_m']}")
    if abs(float(setup.crystal_length_mm) - C_LAB_CRYSTAL_LENGTH_MM) > 1e-12:
        raise RuntimeError(f"setup crystal_length_mm consumed {setup.crystal_length_mm}")
    if abs(waist_um - 1000.0) > 1e-6:
        raise RuntimeError(f"pump waist consumed {waist_um}")
    if abs(float(setup.T_crystal_C) - 84.3) > 1e-6:
        raise RuntimeError(f"temperature consumed {setup.T_crystal_C}")
    if abs(float(setup.poling_period_um) - 9.675) > 1e-9:
        raise RuntimeError(f"poling consumed {setup.poling_period_um}")
    for name, expected in (("lambda_p_nm", 532.0), ("lambda_s_nm", 810.0), ("lambda_i_nm", 1550.0)):
        if abs(float(getattr(setup, name)) - expected) > 1e-9:
            raise RuntimeError(f"{name} consumed {getattr(setup, name)}")

    lambda_s_m = 810.0e-9
    lambda_p_m = 532.0e-9
    lambda_i_m = 1550.0e-9
    k_p = 2.0 * np.pi * float(crystal["n_pump"]) / lambda_p_m
    k_s = 2.0 * np.pi * float(crystal["n_signal"]) / lambda_s_m
    k_i = 2.0 * np.pi * float(crystal["n_idler"]) / lambda_i_m
    k_s_air = 2.0 * np.pi / lambda_s_m
    f_s_m = float(mapping["far_field_signal_focal_mm"]) * 1e-3
    f_i_m = float(mapping["far_field_idler_focal_mm"]) * 1e-3
    m_s = float(mapping["far_field_signal_relay_magnification"])
    m_i = float(mapping["far_field_idler_relay_magnification"])
    magnification = float(mapping["signed_object_to_camera_magnification"])
    s_index = magnification * PITCH_UM / PITCH_UM
    if abs(s_index - magnification) > 1e-12:
        raise RuntimeError(f"s_index {s_index} does not match M {magnification}")
    q_per_pixel = k_s_air * (PITCH_UM * 1e-6) / (abs(m_s) * f_s_m)
    q_formula = (2.0 * np.pi / lambda_s_m) * (PITCH_UM * 1e-6) / (abs(m_s) * f_s_m)
    if abs(q_per_pixel - q_formula) > 1e-9:
        raise RuntimeError("q_s camera mapping disagrees with k_s_air*x/(M_s*f_s)")
    corner = 0.5 * np.hypot(N - 1, N - 1)
    q_s_max = float(corner * q_per_pixel)

    profile = calibrated_radial_marginal(
        n_radial=N,
        q_s_max=q_s_max,
        waist_m=waist_um * 1e-6,
        length_m=length_m,
        delta_k0=delta_k0,
        k_p=k_p,
        k_s=k_s,
        k_i=k_i,
        n_pump=72,
        n_theta=96,
    )
    signal = sample_camera(profile, q_s_max, q_per_pixel)
    refined = calibrated_radial_marginal(
        n_radial=N * 2,
        q_s_max=q_s_max,
        waist_m=waist_um * 1e-6,
        length_m=length_m,
        delta_k0=delta_k0,
        k_p=k_p,
        k_s=k_s,
        k_i=k_i,
        n_pump=72,
        n_theta=96,
    )
    widths = signal_widths(signal, PITCH_UM)
    q_refined = np.linspace(0.0, q_s_max, refined.size)
    i_half = int(np.flatnonzero(refined >= 0.5)[-1])
    if i_half < refined.size - 1 and refined[i_half] != refined[i_half + 1]:
        frac = (0.5 - refined[i_half]) / (refined[i_half + 1] - refined[i_half])
        q_half = q_refined[i_half] + frac * (q_refined[i_half + 1] - q_refined[i_half])
    else:
        q_half = q_refined[i_half]
    refined_fwhm_px = 2.0 * float(q_half / q_per_pixel)
    widths["radial_refinement_fwhm_diameter_px"] = refined_fwhm_px
    widths["radial_refinement_minus_primary_px"] = refined_fwhm_px - widths["fwhm_radial_diameter_px"]

    x_um = axis_um(N, PITCH_UM)
    y_um = axis_um(N, PITCH_UM)
    loaded = load_c_lab_y1_object_map(_C_LAB_SETUP, mask_path=Y1_MASK, metadata_path=Y1_META)
    native = np.asarray(loaded["mask"], dtype=np.float64)
    src_x = np.asarray(loaded["x_um"], dtype=np.float64)
    src_y = np.asarray(loaded["y_um"], dtype=np.float64)
    from scipy.ndimage import map_coordinates

    xx, yy = np.meshgrid(x_um, y_um)
    col = (xx - float(src_x[0])) / float(src_x[1] - src_x[0])
    row = (yy - float(src_y[0])) / float(src_y[1] - src_y[0])
    object_mask = map_coordinates(native, [row, col], order=0, mode="constant", cval=0.0) >= 0.5
    camera_mask = project_once(object_mask, magnification)
    object_meas = measures(object_mask, PITCH_UM, x_um, y_um)
    camera_meas = measures(camera_mask, PITCH_UM, x_um, y_um)
    native_active = native >= 0.5
    src_xx, src_yy = np.meshgrid(src_x, src_y)
    half_span_um = 0.5 * SPAN_MM * 1e3
    outside_tx = (
        (np.abs(src_xx[native_active]) > half_span_um) | (np.abs(src_yy[native_active]) > half_span_um)
    )
    canvas_outside = bool(np.max(np.abs(src_x)) > half_span_um or np.max(np.abs(src_y)) > half_span_um)
    rabbit_fits = bool(int(outside_tx.sum()) == 0 and not canvas_outside and object_meas["edge_margin_px"] > 0)

    square = square_mask(x_um, y_um, 0.0, 2.0)
    square_cam = project_once(square, magnification)
    shifted = square_mask(x_um, y_um, 1.0, 2.0)
    shifted_cam = project_once(shifted, magnification)
    square_obj = measures(square, PITCH_UM, x_um, y_um)
    square_cam_meas = measures(square_cam, PITCH_UM, x_um, y_um)
    shift_px = measures(shifted_cam, PITCH_UM, x_um, y_um)["centroid_x_mm"] / (PITCH_UM * 1e-3)
    shift_code = float(spatial_object_um_to_camera_px(1000.0, setup))
    shift_analytic = magnification * 1000.0 / PITCH_UM
    s_old = magnification * P_OLD_OBJECT_UM / PITCH_UM
    wrong_effective_m = -0.5225806451612903 * PITCH_UM / P_OLD_OBJECT_UM

    code_m = float(mapping["signed_object_to_camera_magnification"])
    code_s = float(mapping["object_to_camera_index_scale"])
    code_camera = project_once(object_mask, code_m)
    disagree = int(np.count_nonzero(code_camera != camera_mask))
    ratio = camera_meas["aperture_width_px"] / max(object_meas["aperture_width_px"], 1)
    geometry_pass = bool(
        rabbit_fits
        and abs(object_meas["aperture_width_px"] - APERTURE_W_MM / (PITCH_UM * 1e-3)) <= 2
        and abs(object_meas["aperture_height_px"] - APERTURE_H_MM / (PITCH_UM * 1e-3)) <= 2
        and abs(camera_meas["aperture_width_px"] - abs(magnification) * APERTURE_W_MM / (PITCH_UM * 1e-3)) <= 2
        and abs(camera_meas["aperture_height_px"] - abs(magnification) * APERTURE_H_MM / (PITCH_UM * 1e-3)) <= 2
        and abs(code_m - magnification) < 1e-12
        and abs(code_s - s_index) < 1e-12
        and disagree == 0
        and abs(ratio - abs(magnification)) < 0.03
        and shift_analytic < 0
        and abs(square_obj["aperture_width_px"] - 154) <= 1
        and abs(square_cam_meas["aperture_width_px"] - 80) <= 1
        and abs(shift_px - shift_analytic) < 0.5
    )
    signal_generated = bool(signal.shape == (N, N) and abs(float(signal.max()) - 1.0) < 1e-12)
    common_scale_pass = bool(
        signal_generated
        and camera_mask.shape == signal.shape
        and abs(float(x_um[1] - x_um[0]) - PITCH_UM) < 1e-12
    )
    k_i_air = 2.0 * np.pi / lambda_i_m
    q_edge = (0.5 * D_CLEAR_MM * 1e-3) * k_i_air / f_i_m
    pupil = {
        "status": "PUPIL_ILLUMINATION_INCONCLUSIVE",
        "pupil_off": "ideal infinite aperture",
        "pupil_on_label": "CONSERVATIVE_CLEAR_APERTURE_LOWER_BOUND",
        "D_clear_mm": D_CLEAR_MM,
        "D_clear_is_measured_exact_diameter": False,
        "thorlabs_outer_diameter_mm": 25.4,
        "legacy_software_pupil_mm": LEGACY_PUPIL_MM,
        "legacy_software_pupil_used": False,
        "legacy_origin": "2*0.04751*75 mm, not the Thorlabs clear aperture",
        "matched_illumination_fields_available": False,
        "camera_signal_spots_invented": False,
        "geometric_rabbit_bbox_identical_off_and_on": True,
        "geometric_edge_camera_px": q_edge / q_per_pixel,
        "reason": (
            "No idler-lens propagation at D_clear=22.86 mm exists for the calibrated source weights. "
            "The 7.1265 mm A6.1 pupil result is a different aperture and is not substituted. "
            "The geometric camera rabbit is the same array for pupil OFF and ON."
        ),
        "on_axis_conditional": "NOT_COMPUTED",
        "a3_collected": "NOT_COMPUTED",
        "broad_source": "NOT_COMPUTED",
        "source_weighting_uncertainty_remains": True,
    }

    invalid_meta = json.loads(INVALID_CACHE_META.read_text(encoding="utf-8"))
    parameters = {
        "consumed_delta_k0_rad_per_m": delta_k0,
        "theory_delta_k0_not_used_rad_per_m": theory_dk0,
        "invalid_cache_delta_k0_rad_per_m": invalid_meta.get("delta_k0_per_m"),
        "invalid_cache_crystal_length_mm": invalid_meta.get("crystal_length_mm"),
        "invalid_1024_cache_loaded": False,
        "crystal_length_mm": length_m * 1e3,
        "crystal_length_m_from_spdc_crystal_state": float(crystal["crystal_length_m"]),
        "temperature_C": float(setup.T_crystal_C),
        "poling_period_um": float(setup.poling_period_um),
        "pump_waist_um": waist_um,
        "wavelengths_nm": {"pump": 532.0, "signal": 810.0, "idler": 1550.0},
        "refractive_index": {
            "n_pump": float(crystal["n_pump"]),
            "n_signal": float(crystal["n_signal"]),
            "n_idler": float(crystal["n_idler"]),
        },
        "f_s_mm": f_s_m * 1e3,
        "f_i_mm": f_i_m * 1e3,
        "relay_magnification_signal": m_s,
        "relay_magnification_idler": m_i,
        "M": magnification,
        "s_index": s_index,
        "s_index_matches_M": abs(s_index - magnification) < 1e-12,
        "existing_mapping_M": code_m,
        "existing_mapping_s_index": code_s,
        "object_grid_shape": [N, N],
        "camera_grid_shape": [N, N],
        "pixel_pitch_um": PITCH_UM,
        "native_photograph_pitch_um": P_NATIVE_UM,
        "span_mm": SPAN_MM,
        "axis_center_convention": "(i-(n-1)/2)*pitch_um",
        "q_per_pixel_per_m": q_per_pixel,
        "q_s_formula": "q_s=(2*pi/lambda_s_air)*x_camera/(M_s*f_s)",
        "source_formula": FORMULA,
        "normalization": "peak_one_after_idler_marginalization",
        "quadrature": {"radial_samples": N, "pump_radial_samples": 72, "angular_samples": 96},
        "rabbit_M_applied_to_S_cam": False,
        "old_grid_pitch_um": P_OLD_OBJECT_UM,
        "old_grid_correct_s_index": s_old,
        "forcing_s_minus_0_52258_on_old_grid_effective_M": wrong_effective_m,
        "old_grid_used": False,
        "nominal_mapping_validates_75mm_mask_plane": False,
        "production_monkeypatch": False,
        "production_integration": False,
        "test_2_started": False,
        "test_3_started": False,
    }
    ratios = {
        "rabbit_width_over_signal_fwhm_x": camera_meas["aperture_width_mm"] / widths["fwhm_x_mm"],
        "rabbit_height_over_signal_fwhm_y": camera_meas["aperture_height_mm"] / widths["fwhm_y_mm"],
        "rabbit_width_over_radial_fwhm": camera_meas["aperture_width_mm"] / widths["fwhm_radial_diameter_mm"],
    }
    measurements = {
        "object_rabbit": object_meas,
        "camera_rabbit": camera_meas,
        "rabbit_fits_without_cropping": rabbit_fits,
        "transmitting_pixels_outside_canvas": int(outside_tx.sum()),
        "native_canvas_outside_grid": canvas_outside,
        "mapping_pixel_disagreement": disagree,
        "aperture_width_ratio_camera_over_object": ratio,
        "square_object_px": square_obj["aperture_width_px"],
        "square_camera_px": square_cam_meas["aperture_width_px"],
        "shift_camera_px": shift_px,
        "shift_analytic_px": shift_analytic,
        "shift_existing_mapping_px": shift_code,
        "signal_widths": widths,
        "rabbit_to_signal_ratios": ratios,
        "signal_peak": float(signal.max()),
        "signal_shape": list(signal.shape),
        "i_plus_established": False,
        "fov_claimed_from_fwhm_alone": False,
        "runtime_s": time.perf_counter() - t0,
    }
    decisions = {
        "common_camera_coordinate_scale": "PASS" if common_scale_pass else "FAIL",
        "rabbit_geometry_and_mapping": "PASS" if geometry_pass else "FAIL",
        "calibrated_signal_array": "GENERATED" if signal_generated else "FAILED",
        "finite_pupil_effect": "INCONCLUSIVE",
        "full_observable_I_plus_established": "NO",
    }

    np.save(out / "S_CAM_CALIBRATED_1024.npy", signal.astype(np.float64))
    np.save(out / "RABBIT_OBJECT_1024.npy", object_mask.astype(np.float64))
    np.save(out / "RABBIT_GEOMETRIC_CAMERA_1024.npy", camera_mask.astype(np.float64))
    rgb = np.repeat(np.clip(signal * 255.0, 0.0, 255.0).astype(np.uint8)[..., None], 3, axis=2)
    rgb[outline(camera_mask)] = (255, 0, 0)
    write_rgb_png(out / "COMMON_CAMERA_SCALE_OVERLAY.png", rgb)
    save_full_field_figure(
        out / "OBJECT_VS_CAMERA_FULL_FIELD.png",
        object_mask,
        signal,
        camera_mask,
    )
    save_json(out / "EFFECTIVE_SOURCE_PARAMETERS.json", parameters)
    save_json(out / "RABBIT_SIGNAL_MEASUREMENTS.json", measurements)
    save_json(out / "PUPIL_OFF_ON_DECISION.json", pupil)
    hashes_after = hash_protected()
    save_json(
        out / "PROTECTED_SHA256.json",
        {
            "before": hashes_before,
            "after": hashes_after,
            "unchanged": hashes_before == hashes_after,
            "git": git_before,
            "git_after": git_snapshot(),
        },
    )
    write_decision(out, parameters, measurements, pupil, decisions)
    return out


def save_full_field_figure(
    path: Path,
    object_mask: np.ndarray,
    signal: np.ndarray,
    camera_mask: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = 0.5 * SPAN_MM
    extent = [-half, half, -half, half]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), constrained_layout=True)
    axes[0].imshow(object_mask, origin="upper", extent=extent, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].imshow(signal, origin="upper", extent=extent, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].contour(
        camera_mask,
        levels=[0.5],
        colors="red",
        linewidths=0.4,
        origin="upper",
        extent=extent,
    )
    for axis, title in (
        (axes[0], "Object grid, 13 µm, physical rabbit"),
        (axes[1], "Camera grid, 13 µm, S_cam and projected outline"),
    ):
        axis.set_xlim(-half, half)
        axis.set_ylim(-half, half)
        axis.set_aspect("equal")
        axis.set_xlabel("mm")
        axis.set_ylabel("mm")
        axis.set_title(title, fontsize=9)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_decision(
    out: Path,
    parameters: dict[str, Any],
    measurements: dict[str, Any],
    pupil: dict[str, Any],
    decisions: dict[str, str],
) -> None:
    obj = measurements["object_rabbit"]
    cam = measurements["camera_rabbit"]
    widths = measurements["signal_widths"]
    lines = [
        "# A6.2-CAL calibrated 1024 signal and rabbit scale",
        "",
        (
            f"Common camera scale: {decisions['common_camera_coordinate_scale']}. "
            f"Rabbit geometry: {decisions['rabbit_geometry_and_mapping']}. "
            f"Signal array: {decisions['calibrated_signal_array']}. "
            f"Finite pupil: {decisions['finite_pupil_effect']}. "
            f"I_plus established: {decisions['full_observable_I_plus_established']}."
        ),
        "",
        (
            f"Consumed source: Δk₀ = {parameters['consumed_delta_k0_rad_per_m']} rad/m, "
            f"L = {parameters['crystal_length_mm']} mm, T = {parameters['temperature_C']} C, "
            f"poling = {parameters['poling_period_um']} µm, waist = {parameters['pump_waist_um']} µm, "
            f"wavelengths 532/810/1550 nm. "
            f"Theory Δk₀ {parameters['theory_delta_k0_not_used_rad_per_m']:.6f} rad/m was not used. "
            f"The invalid 1024 cache (L = {parameters['invalid_cache_crystal_length_mm']} mm, "
            f"Δk₀ = {parameters['invalid_cache_delta_k0_rad_per_m']}) was not loaded."
        ),
        (
            f"M = {parameters['M']:.16f} and s_index = {parameters['s_index']:.16f}. "
            f"They match: {parameters['s_index_matches_M']}. "
            "S_cam is sampled from q_s = (2π/λ_s) x_camera / (M_s f_s). The rabbit magnification is not applied to S_cam."
        ),
        (
            f"Object aperture {obj['aperture_width_px']} × {obj['aperture_height_px']} px, "
            f"{obj['aperture_width_mm']:.4f} × {obj['aperture_height_mm']:.4f} mm. "
            f"Camera aperture {cam['aperture_width_px']} × {cam['aperture_height_px']} px, "
            f"{cam['aperture_width_mm']:.4f} × {cam['aperture_height_mm']:.4f} mm. "
            f"Rabbit fits without cropping: {measurements['rabbit_fits_without_cropping']}."
        ),
        (
            f"New S_cam FWHM: horizontal {widths['fwhm_x_px']:.2f} px ({widths['fwhm_x_mm']:.3f} mm), "
            f"vertical {widths['fwhm_y_px']:.2f} px ({widths['fwhm_y_mm']:.3f} mm), "
            f"radial diameter {widths['fwhm_radial_diameter_px']:.2f} px "
            f"({widths['fwhm_radial_diameter_mm']:.3f} mm). "
            f"One radial refinement changes that diameter by {widths['radial_refinement_minus_primary_px']:.3f} px. "
            f"10% contour diameter {widths['contour_10pct_diameter_px']:.1f} px; "
            f"1% contour diameter {widths['contour_1pct_diameter_px']:.1f} px. "
            "A3's 152 px result on a 512 grid is a historical reference, not a target. "
            "FWHM alone does not establish the physical field of view."
        ),
        (
            f"Rabbit-to-signal width ratio {measurements['rabbit_to_signal_ratios']['rabbit_width_over_signal_fwhm_x']:.3f}; "
            f"height ratio {measurements['rabbit_to_signal_ratios']['rabbit_height_over_signal_fwhm_y']:.3f}. "
            "The red outline is the geometric projection, not an I_plus silhouette."
        ),
        (
            f"2 mm square: {measurements['square_object_px']} object px to {measurements['square_camera_px']} camera px. "
            f"+1 mm shift: {measurements['shift_camera_px']:.3f} camera px "
            f"(analytic {measurements['shift_analytic_px']:.3f})."
        ),
        (
            f"On the unused {parameters['old_grid_pitch_um']} µm object grid, the correct s_index would be "
            f"{parameters['old_grid_correct_s_index']:.6f}. Forcing s = -0.52258 there would imply "
            f"M ≈ {parameters['forcing_s_minus_0_52258_on_old_grid_effective_M']:.5f}. That grid was not used."
        ),
        (
            f"Pupil status: {pupil['status']}. ON case is a conservative clear-aperture lower bound "
            f"D = {pupil['D_clear_mm']} mm, not a measured exact diameter and not the {LEGACY_PUPIL_MM} mm software default. "
            f"A collimated ray reaches that lower-bound edge near {pupil['geometric_edge_camera_px']:.1f} camera pixels. "
            "No matched ON/OFF illumination field was available, so no second signal spot was created. "
            "The geometric rabbit box is identical for pupil OFF and ON."
        ),
        "The nominal object-to-camera map does not validate optical transfer at 75 mm after the idler lens.",
        "Test 2 was not started. Test 3 was not started. Production files were not edited.",
        "",
    ]
    (out / "FINAL_DECISION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(run_a62_cal(Path(args.output) if args.output else None))


if __name__ == "__main__":
    main()
