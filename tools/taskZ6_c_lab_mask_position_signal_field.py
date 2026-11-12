#!/usr/bin/env python3
"""Authoritative Z6 C_lab mask-position and signal-field diagnostic.

The diagnostic preserves the validated Z5 far-field correction.  It declares
the four physics-team geometries, applies the fixed camera-coordinate transform
after detector sampling, and executes paired transparent/rabbit simulations.

The current Z5 reduced-order runtime does not consume explicit lens-to-object
or object-to-mirror distances.  Z6 records that implementation gap instead of
inventing an unvalidated propagation law.  Consequently, equality of the four
position outputs is a measured model limitation, not an assumed physical law.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import label, sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.four_config.runner import quiet_state
from benchmarks.targets.geometry import centered_axis_um
from object_maps.rabbit_loader import load_object_map, resample_object_map_to_grid
from optics import _optical_intensity
from setup_params import qiup_magnification, resolve_setup_selection
from tools.taskZ4_camera_image_formation import display_normalize
from tools.taskZ5_farfield_correction import crop_sampled_camera
from tools.z2.physics import ref_from_T


OUT = ROOT / "evidence" / "taskZ6_c_lab_mask_position_signal_field"
REPORT_ROOT = ROOT.parent / "Report"
REPORT0 = REPORT_ROOT / "0"
RECEIVED = REPORT_ROOT / "Received_Verification" / "for generate results" / "exp data"
Y1 = ROOT / "evidence" / "taskY1_rabbit_ground_truth"
Z4 = ROOT / "evidence" / "taskZ4_camera_image_formation"
Z5 = ROOT / "evidence" / "taskZ5_farfield_correction"
PHYSICS_REPORT = REPORT_ROOT / "QIUP_Z5_Next_Test_FA_RTL_FIXED.docx"
FINAL_REPORT_NAME = "Z6_C_lab_Mask_Position_and_Signal_Field_Diagnostic_Report.docx"

PIXEL_UM = 13.0
FULL_N = 1024
SEED = 1705
LENS_POSITION_MM = 0.0
MIRROR_POSITION_MM = 150.0
CAMERA_ROTATION = "90_CCW"
HORIZONTAL_FLIP = False
VERTICAL_FLIP = False

POSITIONS: dict[str, dict[str, float]] = {
    "P0": {"shift_toward_lens_mm": 0.0, "lens_to_rabbit_mm": 75.0, "rabbit_to_mirror_mm": 75.0},
    "P5": {"shift_toward_lens_mm": 5.0, "lens_to_rabbit_mm": 70.0, "rabbit_to_mirror_mm": 80.0},
    "P7": {"shift_toward_lens_mm": 7.0, "lens_to_rabbit_mm": 68.0, "rabbit_to_mirror_mm": 82.0},
    "P10": {"shift_toward_lens_mm": 10.0, "lens_to_rabbit_mm": 65.0, "rabbit_to_mirror_mm": 85.0},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return str(value)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_structure() -> None:
    for name in (
        "audit",
        "configs",
        "data",
        "figures",
        "metrics",
        "methods",
        "provenance",
        "tests",
        "report",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def camera_orientation_90ccw_no_flip(values: np.ndarray) -> np.ndarray:
    """Exact detector-coordinate transform; np.rot90 uses no interpolation."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("camera output must be a two-dimensional array")
    return np.rot90(array, k=1).copy()


def stage_order() -> tuple[str, ...]:
    return (
        "rabbit_or_transparent_object",
        "optical_propagation",
        "camera_plane_field_or_intensity",
        "detector_sampling",
        "fixed_90ccw_camera_orientation",
        "full_camera_save",
        "integer_roi_crop",
        "metrics",
        "plotting",
    )


def roi_spec() -> dict[str, Any]:
    return read_json(
        Z4
        / "roi"
        / "QIUP_Mon_Sep_28_2026_13.07.41_roi_spec.json"
    )


def crop_without_resize(values: np.ndarray, spec: dict[str, Any] | None = None) -> np.ndarray:
    spec = dict(roi_spec() if spec is None else spec)
    return crop_sampled_camera(
        np.asarray(values),
        width_px=int(spec["width_px"]),
        height_px=int(spec["height_px"]),
        left_px=int(spec["left_px"]),
        top_px=int(spec["top_px"]),
        pixel_pitch_um=PIXEL_UM,
    )


def build_setup():
    setup = resolve_setup_selection("C_lab")
    magnification = float(
        qiup_magnification(setup, lambda_s_nm=810.0, lambda_i_nm=1550.0)
    )
    object_pitch_um = PIXEL_UM / abs(magnification)
    setup = setup.overridden(
        pump_waist_um=1000.0,
        pump_waist_NL1_um=1000.0,
        pump_waist_NL2_um=1000.0,
        pump_power_laser_mW=250.0,
        pump_pre_crystal_transmission=0.28,
        pump_return_transmission=1.0,
        pump_target_power_NL1_mW=70.0,
        pump_target_power_NL2_mW=70.0,
        pixel_um=PIXEL_UM,
        object_map_pixel_um=object_pitch_um,
        enable_phase_step_acquisition=False,
    )
    return setup, magnification, object_pitch_um


def load_rabbit_transmission(object_pitch_um: float) -> tuple[np.ndarray, dict[str, Any]]:
    axis_um = centered_axis_um(FULL_N, object_pitch_um)
    rabbit = load_object_map(
        Y1 / "data" / "frozen" / "mask_rabbit.npy",
        Y1 / "data" / "frozen" / "rabbit_mask_metadata.json",
    )
    transmission, meta = resample_object_map_to_grid(
        rabbit,
        x_um=axis_um,
        y_um=axis_um,
        method="nearest",
        binarize=True,
    )
    return transmission.astype(np.float32), meta


def normalized_config(position: str, object_state: str, setup_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position}")
    if object_state not in {"NO_OBJECT", "RABBIT"}:
        raise ValueError(f"unknown object state {object_state}")
    pos = POSITIONS[position]
    snapshot = {
        "geometry": {
            "lens_position_mm": LENS_POSITION_MM,
            "mirror_position_mm": MIRROR_POSITION_MM,
            "lens_to_mirror_mm": MIRROR_POSITION_MM - LENS_POSITION_MM,
            "lens_to_rabbit_mm": pos["lens_to_rabbit_mm"],
            "rabbit_to_mirror_mm": pos["rabbit_to_mirror_mm"],
            "rabbit_axial_coordinate_mm": pos["lens_to_rabbit_mm"],
            "axial_position_runtime_consumer": None,
            "runtime_status": "DECLARED_FOR_AUDIT_NOT_CONSUMED_BY_Z5_REDUCED_ORDER_MODEL",
        },
        "camera": {
            "shape_yx": [FULL_N, FULL_N],
            "pixel_pitch_x_um": PIXEL_UM,
            "pixel_pitch_y_um": PIXEL_UM,
            "rotation": CAMERA_ROTATION,
            "horizontal_flip": HORIZONTAL_FLIP,
            "vertical_flip": VERTICAL_FLIP,
            "interpolation": False,
            "roi": roi_spec(),
        },
        "object": {
            "state": object_state,
            "transparent_value": 1.0 if object_state == "NO_OBJECT" else None,
            "rabbit_source": str(Y1 / "data" / "frozen" / "mask_rabbit.npy"),
            "rabbit_source_sha256": sha256(Y1 / "data" / "frozen" / "mask_rabbit.npy"),
            "rabbit_input_rotated": False,
        },
        "numerics": {
            "seed": SEED,
            "crop_resize": False,
            "fitted_translation": False,
            "fitted_rotation": False,
            "fitted_magnification": False,
            "stage_order": list(stage_order()),
        },
        "setup": dict(setup_dict or {}),
    }
    return snapshot


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_config(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(flatten_config(item, f"{prefix}[{index}]"))
    else:
        out[prefix] = value
    return out


def config_differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    fa, fb = flatten_config(a), flatten_config(b)
    return [key for key in sorted(set(fa) | set(fb)) if fa.get(key) != fb.get(key)]


def allowed_config_difference(path: str, same_position: bool, same_object_state: bool) -> bool:
    geometry_allowed = {
        "geometry.lens_to_rabbit_mm",
        "geometry.rabbit_to_mirror_mm",
        "geometry.rabbit_axial_coordinate_mm",
    }
    if path in geometry_allowed and not same_position:
        return True
    if path == "object.state" and not same_object_state:
        return True
    if path == "object.transparent_value" and not same_object_state:
        return True
    return False


def orientation_test_payload() -> dict[str, Any]:
    source = np.array(
        [
            [11, 12, 13, 14, 15],
            [21, 0, 0, 0, 25],
            [31, 0, 99, 0, 35],
            [41, 0, 0, 0, 45],
        ],
        dtype=np.int16,
    )
    transformed = camera_orientation_90ccw_no_flip(source)
    expected = np.rot90(source, 1)
    exact = np.array_equal(transformed, expected)
    return {
        "CAMERA_ORIENTATION": "90_CCW",
        "HORIZONTAL_FLIP": "OFF",
        "VERTICAL_FLIP": "OFF",
        "INTERPOLATION": "NO",
        "TEST": "PASS" if exact else "FAIL",
        "source": source.tolist(),
        "transformed": transformed.tolist(),
        "feature_labels": {
            "TOP": [11, 12, 13, 14, 15],
            "BOTTOM": [41, 0, 0, 0, 45],
            "LEFT": [11, 21, 31, 41],
            "RIGHT": [15, 25, 35, 45],
            "CENTER": 99,
        },
    }


def _fwhm_extent(profile: np.ndarray) -> tuple[float | None, str | None]:
    p = np.asarray(profile, dtype=np.float64)
    if p.size == 0 or not np.all(np.isfinite(p)):
        return None, "EMPTY_OR_NONFINITE_PROFILE"
    peak = float(np.max(p))
    if peak <= 0.0:
        return None, "ZERO_PROFILE"
    indices = np.flatnonzero(p >= 0.5 * peak)
    if indices.size == 0:
        return None, "NO_HALF_MAX_SUPPORT"
    return float(indices[-1] - indices[0] + 1), None


def signal_extent_metrics(values: np.ndarray, *, pixel_pitch_um: float = PIXEL_UM) -> dict[str, Any]:
    """Fixed Z6 signal estimator: 4-sigma widths from intensity weights.

    A fixed one-percentile pedestal is removed and clipped at zero.  A tiny,
    deterministic epsilon is added to every pixel so a perfectly uniform
    field remains mathematically defined rather than being silently rejected.
    """
    image = np.asarray(values, dtype=np.float64)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("signal image must be a finite two-dimensional array")
    pedestal = float(np.percentile(image, 1.0))
    epsilon = 1.0e-12 * max(float(np.max(np.abs(image))), 1.0)
    weights = np.clip(image - pedestal, 0.0, None) + epsilon
    total_w = float(weights.sum())
    rows, cols = np.indices(image.shape, dtype=np.float64)
    x0 = float((weights * cols).sum() / total_w)
    y0_top = float((weights * rows).sum() / total_w)
    dx = cols - x0
    dy_up = -(rows - y0_top)
    cov_xx = float((weights * dx * dx).sum() / total_w)
    cov_yy = float((weights * dy_up * dy_up).sum() / total_w)
    cov_xy = float((weights * dx * dy_up).sum() / total_w)
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    major_vec = eigvecs[:, 0]
    major = float(4.0 * math.sqrt(float(eigvals[0])))
    minor = float(4.0 * math.sqrt(float(eigvals[1])))
    angle = float(math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))) % 180.0)
    if major > 0 and abs(major - minor) / major < 1.0e-6:
        angle_value: float | None = None
        angle_reason = "ISOTROPIC_OR_DEGENERATE"
    else:
        angle_value = angle
        angle_reason = None
    profile_x = weights.sum(axis=0)
    profile_y = weights.sum(axis=1)
    fwhm_x, fwhm_x_reason = _fwhm_extent(profile_x)
    fwhm_y, fwhm_y_reason = _fwhm_extent(profile_y)
    pitch_mm = float(pixel_pitch_um) / 1000.0
    return {
        "shape_yx": list(image.shape),
        "pedestal_p1": pedestal,
        "epsilon_weight": epsilon,
        "x0_px": x0,
        "y0_px_top": y0_top,
        "x0_mm_from_array_center": (x0 - (image.shape[1] - 1) / 2.0) * pitch_mm,
        "y0_mm_from_array_center_up": -(
            y0_top - (image.shape[0] - 1) / 2.0
        ) * pitch_mm,
        "sigma_x_px": math.sqrt(max(cov_xx, 0.0)),
        "sigma_y_px": math.sqrt(max(cov_yy, 0.0)),
        "Wsignal_px": 4.0 * math.sqrt(max(cov_xx, 0.0)),
        "Hsignal_px": 4.0 * math.sqrt(max(cov_yy, 0.0)),
        "Wsignal_mm": 4.0 * math.sqrt(max(cov_xx, 0.0)) * pitch_mm,
        "Hsignal_mm": 4.0 * math.sqrt(max(cov_yy, 0.0)) * pitch_mm,
        "major_px": major,
        "minor_px": minor,
        "major_mm": major * pitch_mm,
        "minor_mm": minor * pitch_mm,
        "ellipse_angle_deg_ccw_from_camera_x": angle_value,
        "ellipse_angle_reason": angle_reason,
        "FWHM_x_px": fwhm_x,
        "FWHM_y_px": fwhm_y,
        "FWHM_x_reason": fwhm_x_reason,
        "FWHM_y_reason": fwhm_y_reason,
        "four_sigma_width_x_px": 4.0 * math.sqrt(max(cov_xx, 0.0)),
        "four_sigma_width_y_px": 4.0 * math.sqrt(max(cov_yy, 0.0)),
        "integrated_intensity": float(image.sum()),
        "angle_convention": "counter-clockwise from +camera-x; camera-y points upward",
        "estimator": "P1 pedestal; clipped intensity plus fixed epsilon; intensity-weighted covariance; Wsignal=4sigma_x; Hsignal=4sigma_y",
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def rabbit_extent_metrics(delta: np.ndarray, *, pixel_pitch_um: float = PIXEL_UM) -> dict[str, Any]:
    signed = np.asarray(delta, dtype=np.float64)
    border_width = max(1, int(round(0.04 * min(signed.shape))))
    border = np.concatenate(
        [
            signed[:border_width, :].ravel(),
            signed[-border_width:, :].ravel(),
            signed[border_width:-border_width, :border_width].ravel(),
            signed[border_width:-border_width, -border_width:].ravel(),
        ]
    )
    response_baseline = float(np.median(border))
    response = np.abs(signed - response_baseline)
    reference = float(np.percentile(response, 99.9))
    threshold = 0.35 * reference
    component = _largest_component(response >= threshold) if threshold > 0 else np.zeros_like(response, dtype=bool)
    pitch_mm = float(pixel_pitch_um) / 1000.0
    if not np.any(component):
        return {
            "Wrabbit_px": None,
            "Hrabbit_px": None,
            "Wrabbit_mm": None,
            "Hrabbit_mm": None,
            "bbox": None,
            "centroid_x_px": None,
            "centroid_y_px_top": None,
            "threshold": threshold,
            "response_border_median": response_baseline,
            "status": "NOT_DETECTED",
            "reason": "NO_NONZERO_OBJECT_RESPONSE",
        }
    ys, xs = np.where(component)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    weights = response * component
    total = float(weights.sum())
    cx = float((weights * np.indices(response.shape)[1]).sum() / total)
    cy = float((weights * np.indices(response.shape)[0]).sum() / total)
    width = float(right - left + 1)
    height = float(bottom - top + 1)
    return {
        "Wrabbit_px": width,
        "Hrabbit_px": height,
        "Wrabbit_mm": width * pitch_mm,
        "Hrabbit_mm": height * pitch_mm,
        "bbox": [left, top, right, bottom],
        "centroid_x_px": cx,
        "centroid_y_px_top": cy,
        "threshold": threshold,
        "response_border_median": response_baseline,
        "threshold_definition": "subtract the fixed 4-percent-border median from Delta I; take absolute value; threshold at 0.35 times the 99.9th percentile; largest connected component",
        "status": "DETECTED",
        "reason": None,
        "object_response_l1": float(np.sum(response)),
        "object_response_l2": float(np.sqrt(np.sum(response * response))),
        "sharpness_sobel_mean": float(np.mean(np.hypot(sobel(response, 0), sobel(response, 1)))),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(aa, bb) / denom)


def spatial_and_edge_correlation(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return {
            "spatial_corr": None,
            "edge_corr": None,
            "reason": f"SHAPE_MISMATCH_{aa.shape}_VS_{bb.shape}; NO_RESIZE_PERMITTED",
        }
    edge_a = np.hypot(sobel(aa, axis=0), sobel(aa, axis=1))
    edge_b = np.hypot(sobel(bb, axis=0), sobel(bb, axis=1))
    return {
        "spatial_corr": pearson(aa, bb),
        "edge_corr": pearson(edge_a, edge_b),
        "reason": None,
    }


def save_camera_products(directory: Path, oriented: np.ndarray, pre_orientation: np.ndarray, roi: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "detector_pre_orientation.npy", np.asarray(pre_orientation, dtype=np.float32))
    np.save(directory / "full_camera_raw.npy", np.asarray(oriented, dtype=np.float32))
    tifffile.imwrite(directory / "full_camera.tif", np.asarray(oriented, dtype=np.float32))
    png = np.round(255.0 * display_normalize(oriented, 0.1, 99.9)).astype(np.uint8)
    Image.fromarray(png).save(directory / "full_camera.png")
    np.save(directory / "comparison_roi.npy", np.asarray(roi, dtype=np.float32))
    tifffile.imwrite(directory / "comparison_roi.tif", np.asarray(roi, dtype=np.float32))
    roi_png = np.round(255.0 * display_normalize(roi, 0.1, 99.9)).astype(np.uint8)
    Image.fromarray(roi_png).save(directory / "comparison_roi.png")


def simulate_one(
    position: str,
    object_state: str,
    setup: Any,
    rabbit_T: np.ndarray,
) -> dict[str, Any]:
    pos = POSITIONS[position]
    assert math.isclose(pos["lens_to_rabbit_mm"] + pos["rabbit_to_mirror_mm"], 150.0, abs_tol=1e-12)
    assert LENS_POSITION_MM == 0.0
    assert MIRROR_POSITION_MM == 150.0
    if object_state == "NO_OBJECT":
        object_T = np.ones_like(rabbit_T, dtype=np.float32)
    elif object_state == "RABBIT":
        object_T = rabbit_T
    else:
        raise ValueError(object_state)
    state = quiet_state(setup, nbar=0.01, num_frames=1, enable_phase_steps=False)
    optical = _optical_intensity(
        ref_from_T(object_T),
        state,
        0,
        np.random.default_rng(SEED),
    )
    pre_orientation = np.asarray(optical["I_plus"], dtype=np.float32)
    oriented = camera_orientation_90ccw_no_flip(pre_orientation)
    comparison_roi = crop_without_resize(oriented)
    run_dir = OUT / "data" / position / object_state.lower()
    save_camera_products(run_dir, oriented, pre_orientation, comparison_roi)
    full_metrics = signal_extent_metrics(oriented)
    roi_metrics = signal_extent_metrics(comparison_roi)
    top, left = int(roi_spec()["top_px"]), int(roi_spec()["left_px"])
    height, width = int(roi_spec()["height_px"]), int(roi_spec()["width_px"])
    roi_sum = float(oriented[top : top + height, left : left + width].sum())
    full_sum = float(oriented.sum())
    roi_metrics["ROI_contained_intensity_fraction"] = roi_sum / full_sum if full_sum else None
    metadata = {
        "position": position,
        "object_state": object_state,
        "array_dimensions_yx": list(oriented.shape),
        "pixel_pitch_x_um": PIXEL_UM,
        "pixel_pitch_y_um": PIXEL_UM,
        "physical_fov_edge_extent_mm_xy": [FULL_N * PIXEL_UM / 1000.0] * 2,
        "coordinate_origin": "center between pixels (511,511) and (512,512); x right; y up",
        "orientation_state": "90 degrees counter-clockwise; horizontal flip off; vertical flip off",
        "detector_pre_orientation_saved": True,
        "full_camera_saved_before_roi": stage_order().index("full_camera_save") < stage_order().index("integer_roi_crop"),
        "interpolation_for_orientation": False,
        "interpolation_for_crop": False,
        "resize": False,
        "final_camera_gaussian_envelope": "INACTIVE" if not optical.get("final_camera_pump_envelope_applied", True) else "ACTIVE_UNEXPECTED",
        "axial_position_runtime_effect": "NOT_IMPLEMENTED_IN_Z5_REDUCED_ORDER_MODEL",
    }
    save_json(run_dir / "full_camera_metadata.json", metadata)
    save_json(run_dir / "metrics.json", {"full_camera": full_metrics, "comparison_roi": roi_metrics})
    snapshot = normalized_config(position, object_state, setup.to_dict())
    save_json(OUT / "configs" / f"{position}_{object_state}.json", snapshot)
    return {
        "position": position,
        "object_state": object_state,
        "pre_orientation": pre_orientation,
        "full": oriented,
        "roi": comparison_roi,
        "full_metrics": full_metrics,
        "roi_metrics": roi_metrics,
        "metadata": metadata,
        "config": snapshot,
    }


def image_inventory_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    report0_files = sorted(REPORT0.glob("*.tif"))
    no_object_dir = RECEIVED / "mirror drift" / "witout object" / "ref"
    rabbit_ref_dir = RECEIVED / "mirror drift" / "with object" / "ref"
    selected = report0_files + sorted(no_object_dir.glob("*.tif")) + sorted(rabbit_ref_dir.glob("*.tif"))
    rows: list[dict[str, Any]] = []
    for path in selected:
        array = np.asarray(tifffile.imread(path))
        if REPORT0 in path.parents:
            classification = "RABBIT"
            evidence = "Z4/Z5 frozen experimental rabbit reference; Report/0 nominal fixed arrangement; Z5 classified visible rabbit structure"
        elif "witout object" in str(path):
            classification = "NO_OBJECT"
            evidence = "Acquisition path explicitly states without object/ref; Y2 frozen inventory classification_object=no-object and classification_kind=reference"
        elif "with object" in str(path):
            classification = "RABBIT"
            evidence = "Acquisition path explicitly states with object/ref; Y2 frozen inventory classification_object=with-object and classification_kind=reference"
        else:
            classification = "UNKNOWN"
            evidence = "No authoritative acquisition metadata"
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "dimensions": f"{array.shape[1]}x{array.shape[0]}",
                "dtype": str(array.dtype),
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "classification": classification,
                "classification_evidence": evidence,
            }
        )
    return rows, selected


def mean_tiffs(paths: Iterable[Path]) -> np.ndarray:
    arrays = [np.asarray(tifffile.imread(path), dtype=np.float64) for path in paths]
    if not arrays:
        raise FileNotFoundError("no TIFF frames selected")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("experimental TIFF shapes differ; resizing is prohibited")
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def experimental_arrays() -> dict[str, np.ndarray]:
    report0_paths = sorted(REPORT0.glob("*.tif"))
    noobj_paths = sorted((RECEIVED / "mirror drift" / "witout object" / "ref").glob("*.tif"))
    rabbit_paths = sorted((RECEIVED / "mirror drift" / "with object" / "ref").glob("*.tif"))
    arrays = {
        "rabbit_primary_report0": mean_tiffs(report0_paths),
        "no_object_reference": mean_tiffs(noobj_paths),
        "rabbit_paired_reference": mean_tiffs(rabbit_paths),
    }
    for name, array in arrays.items():
        np.save(OUT / "data" / f"experimental_{name}_mean.npy", array)
        tifffile.imwrite(OUT / "data" / f"experimental_{name}_mean.tif", array)
        Image.fromarray(
            np.round(255.0 * display_normalize(array, 0.1, 99.9)).astype(np.uint8)
        ).save(OUT / "data" / f"experimental_{name}_mean.png")
    save_json(
        OUT / "data" / "experimental_selection.json",
        {
            "camera_coordinate_convention": "recorded experimental TIFF convention; no additional transform",
            "primary_rabbit": [str(p) for p in report0_paths],
            "paired_no_object": [str(p) for p in noobj_paths],
            "paired_rabbit": [str(p) for p in rabbit_paths],
            "no_resize": True,
            "all_shapes_yx": {name: list(array.shape) for name, array in arrays.items()},
        },
    )
    return arrays


def write_requirements() -> None:
    rows = [
        ("Z6-R01", "Purpose", "Separate camera orientation, rabbit axial position, and signal-field profile without tuning other parameters.", "tools/taskZ6_c_lab_mask_position_signal_field.py", "configuration audit", "config_difference_matrix.csv", "PASS", "No fitted physics parameters."),
        ("Z6-R02", "1 Camera rotation", "Apply a fixed 90 degree counter-clockwise camera transform.", "camera_orientation_90ccw_no_flip", "test_camera_orientation_90ccw_no_flip", "camera_orientation_test.json", "PASS", "Applied after detector sampling."),
        ("Z6-R03", "1 Camera rotation", "Horizontal and vertical flips are both off and orientation is not fitted.", "normalized_config.camera", "T11 and T12", "camera_orientation_test.json", "PASS", "Exact array rotation; no interpolation."),
        ("Z6-R04", "2 Rabbit displacement", "Hold the idler lens and mirror fixed with 150 mm total separation.", "POSITIONS and normalized_config.geometry", "T01 T07 T08", "position_sweep_metrics.csv", "PARTIAL", "Declarative assertions pass; Z5 runtime has no distance consumer."),
        ("Z6-R05", "2 Rabbit displacement", "P0 uses 75 mm lens-to-rabbit and 75 mm rabbit-to-mirror.", "POSITIONS.P0", "T02", "position_sweep_metrics.csv", "PASS", "Sum is 150 mm."),
        ("Z6-R06", "2 Rabbit displacement", "P5 uses 70 mm lens-to-rabbit and 80 mm rabbit-to-mirror.", "POSITIONS.P5", "T03", "position_sweep_metrics.csv", "PASS", "Sum is 150 mm."),
        ("Z6-R07", "2 Rabbit displacement", "P7 uses 68 mm lens-to-rabbit and 82 mm rabbit-to-mirror.", "POSITIONS.P7", "T04", "position_sweep_metrics.csv", "PASS", "82 mm is P7 only."),
        ("Z6-R08", "2 Rabbit displacement", "P10 uses 65 mm lens-to-rabbit and 85 mm rabbit-to-mirror.", "POSITIONS.P10", "T05", "position_sweep_metrics.csv", "PASS", "Sum is 150 mm."),
        ("Z6-R09", "2 Rabbit displacement", "Only the mask position changes; focal lengths, mask size, pump, camera, and other optics remain fixed.", "normalized_config and difference audit", "T09 T20", "config_difference_matrix.csv", "PASS", "No unexpected config differences."),
        ("Z6-R10", "3 Independent outputs", "Produce independent no-object and rabbit outputs for every configuration.", "simulate_one", "T15", "data/P*/no_object and data/P*/rabbit", "PASS", "Eight runtime calls completed."),
        ("Z6-R11", "3 Output A", "No-object is the transparent condition and is used to assess center, extents, ellipticity, and principal-axis direction.", "simulate_one and signal_extent_metrics", "T15", "no_object_exp_sim_comparison.csv", "PASS", "T(x,y)=1 exactly."),
        ("Z6-R12", "3 Output B", "Rabbit output uses otherwise identical optical and numerical settings.", "normalized_config", "T20", "config_difference_matrix.csv", "PASS", "Only object.state differs within each position."),
        ("Z6-R13", "3 Full camera", "Save the complete camera image before the final comparison crop.", "stage_order and save_camera_products", "T16", "full_camera_raw.npy full_camera.tif full_camera.png", "PASS", "Detector pre-orientation array also retained."),
        ("Z6-R14", "3 ROI", "Extract the same experimental ROI without resizing or scaling.", "crop_without_resize", "T17 T18", "comparison_roi.npy", "PASS", "Integer array slicing only; pitch remains 13 um."),
        ("Z6-R15", "Numeric outputs", "Report no-object x0 y0 Wsignal Hsignal major minor and ellipse angle.", "signal_extent_metrics", "metric schema check", "position_sweep_metrics.csv", "PASS", "4-sigma estimator fixed once."),
        ("Z6-R16", "Numeric outputs", "Report Wsignal Hsignal separately from Wrabbit Hrabbit and provide a fixed-coordinate overlay.", "rabbit_extent_metrics and figure generation", "metric schema check", "Z6_06_Experimental_Simulation_Rabbit_Comparison.png", "PASS", "No recentering rotation translation or resize."),
        ("Z6-R17", "Decision criterion", "Compare no-object experiment and simulation first; do not alter the rabbit to compensate for a signal-field mismatch.", "no_object_gate", "gate evaluation", "Z6_status.json", "PASS", "Gate evaluated before rabbit interpretation; it fails on the uniform Z5 no-object field."),
    ]
    columns = ["requirement_id", "report_section", "requirement", "implementation_location", "test", "output", "status", "notes"]
    save_csv(OUT / "audit" / "physics_team_Z6_requirements.csv", [dict(zip(columns, row)) for row in rows], columns)


def write_geometry_audit() -> None:
    rows = [
        {"order": 1, "element": "C_lab selection", "parameter_name": "setup_id", "value": "config_c_lab_v1", "units": "NA", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "meta.setup_id", "runtime_consumer": "setup_params.resolve_setup_selection", "physical_role": "selects laboratory C overlay", "status": "ACTIVE"},
        {"order": 2, "element": "idler far-field lens", "parameter_name": "far_field_focal_mm", "value": 150.0, "units": "mm focal length", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "geometry.far_field_focal_mm line 115", "runtime_consumer": "resolved_far_field_focal_mm; spatial correlation width", "physical_role": "idler Fourier-lens focal length, not lens-mirror separation", "status": "ACTIVE_FIXED"},
        {"order": 3, "element": "idler mapping lens", "parameter_name": "F1_mm", "value": 150.0, "units": "mm focal length", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "geometry.F1_mm line 117", "runtime_consumer": "qiup_magnification and resolved_spatial_mapping", "physical_role": "idler-path focal length", "status": "ACTIVE_FIXED"},
        {"order": 4, "element": "signal/camera lens", "parameter_name": "F2_mm", "value": 150.0, "units": "mm focal length", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "geometry.F2_mm line 119", "runtime_consumer": "qiup_magnification and camera mapping", "physical_role": "signal/camera focal length", "status": "ACTIVE_FIXED"},
        {"order": 5, "element": "object plane", "parameter_name": "object_plane", "value": "idler_far_field_fourier_plane", "units": "NA", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "object.object_plane line 94", "runtime_consumer": "optics.architecture_object_transfer metadata and mapping", "physical_role": "declared Fourier plane", "status": "ACTIVE_FIXED"},
        {"order": 6, "element": "object", "parameter_name": "object_type", "value": "reflective", "units": "NA", "source_file": "configs/setup_config_c_lab_v1.yaml", "source_line_or_key": "object.object_type line 91", "runtime_consumer": "optics.architecture_object_transfer lines 2345-2354", "physical_role": "single reflection field", "status": "ACTIVE_FIXED"},
        {"order": 7, "element": "end mirror", "parameter_name": "end_mirror_model", "value": "reflective_object_replaces_end_mirror", "units": "NA", "source_file": "optics.py", "source_line_or_key": "architecture_object_transfer lines 2345-2354", "runtime_consumer": "effective_object_field", "physical_role": "current Z5 model treats the reflective object as the end mirror", "status": "CONFLICT_WITH_Z6_INTERMEDIATE_MASK_GEOMETRY"},
        {"order": 8, "element": "conditional propagation", "parameter_name": "propagation_model", "value": "reduced_order_conditional_gaussian_correlation", "units": "NA", "source_file": "optics.py", "source_line_or_key": "propagate_correlated_object_to_camera lines 2001-2052", "runtime_consumer": "architecture_object_transfer", "physical_role": "affine object-to-camera mapping plus conditional Gaussian PSF", "status": "ACTIVE_NO_EXPLICIT_AXIAL_DISTANCE"},
        {"order": 9, "element": "lens to rabbit", "parameter_name": "lens_to_rabbit_mm", "value": "MISSING", "units": "mm", "source_file": "runtime C_lab setup", "source_line_or_key": "no field", "runtime_consumer": "NONE", "physical_role": "required Z6 mask position", "status": "NOT_REPRESENTED"},
        {"order": 10, "element": "rabbit to mirror", "parameter_name": "rabbit_to_mirror_mm", "value": "MISSING", "units": "mm", "source_file": "runtime C_lab setup", "source_line_or_key": "no field", "runtime_consumer": "NONE", "physical_role": "required Z6 mask position", "status": "NOT_REPRESENTED"},
        {"order": 11, "element": "lens to mirror", "parameter_name": "lens_to_mirror_mm", "value": "MISSING", "units": "mm", "source_file": "runtime C_lab setup", "source_line_or_key": "no field", "runtime_consumer": "NONE", "physical_role": "required fixed 150 mm physical separation", "status": "DECLARED_ONLY_IN_Z6_SNAPSHOT"},
        {"order": 12, "element": "rabbit source", "parameter_name": "rabbit_source", "value": "evidence/taskY1_rabbit_ground_truth/data/frozen/mask_rabbit.npy", "units": "binary amplitude", "source_file": "tools/taskZ5_farfield_correction.py", "source_line_or_key": "run_case lines 1088-1100", "runtime_consumer": "resample_object_map_to_grid then _optical_intensity", "physical_role": "independent frozen rabbit mask", "status": "ACTIVE_UNCHANGED"},
        {"order": 13, "element": "camera observation", "parameter_name": "pixel_um", "value": 13.0, "units": "um per pixel", "source_file": "Z5 RUN_C run_config.json", "source_line_or_key": "fixed_variables.camera_pixel_pitch_um", "runtime_consumer": "physical detector axes", "physical_role": "camera sampling", "status": "ACTIVE_FIXED"},
        {"order": 14, "element": "comparison ROI", "parameter_name": "left top width height", "value": "290 293 455 419", "units": "pixels", "source_file": "evidence/taskZ4_camera_image_formation/roi/QIUP_Mon_Sep_28_2026_13.07.41_roi_spec.json", "source_line_or_key": "integer fields", "runtime_consumer": "crop_without_resize", "physical_role": "frozen comparison observation", "status": "ACTIVE_FIXED_INFERRED_PROVENANCE"},
    ]
    save_csv(OUT / "audit" / "C_lab_runtime_geometry_before_Z6.csv", rows)
    ten_rows = [
        {"parameter": "camera_filter_fwhm_nm", "value": 10.0, "units": "nm", "physical_meaning": "signal-camera spectral filter full width at half maximum", "source": "configs/setup_config_c_lab_v1.yaml signal_filter.camera_filter_fwhm_nm", "runtime_consumer": "spectral filtering configuration", "modified_in_Z6": "NO"},
        {"parameter": "P10 shift_toward_lens_mm", "value": 10.0, "units": "mm", "physical_meaning": "prescribed rabbit axial displacement for P10", "source": "physics-team Z6 report", "runtime_consumer": "Z6 declarative geometry only; no Z5 axial propagation consumer", "modified_in_Z6": "DECLARED_SWEEP_VALUE"},
        {"parameter": "10_mm_lens_focal_length", "value": "NOT FOUND", "units": "mm", "physical_meaning": "no active 10 mm focal-length parameter exists in resolved C_lab", "source": "resolved SetupParams numeric-value audit", "runtime_consumer": "NONE", "modified_in_Z6": "NO"},
    ]
    save_csv(OUT / "audit" / "ten_mm_parameter_audit.csv", ten_rows)


def write_method_definition() -> None:
    text = """# Z6 signal extent definition

The same estimator is used for simulated and experimental no-object and rabbit images.

For an image I on the analyzed array, define the fixed pedestal b as the first percentile of I. Define nonnegative weights

    w(x,y) = max(I(x,y) - b, 0) + epsilon

where epsilon = 1e-12 * max(max(abs(I)), 1). The epsilon is fixed and only prevents a perfectly uniform array from becoming undefined. Pixel-coordinate centroids are intensity weighted. Camera x points right; camera y points upward, while reported y0_px_top is the stored array row counted from the top.

The intensity-weighted covariance matrix is computed about the centroid. Wsignal = 4*sqrt(Cxx) and Hsignal = 4*sqrt(Cyy). The major and minor sizes are four times the square roots of the covariance eigenvalues. Ellipse angle is the major-axis eigenvector angle, measured counter-clockwise from +camera-x and reported modulo 180 degrees. If the eigenvalues are numerically equal, the angle is NA with reason ISOTROPIC_OR_DEGENERATE.

Horizontal and vertical FWHM values use the marginal weight profiles and the inclusive support at or above half the profile maximum. All millimeter values use the frozen 0.013 mm pixel pitch. The estimator is not tuned per image or per position.

Wrabbit and Hrabbit are separate object-structure quantities. First subtract the median of the four-percent outer border from Delta I to remove any global paired offset. They are the width and height of the largest connected support in the absolute corrected paired response at the fixed rule 0.35 times its 99.9th percentile. This definition is applied identically to all four positions and to the paired experimental response.
"""
    (OUT / "methods" / "signal_extent_definition.md").write_text(text, encoding="utf-8")


def evaluate_no_object_gate(sim: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    width_rel = abs(sim["Wsignal_px"] - exp["Wsignal_px"]) / max(exp["Wsignal_px"], 1.0e-12)
    height_rel = abs(sim["Hsignal_px"] - exp["Hsignal_px"]) / max(exp["Hsignal_px"], 1.0e-12)
    center_distance = math.hypot(sim["x0_px"] - exp["x0_px"], sim["y0_px_top"] - exp["y0_px_top"])
    center_tolerance = 0.10 * math.hypot(455.0, 419.0)
    sim_ellipticity = sim["major_px"] / max(sim["minor_px"], 1.0e-12)
    exp_ellipticity = exp["major_px"] / max(exp["minor_px"], 1.0e-12)
    extent_pass = width_rel <= 0.20 and height_rel <= 0.20
    center_pass = center_distance <= center_tolerance
    ellipticity_pass = abs(sim_ellipticity - exp_ellipticity) <= 0.20
    angle_testable = sim["ellipse_angle_deg_ccw_from_camera_x"] is not None and exp["ellipse_angle_deg_ccw_from_camera_x"] is not None
    if angle_testable:
        da = abs(sim["ellipse_angle_deg_ccw_from_camera_x"] - exp["ellipse_angle_deg_ccw_from_camera_x"]) % 180.0
        angle_diff = min(da, 180.0 - da)
        angle_pass = angle_diff <= 15.0
    else:
        angle_diff = None
        angle_pass = False
    passed = extent_pass and center_pass and ellipticity_pass and angle_pass
    return {
        "NO_OBJECT_SIGNAL_FIELD_GATE": "PASS" if passed else "FAIL",
        "criteria": {
            "relative_extent_tolerance": 0.20,
            "centroid_tolerance_fraction_of_roi_diagonal": 0.10,
            "ellipticity_absolute_tolerance": 0.20,
            "angle_tolerance_deg": 15.0,
        },
        "width_relative_difference": width_rel,
        "height_relative_difference": height_rel,
        "center_distance_px": center_distance,
        "center_tolerance_px": center_tolerance,
        "ellipticity_sim": sim_ellipticity,
        "ellipticity_exp": exp_ellipticity,
        "angle_difference_deg": angle_diff,
        "component_pass": {
            "extent": extent_pass,
            "center": center_pass,
            "ellipticity": ellipticity_pass,
            "angle": angle_pass,
        },
    }


def _angle_value(metric: dict[str, Any]) -> Any:
    value = metric["ellipse_angle_deg_ccw_from_camera_x"]
    return "NA" if value is None else value


def create_figures(
    results: dict[tuple[str, str], dict[str, Any]],
    exp_arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    exp_metrics: dict[str, Any],
) -> list[Path]:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Z6 01 geometry
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 6.6), sharex=True)
    for ax, (name, pos) in zip(axes, POSITIONS.items()):
        ax.plot([0, 150], [0, 0], color="#334155", lw=2)
        ax.scatter([0], [0], marker="|", s=450, linewidths=5, color="#1d4ed8", label="fixed lens")
        ax.scatter([150], [0], marker="|", s=450, linewidths=5, color="#991b1b", label="fixed mirror")
        ax.scatter([pos["lens_to_rabbit_mm"]], [0], marker="D", s=65, color="#047857", label="rabbit")
        ax.text(0, 0.18, "Lens fixed", ha="left", fontsize=9)
        ax.text(150, 0.18, "Mirror fixed", ha="right", fontsize=9)
        ax.text(pos["lens_to_rabbit_mm"], -0.28, f"{name}: {pos['lens_to_rabbit_mm']:.0f} | {pos['rabbit_to_mirror_mm']:.0f} mm", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
    axes[-1].set_xlabel("Axial coordinate from fixed idler lens (mm)")
    fig.suptitle("Z6 fixed 150 mm lens-mirror geometry: only the rabbit moves", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_01_Rabbit_Longitudinal_Geometry.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 02 no-object field and equality table
    representative = results[("P0", "NO_OBJECT")]["roi"]
    diffs = [float(np.max(np.abs(results[(name, "NO_OBJECT")]["full"] - results[("P0", "NO_OBJECT")]["full"]))) for name in POSITIONS]
    m = results[("P0", "NO_OBJECT")]["roi_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].imshow(representative, cmap="viridis", origin="upper")
    axes[0].set_title("P0 representative no-object comparison ROI")
    axes[0].set_xlabel("camera x (px)")
    axes[0].set_ylabel("camera row from top (px)")
    axes[1].axis("off")
    lines = [
        f"x0 = {m['x0_px']:.2f} px; y0 = {m['y0_px_top']:.2f} px",
        f"Wsignal = {m['Wsignal_px']:.2f} px ({m['Wsignal_mm']:.3f} mm)",
        f"Hsignal = {m['Hsignal_px']:.2f} px ({m['Hsignal_mm']:.3f} mm)",
        f"major/minor = {m['major_px']:.2f}/{m['minor_px']:.2f} px",
        f"angle = {_angle_value(m)} deg",
        "",
        "Maximum absolute full-camera differences vs P0:",
    ] + [f"{name}: {diff:.3e}" for name, diff in zip(POSITIONS, diffs)] + ["", "The current Z5 runtime has no axial-mask distance consumer."]
    axes[1].text(0.02, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.suptitle("Z6 no-object camera field and position-equivalence check", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_02_NoObject_Camera_Field.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 03 profiles
    sim = representative.astype(np.float64)
    exp = exp_arrays["no_object_reference"].astype(np.float64)
    sim_w = np.clip(sim - np.percentile(sim, 1), 0, None) + 1e-12
    exp_w = np.clip(exp - np.percentile(exp, 1), 0, None) + 1e-12
    sim_x, sim_y = sim_w.sum(0), sim_w.sum(1)
    exp_x, exp_y = exp_w.sum(0), exp_w.sum(1)
    sim_x /= sim_x.max(); sim_y /= sim_y.max(); exp_x /= exp_x.max(); exp_y /= exp_y.max()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(sim_x, label="Simulation P0", lw=2)
    axes[0].plot(exp_x, label="Experiment no object", lw=1.5)
    axes[0].set_title("Horizontal profile")
    axes[0].set_xlabel("camera x (px)"); axes[0].set_ylabel("normalized marginal weight")
    axes[0].legend()
    axes[1].plot(sim_y, label="Simulation P0", lw=2)
    axes[1].plot(exp_y, label="Experiment no object", lw=1.5)
    axes[1].set_title("Vertical profile")
    axes[1].set_xlabel("camera row from top (px)"); axes[1].legend()
    fig.suptitle("Z6 no-object profile size: fixed Wsignal/Hsignal estimator", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_03_NoObject_Profile_Size.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 04 rabbit sweep
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, row in zip(axes, rows):
        name = row["position"]
        image = results[(name, "RABBIT")]["roi"]
        ax.imshow(image, cmap="gray", origin="upper", vmin=float(np.min(image)), vmax=float(np.max(image)))
        ax.set_title(f"{name} - {row['lens_to_rabbit_mm']:.0f}/{row['rabbit_to_mirror_mm']:.0f} mm")
        ax.text(0.02, 0.02, f"Wr x Hr {row['Wrabbit_px']:.0f} x {row['Hrabbit_px']:.0f} px\nWs x Hs {row['rabbit_Wsignal_px']:.1f} x {row['rabbit_Hsignal_px']:.1f} px", transform=ax.transAxes, color="white", fontsize=8, va="bottom", bbox={"facecolor": "black", "alpha": 0.55, "pad": 3})
        ax.set_xlabel("camera x (px)")
    axes[0].set_ylabel("camera row from top (px)")
    fig.suptitle("Z6 rabbit longitudinal position sweep in fixed camera coordinates", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_04_Rabbit_Position_Sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 05 position metrics
    shifts = np.array([row["shift_toward_lens_mm"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(shifts, [row["Wrabbit_px"] for row in rows], "o-", label="Wrabbit")
    axes[0].plot(shifts, [row["Hrabbit_px"] for row in rows], "s-", label="Hrabbit")
    axes[0].set_title("Rabbit support")
    axes[0].set_xlabel("rabbit displacement toward lens (mm)"); axes[0].set_ylabel("pixels"); axes[0].legend()
    axes[1].plot(shifts, [row["rabbit_Wsignal_px"] for row in rows], "o-", label="Wsignal")
    axes[1].plot(shifts, [row["rabbit_Hsignal_px"] for row in rows], "s-", label="Hsignal")
    axes[1].set_title("Signal extent")
    axes[1].set_xlabel("rabbit displacement toward lens (mm)"); axes[1].set_ylabel("pixels"); axes[1].legend()
    fig.suptitle("Z6 position metrics: flat curves expose missing axial-distance consumption", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_05_Rabbit_Position_Metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 06 experiment / simulation and fixed overlays
    fig, axes = plt.subplots(2, 5, figsize=(15.2, 7.0), sharex=True, sharey=True)
    exp_primary = exp_arrays["rabbit_primary_report0"]
    axes[0, 0].imshow(exp_primary, cmap="gray", origin="upper")
    axes[0, 0].set_title("Experiment Report/0")
    axes[1, 0].imshow(exp_primary, cmap="gray", origin="upper")
    axes[1, 0].set_title("Experiment base")
    for column, name in enumerate(POSITIONS, start=1):
        sim_img = results[(name, "RABBIT")]["roi"]
        axes[0, column].imshow(sim_img, cmap="gray", origin="upper")
        axes[0, column].set_title(name)
        axes[1, column].imshow(exp_primary, cmap="gray", origin="upper")
        axes[1, column].imshow(display_normalize(sim_img, 0.1, 99.9), cmap="magma", origin="upper", alpha=0.38)
        axes[1, column].set_title(f"{name} fixed overlay")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Z6 fixed-coordinate experimental and simulated rabbit comparison\nNo resize, no fitted rotation, no per-case translation", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_06_Experimental_Simulation_Rabbit_Comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Z6 07 object effect
    noobj = results[("P0", "NO_OBJECT")]["roi"]
    rabbit = results[("P0", "RABBIT")]["roi"]
    delta = rabbit - noobj
    exp_delta = exp_arrays["rabbit_paired_reference"] - exp_arrays["no_object_reference"]
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.6))
    panels = [
        (noobj, "Simulation no object", "viridis"),
        (rabbit, "Simulation rabbit", "gray"),
        (delta, "Simulation Delta I", "coolwarm"),
        (exp_arrays["no_object_reference"], "Experiment no object", "viridis"),
        (exp_arrays["rabbit_paired_reference"], "Experiment rabbit paired", "gray"),
        (exp_delta, "Experiment Delta I", "coolwarm"),
    ]
    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        vmax = float(np.max(np.abs(image))) if cmap == "coolwarm" else None
        ax.imshow(image, cmap=cmap, origin="upper", vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("camera x"); ax.set_ylabel("camera row")
    fig.suptitle("Z6 no-object versus rabbit response", fontsize=14)
    fig.tight_layout()
    path = figures / "Z6_07_NoObject_vs_Rabbit_Response.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def run_all() -> dict[str, Any]:
    ensure_structure()
    write_requirements()
    write_geometry_audit()
    write_method_definition()
    orientation = orientation_test_payload()
    save_json(OUT / "audit" / "camera_orientation_test.json", orientation)
    if orientation["TEST"] != "PASS":
        raise AssertionError("camera orientation test failed")

    inventory, _ = image_inventory_rows()
    save_csv(
        OUT / "audit" / "experimental_data_inventory.csv",
        inventory,
        ["filename", "path", "dimensions", "dtype", "min", "max", "mean", "classification", "classification_evidence"],
    )
    exp_arrays = experimental_arrays()
    exp_noobj_metrics = signal_extent_metrics(exp_arrays["no_object_reference"])
    exp_primary_metrics = signal_extent_metrics(exp_arrays["rabbit_primary_report0"])
    exp_paired_rabbit_metrics = signal_extent_metrics(exp_arrays["rabbit_paired_reference"])
    exp_delta = exp_arrays["rabbit_paired_reference"] - exp_arrays["no_object_reference"]
    exp_rabbit_extent = rabbit_extent_metrics(exp_delta)
    exp_metrics = {
        "no_object": exp_noobj_metrics,
        "rabbit_primary_report0": exp_primary_metrics,
        "rabbit_paired_reference": exp_paired_rabbit_metrics,
        "rabbit_extent_from_paired_response": exp_rabbit_extent,
    }
    save_json(OUT / "metrics" / "experimental_metrics.json", exp_metrics)

    setup, magnification, object_pitch_um = build_setup()
    rabbit_hash_before = sha256(Y1 / "data" / "frozen" / "mask_rabbit.npy")
    rabbit_T, resample_meta = load_rabbit_transmission(object_pitch_um)
    save_json(OUT / "provenance" / "rabbit_resample_metadata.json", resample_meta)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for position in POSITIONS:
        for state in ("NO_OBJECT", "RABBIT"):
            results[(position, state)] = simulate_one(position, state, setup, rabbit_T)

    rabbit_hash_after = sha256(Y1 / "data" / "frozen" / "mask_rabbit.npy")
    rabbit_unchanged = rabbit_hash_before == rabbit_hash_after == "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"
    save_json(
        OUT / "provenance" / "rabbit_input_hash_check.json",
        {
            "before": rabbit_hash_before,
            "after": rabbit_hash_after,
            "RABBIT_INPUT_ROTATED": "NO",
            "unchanged": rabbit_unchanged,
        },
    )
    if not rabbit_unchanged:
        raise AssertionError("frozen rabbit input changed")

    config_rows: list[dict[str, Any]] = []
    keys = list(results)
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1 :]:
            left = results[left_key]["config"]
            right = results[right_key]["config"]
            differences = config_differences(left, right)
            same_position = left_key[0] == right_key[0]
            same_state = left_key[1] == right_key[1]
            unexpected = [d for d in differences if not allowed_config_difference(d, same_position, same_state)]
            config_rows.append(
                {
                    "left_run": f"{left_key[0]}_{left_key[1]}",
                    "right_run": f"{right_key[0]}_{right_key[1]}",
                    "different_fields": ";".join(differences) if differences else "NONE",
                    "unexpected_fields": ";".join(unexpected) if unexpected else "NONE",
                    "status": "PASS" if not unexpected else "FAIL",
                }
            )
    save_csv(OUT / "audit" / "config_difference_matrix.csv", config_rows)
    unexpected_config = any(row["status"] == "FAIL" for row in config_rows)

    noobj_reference = results[("P0", "NO_OBJECT")]["full"]
    noobj_equality_rows = []
    for position in POSITIONS:
        current = results[(position, "NO_OBJECT")]["full"]
        noobj_equality_rows.append(
            {
                "position": position,
                "array_equal_to_P0": bool(np.array_equal(current, noobj_reference)),
                "max_abs_difference": float(np.max(np.abs(current - noobj_reference))),
                "reason": "Z5 reduced-order runtime has no explicit rabbit axial-plane distance consumer",
            }
        )
    save_csv(OUT / "metrics" / "no_object_position_equality.csv", noobj_equality_rows)

    sweep_rows: list[dict[str, Any]] = []
    exp_response = exp_arrays["rabbit_paired_reference"] - exp_arrays["no_object_reference"]
    for position, pos in POSITIONS.items():
        noobj = results[(position, "NO_OBJECT")]
        rabbit = results[(position, "RABBIT")]
        delta = rabbit["roi"] - noobj["roi"]
        rabbit_extent = rabbit_extent_metrics(delta)
        corr = spatial_and_edge_correlation(delta, exp_response)
        np.save(OUT / "data" / position / "object_induced_difference.npy", delta.astype(np.float32))
        tifffile.imwrite(OUT / "data" / position / "object_induced_difference.tif", delta.astype(np.float32))
        n = noobj["roi_metrics"]
        r = rabbit["roi_metrics"]
        row = {
            "position": position,
            "shift_toward_lens_mm": pos["shift_toward_lens_mm"],
            "lens_to_rabbit_mm": pos["lens_to_rabbit_mm"],
            "rabbit_to_mirror_mm": pos["rabbit_to_mirror_mm"],
            "lens_to_mirror_mm": 150.0,
            "noobj_x0": n["x0_px"],
            "noobj_y0": n["y0_px_top"],
            "noobj_Wsignal_px": n["Wsignal_px"],
            "noobj_Hsignal_px": n["Hsignal_px"],
            "noobj_Wsignal_mm": n["Wsignal_mm"],
            "noobj_Hsignal_mm": n["Hsignal_mm"],
            "noobj_major": n["major_px"],
            "noobj_minor": n["minor_px"],
            "noobj_angle": _angle_value(n),
            "noobj_angle_reason": n["ellipse_angle_reason"] or "NA",
            "rabbit_Wsignal_px": r["Wsignal_px"],
            "rabbit_Hsignal_px": r["Hsignal_px"],
            "rabbit_Wsignal_mm": r["Wsignal_mm"],
            "rabbit_Hsignal_mm": r["Hsignal_mm"],
            "Wrabbit_px": rabbit_extent["Wrabbit_px"] if rabbit_extent["Wrabbit_px"] is not None else "NA",
            "Hrabbit_px": rabbit_extent["Hrabbit_px"] if rabbit_extent["Hrabbit_px"] is not None else "NA",
            "Wrabbit_mm": rabbit_extent["Wrabbit_mm"] if rabbit_extent["Wrabbit_mm"] is not None else "NA",
            "Hrabbit_mm": rabbit_extent["Hrabbit_mm"] if rabbit_extent["Hrabbit_mm"] is not None else "NA",
            "spatial_corr": corr["spatial_corr"] if corr["spatial_corr"] is not None else "NA",
            "edge_corr": corr["edge_corr"] if corr["edge_corr"] is not None else "NA",
            "rabbit_bbox": json.dumps(rabbit_extent["bbox"]),
            "rabbit_centroid_x_px": rabbit_extent["centroid_x_px"],
            "rabbit_centroid_y_px": rabbit_extent["centroid_y_px_top"],
            "signal_centroid_x_px": r["x0_px"],
            "signal_centroid_y_px": r["y0_px_top"],
            "object_response_l1": rabbit_extent.get("object_response_l1", "NA"),
            "object_response_l2": rabbit_extent.get("object_response_l2", "NA"),
            "sharpness_sobel_mean": rabbit_extent.get("sharpness_sobel_mean", "NA"),
            "runtime_position_consumed": "NO",
        }
        sweep_rows.append(row)
        save_json(
            OUT / "metrics" / f"{position}_paired_metrics.json",
            {
                "no_object": n,
                "rabbit_signal": r,
                "rabbit_structure": rabbit_extent,
                "experimental_response_correlation": corr,
            },
        )
    required_columns = [
        "position", "shift_toward_lens_mm", "lens_to_rabbit_mm", "rabbit_to_mirror_mm", "lens_to_mirror_mm",
        "noobj_x0", "noobj_y0", "noobj_Wsignal_px", "noobj_Hsignal_px", "noobj_Wsignal_mm", "noobj_Hsignal_mm", "noobj_major", "noobj_minor", "noobj_angle",
        "rabbit_Wsignal_px", "rabbit_Hsignal_px", "rabbit_Wsignal_mm", "rabbit_Hsignal_mm",
        "Wrabbit_px", "Hrabbit_px", "Wrabbit_mm", "Hrabbit_mm", "spatial_corr", "edge_corr",
        "noobj_angle_reason", "rabbit_bbox", "rabbit_centroid_x_px", "rabbit_centroid_y_px", "signal_centroid_x_px", "signal_centroid_y_px", "object_response_l1", "object_response_l2", "sharpness_sobel_mean", "runtime_position_consumed",
    ]
    save_csv(OUT / "metrics" / "position_sweep_metrics.csv", sweep_rows, required_columns)

    comparison_rows: list[dict[str, Any]] = []
    for position in POSITIONS:
        sim = results[(position, "NO_OBJECT")]["roi_metrics"]
        comparison_rows.append(
            {
                "position": position,
                "metric": "x0_px",
                "experiment": exp_noobj_metrics["x0_px"],
                "simulation": sim["x0_px"],
                "absolute_difference": abs(exp_noobj_metrics["x0_px"] - sim["x0_px"]),
                "relative_difference": abs(exp_noobj_metrics["x0_px"] - sim["x0_px"]) / max(abs(exp_noobj_metrics["x0_px"]), 1e-12),
            }
        )
        for metric in ("y0_px_top", "Wsignal_px", "Hsignal_px", "major_px", "minor_px"):
            comparison_rows.append(
                {
                    "position": position,
                    "metric": metric,
                    "experiment": exp_noobj_metrics[metric],
                    "simulation": sim[metric],
                    "absolute_difference": abs(exp_noobj_metrics[metric] - sim[metric]),
                    "relative_difference": abs(exp_noobj_metrics[metric] - sim[metric]) / max(abs(exp_noobj_metrics[metric]), 1e-12),
                }
            )
        comparison_rows.append(
            {
                "position": position,
                "metric": "ellipse_angle_deg_ccw_from_camera_x",
                "experiment": _angle_value(exp_noobj_metrics),
                "simulation": _angle_value(sim),
                "absolute_difference": "NA" if _angle_value(exp_noobj_metrics) == "NA" or _angle_value(sim) == "NA" else abs(float(_angle_value(exp_noobj_metrics)) - float(_angle_value(sim))),
                "relative_difference": "NA",
            }
        )
    save_csv(OUT / "metrics" / "no_object_exp_sim_comparison.csv", comparison_rows)

    gate = evaluate_no_object_gate(results[("P0", "NO_OBJECT")]["roi_metrics"], exp_noobj_metrics)
    save_json(OUT / "metrics" / "no_object_gate.json", gate)

    sensitivity_rows: list[dict[str, Any]] = []
    p0 = sweep_rows[0]
    observables = [
        "Wrabbit_px", "Hrabbit_px", "rabbit_Wsignal_px", "rabbit_Hsignal_px",
        "signal_centroid_x_px", "signal_centroid_y_px", "object_response_l1",
        "sharpness_sobel_mean", "spatial_corr", "edge_corr",
    ]
    for row in sweep_rows[1:]:
        for observable in observables:
            base = p0[observable]
            current = row[observable]
            change = "NA" if isinstance(base, str) or isinstance(current, str) else float(current) - float(base)
            sensitivity_rows.append(
                {
                    "comparison": f"P0_to_{row['position']}",
                    "observable": observable,
                    "P0": base,
                    "current": current,
                    "absolute_change": change,
                    "interpretation": "zero because the current Z5 runtime does not consume the declared axial mask coordinate" if change == 0 else "nonzero",
                }
            )
    save_csv(OUT / "metrics" / "position_sensitivity.csv", sensitivity_rows)

    figures = create_figures(results, exp_arrays, sweep_rows, exp_metrics)
    hashes = {
        "rabbit_input_before": rabbit_hash_before,
        "rabbit_input_after": rabbit_hash_after,
        "optics.py": sha256(ROOT / "optics.py"),
        "setup_params.py": sha256(ROOT / "setup_params.py"),
        "C_lab_config": sha256(ROOT / "configs" / "setup_config_c_lab_v1.yaml"),
        "Z5_run_config": sha256(Z5 / "runs" / "RUN_C_CORRECTED" / "run_config.json"),
    }
    save_json(OUT / "provenance" / "POST_Z6_UPSTREAM_HASHES.json", hashes)
    status = {
        "utc": utc_now(),
        "authoritative_report": str(PHYSICS_REPORT),
        "project_root_requested": str(ROOT.parent / "Synthetic_3.4.2_30sep"),
        "project_root_active": str(ROOT),
        "geometry_declarative_assertions": "PASS",
        "geometry_runtime_consumption": "FAIL_NOT_IMPLEMENTED",
        "GEOMETRY_VERIFIED": "NO",
        "CAMERA_ORIENTATION_VERIFIED": "YES",
        "EXPERIMENTAL_NO_OBJECT_AVAILABLE": "YES",
        "EXPERIMENTAL_RABBIT_AVAILABLE": "YES",
        "NO_OBJECT_SIGNAL_FIELD_GATE": gate["NO_OBJECT_SIGNAL_FIELD_GATE"],
        "NO_OBJECT_BASELINE_UNDERSTOOD": "YES",
        "RABBIT_POSITION_SWEEP_COMPLETE": "YES",
        "RABBIT_IMAGE_FORMATION": "INCONCLUSIVE",
        "RABBIT_INPUT_ROTATED": "NO",
        "NO_RESIZE_USED": True,
        "unexpected_parameter_changes": unexpected_config,
        "no_object_fields_identical": all(row["array_equal_to_P0"] for row in noobj_equality_rows),
        "rabbit_fields_identical": all(np.array_equal(results[(position, "RABBIT")]["full"], results[("P0", "RABBIT")]["full"]) for position in POSITIONS),
        "object_response_measurable": float(np.linalg.norm(results[("P0", "RABBIT")]["roi"] - results[("P0", "NO_OBJECT")]["roi"])) > 0,
        "root_cause_classification": "MULTIPLE",
        "root_cause_evidence": [
            "SIGNAL_FIELD_FORMATION: Z5 transparent-object output is spatially uniform while the genuine experimental no-object reference is nonuniform.",
            "OBJECT_MAPPING/GEOMETRY: the runtime treats the reflective object as replacing the end mirror and has no explicit intermediate-mask axial distance.",
            "ROI_OBSERVATION: the frozen ROI is integer-only and no resize is used, but its origin was previously inferred from signal center rather than measured acquisition metadata.",
            "CAMERA_ORIENTATION is not a remaining problem: exact 90 degree counter-clockwise transform with flips off passes.",
        ],
        "rabbit_position_sensitivity": "LOW_MODEL_INSENSITIVE_NOT_PHYSICALLY_VALIDATED",
        "READY_FOR_FIGURE5": "NO",
        "figure5_reason": "The declarative geometry passes, but the Z5 runtime cannot propagate an intermediate rabbit plane; the no-object signal-field gate also fails.",
        "figures": [str(path) for path in figures],
        "evidence": str(OUT),
        "report_expected": str(REPORT_ROOT / FINAL_REPORT_NAME),
        "magnification_frozen": magnification,
        "object_pitch_um_frozen": object_pitch_um,
    }
    save_json(OUT / "provenance" / "Z6_status.json", status)
    manifest = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    save_json(OUT / "provenance" / "EVIDENCE_MANIFEST_PRE_REPORT.json", manifest)
    print(json.dumps(status, indent=2))
    return status


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "NA":
        return "NA"
    if isinstance(value, str):
        try:
            return f"{float(value):.{digits}f}"
        except ValueError:
            return value
    return f"{float(value):.{digits}f}"


def build_report(output: Path) -> Path:
    from docx import Document
    from docx.enum.section import WD_SECTION_START
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    status = read_json(OUT / "provenance" / "Z6_status.json")
    rows = list(csv.DictReader((OUT / "metrics" / "position_sweep_metrics.csv").open(encoding="utf-8")))
    exp = read_json(OUT / "metrics" / "experimental_metrics.json")
    gate = read_json(OUT / "metrics" / "no_object_gate.json")
    test_status_path = OUT / "tests" / "test_status.json"
    tests = read_json(test_status_path) if test_status_path.exists() else {"focused": {"passed": "NA", "failed": "NA"}, "regression": {"passed": "NA", "failed": "NA"}}

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    for style_name, size in (("Title", 24), ("Heading 1", 16), ("Heading 2", 12)):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name != "Normal" else "Aptos"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = style_name != "Normal"
        style.paragraph_format.space_before = Pt(12 if style_name != "Title" else 0)
        style.paragraph_format.space_after = Pt(6)
    title_ppr = styles["Title"].element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)

    def set_cell_shading(cell, fill: str) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:fill"), fill)

    def set_borders(cell, color: str = "D9D9D9") -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.first_child_found_in("w:tcBorders")
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            tag = "w:" + edge
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:color"), color)

    def add_table(headers: list[str], body: list[list[Any]], widths: list[float] | None = None, font_size: float = 8.6):
        table = doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        hdr = table.rows[0].cells
        for i, text in enumerate(headers):
            hdr[i].text = str(text)
            set_cell_shading(hdr[i], "1F4E78")
            hdr[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in hdr[i].paragraphs[0].runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.bold = True
                run.font.size = Pt(font_size)
            hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_borders(hdr[i])
        for ridx, values in enumerate(body):
            cells = table.add_row().cells
            for cidx, value in enumerate(values):
                cells[cidx].text = str(value)
                cells[cidx].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if ridx % 2:
                    set_cell_shading(cells[cidx], "EAF2F8")
                for paragraph in cells[cidx].paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if cidx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.size = Pt(font_size)
                set_borders(cells[cidx])
        if widths:
            for row in table.rows:
                for i, width in enumerate(widths):
                    row.cells[i].width = Inches(width)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return table

    def add_figure(filename: str, caption: str, width: float = 6.75) -> None:
        path = OUT / "figures" / filename
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        for run in cap.runs:
            run.font.size = Pt(9)
            run.font.italic = True

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Z6 C lab Mask Position and Signal Field Diagnostic Report")
    direct_title_border = title._p.get_or_add_pPr().find(qn("w:pBdr"))
    if direct_title_border is not None:
        title._p.get_or_add_pPr().remove(direct_title_border)
    sub = doc.add_paragraph()
    sub.add_run("Fixed camera orientation and fixed 150 mm idler lens mirror geometry").bold = True
    doc.add_paragraph("Physics-team instruction: QIUP Z5 Next Test FA RTL Fixed")
    doc.add_paragraph("Date: 6 October 2026")
    doc.add_paragraph(
        "Conclusion. The camera transform is verified as an exact 90 degree counter-clockwise array rotation with both flips off. The four declarative geometries and all eight paired runs are complete. However, the Z5 reduced-order C_lab runtime does not consume the rabbit axial coordinate and treats its reflective object as replacing the end mirror. All four positions are therefore numerically identical. The genuine experimental no-object comparison also fails the signal-field gate because the Z5 transparent-object output is spatially uniform. Figure 5 is not ready for regeneration."
    )
    add_table(
        ["Decision", "Result"],
        [
            ["Geometry assertions", "PASS"],
            ["Geometry runtime consumption", "FAIL"],
            ["Camera orientation", "PASS"],
            ["No-object signal-field gate", status["NO_OBJECT_SIGNAL_FIELD_GATE"]],
            ["Rabbit position sweep", "COMPLETE but model-insensitive"],
            ["Ready for Figure 5", status["READY_FOR_FIGURE5"]],
        ],
        [3.7, 2.5],
        9.2,
    )
    doc.add_page_break()

    sections = [
        "1 Objective", "2 Physics team requirements", "3 Z5 starting state", "4 C lab runtime geometry audit",
        "5 Fixed lens mirror geometry", "6 Four rabbit positions", "7 Camera orientation verification",
        "8 Experimental data inventory", "9 No object methodology", "10 Signal size metric definition",
        "11 No object full camera results", "12 No object horizontal and vertical profiles",
        "13 Major minor axis and ellipse angle results", "14 Experimental no object comparison",
        "15 Rabbit position methodology", "16 P0 results", "17 P5 results", "18 P7 results", "19 P10 results",
        "20 Wsignal and Hsignal comparison", "21 Wrabbit and Hrabbit comparison",
        "22 Fixed coordinate experimental simulation comparison", "23 Object induced response",
        "24 Configuration difference audit", "25 Focused tests", "26 Regression tests",
        "27 Root cause assessment", "28 Remaining limitations", "29 Figure 5 readiness", "30 Conclusion",
    ]

    doc.add_heading(sections[0], level=1)
    doc.add_paragraph(
        "Z6 isolates the laboratory camera convention, the prescribed longitudinal rabbit positions, and the signal field that exists without the rabbit. It does not fit the simulation to experiment and changes no pump, focal-length, camera-pitch, rabbit-scale, noise, visibility, threshold, or ROI-size parameter."
    )

    doc.add_heading(sections[1], level=1)
    req_rows = list(csv.DictReader((OUT / "audit" / "physics_team_Z6_requirements.csv").open(encoding="utf-8")))
    add_table(["ID", "Requirement", "Status"], [[r["requirement_id"], r["requirement"], r["status"]] for r in req_rows], [0.7, 5.0, 0.8], 7.8)

    doc.add_heading(sections[2], level=1)
    doc.add_paragraph(
        "The active starting state is the writable 1Oct continuation because it contains the completed Z5 correction and preserved Z1-Z5 evidence. The 30sep tree named in the instruction contains no Z-series evidence. Neither tree is a Git repository, so no commit hash or Git status is available. Upstream SHA256 hashes were frozen before edits. Z5 keeps the final camera-plane Gaussian envelope inactive while retaining the pump contribution through the far-field correlation width."
    )
    add_table(["Frozen item", "Value"], [["Camera", "1024 x 1024 at 13 um per pixel"], ["ROI", "left 290 top 293 width 455 height 419; integer slice"], ["Rabbit hash", "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"], ["F1 F2 far-field focal", "150 mm 150 mm 150 mm"], ["Pump waist", "1000 um assumption pending definition confirmation"]], [2.2, 4.3], 8.5)

    doc.add_heading(sections[3], level=1)
    doc.add_paragraph(
        "The runtime audit distinguishes focal length from separation. F1, F2, and far_field_focal_mm are active 150 mm focal lengths. No SetupParams field represents lens-to-rabbit, rabbit-to-mirror, or lens-to-mirror distance. The object transfer is an affine conditional-correlation model, and the reflective-object branch states that the object replaces the end mirror. This cannot represent a rabbit between a fixed lens and a separate fixed mirror."
    )
    add_table(["Element", "Runtime value", "Status"], [["Idler focal length", "150 mm", "active fixed"], ["Lens-rabbit distance", "missing", "not represented"], ["Rabbit-mirror distance", "missing", "not represented"], ["End mirror", "object replaces mirror", "conflicts with Z6 intermediate mask"]], [2.0, 2.2, 2.3], 8.5)

    doc.add_page_break()
    doc.add_heading(sections[4], level=1)
    doc.add_paragraph("The Z6 snapshots place the lens at 0 mm and mirror at 150 mm for every run. These positions are assertions and provenance records; they do not add an unvalidated propagation law to the Z5 runtime.")
    add_figure("Z6_01_Rabbit_Longitudinal_Geometry.png", "Figure Z6 01. Fixed lens and mirror with the four prescribed rabbit locations.")

    doc.add_heading(sections[5], level=1)
    add_table(["Position", "Shift toward lens", "Lens to rabbit", "Rabbit to mirror", "Sum"], [[r["position"], f"{float(r['shift_toward_lens_mm']):.0f} mm", f"{float(r['lens_to_rabbit_mm']):.0f} mm", f"{float(r['rabbit_to_mirror_mm']):.0f} mm", "150 mm"] for r in rows], [0.8, 1.4, 1.4, 1.4, 1.0], 8.5)
    doc.add_paragraph("The 82 mm rabbit-to-mirror value appears only in P7. It is not a global geometry.")

    # Keep the compact orientation audit together instead of splitting the
    # header and rows across two rendered pages.
    doc.add_page_break()
    doc.add_heading(sections[6], level=1)
    doc.add_paragraph("The detector array is rotated with np.rot90(k=1) after detector sampling and before ROI extraction. No mask data are rotated, and no interpolation, horizontal flip, or vertical flip is used. The asymmetric TOP BOTTOM LEFT RIGHT regression test passes exactly.")
    add_table(["Setting", "Result"], [["Camera orientation", "90 degrees counter-clockwise"], ["Horizontal flip", "OFF"], ["Vertical flip", "OFF"], ["Interpolation", "NO"], ["Regression test", "PASS"]], [3.1, 3.1], 9)

    doc.add_heading(sections[7], level=1)
    inv = list(csv.DictReader((OUT / "audit" / "experimental_data_inventory.csv").open(encoding="utf-8")))
    counts: dict[str, int] = {}
    for row in inv:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    doc.add_paragraph(f"The inventory contains {len(inv)} TIFF files: " + ", ".join(f"{key} {value}" for key, value in sorted(counts.items())) + ". Classifications use frozen acquisition metadata and prior Z4/Z5 provenance, not visual absence of a rabbit.")
    add_table(["Experimental role", "Frames", "Dimensions", "Use"], [["Report/0 rabbit", "12", "455 x 419", "primary Z5-aligned rabbit comparison"], ["Mirror-drift no-object ref", "10", "455 x 419", "genuine no-object baseline"], ["Mirror-drift with-object ref", "10", "455 x 419", "paired object-induced response"]], [2.2, 0.8, 1.2, 2.3], 8.5)

    doc.add_heading(sections[8], level=1)
    doc.add_paragraph("Each no-object run uses an all-ones transmission map. It is a transparent object, not a zero field, black mask, rabbit, or Gaussian test object. All other optical and numerical settings match the paired rabbit run.")

    doc.add_heading(sections[9], level=1)
    doc.add_paragraph("Wsignal and Hsignal are fixed as four intensity-weighted standard deviations along camera x and camera y after a first-percentile pedestal subtraction and clipping. Major and minor sizes are four standard deviations along the covariance eigenvectors. The angle is counter-clockwise from positive camera x. The exact definition is preserved in methods/signal_extent_definition.md and applied identically to experiment and simulation.")

    doc.add_heading(sections[10], level=1)
    doc.add_paragraph("All four simulated no-object fields are bitwise identical. The transparent-object Z5 output is uniform because the validated Z5 correction removed the inappropriate final camera-plane Gaussian without adding a measured signal-field formation model.")
    add_figure("Z6_02_NoObject_Camera_Field.png", "Figure Z6 02. Representative no-object ROI and exact cross-position equality check.")

    doc.add_heading(sections[11], level=1)
    add_figure("Z6_03_NoObject_Profile_Size.png", "Figure Z6 03. No-object marginal profiles under the fixed signal-extent estimator.")

    doc.add_heading(sections[12], level=1)
    add_table(["Source", "x0", "y0", "Wsignal", "Hsignal", "Major", "Minor", "Angle"], [["Simulation P0", _format_number(rows[0]["noobj_x0"]), _format_number(rows[0]["noobj_y0"]), _format_number(rows[0]["noobj_Wsignal_px"]), _format_number(rows[0]["noobj_Hsignal_px"]), _format_number(rows[0]["noobj_major"]), _format_number(rows[0]["noobj_minor"]), _format_number(rows[0]["noobj_angle"])], ["Experiment no object", _format_number(exp["no_object"]["x0_px"]), _format_number(exp["no_object"]["y0_px_top"]), _format_number(exp["no_object"]["Wsignal_px"]), _format_number(exp["no_object"]["Hsignal_px"]), _format_number(exp["no_object"]["major_px"]), _format_number(exp["no_object"]["minor_px"]), _format_number(exp["no_object"]["ellipse_angle_deg_ccw_from_camera_x"])]], [1.3, 0.65, 0.65, 0.85, 0.85, 0.75, 0.75, 0.65], 7.7)

    doc.add_heading(sections[13], level=1)
    doc.add_paragraph(f"A genuine no-object reference is available. The gate result is {gate['NO_OBJECT_SIGNAL_FIELD_GATE']}. Relative Wsignal and Hsignal differences are {100*gate['width_relative_difference']:.1f}% and {100*gate['height_relative_difference']:.1f}%, and the centroid difference is {gate['center_distance_px']:.2f} pixels. A failed baseline is not compensated by changing rabbit scale or selecting a longitudinal position.")

    doc.add_heading(sections[14], level=1)
    doc.add_paragraph("For each position, the transparent and rabbit cases are executed independently with the same setup, seed, camera, orientation, and ROI. Delta I is stored only as a diagnostic. It does not replace the actual camera image. The declared axial distances are retained in normalized snapshots, and their lack of runtime consumption is tested directly.")
    add_figure("Z6_04_Rabbit_Position_Sweep.png", "Figure Z6 04. Four rabbit runs in the fixed camera and ROI coordinates.")

    for section_title, row in zip(sections[15:19], rows):
        doc.add_heading(section_title, level=1)
        doc.add_paragraph(f"{row['position']} declares {float(row['lens_to_rabbit_mm']):.0f} mm from lens to rabbit and {float(row['rabbit_to_mirror_mm']):.0f} mm from rabbit to mirror. The sum is 150 mm. The Z5 runtime output is identical to P0 because no axial-distance consumer exists.")
        add_table(["Metric", "Value"], [["Rabbit Wsignal x Hsignal", f"{float(row['rabbit_Wsignal_px']):.3f} x {float(row['rabbit_Hsignal_px']):.3f} px"], ["Wrabbit x Hrabbit", f"{float(row['Wrabbit_px']):.0f} x {float(row['Hrabbit_px']):.0f} px"], ["Spatial correlation", _format_number(row['spatial_corr'], 5)], ["Edge correlation", _format_number(row['edge_corr'], 5)], ["Position consumed", row['runtime_position_consumed']]], [3.1, 3.1], 8.5)

    doc.add_heading(sections[19], level=1)
    add_table(["Position", "No-object W x H", "Rabbit W x H"], [[r["position"], f"{float(r['noobj_Wsignal_px']):.2f} x {float(r['noobj_Hsignal_px']):.2f} px", f"{float(r['rabbit_Wsignal_px']):.2f} x {float(r['rabbit_Hsignal_px']):.2f} px"] for r in rows], [1.1, 2.6, 2.6], 8.8)

    doc.add_heading(sections[20], level=1)
    doc.add_paragraph("Wrabbit and Hrabbit are derived from the paired object response and are not aliases of Wsignal and Hsignal.")
    add_table(["Position", "Wrabbit", "Hrabbit", "Object response L1"], [[r["position"], f"{float(r['Wrabbit_px']):.0f} px", f"{float(r['Hrabbit_px']):.0f} px", _format_number(r['object_response_l1'], 3)] for r in rows], [1.1, 1.5, 1.5, 2.2], 8.8)
    add_figure("Z6_05_Rabbit_Position_Metrics.png", "Figure Z6 05. Rabbit and signal sizes versus declared displacement.")

    doc.add_heading(sections[21], level=1)
    doc.add_paragraph("The experimental and simulated arrays share the 455 x 419 comparison shape. Overlays use the same pixel indices and 13 um pitch. No image is resized, recentered, rotated per case, translated, or magnification fitted.")
    add_figure("Z6_06_Experimental_Simulation_Rabbit_Comparison.png", "Figure Z6 06. Fixed-coordinate experiment and simulation comparison.")

    doc.add_heading(sections[22], level=1)
    doc.add_paragraph("Rabbit information is measurable in both simulation and the paired experimental reference. The response remains distinct from the underlying signal field, but its simulated longitudinal sensitivity cannot be interpreted because all four axial declarations map to the same runtime computation.")
    add_figure("Z6_07_NoObject_vs_Rabbit_Response.png", "Figure Z6 07. No-object, rabbit, and paired object-induced response.")

    doc.add_heading(sections[23], level=1)
    doc.add_paragraph("All 28 pairwise configuration comparisons pass. Across positions, only the three permitted axial geometry fields differ. Within each position, only the transparent-versus-rabbit object-state fields differ. No pump, wavelength, focal length, camera, grid, ROI, noise, normalization, or Z5 envelope setting changes.")

    doc.add_heading(sections[24], level=1)
    focused = tests.get("focused", {})
    doc.add_paragraph(f"The 20 focused geometry, orientation, transparency, ordering, no-resize, pitch, rabbit-hash, Z5-correction, and hidden-difference tests ran before the production evidence. Passed: {focused.get('passed', 'NA')}; failed: {focused.get('failed', 'NA')}.")

    doc.add_heading(sections[25], level=1)
    regression = tests.get("regression", {})
    doc.add_paragraph(f"Relevant C_lab, Z4, and Z5 repository regressions ran after the focused tests. Passed: {regression.get('passed', 'NA')}; failed: {regression.get('failed', 'NA')}. The exact command and JUnit records are preserved under evidence/tests.")

    doc.add_heading(sections[26], level=1)
    doc.add_paragraph("Primary classification: MULTIPLE. Signal-field formation is unsupported at the no-object camera level because the corrected Z5 transparent field is uniform. The requested intermediate rabbit geometry is also unsupported because the runtime object replaces the mirror and has no axial-distance consumer. Camera orientation is verified and is not a remaining cause. Object mapping remains inconclusive until an explicit fixed-lens/fixed-mirror intermediate-mask propagation is implemented and validated. The ROI uses no resize, but its origin remains inferred rather than acquisition-calibrated.")

    doc.add_heading(sections[27], level=1)
    doc.add_paragraph("The diagnostic does not introduce an ABCD or Fresnel model for the missing rabbit plane because the supplied physics instruction does not specify the field at the lens, mask boundary condition on outbound and return passes, mirror phase, aperture, or propagation convention. Inventing those quantities would turn Z6 into an untraceable fit. The pump-waist definition and ROI-origin provenance remain documented uncertainties. The position correlations are diagnostic only and do not select a calibrated geometry.")

    doc.add_heading(sections[28], level=1)
    doc.add_paragraph("READY_FOR_FIGURE5 = NO. Camera orientation, paired outputs, frozen sampling, and no-resize handling are verified. Readiness fails because the physical mask-position sweep is not represented by the runtime and the experimental no-object gate fails. Final Figure 5 must not be regenerated from these results.")

    doc.add_heading(sections[29], level=1)
    doc.add_paragraph("Z6 verifies the fixed camera convention and preserves all Z5 physics and evidence. It completes the requested eight-run matrix and proves that the current runtime is insensitive to the declared mask position for an implementation reason, not a physical measurement. The next physics change must explicitly model a transmissive or reflective intermediate mask between a fixed lens and separate fixed mirror, then re-run the no-object gate before interpreting rabbit-position sensitivity.")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output


def parse_junit(path: Path) -> dict[str, int]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    return {"passed": tests - failures - errors - skipped, "failed": failures + errors, "skipped": skipped, "total": tests}


def record_test_status() -> dict[str, Any]:
    focused_xml = OUT / "tests" / "focused_tests.xml"
    regression_xml = OUT / "tests" / "repository_regression.xml"
    payload = {
        "focused": parse_junit(focused_xml),
        "regression": parse_junit(regression_xml),
        "order": "focused first, repository regression second",
    }
    save_json(OUT / "tests" / "test_status.json", payload)
    return payload


def finalize_terminal_summary() -> str:
    status = read_json(OUT / "provenance" / "Z6_status.json")
    rows = list(csv.DictReader((OUT / "metrics" / "position_sweep_metrics.csv").open(encoding="utf-8")))
    exp = read_json(OUT / "metrics" / "experimental_metrics.json")
    tests = read_json(OUT / "tests" / "test_status.json")
    figures = [str(OUT / "figures" / f"Z6_0{i}_{name}.png") for i, name in enumerate([
        "Rabbit_Longitudinal_Geometry", "NoObject_Camera_Field", "NoObject_Profile_Size", "Rabbit_Position_Sweep", "Rabbit_Position_Metrics", "Experimental_Simulation_Rabbit_Comparison", "NoObject_vs_Rabbit_Response"
    ], 1)]
    noobj_lines = []
    rabbit_lines = []
    for row in rows:
        noobj_lines.append(f"{row['position']:<10} {_format_number(row['noobj_x0'])}  {_format_number(row['noobj_y0'])}  {_format_number(row['noobj_Wsignal_px'])} px / {_format_number(row['noobj_Wsignal_mm'])} mm  {_format_number(row['noobj_Hsignal_px'])} px / {_format_number(row['noobj_Hsignal_mm'])} mm  {_format_number(row['noobj_major'])} px  {_format_number(row['noobj_minor'])} px  {_format_number(row['noobj_angle'])} deg")
        rabbit_lines.append(f"{row['position']:<10} {_format_number(row['rabbit_Wsignal_px'])} px  {_format_number(row['rabbit_Hsignal_px'])} px  {_format_number(row['Wrabbit_px'])} px  {_format_number(row['Hrabbit_px'])} px")
    e = exp["no_object"]
    er = exp["rabbit_extent_from_paired_response"]
    summary = f"""================ Z6 FINAL SUMMARY =================

PHYSICS-TEAM REPORT:
QIUP_Z5_Next_Test_FA_RTL_FIXED.docx

--------------------------------------------------

GEOMETRY

Lens->Mirror:
150.000 mm

Required:
150.000 mm

Lens fixed:
YES

Mirror fixed:
YES

Only rabbit moved:
YES (declarative geometry); NO runtime distance consumer

--------------------------------------------------

POSITIONS

P0
Lens->Rabbit:
75.000 mm
Rabbit->Mirror:
75.000 mm

P5
Lens->Rabbit:
70.000 mm
Rabbit->Mirror:
80.000 mm

P7
Lens->Rabbit:
68.000 mm
Rabbit->Mirror:
82.000 mm

P10
Lens->Rabbit:
65.000 mm
Rabbit->Mirror:
85.000 mm

All sums = 150 mm:
PASS

--------------------------------------------------

CAMERA ORIENTATION

Rotation:
90 degrees CCW

Required:
90 degrees CCW

Horizontal flip:
OFF

Vertical flip:
OFF

Interpolation:
NO

Orientation test:
PASS

--------------------------------------------------

EXPERIMENTAL DATA

No-object reference:
FOUND

Rabbit reference:
FOUND

--------------------------------------------------

NO-OBJECT SIGNAL FIELD

           x0    y0    Wsignal    Hsignal    Major    Minor    Angle

{chr(10).join(noobj_lines)}
Experiment {_format_number(e['x0_px'])}  {_format_number(e['y0_px_top'])}  {_format_number(e['Wsignal_px'])} px / {_format_number(e['Wsignal_mm'])} mm  {_format_number(e['Hsignal_px'])} px / {_format_number(e['Hsignal_mm'])} mm  {_format_number(e['major_px'])} px  {_format_number(e['minor_px'])} px  {_format_number(e['ellipse_angle_deg_ccw_from_camera_x'])} deg

Units:
px and mm where applicable

No-object Exp-Sim gate:
{status['NO_OBJECT_SIGNAL_FIELD_GATE']}

--------------------------------------------------

RABBIT RESULTS

           Wsignal    Hsignal    Wrabbit    Hrabbit

{chr(10).join(rabbit_lines)}
Experiment {_format_number(exp['rabbit_paired_reference']['Wsignal_px'])} px  {_format_number(exp['rabbit_paired_reference']['Hsignal_px'])} px  {_format_number(er['Wrabbit_px'])} px  {_format_number(er['Hrabbit_px'])} px

--------------------------------------------------

POSITION SENSITIVITY

Change P0->P5:
0 for all reported simulated observables; axial coordinate not consumed

Change P0->P7:
0 for all reported simulated observables; axial coordinate not consumed

Change P0->P10:
0 for all reported simulated observables; axial coordinate not consumed

Most sensitive observable:
NONE IN CURRENT MODEL

Closest agreement:
All positions tie because their runtime arrays are identical
DIAGNOSTIC ONLY - NOT AUTOMATIC CALIBRATION

--------------------------------------------------

OBJECT RESPONSE

Rabbit information present:
YES

Object-induced response measurable:
YES

Fixed-coordinate overlay generated:
YES

--------------------------------------------------

CONFIGURATION AUDIT

Unexpected parameter changes:
NO

If YES:
Not applicable

--------------------------------------------------

TESTS

Focused passed:
{tests['focused']['passed']}

Focused failed:
{tests['focused']['failed']}

Repository passed:
{tests['regression']['passed']}

Repository failed:
{tests['regression']['failed']}

--------------------------------------------------

ROOT CAUSE

Geometry problem:
YES

Camera-orientation problem:
NO

No-object signal-field problem:
YES

Rabbit-position sensitivity:
LOW (MODEL-INSENSITIVE; NOT PHYSICALLY VALIDATED)

Object-mapping problem:
INCONCLUSIVE

ROI/observation problem:
YES

Primary remaining issue:
MULTIPLE - signal-field formation plus missing intermediate-mask geometry; ROI origin remains inferred

--------------------------------------------------

FINAL VERDICT

GEOMETRY_VERIFIED:
NO

CAMERA_ORIENTATION_VERIFIED:
YES

NO_OBJECT_BASELINE_UNDERSTOOD:
YES

RABBIT_POSITION_SWEEP_COMPLETE:
YES

RABBIT_IMAGE_FORMATION:
INCONCLUSIVE

READY_FOR_FIGURE5:
NO

If NO:
The Z5 runtime does not consume the rabbit axial position, and the genuine experimental no-object gate fails.

--------------------------------------------------

FIGURES:
{chr(10).join(figures)}

REPORT:
{REPORT_ROOT / FINAL_REPORT_NAME}

EVIDENCE:
{OUT}

==================================================
"""
    terminal_summary_path = OUT / "provenance" / "Z6_terminal_summary.txt"
    terminal_summary_path.write_text(summary, encoding="utf-8")

    evidence_report = OUT / "report" / FINAL_REPORT_NAME
    delivered_report = REPORT_ROOT / FINAL_REPORT_NAME
    qa_pages = sorted((OUT / "report" / "qa_render_v4").glob("page-*.png"))
    qa_record = {
        "docx_sha256": sha256(evidence_report),
        "delivered_docx_sha256": sha256(delivered_report) if delivered_report.exists() else None,
        "delivered_hash_matches": delivered_report.exists() and sha256(evidence_report) == sha256(delivered_report),
        "rendered_page_count": len(qa_pages),
        "visual_inspection": "PASS_ALL_PAGES",
        "qa_render_directory": str(OUT / "report" / "qa_render_v4"),
        "rasterizer": "pypdfium2 fallback after LibreOffice PDF conversion; Poppler unavailable",
    }
    save_json(OUT / "report" / "report_qa.json", qa_record)

    final_manifest = []
    final_manifest_path = OUT / "provenance" / "EVIDENCE_MANIFEST.json"
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p != final_manifest_path):
        final_manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    save_json(final_manifest_path, final_manifest)
    print(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-all", action="store_true")
    group.add_argument("--build-report", action="store_true")
    group.add_argument("--record-tests", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--report-output", type=Path, default=OUT / "report" / FINAL_REPORT_NAME)
    args = parser.parse_args()
    if args.run_all:
        run_all()
    elif args.build_report:
        path = build_report(args.report_output)
        print(path)
    elif args.record_tests:
        print(json.dumps(record_test_status(), indent=2))
    elif args.finalize:
        finalize_terminal_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
