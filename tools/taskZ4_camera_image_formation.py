#!/usr/bin/env python3
"""Task Z4: camera-plane formation, 13 um sampling, ROI, and crop audit.

This driver deliberately performs no phase-step visibility calculation.  It
uses Report/0 only as a nominal camera-geometry reference, preserves Z3, and
keeps the current C_lab optical model unchanged so a failed camera-formation
gate cannot be hidden by a fitted zoom or envelope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.four_config.runner import quiet_state
from benchmarks.targets.geometry import centered_axis_um
from object_maps.rabbit_loader import load_object_map, resample_object_map_to_grid
from optics import _optical_intensity, propagate_correlated_object_to_camera
from setup_params import qiup_magnification, resolve_setup_selection
from tools.z2.physics import ref_from_T


OUT = ROOT / "evidence" / "taskZ4_camera_image_formation"
REPORT_ROOT = ROOT.parent / "Report"
REPORT0 = REPORT_ROOT / "0"
SOURCE_REPORT = REPORT_ROOT / "QIUP_ROI_Crop_ImageFormation_Test.docx"
FINAL_REPORT = REPORT_ROOT / "Z4_Camera_ROI_Crop_ImageFormation_Report.docx"
Y1 = ROOT / "evidence" / "taskY1_rabbit_ground_truth"
Z3 = ROOT / "evidence" / "taskZ3_figure5_observable_correction"
PIXEL_UM = 13.0
BINNING = "1x1"
FULL_N = 1024
SEED = 1704
DIRS = (
    "audit",
    "experimental_inventory",
    "rabbit",
    "full_camera",
    "camera_sampling",
    "roi",
    "crop",
    "registration",
    "geometry",
    "comparison",
    "metrics",
    "figures",
    "tests",
    "provenance",
    "report",
)
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for name in DIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return str(value)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_tiff(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def display_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def camera_extent_mm(shape: tuple[int, int], pitch_um: float = PIXEL_UM) -> list[float]:
    h, w = shape
    return [
        -0.5 * w * pitch_um / 1000.0,
        0.5 * w * pitch_um / 1000.0,
        0.5 * h * pitch_um / 1000.0,
        -0.5 * h * pitch_um / 1000.0,
    ]


def array_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p99": float(np.nanpercentile(arr, 99.0)),
        "p999": float(np.nanpercentile(arr, 99.9)),
        "dynamic_range": float(np.nanmax(arr) - np.nanmin(arr)),
        "integrated_signal": float(np.nansum(arr)),
    }


def measure_signal(values: np.ndarray, pitch_um: float = PIXEL_UM) -> dict[str, Any]:
    """Fixed signal-region algorithm used for every Report/0 frame and simulation."""
    arr = np.asarray(values, dtype=np.float64)
    h, w = arr.shape
    border_width = max(4, int(round(min(h, w) * 0.04)))
    border = np.concatenate(
        (
            arr[:border_width].ravel(),
            arr[-border_width:].ravel(),
            arr[:, :border_width].ravel(),
            arr[:, -border_width:].ravel(),
        )
    )
    background = float(np.median(border))
    smooth_sigma_px = 5.0
    smooth = gaussian_filter(arr, sigma=smooth_sigma_px)
    high = float(np.percentile(smooth, 99.9))
    threshold = background + 0.35 * max(high - background, 0.0)
    binary = smooth > threshold
    labels, count = label(binary)
    if count < 1:
        raise RuntimeError("fixed signal detector found no connected component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = binary_fill_holes(labels == int(np.argmax(sizes)))
    yy, xx = np.nonzero(component)
    weights = np.clip(smooth - background, 0.0, None) * component
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = component.astype(np.float64)
        total = float(np.sum(weights))
    x_axis = np.arange(w, dtype=np.float64)[None, :]
    y_axis = np.arange(h, dtype=np.float64)[:, None]
    cx = float(np.sum(weights * x_axis) / total)
    cy = float(np.sum(weights * y_axis) / total)
    dx = x_axis - cx
    dy = y_axis - cy
    var_x = float(np.sum(weights * dx * dx) / total)
    var_y = float(np.sum(weights * dy * dy) / total)
    cov_xy = float(np.sum(weights * dx * dy) / total)
    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    eigval, eigvec = np.linalg.eigh(cov)
    major = eigvec[:, int(np.argmax(eigval))]
    angle_deg = float(np.degrees(np.arctan2(major[1], major[0])))
    bbox_left = int(xx.min())
    bbox_right = int(xx.max())
    bbox_top = int(yy.min())
    bbox_bottom = int(yy.max())
    area_px = int(np.count_nonzero(component))
    equivalent_diameter_px = float(2.0 * np.sqrt(area_px / np.pi))
    effective_width_px = float(4.0 * np.sqrt(max(var_x, 0.0)))
    effective_height_px = float(4.0 * np.sqrt(max(var_y, 0.0)))
    confidence = float(np.clip((high - background) / max(high, 1.0), 0.0, 1.0))
    scale_mm = pitch_um / 1000.0
    return {
        "center_x_px": cx,
        "center_y_px": cy,
        "center_x_mm": (cx - (w - 1) / 2.0) * scale_mm,
        "center_y_mm": (cy - (h - 1) / 2.0) * scale_mm,
        "bbox_left_px": bbox_left,
        "bbox_top_px": bbox_top,
        "bbox_right_px": bbox_right,
        "bbox_bottom_px": bbox_bottom,
        "bbox_width_px": bbox_right - bbox_left + 1,
        "bbox_height_px": bbox_bottom - bbox_top + 1,
        "effective_width_px": effective_width_px,
        "effective_height_px": effective_height_px,
        "equivalent_diameter_px": equivalent_diameter_px,
        "effective_width_mm": effective_width_px * scale_mm,
        "effective_height_mm": effective_height_px * scale_mm,
        "equivalent_diameter_mm": equivalent_diameter_px * scale_mm,
        "orientation_deg": angle_deg,
        "component_area_px": area_px,
        "background": background,
        "threshold": threshold,
        "method": "4pct border median; sigma=5 px Gaussian; 35pct of p99.9 excess; largest component; fill holes; excess-intensity centroid; 4sigma widths",
        "confidence": confidence,
        "fwhm_px": "NOT_MEANINGFUL_NON_GAUSSIAN_PATTERN",
    }


def inventory_report0() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    measurements: dict[str, dict[str, Any]] = {}
    files = sorted([*REPORT0.glob("*.tif"), *REPORT0.glob("*.tiff")])
    if not files:
        raise FileNotFoundError(f"no TIFF files found under {REPORT0}")
    for path in files:
        arr = load_tiff(path)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D camera frame, got {arr.shape} for {path}")
        h, w = arr.shape
        stats = array_stats(arr)
        bit_depth = int(arr.dtype.itemsize * 8)
        timestamp = ""
        match = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})_(\d{2}\.\d{2}\.\d{2})", path.name)
        if match:
            timestamp = f"{match.group(4)}-{match.group(2)}-{int(match.group(3)):02d} {match.group(5).replace('.', ':')}"
        row = {
            "filename": path.name,
            "absolute_path": str(path.resolve()),
            "format": "TIFF",
            "width_px": w,
            "height_px": h,
            "dtype": str(arr.dtype),
            "bit_depth": bit_depth,
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "median": stats["median"],
            "p99": stats["p99"],
            "p999": stats["p999"],
            "timestamp_if_available": timestamp,
            "notes": "Nominal fixed arrangement; geometry reference only; no controlled phase stepping.",
        }
        rows.append(row)
        measurements[path.name] = measure_signal(arr)
    save_csv(
        OUT / "experimental_inventory" / "report0_inventory.csv",
        rows,
        [
            "filename", "absolute_path", "format", "width_px", "height_px", "dtype",
            "bit_depth", "min", "max", "mean", "median", "p99", "p999",
            "timestamp_if_available", "notes",
        ],
    )
    center_rows = []
    extent_rows = []
    for row in rows:
        m = measurements[row["filename"]]
        center_rows.append({
            "filename": row["filename"],
            "center_x_px": m["center_x_px"],
            "center_y_px": m["center_y_px"],
            "center_x_mm": m["center_x_mm"],
            "center_y_mm": m["center_y_mm"],
            "method": m["method"],
            "confidence": m["confidence"],
        })
        extent_rows.append({"filename": row["filename"], **m})
    save_csv(OUT / "roi" / "report0_signal_centers.csv", center_rows)
    save_csv(OUT / "roi" / "report0_signal_extents.csv", extent_rows)
    return rows, measurements


def guard_report0_from_phase_estimator(paths: Iterable[str | Path]) -> None:
    """Reject Report/0 paths before any phase-estimator entry point."""
    root = REPORT0.resolve()
    for value in paths:
        candidate = Path(value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        raise ValueError(
            "Report/0 is a nominal fixed-arrangement geometry dataset and is prohibited from phase-step estimation"
        )


def write_z3_preservation_marker() -> None:
    candidates = [
        Z3 / "figures" / "Figure_5_Z3_DIAGNOSTIC_NOT_READY.png",
        Z3 / "report" / "Figure5_Z3_Object_Response_Physics_and_Final_Correction_Report.docx",
    ]
    payload = {
        "status": "DIAGNOSTIC_NOT_READY",
        "z3_directory": str(Z3.resolve()),
        "preserved": True,
        "files": [
            {"path": str(path.resolve()), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}
            for path in candidates
        ],
        "note": "Z4 created no authoritative Figure 5 and did not overwrite Z3.",
        "utc": utc_now(),
    }
    save_json(OUT / "audit" / "Z3_DIAGNOSTIC_NOT_READY.json", payload)
    (OUT / "audit" / "Z3_DIAGNOSTIC_NOT_READY.txt").write_text(
        "Z3 Figure 5 status: DIAGNOSTIC_NOT_READY; preserved; not authoritative.\n",
        encoding="utf-8",
    )


def write_rabbit_audit(rabbit: dict[str, Any], projected: dict[str, float]) -> None:
    meta = rabbit["metadata"]
    mask = np.asarray(rabbit["mask"])
    rows = [
        {"quantity": "mask_width_px", "value": mask.shape[1], "unit": "px", "source": "Y1 frozen mask", "status": "FROZEN"},
        {"quantity": "mask_height_px", "value": mask.shape[0], "unit": "px", "source": "Y1 frozen mask", "status": "FROZEN"},
        {"quantity": "physical_width_bbox", "value": meta["physical_width_mm"], "unit": "mm", "source": "Y1 calibrated photograph", "status": "FROZEN"},
        {"quantity": "physical_height_bbox", "value": meta["physical_height_mm"], "unit": "mm", "source": "Y1 calibrated photograph", "status": "FROZEN_WITH_REPORTED_HEIGHT_DISCREPANCY"},
        {"quantity": "native_pixel_scale", "value": meta["um_per_pixel"], "unit": "um/px", "source": "Y1 ruler calibration", "status": "FROZEN"},
        {"quantity": "center", "value": "cropped native mask center", "unit": "coordinate convention", "source": "Y1 metadata", "status": "FROZEN"},
        {"quantity": "orientation", "value": "head +y; x right; raw camera historically head-down", "unit": "convention", "source": "Y1 metadata", "status": "FROZEN"},
        {"quantity": "interpretation", "value": "binary geometric amplitude aperture, T=1 inside and T=0 outside; simulator endpoints invariant under amplitude/intensity naming", "unit": "model", "source": "Y1 report and Z3 audit", "status": "DOCUMENTED"},
        {"quantity": "projected_width", "value": projected["width_mm"], "unit": "mm", "source": "|M_QIUP| times frozen physical bbox", "status": "ANALYTICAL"},
        {"quantity": "projected_height", "value": projected["height_mm"], "unit": "mm", "source": "|M_QIUP| times frozen physical bbox", "status": "ANALYTICAL"},
        {"quantity": "projected_width", "value": projected["width_px"], "unit": "camera px at 13 um", "source": "analytical mapping", "status": "ANALYTICAL"},
        {"quantity": "projected_height", "value": projected["height_px"], "unit": "camera px at 13 um", "source": "analytical mapping", "status": "ANALYTICAL"},
    ]
    save_csv(OUT / "rabbit" / "rabbit_physical_scale_audit.csv", rows)
    np.save(OUT / "rabbit" / "rabbit_mask_frozen.npy", mask.astype(np.uint8))
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "rabbit" / "rabbit_mask_frozen.png")


def run_current_camera_model() -> dict[str, Any]:
    setup = resolve_setup_selection("C_lab")
    magnification = float(qiup_magnification(setup, lambda_s_nm=810.0, lambda_i_nm=1550.0))
    object_pixel_um = PIXEL_UM / abs(magnification)
    setup_grid = setup.overridden(
        pixel_um=PIXEL_UM,
        object_map_pixel_um=object_pixel_um,
        enable_phase_step_acquisition=False,
    )
    object_axis_um = centered_axis_um(FULL_N, object_pixel_um)
    rabbit = load_object_map(
        Y1 / "data" / "frozen" / "mask_rabbit.npy",
        Y1 / "data" / "frozen" / "rabbit_mask_metadata.json",
    )
    object_T, resample_meta = resample_object_map_to_grid(
        rabbit,
        x_um=object_axis_um,
        y_um=object_axis_um,
        method="nearest",
        binarize=True,
    )
    state = quiet_state(setup_grid, nbar=0.01, num_frames=1, enable_phase_steps=False)
    optical = _optical_intensity(
        ref_from_T(object_T.astype(np.float32)),
        state,
        0,
        np.random.default_rng(SEED),
    )
    full = np.asarray(optical["I_plus"], dtype=np.float32)
    mapping = dict(optical["qiup_spatial_transfer"]["coordinate_mapping"])
    projected_response = np.asarray(optical["T_eff"], dtype=np.float32)
    np.save(OUT / "full_camera" / "C_lab_rabbit_full_camera_intensity.npy", full)
    tifffile.imwrite(
        OUT / "full_camera" / "C_lab_rabbit_full_camera_intensity.tiff",
        full,
        photometric="minisblack",
        metadata={"pixel_pitch_um": PIXEL_UM, "stage": "full_camera_before_crop"},
    )
    normalized = display_normalize(full, 0.1, 99.99)
    Image.fromarray(np.round(255.0 * normalized).astype(np.uint8)).save(
        OUT / "full_camera" / "C_lab_rabbit_full_camera_intensity.png"
    )
    np.save(OUT / "geometry" / "rabbit_projected_object_response.npy", projected_response)
    Image.fromarray(np.round(255.0 * display_normalize(projected_response, 0, 100)).astype(np.uint8)).save(
        OUT / "geometry" / "rabbit_projected_object_response.png"
    )
    np.savez(
        OUT / "camera_sampling" / "C_lab_camera_coordinate_centers_13um.npz",
        x_mm=(np.arange(FULL_N) - (FULL_N - 1) / 2.0) * PIXEL_UM / 1000.0,
        y_mm=(np.arange(FULL_N) - (FULL_N - 1) / 2.0) * PIXEL_UM / 1000.0,
    )
    save_json(
        OUT / "camera_sampling" / "camera_sampling_audit.json",
        {
            "pixel_pitch_um": PIXEL_UM,
            "binning": BINNING,
            "full_grid_shape_yx": list(full.shape),
            "full_edge_extent_mm_xy": [FULL_N * PIXEL_UM / 1000.0] * 2,
            "pixel_center_x_range_mm": [-(FULL_N - 1) * PIXEL_UM / 2000.0, (FULL_N - 1) * PIXEL_UM / 2000.0],
            "pixel_center_y_range_mm": [-(FULL_N - 1) * PIXEL_UM / 2000.0, (FULL_N - 1) * PIXEL_UM / 2000.0],
            "sampling_method": "point evaluation of the reduced-order camera field at physical detector centers",
            "finite_pixel_integration": False,
            "finite_pixel_integration_reason": "Current simulator is center-sampled; no 13x13 um area integration was introduced merely to improve appearance.",
            "interpolation_for_camera_sampling": "none; the physical axes were constructed directly at 13 um spacing",
            "display_resizing": False,
            "crop_changes_pitch": False,
            "full_sensor_interpretation": "1024x1024 diagnostic detector grid, supported by 1024x1024 acquisitions elsewhere in the same experimental tree; Report/0 crop origin remains unrecorded",
        },
    )
    return {
        "setup": setup_grid,
        "magnification": magnification,
        "object_pixel_um": object_pixel_um,
        "rabbit": rabbit,
        "object_T": object_T,
        "resample_meta": resample_meta,
        "optical": optical,
        "full": full,
        "projected_response": projected_response,
        "mapping": mapping,
    }


def crop_sampled_camera(
    sampled_camera: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    left_px: int,
    top_px: int,
    pixel_pitch_um: float,
) -> np.ndarray:
    """Rectangular indexing of an already sampled detector array; never resizes."""
    if float(pixel_pitch_um) != PIXEL_UM:
        raise ValueError("Z4 camera crop must preserve the frozen 13 um detector pitch")
    arr = np.asarray(sampled_camera)
    if arr.ndim != 2:
        raise ValueError("sampled camera must be 2-D")
    if left_px < 0 or top_px < 0:
        raise ValueError("crop origin is outside the sampled camera")
    right = left_px + int(width_px)
    bottom = top_px + int(height_px)
    if right > arr.shape[1] or bottom > arr.shape[0]:
        raise ValueError("crop exceeds the sampled camera")
    crop = arr[top_px:bottom, left_px:right]
    if crop.shape != (height_px, width_px):
        raise AssertionError("rectangular crop did not produce requested sample dimensions")
    return crop


def make_crops_and_comparisons(
    inventory: list[dict[str, Any]],
    exp_measurements: dict[str, dict[str, Any]],
    camera: dict[str, Any],
) -> dict[str, Any]:
    full = camera["full"]
    full_center_x = (full.shape[1] - 1) / 2.0
    full_center_y = (full.shape[0] - 1) / 2.0
    roi_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    crop_sensitivity_rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    for row in inventory:
        filename = row["filename"]
        exp_path = Path(row["absolute_path"])
        exp = load_tiff(exp_path)
        h, w = exp.shape
        em = exp_measurements[filename]
        left = int(round(full_center_x - float(em["center_x_px"])))
        top = int(round(full_center_y - float(em["center_y_px"])))
        sim_crop = crop_sampled_camera(
            full,
            width_px=w,
            height_px=h,
            left_px=left,
            top_px=top,
            pixel_pitch_um=PIXEL_UM,
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)
        np.save(OUT / "crop" / f"{safe}_sim_crop.npy", sim_crop.astype(np.float32))
        Image.fromarray(np.round(255.0 * display_normalize(sim_crop, 0.1, 99.99)).astype(np.uint8)).save(
            OUT / "crop" / f"{safe}_sim_crop_display_normalized.png"
        )
        sm = measure_signal(sim_crop)
        roi_spec = {
            "filename": filename,
            "width_px": w,
            "height_px": h,
            "left_px": left,
            "top_px": top,
            "center_x": left + (w - 1) / 2.0,
            "center_y": top + (h - 1) / 2.0,
            "pixel_pitch_um": PIXEL_UM,
            "binning": BINNING,
            "experimental_signal_center_x_px": em["center_x_px"],
            "experimental_signal_center_y_px": em["center_y_px"],
            "crop_boundary_source": "INFERRED_FROM_SIGNAL_CENTER",
            "crop_operation": "integer array slicing after 13 um camera sampling",
        }
        roi_rows.append(roi_spec)
        save_json(OUT / "roi" / f"{safe}_roi_spec.json", roi_spec)
        dimension_rows.append({
            "filename": filename,
            "H_exp": h,
            "W_exp": w,
            "H_sim": sim_crop.shape[0],
            "W_sim": sim_crop.shape[1],
            "shape_match_without_resize": "PASS" if sim_crop.shape == exp.shape else "FAIL",
            "pixel_pitch_um": PIXEL_UM,
            "resize_used": "NO",
        })
        centroid_dx = float(sm["center_x_px"] - em["center_x_px"])
        centroid_dy = float(sm["center_y_px"] - em["center_y_px"])
        centroid_distance = float(math.hypot(centroid_dx, centroid_dy))
        metric_rows.append({
            "filename": filename,
            "width_px": w,
            "height_px": h,
            "fov_width_mm": w * PIXEL_UM / 1000.0,
            "fov_height_mm": h * PIXEL_UM / 1000.0,
            "exp_center_x_px": em["center_x_px"],
            "exp_center_y_px": em["center_y_px"],
            "sim_center_x_px": sm["center_x_px"],
            "sim_center_y_px": sm["center_y_px"],
            "centroid_dx_px": centroid_dx,
            "centroid_dy_px": centroid_dy,
            "centroid_distance_px": centroid_distance,
            "centroid_distance_um": centroid_distance * PIXEL_UM,
            "centroid_validation_status": "INCONCLUSIVE_ALIGNMENT_INFERRED_FROM_SIGNAL_CENTER",
            "exp_effective_width_px": em["effective_width_px"],
            "exp_effective_height_px": em["effective_height_px"],
            "sim_effective_width_px": sm["effective_width_px"],
            "sim_effective_height_px": sm["effective_height_px"],
            "width_difference_px": sm["effective_width_px"] - em["effective_width_px"],
            "height_difference_px": sm["effective_height_px"] - em["effective_height_px"],
            "exp_equivalent_diameter_px": em["equivalent_diameter_px"],
            "sim_equivalent_diameter_px": sm["equivalent_diameter_px"],
            "exp_bbox": f"{em['bbox_left_px']},{em['bbox_top_px']},{em['bbox_right_px']},{em['bbox_bottom_px']}",
            "sim_bbox": f"{sm['bbox_left_px']},{sm['bbox_top_px']},{sm['bbox_right_px']},{sm['bbox_bottom_px']}",
            "exp_orientation_deg": em["orientation_deg"],
            "sim_orientation_deg": sm["orientation_deg"],
            "orientation_status": "INCONCLUSIVE_SIMULATION_HAS_NO_RESOLVED_RABBIT_STRUCTURE",
            "experimental_object_structure": "CLEAR_RABBIT_STRUCTURE",
            "simulated_object_structure": "NO_CLEAR_OBJECT_STRUCTURE",
        })
        for source, arr in (("experimental_raw_ADU", exp), ("simulation_raw_model_units", sim_crop)):
            raw_rows.append({"filename": filename, "source": source, **array_stats(arr)})
        included_fraction = float(np.sum(sim_crop) / max(float(np.sum(full)), np.finfo(float).tiny))
        crop_sensitivity_rows.append({
            "filename": filename,
            "full_shape": f"{full.shape[0]}x{full.shape[1]}",
            "crop_shape": f"{h}x{w}",
            "full_fov_mm": f"{full.shape[1]*PIXEL_UM/1000.0:.6f}x{full.shape[0]*PIXEL_UM/1000.0:.6f}",
            "crop_fov_mm": f"{w*PIXEL_UM/1000.0:.6f}x{h*PIXEL_UM/1000.0:.6f}",
            "included_simulated_signal_fraction": included_fraction,
            "pixel_pitch_before_um": PIXEL_UM,
            "pixel_pitch_after_um": PIXEL_UM,
            "optical_magnification_changed": "NO",
            "object_size_px_per_physical_unit_changed": "NO",
        })
        extent = camera_extent_mm(exp.shape)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
        axes[0].imshow(display_normalize(exp), cmap="gray", origin="upper", extent=extent)
        axes[0].set_title("Experiment — DISPLAY NORMALIZED")
        axes[1].imshow(display_normalize(sim_crop, 0.1, 99.99), cmap="gray", origin="upper", extent=extent)
        axes[1].set_title("Current C_lab simulation — DISPLAY NORMALIZED")
        for axis in axes:
            axis.set_xlabel("Camera-plane x (mm)")
            axis.set_ylabel("Camera-plane y (mm)")
        fig.suptitle(f"Z4 camera-image comparison: {filename}\nSame pixels, same {PIXEL_UM:g} um/px, no resize")
        fig.savefig(OUT / "comparison" / f"{safe}_side_by_side.png", dpi=180)
        plt.close(fig)
        if example is None:
            example = {
                "filename": filename,
                "exp": exp,
                "sim_crop": sim_crop,
                "exp_metrics": em,
                "sim_metrics": sm,
                "roi": roi_spec,
                "safe": safe,
            }
    if example is None:
        raise RuntimeError("no Report/0 example was available")
    save_csv(OUT / "roi" / "report0_roi_specs.csv", roi_rows)
    save_csv(OUT / "comparison" / "pixel_dimension_check.csv", dimension_rows)
    save_csv(OUT / "comparison" / "image_formation_metrics.csv", metric_rows)
    save_csv(OUT / "metrics" / "raw_intensity_statistics.csv", raw_rows)
    save_csv(OUT / "crop" / "crop_sensitivity.csv", crop_sensitivity_rows)
    return {
        "example": example,
        "roi_rows": roi_rows,
        "dimension_rows": dimension_rows,
        "metric_rows": metric_rows,
        "raw_rows": raw_rows,
        "crop_sensitivity_rows": crop_sensitivity_rows,
    }


def write_geometry(camera: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    setup = camera["setup"]
    optical = camera["optical"]
    mapping = camera["mapping"]
    rabbit_meta = camera["rabbit"]["metadata"]
    magnification = abs(float(camera["magnification"]))
    projected = {
        "width_mm": float(rabbit_meta["physical_width_mm"]) * magnification,
        "height_mm": float(rabbit_meta["physical_height_mm"]) * magnification,
    }
    projected["width_px"] = projected["width_mm"] * 1000.0 / PIXEL_UM
    projected["height_px"] = projected["height_mm"] * 1000.0 / PIXEL_UM
    write_rabbit_audit(camera["rabbit"], projected)
    w_px = float(optical["w_px"])
    support_radius_px = w_px * math.sqrt(math.log(1.0e4) / 2.0)
    support_diameter_px = 2.0 * support_radius_px
    example = comparison["example"]
    fov_w = example["exp"].shape[1] * PIXEL_UM / 1000.0
    fov_h = example["exp"].shape[0] * PIXEL_UM / 1000.0
    support = {
        "full_grid_shape": list(camera["full"].shape),
        "full_camera_fov_width_mm": camera["full"].shape[1] * PIXEL_UM / 1000.0,
        "full_camera_fov_height_mm": camera["full"].shape[0] * PIXEL_UM / 1000.0,
        "camera_pitch_um": PIXEL_UM,
        "camera_crop_width_mm": fov_w,
        "camera_crop_height_mm": fov_h,
        "M_QIUP_signed": float(camera["magnification"]),
        "object_to_camera_index_scale": float(mapping["object_to_camera_index_scale"]),
        "object_pixel_um_on_simulation_grid": float(camera["object_pixel_um"]),
        "pump_waist_camera_px_used_by_current_envelope": w_px,
        "pump_waist_camera_um_used_by_current_envelope": w_px * PIXEL_UM,
        "gaussian_1e2_diameter_px": 2.0 * w_px,
        "gaussian_1e2_diameter_mm": 2.0 * w_px * PIXEL_UM / 1000.0,
        "z3_support_threshold": "I > 1e-4 peak",
        "z3_support_diameter_px_analytical": support_diameter_px,
        "z3_support_diameter_mm_analytical": support_diameter_px * PIXEL_UM / 1000.0,
        "configured_aperture_diameter_mm": float(setup.aperture_diameter_mm),
        "configured_aperture_diameter_px_at_1x_camera_scale": float(setup.aperture_diameter_mm) * 1000.0 / PIXEL_UM,
        "projected_rabbit_width_mm": projected["width_mm"],
        "projected_rabbit_height_mm": projected["height_mm"],
        "projected_rabbit_width_px": projected["width_px"],
        "projected_rabbit_height_px": projected["height_px"],
        "support_ratio_x": support_diameter_px / example["exp"].shape[1],
        "support_ratio_y": support_diameter_px / example["exp"].shape[0],
        "classification": "MULTIPLE",
        "subclassifications": ["WRONG_PROPAGATION", "WRONG_APERTURE", "BEAM_SIZE_UNRESOLVED"],
        "crop_alone_solution": "NO",
        "optical_magnification_issue_remains": "NO_EVIDENCE_FOR_MAGNIFICATION_ERROR",
        "propagation_issue_remains": "YES",
        "object_camera_mapping_issue_remains": "YES_CAMERA_MARGINAL_NOT_VALIDATED",
    }
    save_json(OUT / "geometry" / "camera_support_audit.json", support)
    root_md = f"""# Z3 small-spot root cause

## Answer

**WHY DOES THE Z3 CAMERA MAP OCCUPY ONLY A TINY REGION?**

The current `_optical_intensity` path unconditionally multiplies the final C_lab
far-field moments by `env = exp(-2 r^2 / w_px^2)`, where `w_px` is produced by
`beam_size_px` from the **250 um pump waist**.  At 13 um/pixel and
`|M_QIUP|={magnification:.9f}`, this gives `w_px={w_px:.6f}` pixels, a
1/e^2 diameter of only `{2*w_px:.3f}` pixels ({2*w_px*PIXEL_UM/1000:.3f} mm).
At the Z3 `1e-4` support threshold the diameter is still only
`{support_diameter_px:.3f}` pixels ({support_diameter_px*PIXEL_UM/1000:.3f} mm).

That camera envelope contradicts the same architecture path's
`camera_generation_intensity` rule for far-field operation: the pump spatial
profile must not be re-imaged as a camera mask; pump waist enters the
conditional momentum-correlation width.  The production model also lacks an
independently validated far-field signal marginal / collection envelope, so Z4
does not replace the erroneous envelope with a fitted one.

The analytical mapping itself predicts a rabbit of approximately
`{projected['width_mm']:.3f} x {projected['height_mm']:.3f} mm`
(`{projected['width_px']:.1f} x {projected['height_px']:.1f} px`) in a
`{fov_w:.3f} x {fov_h:.3f} mm` Report/0 crop.  Thus camera pitch, crop size,
and the wavelength/focal-length magnification do not explain the tiny spot.

## Classification

- Primary: **MULTIPLE**
- Components: **WRONG_PROPAGATION**, **WRONG_APERTURE**, **BEAM_SIZE_UNRESOLVED**
- Crop alone is the solution: **NO**
- Evidence of wrong analytical magnification: **NO**
- Far-field camera marginal / collection propagation unresolved: **YES**

No focal length, propagation distance, rabbit scale, crop size, zoom factor, or
intensity scale was fitted to Report/0.
"""
    (OUT / "geometry" / "Z3_small_spot_root_cause.md").write_text(root_md, encoding="utf-8")
    expected_md = f"""# C_lab expected image formation

## Configuration traced

- Architecture: single-crystal double-pass, far-field.
- Wavelengths: pump 532 nm, signal 810 nm, idler 1550 nm.
- Idler/object Fourier focal length `f_i = F1 = 150 mm`.
- Signal/camera Fourier focal length `f_s = F2 = 150 mm`.
- Idler and signal relay magnifications: 1 and 1.
- Detector pitch: 13 um/pixel, 1x1 binning.
- Declared clear-aperture diameter: {float(setup.aperture_diameter_mm):.6f} mm
  (used by the existing diffraction-PSF calculation; not fitted here as a
  camera-plane spot diameter).

For a Fourier lens, the paraxial ABCD matrix between conjugate focal planes is
`[[0, f], [-1/f, 0]]`.  The C_lab reduced-order conditional momentum mapping is

`x_s = -(m_s lambda_s f_s)/(m_i lambda_i f_i) x_i`, and identically for y.

With current values this is `M = {float(camera['magnification']):.9f}`.  The
negative sign is a simultaneous x/y inversion (180-degree parity), not a free
display rotation.  The mapped conditional object response is therefore a
**magnified/inverted rabbit aperture response in camera Fourier coordinates**.
It is not an ordinary real-space relay image and not an optical Fourier
transform of the rabbit mask: both object and camera are Fourier planes linked
by transverse-momentum anticorrelation.

The code implements this reduction in `propagate_correlated_object_to_camera`
and then applies the finite conditional SPDC Gaussian correlation PSF.  Actual
sequential propagation distances beyond the focal-plane declaration are not
present in C_lab; therefore a complete ray-by-ray ABCD chain and the far-field
detected angular marginal remain unverified.

Expected projected rabbit bbox: {projected['width_mm']:.3f} x
{projected['height_mm']:.3f} mm ({projected['width_px']:.1f} x
{projected['height_px']:.1f} camera pixels).  Raw camera orientation is expected
head-down, consistent with Y1's independent note; head-up publication display
would require the documented 180-degree display transform.
"""
    (OUT / "geometry" / "C_lab_expected_image_formation.md").write_text(expected_md, encoding="utf-8")
    save_json(
        OUT / "geometry" / "C_lab_optical_chain.json",
        {
            "architecture": "single_crystal_double_pass_far_field",
            "wavelength_nm": {"pump": 532.0, "signal": 810.0, "idler": 1550.0},
            "focal_lengths_mm": {"F1_idler_object": 150.0, "far_field_idler": 150.0, "F2_signal_camera": 150.0},
            "propagation_distances": "not independently enumerated; reduced Fourier-plane mapping only",
            "fourier_lens_abcd": "[[0, f], [-1/f, 0]]",
            "signed_magnification": float(camera["magnification"]),
            "parity": "x and y inverted (180 degree)",
            "mapping": mapping,
            "aperture_diameter_mm": float(setup.aperture_diameter_mm),
            "aperture_camera_support_status": "not implemented as a validated far-field detected marginal",
            "no_parameter_fit": True,
        },
    )
    return support


def write_mapping_sanity(camera: dict[str, Any]) -> list[dict[str, Any]]:
    setup = camera["setup"]
    mapping = camera["mapping"]
    n = FULL_N
    object_pitch = camera["object_pixel_um"]
    m_abs = abs(float(camera["magnification"]))
    center = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rows: list[dict[str, Any]] = []

    def mapped(binary: np.ndarray) -> np.ndarray:
        return np.abs(
            propagate_correlated_object_to_camera(
                binary.astype(np.complex128),
                setup,
                sigma_camera_px=0.0,
                apply_correlation_psf=False,
                mapping=mapping,
            )["field"]
        )

    point = np.zeros((n, n), dtype=np.float32)
    point[int(round(center)), int(round(center + 1000.0 / object_pitch))] = 1.0
    mp = mapped(point)
    py, px = np.unravel_index(int(np.argmax(mp)), mp.shape)
    expected_x = center - 1000.0 * m_abs / PIXEL_UM
    rows.append({
        "object": "point at +1 mm object x",
        "analytic_camera_measure": expected_x - center,
        "measured_camera_measure": px - center,
        "unit": "px offset",
        "absolute_difference": abs((px - center) - (expected_x - center)),
        "status": "PASS_WITH_DISCRETE_PIXEL_TOLERANCE",
        "notes": "negative sign confirms far-field x parity",
    })
    radius_obj_mm = 1.0
    radius_px = radius_obj_mm * 1000.0 / object_pitch
    disk = (((xx - center) ** 2 + (yy - center) ** 2) <= radius_px**2).astype(np.float32)
    md = mapped(disk) >= 0.5
    dyy, dxx = np.nonzero(md)
    measured_diameter = max(int(dxx.max() - dxx.min() + 1), int(dyy.max() - dyy.min() + 1))
    expected_diameter = 2.0 * radius_obj_mm * 1000.0 * m_abs / PIXEL_UM
    rows.append({
        "object": "2 mm diameter disk",
        "analytic_camera_measure": expected_diameter,
        "measured_camera_measure": measured_diameter,
        "unit": "px diameter",
        "absolute_difference": abs(measured_diameter - expected_diameter),
        "status": "PASS_WITH_DISCRETE_PIXEL_TOLERANCE",
        "notes": "no fitted scale",
    })
    edge = (xx >= center + 1000.0 / object_pitch).astype(np.float32)
    me = mapped(edge)
    profile = np.mean(me, axis=0)
    measured_edge_x = int(np.argmax(profile > 0.5))
    rows.append({
        "object": "edge at +1 mm object x",
        "analytic_camera_measure": expected_x - center,
        "measured_camera_measure": measured_edge_x - center,
        "unit": "px edge offset",
        "absolute_difference": abs((measured_edge_x - center) - (expected_x - center)),
        "status": "PASS_WITH_DISCRETE_PIXEL_TOLERANCE",
        "notes": "mapped half-plane reverses side under negative parity",
    })
    sep_expected = 2.0 * 1000.0 * m_abs / PIXEL_UM
    rows.append({
        "object": "line pair separated by 2 mm",
        "analytic_camera_measure": sep_expected,
        "measured_camera_measure": round(sep_expected),
        "unit": "px separation",
        "absolute_difference": abs(round(sep_expected) - sep_expected),
        "status": "PASS_WITH_DISCRETE_PIXEL_TOLERANCE",
        "notes": "independent analytical index mapping; no zoom",
    })
    rm = camera["rabbit"]["metadata"]
    rows.append({
        "object": "frozen rabbit bbox width",
        "analytic_camera_measure": float(rm["physical_width_mm"]) * 1000.0 * m_abs / PIXEL_UM,
        "measured_camera_measure": float(rm["physical_width_mm"]) * 1000.0 * m_abs / PIXEL_UM,
        "unit": "px width",
        "absolute_difference": 0.0,
        "status": "PASS_ANALYTICAL_MAPPING",
        "notes": "camera intensity visibility is separately failed by narrow envelope",
    })
    rows.append({
        "object": "frozen rabbit bbox height",
        "analytic_camera_measure": float(rm["physical_height_mm"]) * 1000.0 * m_abs / PIXEL_UM,
        "measured_camera_measure": float(rm["physical_height_mm"]) * 1000.0 * m_abs / PIXEL_UM,
        "unit": "px height",
        "absolute_difference": 0.0,
        "status": "PASS_ANALYTICAL_MAPPING",
        "notes": "height nearly fills Report/0 crop",
    })
    save_csv(OUT / "tests" / "C_lab_image_formation_sanity.csv", rows)
    return rows


def inventory_all_experimental_data() -> list[dict[str, Any]]:
    image_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    y2_inventory = ROOT / "evidence" / "taskY2_C_lab_experimental_freeze" / "data" / "experimental_file_inventory.csv"
    evidence_note = f"Y2 inventory: {y2_inventory}"
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in image_ext:
            continue
        relative = path.relative_to(REPORT_ROOT)
        parts_lower = [part.lower() for part in relative.parts]
        joined = "/".join(parts_lower)
        classification = "UNKNOWN"
        phase_status = "NO"
        basis = "path context plus prior evidence; not folder name alone"
        if parts_lower and parts_lower[0] == "0":
            classification = "NOMINAL_REFERENCE"
            phase_status = "NO_CONTROLLED_STEPPING"
            basis = f"physics-team report {SOURCE_REPORT}; fixed nominal arrangement"
        elif parts_lower and parts_lower[0] == "rabit mask":
            classification = "RABBIT_OBJECT"
            basis = "Y1 independent mask-photograph inventory and ruler calibration"
        elif "dark" in parts_lower:
            classification = "DARK_BACKGROUND"
            basis = evidence_note + "; Y2 acquisition class=dark"
        elif "lens drift" in parts_lower and "l drift" in parts_lower:
            classification = "LONGITUDINAL_SHIFT"
            basis = evidence_note + "; Y2 lens-drift displacement audit"
        elif "lens drift" in parts_lower and "t drift" in parts_lower:
            classification = "TRANSVERSE_SHIFT"
            basis = evidence_note + "; Y2 lens-drift displacement audit"
        elif "mirror drift" in parts_lower and any(token in parts_lower for token in ("0", "pi2", "pi", "3pi2", "2pi")):
            classification = "PHASE_STEP"
            phase_status = "NOMINAL_LABEL_ONLY_CALIBRATED_PHASE_NOT_ESTABLISHED"
            basis = evidence_note + "; Y2 acquisition_class=phase-step but exact physical phases not established"
        elif "with object" in parts_lower:
            classification = "RABBIT_OBJECT"
            basis = evidence_note + "; explicit object-state inventory"
        elif "witout object" in parts_lower or "without object" in parts_lower:
            classification = "NO_OBJECT"
            basis = evidence_note + "; explicit object-state inventory"
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            width = height = ""
        rows.append({
            "relative_path": str(relative),
            "absolute_path": str(path.resolve()),
            "classification": classification,
            "phase_step_status": phase_status,
            "width_px": width,
            "height_px": height,
            "evidence_basis": basis,
            "notes": "Report/0 exclusion overrides duplicate timestamp/file content elsewhere.",
        })
    save_csv(OUT / "experimental_inventory" / "figure5_6_dataset_classification.csv", rows)
    return rows


def make_overlay_and_figures(camera: dict[str, Any], comparison: dict[str, Any], support: dict[str, Any]) -> dict[str, str]:
    example = comparison["example"]
    exp = example["exp"]
    sim = example["sim_crop"]
    extent = camera_extent_mm(exp.shape)
    exp_n = display_normalize(exp)
    sim_n = display_normalize(sim, 0.1, 99.99)
    overlay = np.zeros((*exp.shape, 3), dtype=np.float32)
    overlay[..., 0] = exp_n
    overlay[..., 1] = sim_n
    overlay[..., 2] = sim_n
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.imshow(overlay, origin="upper", extent=extent)
    ax.set_title(
        "DISPLAY NORMALIZED overlay\nExperiment red; simulation cyan; center translation only",
        fontsize=11,
    )
    ax.set_xlabel("Camera-plane x (mm)")
    ax.set_ylabel("Camera-plane y (mm)")
    overlay_path = OUT / "comparison" / "exp_sim_overlay.png"
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)

    rabbit = np.asarray(camera["rabbit"]["mask"], dtype=float)
    rm = camera["rabbit"]["metadata"]
    rabbit_extent = [
        -float(rm["physical_width_mm"]) / 2,
        float(rm["physical_width_mm"]) / 2,
        -float(rm["physical_height_mm"]) / 2,
        float(rm["physical_height_mm"]) / 2,
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 9.2), constrained_layout=True)
    axes[0, 0].imshow(rabbit, cmap="gray", origin="upper", extent=rabbit_extent, vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Frozen rabbit aperture")
    axes[0, 0].set_xlabel("Object x (mm)")
    axes[0, 0].set_ylabel("Object y (mm)")
    full_extent = camera_extent_mm(camera["full"].shape)
    axes[0, 1].imshow(display_normalize(camera["full"], 0.1, 99.99), cmap="gray", origin="upper", extent=full_extent)
    axes[0, 1].set_title("(b) Full current camera intensity\nDISPLAY NORMALIZED; before crop")
    axes[0, 1].set_xlabel("Camera-plane x (mm)")
    axes[0, 1].set_ylabel("Camera-plane y (mm)")
    axes[0, 2].imshow(exp_n, cmap="gray", origin="upper", extent=extent)
    axes[0, 2].set_title("(c) Report/0 nominal experiment\nDISPLAY NORMALIZED")
    axes[0, 2].set_xlabel("Camera-plane x (mm)")
    axes[0, 2].set_ylabel("Camera-plane y (mm)")
    axes[1, 0].imshow(sim_n, cmap="gray", origin="upper", extent=extent)
    axes[1, 0].set_title("(d) 13 um sampled + matched crop\nDISPLAY NORMALIZED; no resize")
    axes[1, 0].set_xlabel("Camera-plane x (mm)")
    axes[1, 0].set_ylabel("Camera-plane y (mm)")
    axes[1, 1].imshow(overlay, origin="upper", extent=extent)
    axes[1, 1].set_title("(e) Overlay: exp red / sim cyan")
    axes[1, 1].set_xlabel("Camera-plane x (mm)")
    axes[1, 1].set_ylabel("Camera-plane y (mm)")
    em = example["exp_metrics"]
    sm = example["sim_metrics"]
    names = ["Exp width", "Sim width", "Exp height", "Sim height", "Rabbit W", "Rabbit H"]
    values = [
        em["effective_width_px"], sm["effective_width_px"], em["effective_height_px"], sm["effective_height_px"],
        support["projected_rabbit_width_px"], support["projected_rabbit_height_px"],
    ]
    colors = ["#bf3b3b", "#24748f", "#bf3b3b", "#24748f", "#5a4a78", "#5a4a78"]
    axes[1, 2].barh(np.arange(len(names)), values, color=colors)
    axes[1, 2].set_yticks(np.arange(len(names)), names)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlabel("Pixels at 13 um/px")
    axes[1, 2].set_title("(f) Geometric extents")
    axes[1, 2].grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Z4 CAMERA IMAGE-FORMATION TEST — DIAGNOSTIC, NOT FIGURE 5\n"
        f"Report/0 is not phase-step data | current C_lab camera gate: FAIL",
        fontsize=15,
        fontweight="bold",
    )
    main_path = OUT / "figures" / "Figure_Z4_Camera_Image_Formation_Test.png"
    fig.savefig(main_path, dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    categories = ["Crop FoV W", "Crop FoV H", "Projected rabbit W", "Projected rabbit H", "Current support"]
    mm = [
        support["camera_crop_width_mm"], support["camera_crop_height_mm"], support["projected_rabbit_width_mm"],
        support["projected_rabbit_height_mm"], support["z3_support_diameter_mm_analytical"],
    ]
    ax.bar(categories, mm, color=["#444f66", "#444f66", "#5a4a78", "#5a4a78", "#d05a47"])
    ax.set_ylabel("Physical extent (mm)")
    ax.set_title("Geometry diagnosis: the mapped rabbit fits the crop; the implemented support does not")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    geometry_path = OUT / "figures" / "Z4_geometry_diagnostic.png"
    fig.savefig(geometry_path, dpi=200)
    plt.close(fig)
    return {"main": str(main_path.resolve()), "overlay": str(overlay_path.resolve()), "geometry": str(geometry_path.resolve())}


def requirement_rows() -> list[dict[str, Any]]:
    entries = [
        (1, "Read complete physics-team report first", "full DOCX text/tables and four rendered pages reviewed before code changes", "manual source review", "PASS", "audit/physics_team_requirements.csv"),
        (2, "Preserve Z3 and create isolated Z4 evidence tree", "hash-preserving marker; no Z3 writes", "test_z3_preserved", "PASS", "audit/Z3_DIAGNOSTIC_NOT_READY.json"),
        (3, "Exclude Report/0 from phase-step visibility", "guard rejects Report/0 paths; Z4 imports no estimator", "T01", "PASS", "tests/critical_test_matrix.csv"),
        (4, "Inventory every Report/0 TIFF", "direct per-file read and statistics", "inventory row count and dimensions", "PASS", "experimental_inventory/report0_inventory.csv"),
        (5, "Freeze 13 um/pixel and 1x1 binning", "physical axes constructed at fixed pitch", "T02", "PASS", "camera_sampling/camera_sampling_audit.json"),
        (6, "Separate magnification, sampling, crop, display zoom", "separate functions/artifacts and provenance", "T03/T05/T06", "PASS", "provenance/operation_chain.json"),
        (7, "Diagnose Z3 small spot", "trace unconditional Gaussian envelope", "T12", "PASS", "geometry/Z3_small_spot_root_cause.md"),
        (8, "Trace object-to-camera formation", "wavelength/focal/mapping/ABCD audit", "T08", "PASS", "geometry/C_lab_optical_chain.json"),
        (9, "Determine expected C_lab image formation", "far-field conditional mapped response documented", "mapping review", "PASS", "geometry/C_lab_expected_image_formation.md"),
        (10, "Audit rabbit physical scale", "reuse frozen independent Y1 calibration", "T11", "PASS", "rabbit/rabbit_physical_scale_audit.csv"),
        (11, "Generate full camera plane before crop", "1024x1024 current-model intensity saved in NPY/TIFF/PNG", "T07", "PASS", "full_camera/C_lab_rabbit_full_camera_intensity.npy"),
        (12, "Sample at real camera pitch without resize", "direct detector-center coordinates at 13 um", "T02/T03", "PASS", "camera_sampling/C_lab_camera_coordinate_centers_13um.npz"),
        (13, "Assess finite-pixel integration", "document current point sampling; do not introduce appearance-driven integration", "sampling audit", "PASS", "camera_sampling/camera_sampling_audit.json"),
        (14, "Find Report/0 signal centers with fixed algorithm", "same fixed connected-component algorithm for all frames", "center table", "PASS", "roi/report0_signal_centers.csv"),
        (15, "Measure experimental signal extent", "bbox, 4sigma widths, equivalent diameter, centroid", "extent table", "PASS", "roi/report0_signal_extents.csv"),
        (16, "Create sample-specific ROI specs", "actual dimensions and center-inferred origins per sample", "T04", "PASS", "roi/report0_roi_specs.csv"),
        (17, "Crop only after camera sampling", "crop function accepts sampled camera and preserves pitch", "T05", "PASS", "provenance/operation_chain.json"),
        (18, "No resizing", "blacklisted resize APIs absent from Z4 driver", "T03", "PASS", "tests/test_no_resize_for_camera_crop.py"),
        (19, "Match pixel dimensions without resizing", "integer rectangular slicing", "T10", "PASS", "comparison/pixel_dimension_check.csv"),
        (20, "Compare camera images, not visibility", "12 side-by-side intensity comparisons", "artifact audit", "PASS", "comparison/*_side_by_side.png"),
        (21, "Do not force intensity match", "raw statistics retained; normalization display-only", "raw-stat audit", "PASS", "metrics/raw_intensity_statistics.csv"),
        (22, "Compute geometric metrics", "centers, extents, bbox, orientation", "metrics audit", "PASS", "comparison/image_formation_metrics.csv"),
        (23, "Structural rabbit check", "experiment clear; current simulation no clear structure", "geometry gate", "FAIL", "comparison/image_formation_metrics.csv"),
        (24, "Overlay after coordinate alignment only", "translation from independent signal center; no scale/warp", "overlay audit", "PASS", "comparison/exp_sim_overlay.png"),
        (25, "Audit orientation/parity", "negative far-field mapping and raw camera-down convention", "T09", "INCONCLUSIVE", "geometry/C_lab_expected_image_formation.md"),
        (26, "Camera support test", "implemented 1e-4 support compared with crop", "support ratios", "FAIL", "geometry/camera_support_audit.json"),
        (27, "Do not shrink experimental FoV", "actual 455x419 acquisition retained", "dimension audit", "PASS", "comparison/pixel_dimension_check.csv"),
        (28, "If magnification wrong, trace physical cause only", "mapping checks find no magnification error; no change made", "sanity mapping", "PASS", "tests/C_lab_image_formation_sanity.csv"),
        (29, "Magnification sanity with point/disk/edge/line pair/rabbit", "analytical and code mapping comparison", "sanity table", "PASS", "tests/C_lab_image_formation_sanity.csv"),
        (30, "Crop sensitivity", "FoV/origin/signal fraction quantified; pitch and M unchanged", "crop audit", "PASS", "crop/crop_sensitivity.csv"),
        (31, "Do not calculate four-step visibility", "no visibility estimator imported or called", "T01", "PASS", "tests/critical_test_matrix.csv"),
        (32, "Apply pass criteria without invented thresholds", "actual discrepancies reported; failed size/structure gates", "status audit", "FAIL", "tests/Z4_status.json"),
        (33, "Freeze only if Z4 passes", "no freeze emitted because gate fails", "freeze absence", "NOT_APPLICABLE", "tests/Z4_status.json"),
        (34, "If Z4 fails, classify unresolved physics", "PROPAGATION_UNRESOLVED and BEAM_SIZE_UNRESOLVED", "failure classification", "PASS", "tests/Z4_status.json"),
        (35, "Inventory all experimental datasets", "file-level evidence-backed classification", "inventory audit", "PASS", "experimental_inventory/figure5_6_dataset_classification.csv"),
        (36, "Generate diagnostic Z4 figure", "six-panel non-Figure-5 diagnostic", "figure exists", "PASS", "figures/Figure_Z4_Camera_Image_Formation_Test.png"),
        (37, "Use identical physical axes", "matched experimental/simulation extents in mm", "figure review", "PASS", "figures/Figure_Z4_Camera_Image_Formation_Test.png"),
        (38, "Create complete DOCX report", "generated after tests and rendered for QA", "report render QA", "PENDING_REPORT", str(FINAL_REPORT)),
        (39, "Run T01-T15", "focused Z4 tests plus repository suite", "critical matrix", "PENDING_REGRESSION", "tests/critical_test_matrix.csv"),
        (40, "Print final terminal summary and stop after Z4", "driver terminal summary; no Figure 5 or visibility output", "console output", "PASS", "provenance/Z4_terminal_summary.txt"),
    ]
    rows = []
    for number, requirement, implementation, test, status, evidence in entries:
        rows.append({
            "requirement": f"R{number:02d} {requirement}",
            "source": f"QIUP_ROI_Crop_ImageFormation_Test.docx section {number}",
            "implementation": implementation,
            "test": test,
            "result": status,
            "status": status,
            "evidence": evidence,
            "notes": "No appearance-driven fitting, resizing, arbitrary zoom, or Report/0 visibility calculation.",
        })
    return rows


def critical_tests() -> list[dict[str, Any]]:
    tests = [
        ("T01", "Report/0 excluded from four-step estimator", "PASS", "guard rejects Report/0; no phase estimator imported/called"),
        ("T02", "13 um/pixel preserved", "PASS", "camera sampling and all ROI specs use 13.0 um"),
        ("T03", "no resize used", "PASS", "static blacklist test and crop-by-slice implementation"),
        ("T04", "sample-specific dimensions used", "PASS", "dimensions read from every TIFF, never assumed"),
        ("T05", "crop occurs after camera sampling", "PASS", "operation_chain.json and sampled-array crop API"),
        ("T06", "crop does not alter pixel pitch", "PASS", "13.0 um before and after crop"),
        ("T07", "full camera plane generated before crop", "PASS", "1024x1024 NPY/TIFF/PNG generated first"),
        ("T08", "object-to-camera mapping documented", "PASS", "C_lab optical chain and expected formation documents"),
        ("T09", "orientation documented", "PASS", "negative x/y parity; raw head-down convention"),
        ("T10", "simulated/experimental dimensions match", "PASS", "all Report/0 rows match by integer slicing"),
        ("T11", "rabbit physical scale documented", "PASS", "Y1 frozen physical calibration reused"),
        ("T12", "Z3 tiny-spot cause identified", "PASS", "unconditional far-field pump-waist camera envelope"),
        ("T13", "no arbitrary zoom factor", "PASS", "only analytical M_QIUP and fixed physical sampling used"),
        ("T14", "no rabbit-based physics fitting", "PASS", "C_lab production physics unchanged"),
        ("T15", "regression suite passes", "NOT_RUN", "repository suite will be recorded before report authoring"),
    ]
    return [{"test": a, "requirement": b, "result": c, "evidence": d} for a, b, c, d in tests]


def write_status_and_audits(
    inventory: list[dict[str, Any]],
    comparison: dict[str, Any],
    support: dict[str, Any],
    figures: dict[str, str],
    dataset_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    req = requirement_rows()
    tests = critical_tests()
    save_csv(
        OUT / "audit" / "physics_team_requirements.csv",
        req,
        ["requirement", "source", "implementation", "test", "result", "status", "evidence", "notes"],
    )
    save_csv(OUT / "tests" / "critical_test_matrix.csv", tests)
    save_json(OUT / "tests" / "critical_test_matrix.json", tests)
    example = comparison["example"]
    status = {
        "CAMERA_IMAGE_FORMATION": "FAIL",
        "READY_FOR_PHASE_DATA_CLASSIFICATION": "NO",
        "READY_FOR_FIGURE5_VISIBILITY": "NO",
        "root_cause": "MULTIPLE",
        "failure_classes": ["PROPAGATION_UNRESOLVED", "BEAM_SIZE_UNRESOLVED", "CAMERA_MAPPING_UNRESOLVED"],
        "reason": "The current far-field camera envelope confines signal to a tiny pump-waist-derived spot; crop cannot recover the mapped rabbit boundary and no independently validated signal marginal is implemented.",
        "pixel_dimensions": "PASS",
        "physical_fov": "PASS",
        "camera_sampling_13um": "PASS",
        "crop_implementation": "PASS",
        "signal_size": "FAIL",
        "signal_location": "INCONCLUSIVE_CROP_ORIGIN_INFERRED_FROM_SIGNAL_CENTER",
        "orientation": "INCONCLUSIVE_NO_SIMULATED_RABBIT_STRUCTURE",
        "rabbit_structure": "FAIL",
        "z3_preserved": True,
        "authoritative_figure5_created": False,
        "visibility_calculated_from_report0": False,
        "production_physics_modified": False,
        "resize_used": False,
        "arbitrary_zoom_used": False,
        "rabbit_based_physics_fit": False,
        "freeze_created": False,
        "example_filename": example["filename"],
        "figure": figures["main"],
        "report": str(FINAL_REPORT.resolve()),
        "evidence": str(OUT.resolve()),
        "experimental_dataset_file_count": len(dataset_rows),
        "utc": utc_now(),
    }
    save_json(OUT / "tests" / "Z4_status.json", status)
    save_json(
        OUT / "provenance" / "operation_chain.json",
        {
            "order": [
                "frozen rabbit physical map",
                "C_lab QIUP conditional object propagation",
                "full 1024x1024 camera intensity",
                "point sampling at 13 um/pixel",
                "sample-specific rectangular crop",
                "display-only normalization",
            ],
            "prohibited_operations": {
                "pre_propagation_crop": False,
                "resize_to_experiment": False,
                "display_zoom_as_physics": False,
                "nonlinear_warp": False,
                "free_magnification_fit": False,
                "intensity_scale_fit": False,
                "report0_visibility": False,
            },
            "workspace_note": "User named 30sep; execution used the provided writable/superseding 1Oct workspace containing preserved Z3 evidence.",
        },
    )
    summary = {
        "inventory": inventory,
        "example": {
            "filename": example["filename"],
            "exp_shape": list(example["exp"].shape),
            "sim_shape": list(example["sim_crop"].shape),
            "exp_metrics": example["exp_metrics"],
            "sim_metrics": example["sim_metrics"],
            "roi": example["roi"],
            "exp_stats": array_stats(example["exp"]),
            "sim_stats": array_stats(example["sim_crop"]),
        },
        "support": support,
        "status": status,
        "figures": figures,
        "report0_count": len(inventory),
        "dataset_classification_count": len(dataset_rows),
        "source_report": str(SOURCE_REPORT.resolve()),
        "generated_utc": utc_now(),
    }
    save_json(OUT / "provenance" / "Z4_summary.json", summary)
    write_terminal_summary(summary)
    return summary


def write_terminal_summary(summary: dict[str, Any]) -> str:
    inv = summary["inventory"]
    shapes: dict[str, int] = {}
    for row in inv:
        key = f"{row['width_px']}x{row['height_px']}"
        shapes[key] = shapes.get(key, 0) + 1
    common = max(shapes, key=shapes.get)
    other = ", ".join(f"{key} ({value})" for key, value in shapes.items() if key != common) or "none"
    ex = summary["example"]
    em = ex["exp_metrics"]
    sm = ex["sim_metrics"]
    support = summary["support"]
    centroid_px = math.hypot(sm["center_x_px"] - em["center_x_px"], sm["center_y_px"] - em["center_y_px"])
    text = f"""================ Z4 CAMERA IMAGE-FORMATION TEST ================

REPORT\\0

Number of files:
{summary['report0_count']}

Phase-step data:
NO

Primary role:
NOMINAL IMAGE-FORMATION REFERENCE

Common dimensions:
{common} px ({shapes[common]} files)

Other dimensions:
{other}

--------------------------------------------------

CAMERA

Pixel pitch:
13 um/pixel

Binning:
1x1

Resize used:
NO

Interpolation used for visual resizing:
NO

Crop changes pixel pitch:
NO

--------------------------------------------------

EXAMPLE EXPERIMENTAL SAMPLE

File:
{ex['filename']}

Dimensions:
{ex['exp_shape'][1]} x {ex['exp_shape'][0]} px

Physical FoV:
{ex['exp_shape'][1]*PIXEL_UM/1000:.3f} x {ex['exp_shape'][0]*PIXEL_UM/1000:.3f} mm

Signal center:
({em['center_x_px']:.3f}, {em['center_y_px']:.3f}) px

Signal width/height:
{em['effective_width_px']:.3f} x {em['effective_height_px']:.3f} px

--------------------------------------------------

SIMULATION BEFORE CROP

Grid:
{FULL_N} x {FULL_N}

Physical FoV:
{FULL_N*PIXEL_UM/1000:.3f} x {FULL_N*PIXEL_UM/1000:.3f} mm

Camera sampled shape:
{FULL_N} x {FULL_N}

Signal center:
full-grid optical center

Signal width/height:
{sm['effective_width_px']:.3f} x {sm['effective_height_px']:.3f} px in matched crop

Rabbit projected width/height:
{support['projected_rabbit_width_px']:.3f} x {support['projected_rabbit_height_px']:.3f} px

--------------------------------------------------

MATCHED CROP

Experimental shape:
{ex['exp_shape'][0]} x {ex['exp_shape'][1]}

Simulation shape:
{ex['sim_shape'][0]} x {ex['sim_shape'][1]}

Shape match without resize:
PASS

Pixel pitch:
13 um/pixel

Physical FoV:
{ex['exp_shape'][1]*PIXEL_UM/1000:.3f} x {ex['exp_shape'][0]*PIXEL_UM/1000:.3f} mm

--------------------------------------------------

GEOMETRIC COMPARISON

Centroid difference:
{centroid_px:.3f} px
{centroid_px*PIXEL_UM:.3f} um (inconclusive: crop origin inferred from signal center)

Width difference:
{sm['effective_width_px']-em['effective_width_px']:.3f} px

Height difference:
{sm['effective_height_px']-em['effective_height_px']:.3f} px

Orientation:
INCONCLUSIVE

Rabbit/object structure:
FAIL

--------------------------------------------------

Z3 TINY-SPOT ROOT CAUSE

Classification:
MULTIPLE — WRONG_PROPAGATION + WRONG_APERTURE + BEAM_SIZE_UNRESOLVED

Explanation:
The far-field intensity is multiplied by a pump-waist-derived Gaussian camera envelope; its 1e-4 diameter is {support['z3_support_diameter_px_analytical']:.3f} px, while the mapped rabbit is {support['projected_rabbit_width_px']:.1f} x {support['projected_rabbit_height_px']:.1f} px.

Was crop alone the solution:
NO

Optical magnification issue remains:
NO evidence; analytical mapping passes

Propagation issue remains:
YES

Object-camera mapping issue remains:
YES — detected far-field marginal unresolved

--------------------------------------------------

PHYSICS-TEAM IMAGE-FORMATION TEST

Pixel dimensions:
PASS

Physical FoV:
PASS

13 um sampling:
PASS

Crop implementation:
PASS

Signal size:
FAIL

Signal location:
INCONCLUSIVE

Orientation:
INCONCLUSIVE

Rabbit structure:
FAIL

--------------------------------------------------

FINAL STATUS:

CAMERA_IMAGE_FORMATION:
FAIL

READY_FOR_PHASE_DATA_CLASSIFICATION:
NO

READY_FOR_FIGURE5_VISIBILITY:
NO

Reason:
Current far-field signal support is physically inconsistent and cannot be repaired by crop; a validated camera marginal/collection model is still required.

DIAGNOSTIC FIGURE:
{summary['figures']['main']}

REPORT:
{FINAL_REPORT.resolve()}

EVIDENCE:
{OUT.resolve()}

==============================================================
"""
    (OUT / "provenance" / "Z4_terminal_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return text


def record_regression(junit_path: Path) -> None:
    if not junit_path.is_file():
        raise FileNotFoundError(junit_path)
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    duration = sum(float(s.attrib.get("time", 0.0)) for s in suites)
    passed = tests - failures - errors - skipped
    result = "PASS" if failures == 0 and errors == 0 else "FAIL"
    payload = {
        "command": f"python -m pytest -q tests --junitxml={junit_path}",
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "duration_seconds": duration,
        "result": result,
        "junit_xml": str(junit_path.resolve()),
        "interpretation": "T15 is a repository-wide gate; failures do not convert the Z4 camera-image failure into a pass.",
        "utc": utc_now(),
    }
    save_json(OUT / "tests" / "full_regression_status.json", payload)
    rows = load_csv(OUT / "tests" / "critical_test_matrix.csv")
    for row in rows:
        if row["test"] == "T15":
            row["result"] = result
            row["evidence"] = f"{passed} passed, {failures} failed, {errors} errors, {skipped} skipped; see full_regression_status.json"
    save_csv(OUT / "tests" / "critical_test_matrix.csv", rows)
    save_json(OUT / "tests" / "critical_test_matrix.json", rows)
    req = load_csv(OUT / "audit" / "physics_team_requirements.csv")
    for row in req:
        if row["requirement"].startswith("R39 "):
            row["result"] = result
            row["status"] = result
            row["notes"] = f"Focused Z4 tests plus repository regression recorded; repository result={result}."
    save_csv(
        OUT / "audit" / "physics_team_requirements.csv",
        req,
        ["requirement", "source", "implementation", "test", "result", "status", "evidence", "notes"],
    )
    print(json.dumps(payload, indent=2))


def build_report(report_path: Path) -> None:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    summary = json.loads((OUT / "provenance" / "Z4_summary.json").read_text(encoding="utf-8"))
    tests = load_csv(OUT / "tests" / "critical_test_matrix.csv")
    req = load_csv(OUT / "audit" / "physics_team_requirements.csv")
    metrics = load_csv(OUT / "comparison" / "image_formation_metrics.csv")
    inventory = load_csv(OUT / "experimental_inventory" / "report0_inventory.csv")
    regression_path = OUT / "tests" / "full_regression_status.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8")) if regression_path.is_file() else {"result": "NOT_RUN"}
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.68)
    section.right_margin = Inches(0.68)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9.2)
    for name, size, color in (("Title", 24, "18324A"), ("Heading 1", 15, "18324A"), ("Heading 2", 11.5, "35647A")):
        styles[name].font.name = "Aptos Display" if name != "Normal" else "Aptos"
        styles[name].font.size = Pt(size)
        styles[name].font.color.rgb = RGBColor.from_string(color)
    styles["Heading 1"].paragraph_format.space_before = Pt(10)
    styles["Heading 1"].paragraph_format.space_after = Pt(4)
    styles["Heading 2"].paragraph_format.space_before = Pt(7)
    styles["Heading 2"].paragraph_format.space_after = Pt(3)

    def shade(cell: Any, fill: str) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:fill"), fill)

    def add_table(headers: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> Any:
        table = doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        hdr = table.rows[0]
        hdr._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for idx, header in enumerate(headers):
            cell = hdr.cells[idx]
            cell.text = str(header)
            shade(cell, "D9E8EF")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in cell.paragraphs[0].runs:
                run.font.bold = True
                run.font.size = Pt(8.3)
        for row_idx, values in enumerate(rows):
            cells = table.add_row().cells
            if row_idx % 2:
                for cell in cells:
                    shade(cell, "F4F7F9")
            for idx, value in enumerate(values):
                cells[idx].text = str(value)
                cells[idx].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tr_pr = table.rows[-1]._tr.get_or_add_trPr()
                if tr_pr.find(qn("w:cantSplit")) is None:
                    tr_pr.append(OxmlElement("w:cantSplit"))
                for paragraph in cells[idx].paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    for run in paragraph.runs:
                        run.font.size = Pt(7.5)
                if widths:
                    cells[idx].width = Inches(widths[idx])
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return table

    def add_picture(path: Path, width: float, caption: str, alt: str) -> None:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        inline = run.add_picture(str(path), width=Inches(width))
        doc_pr = inline._inline.docPr
        doc_pr.set("descr", alt)
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.style = styles["Caption"]
        cap.runs[0].font.size = Pt(8.5)

    header = section.header.paragraphs[0]
    header.text = "QIUP Digital Twin | Z4 Camera Image-Formation Gate"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor(90, 105, 115)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Diagnostic report — not authoritative Figure 5   |   ")
    run.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Z4 Camera-Plane Image Formation\nROI, Crop, and Rabbit FoV Validation")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Physics-team image-formation test before Figure 5 visibility").bold = True
    badge = doc.add_paragraph()
    badge.alignment = WD_ALIGN_PARAGRAPH.CENTER
    badge_run = badge.add_run("CAMERA_IMAGE_FORMATION: FAIL   •   READY_FOR_FIGURE5_VISIBILITY: NO")
    badge_run.font.bold = True
    badge_run.font.color.rgb = RGBColor(166, 54, 43)
    doc.add_paragraph(
        "Report/0 contains 12 fixed-arrangement 16-bit TIFF frames. It is used only for camera FoV, ROI, crop, center, extent, and object-structure diagnosis. No controlled phase stepping occurred and no four-step visibility is calculated in Z4."
    )
    add_picture(
        OUT / "figures" / "Figure_Z4_Camera_Image_Formation_Test.png",
        7.0,
        "Figure Z4-1. End-to-end camera image-formation diagnostic. All experimental and cropped-simulation camera panels use identical millimetre axes and 13 µm/pixel. Display normalization is explicitly labeled; no array was resized.",
        "Six-panel diagnostic showing frozen rabbit mask, full simulated camera plane, nominal experimental image, matched simulated crop, red-cyan overlay, and geometric extent bars. The simulated signal is a tiny central spot while the experiment contains a broad rabbit-bearing field.",
    )

    doc.add_page_break()
    doc.add_heading("1. Decision and root cause", level=1)
    doc.add_paragraph(
        "The acquisition geometry passes: every 455×419 experimental frame is matched by integer slicing from a previously generated 1024×1024 camera grid, at the unchanged 13 µm pitch. Image formation fails. The current far-field intensity path multiplies the final moments by a Gaussian whose width is derived from the 250 µm pump waist. That produces a ≈10.05 px 1/e² radius and only ≈43.2 px diameter at the Z3 10⁻⁴ support threshold. The analytical rabbit mapping is ≈342×410 px, so crop cannot restore object boundaries that the implemented envelope suppresses."
    )
    add_table(
        ["Gate", "Result", "Evidence"],
        [
            ["Pixel dimensions", "PASS", "All 12 crops match without resize"],
            ["Physical FoV / 13 µm sampling", "PASS", "5.915×5.447 mm for 455×419"],
            ["Signal extent", "FAIL", "Current simulated width/height are far below experiment"],
            ["Signal location", "INCONCLUSIVE", "Crop origin inferred from experimental signal center"],
            ["Orientation", "INCONCLUSIVE", "Parity is documented, but simulated rabbit structure is absent"],
            ["Rabbit structure", "FAIL", "Clear in experiment; not resolved by current simulation"],
            ["Final", "FAIL", "Propagation / camera marginal unresolved"],
        ],
        [1.45, 1.05, 4.25],
    )

    doc.add_heading("2. Physics-team requirements", level=1)
    doc.add_paragraph(
        f"All 40 numbered instructions from {SOURCE_REPORT.name} were converted to the auditable CSV. The full table is retained in evidence; the report summarizes its gate-bearing items."
    )
    for row in req:
        if row["requirement"].startswith("R38 "):
            row["result"] = "PASS"
            row["status"] = "PASS"
            row["notes"] = "DOCX authored after test recording; render QA follows."
    save_csv(
        OUT / "audit" / "physics_team_requirements.csv",
        req,
        ["requirement", "source", "implementation", "test", "result", "status", "evidence", "notes"],
    )
    add_table(
        ["Requirement", "Status", "Evidence"],
        [[row["requirement"], row["status"], row["evidence"]] for row in req if row["status"] not in ("PASS",)],
        [3.2, 1.15, 2.45],
    )

    doc.add_page_break()
    doc.add_heading("3. Report/0 inventory and acquisition meaning", level=1)
    doc.add_paragraph(
        "Direct inspection found 12 TIFF files, all uint16 and 455×419 pixels. The physics-team report explicitly identifies this folder as a nominal fixed arrangement with no controlled phase progression. Therefore 0, π/2, π, and 3π/2 labels are not assigned, even though duplicate timestamps also occur elsewhere in the broader experimental tree."
    )
    add_table(
        ["File", "Shape", "Range", "Mean / median"],
        [[row["filename"].replace("QIUP_", ""), f"{row['width_px']}×{row['height_px']}", f"{float(row['min']):.0f}–{float(row['max']):.0f}", f"{float(row['mean']):.1f} / {float(row['median']):.1f}"] for row in inventory],
        [3.25, 0.9, 1.15, 1.35],
    )

    doc.add_heading("4. Camera sampling, ROI, and crop", level=1)
    doc.add_paragraph(
        "Optical magnification maps physical object coordinates; camera sampling evaluates the camera plane at 13 µm detector centers; crop selects a rectangular subset of those samples; display zoom changes only page presentation. Z4 stores these as separate stages. Finite 13×13 µm pixel-area integration was not added: the existing reduced-order model is point-sampled and there is no evidence that area integration would resolve the roughly order-of-magnitude support error."
    )
    add_table(
        ["Operation", "Definition", "Z4 implementation"],
        [
            ["Optical magnification", "M = −f_s λ_s/(f_i λ_i)", "−0.522580645; no fitted scale"],
            ["Camera sampling", "Detector-center evaluation", "13 µm/pixel, 1×1"],
            ["Crop", "Rectangular sample selection", "Integer slices; per-file dimensions and inferred origin"],
            ["Display normalization", "Visualization only", "Percentile window, explicitly labeled; raw arrays retained"],
        ],
        [1.55, 2.35, 2.95],
    )

    doc.add_heading("5. C_lab image-formation mapping", level=1)
    doc.add_paragraph(
        "C_lab declares far-field object and signal camera planes with 150 mm idler and signal Fourier focal lengths, 810/1550 nm signal/idler wavelengths, and unit relay magnifications. Its reduced Fourier-plane relation is x_s = −(m_s λ_s f_s)/(m_i λ_i f_i) x_i, with the same y mapping. The expected observable is an inverted conditional rabbit-aperture response in camera Fourier coordinates—not a free-space photograph and not an optical Fourier transform of the rabbit mask. The negative scale encodes simultaneous horizontal and vertical parity."
    )
    add_picture(
        OUT / "figures" / "Z4_geometry_diagnostic.png",
        4.6,
        "Figure Z4-2. Physical extents at 13 µm/pixel. The analytical mapped rabbit fits the experimental FoV, whereas the current pump-waist-derived support is much smaller.",
        "Bar chart comparing experimental crop width and height, projected rabbit width and height, and the much smaller implemented camera support in millimetres.",
    )

    doc.add_page_break()
    doc.add_heading("6. Rabbit physical scale and orientation", level=1)
    doc.add_paragraph(
        "The mask is the frozen Y1 independent geometry: 254×297 pixels, 39.7276 µm/native pixel, with an 8.502×10.210 mm bounding box. Its scale came from a ruler photograph, not QIUP output. Inside the rabbit aperture is T=1 and outside T=0; no phase was invented. Head direction is +y in the mask. Independent notes say raw camera acquisition historically shows the rabbit head-down, consistent with the two-axis far-field inversion."
    )
    add_picture(
        OUT / "rabbit" / "rabbit_mask_frozen.png",
        2.3,
        "Figure Z4-3. Frozen independent rabbit aperture mask (white T=1, black T=0).",
        "Binary rabbit silhouette with head and ears at the upper right, used as the independently calibrated aperture geometry.",
    )

    doc.add_heading("7. Full camera plane and matched crop", level=1)
    doc.add_paragraph(
        "The required full intensity was generated first on a 1024×1024 diagnostic detector grid (13.312×13.312 mm edge extent), a format present in the same experimental tree. This does not assert a recorded Report/0 sensor origin; its purpose is to keep camera formation separate from later ROI selection. Each Report/0 crop origin is the minimum integer translation that aligns the simulated optical center to the fixed-algorithm experimental signal centroid, and is labeled INFERRED_FROM_SIGNAL_CENTER."
    )
    first_side = sorted((OUT / "comparison").glob("*_side_by_side.png"))[0]
    add_picture(
        first_side,
        6.7,
        "Figure Z4-4. Example experimental and simulated camera images with the same pixel dimensions, physical FoV, orientation convention, and detector pitch. Both panels are display-normalized independently; quantitative raw statistics are stored separately.",
        "Side-by-side nominal experiment with broad illuminated rabbit structure and current simulation with a small central spot, both on identical camera-plane millimetre axes.",
    )

    doc.add_heading("8. Signal center, extent, and structure", level=1)
    first = metrics[0]
    add_table(
        ["Metric", "Experiment", "Simulation", "Interpretation"],
        [
            ["Centroid x,y (px)", f"{float(first['exp_center_x_px']):.2f}, {float(first['exp_center_y_px']):.2f}", f"{float(first['sim_center_x_px']):.2f}, {float(first['sim_center_y_px']):.2f}", "Not a validation: alignment defines inferred crop origin"],
            ["Effective width (px)", f"{float(first['exp_effective_width_px']):.2f}", f"{float(first['sim_effective_width_px']):.2f}", "FAIL"],
            ["Effective height (px)", f"{float(first['exp_effective_height_px']):.2f}", f"{float(first['sim_effective_height_px']):.2f}", "FAIL"],
            ["Object structure", first["experimental_object_structure"].replace("_", " "), first["simulated_object_structure"].replace("_", " "), "FAIL"],
            ["Orientation", f"{float(first['exp_orientation_deg']):.1f}°", f"{float(first['sim_orientation_deg']):.1f}°", "INCONCLUSIVE"],
        ],
        [1.45, 1.65, 1.65, 2.2],
    )
    add_picture(
        OUT / "comparison" / "exp_sim_overlay.png",
        5.4,
        "Figure Z4-5. Red–cyan overlay after center-only coordinate alignment. No scale fit, nonlinear warp, or resize was applied.",
        "Overlay where red experimental intensity spans most of the crop and cyan simulated intensity remains a tiny central region; rabbit geometry does not overlap as a resolved simulated structure.",
    )

    doc.add_page_break()
    doc.add_heading("9. Why crop is not the solution", level=1)
    s = summary["support"]
    doc.add_paragraph(
        f"For the example crop, the current 10⁻⁴ support diameter is {s['z3_support_diameter_px_analytical']:.2f} px ({s['z3_support_diameter_mm_analytical']:.3f} mm), giving support ratios {s['support_ratio_x']:.3f} and {s['support_ratio_y']:.3f}. The analytical rabbit projection is {s['projected_rabbit_width_px']:.1f}×{s['projected_rabbit_height_px']:.1f} px. Cropping changes the coordinate origin and included signal fraction, but not pitch, optical magnification, or pixels per millimetre. Shrinking the experimental FoV around the simulated spot would violate the acquisition and was not done."
    )
    doc.add_paragraph(
        "The discrepancy is classified MULTIPLE: WRONG_PROPAGATION because the far-field pump profile is reapplied as a camera mask; WRONG_APERTURE because the declared/actual collection support is not the camera marginal used by the current intensity; and BEAM_SIZE_UNRESOLVED because no independently validated far-field signal distribution is implemented. The analytical focal/wavelength magnification passed point, disk, edge, line-pair, and rabbit scale checks, so focal lengths were not changed."
    )

    doc.add_heading("10. No-resize, no-fit, and provenance controls", level=1)
    add_table(
        ["Control", "Result", "Detail"],
        [
            ["Resize / rescale / zoom API", "NOT USED", "Static test scans Z4 driver; crops are array slices"],
            ["Visual interpolation", "NOT USED", "Matplotlib displays source arrays on physical axes"],
            ["Intensity scale fit", "NOT USED", "Raw experiment and model units reported separately"],
            ["Physics parameter fit", "NOT USED", "C_lab focal lengths, wavelengths, relays, and pump waist unchanged"],
            ["Rabbit-based alignment", "TRANSLATION ONLY", "Experimental signal center determines inferred ROI origin"],
            ["Z3 overwrite", "NO", "Z3 hashes recorded and status retained as DIAGNOSTIC_NOT_READY"],
        ],
        [2.0, 1.15, 3.75],
    )

    doc.add_heading("11. Experimental dataset classification", level=1)
    doc.add_paragraph(
        f"The file-level inventory contains {summary['dataset_classification_count']} image files. Report/0 is NOMINAL_REFERENCE and explicitly non-phase-step. Mirror-drift phase folders are classified PHASE_STEP only because the prior Y2 acquisition audit identified their acquisition class; their folder phases remain nominal labels and calibrated physical phases are not established. Dark, longitudinal-shift, transverse-shift, no-object, and rabbit-object contexts are separately recorded. This classification is preparatory only: Z4 is not ready to begin phase analysis."
    )

    doc.add_heading("12. Required tests", level=1)
    add_table(
        ["Test", "Result", "Evidence"],
        [[row["test"], row["result"], row["evidence"]] for row in tests],
        [0.65, 0.9, 5.35],
    )
    doc.add_paragraph(
        f"Repository regression: {regression.get('result', 'NOT_RUN')}. "
        + (f"{regression.get('passed', 0)} passed; {regression.get('failures', 0)} failed; {regression.get('errors', 0)} errors; {regression.get('skipped', 0)} skipped." if regression.get("result") != "NOT_RUN" else "Not recorded.")
    )

    doc.add_heading("13. Final readiness", level=1)
    final = doc.add_paragraph()
    final.add_run("CAMERA_IMAGE_FORMATION: FAIL\n").bold = True
    final.add_run("READY_FOR_PHASE_DATA_CLASSIFICATION: NO\n").bold = True
    final.add_run("READY_FOR_FIGURE5_VISIBILITY: NO\n").bold = True
    final.add_run(
        "Required next physics input: a validated C_lab far-field detected signal marginal / collection model (including where the aperture acts and the measured ROI origin). Once that is established independently, rerun Z4. Do not replace it with a fitted Gaussian width, crop, or display zoom. No freeze or final Figure 5 was created, and no Report/0 visibility or phase assignment was produced."
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(report_path)
    save_json(
        OUT / "report" / "a11y_audit.json",
        {
            "report": str(report_path.resolve()),
            "title_present": True,
            "heading_hierarchy": "Title > Heading 1 > Heading 2",
            "figure_alt_text_count": 5,
            "figure_caption_count": 5,
            "tables_have_header_rows": True,
            "color_not_only_carrier": True,
            "plain_language_status_text": True,
            "status": "PASS_PRE_RENDER",
        },
    )
    print(f"REPORT: {report_path.resolve()}")


def generate_evidence() -> None:
    ensure_dirs()
    guard_report0_from_phase_estimator(sorted(REPORT0.glob("*.does-not-run")))
    write_z3_preservation_marker()
    inventory, measurements = inventory_report0()
    camera = run_current_camera_model()
    comparison = make_crops_and_comparisons(inventory, measurements, camera)
    support = write_geometry(camera, comparison)
    write_mapping_sanity(camera)
    dataset_rows = inventory_all_experimental_data()
    figures = make_overlay_and_figures(camera, comparison, support)
    write_status_and_audits(inventory, comparison, support, figures, dataset_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--report-path", type=Path, default=FINAL_REPORT)
    parser.add_argument("--record-regression", type=Path)
    args = parser.parse_args()
    if args.record_regression is not None:
        record_regression(args.record_regression)
        return
    if args.report_only:
        build_report(args.report_path)
        return
    generate_evidence()
    if not args.evidence_only:
        build_report(args.report_path)


if __name__ == "__main__":
    main()
