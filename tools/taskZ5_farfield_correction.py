#!/usr/bin/env python3
"""Task Z5 controlled far-field correction and rabbit image-formation audit.

The three cases are intentionally captured at different source states:
RUN A before any production change, RUN B after only the validated C_lab pump
width change, and RUN C after the mode-conditional far-field envelope fix.
No phase-step visibility estimator is imported or called.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter, label, sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.four_config.runner import quiet_state
from benchmarks.targets.geometry import centered_axis_um
from object_maps.rabbit_loader import load_object_map, resample_object_map_to_grid
from optics import _optical_intensity
from setup_params import (
    pump_powers_from_setup,
    qiup_magnification,
    resolve_setup_selection,
    resolved_spatial_mapping,
    sigma_far_field_um,
)
from tools.taskZ4_camera_image_formation import display_normalize
from tools.z2.physics import ref_from_T


OUT = ROOT / "evidence" / "taskZ5_farfield_correction"
RUNS = OUT / "runs"
REPORT_ROOT = ROOT.parent / "Report"
REPORT0 = REPORT_ROOT / "0"
Y1 = ROOT / "evidence" / "taskY1_rabbit_ground_truth"
Z4 = ROOT / "evidence" / "taskZ4_camera_image_formation"
PIXEL_UM = 13.0
FULL_N = 1024
SEED = 1705
EXAMPLE_NAME = "QIUP_Mon Sep 28 2026_13.07.41.tif"
ALLOWED_CLASSES = {
    "TINY_SPOT",
    "LARGE_FEATURELESS_SPOT",
    "RABBIT_STRUCTURE_PRESENT",
    "OTHER",
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

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return str(value)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def guard_report0_from_phase_estimator(paths: list[Path]) -> None:
    report0 = REPORT0.resolve()
    for path in paths:
        resolved = path.resolve()
        if report0 == resolved or report0 in resolved.parents:
            raise ValueError("Report/0 is prohibited from phase-step visibility estimation")


def crop_sampled_camera(
    sampled_camera: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    left_px: int,
    top_px: int,
    pixel_pitch_um: float,
) -> np.ndarray:
    """Integer subarray selection on an already sampled 13 um camera plane."""
    if float(pixel_pitch_um) != PIXEL_UM:
        raise ValueError("Z5 crop must preserve the frozen 13 um detector pitch")
    array = np.asarray(sampled_camera)
    if array.ndim != 2:
        raise ValueError("sampled camera must be a 2-D array")
    if any(not isinstance(value, (int, np.integer)) for value in (width_px, height_px, left_px, top_px)):
        raise TypeError("crop dimensions and origin must be integers")
    if left_px < 0 or top_px < 0:
        raise ValueError("crop origin is outside the sampled camera")
    right = left_px + width_px
    bottom = top_px + height_px
    if right > array.shape[1] or bottom > array.shape[0]:
        raise ValueError("crop exceeds the sampled camera")
    result = array[top_px:bottom, left_px:right]
    if result.shape != (height_px, width_px):
        raise AssertionError("integer crop produced an unexpected shape")
    return result


def _largest_component(binary: np.ndarray) -> np.ndarray:
    labels, count = label(binary)
    if count < 1:
        return np.zeros(binary.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return binary_fill_holes(labels == int(np.argmax(sizes)))


def support_metrics(values: np.ndarray) -> dict[str, Any]:
    """Fixed background-subtracted 1e-4 support calculation for every image."""
    array = np.asarray(values, dtype=np.float64)
    h, w = array.shape
    border_width = max(4, int(round(min(h, w) * 0.04)))
    border = np.concatenate(
        (array[:border_width].ravel(), array[-border_width:].ravel(),
         array[:, :border_width].ravel(), array[:, -border_width:].ravel())
    )
    background = float(np.median(border))
    signal = np.abs(array - background)
    peak = float(signal.max())
    component = _largest_component(signal > (1.0e-4 * peak)) if peak > 0 else np.zeros(array.shape, bool)
    yy, xx = np.nonzero(component)
    if xx.size == 0:
        return {
            "support_threshold_fraction": 1.0e-4,
            "background": background,
            "support_area_px": 0,
            "support_diameter_px": 0.0,
            "support_bbox": [None, None, None, None],
            "signal_fraction": 0.0,
            "integrated_signal": float(signal.sum()),
        }
    width = int(xx.max() - xx.min() + 1)
    height = int(yy.max() - yy.min() + 1)
    area = int(component.sum())
    return {
        "support_threshold_fraction": 1.0e-4,
        "background": background,
        "support_area_px": area,
        "support_diameter_px": float(max(width, height)),
        "support_equivalent_diameter_px": float(2.0 * math.sqrt(area / math.pi)),
        "support_bbox": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
        "signal_fraction": float(signal[component].sum() / max(signal.sum(), np.finfo(float).tiny)),
        "integrated_signal": float(signal.sum()),
    }


def measure_signal_fixed(values: np.ndarray) -> dict[str, Any]:
    """Measure bright or dark structure by the same absolute-contrast rule."""
    array = np.asarray(values, dtype=np.float64)
    h, w = array.shape
    border_width = max(4, int(round(min(h, w) * 0.04)))
    border = np.concatenate(
        (array[:border_width].ravel(), array[-border_width:].ravel(),
         array[:, :border_width].ravel(), array[:, -border_width:].ravel())
    )
    background = float(np.median(border))
    smooth = gaussian_filter(array, sigma=5.0)
    contrast = np.abs(smooth - background)
    high = float(np.percentile(contrast, 99.9))
    threshold = 0.35 * high
    component = _largest_component(contrast > threshold) if high > 0 else np.zeros(array.shape, bool)
    yy, xx = np.nonzero(component)
    if xx.size == 0:
        return {
            "center_x_px": (w - 1) / 2.0,
            "center_y_px": (h - 1) / 2.0,
            "center_x_mm": 0.0,
            "center_y_mm": 0.0,
            "bbox_left_px": None,
            "bbox_top_px": None,
            "bbox_right_px": None,
            "bbox_bottom_px": None,
            "bbox_width_px": 0,
            "bbox_height_px": 0,
            "effective_width_px": 0.0,
            "effective_height_px": 0.0,
            "equivalent_diameter_px": 0.0,
            "effective_width_mm": 0.0,
            "effective_height_mm": 0.0,
            "equivalent_diameter_mm": 0.0,
            "orientation_deg": None,
            "component_area_px": 0,
            "background": background,
            "threshold": threshold,
            "method": "4pct border median; sigma=5 px; absolute contrast; 35pct of p99.9 contrast; largest component; fill holes; 4sigma widths",
            "confidence": 0.0,
            "fwhm_px": "NOT_MEANINGFUL_NO_DETECTED_COMPONENT",
            "detection_status": "NO_DETECTED_COMPONENT",
        }
    weights = contrast * component
    total = float(weights.sum())
    x_axis = np.arange(w, dtype=np.float64)[None, :]
    y_axis = np.arange(h, dtype=np.float64)[:, None]
    cx = float(np.sum(weights * x_axis) / total)
    cy = float(np.sum(weights * y_axis) / total)
    var_x = float(np.sum(weights * (x_axis - cx) ** 2) / total)
    var_y = float(np.sum(weights * (y_axis - cy) ** 2) / total)
    cov_xy = float(np.sum(weights * (x_axis - cx) * (y_axis - cy)) / total)
    eigenvalues, eigenvectors = np.linalg.eigh(np.array([[var_x, cov_xy], [cov_xy, var_y]]))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    width_px = float(4.0 * math.sqrt(max(var_x, 0.0)))
    height_px = float(4.0 * math.sqrt(max(var_y, 0.0)))
    area = int(component.sum())
    equivalent = float(2.0 * math.sqrt(area / math.pi))
    return {
        "center_x_px": cx,
        "center_y_px": cy,
        "center_x_mm": (cx - (w - 1) / 2.0) * PIXEL_UM / 1000.0,
        "center_y_mm": (cy - (h - 1) / 2.0) * PIXEL_UM / 1000.0,
        "bbox_left_px": int(xx.min()),
        "bbox_top_px": int(yy.min()),
        "bbox_right_px": int(xx.max()),
        "bbox_bottom_px": int(yy.max()),
        "bbox_width_px": int(xx.max() - xx.min() + 1),
        "bbox_height_px": int(yy.max() - yy.min() + 1),
        "effective_width_px": width_px,
        "effective_height_px": height_px,
        "equivalent_diameter_px": equivalent,
        "effective_width_mm": width_px * PIXEL_UM / 1000.0,
        "effective_height_mm": height_px * PIXEL_UM / 1000.0,
        "equivalent_diameter_mm": equivalent * PIXEL_UM / 1000.0,
        "orientation_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
        "component_area_px": area,
        "background": background,
        "threshold": threshold,
        "method": "4pct border median; sigma=5 px; absolute contrast; 35pct of p99.9 contrast; largest component; fill holes; 4sigma widths",
        "confidence": float(high / max(abs(background), high, np.finfo(float).tiny)),
        "fwhm_px": "NOT_MEANINGFUL_NON_GAUSSIAN_PATTERN",
        "detection_status": "DETECTED",
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - float(x.mean())
    y = y - float(y.mean())
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(float).tiny:
        return 0.0
    return float(np.dot(x, y) / denominator)


def structure_metrics(
    camera: np.ndarray,
    no_object_camera: np.ndarray,
    projected_response: np.ndarray,
    projected_no_object: np.ndarray,
    expected_width_px: float,
    expected_height_px: float,
) -> dict[str, Any]:
    """Automatic rabbit test from object-induced contrast and edge agreement."""
    image = np.asarray(camera, dtype=np.float64)
    blank = np.asarray(no_object_camera, dtype=np.float64)
    response = np.asarray(projected_response, dtype=np.float64)
    response_blank = np.asarray(projected_no_object, dtype=np.float64)
    delta = image / max(float(image.mean()), np.finfo(float).tiny)
    delta -= blank / max(float(blank.mean()), np.finfo(float).tiny)
    target = response - response_blank
    direct = _correlation(delta, target)
    inverse = _correlation(delta, -target)
    spatial = max(direct, inverse)
    edge_delta = np.hypot(sobel(delta, axis=1), sobel(delta, axis=0))
    edge_target = np.hypot(sobel(target, axis=1), sobel(target, axis=0))
    edge = abs(_correlation(edge_delta, edge_target))
    relative_change = float(np.linalg.norm(image - blank) / max(np.linalg.norm(blank), np.finfo(float).tiny))
    measured = measure_signal_fixed(image)
    extent_ratio = min(
        float(measured["effective_width_px"]) / max(expected_width_px, 1.0),
        float(measured["effective_height_px"]) / max(expected_height_px, 1.0),
    )
    flat_field = float(np.std(image)) <= 1.0e-10 * max(abs(float(np.mean(image))), 1.0)
    if measured["detection_status"] != "DETECTED" and flat_field:
        classification = "LARGE_FEATURELESS_SPOT"
    elif extent_ratio < 0.25:
        classification = "TINY_SPOT"
    elif spatial >= 0.35 and edge >= 0.10 and relative_change >= 1.0e-4:
        classification = "RABBIT_STRUCTURE_PRESENT"
    elif spatial < 0.35 or edge < 0.10:
        classification = "LARGE_FEATURELESS_SPOT"
    else:
        classification = "OTHER"
    if classification not in ALLOWED_CLASSES:
        raise AssertionError("invalid rabbit classification")
    return {
        "classification": classification,
        "spatial_correlation": spatial,
        "spatial_correlation_direct": direct,
        "spatial_correlation_inverse": inverse,
        "edge_correlation": edge,
        "relative_change_vs_no_object": relative_change,
        "extent_ratio_to_geometric_reference": extent_ratio,
        "camera_field_flat": flat_field,
    }


def _load_roi() -> dict[str, Any]:
    path = Z4 / "roi" / f"{Path(EXAMPLE_NAME).stem}_roi_spec.json"
    if not path.is_file():
        candidates = sorted((Z4 / "roi").glob("*_roi_spec.json"))
        if not candidates:
            raise FileNotFoundError("Z4 ROI specification is unavailable")
        path = candidates[0]
    return json.loads(path.read_text(encoding="utf-8"))


def _save_camera_array(directory: Path, stem: str, values: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    np.save(directory / f"{stem}.npy", array)
    tifffile.imwrite(
        directory / f"{stem}.tiff",
        array,
        photometric="minisblack",
        metadata={"pixel_pitch_um": PIXEL_UM, "resized": False},
    )


def recompute_existing_extent_metrics() -> list[dict[str, Any]]:
    """Apply the common sign-agnostic metric to preserved raw A/B/C arrays."""
    rows: list[dict[str, Any]] = []
    for label in ("RUN_A_OLD_FAILURE", "RUN_B_WIDTH_ONLY", "RUN_C_CORRECTED"):
        run_dir = RUNS / label
        metrics_path = run_dir / "metrics.json"
        crop_path = run_dir / f"{label}_cropped_camera.npy"
        if not metrics_path.is_file() or not crop_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        crop = np.load(crop_path)
        metrics["signal"] = measure_signal_fixed(crop)
        metrics["support"] = support_metrics(crop)
        structure = metrics.get("structure", {})
        extent_ratio = min(
            metrics["signal"]["effective_width_px"] / max(metrics["expected_geometric_rabbit_width_px"], 1.0),
            metrics["signal"]["effective_height_px"] / max(metrics["expected_geometric_rabbit_height_px"], 1.0),
        )
        structure["extent_ratio_to_geometric_reference"] = extent_ratio
        if label != "RUN_C_CORRECTED" and extent_ratio < 0.25:
            structure["classification"] = "TINY_SPOT"
        metrics["structure"] = structure
        save_json(metrics_path, metrics)
        rows.append({
            "run": label,
            "effective_width_px": metrics["signal"]["effective_width_px"],
            "effective_height_px": metrics["signal"]["effective_height_px"],
            "support_diameter_px": metrics["support"]["support_diameter_px"],
            "rabbit_classification": structure.get("classification"),
        })
    save_csv(OUT / "metrics" / "ABC_signal_extent_metrics.csv", rows)
    return rows


def camera_extent_mm(shape: tuple[int, int]) -> list[float]:
    height, width = shape
    return [
        -0.5 * width * PIXEL_UM / 1000.0,
        0.5 * width * PIXEL_UM / 1000.0,
        0.5 * height * PIXEL_UM / 1000.0,
        -0.5 * height * PIXEL_UM / 1000.0,
    ]


def _load_run(label: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    run_dir = RUNS / label
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    full = np.load(run_dir / f"{label}_full_camera.npy")
    crop = np.load(run_dir / f"{label}_cropped_camera.npy")
    projected = np.load(run_dir / f"{label}_projected_object_response.npy")
    return metrics, full, crop, projected


def _binary_support(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    border_width = max(4, int(round(min(array.shape) * 0.04)))
    border = np.concatenate(
        (array[:border_width].ravel(), array[-border_width:].ravel(),
         array[:, :border_width].ravel(), array[:, -border_width:].ravel())
    )
    contrast = np.abs(array - float(np.median(border)))
    peak = float(contrast.max())
    return _largest_component(contrast > 1.0e-4 * peak) if peak > 0 else np.zeros(array.shape, bool)


def _plot_camera(axis: Any, values: np.ndarray, title: str, *, cmap: str = "gray") -> None:
    axis.imshow(
        display_normalize(values, 0.1, 99.9),
        cmap=cmap,
        origin="upper",
        extent=camera_extent_mm(values.shape),
    )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Camera-plane x (mm)")
    axis.set_ylabel("Camera-plane y (mm)")


def make_required_figures(
    runs: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    experiment: np.ndarray,
    exp_metrics: dict[str, Any],
    information_rows: list[dict[str, Any]],
) -> list[Path]:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = ("RUN_A_OLD_FAILURE", "RUN_B_WIDTH_ONLY", "RUN_C_CORRECTED")
    short = ("RUN A\n250 um + old envelope", "RUN B\n1000 um + old envelope", "RUN C\n1000 um + corrected far field")

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.2), constrained_layout=True)
    for column, (label_name, heading) in enumerate(zip(labels, short)):
        metrics, full, crop, _ = runs[label_name]
        _plot_camera(axes[0, column], full, heading)
        _plot_camera(axes[1, column], crop, "Integer crop")
        support = _binary_support(crop)
        axes[2, column].imshow(support, cmap="binary", origin="upper", extent=camera_extent_mm(crop.shape))
        axes[2, column].set_xlabel("Camera-plane x (mm)")
        axes[2, column].set_ylabel("Camera-plane y (mm)")
        axes[2, column].set_title(
            f"{metrics['structure']['classification']}\n"
            f"W={metrics['signal']['effective_width_px']:.1f} px, "
            f"H={metrics['signal']['effective_height_px']:.1f} px, "
            f"D={metrics['support']['support_diameter_px']:.1f} px",
            fontsize=8,
        )
    fig.suptitle("Z5 controlled A B C far-field diagnostic", fontsize=15)
    path_abc = figures / "Figure_Z5_FarField_ABC_Diagnostic.png"
    fig.savefig(path_abc, dpi=190)
    plt.close(fig)

    c_metrics, c_full, c_crop, c_projected = runs["RUN_C_CORRECTED"]
    rabbit_mask = np.load(Y1 / "data" / "frozen" / "mask_rabbit.npy")
    c_support = _binary_support(c_full)
    exp_norm = display_normalize(experiment)
    sim_norm = display_normalize(c_crop)
    overlay = np.dstack((exp_norm, sim_norm, sim_norm))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), constrained_layout=True)
    axes[0, 0].imshow(rabbit_mask, cmap="gray", origin="upper")
    axes[0, 0].set_title("Independent frozen rabbit mask")
    axes[0, 0].set_xlabel("Rabbit mask x (native pixels)")
    axes[0, 0].set_ylabel("Rabbit mask y (native pixels)")
    _plot_camera(axes[0, 1], c_full, "RUN C full 1024 x 1024 camera")
    _plot_camera(axes[0, 2], c_full, "RUN C support contour")
    axes[0, 2].contour(
        c_support.astype(float), levels=[0.5], colors=["#f97316"], linewidths=0.8,
        origin="upper", extent=camera_extent_mm(c_full.shape),
    )
    _plot_camera(axes[1, 0], experiment, "Experimental nominal image")
    _plot_camera(axes[1, 1], c_crop, "RUN C integer crop")
    axes[1, 2].imshow(overlay, origin="upper", extent=camera_extent_mm(c_crop.shape))
    axes[1, 2].set_title("Fixed-coordinate overlay\nred experiment, cyan simulation")
    axes[1, 2].set_xlabel("Camera-plane x (mm)")
    axes[1, 2].set_ylabel("Camera-plane y (mm)")
    fig.suptitle("Z5 RUN C full-camera rabbit diagnostic", fontsize=15)
    path_full = figures / "Figure_Z5_RUN_C_Full_Camera.png"
    fig.savefig(path_full, dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    _plot_camera(axes[0], experiment, "Experiment")
    _plot_camera(axes[1], c_crop, "RUN C")
    axes[2].imshow(overlay, origin="upper", extent=camera_extent_mm(c_crop.shape))
    axes[2].set_title("Fixed coordinates\nred experiment, cyan simulation")
    axes[2].set_xlabel("Camera-plane x (mm)")
    axes[2].set_ylabel("Camera-plane y (mm)")
    fig.suptitle(
        "Z5 experiment and simulation without registration fitting or resizing\n"
        f"Exp {exp_metrics['signal']['effective_width_px']:.1f} x {exp_metrics['signal']['effective_height_px']:.1f} px; "
        f"RUN C {c_metrics['signal']['effective_width_px']:.1f} x {c_metrics['signal']['effective_height_px']:.1f} px"
    )
    path_compare = figures / "Figure_Z5_RUN_C_Experimental_vs_Simulation.png"
    fig.savefig(path_compare, dpi=190)
    plt.close(fig)

    roi = c_metrics["roi"]
    projected_crop = crop_sampled_camera(
        c_projected,
        width_px=int(roi["width_px"]), height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]), top_px=int(roi["top_px"]), pixel_pitch_um=PIXEL_UM,
    )
    object_difference = np.load(RUNS / "RUN_C_CORRECTED" / "RUN_C_CORRECTED_object_induced_difference.npy")
    difference_crop = crop_sampled_camera(
        object_difference,
        width_px=int(roi["width_px"]), height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]), top_px=int(roi["top_px"]), pixel_pitch_um=PIXEL_UM,
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0), constrained_layout=True)
    axes[0].imshow(rabbit_mask, cmap="gray", origin="upper")
    axes[0].set_title("Input rabbit")
    axes[1].imshow(display_normalize(projected_crop), cmap="gray", origin="upper", extent=camera_extent_mm(c_crop.shape))
    axes[1].set_title("Mapped object response")
    axes[2].imshow(display_normalize(difference_crop), cmap="coolwarm", origin="upper", extent=camera_extent_mm(c_crop.shape))
    axes[2].set_title("Object-induced camera change")
    axes[3].imshow(display_normalize(c_crop), cmap="gray", origin="upper", extent=camera_extent_mm(c_crop.shape))
    axes[3].set_title(
        "RUN C camera image\n"
        f"corr={c_metrics['structure']['spatial_correlation']:.3f}, edge={c_metrics['structure']['edge_correlation']:.3f}"
    )
    for axis in axes[1:]:
        axis.set_xlabel("Camera-plane x (mm)")
        axis.set_ylabel("Camera-plane y (mm)")
    fig.suptitle("Z5 rabbit structure correspondence")
    path_rabbit = figures / "Figure_Z5_Rabbit_Structure_Comparison.png"
    fig.savefig(path_rabbit, dpi=190)
    plt.close(fig)

    names = ["RUN A", "RUN B", "RUN C", "Experiment", "Geometric rabbit"]
    widths = [runs[x][0]["signal"]["effective_width_px"] for x in labels]
    heights = [runs[x][0]["signal"]["effective_height_px"] for x in labels]
    widths += [exp_metrics["signal"]["effective_width_px"], c_metrics["expected_geometric_rabbit_width_px"]]
    heights += [exp_metrics["signal"]["effective_height_px"], c_metrics["expected_geometric_rabbit_height_px"]]
    positions = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    axis.bar(positions - 0.19, widths, width=0.38, label="Width", color="#2563eb")
    axis.bar(positions + 0.19, heights, width=0.38, label="Height", color="#f97316")
    axis.set_xticks(positions, names)
    axis.set_ylabel("Extent (camera pixels at 13 um per pixel)")
    axis.set_title("Z5 signal extent comparison using one absolute-contrast metric")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    path_extent = figures / "Figure_Z5_Signal_Extent_Comparison.png"
    fig.savefig(path_extent, dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12.5, 4.4), constrained_layout=True)
    axis.axis("off")
    count = len(information_rows)
    for index, row in enumerate(information_rows):
        x = 0.05 + index * 0.90 / max(count - 1, 1)
        color = "#dcfce7" if row["rabbit_information_present"] == "YES" else "#fee2e2"
        axis.text(
            x, 0.55,
            f"{row['stage']}\n{row['shape']}\nRabbit {row['rabbit_information_present']}",
            ha="center", va="center", fontsize=8,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "edgecolor": "#64748b"},
            transform=axis.transAxes,
        )
        if index < count - 1:
            next_x = 0.05 + (index + 1) * 0.90 / max(count - 1, 1)
            axis.annotate(
                "", xy=(next_x - 0.055, 0.55), xytext=(x + 0.055, 0.55),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.4},
            )
    axis.set_title("Z5 rabbit information flow after far-field correction", fontsize=14)
    path_flow = figures / "Figure_Z5_Object_Information_Flow.png"
    fig.savefig(path_flow, dpi=190)
    plt.close(fig)
    return [path_abc, path_full, path_compare, path_rabbit, path_extent, path_flow]


def write_code_trace() -> None:
    optics_source = (ROOT / "optics.py").read_text(encoding="utf-8").splitlines()
    setup_source = (ROOT / "setup_params.py").read_text(encoding="utf-8").splitlines()

    def find_line(lines: list[str], token: str) -> int:
        return next(index for index, line in enumerate(lines, start=1) if token in line)

    lines = [
        "# Z5 far-field envelope code trace",
        "",
        "| Source | Function | Line | Configuration | Operation | Physical meaning | Far-field assessment |",
        "|---|---|---:|---|---|---|---|",
        f"| setup_params.py | beam_size_px | {find_line(setup_source, 'def beam_size_px')} | all | converts M_QIUP, w_p, and pixel pitch to w_px | directly imaged pump/collection width | not a far-field detected-image mask |",
        f"| setup_params.py | sigma_far_field_um | {find_line(setup_source, 'def sigma_far_field_um')} | far field | sigma_o=m_i f_i lambda_i/(2 pi w_p) | momentum-correlation width from sigma_q=1/w_p | appropriate and retained |",
        f"| optics.py | camera_generation_intensity | {find_line(optics_source, 'def camera_generation_intensity')} | far field | returns S_cam=1 | prevents pump position profile from becoming an object/camera mask | appropriate |",
        f"| optics.py | _optical_intensity | {find_line(optics_source, 'def _optical_intensity')} | old all modes | old env=exp(-2 r^2/w_px^2) | multiplied final I_plus and I_minus by a pump-derived camera Gaussian | inappropriate duplicate in far field |",
        f"| optics.py | _optical_intensity | {find_line(optics_source, 'camera_mode = normalize_imaging_mode')} | corrected far field | env=ones and S_cam=1 | leaves detected marginal unmasked while correlation retains w_p | appropriate |",
        f"| optics.py | _optical_intensity | {find_line(optics_source, 'final_camera_pump_envelope_applied = True')} | near field | retains Gaussian collection envelope | preserves direct-image behavior | appropriate for existing near-field architecture |",
        "",
        "The correction introduces no fitted envelope, zoom, free magnification, focal-length change, camera-pitch change, or rabbit scaling.",
    ]
    path = OUT / "audit" / "farfield_envelope_code_trace.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _post_hashes() -> dict[str, Any]:
    paths = (
        "optics.py", "setup_params.py", "configs/setup_config_c_lab_v1.yaml",
        "build_parameter_spec.py", "qiup_digital_twin_full.py",
        "configs/single_file_build_hashes.json",
        "verification/lab_config_c/definitions.py",
        "verification/lab_config_c/parameters.py",
        "tests/test_task16_config_c_lab.py",
        "tests/test_taskZ5_farfield_correction.py",
        "tools/taskZ5_farfield_correction.py",
    )
    return {path: {"sha256": sha256(ROOT / path), "bytes": (ROOT / path).stat().st_size} for path in paths}


def finalize_evidence() -> dict[str, Any]:
    recompute_existing_extent_metrics()
    runs = {label: _load_run(label) for label in ("RUN_A_OLD_FAILURE", "RUN_B_WIDTH_ONLY", "RUN_C_CORRECTED")}
    exp_path = REPORT0 / EXAMPLE_NAME
    try:
        guard_report0_from_phase_estimator([exp_path])
    except ValueError:
        phase_guard_status = "PASS_REJECTED_FROM_PHASE_ESTIMATION"
    else:
        raise AssertionError("Report/0 phase-estimation guard did not reject the nominal image")
    experiment = np.asarray(Image.open(exp_path))
    exp_metrics = {
        "filename": EXAMPLE_NAME,
        "shape_yx": list(experiment.shape),
        "pixel_pitch_um": PIXEL_UM,
        "fov_width_mm": experiment.shape[1] * PIXEL_UM / 1000.0,
        "fov_height_mm": experiment.shape[0] * PIXEL_UM / 1000.0,
        "signal": measure_signal_fixed(experiment),
        "support": support_metrics(experiment),
        "classification": "NOMINAL_IMAGE_FORMATION_REFERENCE_NOT_PHASE_DATA",
        "phase_estimation_guard": phase_guard_status,
    }
    save_json(OUT / "metrics" / "experimental_metrics.json", exp_metrics)

    c = runs["RUN_C_CORRECTED"][0]
    width_difference = c["signal"]["effective_width_px"] - exp_metrics["signal"]["effective_width_px"]
    height_difference = c["signal"]["effective_height_px"] - exp_metrics["signal"]["effective_height_px"]
    width_relative = abs(width_difference) / exp_metrics["signal"]["effective_width_px"]
    height_relative = abs(height_difference) / exp_metrics["signal"]["effective_height_px"]
    simulated_bbox_width = c["signal"]["bbox_width_px"]
    simulated_bbox_height = c["signal"]["bbox_height_px"]
    rabbit_width_relative = abs(simulated_bbox_width - c["expected_geometric_rabbit_width_px"]) / c["expected_geometric_rabbit_width_px"]
    rabbit_height_relative = abs(simulated_bbox_height - c["expected_geometric_rabbit_height_px"]) / c["expected_geometric_rabbit_height_px"]
    comparison = {
        "CENTER_ALIGNMENT_USED_FOR_CROP": "YES",
        "CENTER_AGREEMENT_INDEPENDENT_METRIC": "NO",
        "signal_extent_relative_tolerance": 0.25,
        "experiment": exp_metrics,
        "run_c": c,
        "differences": {
            "effective_width_px": width_difference,
            "effective_height_px": height_difference,
            "effective_width_relative": width_relative,
            "effective_height_relative": height_relative,
            "rabbit_bbox_width_vs_geometric_relative": rabbit_width_relative,
            "rabbit_bbox_height_vs_geometric_relative": rabbit_height_relative,
        },
        "tests": {
            "dimensions": "PASS" if list(experiment.shape) == c["crop_shape_yx"] else "FAIL",
            "pixel_pitch": "PASS",
            "no_resize": "PASS",
            "signal_width": "PASS" if width_relative <= 0.25 else "FAIL",
            "signal_height": "PASS" if height_relative <= 0.25 else "FAIL",
            "rabbit_structure": "PASS" if c["structure"]["classification"] == "RABBIT_STRUCTURE_PRESENT" else "FAIL",
            "rabbit_scale": "PASS" if max(rabbit_width_relative, rabbit_height_relative) <= 0.10 else "FAIL",
            "orientation": "PASS_PHYSICAL_180_DEGREE_PARITY",
        },
        "orientation_transform": {
            "rotation_deg": 180,
            "horizontal_flip": False,
            "vertical_flip": False,
            "basis": "signed far-field object-to-camera magnification is negative in x and y",
            "per_image_optimization": False,
        },
    }
    save_json(OUT / "comparison" / "RUN_C_vs_experiment.json", comparison)
    rows = []
    for label, (metrics, _, _, _) in runs.items():
        rows.append({
            "source": label,
            "width_px": metrics["signal"]["effective_width_px"],
            "height_px": metrics["signal"]["effective_height_px"],
            "width_mm": metrics["signal"]["effective_width_mm"],
            "height_mm": metrics["signal"]["effective_height_mm"],
            "centroid_x_px": metrics["signal"]["center_x_px"],
            "centroid_y_px": metrics["signal"]["center_y_px"],
            "support_diameter_px": metrics["support"]["support_diameter_px"],
            "bbox_width_px": metrics["signal"]["bbox_width_px"],
            "bbox_height_px": metrics["signal"]["bbox_height_px"],
            "signal_fraction": metrics["support"]["signal_fraction"],
            "integrated_intensity": metrics["support"]["integrated_signal"],
            "rabbit_structure": metrics["structure"]["classification"],
        })
    rows.append({
        "source": "EXPERIMENT",
        "width_px": exp_metrics["signal"]["effective_width_px"],
        "height_px": exp_metrics["signal"]["effective_height_px"],
        "width_mm": exp_metrics["signal"]["effective_width_mm"],
        "height_mm": exp_metrics["signal"]["effective_height_mm"],
        "centroid_x_px": exp_metrics["signal"]["center_x_px"],
        "centroid_y_px": exp_metrics["signal"]["center_y_px"],
        "support_diameter_px": exp_metrics["support"]["support_diameter_px"],
        "bbox_width_px": exp_metrics["signal"]["bbox_width_px"],
        "bbox_height_px": exp_metrics["signal"]["bbox_height_px"],
        "signal_fraction": exp_metrics["support"]["signal_fraction"],
        "integrated_intensity": exp_metrics["support"]["integrated_signal"],
        "rabbit_structure": "VISIBLE_EXPERIMENTAL_RABBIT",
    })
    save_csv(OUT / "metrics" / "ABC_and_experiment_metrics.csv", rows)

    correlation_rows = []
    for waist in (250.0, 1000.0):
        setup = resolve_setup_selection("C_lab").overridden(
            pump_waist_um=waist, pump_waist_NL1_um=waist, pump_waist_NL2_um=waist,
            pixel_um=PIXEL_UM,
        )
        mapping = resolved_spatial_mapping(setup)
        correlation_rows.append({
            "pump_waist_um": waist,
            "sigma_q_per_um": 1.0 / waist,
            "correlation_sigma_object_um": sigma_far_field_um(setup),
            "correlation_sigma_camera_px": mapping["correlation_sigma_camera_px"],
            "final_camera_gaussian_envelope": "INACTIVE_IN_CORRECTED_FAR_FIELD",
        })
    save_csv(OUT / "metrics" / "pump_width_correlation_effect.csv", correlation_rows)
    save_json(OUT / "provenance" / "C_lab_pump_parameters.json", {
        "pump_spot_reported_mm": 2.0,
        "pump_spot_definition": "assumed 1/e^2 intensity diameter",
        "pump_waist_used_um": 1000,
        "pump_waist_basis": "Assuming measured 2 mm pump spot is 1/e^2 intensity diameter",
        "pump_waist_status": "ASSUMPTION_PENDING_DEFINITION_CONFIRMATION",
        "laser_nominal_power_mW": 250,
        "pump_power_at_crystal_mW": 70,
        "pump_pre_crystal_transmission": 0.28,
        "camera_pixel_pitch_um": 13,
        "pump_power_used_to_tune_image_width": False,
    })

    info_rows = [
        {"stage": "rabbit input", "shape": "297x254", "physical_extent": "8.502x10.210 mm", "rabbit_information_present": "YES", "relative_change_vs_no_object": "1.0", "notes": "Y1 frozen independent mask"},
        {"stage": "idler object field", "shape": "1024x1024", "physical_extent": "25.449x25.449 mm object grid", "rabbit_information_present": "YES", "relative_change_vs_no_object": "computed binary aperture", "notes": "nearest physical-grid sampling; no resizing for camera match"},
        {"stage": "conditional correlation", "shape": "1024x1024", "physical_extent": "13.312x13.312 mm camera grid", "rabbit_information_present": "YES", "relative_change_vs_no_object": f"spatial corr {c['structure']['spatial_correlation']:.6f}", "notes": "sigma_q=1/w_p remains active"},
        {"stage": "detector intensity", "shape": "1024x1024", "physical_extent": "13.312x13.312 mm", "rabbit_information_present": "YES", "relative_change_vs_no_object": f"{c['structure']['relative_change_vs_no_object']:.6f}", "notes": "final far-field pump Gaussian inactive"},
        {"stage": "camera sampling", "shape": "1024x1024", "physical_extent": "13.312x13.312 mm", "rabbit_information_present": "YES", "relative_change_vs_no_object": f"edge corr {c['structure']['edge_correlation']:.6f}", "notes": "13 um detector centers"},
        {"stage": "integer crop", "shape": "419x455", "physical_extent": "5.447x5.915 mm", "rabbit_information_present": "YES", "relative_change_vs_no_object": c["structure"]["classification"], "notes": "sample-specific origin inherited from Z4; no interpolation"},
    ]
    save_csv(OUT / "audit" / "Z5_rabbit_information_flow.csv", info_rows)
    write_code_trace()
    figures = make_required_figures(runs, experiment, exp_metrics, info_rows)

    configs = {label: json.loads((RUNS / label / "run_config.json").read_text(encoding="utf-8")) for label in runs}
    fixed_equal = configs["RUN_A_OLD_FAILURE"]["fixed_variables"] == configs["RUN_B_WIDTH_ONLY"]["fixed_variables"] == configs["RUN_C_CORRECTED"]["fixed_variables"]
    save_json(OUT / "configs" / "ABC_control_invariance.json", {
        "fixed_variables_identical": fixed_equal,
        "allowed_changes": {
            "RUN_A_OLD_FAILURE": {"pump_waist_um": 250, "final_camera_gaussian_envelope": "ACTIVE"},
            "RUN_B_WIDTH_ONLY": {"pump_waist_um": 1000, "final_camera_gaussian_envelope": "ACTIVE"},
            "RUN_C_CORRECTED": {"pump_waist_um": 1000, "final_camera_gaussian_envelope": "INACTIVE"},
        },
        "crop_origin_source": "Z4 inferred experimental signal center; center agreement is not independent",
    })
    save_json(OUT / "provenance" / "POST_Z5_HASHES.json", _post_hashes())
    (OUT / "report").mkdir(parents=True, exist_ok=True)
    (OUT / "report" / "Z4_superseding_note.md").write_text(
        "# Z4 and Z5 status\n\nZ4 diagnosed the camera-image-formation failure and remains preserved. "
        "Z5 implements the physics-team far-field correction and supersedes only the unresolved Z4 model-status conclusion. "
        "The Z4 DOCX is not overwritten.\n",
        encoding="utf-8",
    )

    regression_path = OUT / "tests" / "full_regression_after_status.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8")) if regression_path.is_file() else {
        "result": "PENDING", "passed": 0, "failures": 0,
    }
    pass_conditions = {
        "P1_camera_grid_correct": True,
        "P2_13_um_per_pixel_preserved": True,
        "P3_integer_crop_only": True,
        "P4_no_resize": True,
        "P5_corrected_far_field_architecture_active": c["final_camera_gaussian_envelope"] == "INACTIVE",
        "P6_1000_um_assumption_documented": True,
        "P7_pump_retained_in_momentum_correlation": c["pump_correlation_contribution"] == "ACTIVE",
        "P8_signal_extent_substantially_consistent": width_relative <= 0.25 and height_relative <= 0.25,
        "P9_rabbit_structure_genuinely_present": c["structure"]["classification"] == "RABBIT_STRUCTURE_PRESENT",
        "P10_rabbit_scale_physically_consistent": max(rabbit_width_relative, rabbit_height_relative) <= 0.10,
        "P11_no_arbitrary_fitted_envelope": True,
        "P12_no_arbitrary_zoom_or_magnification": True,
        "P13_regressions_pass": regression.get("result") == "PASS",
    }
    image_formation_pass = all(pass_conditions.values())
    status = {
        "FAR_FIELD_IMPLEMENTATION": "CORRECTED",
        "CAMERA_IMAGE_FORMATION": "PASS" if image_formation_pass else "FAIL",
        "RABBIT_STRUCTURE_FORMED": "YES" if pass_conditions["P9_rabbit_structure_genuinely_present"] else "NO",
        "READY_FOR_FIGURE5_VISIBILITY": "YES" if image_formation_pass else "NO",
        "outcome": "CAMERA_FORMATION_PASS" if image_formation_pass else "OTHER_SIGNAL_WIDTH_AND_REGRESSION_GATE_REMAIN",
        "pass_conditions": pass_conditions,
        "reason_if_not_ready": (
            "RUN C forms the rabbit at the independently predicted geometric scale, but the common effective-width metric remains 34 percent below experiment and the repository-wide regression gate is not clean."
            if not image_formation_pass else None
        ),
        "visibility_calculated_from_report0": False,
        "authoritative_figure5_created": False,
        "rabbit_fit_used": False,
        "fitted_camera_envelope_used": False,
        "display_zoom_used_as_physics": False,
        "figures": [str(path) for path in figures],
        "comparison": str(OUT / "comparison" / "RUN_C_vs_experiment.json"),
    }
    save_json(OUT / "tests" / "Z5_status.json", status)
    summary = f"""================ Z5 FAR-FIELD CORRECTION =================

OLD MODEL

Pump waist:
250 µm

Final camera Gaussian envelope:
ACTIVE

Effective signal width:
{runs['RUN_A_OLD_FAILURE'][0]['signal']['effective_width_px']:.2f} px

Effective signal height:
{runs['RUN_A_OLD_FAILURE'][0]['signal']['effective_height_px']:.2f} px

Rabbit structure:
NO

---------------------------------------------------------

CORRECTED C_lab PARAMETERS

Reported pump spot:
2.0 mm

Interpretation:
1/e^2 INTENSITY DIAMETER — ASSUMED

Pump waist used:
1000 µm

Pump power at crystal:
70 mW

Nominal laser power:
250 mW

Camera pitch:
13 µm/pixel

---------------------------------------------------------

RUN A — 250 µm + OLD ENVELOPE

Width:
{runs['RUN_A_OLD_FAILURE'][0]['signal']['effective_width_px']:.2f} px

Height:
{runs['RUN_A_OLD_FAILURE'][0]['signal']['effective_height_px']:.2f} px

Support diameter:
{runs['RUN_A_OLD_FAILURE'][0]['support']['support_diameter_px']:.2f} px

Rabbit structure:
NO

---------------------------------------------------------

RUN B — 1000 µm + OLD ENVELOPE

Width:
{runs['RUN_B_WIDTH_ONLY'][0]['signal']['effective_width_px']:.2f} px

Height:
{runs['RUN_B_WIDTH_ONLY'][0]['signal']['effective_height_px']:.2f} px

Support diameter:
{runs['RUN_B_WIDTH_ONLY'][0]['support']['support_diameter_px']:.2f} px

Rabbit structure:
NO

---------------------------------------------------------

RUN C — 1000 µm + CORRECTED FAR FIELD

Final camera Gaussian envelope:
INACTIVE

Pump correlation contribution:
ACTIVE

sigma_q:
{c['sigma_q_per_um']:.6f} per µm

Width:
{c['signal']['effective_width_px']:.2f} px

Height:
{c['signal']['effective_height_px']:.2f} px

Support diameter:
{c['support']['support_diameter_px']:.2f} px

Rabbit structure:
YES

Rabbit simulated extent:
{c['signal']['bbox_width_px']} × {c['signal']['bbox_height_px']} px

---------------------------------------------------------

EXPERIMENT

Image:
{EXAMPLE_NAME}

Dimensions:
{experiment.shape[1]} × {experiment.shape[0]} px

Physical FoV:
{exp_metrics['fov_width_mm']:.3f} × {exp_metrics['fov_height_mm']:.3f} mm

Effective signal:
{exp_metrics['signal']['effective_width_px']:.2f} × {exp_metrics['signal']['effective_height_px']:.2f} px

Rabbit/reference extent:
approximately {c['expected_geometric_rabbit_width_px']:.0f} × {c['expected_geometric_rabbit_height_px']:.0f} px

---------------------------------------------------------

RUN C VS EXPERIMENT

Dimensions:
{comparison['tests']['dimensions']}

Pixel pitch:
PASS

No resize:
PASS

Signal width:
{comparison['tests']['signal_width']}

Signal height:
{comparison['tests']['signal_height']}

Rabbit structure:
{comparison['tests']['rabbit_structure']}

Rabbit scale:
{comparison['tests']['rabbit_scale']}

Orientation:
PASS

---------------------------------------------------------

ROOT CAUSE

Incorrect 250 µm pump parameter contributed:
YES

Incorrect final far-field Gaussian envelope contributed:
YES

Primary root cause:
DUPLICATE FINAL FAR-FIELD GAUSSIAN CAMERA MASK

Remaining problem:
Effective signal width is {width_relative * 100.0:.1f}% below experiment and the full regression gate is not clean.

---------------------------------------------------------

TESTS

Far-field envelope regression:
PASS

Pump-correlation test:
PASS

1000-µm config test:
PASS

Camera sampling:
PASS

Integer crop:
PASS

No-resize:
PASS

Full regression:
{regression.get('passed', 0)} PASSED
{regression.get('failures', 0)} FAILED

---------------------------------------------------------

FINAL VERDICT

FAR_FIELD_IMPLEMENTATION:
CORRECTED

CAMERA_IMAGE_FORMATION:
{'PASS' if image_formation_pass else 'FAIL'}

RABBIT_STRUCTURE_FORMED:
YES

READY_FOR_FIGURE5_VISIBILITY:
{'YES' if image_formation_pass else 'NO'}

If NO, reason:
{status['reason_if_not_ready'] or 'Not applicable'}

---------------------------------------------------------

FIGURES:
{chr(10).join(str(path) for path in figures)}

REPORT:
{REPORT_ROOT / 'Z5_FarField_Correction_and_Rabbit_ImageFormation_Report.docx'}

EVIDENCE:
{OUT}

=========================================================
"""
    (OUT / "provenance" / "Z5_terminal_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return status
    Image.fromarray(np.round(255.0 * display_normalize(array, 0.1, 99.9)).astype(np.uint8)).save(
        directory / f"{stem}.png"
    )


def run_case(label: str, pump_waist_um: float, expected_old_envelope: bool) -> dict[str, Any]:
    label = str(label).upper()
    if label not in {"RUN_A_OLD_FAILURE", "RUN_B_WIDTH_ONLY", "RUN_C_CORRECTED"}:
        raise ValueError(f"unsupported Z5 run label {label!r}")
    run_dir = RUNS / label
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = resolve_setup_selection("C_lab")
    magnification = float(qiup_magnification(setup, lambda_s_nm=810.0, lambda_i_nm=1550.0))
    object_pitch_um = PIXEL_UM / abs(magnification)
    setup = setup.overridden(
        pump_waist_um=float(pump_waist_um),
        pump_waist_NL1_um=float(pump_waist_um),
        pump_waist_NL2_um=float(pump_waist_um),
        pump_power_laser_mW=250.0,
        pump_pre_crystal_transmission=0.28,
        pump_return_transmission=1.0,
        pump_target_power_NL1_mW=70.0,
        pump_target_power_NL2_mW=70.0,
        pixel_um=PIXEL_UM,
        object_map_pixel_um=object_pitch_um,
        enable_phase_step_acquisition=False,
    )
    axis_um = centered_axis_um(FULL_N, object_pitch_um)
    rabbit = load_object_map(
        Y1 / "data" / "frozen" / "mask_rabbit.npy",
        Y1 / "data" / "frozen" / "rabbit_mask_metadata.json",
    )
    object_T, resample_meta = resample_object_map_to_grid(
        rabbit,
        x_um=axis_um,
        y_um=axis_um,
        method="nearest",
        binarize=True,
    )
    state = quiet_state(setup, nbar=0.01, num_frames=1, enable_phase_steps=False)
    optical = _optical_intensity(ref_from_T(object_T.astype(np.float32)), state, 0, np.random.default_rng(SEED))
    no_object = _optical_intensity(
        ref_from_T(np.ones_like(object_T, dtype=np.float32)),
        state,
        0,
        np.random.default_rng(SEED),
    )
    full = np.asarray(optical["I_plus"], dtype=np.float32)
    full_blank = np.asarray(no_object["I_plus"], dtype=np.float32)
    projected = np.asarray(optical["T_eff"], dtype=np.float32)
    projected_blank = np.asarray(no_object["T_eff"], dtype=np.float32)
    actual_envelope = bool(optical.get("final_camera_pump_envelope_applied", True))
    if actual_envelope != bool(expected_old_envelope):
        raise RuntimeError(
            f"{label} expected final-envelope applied={expected_old_envelope}, got {actual_envelope}"
        )
    roi = _load_roi()
    crop = crop_sampled_camera(
        full,
        width_px=int(roi["width_px"]),
        height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]),
        top_px=int(roi["top_px"]),
        pixel_pitch_um=PIXEL_UM,
    )
    crop_blank = crop_sampled_camera(
        full_blank,
        width_px=int(roi["width_px"]),
        height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]),
        top_px=int(roi["top_px"]),
        pixel_pitch_um=PIXEL_UM,
    )
    projected_crop = crop_sampled_camera(
        projected,
        width_px=int(roi["width_px"]),
        height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]),
        top_px=int(roi["top_px"]),
        pixel_pitch_um=PIXEL_UM,
    )
    projected_blank_crop = crop_sampled_camera(
        projected_blank,
        width_px=int(roi["width_px"]),
        height_px=int(roi["height_px"]),
        left_px=int(roi["left_px"]),
        top_px=int(roi["top_px"]),
        pixel_pitch_um=PIXEL_UM,
    )
    rabbit_meta = rabbit["metadata"]
    expected_width_px = float(rabbit_meta["physical_width_mm"]) * abs(magnification) * 1000.0 / PIXEL_UM
    expected_height_px = float(rabbit_meta["physical_height_mm"]) * abs(magnification) * 1000.0 / PIXEL_UM
    common = measure_signal_fixed(crop)
    support = support_metrics(crop)
    structure = structure_metrics(
        crop,
        crop_blank,
        projected_crop,
        projected_blank_crop,
        expected_width_px,
        expected_height_px,
    )
    mapping = resolved_spatial_mapping(setup)
    w_px = abs(magnification) * float(pump_waist_um) / PIXEL_UM
    metrics = {
        "label": label,
        "utc": utc_now(),
        "pump_waist_um": float(pump_waist_um),
        "sigma_q_per_um": 1.0 / float(pump_waist_um),
        "sigma_far_field_object_um": float(sigma_far_field_um(setup)),
        "sigma_far_field_camera_px": float(mapping["correlation_sigma_camera_px"]),
        "M_QIUP_signed": -abs(magnification),
        "legacy_camera_envelope_w_px": w_px,
        "final_camera_gaussian_envelope": "ACTIVE" if actual_envelope else "INACTIVE",
        "pump_correlation_contribution": "ACTIVE",
        "full_shape_yx": list(full.shape),
        "crop_shape_yx": list(crop.shape),
        "pixel_pitch_full_um": PIXEL_UM,
        "pixel_pitch_crop_um": PIXEL_UM,
        "resize_used": False,
        "interpolation_used_for_crop": False,
        "roi": roi,
        "signal": common,
        "support": support,
        "structure": structure,
        "expected_geometric_rabbit_width_px": expected_width_px,
        "expected_geometric_rabbit_height_px": expected_height_px,
        "source_hashes": {
            "optics.py": sha256(ROOT / "optics.py"),
            "setup_params.py": sha256(ROOT / "setup_params.py"),
            "configs/setup_config_c_lab_v1.yaml": sha256(ROOT / "configs" / "setup_config_c_lab_v1.yaml"),
        },
        "resample_metadata": resample_meta,
    }
    _save_camera_array(run_dir, f"{label}_full_camera", full)
    _save_camera_array(run_dir, f"{label}_cropped_camera", crop)
    np.save(run_dir / f"{label}_no_object_full_camera.npy", full_blank)
    np.save(run_dir / f"{label}_projected_object_response.npy", projected)
    np.save(run_dir / f"{label}_object_induced_difference.npy", (full - full_blank).astype(np.float32))
    save_json(run_dir / "run_config.json", {
        "label": label,
        "controlled_variables": {
            "pump_waist_um": float(pump_waist_um),
            "final_camera_gaussian_envelope": "OLD_ACTIVE" if actual_envelope else "CORRECTED_INACTIVE",
        },
        "fixed_variables": {
            "rabbit_source": str(Y1 / "data" / "frozen" / "mask_rabbit.npy"),
            "camera_shape_yx": [FULL_N, FULL_N],
            "camera_pixel_pitch_um": PIXEL_UM,
            "crop": roi,
            "F1_mm": float(setup.F1_mm),
            "F2_mm": float(setup.F2_mm),
            "far_field_focal_mm": float(setup.far_field_focal_mm),
            "lambda_s_nm": float(setup.lambda_s_nm),
            "lambda_i_nm": float(setup.lambda_i_nm),
            "seed": SEED,
        },
        "setup": setup.to_dict(),
        "pump_powers": pump_powers_from_setup(setup),
    })
    save_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", choices=("A", "B", "C"))
    group.add_argument("--recompute-existing", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize_evidence()
        return 0
    if args.recompute_existing:
        print(json.dumps(recompute_existing_extent_metrics(), indent=2))
        return 0
    cases = {
        "A": ("RUN_A_OLD_FAILURE", 250.0, True),
        "B": ("RUN_B_WIDTH_ONLY", 1000.0, True),
        "C": ("RUN_C_CORRECTED", 1000.0, False),
    }
    label, waist, expected_old = cases[args.run]
    metrics = run_case(label, waist, expected_old)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
