#!/usr/bin/env python3
"""Emit C_lab Y1 rabbit final physical-dimension freeze evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from object_maps.c_lab_y1_rabbit_physical import (
    Y1_MASK_DEFAULT,
    Y1_META_DEFAULT,
    aperture_bbox_px,
    load_c_lab_y1_object_map,
    predicted_camera_footprint_mm,
    resolved_aperture_mm,
)
from setup_params import (
    delta_k_qpm_per_m_from_setup,
    resolve_setup_selection,
    resolved_spatial_mapping,
    setup_selection_info,
)

EVIDENCE_ROOT = ROOT / "evidence" / "c_lab_rabbit_final_dimensions_freeze"
PRODUCTION_PATHS = [
    ROOT / "configs" / "setup_config_c_lab_v1.yaml",
    ROOT / "object_maps" / "rabbit_loader.py",
    ROOT / "object_maps" / "c_lab_y1_rabbit_physical.py",
    ROOT / "object_maps" / "__init__.py",
    ROOT / "setup_params.py",
    ROOT / "tools" / "c_lab_a6_2_calibrated_1024_audit.py",
    ROOT / "tools" / "c_lab_a6_2_lite_scale_pupil.py",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = EVIDENCE_ROOT / utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    setup = resolve_setup_selection("C_lab")
    w_mm, h_mm = resolved_aperture_mm(setup)
    mask = np.load(Y1_MASK_DEFAULT)
    mask_hash = sha256_file(Y1_MASK_DEFAULT)
    bbox = aperture_bbox_px(mask)
    loaded = load_c_lab_y1_object_map(setup)
    cal = loaded["physical_calibration"]
    foot = predicted_camera_footprint_mm(setup)
    mapping = resolved_spatial_mapping(setup)

    report = {
        "authoritative_aperture_mm": {"width_x": w_mm, "height_y": h_mm},
        "orientation": cal["axis_convention"],
        "native_mask": {
            "path": str(Y1_MASK_DEFAULT.relative_to(ROOT)),
            "sha256": mask_hash,
            "shape_hw": list(mask.shape),
            "transmission_bbox_px": bbox,
        },
        "aspect_ratio": {
            "bbox_wh": cal["aspect_ratio_bbox"],
            "caliper_wh": cal["aspect_ratio_caliper"],
            "discrepancy": cal["aspect_ratio_discrepancy"],
            "resolution": "Independent um/px_x and um/px_y; contour not rescaled.",
        },
        "source_plane_pitch": {
            "um_per_px_x": cal["um_per_px_x"],
            "um_per_px_y": cal["um_per_px_y"],
        },
        "optical_geometry_unchanged": {
            "L_mm": float(setup.crystal_length_mm),
            "Ms": float(setup.far_field_signal_relay_magnification),
            "Mi": float(setup.far_field_idler_relay_magnification),
            "delta_k0_ix1": float(delta_k_qpm_per_m_from_setup(setup, interaction=1)),
            "delta_k0_ix2": float(delta_k_qpm_per_m_from_setup(setup, interaction=2)),
        },
        "camera_footprint_prediction": foot,
        "M_signed": float(mapping["signed_object_to_camera_magnification"]),
        "yaml_selection": setup_selection_info(setup),
        "superseded_values_mm": {"width": 7.389330306469919, "height": 10.051078320090804},
    }
    (out / "RABBIT_FINAL_DIMENSIONS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in PRODUCTION_PATHS}
    manifest["evidence/taskY1_rabbit_ground_truth/data/frozen/mask_rabbit.npy"] = mask_hash
    (out / "SHA256_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    md = [
        "# C_lab Y1 rabbit — final physical dimensions freeze",
        "",
        f"| Axis | mm |",
        f"|------|-----:|",
        f"| **x (width, shorter)** | **{w_mm:.3f}** |",
        f"| **y (height, longer)** | **{h_mm:.3f}** |",
        "",
        f"- Native mask SHA256 unchanged: `{mask_hash[:16]}…`",
        f"- Image shape (H×W): {mask.shape[0]}×{mask.shape[1]} px; transmission bbox: {bbox['width_px']}×{bbox['height_px']} px",
        f"- Pitch: {cal['um_per_px_x']:.4f} µm/px (x), {cal['um_per_px_y']:.4f} µm/px (y)",
        f"- Aspect discrepancy (bbox−caliper): {cal['aspect_ratio_discrepancy']:.4f}",
        f"- Predicted camera footprint: {foot['camera_aperture_width_mm']:.3f} × {foot['camera_aperture_height_mm']:.3f} mm "
        f"({foot['camera_aperture_width_px']:.1f} × {foot['camera_aperture_height_px']:.1f} px @ 13 µm)",
        f"- M magnitude: {foot['M_magnitude']:.6f}",
    ]
    (out / "RABBIT_FINAL_DIMENSIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"evidence={out}")


if __name__ == "__main__":
    main()
