"""Discover axial idler-lens positions from experimental TIFF filenames."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import tifffile

from tools.c_lab_figure6_lens_displacement_validation import parse_abs_mm

# Validated laboratory reference stage coordinate (L = 6.910 mm series).
Z_REFERENCE_MM = 6.910
AXIAL_EXP_FOLDER_NAME = "L drift"
ACQUISITION = "without object, axial idler-lens drift"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_filename_stage_mm(name: str) -> tuple[float | None, str]:
    """Return (absolute stage mm, note)."""
    abs_mm = parse_abs_mm(name)
    if abs_mm is not None:
        return abs_mm, "absolute stage coordinate from filename pattern"
    m_alt = re.search(r"(\d+\.?\d*)\s*mm", name, re.I)
    if m_alt:
        return float(m_alt.group(1)), "alternate mm token in filename"
    return None, "unparsed — ambiguous filename"


def build_provenance_table(folder: Path) -> tuple[list[float], list[dict[str, Any]]]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows: list[dict[str, Any]] = []
    by_dz: dict[float, list[Path]] = {}
    seen_hash: dict[float, set[str]] = {}

    for p in sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff")):
        abs_mm, parse_note = parse_filename_stage_mm(p.name)
        if abs_mm is None:
            rows.append(
                {
                    "filename": p.name,
                    "absolute_path": str(p.resolve()),
                    "parsed_numerical_value": "",
                    "parsed_unit": "",
                    "value_kind": "ambiguous",
                    "z_reference_mm": Z_REFERENCE_MM,
                    "delta_z_lens_mm": "",
                    "acquisition_condition": ACQUISITION,
                    "sha256": sha256_file(p),
                    "inclusion_status": "excluded_unparsed",
                }
            )
            continue
        dz = round(abs_mm - Z_REFERENCE_MM, 6)
        digest = sha256_file(p)
        dup = digest in seen_hash.setdefault(dz, set())
        if not dup:
            seen_hash[dz].add(digest)
            by_dz.setdefault(dz, []).append(p)
        rows.append(
            {
                "filename": p.name,
                "absolute_path": str(p.resolve()),
                "parsed_numerical_value": abs_mm,
                "parsed_unit": "mm",
                "value_kind": "absolute_stage_coordinate",
                "z_reference_mm": Z_REFERENCE_MM,
                "delta_z_lens_mm": dz,
                "acquisition_condition": ACQUISITION,
                "sha256": digest,
                "inclusion_status": "duplicate_excluded_from_mean"
                if dup
                else "included",
                "parse_note": parse_note,
            }
        )

    dz_list = sorted(by_dz.keys())
    if not dz_list:
        raise FileNotFoundError(f"No parseable axial TIFFs in {folder}")
    return dz_list, rows


def mean_stack_deduped(paths: list[Path]) -> tuple[Any, int]:
    import numpy as np

    frames: list[np.ndarray] = []
    seen: set[str] = set()
    for p in paths:
        d = sha256_file(p)
        if d in seen:
            continue
        seen.add(d)
        frames.append(tifffile.imread(p).astype(np.float64))
    if not frames:
        raise ValueError("No unique frames")
    stack = np.stack(frames, axis=0)
    return np.mean(stack, axis=0), len(frames)


def load_experimental_means(folder: Path) -> tuple[list[float], dict[float, Any], dict[float, int], list[dict]]:
    dz_list, audit = build_provenance_table(folder)
    means: dict[float, Any] = {}
    n_frames: dict[float, int] = {}
    for dz in dz_list:
        paths = sorted(
            {
                Path(r["absolute_path"])
                for r in audit
                if r.get("delta_z_lens_mm") == dz and r.get("inclusion_status") == "included"
            }
        )
        img, n = mean_stack_deduped(paths)
        means[dz] = img
        n_frames[dz] = n
    return dz_list, means, n_frames, audit
