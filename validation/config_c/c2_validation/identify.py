"""Task 19 — identify C2 longitudinal/transverse files from Task 16 inventory.

Mappings use explicit freeze-plan stage coordinates that appear in filenames:
  L: z0=6.910 mm; z∈{6.910,9.310,10.490,12.722} → δ={0,2.400,3.580,5.812} mm
  T: x0=4.340 mm; x∈{4.340…4.520} → Δx={0,20,30,40,60,180} µm
Folder token ``witout object/0`` is an additional δ=0 / aligned set.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Z0_MM = 6.910
LONGITUDINAL_TARGETS_MM: dict[float, float] = {
    6.910: 0.0,
    9.310: 2.400,
    10.490: 3.580,
    12.722: 5.812,
}
TARGET_DELTA_Z_MM: tuple[float, ...] = (0.0, 2.400, 3.580, 5.812)

X0_MM = 4.340
TRANSVERSE_TARGETS_MM: dict[float, float] = {
    4.340: 0.0,
    4.360: 20.0,
    4.370: 30.0,
    4.380: 40.0,
    4.400: 60.0,
    4.520: 180.0,
}
TARGET_DELTA_X_UM: tuple[float, ...] = (0.0, 20.0, 30.0, 40.0, 60.0, 180.0)


def _parse_stage_mm_from_name(name: str) -> float | None:
    """Parse stage millimetre readings embedded in filenames (explicit digits)."""
    stem = Path(name).stem
    # Patterns: L=6_910mm, 6_910mm, 10_490, T=4_340mm, 4-360, 4_400
    patterns = [
        r"(?:L=)?(\d+)_(\d+)mm",
        r"(?:T=)?(\d+)_(\d+)mm",
        r"(?:T=)?(\d+)_(\d+)\b",
        r"(?:T=)?(\d+)-(\d+)\b",
        r"(\d+)_(\d+)mm",
        r"(\d+)_(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            whole = float(f"{m.group(1)}.{m.group(2)}")
            return whole
    return None


@dataclass
class FileRec:
    relative_path: str
    absolute_path: str
    filename: str
    object_token: str
    drift_token: str
    dimensions: str
    height: int
    width: int
    sha256: str
    file_size_bytes: int
    stage_mm: float | None
    delta_z_mm: float | None = None
    delta_x_um: float | None = None


@dataclass
class C2Identification:
    experimental_root: str
    inventory_path: str
    longitudinal: dict[str, Any] = field(default_factory=dict)
    transverse: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def identify_c2(
    *,
    experimental_root: Path,
    inventory_csv: Path,
) -> C2Identification:
    root = Path(experimental_root)
    rows = [
        r
        for r in csv.DictReader(inventory_csv.open(encoding="utf-8"))
        if r.get("token_level_0") == "lens drift"
    ]
    notes = [
        "Classification uses Task 16 inventory + freeze-plan stage coordinates "
        "that appear explicitly in filenames.",
        "Folder spelling 'witout object' is preserved.",
        "Experimental camera pixel pitch remains UNKNOWN (Task 16); physical "
        "µm conversions for experiment use ASSUMED stock pixel_um=16 only when "
        "labeled as such.",
    ]

    longs: dict[tuple[str, float], list[FileRec]] = defaultdict(list)
    trans: dict[tuple[str, float], list[FileRec]] = defaultdict(list)
    unmatched: list[str] = []

    for r in rows:
        abs_path = root / Path(r["relative_path"])
        obj = r["token_level_1"]
        drift = r["token_level_2"]
        # Folder '0' under witout/with object is aligned zero (no stage digits).
        # Do not parse QIUP timestamps as stage readings.
        if drift == "0":
            stage = Z0_MM
        elif drift in {"L drift", "T drift"}:
            stage = _parse_stage_mm_from_name(r["filename"])
            # Reject timestamp-like parses (year.month from QIUP_Mon Sep 28 2026_...)
            if stage is not None and stage > 100.0:
                stage = None
        else:
            stage = None
        rec = FileRec(
            relative_path=r["relative_path"],
            absolute_path=str(abs_path),
            filename=r["filename"],
            object_token=obj,
            drift_token=drift,
            dimensions=r["dimensions"],
            height=int(float(r["height"])),
            width=int(float(r["width"])),
            sha256=r["sha256"],
            file_size_bytes=int(float(r["file_size_bytes"])),
            stage_mm=stage,
        )
        if drift in {"L drift", "0"} and stage is not None:
            # Nearest known longitudinal stage (avoid float-tolerance collisions)
            best_z = min(
                LONGITUDINAL_TARGETS_MM.keys(),
                key=lambda z: abs(stage - z),
            )
            if abs(stage - best_z) <= 0.05:
                dlt = LONGITUDINAL_TARGETS_MM[best_z]
                rec.delta_z_mm = dlt
                longs[(obj, dlt)].append(rec)
            else:
                unmatched.append(r["relative_path"])
        elif drift == "T drift" and stage is not None:
            best_x = min(
                TRANSVERSE_TARGETS_MM.keys(),
                key=lambda x: abs(stage - x),
            )
            if abs(stage - best_x) <= 0.005:
                dlt = TRANSVERSE_TARGETS_MM[best_x]
                rec.delta_x_um = dlt
                trans[(obj, dlt)].append(rec)
            else:
                unmatched.append(r["relative_path"])
        else:
            unmatched.append(r["relative_path"])

    def _pack_long() -> dict[str, Any]:
        out: dict[str, Any] = {"targets_mm": list(TARGET_DELTA_Z_MM), "groups": {}}
        for obj in ("witout object", "with object"):
            for dlt in TARGET_DELTA_Z_MM:
                files = longs.get((obj, dlt), [])
                dims: dict[str, int] = defaultdict(int)
                for f in files:
                    dims[f.dimensions] += 1
                # prefer largest consistent group shape
                primary = None
                if dims:
                    primary = max(dims.items(), key=lambda kv: kv[1])[0]
                primary_files = [f for f in files if f.dimensions == primary]
                out["groups"][f"{obj}::dz_{dlt:.3f}mm"] = {
                    "object_token": obj,
                    "delta_z_mm": dlt,
                    "delta_z_um": dlt * 1.0e3,
                    "n_files": len(files),
                    "dimension_counts": dict(dims),
                    "primary_shape": primary,
                    "files_primary": [asdict(f) for f in primary_files],
                    "status": "OK" if primary_files else "NO_DATA",
                }
        return out

    def _pack_trans() -> dict[str, Any]:
        out: dict[str, Any] = {"targets_um": list(TARGET_DELTA_X_UM), "groups": {}}
        for obj in ("witout object", "with object"):
            for dlt in TARGET_DELTA_X_UM:
                files = trans.get((obj, dlt), [])
                dims: dict[str, int] = defaultdict(int)
                for f in files:
                    dims[f.dimensions] += 1
                primary = max(dims.items(), key=lambda kv: kv[1])[0] if dims else None
                primary_files = [f for f in files if f.dimensions == primary]
                out["groups"][f"{obj}::dx_{dlt:.0f}um"] = {
                    "object_token": obj,
                    "delta_x_um": dlt,
                    "n_files": len(files),
                    "dimension_counts": dict(dims),
                    "primary_shape": primary,
                    "files_primary": [asdict(f) for f in primary_files],
                    "status": "OK" if primary_files else "NO_DATA",
                }
        return out

    if unmatched:
        notes.append(
            f"Unmatched lens-drift paths (not in target map): {len(unmatched)}"
        )

    return C2Identification(
        experimental_root=str(root),
        inventory_path=str(inventory_csv),
        longitudinal=_pack_long(),
        transverse=_pack_trans(),
        notes=notes,
    )
