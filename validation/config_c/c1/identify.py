"""Task 18 Stage 1 — identify C1 mirror-drift recordings from Task 16 inventory.

No filename/meaning guessing: only explicit directory tokens and the Task 16
inventory columns are used. Micrometre stage readings are not present in the
TIFF tree; see freeze-plan provenance notes in the runner report.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Explicit acquisition tokens present under mirror drift/ (Task 16 inventory).
PHASE_LABELS: tuple[str, ...] = ("0", "pi2", "pi", "3pi2", "2pi")
SPECIAL_LABELS: tuple[str, ...] = ("dark", "ref")
OBJECT_TOKENS: tuple[str, ...] = ("witout object", "with object")

NOMINAL_PHASE_RAD: dict[str, float] = {
    "0": 0.0,
    "pi2": 0.5 * 3.141592653589793,
    "pi": 3.141592653589793,
    "3pi2": 1.5 * 3.141592653589793,
    "2pi": 2.0 * 3.141592653589793,
}

TARGET_STAGE_UM: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 150.0, 200.0)

# Provisional stage assignment from freeze-plan experimenter report:
# "25 µm stage increments produced approximately the intended sequence of
# phase states (approximately 0, π/2, π, 3π/2, 2π)".
# This is NOT measured per-file metadata and is NOT the ideal geometric law.
PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN: dict[str, float] = {
    "0": 0.0,
    "pi2": 25.0,
    "pi": 50.0,
    "3pi2": 75.0,  # consecutive 25 µm step; not in Task-18 target list
    "2pi": 100.0,
}

PRIMARY_SHAPE = "419x455"


@dataclass
class FileRecord:
    relative_path: str
    absolute_path: str
    filename: str
    object_token: str
    acquisition_label: str
    dimensions: str
    height: int
    width: int
    sha256: str
    file_size_bytes: int
    ok: bool


@dataclass
class IdentificationResult:
    experimental_root: str
    inventory_path: str
    n_mirror_files: int
    groups: dict[str, Any] = field(default_factory=dict)
    stage_um_status: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_task16_mirror_rows(inventory_csv: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(inventory_csv.open(encoding="utf-8")))
    return [r for r in rows if r.get("token_level_0") == "mirror drift"]


def identify_c1_sources(
    *,
    experimental_root: Path,
    inventory_csv: Path,
) -> IdentificationResult:
    root = Path(experimental_root)
    rows = load_task16_mirror_rows(inventory_csv)
    notes: list[str] = [
        "Classification uses Task 16 inventory token_level_* only.",
        "Folder token 'witout object' is the on-disk spelling (not corrected).",
        "Micrometre stage readings are absent from paths and TIFF metadata.",
    ]

    records: list[FileRecord] = []
    for r in rows:
        abs_path = root / Path(r["relative_path"])
        records.append(
            FileRecord(
                relative_path=r["relative_path"],
                absolute_path=str(abs_path),
                filename=r["filename"],
                object_token=r["token_level_1"],
                acquisition_label=r["token_level_2"],
                dimensions=r["dimensions"],
                height=int(float(r["height"])),
                width=int(float(r["width"])),
                sha256=r["sha256"],
                file_size_bytes=int(float(r["file_size_bytes"])),
                ok=str(r.get("ok", "True")).lower() in {"1", "true", "yes"},
            )
        )

    groups: dict[str, Any] = {}
    by_key: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for rec in records:
        by_key[(rec.object_token, rec.acquisition_label)].append(rec)

    for obj in OBJECT_TOKENS:
        for lab in list(PHASE_LABELS) + list(SPECIAL_LABELS):
            files = by_key.get((obj, lab), [])
            dim_counts: dict[str, int] = defaultdict(int)
            for f in files:
                dim_counts[f.dimensions] += 1
            primary = [f for f in files if f.dimensions == PRIMARY_SHAPE]
            groups[f"{obj}::{lab}"] = {
                "object_token": obj,
                "acquisition_label": lab,
                "n_files_all_shapes": len(files),
                "n_files_primary_shape": len(primary),
                "primary_shape": PRIMARY_SHAPE,
                "dimension_counts": dict(dim_counts),
                "nominal_phase_rad": NOMINAL_PHASE_RAD.get(lab),
                "is_phase_step": lab in PHASE_LABELS,
                "is_dark": lab == "dark",
                "is_ref": lab == "ref",
                "files_primary": [asdict(f) for f in primary],
                "files_other_shapes": [
                    asdict(f) for f in files if f.dimensions != PRIMARY_SHAPE
                ],
            }

    stage_status: dict[str, Any] = {
        "target_stage_um": list(TARGET_STAGE_UM),
        "file_level_stage_um": "UNRESOLVED",
        "file_level_reason": (
            "No directory or filename encodes stage position in micrometres; "
            "mirror-drift folders are labeled by phase-state tokens only."
        ),
        "provisional_mapping_source": (
            "Configuration_C_Experimental_Freeze_and_Validation_Plan §6 "
            "experimenter report: 25 µm increments ≈ {0, π/2, π, 3π/2, 2π}."
        ),
        "provisional_label_to_stage_um": dict(PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN),
        "per_target": {},
    }
    stage_to_label = {v: k for k, v in PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.items()}
    for z in TARGET_STAGE_UM:
        if z in stage_to_label:
            lab = stage_to_label[z]
            stage_status["per_target"][str(z)] = {
                "status": "PROVISIONAL_FROM_FREEZE_PLAN",
                "acquisition_label": lab,
                "note": (
                    "Assigned only via freeze-plan 25 µm quarter-wave sequence; "
                    "not read from file metadata."
                ),
            }
        else:
            stage_status["per_target"][str(z)] = {
                "status": "NO_DATA_IDENTIFIED",
                "acquisition_label": None,
                "note": (
                    "No mirror-drift folder uniquely identifies this stage "
                    "position; analysis for this target is STOPPED."
                ),
            }

    notes.append(
        "Primary analysis restricted to frames with shape 419x455 "
        "(dominant consistent mirror-drift size). Other shapes are indexed "
        "but not resized into the primary set."
    )

    return IdentificationResult(
        experimental_root=str(root),
        inventory_path=str(inventory_csv),
        n_mirror_files=len(records),
        groups=groups,
        stage_um_status=stage_status,
        notes=notes,
    )


def write_source_index(result: IdentificationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experimental_root": result.experimental_root,
        "inventory_path": result.inventory_path,
        "n_mirror_files": result.n_mirror_files,
        "notes": result.notes,
        "stage_um_status": result.stage_um_status,
        "groups": result.groups,
    }
    (out_dir / "c1_source_identification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = out_dir / "c1_primary_shape_file_index.csv"
    fields = [
        "object_token",
        "acquisition_label",
        "relative_path",
        "dimensions",
        "sha256",
        "file_size_bytes",
        "nominal_phase_rad",
        "provisional_stage_um",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for g in result.groups.values():
            for f in g["files_primary"]:
                lab = g["acquisition_label"]
                w.writerow(
                    {
                        "object_token": g["object_token"],
                        "acquisition_label": lab,
                        "relative_path": f["relative_path"],
                        "dimensions": f["dimensions"],
                        "sha256": f["sha256"],
                        "file_size_bytes": f["file_size_bytes"],
                        "nominal_phase_rad": g["nominal_phase_rad"],
                        "provisional_stage_um": PROVISIONAL_STAGE_UM_FROM_FREEZE_PLAN.get(
                            lab
                        ),
                    }
                )
