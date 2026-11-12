"""Shared paths and I/O for Task Z2."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT.parent / "Report"
EXP_ROOT = REPORT / "Received_Verification" / "for generate results" / "exp data"
Z1 = ROOT / "evidence" / "taskZ1_figure5_quantitative_reconstruction"
OUT = ROOT / "evidence" / "taskZ2_figure5_rootcause_and_rebuild"
Y1 = ROOT / "evidence" / "taskY1_rabbit_ground_truth"
X9 = ROOT / "evidence" / "taskX9_propagation"

RABBIT_HASH = "2302a5e97c352a92c812c8f83c84d4f7d21b9157e05c7fa712f88c4dc2590683"
PIXEL_UM = 13.0  # Andor iXon Ultra 888 (camera plane)
SENSOR_N = 1024
GRID_N = 128
NBAR = 0.01
SEED = 20261004
LAMBDA_I_NM = 1550.0
PHASE_LABELS = ("0", "pi2", "pi", "3pi2")
PHASE_RAD = (0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2)
STAGE_UM_PROVISIONAL = (0.0, 25.0, 50.0, 75.0)

Z2_DIRS = (
    "audit",
    "baseline",
    "camera_model",
    "coherence",
    "detector",
    "experiment",
    "simulation",
    "registration",
    "phase",
    "object",
    "metrics",
    "figures",
    "supplement",
    "tests",
    "provenance",
    "report",
    "data",
    "logs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def ensure_dirs() -> None:
    for d in Z2_DIRS:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr[0]
    return arr


def rotate180(a: np.ndarray) -> np.ndarray:
    return np.rot90(a, 2)


def export_figure(fig, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
