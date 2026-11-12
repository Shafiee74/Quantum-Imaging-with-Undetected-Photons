"""READ-ONLY recursive audit of the experimental-data tree."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from verification.lab_config_c.definitions import EXPERIMENTAL_DATA_ROOT_DEFAULT


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def infer_acquisition_group(rel_posix: str) -> dict[str, Any]:
    """Parse naming tokens only — no experimental semantics beyond path text."""
    parts = [p for p in Path(rel_posix).parts]
    filename = parts[-1] if parts else ""
    stem = Path(filename).stem
    group: dict[str, Any] = {
        "relative_dir": "/".join(parts[:-1]) if len(parts) > 1 else "",
        "path_tokens": parts[:-1],
        "filename_stem": stem,
        "inferred_from": "directory_and_filename_tokens_only",
    }
    # Explicit top-level folders present in this dataset
    if parts:
        group["token_level_0"] = parts[0]
    if len(parts) > 1:
        group["token_level_1"] = parts[1]
    if len(parts) > 2:
        group["token_level_2"] = parts[2]
    if len(parts) > 3:
        group["token_level_3"] = parts[3]
    # Preserve spelling as on disk (including 'witout')
    group["acquisition_group_path"] = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return group


def _open_tiff_stats(path: Path) -> dict[str, Any]:
    """Read TIFF metadata/statistics without modifying the file."""
    unresolved: list[str] = []
    try:
        with Image.open(path) as img:
            n_frames = int(getattr(img, "n_frames", 1) or 1)
            mode = str(img.mode)
            # Collect per-page shapes; do not rewrite anything.
            frames_meta: list[dict[str, Any]] = []
            mins: list[float] = []
            maxs: list[float] = []
            means: list[float] = []
            stds: list[float] = []
            for i in range(n_frames):
                try:
                    img.seek(i)
                except EOFError:
                    unresolved.append(f"frame_seek_eof_at_{i}")
                    break
                arr = np.asarray(img)
                frames_meta.append(
                    {
                        "frame_index": i,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                )
                flat = arr.astype(np.float64).ravel()
                mins.append(float(np.min(flat)))
                maxs.append(float(np.max(flat)))
                means.append(float(np.mean(flat)))
                stds.append(float(np.std(flat)))
            # Primary page for dimension inventory
            primary = frames_meta[0] if frames_meta else {
                "shape": None,
                "dtype": None,
            }
            # PIL info / tags often sparse for these lab TIFFs
            info_keys = sorted(str(k) for k in (img.info or {}).keys())
            if not info_keys:
                unresolved.append("no_pil_info_keys")
            # Camera EXIF typically absent
            exif = getattr(img, "getexif", lambda: None)()
            if not exif:
                unresolved.append("no_exif_camera_metadata")
            return {
                "ok": True,
                "error": None,
                "image_mode": mode,
                "n_frames": n_frames,
                "height": int(primary["shape"][0]) if primary.get("shape") else None,
                "width": int(primary["shape"][1]) if primary.get("shape") and len(primary["shape"]) > 1 else None,
                "shape": primary.get("shape"),
                "dtype": primary.get("dtype"),
                "min": float(min(mins)) if mins else None,
                "max": float(max(maxs)) if maxs else None,
                "mean": float(np.mean(means)) if means else None,
                "std": float(np.mean(stds)) if stds else None,
                "frames_meta": frames_meta,
                "pil_info_keys": info_keys,
                "unresolved_metadata": unresolved,
            }
    except Exception as exc:  # noqa: BLE001 — audit must continue
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "image_mode": None,
            "n_frames": None,
            "height": None,
            "width": None,
            "shape": None,
            "dtype": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "frames_meta": [],
            "pil_info_keys": [],
            "unresolved_metadata": ["read_failed", str(exc)],
        }


def audit_one_file(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    st = path.stat()
    # Ensure open is read-only at OS level where possible
    stats = _open_tiff_stats(path)
    group = infer_acquisition_group(rel)
    return {
        "relative_path": rel,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": int(st.st_size),
        "sha256": sha256_file(path),
        "dimensions_hw": (
            [stats["height"], stats["width"]]
            if stats["height"] is not None and stats["width"] is not None
            else None
        ),
        "height": stats["height"],
        "width": stats["width"],
        "dtype": stats["dtype"],
        "image_mode": stats["image_mode"],
        "n_frames": stats["n_frames"],
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"],
        "ok": stats["ok"],
        "error": stats["error"],
        "pil_info_keys": stats["pil_info_keys"],
        "unresolved_metadata": stats["unresolved_metadata"],
        "acquisition_group_path": group["acquisition_group_path"],
        "token_level_0": group.get("token_level_0"),
        "token_level_1": group.get("token_level_1"),
        "token_level_2": group.get("token_level_2"),
        "token_level_3": group.get("token_level_3"),
        "filename_stem": group["filename_stem"],
        "acquisition_inference_policy": group["inferred_from"],
        # Always unresolved at file level unless present in filename/dir
        "roi": None,
        "binning": None,
        "crop_origin": None,
        "preprocessing": None,
        "camera_metadata": None,
    }


def audit_experimental_tree(
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Walk the experimental tree read-only and return inventory records."""
    root_path = Path(root) if root is not None else EXPERIMENTAL_DATA_ROOT_DEFAULT
    if not root_path.is_dir():
        raise FileNotFoundError(f"experimental data root not found: {root_path}")

    # Refuse write-intent: open directory listing only
    if not os.access(root_path, os.R_OK):
        raise PermissionError(f"experimental root not readable: {root_path}")

    records: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            # Skip non-files
            if not path.is_file():
                continue
            records.append(audit_one_file(root_path, path))

    records.sort(key=lambda r: r["relative_path"])
    return {
        "root": str(root_path),
        "n_files": len(records),
        "records": records,
        "read_only": True,
    }
