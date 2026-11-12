#!/usr/bin/env python3
"""Replace full MNIST train idx files with a small random subset (same format).

Keeps torchvision-compatible paths under ``data/MNIST/raw/``:
  train-images-idx3-ubyte, train-labels-idx1-ubyte

Removes unused ``t10k-*`` test files to shrink the tree.
"""
from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "MNIST" / "raw"
IMAGES = RAW / "train-images-idx3-ubyte"
LABELS = RAW / "train-labels-idx1-ubyte"
MANIFEST = ROOT / "data" / "MNIST" / "subset_manifest.json"


def _read_images(path: Path) -> tuple[int, int, int, np.ndarray]:
    with path.open("rb") as fh:
        magic, count, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise RuntimeError(f"bad image magic {magic} in {path}")
        data = np.frombuffer(fh.read(count * rows * cols), dtype=np.uint8)
        if data.size != count * rows * cols:
            raise RuntimeError(f"truncated image payload in {path}")
    return count, rows, cols, data.reshape(count, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise RuntimeError(f"bad label magic {magic} in {path}")
        labels = np.frombuffer(fh.read(count), dtype=np.uint8)
        if labels.size != count:
            raise RuntimeError(f"truncated label payload in {path}")
    return labels


def _write_images(path: Path, images: np.ndarray) -> None:
    count, rows, cols = images.shape
    with path.open("wb") as fh:
        fh.write(struct.pack(">IIII", 2051, count, rows, cols))
        fh.write(np.ascontiguousarray(images, dtype=np.uint8).tobytes())


def _write_labels(path: Path, labels: np.ndarray) -> None:
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    with path.open("wb") as fh:
        fh.write(struct.pack(">II", 2049, labels.size))
        fh.write(labels.tobytes())


def build_subset(n: int, seed: int | None) -> dict:
    if not IMAGES.is_file() or not LABELS.is_file():
        raise FileNotFoundError(f"need full MNIST train files at {RAW}")
    count, rows, cols, images = _read_images(IMAGES)
    labels = _read_labels(LABELS)
    if labels.size != count:
        raise RuntimeError("image/label count mismatch")
    if n > count:
        raise ValueError(f"n={n} exceeds available train samples ({count})")

    rng = np.random.default_rng(seed)
    # True random draw without replacement from the current train pool.
    chosen = np.sort(rng.choice(count, size=n, replace=False))
    sub_images = images[chosen]
    sub_labels = labels[chosen]

    RAW.mkdir(parents=True, exist_ok=True)
    _write_images(IMAGES, sub_images)
    _write_labels(LABELS, sub_labels)

    # Drop unused test split (~8 MB).
    for name in (
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ):
        path = RAW / name
        if path.is_file():
            path.unlink()

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subset_size": int(n),
        "seed": seed,
        "source_pool_count": int(count),
        "image_shape": [int(rows), int(cols)],
        "source_indices": [int(i) for i in chosen.tolist()],
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(sub_labels, return_counts=True))},
        "files": {
            "images": str(IMAGES.relative_to(ROOT).as_posix()),
            "labels": str(LABELS.relative_to(ROOT).as_posix()),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of random train images")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (omit for a non-reproducible random draw)",
    )
    args = parser.parse_args()
    seed = args.seed
    if seed is None:
        # One-shot unpredictable seed, still recorded in the manifest.
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    manifest = build_subset(args.n, seed)
    print(
        f"wrote {manifest['subset_size']} images "
        f"(seed={manifest['seed']}) -> {IMAGES}"
    )
    print(f"manifest {MANIFEST}")
    print("label_counts", manifest["label_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
