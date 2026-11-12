# -*- coding: utf-8 -*-
"""U15 publication-grade runtime scaling benchmark (Task 23)."""
from __future__ import annotations

import platform
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from typing import Any

import numpy as np

from benchmarks.targets import build_target
from setup_params import resolve_setup_selection
from simulate import simulate_forward
from verification.physics.u1_u9_runner import quiet_state
from verification.resolution.quantities import intensity_map_from_result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": None,
        "ram_total_gb": None,
        "numpy": np.__version__,
        "torch": None,
        "cuda_available": False,
        "gpu": None,
        "timestamp_utc": utc_now(),
    }
    try:
        import os

        info["cpu_count_logical"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _setup_for_n(profile: str, n: int) -> tuple[Any, float]:
    """Return setup with object-map pitch scaled so N*dx ≈ fixed domain."""
    setup = resolve_setup_selection(profile)
    stock_n = 64
    stock_px = float(getattr(setup, "object_map_pixel_um", None) or getattr(setup, "pixel_um", 16.0))
    domain = float(stock_n) * stock_px
    px = domain / float(n)
    return setup.overridden(object_map_pixel_um=float(px)), float(px)


def _one_forward(profile: str, n: int, *, seed: int, nbar: float) -> dict[str, Any]:
    setup, px = _setup_for_n(profile, n)
    tgt = build_target("transparent", int(n), pixel_um=float(px))
    state = quiet_state(setup, nbar=float(nbar))
    state.num_frames = 1
    t0 = time.perf_counter()
    tracemalloc.start()
    res = simulate_forward(
        tgt.to_simulation_reference(), state, seed=int(seed), detector_noise=False
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dt = time.perf_counter() - t0
    I = intensity_map_from_result(res)
    return {
        "runtime_s": float(dt),
        "peak_mem_bytes": int(peak),
        "current_mem_bytes": int(current),
        "finite": bool(np.isfinite(I).all()),
        "shape": list(I.shape),
        "intensity_mean": float(np.mean(I)),
    }


def time_grid_scaling(
    *,
    profile: str = "A",
    grids: tuple[int, ...] = (24, 32, 48, 64),
    repeats: int = 3,
    nbar: float = 0.01,
    seed0: int = 230015,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _one_forward(profile, grids[0], seed=seed0, nbar=nbar)
    for n in grids:
        times: list[float] = []
        mems: list[int] = []
        last: dict[str, Any] = {}
        for r in range(repeats):
            out = _one_forward(profile, int(n), seed=seed0 + r, nbar=nbar)
            times.append(out["runtime_s"])
            mems.append(out["peak_mem_bytes"])
            last = out
        rows.append(
            {
                "profile": profile,
                "grid_n": int(n),
                "n_pixels": int(n) * int(n),
                "repeats": repeats,
                "runtime_mean_s": float(statistics.mean(times)),
                "runtime_std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
                "runtime_median_s": float(statistics.median(times)),
                "runtime_min_s": float(min(times)),
                "runtime_max_s": float(max(times)),
                "peak_mem_mean_bytes": float(statistics.mean(mems)),
                "peak_mem_max_bytes": int(max(mems)),
                "finite": last.get("finite"),
                "shape": str(last.get("shape")),
            }
        )
    return rows


def time_frame_scaling(
    *,
    profile: str = "A",
    grid_n: int = 32,
    frame_counts: tuple[int, ...] = (1, 2, 4, 8),
    repeats: int = 3,
    nbar: float = 0.01,
    seed0: int = 230115,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _one_forward(profile, grid_n, seed=seed0, nbar=nbar)
    for nf in frame_counts:
        times: list[float] = []
        for r in range(repeats):
            t0 = time.perf_counter()
            for f in range(int(nf)):
                _one_forward(profile, grid_n, seed=seed0 + 1000 * r + f, nbar=nbar)
            times.append(time.perf_counter() - t0)
        rows.append(
            {
                "profile": profile,
                "grid_n": int(grid_n),
                "n_frames": int(nf),
                "repeats": repeats,
                "runtime_mean_s": float(statistics.mean(times)),
                "runtime_std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
                "runtime_median_s": float(statistics.median(times)),
                "runtime_min_s": float(min(times)),
                "runtime_max_s": float(max(times)),
                "runtime_per_frame_mean_s": float(statistics.mean(times)) / float(nf),
            }
        )
    return rows


def runtime_error_tradeoff(
    grid_rows: list[dict[str, Any]],
    *,
    reference_grid: int | None = None,
) -> list[dict[str, Any]]:
    if not grid_rows:
        return []
    ref_n = reference_grid or max(int(r["grid_n"]) for r in grid_rows)
    means: dict[int, float | None] = {}
    for r in grid_rows:
        n = int(r["grid_n"])
        out = _one_forward(r["profile"], n, seed=99, nbar=0.01)
        means[n] = out.get("intensity_mean")
    ref = means.get(ref_n)
    out_rows = []
    for r in grid_rows:
        n = int(r["grid_n"])
        m = means.get(n)
        rel = None
        if ref is not None and m is not None and abs(ref) > 0:
            rel = abs(float(m) - float(ref)) / abs(float(ref))
        out_rows.append(
            {
                "grid_n": n,
                "runtime_mean_s": r["runtime_mean_s"],
                "intensity_mean": m,
                "reference_grid_n": ref_n,
                "rel_intensity_mean_error_vs_ref": rel,
                "note": "Proxy numerical error = |mean(I)-mean(I_ref)|/|mean(I_ref)|; not a physical acceptance criterion",
            }
        )
    return out_rows
