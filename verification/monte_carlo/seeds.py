"""Deterministic SeedSequence management for independent MC realizations."""
from __future__ import annotations

from typing import Any

import numpy as np

from verification.monte_carlo.definitions import CHANNELS, MASTER_SEED, SEED_POLICY


def master_seed_sequence(master_seed: int = MASTER_SEED) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(master_seed))


def channel_root_sequence(
    channel: str,
    *,
    master_seed: int = MASTER_SEED,
) -> np.random.SeedSequence:
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}")
    idx = list(CHANNELS).index(channel)
    # Spawn one pool for channels from the master.
    channel_pool = master_seed_sequence(master_seed).spawn(len(CHANNELS))
    return channel_pool[idx]


def realization_seed_sequence(
    channel: str,
    realization_index: int,
    *,
    master_seed: int = MASTER_SEED,
    max_realizations: int,
) -> np.random.SeedSequence:
    """Unique child seed for (channel, realization_index)."""
    i = int(realization_index)
    n = int(max_realizations)
    if i < 0 or i >= n:
        raise IndexError(f"realization_index {i} out of range [0, {n})")
    root = channel_root_sequence(channel, master_seed=master_seed)
    children = root.spawn(n)
    return children[i]


def realization_seed_int(
    channel: str,
    realization_index: int,
    *,
    master_seed: int = MASTER_SEED,
    max_realizations: int,
) -> int:
    """Integer seed suitable for ``simulate_forward(..., seed=...)``."""
    ss = realization_seed_sequence(
        channel,
        realization_index,
        master_seed=master_seed,
        max_realizations=max_realizations,
    )
    # Stable 63-bit integer from SeedSequence entropy.
    return int(ss.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF))


def drift_phase_offset_rad(
    channel: str,
    realization_index: int,
    *,
    master_seed: int,
    max_realizations: int,
    rms_rad: float,
) -> float:
    """Independent Gaussian drift phase for the optional drift channel."""
    ss = realization_seed_sequence(
        channel,
        realization_index,
        master_seed=master_seed,
        max_realizations=max_realizations,
    )
    # Separate spawn so drift draw does not collide with simulate_forward seed int.
    drift_rng = np.random.default_rng(ss.spawn(1)[0])
    return float(drift_rng.normal(0.0, float(rms_rad)))


def seed_manifest(
    *,
    master_seed: int = MASTER_SEED,
    frame_counts: list[int],
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, Any]:
    max_n = int(max(frame_counts))
    rows = []
    for ch in channels:
        for i in range(max_n):
            rows.append(
                {
                    "channel": ch,
                    "realization_index": i,
                    "seed_int": realization_seed_int(
                        ch, i, master_seed=master_seed, max_realizations=max_n
                    ),
                }
            )
    return {
        "policy": SEED_POLICY,
        "master_seed": int(master_seed),
        "max_realizations": max_n,
        "frame_counts": list(frame_counts),
        "channels": list(channels),
        "realization_seeds": rows,
    }
