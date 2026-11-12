"""Aggregate statistics and confidence intervals for MC sweeps."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verification.monte_carlo.definitions import CI_LEVEL, CI_Z


def _as_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def sample_mean(x: Sequence[float] | np.ndarray) -> float:
    a = _as_1d(x)
    return float(np.mean(a)) if a.size else float("nan")


def sample_variance(x: Sequence[float] | np.ndarray, *, ddof: int = 1) -> float:
    a = _as_1d(x)
    if a.size <= ddof:
        return float("nan")
    return float(np.var(a, ddof=ddof))


def sample_std(x: Sequence[float] | np.ndarray, *, ddof: int = 1) -> float:
    v = sample_variance(x, ddof=ddof)
    return float(np.sqrt(v)) if np.isfinite(v) else float("nan")


def confidence_interval_mean(
    x: Sequence[float] | np.ndarray,
    *,
    z: float = CI_Z,
    level: float = CI_LEVEL,
) -> dict[str, Any]:
    a = _as_1d(x)
    n = int(a.size)
    mean = sample_mean(a)
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "level": float(level),
            "z": float(z),
        }
    sem = float(sample_std(a) / np.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci_low": float(mean - z * sem),
        "ci_high": float(mean + z * sem),
        "level": float(level),
        "z": float(z),
    }


def covariance_pair(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    ddof: int = 1,
) -> float:
    a = _as_1d(x)
    b = _as_1d(y)
    if a.size != b.size or a.size <= ddof:
        return float("nan")
    return float(np.cov(a, b, ddof=ddof)[0, 1])


def normalized_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    a = _as_1d(x)
    b = _as_1d(y)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_scalar_series(
    name: str,
    values: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    a = _as_1d(values)
    ci = confidence_interval_mean(a)
    return {
        "metric": name,
        "n": int(a.size),
        "mean": sample_mean(a),
        "variance": sample_variance(a),
        "std": sample_std(a),
        "ci": ci,
        "values": a.tolist(),
    }
