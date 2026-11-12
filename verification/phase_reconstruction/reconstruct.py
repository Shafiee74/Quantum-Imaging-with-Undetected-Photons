"""Shared four-step / N-step phase reconstruction (sim + experiment).

Model per pixel
---------------
``I(θ) = A + C cos θ + S sin θ = A + B cos(θ + φ)``

with
``B = hypot(C, S)``, ``φ = atan2(-S, C)``, ``V = B / |A|``.

For the canonical four-step schedule ``θ ∈ {0, π/2, π, 3π/2}`` the
closed-form coefficients are used; for other schedules a least-squares
fit of the same model is used. Both paths share the same naming and
outputs so experimental Configuration-C frames use this module unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from verification.phase_reconstruction.constants import (
    FOUR_STEP_LABELS,
    FOUR_STEP_PHASES_RAD,
)
from verification.phase_reconstruction.metrics import visibility_from_ab, wrap_phase
from verification.phase_reconstruction.unwrap import maybe_unwrap


@dataclass(frozen=True)
class PhaseReconstructionResult:
    """Pixel-wise four-step / N-step reconstruction products."""

    dc_A: np.ndarray
    cos_C: np.ndarray
    sin_S: np.ndarray
    modulation_B: np.ndarray
    visibility_V: np.ndarray
    phase_wrapped_rad: np.ndarray
    phase_unwrapped_rad: np.ndarray | None
    residual_rms: np.ndarray
    snr: np.ndarray
    n_steps: int
    phases_rad: tuple[float, ...]
    method: str
    unwrap_enabled: bool
    unwrap_reason: str
    frames_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep arrays as arrays for callers; JSON export handled separately.
        return d

    def array_bundle(self) -> dict[str, np.ndarray | None]:
        return {
            "dc_A": self.dc_A,
            "cos_C": self.cos_C,
            "sin_S": self.sin_S,
            "modulation_B": self.modulation_B,
            "visibility_V": self.visibility_V,
            "phase_wrapped_rad": self.phase_wrapped_rad,
            "phase_unwrapped_rad": self.phase_unwrapped_rad,
            "residual_rms": self.residual_rms,
            "snr": self.snr,
        }


def _as_frame_stack(
    frames: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
) -> np.ndarray:
    if isinstance(frames, Mapping):
        missing = [k for k in FOUR_STEP_LABELS if k not in frames]
        # Also accept short aliases.
        aliases = {
            "I0": "I0",
            "I_pi_2": "I_pi_2",
            "Iπ/2": "I_pi_2",
            "I_pi2": "I_pi_2",
            "Ipi2": "I_pi_2",
            "I_pi": "I_pi",
            "Iπ": "I_pi",
            "Ipi": "I_pi",
            "I_3pi_2": "I_3pi_2",
            "I3π/2": "I_3pi_2",
            "I_3pi2": "I_3pi_2",
            "I3pi2": "I_3pi_2",
        }
        ordered: list[np.ndarray] = []
        if not missing and all(k in frames for k in FOUR_STEP_LABELS):
            for key in FOUR_STEP_LABELS:
                ordered.append(np.asarray(frames[key], dtype=np.float64))
        else:
            # Try alias resolution in FOUR_STEP order.
            by_canon: dict[str, np.ndarray] = {}
            for key, arr in frames.items():
                canon = aliases.get(str(key), str(key))
                by_canon[canon] = np.asarray(arr, dtype=np.float64)
            for key in FOUR_STEP_LABELS:
                if key not in by_canon:
                    raise KeyError(
                        f"four-step frames missing {key!r}; "
                        f"got keys {sorted(map(str, frames.keys()))}"
                    )
                ordered.append(by_canon[key])
        stack = np.stack(ordered, axis=0)
    elif isinstance(frames, np.ndarray):
        stack = np.asarray(frames, dtype=np.float64)
    else:
        stack = np.stack(
            [np.asarray(f, dtype=np.float64) for f in frames], axis=0
        )
    if stack.ndim != 3:
        raise ValueError(
            f"phase-step frames must have shape (K, H, W); got {stack.shape}"
        )
    if not np.all(np.isfinite(stack)):
        raise ValueError("phase-step frames contain NaN/Inf")
    return stack


def _is_four_step_schedule(phases: np.ndarray) -> bool:
    if phases.size != 4:
        return False
    ref = np.asarray(FOUR_STEP_PHASES_RAD, dtype=np.float64)
    return bool(np.allclose(phases, ref, rtol=0.0, atol=1.0e-12))


def _four_step_closed_form(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact four-bucket coefficients for θ = 0, π/2, π, 3π/2."""
    I0, I1, I2, I3 = stack[0], stack[1], stack[2], stack[3]
    A = 0.25 * (I0 + I1 + I2 + I3)
    C = 0.5 * (I0 - I2)
    S = 0.5 * (I1 - I3)
    return A, C, S


def _n_step_least_squares(
    stack: np.ndarray, phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack(
        (np.ones(phases.size), np.cos(phases), np.sin(phases))
    )
    pixels = stack.reshape(phases.size, -1)
    coeffs, *_ = np.linalg.lstsq(design, pixels, rcond=None)
    h, w = stack.shape[1], stack.shape[2]
    A = coeffs[0].reshape(h, w)
    C = coeffs[1].reshape(h, w)
    S = coeffs[2].reshape(h, w)
    return A, C, S


def reconstruct_phase_steps(
    frames: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
    phases_rad: Sequence[float] | np.ndarray | None = None,
    *,
    unwrap: bool = False,
    unwrap_reason: str | None = None,
) -> PhaseReconstructionResult:
    """Reconstruct A, C, S, B, V, and wrapped (optional unwrapped) phase.

    Parameters
    ----------
    frames
        ``(K,H,W)`` stack, sequence of K arrays, or mapping with keys
        ``I0``, ``I_pi_2``, ``I_pi``, ``I_3pi_2`` (aliases accepted).
    phases_rad
        Interferometer phase steps θ. Defaults to the canonical four-step
        schedule when ``frames`` provides four frames / named keys.
    unwrap
        If True, also return a 2-D unwrapped phase map.
    unwrap_reason
        Provenance string explaining why unwrap was enabled.
    """
    stack = _as_frame_stack(frames)
    if phases_rad is None:
        if stack.shape[0] != 4:
            raise ValueError(
                "phases_rad is required when frame count is not 4"
            )
        phases = np.asarray(FOUR_STEP_PHASES_RAD, dtype=np.float64)
    else:
        phases = np.asarray(list(phases_rad), dtype=np.float64).reshape(-1)
    if phases.size != stack.shape[0]:
        raise ValueError(
            f"phase-step count {phases.size} does not match frames {stack.shape[0]}"
        )
    if phases.size < 3:
        raise ValueError("phase retrieval requires at least 3 steps")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phase-step schedule must be finite")

    if _is_four_step_schedule(phases):
        A, C, S = _four_step_closed_form(stack)
        method = "four_step_closed_form"
    else:
        A, C, S = _n_step_least_squares(stack, phases)
        method = "least_squares_n_step"

    B = np.hypot(C, S)
    phase = np.arctan2(-S, C)
    V = visibility_from_ab(B, A)

    # Residual against the fitted model.
    design = np.column_stack(
        (np.ones(phases.size), np.cos(phases), np.sin(phases))
    )
    fitted = (
        A[None, :, :]
        + C[None, :, :] * np.cos(phases)[:, None, None]
        + S[None, :, :] * np.sin(phases)[:, None, None]
    )
    residual = stack - fitted
    residual_rms = np.sqrt(np.mean(residual * residual, axis=0))
    snr = B / np.maximum(residual_rms, 1.0e-12)

    unwrap_pack = maybe_unwrap(
        phase,
        enabled=unwrap,
        reason=unwrap_reason
        or ("physically_continuous_map" if unwrap else "not_requested"),
    )

    return PhaseReconstructionResult(
        dc_A=A.astype(np.float64),
        cos_C=C.astype(np.float64),
        sin_S=S.astype(np.float64),
        modulation_B=B.astype(np.float64),
        visibility_V=V.astype(np.float64),
        phase_wrapped_rad=np.asarray(
            unwrap_pack["phase_wrapped_rad"], dtype=np.float64
        ),
        phase_unwrapped_rad=(
            None
            if unwrap_pack["phase_unwrapped_rad"] is None
            else np.asarray(unwrap_pack["phase_unwrapped_rad"], dtype=np.float64)
        ),
        residual_rms=residual_rms.astype(np.float64),
        snr=snr.astype(np.float64),
        n_steps=int(phases.size),
        phases_rad=tuple(float(p) for p in phases),
        method=method,
        unwrap_enabled=bool(unwrap_pack["unwrap_enabled"]),
        unwrap_reason=str(unwrap_pack["unwrap_reason"]),
        frames_shape=tuple(int(x) for x in stack.shape),
    )


def reconstruct_four_step(
    I0: np.ndarray,
    I_pi_2: np.ndarray,
    I_pi: np.ndarray,
    I_3pi_2: np.ndarray,
    *,
    unwrap: bool = False,
    unwrap_reason: str | None = None,
) -> PhaseReconstructionResult:
    """Convenience API for named four-step intensity frames."""
    return reconstruct_phase_steps(
        {
            "I0": I0,
            "I_pi_2": I_pi_2,
            "I_pi": I_pi,
            "I_3pi_2": I_3pi_2,
        },
        FOUR_STEP_PHASES_RAD,
        unwrap=unwrap,
        unwrap_reason=unwrap_reason,
    )


# Silence unused import lint for wrap_phase re-export convenience in tests.
__all__ = [
    "PhaseReconstructionResult",
    "reconstruct_four_step",
    "reconstruct_phase_steps",
    "wrap_phase",
]
