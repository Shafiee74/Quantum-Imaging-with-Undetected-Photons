"""Five publication examples for shared four-step reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from verification.phase_reconstruction.metrics import (
    circular_phase_error,
    phase_error_metrics,
)
from verification.phase_reconstruction.reconstruct import (
    PhaseReconstructionResult,
    reconstruct_phase_steps,
)
from verification.phase_reconstruction.synthetic import (
    amplitude_phase_maps,
    constant_phase_maps,
    phase_ramp_maps,
    phase_step_maps,
    synthesize_four_step_frames,
)


@dataclass(frozen=True)
class ExampleSpec:
    example_id: str
    data_dirname: str
    title: str
    unwrap: bool
    unwrap_reason: str
    use_unwrapped_for_error: bool
    noise_sigma: float
    seed: int
    known_phase_rad: float | None
    builder: Callable[[tuple[int, int]], dict[str, np.ndarray]]


def _build_constant(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return constant_phase_maps(shape, phase_rad=0.37, dc_A=2.0, modulation_B=0.8)


def _build_ramp(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return phase_ramp_maps(shape)


def _build_step(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return phase_step_maps(shape)


def _build_amp_phase(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return amplitude_phase_maps(shape)


def _build_noisy(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return constant_phase_maps(shape, phase_rad=-1.1, dc_A=1.5, modulation_B=0.6)


EXAMPLE_SPECS: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        example_id="constant_known_phase",
        data_dirname="ideal",
        title="Constant known phase",
        unwrap=False,
        unwrap_reason="not_requested",
        use_unwrapped_for_error=False,
        noise_sigma=0.0,
        seed=101,
        known_phase_rad=0.37,
        builder=_build_constant,
    ),
    ExampleSpec(
        example_id="phase_ramp",
        data_dirname="phase_ramp",
        title="Phase ramp (>2π; unwrap justified)",
        unwrap=True,
        unwrap_reason="physically_continuous_horizontal_ramp",
        use_unwrapped_for_error=True,
        noise_sigma=0.0,
        seed=102,
        known_phase_rad=None,
        builder=_build_ramp,
    ),
    ExampleSpec(
        example_id="phase_step",
        data_dirname="phase_step",
        title="Phase step (wrapped comparison)",
        unwrap=False,
        unwrap_reason="discontinuous_object_keep_wrapped",
        use_unwrapped_for_error=False,
        noise_sigma=0.0,
        seed=103,
        known_phase_rad=None,
        builder=_build_step,
    ),
    ExampleSpec(
        example_id="amplitude_phase_object",
        data_dirname="amplitude_phase",
        title="Amplitude + phase object",
        unwrap=False,
        unwrap_reason="smooth_but_wrapped_sufficient",
        use_unwrapped_for_error=False,
        noise_sigma=0.0,
        seed=104,
        known_phase_rad=None,
        builder=_build_amp_phase,
    ),
    ExampleSpec(
        example_id="noisy_phase_reconstruction",
        data_dirname="noisy",
        title="Noisy phase reconstruction",
        unwrap=False,
        unwrap_reason="not_requested",
        use_unwrapped_for_error=False,
        noise_sigma=0.05,
        seed=105,
        known_phase_rad=-1.1,
        builder=_build_noisy,
    ),
)


def run_example(
    spec: ExampleSpec,
    *,
    shape: tuple[int, int] = (48, 48),
) -> dict[str, Any]:
    """Synthesize frames, reconstruct with the shared algorithm, score errors."""
    maps = spec.builder(shape)
    rng = np.random.default_rng(int(spec.seed))
    synth = synthesize_four_step_frames(
        maps["dc_A"],
        maps["modulation_B"],
        maps["phase_rad"],
        noise_sigma=float(spec.noise_sigma),
        rng=rng,
    )
    frames = synth["frames"]
    rec: PhaseReconstructionResult = reconstruct_phase_steps(
        {
            "I0": frames[0],
            "I_pi_2": frames[1],
            "I_pi": frames[2],
            "I_3pi_2": frames[3],
        },
        synth["phases_rad"],
        unwrap=spec.unwrap,
        unwrap_reason=spec.unwrap_reason,
    )

    if spec.use_unwrapped_for_error and rec.phase_unwrapped_rad is not None:
        estimate = rec.phase_unwrapped_rad
        truth = synth["truth_phase_unwrapped_rad"]
        piston = float(np.mean(estimate - truth))
        error_map = (estimate - piston) - truth
        err_metrics = {
            "n_pixels": int(error_map.size),
            "rmse_rad": float(np.sqrt(np.mean(error_map * error_map))),
            "mae_rad": float(np.mean(np.abs(error_map))),
            "max_abs_rad": float(np.max(np.abs(error_map))),
            "piston_removed": True,
        }
    else:
        estimate = rec.phase_wrapped_rad
        truth = synth["truth_phase_rad"]
        err_metrics = phase_error_metrics(estimate, truth, remove_piston=False)
        error_map = circular_phase_error(estimate, truth)

    return {
        "spec": spec,
        "shape": shape,
        "frames": frames,
        "phases_rad": synth["phases_rad"],
        "truth_phase_rad": synth["truth_phase_rad"],
        "truth_phase_unwrapped_rad": synth["truth_phase_unwrapped_rad"],
        "truth_dc_A": synth["truth_dc_A"],
        "truth_modulation_B": synth["truth_modulation_B"],
        "reconstruction": rec,
        "error_map_rad": np.asarray(error_map, dtype=np.float64),
        "error_metrics": err_metrics,
        "noise_sigma": float(spec.noise_sigma),
        "seed": int(spec.seed),
    }


def run_all_examples(
    *, shape: tuple[int, int] = (48, 48)
) -> list[dict[str, Any]]:
    return [run_example(spec, shape=shape) for spec in EXAMPLE_SPECS]
