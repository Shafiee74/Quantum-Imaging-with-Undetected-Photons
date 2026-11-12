"""BenchmarkTarget container — physical ground truth for all configs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.targets.geometry import geometry_metadata


@dataclass
class BenchmarkTarget:
    """Deterministic synthetic object with physical metadata.

    ``T`` is intensity transmission in [0, 1].
    ``amplitude`` is ``√T`` (complex-object modulus).
    ``phi_rad`` is the object phase map [rad].
    ``O_complex = amplitude * exp(1j * phi_rad)``.
    """

    target_id: str
    T: np.ndarray
    phi_rad: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    X_um: np.ndarray
    Y_um: np.ndarray
    pixel_um: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T, dtype=np.float64)
        self.phi_rad = np.asarray(self.phi_rad, dtype=np.float64)
        if self.T.shape != self.phi_rad.shape:
            raise ValueError(
                f"T shape {self.T.shape} != phi shape {self.phi_rad.shape}"
            )
        if self.T.ndim != 2:
            raise ValueError("T/phi must be 2-D")
        if not np.all(np.isfinite(self.T)) or not np.all(np.isfinite(self.phi_rad)):
            raise ValueError("T/phi must be finite")
        if np.any(self.T < 0.0) or np.any(self.T > 1.0 + 1.0e-12):
            raise ValueError("T must lie in [0, 1]")
        self.T = np.clip(self.T, 0.0, 1.0)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.T.shape[0]), int(self.T.shape[1]))

    @property
    def amplitude(self) -> np.ndarray:
        return np.sqrt(self.T)

    @property
    def O_complex(self) -> np.ndarray:
        return (self.amplitude * np.exp(1j * self.phi_rad)).astype(np.complex128)

    @property
    def M(self) -> np.ndarray:
        """Mask / SLM drive map — identical to intensity transmission here."""
        return self.T.copy()

    @property
    def width_um(self) -> float:
        return float(self.shape[1]) * float(self.pixel_um)

    @property
    def height_um(self) -> float:
        return float(self.shape[0]) * float(self.pixel_um)

    def full_metadata(self) -> dict[str, Any]:
        geo = geometry_metadata(self.shape, float(self.pixel_um))
        out = {
            "target_id": self.target_id,
            "geometry": geo,
            "T_min": float(np.min(self.T)),
            "T_max": float(np.max(self.T)),
            "phi_min_rad": float(np.min(self.phi_rad)),
            "phi_max_rad": float(np.max(self.phi_rad)),
            "amplitude_min": float(np.min(self.amplitude)),
            "amplitude_max": float(np.max(self.amplitude)),
        }
        out.update(dict(self.metadata))
        return out

    def to_simulation_reference(self) -> dict[str, Any]:
        """Pack into the production ``simulate_forward`` reference schema.

        Same physical ground truth for every configuration; only the sampling
        grid (chosen when the target was built) is embedded.
        """
        h, w = self.shape
        zeros = np.zeros((h, w), dtype=np.float32)
        return {
            "M": self.M.astype(np.float32),
            "T": self.T.astype(np.float32),
            "phi": self.phi_rad.astype(np.float32),
            "O_i": self.O_complex.astype(np.complex64),
            "D": zeros,
            "target_size": int(h) if h == w else (h, w),
            "benchmark_target_id": self.target_id,
            "benchmark_pixel_um": float(self.pixel_um),
            "benchmark_metadata": self.full_metadata(),
        }

    def array_bundle(self) -> dict[str, np.ndarray]:
        return {
            "transmission_T": self.T,
            "amplitude": self.amplitude,
            "phase_rad": self.phi_rad,
            "complex_object": self.O_complex,
            "M": self.M,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "X_um": self.X_um,
            "Y_um": self.Y_um,
        }
