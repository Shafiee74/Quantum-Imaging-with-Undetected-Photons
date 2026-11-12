"""Opt-in C_lab Rabbit70 synthetic object map (SPDC-designed; not Y1 caliper)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from setup_params import SetupParams, resolve_setup_selection

DEFAULT_MASK = (
    Path(__file__).resolve().parents[1]
    / "C_lab_JointSPDC_Rabbit70_v1_27Oct"
    / "Rabbit_Physical_Mask.npz"
)


def load_rabbit70_transmission_grid(
    setup: SetupParams | str | None = None,
    *,
    grid_n: int | None = None,
) -> np.ndarray:
    """Return frozen Rabbit70 T(x,y) resampled to the requested object grid.

    Physical width/height/center are defined in the NPZ / JSON freeze and are
    independent of Mi, camera pitch, and cropping. Resampling only changes
    rasterization of the same physical field.
    """
    if not isinstance(setup, SetupParams):
        setup = resolve_setup_selection(setup or "C_lab_rabbit70")
    rel = getattr(setup, "rabbit70_mask_npz", None)
    path = Path(rel) if rel else DEFAULT_MASK
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Rabbit70 mask not found at {path}. Run "
            "C_lab_JointSPDC_Rabbit70_v1_27Oct/build_and_verify.py first."
        )
    data = np.load(path)
    T = np.asarray(data["T"], dtype=np.float64)
    n = int(grid_n or T.shape[0])
    if n != T.shape[0]:
        T = zoom(T, n / T.shape[0], order=0)
    return np.clip(T, 0.0, 1.0)


def rabbit70_metadata(setup: SetupParams | str | None = None) -> dict[str, Any]:
    if not isinstance(setup, SetupParams):
        setup = resolve_setup_selection(setup or "C_lab_rabbit70")
    root = Path(__file__).resolve().parents[1] / "C_lab_JointSPDC_Rabbit70_v1_27Oct"
    import json

    return json.loads((root / "Rabbit_Physical_Definition.json").read_text(encoding="utf-8"))
