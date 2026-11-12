"""Production rabbit object mask helper used by C_lab figure generators.

This module previously hosted a longer L/M_s/M_i sweep workflow. The release
cleanup kept only the shared ``object_mask_original`` helper that downstream
Figure 5/6 tools import.
"""
from __future__ import annotations

import numpy as np

from object_maps.c_lab_production_object_map import c_lab_rabbit_transmission_grid
from setup_params import resolve_setup_selection

N = 1024


def object_mask_original() -> np.ndarray:
    """Return the production C_lab rabbit transmission grid on the 1024 grid."""
    setup = resolve_setup_selection("C_lab")
    mask = np.asarray(c_lab_rabbit_transmission_grid(setup), dtype=np.float64)
    if mask.shape != (N, N):
        raise RuntimeError(f"unexpected rabbit mask shape {mask.shape}, expected {(N, N)}")
    return mask
