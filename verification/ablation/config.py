"""Build merged ablation configurations from the mechanism catalog."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from verification.ablation.definitions import (
    CLEAN_SETUP_BASE,
    MECHANISMS,
)


def merge_mechanisms(names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Merge one or more mechanism specs (later entries override scalars)."""
    detector_noise = False
    setup = dict(CLEAN_SETUP_BASE)
    latents: dict[str, float] = {
        "s_pump": 0.0,
        "s_align": 0.0,
        "s_thermal": 0.0,
        "s_det": 0.0,
        "s_cool": 0.0,
        "s_focus": 0.0,
    }
    em_gain = 1.0
    phase_offset = 0.0
    sources: list[str] = []

    for name in names:
        if name not in MECHANISMS:
            raise KeyError(f"unknown mechanism {name!r}")
        m = MECHANISMS[name]
        sources.append(str(m["source"]))
        if m["controls"].get("detector_noise"):
            detector_noise = True
        for k, v in (m.get("setup") or {}).items():
            # For isolation merges: take max of noise strengths where both set
            if k in setup and isinstance(v, (int, float)) and isinstance(setup[k], (int, float)):
                if k in ("dark0", "cic0", "read_noise0_e", "prnu_sigma", "hot_frac_at_s1"):
                    setup[k] = max(float(setup[k]), float(v))
                else:
                    setup[k] = v
            else:
                setup[k] = v
        for k, v in (m.get("latents") or {}).items():
            latents[k] = max(float(latents.get(k, 0.0)), float(v))
        em_gain = max(float(em_gain), float(m.get("em_gain", 1.0)))
        phase_offset += float(m.get("object_phase_offset_rad", 0.0))

    return {
        "mechanism_names": list(names),
        "detector_noise": bool(detector_noise),
        "setup": setup,
        "latents": latents,
        "em_gain": float(em_gain),
        "object_phase_offset_rad": float(phase_offset),
        "sources": sources,
    }


def clean_config() -> dict[str, Any]:
    return merge_mechanisms(())
