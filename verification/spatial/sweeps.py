"""Parameter sweeps for FoV / spatial-mode dependence (Task 09)."""
from __future__ import annotations

from typing import Any, Iterable

from setup_params import resolve_setup_selection
from verification.spatial.analysis import evaluate_profile

# Allowed ranges constrained by production validators.
PUMP_WAIST_SWEEP_UM: tuple[float, ...] = (150.0, 200.0, 250.0, 300.0, 350.0)
CRYSTAL_LENGTH_SWEEP_MM: tuple[float, ...] = (2.0, 4.0, 6.0, 10.0, 15.0)
NEAR_FIELD_MS_SWEEP: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
DETECTOR_PIXEL_SWEEP_UM: tuple[float, ...] = (8.0, 13.0, 16.0, 20.0)


def _safe_override(setup: Any, **kwargs: Any) -> Any | None:
    try:
        return setup.overridden(**kwargs)
    except Exception as exc:  # noqa: BLE001 — record and skip illegal points
        return None


def sweep_pump_waist(profile: str) -> list[dict[str, Any]]:
    base = resolve_setup_selection(profile)
    rows: list[dict[str, Any]] = []
    for w in PUMP_WAIST_SWEEP_UM:
        setup = _safe_override(
            base, pump_waist_um=float(w), pump_waist_NL1_um=float(w)
        )
        if setup is None:
            rows.append(
                {
                    "profile": profile,
                    "sweep": "pump_waist",
                    "pump_waist_um": float(w),
                    "status": "skipped_invalid",
                }
            )
            continue
        ev = evaluate_profile(profile, setup=setup)
        rows.append(
            {
                "profile": profile,
                "sweep": "pump_waist",
                "status": "ok",
                "pump_waist_um": float(w),
                "D_FoV_um": ev["fov"]["D_FoV_um"],
                "sigma_object_um": ev["correlation_sigma_object_um"],
                "N_modes": ev["modes"]["N_modes"],
                "N_modes_1d": ev["modes"]["N_modes_1d"],
                "signed_magnification": ev["signed_magnification"],
            }
        )
    return rows


def sweep_crystal_length(profile: str) -> list[dict[str, Any]]:
    """Near-field σ depends on L; far-field σ does not (still record both)."""
    base = resolve_setup_selection(profile)
    rows: list[dict[str, Any]] = []
    for L in CRYSTAL_LENGTH_SWEEP_MM:
        setup = _safe_override(base, crystal_length_mm=float(L))
        if setup is None:
            rows.append(
                {
                    "profile": profile,
                    "sweep": "crystal_length",
                    "crystal_length_mm": float(L),
                    "status": "skipped_invalid",
                }
            )
            continue
        ev = evaluate_profile(profile, setup=setup)
        rows.append(
            {
                "profile": profile,
                "sweep": "crystal_length",
                "status": "ok",
                "crystal_length_mm": float(L),
                "imaging_mode": ev["imaging_mode"],
                "D_FoV_um": ev["fov"]["D_FoV_um"],
                "sigma_object_um": ev["correlation_sigma_object_um"],
                "N_modes": ev["modes"]["N_modes"],
                "N_modes_1d": ev["modes"]["N_modes_1d"],
            }
        )
    return rows


def sweep_magnification(profile: str) -> list[dict[str, Any]]:
    """Vary near-field Ms where applicable; far-field records baseline only."""
    base = resolve_setup_selection(profile)
    rows: list[dict[str, Any]] = []
    mode = str(getattr(base, "imaging_mode", ""))
    if "near" in mode:
        values: Iterable[float] = NEAR_FIELD_MS_SWEEP
        for Ms in values:
            setup = _safe_override(base, near_field_signal_magnification=float(Ms))
            if setup is None:
                rows.append(
                    {
                        "profile": profile,
                        "sweep": "magnification",
                        "Ms": float(Ms),
                        "status": "skipped_invalid",
                    }
                )
                continue
            ev = evaluate_profile(profile, setup=setup)
            rows.append(
                {
                    "profile": profile,
                    "sweep": "magnification",
                    "status": "ok",
                    "Ms": float(Ms),
                    "signed_magnification": ev["signed_magnification"],
                    "sigma_object_um": ev["correlation_sigma_object_um"],
                    "sigma_camera_um": ev["correlation_sigma_camera_um"],
                    "D_FoV_object_um": ev["fov"]["D_FoV_um"],
                    "D_FoV_camera_um": ev["modes"].get("D_FoV_camera_um"),
                    "N_modes": ev["modes"]["N_modes"],
                    "samples_per_sigma_camera": ev["modes"].get(
                        "samples_per_sigma_camera"
                    ),
                }
            )
    else:
        # Far-field: vary far-field signal relay magnification if present.
        for Ms in (0.5, 1.0, 1.5, 2.0):
            kwargs = {"far_field_signal_relay_magnification": float(Ms)}
            setup = _safe_override(base, **kwargs)
            if setup is None:
                rows.append(
                    {
                        "profile": profile,
                        "sweep": "magnification",
                        "far_field_Ms": float(Ms),
                        "status": "skipped_invalid",
                    }
                )
                continue
            ev = evaluate_profile(profile, setup=setup)
            rows.append(
                {
                    "profile": profile,
                    "sweep": "magnification",
                    "status": "ok",
                    "far_field_Ms": float(Ms),
                    "signed_magnification": ev["signed_magnification"],
                    "sigma_object_um": ev["correlation_sigma_object_um"],
                    "sigma_camera_um": ev["correlation_sigma_camera_um"],
                    "D_FoV_object_um": ev["fov"]["D_FoV_um"],
                    "D_FoV_camera_um": ev["modes"].get("D_FoV_camera_um"),
                    "N_modes": ev["modes"]["N_modes"],
                    "samples_per_sigma_camera": ev["modes"].get(
                        "samples_per_sigma_camera"
                    ),
                }
            )
    return rows


def sweep_detector_sampling(profile: str) -> list[dict[str, Any]]:
    base = resolve_setup_selection(profile)
    rows: list[dict[str, Any]] = []
    for pix in DETECTOR_PIXEL_SWEEP_UM:
        setup = _safe_override(base, pixel_um=float(pix))
        if setup is None:
            rows.append(
                {
                    "profile": profile,
                    "sweep": "detector_sampling",
                    "pixel_um": float(pix),
                    "status": "skipped_invalid",
                }
            )
            continue
        ev = evaluate_profile(profile, setup=setup)
        rows.append(
            {
                "profile": profile,
                "sweep": "detector_sampling",
                "status": "ok",
                "pixel_um": float(pix),
                "D_FoV_um": ev["fov"]["D_FoV_um"],
                "sigma_camera_um": ev["correlation_sigma_camera_um"],
                "sigma_camera_px": ev["modes"].get("sigma_camera_px"),
                "samples_per_sigma_camera": ev["modes"].get("samples_per_sigma_camera"),
                "N_modes": ev["modes"]["N_modes"],
                "array_sampling_window_um": ev["modes"].get("array_sampling_window_um"),
                "note": "FoV unchanged by detector pitch; sampling diagnostics change",
            }
        )
    return rows


def sweep_correlation_width_via_waist_and_length(
    profile: str,
) -> list[dict[str, Any]]:
    """Cross-check: σ from mapping while FoV from waist (FF) or L (NF)."""
    rows: list[dict[str, Any]] = []
    rows.extend(sweep_pump_waist(profile))
    rows.extend(sweep_crystal_length(profile))
    return rows


def run_all_sweeps(profiles: tuple[str, ...] = ("A", "B", "C", "D")) -> dict[str, Any]:
    out: dict[str, Any] = {"pump_waist": [], "crystal_length": [], "magnification": [], "detector_sampling": []}
    for p in profiles:
        out["pump_waist"].extend(sweep_pump_waist(p))
        out["crystal_length"].extend(sweep_crystal_length(p))
        out["magnification"].extend(sweep_magnification(p))
        out["detector_sampling"].extend(sweep_detector_sampling(p))
    return out
