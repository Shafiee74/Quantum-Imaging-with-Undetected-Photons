"""JSON bridge for the QIUP UI (Django HTTP API and optional CLI).

Service helpers (used by Django)::

    defaults() -> dict
    preview_event(config) -> dict
    iter_run_events(config) -> Iterator[dict]

CLI (debug / legacy)::

    python qiup_ui_backend.py defaults
    python qiup_ui_backend.py run < cfg.json
    python qiup_ui_backend.py preview < obj.json

Every CLI stdout line is a complete JSON object. Diagnostics go to stderr.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import generate_mnist5_four_arrangement_images as gen  # noqa: E402
from latents import LatentState  # noqa: E402
from provenance import (  # noqa: E402
    CALIBRATION_STATUS,
    PACKAGE_VERSION,
    RELEASE_CLASSIFICATION,
)
from setup_params import (  # noqa: E402
    crystal_thermal_state,
    is_two_crystal_architecture,
    object_mask_intensity,
    resolve_setup_selection,
)
from simulate import simulate_forward  # noqa: E402

PROFILE_ORDER = ("A", "B", "C", "D")
UI_RUNS = ROOT / "outputs" / "ui_runs"
SIZE_RANGE = (32, 128)
RABBIT_MASK_PNG = ROOT / "qiup-ui" / "public" / "test-masks" / "rabbit-slm-mask.png"
# Accept nearly-grayscale RGB (common PNG export); reject true color images.
_GRAYSCALE_CHANNEL_TOL = 8.0


def _intensity_from_pil(image) -> np.ndarray:
    """Convert a PIL image to float intensity in [0, 1], requiring a mask (not RGB color)."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("expected a PIL Image")
    mode = image.mode
    arr = np.asarray(image)
    if mode in ("1", "L", "I", "F"):
        data = arr.astype(np.float64)
        if mode == "1":
            return data
        if data.max() > 1.0:
            data = data / (255.0 if data.max() <= 255.0 else float(data.max()))
        return np.clip(data, 0.0, 1.0)
    if mode in ("LA", "PA"):
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        return np.clip(gray, 0.0, 1.0)
    if mode in ("RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
        # True color masks (distinct R/G/B) are rejected; grayscale-looking RGB is OK.
        spread = float(np.max(np.abs(rgb[..., 0] - rgb[..., 1])) + np.max(np.abs(rgb[..., 1] - rgb[..., 2])))
        if spread > _GRAYSCALE_CHANNEL_TOL:
            raise ValueError(
                "uploaded image looks like an RGB color photo; "
                "provide a single-channel amplitude mask (grayscale / binary PNG)"
            )
        gray = rgb.mean(axis=2) / 255.0
        return np.clip(gray, 0.0, 1.0)
    # Fallback: convert to L and warn via ValueError if conversion fails
    try:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unsupported image mode {mode!r} for SLM mask") from exc
    return np.clip(gray, 0.0, 1.0)


def _resize_mask_to_grid(mask: np.ndarray, size: int) -> np.ndarray:
    """Fit mask into a square grid, preserving aspect ratio and zero-padding."""
    from PIL import Image

    data = np.asarray(mask, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("SLM mask must be a 2-D array")
    if not np.isfinite(data).all():
        raise ValueError("SLM mask contains non-finite values")
    data = np.clip(data, 0.0, 1.0)
    h, w = data.shape
    if h <= 0 or w <= 0:
        raise ValueError("SLM mask is empty")
    scale = min(size / float(h), size / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    # Keep new size inside the grid even if rounding pushes over.
    new_h = min(new_h, size)
    new_w = min(new_w, size)
    pil = Image.fromarray((data * 255.0).astype(np.uint8), mode="L")
    resized = np.asarray(pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    out = np.zeros((size, size), dtype=np.float32)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized.astype(np.float32)
    return out


def _mask_reference(T: np.ndarray, size: int, label: str) -> dict[str, Any]:
    T = np.asarray(T, dtype=np.float32)
    if T.shape != (size, size):
        raise ValueError(f"mask shape {T.shape} does not match grid {(size, size)}")
    return {
        "M": T.copy(),
        "T": T,
        "phi": np.zeros_like(T),
        "O_i": np.sqrt(np.clip(T, 0.0, 1.0)).astype(np.float32),
        "D": np.zeros_like(T),
        "target_size": size,
        "label": label,
    }


def _load_png_mask_bytes(raw: bytes, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        intensity = _intensity_from_pil(image)
    return _resize_mask_to_grid(intensity, size)


def _load_png_mask_path(path: Path, size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"rabbit mask not found: {path}")
    return _load_png_mask_bytes(path.read_bytes(), size)


def _mnist_train_count() -> int:
    """Local shipped MNIST subset size (not the full 60 000-image corpus)."""
    return int(gen.mnist_train_count())


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, default=_json_default) + "\n")
    sys.stdout.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _png_base64(
    image: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    cmap: str = "inferno",
    signed: bool = False,
    min_side: int = 256,
) -> str:
    """Render a float image to PNG. Constant fields stay visible (not black)."""
    from matplotlib import colormaps
    from PIL import Image

    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"expected 2-D image, got shape {list(np.asarray(image).shape)}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if signed:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if not np.isfinite(peak) or peak <= 1e-15:
            peak = 1.0
        lo = float(vmin) if vmin is not None else -peak
        hi = float(vmax) if vmax is not None else peak
        if hi <= lo:
            hi = lo + 1.0
        cmap_name = cmap if cmap != "inferno" else "coolwarm"
    else:
        lo = float(np.min(data)) if vmin is None else float(vmin)
        hi = float(np.max(data)) if vmax is None else float(vmax)
        cmap_name = cmap

    span = hi - lo
    if not np.isfinite(span) or span <= 1e-15:
        # Constant array: mid-colormap tone so the panel is not a black square.
        norm = np.full(data.shape, 0.62, dtype=np.float64)
    else:
        norm = np.clip((data - lo) / span, 0.0, 1.0)

    rgba = (colormaps[cmap_name](norm) * 255.0).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    h, w = int(data.shape[0]), int(data.shape[1])
    scale = max(1, int(np.ceil(float(min_side) / float(max(h, w, 1)))))
    if scale > 1:
        img = img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_payload(
    image: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    cmap: str = "inferno",
    signed: bool = False,
    min_side: int = 256,
) -> dict[str, Any]:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        data = np.squeeze(data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(data)) if data.size else 0.0
    hi = float(np.max(data)) if data.size else 0.0
    return {
        "png": _png_base64(data, vmin, vmax, cmap=cmap, signed=signed, min_side=min_side),
        "min": lo,
        "max": hi,
        "mean": float(np.mean(data)) if data.size else 0.0,
        "std": float(np.std(data)) if data.size else 0.0,
        "shape": list(data.shape),
        "constant": bool(hi - lo <= 1e-15),
        "signed": bool(signed),
    }


def _line_profiles(image: np.ndarray) -> dict[str, Any]:
    data = np.asarray(image, dtype=np.float64)
    h, w = data.shape
    row = data[h // 2, :].tolist()
    col = data[:, w // 2].tolist()
    return {
        "row_y": int(h // 2),
        "col_x": int(w // 2),
        "row": [float(v) for v in row],
        "col": [float(v) for v in col],
    }


def _histogram(image: np.ndarray, bins: int = 40) -> dict[str, Any]:
    data = np.asarray(image, dtype=np.float64).ravel()
    counts, edges = np.histogram(data, bins=int(bins))
    return {
        "bins": int(bins),
        "counts": [int(v) for v in counts.tolist()],
        "edges": [float(v) for v in edges.tolist()],
    }


def _centroid_yx(image: np.ndarray) -> list[float]:
    data = np.asarray(image, dtype=np.float64)
    data = np.clip(data - float(np.min(data)), 0.0, None)
    total = float(np.sum(data))
    if total <= 0.0:
        h, w = data.shape
        return [0.5 * (h - 1), 0.5 * (w - 1)]
    yy, xx = np.indices(data.shape)
    return [float(np.sum(yy * data) / total), float(np.sum(xx * data) / total)]


def _snr_mean_over_std(image: np.ndarray) -> float | None:
    data = np.asarray(image, dtype=np.float64)
    std = float(np.std(data))
    if std <= 1e-12:
        return None
    return float(np.mean(data) / std)


def _residual_rms(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def _compact_visibility(meta: dict[str, Any]) -> dict[str, Any]:
    vis = dict(meta.get("visibility_observables") or {})
    keys = (
        "V_50_50",
        "N_1",
        "N_2",
        "C_12_abs",
        "nbar_1",
        "nbar_2",
        "gain_regime_1",
        "gain_regime_2",
        "visibility_equation",
    )
    return {k: vis.get(k) for k in keys if k in vis}


def _diagnostics_payload(
    *,
    result: Any,
    image: np.ndarray,
    mode: str,
    slm: np.ndarray | None = None,
    ideal_ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compact science diagnostics for the web UI (no new physics)."""
    meta = dict(getattr(result, "meta", {}) or {})
    image = np.asarray(image, dtype=np.float64)
    vis = _compact_visibility(meta)
    op = meta.get("spdc_operating_point") or meta.get("visibility_observables") or {}
    out: dict[str, Any] = {
        "visibility": vis,
        "snr": _snr_mean_over_std(image),
        "centroid_yx": _centroid_yx(image),
        "profiles": _line_profiles(image),
        "histogram": _histogram(image),
        "transfer": meta.get("effective_object_transfer"),
        "phase_reconstruction": meta.get("phase_reconstruction"),
        "selected_output_port": getattr(
            result, "selected_output_port", meta.get("selected_output_port")
        ),
        "operating_point": {
            "nbar_1": op.get("nbar_1", op.get("mean_photons_per_mode_NL1")),
            "nbar_2": op.get("nbar_2", op.get("mean_photons_per_mode_NL2")),
            "gain_r_1": op.get("gain_r_1", op.get("spdc_gain_r_NL1")),
            "gain_r_2": op.get("gain_r_2", op.get("spdc_gain_r_NL2")),
            "gain_regime_1": op.get("gain_regime_1", op.get("regime_NL1")),
            "gain_regime_2": op.get("gain_regime_2", op.get("regime_NL2")),
            "V_50_50": vis.get("V_50_50"),
            "detector_mode": meta.get("detector_mode"),
            "selected_output_port": getattr(
                result, "selected_output_port", meta.get("selected_output_port")
            ),
        },
    }
    if slm is not None:
        out["slm_profiles"] = _line_profiles(slm)
    # Interferometric difference port (optical clean_D) — signed teaching map.
    clean_d = getattr(result, "clean_D", None)
    if clean_d is not None:
        out["difference_port"] = _image_payload(
            np.asarray(clean_d, dtype=np.float64), signed=True, cmap="coolwarm"
        )
    # Mid-frame ± ports (same twin stacks as SimResult).
    try:
        fp = np.asarray(getattr(result, "frames_plus", None), dtype=np.float64)
        fm = np.asarray(getattr(result, "frames_minus", None), dtype=np.float64)
        if fp.ndim == 3 and fp.shape[0] > 0:
            mid = int(fp.shape[0] // 2)
            out["plus_port"] = _image_payload(fp[mid], cmap="gray")
            out["minus_port"] = _image_payload(fm[mid], cmap="gray")
    except Exception:  # noqa: BLE001
        pass
    if ideal_ref is not None:
        ideal_ref = np.asarray(ideal_ref, dtype=np.float64)
        residual = image - ideal_ref
        out["residual_to_ideal"] = _image_payload(residual, signed=True, cmap="coolwarm")
        out["residual_rms"] = _residual_rms(image, ideal_ref)
    # Spatial teaching maps already computed by the twin forward path.
    v_map = getattr(result, "visibility_map", None)
    if v_map is not None:
        out["visibility_map"] = _image_payload(
            np.asarray(v_map, dtype=np.float64), vmin=0.0, vmax=1.0, cmap="viridis"
        )
    t_eff = getattr(result, "T_eff_map", None)
    if t_eff is not None:
        out["T_eff_map"] = _image_payload(
            np.asarray(t_eff, dtype=np.float64), vmin=0.0, vmax=1.0, cmap="magma"
        )
    phi_eff = getattr(result, "phi_eff_map", None)
    if phi_eff is not None:
        out["phi_eff_map"] = _image_payload(
            np.asarray(phi_eff, dtype=np.float64), signed=True, cmap="coolwarm"
        )
    phase = getattr(result, "reconstructed_phase_rad", None)
    if phase is not None:
        out["phase_map"] = _image_payload(
            np.asarray(phase, dtype=np.float64), signed=True, cmap="twilight"
        )
    modulation = getattr(result, "phase_modulation", None)
    if modulation is not None:
        out["phase_modulation_map"] = _image_payload(
            np.asarray(modulation, dtype=np.float64), cmap="viridis"
        )
    return out


SWEEP_PARAMS: dict[str, dict[str, Any]] = {
    # Optics/drift: run on a clean optical base. Lab profiles lock delta_z_um and
    # use_physical_thermal, so s_focus/s_thermal need explicit setup mapping.
    "s_focus": {
        "target": "latents",
        "lo": 0.0,
        "hi": 1.0,
        "label": "Focus drift s_focus",
        "family": "optics",
    },
    "s_align": {
        "target": "latents",
        "lo": 0.0,
        "hi": 1.0,
        "label": "Alignment s_align",
        "family": "optics",
    },
    "s_thermal": {
        "target": "latents",
        "lo": 0.0,
        "hi": 0.99,
        "label": "Thermal drift s_thermal",
        "family": "optics",
    },
    "em_gain": {
        "target": "latents",
        "lo": 1.0,
        "hi": 200.0,
        "label": "EM gain",
        "family": "detector",
    },
    "mean_photons_per_mode": {
        "target": "latents",
        "lo": 0.1,
        "hi": 10.0,
        "label": "Mean photons / mode",
        "family": "detector",
    },
    "read_noise0_e": {
        "target": "setup",
        "lo": 0.0,
        "hi": 40.0,
        "label": "Read noise (e⁻)",
        "family": "detector",
    },
    "sigma_blur_extra_px": {
        "target": "setup",
        "lo": 0.0,
        "hi": 3.0,
        "label": "Extra blur σ (px)",
        "family": "optics",
    },
}


_OPTICS_LATENT_KEYS = (
    "s_pump",
    "s_gain",
    "s_align",
    "s_thermal",
    "s_det",
    "s_cool",
    "s_focus",
    "s_slm",
    "delta_phi_rms",
)


def _prepare_sweep_setup(
    *,
    setup,
    param: str,
    value: float,
):
    """Map optics sweep knobs onto fields the physical twin actually uses."""
    if param == "s_focus":
        # Profiles A–D use use_physical_thermal, which ignores latent s_focus unless
        # mechanical Δz is set (Δz = Δf(T) + Δz_mech).
        zmax = float(getattr(setup, "delta_z_max_um", 100.0) or 100.0)
        return replace(setup, delta_z_um=float(value) * zmax)
    if param == "s_thermal":
        # Keep T_pm inside the allowed crystal window while scaling detuning.
        t_pm0 = float(getattr(setup, "T_pm0_NL1_C", 83.7) or 83.7)
        tmax = max(0.0, 90.0 - t_pm0 - 0.1)
        span = min(float(getattr(setup, "T_ref_span_C", 8.0) or 8.0), tmax)
        phase_at_s1 = max(float(getattr(setup, "phase_drift_rad_per_frame_at_s1", 0.0) or 0.0), 0.15)
        vis_at_s1 = max(float(getattr(setup, "vis_walk_per_frame_at_s1", 0.0) or 0.0), 0.02)
        return replace(
            setup,
            delta_T_NL1_C=float(value) * span,
            delta_T_NL2_C=float(value) * span * 0.5,
            phase_drift_rad_per_frame_at_s1=phase_at_s1,
            vis_walk_per_frame_at_s1=vis_at_s1,
        )
    if param == "s_align":
        span = max(float(getattr(setup, "delta_phi_align_span_rad", 0.0) or 0.0), 0.25)
        return replace(setup, delta_phi_align_span_rad=span)
    return setup


def _architecture_payload(setup) -> dict[str, Any]:
    return {
        "architecture": str(setup.spdc_architecture),
        "imaging_mode": str(setup.imaging_mode),
        "object_coefficient": "T" if is_two_crystal_architecture(setup) else "R",
        "object_plane": str(setup.object_plane),
    }


def _apply_phase_offset(ref: dict[str, Any], offset_rad: float) -> dict[str, Any]:
    if abs(float(offset_rad)) < 1e-15:
        return ref
    out = dict(ref)
    if "phi" in out and out["phi"] is not None:
        out["phi"] = np.asarray(out["phi"], dtype=np.float64) + float(offset_rad)
    return out


# Fixed stage ladder for the Noise Stage Explorer (ablation-style, existing mechanisms).
STAGE_LADDER: tuple[dict[str, Any], ...] = (
    {
        "id": "ideal",
        "label": "Ideal optics",
        "family": "reference",
        "detector_noise": False,
        "setup": {},
        "latents": {},
        "explain": "Optical image only — detector nuisances off.",
    },
    {
        "id": "optics",
        "label": "Optics blur / align",
        "family": "optics",
        "detector_noise": False,
        "setup": {
            "sigma_blur_extra_px": 0.6,
            "sigma_diffraction_px": 0.35,
            "sigma_aberration_px": 0.25,
            "delta_phi_align_span_rad": 0.12,
        },
        "latents": {"s_focus": 0.45, "s_align": 0.25},
        "explain": "PSF / focus / alignment soften structure (little grain).",
    },
    {
        "id": "drift",
        "label": "Pump / thermal drift",
        "family": "drift",
        "detector_noise": False,
        "setup": {},
        "latents": {
            "s_pump": 0.3,
            "s_thermal": 0.35,
            "s_align": 0.2,
            "delta_phi_rms": 0.08,
        },
        "explain": "Drift latents walk intensity / phase without EMCCD grain.",
    },
    {
        "id": "shot_read",
        "label": "Shot + read noise",
        "family": "detector",
        "detector_noise": True,
        "setup": {
            "dark0": 0.0,
            "cic0": 0.0,
            "prnu_sigma": 0.0,
            "hot_frac_at_s1": 0.0,
            "accidental_e_at_s1": 0.0,
            "read_noise0_e": 6.0,
            "read_noise_span_e": 10.0,
        },
        "latents": {"s_det": 0.35, "em_gain": 20.0, "mean_photons_per_mode": 2.0},
        "explain": "Photon shot + post-EM read noise (statistical grain).",
    },
    {
        "id": "detector_full",
        "label": "Detector pedestal + PRNU",
        "family": "detector",
        "detector_noise": True,
        "setup": {
            "dark0": 0.0015,
            "cic0": 0.003,
            "prnu_sigma": 0.02,
            "hot_frac_at_s1": 0.0008,
            "read_noise0_e": 5.0,
            "read_noise_span_e": 12.0,
        },
        "latents": {
            "s_det": 0.5,
            "s_cool": 0.4,
            "em_gain": 40.0,
            "mean_photons_per_mode": 2.0,
        },
        "explain": "Dark/CIC pedestal + fixed-pattern / hot pixels + EM.",
    },
    {
        "id": "full_noisy",
        "label": "Full noisy preset",
        "family": "full",
        "detector_noise": True,
        "setup": "noisy_setup",
        "latents": "noisy_latents",
        "explain": "Combined UI noisy setup (optics + drift + detector).",
    },
)


def _crystals(setup) -> list[str]:
    return ["NL1", "NL2"] if is_two_crystal_architecture(setup) else ["NL"]


def _thermal_defaults(setup) -> dict[str, float]:
    if is_two_crystal_architecture(setup):
        return {
            "T_pm0_NL1_C": float(setup.T_pm0_NL1_C),
            "T_pm0_NL2_C": float(setup.T_pm0_NL2_C),
            "dT_NL1_C": float(getattr(setup, "dT_NL1_C", 0.0) or 0.0),
            "dT_NL2_C": float(getattr(setup, "dT_NL2_C", 0.0) or 0.0),
        }
    return {
        "T_pm0_NL_C": float(setup.T_pm0_NL1_C),
        "dT_NL_C": float(getattr(setup, "dT_NL1_C", 0.0) or 0.0),
    }


def _thermal_overrides(setup, thermal: dict[str, Any] | None) -> dict[str, float]:
    if not thermal:
        return {}
    if is_two_crystal_architecture(setup):
        out = {}
        for key in ("T_pm0_NL1_C", "T_pm0_NL2_C", "dT_NL1_C", "dT_NL2_C"):
            if thermal.get(key) is not None:
                out[key] = float(thermal[key])
        return out
    out = {}
    if thermal.get("T_pm0_NL_C") is not None:
        out["T_pm0_C"] = float(thermal["T_pm0_NL_C"])
    if thermal.get("dT_NL_C") is not None:
        out["dT_C"] = float(thermal["dT_NL_C"])
    return out


def _thermal_report(setup) -> list[dict[str, Any]]:
    state = crystal_thermal_state(setup)
    rows = []
    for index, label in enumerate(_crystals(setup), start=1):
        rows.append({
            "crystal": label,
            "T_pm0_C": state.get(f"T_pm0_NL{index}_C"),
            "T_pm_C": state.get(f"T_pm_NL{index}_C"),
            "T_crystal_C": state.get(f"T_crystal_NL{index}_C"),
            "dT_C": state.get(f"dT_NL{index}_C"),
        })
    return rows


def defaults() -> dict[str, Any]:
    labels = {selector: (slug, label) for selector, slug, label in gen.PROFILES}
    profiles = {}
    for selector in PROFILE_ORDER:
        setup = resolve_setup_selection(selector)
        two = is_two_crystal_architecture(setup)
        profiles[selector] = {
            "id": selector,
            "slug": labels[selector][0],
            "label": labels[selector][1],
            "architecture": str(setup.spdc_architecture),
            "imaging_mode": str(setup.imaging_mode),
            "crystals": _crystals(setup),
            "object_coefficient": "T" if two else "R",
            "object_type": str(setup.object_type),
            "object_plane": str(setup.object_plane),
            "thermal": _thermal_defaults(setup),
            "lambda_nm": {
                "pump": float(setup.lambda_p_nm),
                "signal": float(setup.lambda_s_nm),
                "idler": float(setup.lambda_i_nm),
            },
        }
    return {
        "package_version": PACKAGE_VERSION,
        "release_classification": RELEASE_CLASSIFICATION,
        "calibration_status": CALIBRATION_STATUS,
        "profiles": profiles,
        "ideal_setup": copy.deepcopy(gen.IDEAL_SETUP),
        "ideal_latents": copy.deepcopy(gen.IDEAL_LATENTS),
        "noisy_setup": copy.deepcopy(gen.NOISY_SETUP),
        "noisy_latents": copy.deepcopy(gen.NOISY_LATENTS),
        "size": gen.SIZE,
        "seed": gen.SEED,
        "mnist_index": min(int(gen.MNIST_INDEX), max(0, _mnist_train_count() - 1)),
        "size_range": list(SIZE_RANGE),
        "mnist_train_count": _mnist_train_count(),
    }


def _build_reference(obj: dict[str, Any], size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source = str(obj.get("source", "mnist"))
    if source == "mnist":
        index = int(obj.get("mnist_index", gen.MNIST_INDEX))
        n_mnist = _mnist_train_count()
        if not 0 <= index < n_mnist:
            raise ValueError(f"mnist_index must be in 0..{n_mnist - 1}")
        ref, _raw, label, origin = gen.mnist_reference(index, size)
        info = {"source": "mnist", "mnist_index": index, "label": label, "origin": origin}
        return ref, info
    if source == "mask":
        mask_id = int(obj.get("mask_id", 5))
        T = object_mask_intensity(mask_id, (size, size)).astype(np.float32)
        ref = _mask_reference(T, size, f"mask_{mask_id}")
        return ref, {"source": "mask", "mask_id": mask_id, "label": f"catalog mask {mask_id}"}
    if source == "rabbit":
        T = _load_png_mask_path(RABBIT_MASK_PNG, size)
        ref = _mask_reference(T, size, "rabbit")
        return ref, {
            "source": "rabbit",
            "label": "rabbit SLM mask",
            "origin": str(RABBIT_MASK_PNG),
        }
    if source == "upload":
        b64 = obj.get("png_base64") or obj.get("image_base64") or ""
        if not isinstance(b64, str) or not b64.strip():
            raise ValueError("upload source requires png_base64 (base64-encoded mask PNG)")
        # Allow data-URL prefix from the browser.
        if "," in b64 and b64.strip().lower().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid png_base64: {exc}") from exc
        if not raw:
            raise ValueError("uploaded mask is empty")
        try:
            T = _load_png_mask_bytes(raw, size)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not decode uploaded mask: {exc}") from exc
        file_name = str(obj.get("file_name") or "upload.png")
        ref = _mask_reference(T, size, file_name)
        return ref, {
            "source": "upload",
            "label": file_name,
            "file_name": file_name,
        }
    raise ValueError(
        f"unknown object source {source!r}; use 'mnist', 'mask', 'rabbit', or 'upload'"
    )


def _latents(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(gen.IDEAL_LATENTS) | set(gen.NOISY_LATENTS)
    out = {key: value for key, value in raw.items() if key in allowed}
    if "num_frames" in out:
        out["num_frames"] = int(out["num_frames"])
    return out


def _sync_pump_waists(setup_raw: dict[str, Any]) -> dict[str, Any]:
    """Keep NL1/NL2 pump waists aligned with the single UI pump_waist_um control."""
    out = dict(setup_raw)
    if "pump_waist_um" not in out:
        return out
    w = float(out["pump_waist_um"])
    out["pump_waist_NL1_um"] = w
    out["pump_waist_NL2_um"] = w
    return out


def _bridge_setup_for_web_latents(setup, latents: dict[str, Any]):
    """Map UI latents onto fields the physical A–D twin actually uses.

    Lab profiles use ``use_physical_thermal`` and often lock ``delta_z_um=0``, which
    ignores ``s_focus``. When mechanical Δz is ~0, drive it from ``s_focus``.
    """
    sf = float(latents.get("s_focus", 0.0) or 0.0)
    dz = getattr(setup, "delta_z_um", None)
    if abs(sf) > 1e-12 and (dz is None or abs(float(dz)) < 1e-12):
        zmax = float(getattr(setup, "delta_z_max_um", 100.0) or 100.0)
        setup = replace(setup, delta_z_um=float(sf) * zmax)
    return setup


def preview_event(config: dict[str, Any]) -> dict[str, Any]:
    """Build the SLM object preview event (no stdout side effects)."""
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    return {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}


def preview(config: dict[str, Any]) -> None:
    _emit(preview_event(config))


def iter_run_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield NDJSON run events for Django streaming or CLI emission."""
    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    modes = config.get("modes") or {"ideal": True, "noisy": True}
    run_modes = [m for m in ("ideal", "noisy") if modes.get(m)]
    if not run_modes:
        raise ValueError("enable at least one of ideal / noisy")
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))

    setups = {
        "ideal": _sync_pump_waists({**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}),
        "noisy": _sync_pump_waists({**gen.NOISY_SETUP, **(config.get("noisy_setup") or {})}),
    }
    latents = {
        "ideal": _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})}),
        "noisy": _latents({**gen.NOISY_LATENTS, **(config.get("noisy_latents") or {})}),
    }
    thermal = config.get("thermal") or {}

    # Display-only mode: stream results to the browser, do not write inputs/outputs
    # under outputs/ui_runs (avoids filling host disk on repeated web runs).
    # Set QIUP_UI_PERSIST=1 to restore local file dumps for debugging.
    persist = str(os.environ.get("QIUP_UI_PERSIST", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    out_dir: Path | None = None
    if persist:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = UI_RUNS / stamp
        out_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "type": "start",
        "total": len(profiles) * len(run_modes),
        "profiles": profiles,
        "modes": run_modes,
        "output_dir": str(out_dir) if out_dir is not None else None,
        "persisted": persist,
    }

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    if out_dir is not None:
        np.save(out_dir / "slm_object.npy", slm.astype(np.float32))
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    summary: dict[str, Any] = {
        "object": object_info,
        "size": size,
        "seed": seed,
        "config": config,
        "results": {},
        "persisted": persist,
    }
    ideal_images: dict[str, np.ndarray] = {}
    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        for mode in run_modes:
            t0 = time.perf_counter()
            yield {"type": "progress", "profile": selector, "mode": mode}
            try:
                overrides = {
                    **gen._allowed_overrides(base, setups[mode]),
                    **thermal_override,
                }
                setup = _bridge_setup_for_web_latents(
                    base.overridden(**overrides),
                    latents[mode],
                )
                result = simulate_forward(
                    ref,
                    LatentState(setup=setup, **latents[mode]),
                    seed=seed,
                    detector_noise=(mode == "noisy"),
                )
                image = np.asarray(
                    result.ideal_signal_image if mode == "ideal" else result.detector_signal_image,
                    dtype=np.float64,
                )
                if mode == "ideal":
                    ideal_images[selector] = image
                if out_dir is not None:
                    np.save(out_dir / f"{selector}_{mode}.npy", image.astype(np.float32))
                payload = _image_payload(image)
                diagnostics = _diagnostics_payload(
                    result=result,
                    image=image,
                    mode=mode,
                    slm=slm,
                    ideal_ref=ideal_images.get(selector) if mode == "noisy" else None,
                )
                event = {
                    "type": "result",
                    "profile": selector,
                    "mode": mode,
                    "image": payload,
                    "elapsed_s": time.perf_counter() - t0,
                    "sha256": result.meta.get(
                        "ideal_signal_image_sha256" if mode == "ideal" else "detector_signal_image_sha256"
                    ),
                    "thermal": _thermal_report(setup),
                    "diagnostics": diagnostics,
                    "architecture": {
                        "architecture": str(setup.spdc_architecture),
                        "imaging_mode": str(setup.imaging_mode),
                        "object_coefficient": "T" if is_two_crystal_architecture(setup) else "R",
                        "object_plane": str(setup.object_plane),
                    },
                }
                summary["results"][f"{selector}_{mode}"] = {
                    key: event[key] for key in ("elapsed_s", "sha256")
                } | {k: payload[k] for k in ("min", "max", "mean", "std")}
                yield event
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {"type": "error", "profile": selector, "mode": mode, "message": str(exc)}

    if out_dir is not None:
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": str(out_dir) if out_dir is not None else None,
        "persisted": persist,
    }


def iter_stage_ladder_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Run the fixed noise-stage ladder for selected arrangements (demo/lab teaching)."""
    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    # Keep web latency manageable: default to first selected profile unless many requested.
    if not config.get("stage_all_profiles"):
        profiles = profiles[:1]
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))
    noisy_setup = {**gen.NOISY_SETUP, **(config.get("noisy_setup") or {})}
    noisy_latents = _latents({**gen.NOISY_LATENTS, **(config.get("noisy_latents") or {})})
    ideal_setup = {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}
    ideal_latents = _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})})
    thermal = config.get("thermal") or {}

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    yield {
        "type": "start",
        "total": len(profiles) * len(STAGE_LADDER),
        "profiles": profiles,
        "modes": ["stage"],
        "diagnostic": "stage_ladder",
        "stages": [{"id": s["id"], "label": s["label"], "family": s["family"]} for s in STAGE_LADDER],
        "output_dir": None,
        "persisted": False,
    }
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        ideal_image: np.ndarray | None = None
        for stage in STAGE_LADDER:
            t0 = time.perf_counter()
            yield {
                "type": "progress",
                "profile": selector,
                "mode": "stage",
                "stage_id": stage["id"],
                "stage_label": stage["label"],
            }
            try:
                if stage["setup"] == "noisy_setup":
                    setup_raw = dict(noisy_setup)
                else:
                    setup_raw = {**ideal_setup, **(stage.get("setup") or {})}
                if stage["latents"] == "noisy_latents":
                    lat_raw = dict(noisy_latents)
                else:
                    lat_raw = {**ideal_latents, **(stage.get("latents") or {})}
                overrides = {**gen._allowed_overrides(base, setup_raw), **thermal_override}
                setup = base.overridden(**overrides)
                result = simulate_forward(
                    ref,
                    LatentState(setup=setup, **_latents(lat_raw)),
                    seed=seed,
                    detector_noise=bool(stage["detector_noise"]),
                )
                image = np.asarray(
                    result.ideal_signal_image
                    if not stage["detector_noise"]
                    else result.detector_signal_image,
                    dtype=np.float64,
                )
                if stage["id"] == "ideal":
                    ideal_image = image
                diagnostics = _diagnostics_payload(
                    result=result,
                    image=image,
                    mode="stage",
                    slm=slm,
                    ideal_ref=ideal_image,
                )
                yield {
                    "type": "result",
                    "profile": selector,
                    "mode": "stage",
                    "stage_id": stage["id"],
                    "stage_label": stage["label"],
                    "stage_family": stage["family"],
                    "stage_explain": stage["explain"],
                    "image": _image_payload(image),
                    "elapsed_s": time.perf_counter() - t0,
                    "thermal": _thermal_report(setup),
                    "diagnostics": diagnostics,
                    "architecture": {
                        "architecture": str(setup.spdc_architecture),
                        "imaging_mode": str(setup.imaging_mode),
                        "object_coefficient": "T" if is_two_crystal_architecture(setup) else "R",
                        "object_plane": str(setup.object_plane),
                    },
                }
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {
                    "type": "error",
                    "profile": selector,
                    "mode": "stage",
                    "stage_id": stage["id"],
                    "message": str(exc),
                }

    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": None,
        "persisted": False,
        "diagnostic": "stage_ladder",
    }


def iter_mechanism_matrix_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Run clean + Stage-B ablation mechanisms on the user's SLM object."""
    from verification.ablation.definitions import MECHANISMS
    from verification.ablation.config import clean_config, merge_mechanisms

    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    profiles = profiles[:1]
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))
    thermal = config.get("thermal") or {}
    ideal_setup = {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}
    ideal_latents = _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})})

    mechanism_ids = [
        mid for mid, spec in MECHANISMS.items() if spec.get("stage_b", True)
    ]
    cases = [{"id": "clean", "label": "Clean baseline", "family": "reference", "names": ()}]
    for mid in mechanism_ids:
        cases.append({
            "id": mid,
            "label": mid.replace("_", " "),
            "family": "mechanism",
            "names": (mid,),
        })

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    yield {
        "type": "start",
        "total": len(profiles) * len(cases),
        "profiles": profiles,
        "modes": ["mechanism"],
        "diagnostic": "mechanism_matrix",
        "stages": [{"id": c["id"], "label": c["label"], "family": c["family"]} for c in cases],
        "output_dir": None,
        "persisted": False,
    }
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        ideal_image: np.ndarray | None = None
        for case in cases:
            t0 = time.perf_counter()
            yield {
                "type": "progress",
                "profile": selector,
                "mode": "mechanism",
                "stage_id": case["id"],
                "stage_label": case["label"],
            }
            try:
                merged = clean_config() if not case["names"] else merge_mechanisms(case["names"])
                setup_raw = {**ideal_setup, **merged["setup"]}
                lat_raw = {
                    **ideal_latents,
                    **merged["latents"],
                    "em_gain": float(merged["em_gain"]),
                    "num_frames": int(ideal_latents.get("num_frames", 1) or 1),
                }
                # Ablation clean baseline enables phase-step; keep web matrix single-frame unless requested.
                setup_raw["enable_phase_step_acquisition"] = bool(
                    config.get("enable_phase_step_acquisition", False)
                )
                overrides = {**gen._allowed_overrides(base, setup_raw), **thermal_override}
                setup = base.overridden(**overrides)
                case_ref = _apply_phase_offset(ref, float(merged.get("object_phase_offset_rad", 0.0)))
                result = simulate_forward(
                    case_ref,
                    LatentState(setup=setup, **_latents(lat_raw)),
                    seed=seed,
                    detector_noise=bool(merged["detector_noise"]),
                )
                image = np.asarray(
                    result.ideal_signal_image
                    if not merged["detector_noise"]
                    else result.detector_signal_image,
                    dtype=np.float64,
                )
                if case["id"] == "clean":
                    ideal_image = image
                diagnostics = _diagnostics_payload(
                    result=result,
                    image=image,
                    mode="mechanism",
                    slm=slm,
                    ideal_ref=ideal_image,
                )
                yield {
                    "type": "result",
                    "profile": selector,
                    "mode": "mechanism",
                    "stage_id": case["id"],
                    "stage_label": case["label"],
                    "stage_family": case["family"],
                    "stage_explain": (
                        "Clean baseline (detector nuisances off)."
                        if case["id"] == "clean"
                        else str((MECHANISMS.get(case["id"]) or {}).get("source") or case["label"])
                    ),
                    "image": _image_payload(image),
                    "elapsed_s": time.perf_counter() - t0,
                    "thermal": _thermal_report(setup),
                    "diagnostics": diagnostics,
                    "architecture": _architecture_payload(setup),
                }
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {
                    "type": "error",
                    "profile": selector,
                    "mode": "mechanism",
                    "stage_id": case["id"],
                    "message": str(exc),
                }

    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": None,
        "persisted": False,
        "diagnostic": "mechanism_matrix",
    }


def iter_phase_step_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Four-step phase acquisition → φ(x,y) and modulation maps."""
    from setup_params import FOUR_STEP_PSI_SCHEDULE_RAD

    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    profiles = profiles[:1]
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))
    thermal = config.get("thermal") or {}
    # Prefer ideal optical path for clean phase maps unless noisy requested.
    use_noisy = bool((config.get("modes") or {}).get("noisy"))
    setup_raw = {
        **(gen.NOISY_SETUP if use_noisy else gen.IDEAL_SETUP),
        **((config.get("noisy_setup") if use_noisy else config.get("ideal_setup")) or {}),
        "enable_phase_step_acquisition": True,
        "phase_step_schedule_rad": list(FOUR_STEP_PSI_SCHEDULE_RAD),
    }
    lat_raw = {
        **(gen.NOISY_LATENTS if use_noisy else gen.IDEAL_LATENTS),
        **((config.get("noisy_latents") if use_noisy else config.get("ideal_latents")) or {}),
        "num_frames": len(FOUR_STEP_PSI_SCHEDULE_RAD),
    }

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    yield {
        "type": "start",
        "total": len(profiles),
        "profiles": profiles,
        "modes": ["phase_step"],
        "diagnostic": "phase_step",
        "stages": [{"id": "phase_step", "label": "Four-step phase", "family": "lab"}],
        "output_dir": None,
        "persisted": False,
    }
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    for selector in profiles:
        t0 = time.perf_counter()
        yield {"type": "progress", "profile": selector, "mode": "phase_step", "stage_id": "phase_step"}
        try:
            base = resolve_setup_selection(selector)
            thermal_override = _thermal_overrides(base, thermal.get(selector))
            overrides = {**gen._allowed_overrides(base, setup_raw), **thermal_override}
            # to_dict() omits disabled phase-step keys, so force them after filtering.
            overrides["enable_phase_step_acquisition"] = True
            overrides["phase_step_schedule_rad"] = tuple(FOUR_STEP_PSI_SCHEDULE_RAD)
            setup = base.overridden(**overrides)
            result = simulate_forward(
                ref,
                LatentState(setup=setup, **_latents(lat_raw)),
                seed=seed,
                detector_noise=use_noisy,
            )
            image = np.asarray(
                result.detector_signal_image if use_noisy else result.ideal_signal_image,
                dtype=np.float64,
            )
            diagnostics = _diagnostics_payload(
                result=result, image=image, mode="phase_step", slm=slm
            )
            yield {
                "type": "result",
                "profile": selector,
                "mode": "phase_step",
                "stage_id": "phase_step",
                "stage_label": "Four-step phase",
                "stage_family": "lab",
                "stage_explain": "enable_phase_step_acquisition with 4-step ψ schedule → φ and modulation maps.",
                "image": _image_payload(image),
                "elapsed_s": time.perf_counter() - t0,
                "thermal": _thermal_report(setup),
                "diagnostics": diagnostics,
                "architecture": _architecture_payload(setup),
            }
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            yield {"type": "error", "profile": selector, "mode": "phase_step", "message": str(exc)}

    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": None,
        "persisted": False,
        "diagnostic": "phase_step",
    }


def iter_sweep_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """One-parameter metric sweep (SNR, V, mean, RMS residual)."""
    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    profiles = profiles[:1]
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))
    thermal = config.get("thermal") or {}
    param = str(config.get("sweep_param") or "s_focus")
    if param not in SWEEP_PARAMS:
        raise ValueError(f"unsupported sweep_param {param!r}")
    spec = SWEEP_PARAMS[param]
    n_points = int(config.get("sweep_points") or 7)
    n_points = max(3, min(11, n_points))
    values = np.linspace(float(spec["lo"]), float(spec["hi"]), n_points)

    # Detector knobs need the noisy path; optics/drift knobs are masked by it.
    is_optics = str(spec.get("family") or "") == "optics"
    use_noisy = not is_optics
    if is_optics:
        setup_base = {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}
        lat_base = _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})})
        for key in _OPTICS_LATENT_KEYS:
            if key != param:
                lat_base[key] = 0.0
        lat_base["num_frames"] = max(int(lat_base.get("num_frames") or 1), 8)
        lat_base["exposure_s"] = max(float(lat_base.get("exposure_s") or 0.02), 0.05)
        # Keep acquisition fixed across the curve (avoids frame-count jumps looking
        # like a thermal effect). F5 needs T_acq ≳ 30 s when s_thermal > 0.55.
        if param == "s_thermal":
            lat_base["num_frames"] = max(int(lat_base.get("num_frames") or 1), 64)
            lat_base["exposure_s"] = max(float(lat_base.get("exposure_s") or 0.02), 0.5)
    else:
        setup_base = {**gen.NOISY_SETUP, **(config.get("noisy_setup") or {})}
        lat_base = _latents({**gen.NOISY_LATENTS, **(config.get("noisy_latents") or {})})
    ideal_setup = {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}
    ideal_latents = _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})})

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    # Flat SLM masks hide focus/blur curves; fall back to a structured digit.
    if is_optics and float(np.std(slm)) < 1e-3:
        ref, object_info = _build_reference({"source": "mnist", "digit": 5}, size)
        slm = np.asarray(ref["T"], dtype=np.float64)
        object_info = {
            **object_info,
            "note": "optics sweep used MNIST digit (selected mask was spatially flat)",
        }
    yield {
        "type": "start",
        "total": len(profiles) * (1 + n_points),
        "profiles": profiles,
        "modes": ["sweep"],
        "diagnostic": "sweep",
        "sweep": {"param": param, "label": spec["label"], "values": [float(v) for v in values]},
        "stages": [{"id": "ideal_ref", "label": "Ideal reference", "family": "reference"}]
        + [{"id": f"p{i}", "label": f"{spec['label']}={float(v):.3g}", "family": "sweep"} for i, v in enumerate(values)],
        "output_dir": None,
        "persisted": False,
    }
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        # Ideal reference once
        t0 = time.perf_counter()
        yield {"type": "progress", "profile": selector, "mode": "sweep", "stage_id": "ideal_ref"}
        try:
            # Match acquisition / base to the sweep family so residual_rms is
            # about the swept knob, not frame-count / gain mismatch.
            if is_optics:
                ref_setup_raw = dict(setup_base)
                ref_lat_raw = dict(lat_base)
                if spec["target"] == "latents":
                    ref_lat_raw[param] = float(spec["lo"])
                else:
                    ref_setup_raw[param] = float(spec["lo"])
                ref_overrides = {
                    **gen._allowed_overrides(base, ref_setup_raw),
                    **thermal_override,
                }
                ideal_setup_obj = _prepare_sweep_setup(
                    setup=base.overridden(**ref_overrides),
                    param=param,
                    value=float(spec["lo"]),
                )
                ideal_result = simulate_forward(
                    ref,
                    LatentState(setup=ideal_setup_obj, **_latents(ref_lat_raw)),
                    seed=seed,
                    detector_noise=False,
                    raise_on_forbidden=False,
                )
            else:
                ideal_overrides = {
                    **gen._allowed_overrides(base, ideal_setup),
                    **thermal_override,
                }
                ideal_setup_obj = base.overridden(**ideal_overrides)
                ideal_result = simulate_forward(
                    ref,
                    LatentState(setup=ideal_setup_obj, **ideal_latents),
                    seed=seed,
                    detector_noise=False,
                )
            ideal_image = np.asarray(ideal_result.ideal_signal_image, dtype=np.float64)
            yield {
                "type": "result",
                "profile": selector,
                "mode": "sweep",
                "stage_id": "ideal_ref",
                "stage_label": "Ideal reference",
                "stage_family": "reference",
                "image": _image_payload(ideal_image),
                "elapsed_s": time.perf_counter() - t0,
                "diagnostics": _diagnostics_payload(
                    result=ideal_result, image=ideal_image, mode="sweep", slm=slm
                ),
                "architecture": _architecture_payload(ideal_setup_obj),
                "sweep_point": None,
            }
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            yield {"type": "error", "profile": selector, "mode": "sweep", "message": str(exc)}
            continue

        curve: list[dict[str, Any]] = []
        for i, value in enumerate(values):
            t0 = time.perf_counter()
            stage_id = f"p{i}"
            yield {
                "type": "progress",
                "profile": selector,
                "mode": "sweep",
                "stage_id": stage_id,
                "stage_label": f"{spec['label']}={float(value):.3g}",
            }
            try:
                setup_raw = dict(setup_base)
                lat_raw = dict(lat_base)
                if spec["target"] == "setup":
                    setup_raw[param] = float(value)
                else:
                    lat_raw[param] = float(value)
                overrides = {**gen._allowed_overrides(base, setup_raw), **thermal_override}
                setup = _prepare_sweep_setup(
                    setup=base.overridden(**overrides),
                    param=param,
                    value=float(value),
                )
                result = simulate_forward(
                    ref,
                    LatentState(setup=setup, **_latents(lat_raw)),
                    seed=seed,
                    detector_noise=use_noisy,
                    raise_on_forbidden=False,
                )
                if use_noisy:
                    image = np.asarray(result.detector_signal_image, dtype=np.float64)
                else:
                    image = np.asarray(result.ideal_signal_image, dtype=np.float64)
                diagnostics = _diagnostics_payload(
                    result=result,
                    image=image,
                    mode="sweep",
                    slm=slm,
                    ideal_ref=ideal_image,
                )
                point = {
                    "value": float(value),
                    "snr": diagnostics.get("snr"),
                    "V_50_50": (diagnostics.get("visibility") or {}).get("V_50_50"),
                    "mean": float(np.mean(image)),
                    "residual_rms": diagnostics.get("residual_rms"),
                }
                curve.append(point)
                yield {
                    "type": "result",
                    "profile": selector,
                    "mode": "sweep",
                    "stage_id": stage_id,
                    "stage_label": f"{spec['label']}={float(value):.3g}",
                    "stage_family": "sweep",
                    "image": _image_payload(image),
                    "elapsed_s": time.perf_counter() - t0,
                    "diagnostics": diagnostics,
                    "architecture": _architecture_payload(setup),
                    "sweep_point": point,
                }
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {
                    "type": "error",
                    "profile": selector,
                    "mode": "sweep",
                    "stage_id": stage_id,
                    "message": str(exc),
                }

        yield {
            "type": "sweep_curve",
            "profile": selector,
            "param": param,
            "label": spec["label"],
            "curve": curve,
        }

    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": None,
        "persisted": False,
        "diagnostic": "sweep",
    }


def iter_fringe_curve_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Teaching I(φ) curve: mean I₊ / I₋ / D vs object phase offset.

    Uses the existing forward model with a uniform object-phase offset — no new
    physics. This closes the main Web-vs-Twin gap for interferometric demos.
    """
    started = time.perf_counter()
    profiles = [p for p in PROFILE_ORDER if p in set(config.get("profiles") or PROFILE_ORDER)]
    if not profiles:
        raise ValueError("select at least one arrangement")
    profiles = profiles[:1]
    size = int(config.get("size", gen.SIZE))
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be in {SIZE_RANGE[0]}..{SIZE_RANGE[1]}")
    seed = int(config.get("seed", gen.SEED))
    thermal = config.get("thermal") or {}
    n_points = int(config.get("fringe_points") or config.get("sweep_points") or 9)
    n_points = max(5, min(17, n_points))
    phases = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)

    # Clean optical path so the fringe shape is readable.
    setup_raw = {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})}
    lat_raw = _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {}), "num_frames": 1})

    ref0, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref0["T"], dtype=np.float64)
    yield {
        "type": "start",
        "total": len(profiles) * n_points,
        "profiles": profiles,
        "modes": ["fringe"],
        "diagnostic": "fringe_curve",
        "fringe": {"points": n_points, "phi_rad": [float(p) for p in phases]},
        "stages": [
            {"id": f"phi{i}", "label": f"φ={float(p):.2f} rad", "family": "fringe"}
            for i, p in enumerate(phases)
        ],
        "output_dir": None,
        "persisted": False,
    }
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        overrides = {**gen._allowed_overrides(base, setup_raw), **thermal_override}
        setup = base.overridden(**overrides)
        curve: list[dict[str, Any]] = []
        for i, phi in enumerate(phases):
            t0 = time.perf_counter()
            stage_id = f"phi{i}"
            yield {
                "type": "progress",
                "profile": selector,
                "mode": "fringe",
                "stage_id": stage_id,
                "stage_label": f"φ={float(phi):.2f} rad",
            }
            try:
                ref = _apply_phase_offset(ref0, float(phi))
                result = simulate_forward(
                    ref,
                    LatentState(setup=setup, **lat_raw),
                    seed=seed,
                    detector_noise=False,
                    raise_on_forbidden=False,
                )
                fp = np.asarray(result.frames_plus[0], dtype=np.float64)
                fm = np.asarray(result.frames_minus[0], dtype=np.float64)
                fd = np.asarray(result.frames_D[0], dtype=np.float64)
                vis = _compact_visibility(dict(getattr(result, "meta", {}) or {}))
                point = {
                    "phi_rad": float(phi),
                    "mean_I_plus": float(np.mean(fp)),
                    "mean_I_minus": float(np.mean(fm)),
                    "mean_D": float(np.mean(fd)),
                    "V_50_50": vis.get("V_50_50"),
                }
                curve.append(point)
                yield {
                    "type": "result",
                    "profile": selector,
                    "mode": "fringe",
                    "stage_id": stage_id,
                    "stage_label": f"φ={float(phi):.2f} rad",
                    "stage_family": "fringe",
                    "image": _image_payload(fp),
                    "elapsed_s": time.perf_counter() - t0,
                    "diagnostics": _diagnostics_payload(
                        result=result, image=fp, mode="fringe", slm=slm
                    ),
                    "architecture": _architecture_payload(setup),
                    "fringe_point": point,
                }
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {
                    "type": "error",
                    "profile": selector,
                    "mode": "fringe",
                    "stage_id": stage_id,
                    "message": str(exc),
                }

        def _michelson(key: str) -> float | None:
            vals = [float(p[key]) for p in curve if p.get(key) is not None]
            if len(vals) < 2:
                return None
            lo = min(vals)
            hi = max(vals)
            den = abs(hi) + abs(lo)
            if den < 1e-15:
                return 0.0
            return float((hi - lo) / den)

        yield {
            "type": "fringe_curve",
            "profile": selector,
            "label": "Object phase φ (rad)",
            "curve": curve,
            "summary": {
                "michelson_I_plus": _michelson("mean_I_plus"),
                "michelson_I_minus": _michelson("mean_I_minus"),
                "michelson_D": _michelson("mean_D"),
                "n_points": len(curve),
            },
        }

    yield {
        "type": "done",
        "elapsed_s": time.perf_counter() - started,
        "output_dir": None,
        "persisted": False,
        "diagnostic": "fringe_curve",
    }


def iter_diagnostics_events(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Dispatch lab/diagnostics streams by config['diagnostic']."""
    kind = str(config.get("diagnostic") or "stage_ladder").strip().lower()
    if kind in {"stage_ladder", "stages", "ladder"}:
        yield from iter_stage_ladder_events(config)
        return
    if kind in {"mechanism_matrix", "mechanisms", "ablation"}:
        yield from iter_mechanism_matrix_events(config)
        return
    if kind in {"phase_step", "phase"}:
        yield from iter_phase_step_events(config)
        return
    if kind in {"sweep", "param_sweep"}:
        yield from iter_sweep_events(config)
        return
    if kind in {"fringe_curve", "fringe", "i_phi", "visibility_curve"}:
        yield from iter_fringe_curve_events(config)
        return
    raise ValueError(
        "diagnostic must be one of: stage_ladder, mechanism_matrix, phase_step, "
        "sweep, fringe_curve"
    )


def run(config: dict[str, Any]) -> None:
    for event in iter_run_events(config):
        _emit(event)


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if command == "defaults":
            _emit(defaults())
            return 0
        if command == "run":
            run(json.loads(sys.stdin.read() or "{}"))
            return 0
        if command == "preview":
            preview(json.loads(sys.stdin.read() or "{}"))
            return 0
        _emit({"type": "error", "message": "usage: qiup_ui_backend.py defaults|run|preview"})
        return 2
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
