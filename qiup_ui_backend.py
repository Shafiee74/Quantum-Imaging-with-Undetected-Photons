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
import sys
import time
import traceback
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


def _png_base64(image: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> str:
    from matplotlib import colormaps
    from PIL import Image

    data = np.asarray(image, dtype=np.float64)
    lo = float(np.min(data)) if vmin is None else float(vmin)
    hi = float(np.max(data)) if vmax is None else float(vmax)
    norm = np.zeros_like(data) if hi <= lo else np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    rgba = (colormaps["inferno"](norm) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_payload(image: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> dict[str, Any]:
    data = np.asarray(image, dtype=np.float64)
    return {
        "png": _png_base64(data, vmin, vmax),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "shape": list(data.shape),
    }


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
        ref = {
            "M": T.copy(),
            "T": T,
            "phi": np.zeros_like(T),
            "O_i": np.sqrt(T).astype(np.float32),
            "D": np.zeros_like(T),
            "target_size": size,
            "label": f"mask_{mask_id}",
        }
        return ref, {"source": "mask", "mask_id": mask_id, "label": f"catalog mask {mask_id}"}
    raise ValueError(f"unknown object source {source!r}; use 'mnist' or 'mask'")


def _latents(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(gen.IDEAL_LATENTS) | set(gen.NOISY_LATENTS)
    out = {key: value for key, value in raw.items() if key in allowed}
    if "num_frames" in out:
        out["num_frames"] = int(out["num_frames"])
    return out


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
        "ideal": {**gen.IDEAL_SETUP, **(config.get("ideal_setup") or {})},
        "noisy": {**gen.NOISY_SETUP, **(config.get("noisy_setup") or {})},
    }
    latents = {
        "ideal": _latents({**gen.IDEAL_LATENTS, **(config.get("ideal_latents") or {})}),
        "noisy": _latents({**gen.NOISY_LATENTS, **(config.get("noisy_latents") or {})}),
    }
    thermal = config.get("thermal") or {}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = UI_RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "type": "start",
        "total": len(profiles) * len(run_modes),
        "profiles": profiles,
        "modes": run_modes,
        "output_dir": str(out_dir),
    }

    ref, object_info = _build_reference(config.get("object") or {}, size)
    slm = np.asarray(ref["T"], dtype=np.float64)
    np.save(out_dir / "slm_object.npy", slm.astype(np.float32))
    yield {"type": "object", **object_info, "image": _image_payload(slm, 0.0, 1.0)}

    summary: dict[str, Any] = {
        "object": object_info,
        "size": size,
        "seed": seed,
        "config": config,
        "results": {},
    }
    for selector in profiles:
        base = resolve_setup_selection(selector)
        thermal_override = _thermal_overrides(base, thermal.get(selector))
        for mode in run_modes:
            t0 = time.perf_counter()
            yield {"type": "progress", "profile": selector, "mode": mode}
            try:
                overrides = {**gen._allowed_overrides(base, setups[mode]), **thermal_override}
                setup = base.overridden(**overrides)
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
                np.save(out_dir / f"{selector}_{mode}.npy", image.astype(np.float32))
                payload = _image_payload(image)
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
                }
                summary["results"][f"{selector}_{mode}"] = {
                    key: event[key] for key in ("elapsed_s", "sha256")
                } | {k: payload[k] for k in ("min", "max", "mean", "std")}
                yield event
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                yield {"type": "error", "profile": selector, "mode": mode, "message": str(exc)}

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    yield {"type": "done", "elapsed_s": time.perf_counter() - started, "output_dir": str(out_dir)}


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
