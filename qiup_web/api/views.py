"""HTTP API replacing the former Next.js/Node Python-spawn bridge."""
from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from django.conf import settings
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_GET, require_POST

from api.ui_gate import require_ui_gate

# Repository root must be importable for qiup_ui_backend / simulator modules.
_REPO = Path(settings.QIUP_REPO_ROOT)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import qiup_ui_backend as backend  # noqa: E402

# Clamp only abuse-prone fields to the web UI ranges (no physics model changes).
_CLAMP_LATENTS: dict[str, tuple[float, float, bool]] = {
    # key: (min, max, integer?)
    "mean_photons_per_mode": (0.1, 20.0, False),
    "exposure_s": (0.005, 1.0, False),
    "em_gain": (1.0, 1000.0, True),
    "num_frames": (1.0, 64.0, True),
    "s_thermal": (0.0, 0.99, False),
}
_CLAMP_SETUP: dict[str, tuple[float, float, bool]] = {
    "pump_waist_um": (100.0, 350.0, False),
    "pixel_um": (6.5, 24.0, False),
    "object_map_pixel_um": (2.0, 24.0, False),
}
_THERMAL_LIMITS = (20.0, 90.0)
_VALID_PROFILES = {"A", "B", "C", "D"}
_RUN_LOCK = threading.Lock()
_MAX_UPLOAD_CHARS = 2_500_000  # ~1.8 MB binary after base64; web abuse guard only


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"error": message, "type": "error", "message": message}, status=status)


def _reject_cross_site_post(request: HttpRequest) -> JsonResponse | None:
    """Require same-site Origin/Referer for browser-like POSTs."""
    host = (request.get_host() or "").lower()
    origin = (request.META.get("HTTP_ORIGIN") or "").strip()
    referer = (request.META.get("HTTP_REFERER") or "").strip()

    def _host_of(url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:  # noqa: BLE001
            return ""

    if origin:
        origin_host = _host_of(origin)
        if not origin_host or origin_host != host:
            return _json_error("cross-origin API posts are not allowed", status=403)
        return None

    if referer:
        referer_host = _host_of(referer)
        if not referer_host or referer_host != host:
            return _json_error("cross-origin API posts are not allowed", status=403)
        return None

    # No Origin/Referer: block outside DEBUG (curl/scripts). Django tests run with DEBUG=True.
    if not settings.DEBUG:
        return _json_error("API posts require the website UI (missing Origin/Referer)", status=403)
    return None


def _require_json_content_type(request: HttpRequest) -> None:
    ctype = (request.META.get("CONTENT_TYPE") or "").split(";")[0].strip().lower()
    if request.body and ctype and ctype not in {"application/json", "text/json"}:
        raise ValueError("Content-Type must be application/json")


def _parse_json_body(request: HttpRequest) -> dict[str, Any]:
    _require_json_content_type(request)
    raw = request.body.decode("utf-8") if request.body else "{}"
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _friendly_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    if "invalid literal for int()" in msg:
        return "invalid integer value in request"
    if "could not convert string to float" in msg:
        return "invalid number value in request"
    return msg


def _clamp_number(value: Any, lo: float, hi: float, *, integer: bool = False) -> float | int:
    x = float(value)
    x = min(hi, max(lo, x))
    return int(round(x)) if integer else x


def _clamp_known(raw: Any, rules: dict[str, tuple[float, float, bool]]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out = dict(raw)
    for key, (lo, hi, integer) in rules.items():
        if key in out and isinstance(out[key], (int, float)) and not isinstance(out[key], bool):
            out[key] = _clamp_number(out[key], lo, hi, integer=integer)
    return out


def _sanitize_thermal(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    lo, hi = _THERMAL_LIMITS
    out: dict[str, Any] = {}
    for profile, vals in raw.items():
        if profile not in _VALID_PROFILES or not isinstance(vals, dict):
            continue
        cleaned: dict[str, float] = {}
        for key, value in vals.items():
            if not isinstance(key, str):
                continue
            if not (key.startswith("T_pm0_") or key.startswith("dT_")):
                continue
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                continue
            if key.startswith("T_pm0_"):
                cleaned[key] = float(_clamp_number(value, lo, hi))
            else:
                # ΔT offset; keep modest to avoid leaving the 20–90 °C crystal window.
                cleaned[key] = float(_clamp_number(value, -30.0, 30.0))
        out[profile] = cleaned
    return out


def _sanitize_run_config(config: dict[str, Any]) -> dict[str, Any]:
    profiles_in = config.get("profiles") or list(_VALID_PROFILES)
    if not isinstance(profiles_in, list):
        raise ValueError("profiles must be a list")
    unknown_profiles = [str(p) for p in profiles_in if p not in _VALID_PROFILES]
    if unknown_profiles:
        raise ValueError(f"unknown arrangement id(s): {', '.join(unknown_profiles)}")
    profiles = [p for p in profiles_in if p in _VALID_PROFILES]
    if not profiles:
        raise ValueError("select at least one arrangement (A–D)")

    modes = config.get("modes") or {"ideal": True, "noisy": True}
    if not isinstance(modes, dict):
        raise ValueError("modes must be an object")
    modes = {"ideal": bool(modes.get("ideal")), "noisy": bool(modes.get("noisy"))}
    if not modes["ideal"] and not modes["noisy"]:
        raise ValueError("enable at least one of ideal / noisy")

    size = int(_clamp_number(config.get("size", 64), 32, 128, integer=True))
    seed = int(_clamp_number(config.get("seed", 0), 0, 2**31 - 1, integer=True))

    return {
        **config,
        "profiles": profiles,
        "modes": modes,
        "object": config.get("object") or {"source": "mnist", "mnist_index": 5},
        "size": size,
        "seed": seed,
        "thermal": _sanitize_thermal(config.get("thermal")),
        "ideal_setup": _clamp_known(config.get("ideal_setup") or {}, _CLAMP_SETUP),
        "noisy_setup": _clamp_known(config.get("noisy_setup") or {}, _CLAMP_SETUP),
        "ideal_latents": _clamp_known(config.get("ideal_latents") or {}, _CLAMP_LATENTS),
        "noisy_latents": _clamp_known(config.get("noisy_latents") or {}, _CLAMP_LATENTS),
    }


@require_GET
def defaults_view(request: HttpRequest) -> JsonResponse:
    gated = require_ui_gate(request)
    if gated is not None:
        return gated
    try:
        payload = backend.defaults()
        # Web UI: do not expose prototype / calibration banners in the API payload.
        payload.pop("model_notes", None)
        payload.pop("calibration_status", None)
        payload.pop("release_classification", None)
        payload = {
            **payload,
            "backend": {
                "framework": "django",
                "python": "python",
                # Do not expose absolute host filesystem paths (review S3).
                "root": "local",
            },
            "ui_limits": {
                "size": [32, 128],
                "thermal_C": list(_THERMAL_LIMITS),
                "num_frames": [1, 64],
                "mean_photons_per_mode": [0.1, 20.0],
                "pump_waist_um": [100.0, 350.0],
            },
        }
        return JsonResponse(payload)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _json_error(_friendly_error(exc), status=500)


def _guard_upload_size(config: dict[str, Any]) -> None:
    obj = config.get("object") if isinstance(config.get("object"), dict) else {}
    b64 = obj.get("png_base64") if isinstance(obj, dict) else None
    if isinstance(b64, str) and len(b64) > _MAX_UPLOAD_CHARS:
        raise ValueError("uploaded mask is too large for the web UI")


@require_POST
def preview_view(request: HttpRequest) -> JsonResponse:
    gated = require_ui_gate(request)
    if gated is not None:
        return gated
    blocked = _reject_cross_site_post(request)
    if blocked is not None:
        return blocked
    try:
        config = _parse_json_body(request)
        _guard_upload_size(config)
        if "size" in config:
            size = int(config["size"])
            if size < 32 or size > 128:
                raise ValueError("size must be in 32..128")
            config["size"] = size
        event = backend.preview_event(config)
        return JsonResponse(event)
    except ValueError as exc:
        return _json_error(_friendly_error(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _json_error(_friendly_error(exc), status=400)


def _ndjson_stream(config: dict[str, Any]) -> Iterator[bytes]:
    try:
        for event in backend.iter_run_events(config):
            if isinstance(event, dict) and event.get("output_dir"):
                event = {**event, "output_dir": None}
            yield (json.dumps(event, default=backend._json_default) + "\n").encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield (
            json.dumps(
                {"type": "error", "message": _friendly_error(exc)},
                default=backend._json_default,
            )
            + "\n"
        ).encode("utf-8")


def _ndjson_stream_locked(config: dict[str, Any]) -> Iterator[bytes]:
    try:
        yield from _ndjson_stream(config)
    finally:
        _RUN_LOCK.release()


@require_POST
def run_view(request: HttpRequest) -> StreamingHttpResponse | JsonResponse:
    gated = require_ui_gate(request)
    if gated is not None:
        return gated
    blocked = _reject_cross_site_post(request)
    if blocked is not None:
        return blocked
    try:
        config = _sanitize_run_config(_parse_json_body(request))
        _guard_upload_size(config)
    except ValueError as exc:
        return _json_error(_friendly_error(exc), status=400)

    if not _RUN_LOCK.acquire(blocking=False):
        return _json_error("another simulation is already running; try again shortly", status=429)

    response = StreamingHttpResponse(
        _ndjson_stream_locked(config),
        content_type="application/x-ndjson; charset=utf-8",
    )
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"
    return response


def _ndjson_diagnostics_locked(config: dict[str, Any]) -> Iterator[bytes]:
    try:
        for event in backend.iter_diagnostics_events(config):
            if isinstance(event, dict) and event.get("output_dir"):
                event = {**event, "output_dir": None}
            yield (json.dumps(event, default=backend._json_default) + "\n").encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield (
            json.dumps(
                {"type": "error", "message": _friendly_error(exc)},
                default=backend._json_default,
            )
            + "\n"
        ).encode("utf-8")
    finally:
        _RUN_LOCK.release()


@require_POST
def diagnostics_view(request: HttpRequest) -> StreamingHttpResponse | JsonResponse:
    """Lab/diagnostics streams: stage ladder, mechanisms, phase-step, fringe, sweeps."""
    gated = require_ui_gate(request)
    if gated is not None:
        return gated
    blocked = _reject_cross_site_post(request)
    if blocked is not None:
        return blocked
    try:
        raw = _parse_json_body(request)
        config = _sanitize_run_config(raw)
        # Preserve lab-only keys stripped by sanitize defaults.
        if "diagnostic" in raw:
            config["diagnostic"] = str(raw.get("diagnostic") or "stage_ladder")
        if "sweep_param" in raw:
            config["sweep_param"] = str(raw.get("sweep_param") or "s_focus")
        if "sweep_points" in raw and isinstance(raw.get("sweep_points"), (int, float)):
            config["sweep_points"] = int(raw["sweep_points"])
        if "fringe_points" in raw and isinstance(raw.get("fringe_points"), (int, float)):
            config["fringe_points"] = int(raw["fringe_points"])
        if "enable_phase_step_acquisition" in raw:
            config["enable_phase_step_acquisition"] = bool(raw.get("enable_phase_step_acquisition"))
        _guard_upload_size(config)
    except ValueError as exc:
        return _json_error(_friendly_error(exc), status=400)

    if not _RUN_LOCK.acquire(blocking=False):
        return _json_error("another simulation is already running; try again shortly", status=429)

    response = StreamingHttpResponse(
        _ndjson_diagnostics_locked(config),
        content_type="application/x-ndjson; charset=utf-8",
    )
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"
    return response
