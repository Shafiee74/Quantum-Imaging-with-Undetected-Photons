"""HTTP API replacing the former Next.js/Node Python-spawn bridge."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator

from django.conf import settings
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Repository root must be importable for qiup_ui_backend / simulator modules.
_REPO = Path(settings.QIUP_REPO_ROOT)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import qiup_ui_backend as backend  # noqa: E402


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"error": message, "type": "error", "message": message}, status=status)


def _parse_json_body(request: HttpRequest) -> dict[str, Any]:
    raw = request.body.decode("utf-8") if request.body else "{}"
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


@csrf_exempt
@require_GET
def defaults_view(request: HttpRequest) -> JsonResponse:
    try:
        payload = backend.defaults()
        payload = {
            **payload,
            "backend": {
                "framework": "django",
                "python": settings.QIUP_PYTHON,
                "root": str(settings.QIUP_REPO_ROOT),
            },
        }
        return JsonResponse(payload)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _json_error(str(exc), status=500)


@csrf_exempt
@require_POST
def preview_view(request: HttpRequest) -> JsonResponse:
    try:
        config = _parse_json_body(request)
        event = backend.preview_event(config)
        return JsonResponse(event)
    except json.JSONDecodeError as exc:
        return _json_error(f"invalid JSON: {exc}", status=400)
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _json_error(str(exc), status=400)


def _ndjson_stream(config: dict[str, Any]) -> Iterator[bytes]:
    try:
        for event in backend.iter_run_events(config):
            yield (json.dumps(event, default=backend._json_default) + "\n").encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        yield (
            json.dumps({"type": "error", "message": str(exc)}, default=backend._json_default) + "\n"
        ).encode("utf-8")


@csrf_exempt
@require_POST
def run_view(request: HttpRequest) -> StreamingHttpResponse | JsonResponse:
    try:
        config = _parse_json_body(request)
    except json.JSONDecodeError as exc:
        return _json_error(f"invalid JSON: {exc}", status=400)
    except ValueError as exc:
        return _json_error(str(exc), status=400)

    response = StreamingHttpResponse(
        _ndjson_stream(config),
        content_type="application/x-ndjson; charset=utf-8",
    )
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"
    return response
