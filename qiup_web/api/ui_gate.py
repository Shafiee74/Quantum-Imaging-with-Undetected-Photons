"""UI-only API gate: require a cookie issued by the Django website page."""
from __future__ import annotations

from django.conf import settings
from django.core import signing
from django.http import HttpRequest, HttpResponse, JsonResponse

UI_GATE_COOKIE = "qiup_ui_gate"
UI_GATE_SALT = "qiup-ui-gate"
UI_GATE_MAX_AGE = 60 * 60 * 12  # 12 hours


def issue_ui_gate_cookie(response: HttpResponse) -> HttpResponse:
    token = signing.TimestampSigner(salt=UI_GATE_SALT).sign("ui")
    response.set_cookie(
        UI_GATE_COOKIE,
        token,
        max_age=UI_GATE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=not settings.DEBUG,
        path="/",
    )
    return response


def require_ui_gate(request: HttpRequest) -> JsonResponse | None:
    """Return a 403 response when the caller did not open the website UI first."""
    raw = request.COOKIES.get(UI_GATE_COOKIE)
    if not raw:
        return JsonResponse(
            {
                "error": "API is only available from the QIUP website UI. Open the site first.",
                "type": "error",
                "message": "API is only available from the QIUP website UI. Open the site first.",
            },
            status=403,
        )
    try:
        signing.TimestampSigner(salt=UI_GATE_SALT).unsign(raw, max_age=UI_GATE_MAX_AGE)
    except signing.BadSignature:
        return JsonResponse(
            {
                "error": "invalid or expired UI session; reload the website",
                "type": "error",
                "message": "invalid or expired UI session; reload the website",
            },
            status=403,
        )
    return None
