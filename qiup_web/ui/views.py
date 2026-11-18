from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from api.ui_gate import issue_ui_gate_cookie


@ensure_csrf_cookie
def home(request):
    """Serve the simulator UI and issue cookies required by /api/*."""
    # Ensure a CSRF cookie + token exist before the page JS calls the API.
    get_token(request)
    response = render(request, "ui/index.html")
    return issue_ui_gate_cookie(response)
