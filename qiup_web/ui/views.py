from django.shortcuts import render


def home(request):
    """Serve the simulator UI (static JS talks to /api/*)."""
    return render(request, "ui/index.html")
