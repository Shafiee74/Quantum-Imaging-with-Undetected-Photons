from django.urls import include, path

urlpatterns = [
    path("", include("ui.urls")),
    path("api/", include("api.urls")),
]
