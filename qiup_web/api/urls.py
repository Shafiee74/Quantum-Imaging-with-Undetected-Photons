from django.urls import path

from . import views

urlpatterns = [
    path("defaults", views.defaults_view, name="api-defaults"),
    path("defaults/", views.defaults_view, name="api-defaults-slash"),
    path("preview", views.preview_view, name="api-preview"),
    path("preview/", views.preview_view, name="api-preview-slash"),
    path("run", views.run_view, name="api-run"),
    path("run/", views.run_view, name="api-run-slash"),
]
