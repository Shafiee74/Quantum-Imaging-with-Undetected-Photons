"""Django settings for the QIUP web API (UI backend)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

_DEV_SECRET = "qiup-dev-only-change-me-before-any-public-deploy"
SECRET_KEY = os.environ.get("QIUP_DJANGO_SECRET_KEY", _DEV_SECRET)
DEBUG = os.environ.get("QIUP_DJANGO_DEBUG", "1") == "1"
if not DEBUG and SECRET_KEY == _DEV_SECRET:
    raise RuntimeError(
        "Refusing to start with the default SECRET_KEY while QIUP_DJANGO_DEBUG=0. "
        "Set QIUP_DJANGO_SECRET_KEY to a long random value before public deploy."
    )

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("QIUP_DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "corsheaders",
    "api",
    "ui",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "qiup_web.urls"
WSGI_APPLICATION = "qiup_web.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Cap JSON body size for SLM upload base64 (web abuse guard; not a physics change).
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get("QIUP_MAX_UPLOAD_BYTES", str(3 * 1024 * 1024)))

# Session cookie used with CSRF so API POSTs only work from the website UI.
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_HTTPONLY = False  # JS must read csrftoken for X-CSRFToken
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "QIUP_CSRF_TRUSTED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://testserver",
    ).split(",")
    if origin.strip()
]

# Hardening when not in local debug mode.
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = "DENY"
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = os.environ.get("QIUP_DJANGO_SSL_REDIRECT", "0") == "1"

# Keep CORS empty by default so only the same-origin Django UI can call the API.
# Set QIUP_CORS_ORIGINS only if you intentionally need a separate front-end origin.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("QIUP_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = False

# Simulator package lives at the repository root (parent of qiup_web/).
QIUP_REPO_ROOT = REPO_ROOT
QIUP_PYTHON = os.environ.get("QIUP_PYTHON", "python")
