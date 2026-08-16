"""Django settings for the Shelfie backend.

Deliberately small: this project has no auth, no admin, no templates and no
static assets. It is a JSON API in front of the vision -> VLM -> matching
pipeline, and the settings reflect only that.

BASE_DIR is backend/, which is also the sys.path root (see manage.py), so
`import vision`, `import vlm` and `import matching` work unchanged.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent


# --------------------------------------------------------------------------
# .env loading
#
# vlm/gemini.py reads GEMINI_API_KEY and GEMINI_MODEL straight from
# os.environ, so something has to put backend/.env there first. This is that
# something -- ten lines instead of a python-dotenv dependency.
#
# setdefault, not assignment: a variable already exported in the shell wins
# over the file, so `GEMINI_MODEL=x manage.py ...` behaves as expected.
# --------------------------------------------------------------------------


def _load_env(path):
    """Parse KEY=VALUE lines from a .env file into os.environ."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env(BASE_DIR / ".env")


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

# No auth, no sessions, no signed cookies in this project, so this key signs
# nothing that matters. It still must not be a committed literal that looks
# production-ready -- override DJANGO_SECRET_KEY if this is ever deployed.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

# The Expo app runs on a phone and reaches this server over the LAN, so the
# host it uses is whatever IP the dev machine happens to have. Locking this
# down would mean editing settings per demo machine; deployment is explicitly
# out of scope for this task.
ALLOWED_HOSTS = ["*"]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

INSTALLED_APPS = [
    # contenttypes/auth carry no features we use, but DRF resolves
    # request.user against AnonymousUser and the test runner expects them.
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "api",
]

# No session, auth, CSRF or message middleware: nothing here is
# cookie-authenticated, so they would only add latency.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TIME_ZONE = "UTC"
USE_TZ = True
USE_I18N = False

# Uploaded shelf photos. detect_spines() takes a path, so every upload is
# written here before the pipeline runs. Gitignored; never cleaned up
# automatically -- see the README note.
MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"

# The canonical catalog lives at the repo root, per the task spec. Loaded
# once by api/catalog.py via matching.load_catalog(); tests repoint this at a
# fixture.
CATALOG_PATH = REPO_ROOT / "catalog.csv"

# Upload ceiling enforced by the scan serializer. A phone photo is a few MB;
# anything past this is a mistake or an attack, and rejecting it early keeps
# it away from the detector.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# --------------------------------------------------------------------------
# DRF
# --------------------------------------------------------------------------

REST_FRAMEWORK = {
    # JSON only. Dropping the browsable API removes the templates and
    # staticfiles apps from this project entirely.
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    # Maps structured VLMErrors onto deliberate HTTP statuses (503/502) with
    # server-owned messages. Non-VLM exceptions are delegated to DRF's own
    # handler, so ValidationError stays a 400 and Http404 stays a 404.
    "EXCEPTION_HANDLER": "api.exception_handler.shelfie_exception_handler",
}


# --------------------------------------------------------------------------
# Logging
#
# Provider error detail is logged here and deliberately never returned in an
# HTTP response body.
# --------------------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        "api": {"handlers": ["console"], "level": "INFO"},
        "vlm": {"handlers": ["console"], "level": "INFO"},
    },
}
