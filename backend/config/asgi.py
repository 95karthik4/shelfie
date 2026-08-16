"""ASGI entry point.

Present for completeness; the project is served synchronously via WSGI. The
scan pipeline is CPU- and network-bound and blocks for seconds, so async
serving would not help without a task queue -- see the README.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_asgi_application()
