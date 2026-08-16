"""App configuration.

Note what is NOT here: no ready() hook importing the pipeline. vision/
loads YOLO weights at module import, which costs seconds, and doing that at
app-registry time would slow every manage.py invocation, every autoreload
and every test run. api/pipeline.py imports vision lazily instead.
"""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
