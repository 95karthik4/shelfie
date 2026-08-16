"""Root URL configuration.

Everything lives under /api/. There is no admin, no browsable API and no
static/media serving -- the Expo client is the only consumer.
"""

from django.urls import include, path

urlpatterns = [
    path("api/", include("api.urls")),
]
