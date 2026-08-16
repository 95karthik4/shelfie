"""API routes.

Exactly three. No auth, no admin, no catalog endpoint, no browsable API.
"""

from django.urls import path

from .views import LibraryListView, ScanCreateView, ScanItemConfirmView

urlpatterns = [
    path("scans/", ScanCreateView.as_view(), name="scan-create"),
    path(
        "scan-items/<int:pk>/confirm/",
        ScanItemConfirmView.as_view(),
        name="scan-item-confirm",
    ),
    path("library/", LibraryListView.as_view(), name="library-list"),
]
