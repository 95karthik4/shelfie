"""HTTP layer. Views own every database write; serializers never write.

Three endpoints, no auth:

    POST /api/scans/                        upload a shelf photo, get results
    POST /api/scan-items/<pk>/confirm/      a human accepts or corrects one
    GET  /api/library/                      what the human actually confirmed

The persistence boundary is deliberate. A scan writes Scan and ScanItem rows
describing what the models *think*. Only a confirmation writes a
ConfirmedBook -- so a high-confidence "auto" item still requires a human
action before it enters the library.
"""

import logging
import os
import re
import uuid
from pathlib import Path

from django.conf import settings
from django.db import IntegrityError, transaction
from django.http import Http404
from django.utils.text import get_valid_filename
from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ConfirmedBook, Scan, ScanItem
from .pipeline import run_scan
from .serializers import (
    ConfirmedBookSerializer,
    ConfirmSerializer,
    ScanSerializer,
    ScanUploadSerializer,
)

logger = logging.getLogger(__name__)

# Uploads land in MEDIA_ROOT/scans/.
UPLOAD_SUBDIR = "scans"

# A file suffix we are willing to put on disk: a dot plus a few alphanumerics
# and nothing else. Anything stranger is replaced rather than trusted.
SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
FALLBACK_SUFFIX = ".img"


# --------------------------------------------------------------------------
# Upload helpers
# --------------------------------------------------------------------------


def _safe_basename(name):
    """A client-supplied filename reduced to something safe to *store*.

    Two independent steps: basename() drops any directory component (so
    "../../etc/passwd" becomes "passwd"), and get_valid_filename() strips
    everything Django considers unsafe in a filename. The result is only
    ever recorded as a display string -- see _storage_name for what actually
    hits the filesystem.
    """
    base = os.path.basename(name or "")
    cleaned = get_valid_filename(base) if base else ""
    return cleaned[:255] or "upload"


def _storage_name(safe_basename):
    """The name the file is written under: a fresh UUID, never the client's.

    Only the suffix survives from the upload, and only if it looks like an
    ordinary extension. Nothing a client sends can influence the path.
    """
    suffix = Path(safe_basename).suffix.lower()
    if not SAFE_SUFFIX.match(suffix):
        suffix = FALLBACK_SUFFIX
    return "{}{}".format(uuid.uuid4().hex, suffix)


def _save_upload(photo, storage_name):
    """Stream the upload to MEDIA_ROOT/scans/. Returns the path."""
    directory = Path(settings.MEDIA_ROOT) / UPLOAD_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / storage_name
    with path.open("wb") as handle:
        for chunk in photo.chunks():
            handle.write(chunk)
    return path


def _delete_quietly(path):
    """Remove a file, ignoring failure. Used to clean up after a failed scan."""
    try:
        Path(path).unlink()
    except OSError:
        logger.warning("could not delete upload %s", path, exc_info=True)


# --------------------------------------------------------------------------
# POST /api/scans/
# --------------------------------------------------------------------------


class ScanCreateView(APIView):
    """Upload a shelf photo and get structured, per-spine results back.

    Slow by design: a CPU detector pass plus one hosted VLM call takes
    seconds. There is no queue or polling -- the client waits, and needs a
    generous timeout.
    """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = ScanUploadSerializer(data=request.data)
        # raise_exception=True -> DRF's handler -> ordinary 400 body.
        serializer.is_valid(raise_exception=True)
        photo = serializer.validated_data["photo"]

        original_filename = _safe_basename(photo.name)
        saved_path = _save_upload(photo, _storage_name(original_filename))

        try:
            result = run_scan(str(saved_path))
            # Nothing is written until the pipeline has succeeded in full,
            # and then all of it is written together.
            scan = self._persist(result, original_filename, saved_path)
        except Exception:
            # Either stage failed, so this upload is an orphan: _persist's
            # transaction has rolled back and no Scan row references the
            # file. Delete it and re-raise completely unchanged -- a
            # VLMError must reach the custom exception handler with its
            # classification intact. Catching here to clean up is not
            # catching here to decide.
            _delete_quietly(saved_path)
            raise

        # Deliberately outside the try: once persistence has succeeded the
        # Scan legitimately references this image, so a serialization
        # failure must not delete it.
        return Response(ScanSerializer(scan).data, status=status.HTTP_201_CREATED)

    @transaction.atomic
    def _persist(self, result, original_filename, saved_path):
        detector = result["detector"]
        vlm = result["vlm"]

        # No model means the hosted stage never ran (zero crops), so there is
        # no latency to report. Storing 0.0 would claim a measurement we did
        # not make. A cache hit still has a model and a real latency.
        latency_ms = vlm.get("latency_ms") if vlm.get("model") else None

        scan = Scan.objects.create(
            original_filename=original_filename,
            image_path=str(saved_path),
            detector_source=detector["source"],
            detector_quality=detector["quality"],
            used_fallback=detector["used_fallback"],
            vlm_latency_ms=latency_ms,
            vlm_requests_made=vlm.get("requests_made", 0),
            # CharField(blank=True), so absence is "" rather than NULL. The
            # serializer turns it back into null on the way out.
            vlm_model=vlm.get("model") or "",
        )

        # create() per item rather than bulk_create(): tens of rows at most,
        # and every item comes back with a real primary key, which the app
        # needs in order to confirm it.
        for item in result["items"]:
            ScanItem.objects.create(
                scan=scan,
                spine_index=item["index"],
                raw_title=item["raw_title"],
                raw_author=item["raw_author"],
                legible=item["legible"],
                matched_catalog_id=item["catalog_id"],
                matched_title=item["matched_title"],
                matched_author=item["matched_author"],
                confidence=item["confidence"],
                status=item["status"],
                reasons=item["reasons"],
            )

        return scan


# --------------------------------------------------------------------------
# POST /api/scan-items/<pk>/confirm/
# --------------------------------------------------------------------------


def _already_confirmed():
    return Response(
        {
            "error": {
                "code": "already_confirmed",
                "message": "This book has already been confirmed.",
                "retryable": False,
            }
        },
        status=status.HTTP_409_CONFLICT,
    )


class ScanItemConfirmView(APIView):
    """The human-in-the-loop boundary: where a scan result becomes a book.

    Accepts a catalog match (which may be a *different* catalog entry from
    the one the matcher proposed) or a manually typed book. The original AI
    prediction is never overwritten -- ScanItem is saved with
    update_fields=["confirmed"] so matched_catalog_id, confidence, reasons
    and the rest survive exactly as the pipeline recorded them, and the
    ConfirmedBook row carries what the human chose.
    """

    def post(self, request, pk):
        try:
            with transaction.atomic():
                try:
                    # select_for_update() states the intent to hold this row
                    # for the duration. Note it is a no-op on SQLite
                    # (has_select_for_update is False, so Django skips it
                    # silently); the OneToOne constraint below is the actual
                    # guarantee, and the IntegrityError branch is what makes
                    # that guarantee visible to the client.
                    item = ScanItem.objects.select_for_update().get(pk=pk)
                except ScanItem.DoesNotExist:
                    raise Http404("No scan item with id {}.".format(pk))

                # Body validation comes *after* the lookup, so an unknown id
                # is a 404 even when the body is also invalid. Resource
                # existence is the outer question.
                serializer = ConfirmSerializer(data=request.data)
                serializer.is_valid(raise_exception=True)
                data = serializer.validated_data

                if item.confirmed or ConfirmedBook.objects.filter(
                    scan_item=item
                ).exists():
                    return _already_confirmed()

                title, author, catalog_id = self._values(data)
                book = ConfirmedBook.objects.create(
                    scan_item=item,
                    catalog_id=catalog_id,
                    title=title,
                    author=author,
                )
                # Only this one column changes. The AI's own fields are left
                # exactly as scanned, for auditability.
                item.confirmed = True
                item.save(update_fields=["confirmed"])
        except IntegrityError:
            # A concurrent confirmation won the race. On a backend with row
            # locks this is unreachable; on SQLite it is the real defence.
            logger.info("duplicate confirmation for scan item %s", pk)
            return _already_confirmed()

        return Response(
            ConfirmedBookSerializer(book).data, status=status.HTTP_201_CREATED
        )

    @staticmethod
    def _values(data):
        """(title, author, catalog_id) for the confirmed row."""
        if data["mode"] == "catalog":
            entry = data["catalog_entry"]
            # Display strings come from the catalog row the user chose, not
            # from what the VLM read off the spine.
            return entry["title"], (entry.get("author") or None), data["catalog_id"]
        # Manual: the user's own words, and no catalog identity to claim.
        return data["title"], data.get("author"), None


# --------------------------------------------------------------------------
# GET /api/library/
# --------------------------------------------------------------------------


class LibraryListView(ListAPIView):
    """Everything the user has confirmed, newest first.

    Proves that review decisions survive: these rows exist only because a
    human confirmed them, and they outlive the scan request that produced
    them.
    """

    serializer_class = ConfirmedBookSerializer
    # Explicit order_by rather than relying on Meta.ordering, so the
    # guarantee lives at the endpoint that promises it.
    queryset = ConfirmedBook.objects.select_related("scan_item").order_by(
        "-confirmed_at"
    )
