"""Offline API tests.

No network, no Gemini, no YOLO weights. Two mocks stand in for the expensive
stages:

  * vision.detector is patched as an attribute on the vision package, so the
    lazy `from vision import detector` inside run_scan() finds our stub and
    never imports the real module -- which would load YOLO weights on every
    test run.
  * api.pipeline.read_spines_detailed is patched to return canned reads or
    raise a structured VLMError.

Everything between those two points is real: the actual matcher, the actual
catalog.csv, the actual serializers, views, transactions and SQLite. The
expected statuses below were verified against the real catalog rather than
assumed -- "Home"/"Harlan Coben" reaches AUTO because its shared title is
disambiguated by author, while "Dune"/"Frank Herbert" is forced to REVIEW by
EDITION_AMBIGUITY between catalog ids 1 and 2.
"""

import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from PIL import Image
from rest_framework.test import APITestCase

from api.models import ConfirmedBook, Scan, ScanItem
from vlm import VLMAPIError, VLMConfigurationError, VLMResponseError


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# Verified against the committed catalog.csv, not assumed. See module docstring.
AUTO_READ = {"index": 0, "title": "Home", "author": "Harlan Coben", "legible": True}
AUTO_CATALOG_ID = "5"

REVIEW_READ = {"index": 1, "title": "Dune", "author": "Frank Herbert", "legible": True}
REVIEW_CATALOG_ID = "1"

ILLEGIBLE_READ = {"index": 2, "title": None, "author": None, "legible": False}

# A genuinely different work, used to prove a human can overrule the matcher
# rather than merely agree with it.
CORRECTION_CATALOG_ID = "6"  # Home / Toni Morrison

LIVE_METRICS = {
    "latency_ms": 4615.48,
    "cache_hit": False,
    "requests_made": 1,
    "validation_attempts": 1,
    "model": "test-model",
    "usage": None,
}

NO_CALL_METRICS = {
    "latency_ms": 0.0,
    "cache_hit": False,
    "crops_total": 0,
    "crops_valid": 0,
    "requests_made": 0,
    "validation_attempts": 0,
    "model": None,
    "usage": None,
}


def image_bytes(size=(48, 64)):
    """Real encoded JPEG bytes, so ImageField's Pillow check actually passes."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def photo(name="shelf.jpg"):
    return SimpleUploadedFile(name, image_bytes(), content_type="image/jpeg")


class ApiTestCase(APITestCase):
    """Base class: uploads go to a temp dir, never to backend/media/."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.media_root = Path(directory.name)
        settings_override = override_settings(MEDIA_ROOT=str(self.media_root))
        settings_override.enable()
        self.addCleanup(settings_override.disable)

    def stored_uploads(self):
        return sorted((self.media_root / "scans").glob("*"))

    @contextmanager
    def mocked_pipeline(
        self,
        reads=(),
        *,
        crop_count=None,
        source="yolov8n_coco",
        quality=0.72,
        used_fallback=False,
        metrics=None,
        vlm_error=None,
    ):
        """Patch the detector and the hosted stage; run everything else for real."""
        reads = list(reads)
        if crop_count is None:
            crop_count = len(reads)

        box = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 50.0, "confidence": 0.9}
        detector = SimpleNamespace(
            detect_spines=lambda path: {
                "boxes": [dict(box) for _ in range(crop_count)],
                "quality": quality,
                "used_fallback": used_fallback,
                "source": source,
            },
            # Crop contents are irrelevant: read_spines_detailed is mocked.
            crop_spines=lambda path, boxes: [object()] * crop_count,
        )

        if vlm_error is not None:
            hosted = mock.Mock(side_effect=vlm_error)
        else:
            resolved = dict(LIVE_METRICS if crop_count else NO_CALL_METRICS)
            resolved.setdefault("crops_total", crop_count)
            resolved.setdefault("crops_valid", crop_count)
            resolved.update(metrics or {})
            hosted = mock.Mock(return_value=(reads, resolved))

        with mock.patch("vision.detector", detector, create=True), mock.patch(
            "api.pipeline.read_spines_detailed", hosted
        ):
            yield hosted

    def post_scan(self, **kwargs):
        with self.mocked_pipeline(**kwargs):
            return self.client.post(
                reverse("scan-create"), {"photo": photo()}, format="multipart"
            )

    def make_scan_item(self, **overrides):
        """A persisted ScanItem, without going through the scan endpoint."""
        scan = Scan.objects.create(
            original_filename="shelf.jpg",
            detector_source="yolov8n_coco",
            detector_quality=0.7,
        )
        fields = {
            "spine_index": 0,
            "raw_title": "Dune",
            "raw_author": "Frank Herbert",
            "legible": True,
            "matched_catalog_id": REVIEW_CATALOG_ID,
            "matched_title": "Dune",
            "matched_author": "Frank Herbert",
            "confidence": 0.95,
            "status": ScanItem.Status.REVIEW,
            "reasons": ["EDITION_AMBIGUITY"],
        }
        fields.update(overrides)
        return ScanItem.objects.create(scan=scan, **fields)


# --------------------------------------------------------------------------
# POST /api/scans/ -- upload validation
# --------------------------------------------------------------------------


class ScanUploadValidationTests(ApiTestCase):
    def test_missing_photo_is_400(self):
        response = self.client.post(reverse("scan-create"), {}, format="multipart")

        self.assertEqual(response.status_code, 400)
        self.assertIn("photo", response.data)
        self.assertEqual(Scan.objects.count(), 0)

    def test_non_image_upload_is_400(self):
        # A .jpg name on bytes that are not an image: caught by decoding, not
        # by the extension.
        fake = SimpleUploadedFile(
            "shelf.jpg", b"not an image", content_type="image/jpeg"
        )

        response = self.client.post(
            reverse("scan-create"), {"photo": fake}, format="multipart"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("photo", response.data)

    @override_settings(MAX_UPLOAD_BYTES=64)
    def test_oversized_photo_is_400(self):
        response = self.client.post(
            reverse("scan-create"), {"photo": photo()}, format="multipart"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("photo", response.data)
        self.assertEqual(Scan.objects.count(), 0)

    def test_client_filename_never_becomes_a_path(self):
        traversal = SimpleUploadedFile(
            "../../../etc/passwd.jpg", image_bytes(), content_type="image/jpeg"
        )
        with self.mocked_pipeline([AUTO_READ]):
            response = self.client.post(
                reverse("scan-create"), {"photo": traversal}, format="multipart"
            )

        self.assertEqual(response.status_code, 201)
        scan = Scan.objects.get()
        # Stored name is a display string with no directory component; the
        # file on disk is a UUID.
        self.assertNotIn("/", scan.original_filename)
        self.assertEqual(len(self.stored_uploads()), 1)
        self.assertNotIn("passwd", self.stored_uploads()[0].name)


# --------------------------------------------------------------------------
# POST /api/scans/ -- success
# --------------------------------------------------------------------------


class ScanSuccessTests(ApiTestCase):
    def test_successful_scan_response_and_persistence(self):
        response = self.post_scan(reads=[AUTO_READ, REVIEW_READ, ILLEGIBLE_READ])

        self.assertEqual(response.status_code, 201)
        body = response.data

        self.assertEqual(body["scan_id"], Scan.objects.get().id)
        self.assertEqual(
            body["detector"],
            {"source": "yolov8n_coco", "quality": 0.72, "used_fallback": False},
        )
        self.assertEqual(body["vlm"]["model"], "test-model")
        self.assertEqual(body["vlm"]["requests_made"], 1)
        self.assertAlmostEqual(body["vlm"]["latency_ms"], 4615.48)

        self.assertEqual([item["index"] for item in body["items"]], [0, 1, 2])
        self.assertEqual(
            [item["status"] for item in body["items"]],
            ["auto", "review", "unmatched"],
        )

        auto, review, unmatched = body["items"]
        self.assertEqual(auto["catalog_id"], AUTO_CATALOG_ID)
        self.assertEqual(auto["matched_author"], "Harlan Coben")
        self.assertEqual(review["catalog_id"], REVIEW_CATALOG_ID)
        self.assertIn("EDITION_AMBIGUITY", review["reasons"])
        self.assertIsNone(unmatched["catalog_id"])
        self.assertIn("NOT_LEGIBLE", unmatched["reasons"])
        self.assertFalse(any(item["confirmed"] for item in body["items"]))

        self.assertEqual(ScanItem.objects.count(), 3)
        self.assertEqual(
            list(ScanItem.objects.values_list("spine_index", flat=True)), [0, 1, 2]
        )

        # Every item must carry a real persisted id: it is the address the
        # Expo client posts to as /api/scan-items/<id>/confirm/. Without it
        # the review step has nothing to act on.
        self.assertTrue(all(isinstance(item["id"], int) for item in body["items"]))
        self.assertEqual(
            sorted(item["id"] for item in body["items"]),
            sorted(ScanItem.objects.values_list("id", flat=True)),
        )

    def test_auto_items_are_not_silently_added_to_the_library(self):
        """The persistence boundary: high confidence still needs a human."""
        response = self.post_scan(reads=[AUTO_READ])

        self.assertEqual(response.data["items"][0]["status"], "auto")
        self.assertEqual(ConfirmedBook.objects.count(), 0)
        self.assertFalse(ScanItem.objects.get().confirmed)

    def test_zero_detected_books_is_a_successful_empty_scan(self):
        with self.mocked_pipeline(reads=[], crop_count=0) as hosted:
            response = self.client.post(
                reverse("scan-create"), {"photo": photo()}, format="multipart"
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["items"], [])
        # The hosted stage is still called, with an empty batch -- and it is
        # read_spines_detailed's own contract that this makes no API request.
        hosted.assert_called_once_with([])
        self.assertIsNone(response.data["vlm"]["model"])
        self.assertIsNone(response.data["vlm"]["latency_ms"])
        self.assertEqual(response.data["vlm"]["requests_made"], 0)
        self.assertEqual(Scan.objects.count(), 1)

    def test_fallback_detector_metadata_is_reported(self):
        response = self.post_scan(
            reads=[AUTO_READ],
            source="opencv_fallback",
            quality=0.31,
            used_fallback=True,
        )

        self.assertEqual(response.data["detector"]["source"], "opencv_fallback")
        self.assertTrue(response.data["detector"]["used_fallback"])
        self.assertTrue(Scan.objects.get().used_fallback)

    def test_invalid_crop_read_is_preserved_as_an_item(self):
        read = {
            "index": 0,
            "title": None,
            "author": None,
            "legible": False,
            "error": "invalid_crop",
        }

        response = self.post_scan(reads=[read])

        item = response.data["items"][0]
        self.assertEqual(item["status"], "unmatched")
        self.assertIn("INVALID_CROP", item["reasons"])


# --------------------------------------------------------------------------
# POST /api/scans/ -- graceful failure
# --------------------------------------------------------------------------


class ScanFailureTests(ApiTestCase):
    def assert_no_partial_state(self):
        self.assertEqual(Scan.objects.count(), 0)
        self.assertEqual(ScanItem.objects.count(), 0)
        self.assertEqual(self.stored_uploads(), [])

    def test_configuration_error_is_503(self):
        response = self.post_scan(
            reads=[AUTO_READ], vlm_error=VLMConfigurationError("GEMINI_API_KEY not set")
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["error"]["code"], "configuration")
        self.assertFalse(response.data["error"]["retryable"])
        self.assertNotIn("Retry-After", response)
        self.assert_no_partial_state()

    def test_retryable_api_error_is_503_with_retry_after(self):
        response = self.post_scan(
            reads=[AUTO_READ],
            vlm_error=VLMAPIError("429 exhausted", status_code=429, retryable=True),
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response["Retry-After"], "30")
        self.assertTrue(response.data["error"]["retryable"])
        self.assert_no_partial_state()

    def test_permanent_api_error_is_502(self):
        response = self.post_scan(
            reads=[AUTO_READ],
            vlm_error=VLMAPIError("401 bad key", status_code=401, retryable=False),
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.data["error"]["code"], "api_failure")
        self.assert_no_partial_state()

    def test_malformed_response_error_is_502(self):
        response = self.post_scan(
            reads=[AUTO_READ], vlm_error=VLMResponseError("index set mismatch")
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.data["error"]["code"], "invalid_response")
        self.assert_no_partial_state()

    def test_provider_detail_never_reaches_the_client(self):
        response = self.post_scan(
            reads=[AUTO_READ],
            vlm_error=VLMAPIError(
                "gemini says: quota exceeded for project 12345",
                status_code=429,
                retryable=True,
            ),
        )

        body = str(response.data)
        self.assertNotIn("gemini", body.lower())
        self.assertNotIn("12345", body)
        self.assertNotIn("quota", body.lower())

    def test_upload_is_cleaned_up_when_persistence_fails(self):
        """The cleanup boundary covers persistence, not only the pipeline.

        The AI stages succeed here and _persist() runs for real, creating the
        Scan row before the first ScanItem write blows up. That exercises two
        guarantees at once: transaction.atomic() rolls the already-created
        Scan back, and ScanCreateView.post() deletes the now-orphaned upload.

        RuntimeError is a stand-in for "something unexpected broke in the
        write path". It is not API behaviour -- the test client re-raises it
        rather than converting it to a response, which is what makes the
        cleanup boundary observable on its own.
        """
        with self.mocked_pipeline([AUTO_READ]), mock.patch(
            "api.views.ScanItem.objects.create",
            side_effect=RuntimeError("write failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.client.post(
                    reverse("scan-create"), {"photo": photo()}, format="multipart"
                )

        self.assertEqual(Scan.objects.count(), 0)
        self.assertEqual(ScanItem.objects.count(), 0)
        self.assertEqual(self.stored_uploads(), [])


# --------------------------------------------------------------------------
# POST /api/scan-items/<pk>/confirm/
# --------------------------------------------------------------------------


class ConfirmTests(ApiTestCase):
    def confirm(self, pk, payload):
        return self.client.post(
            reverse("scan-item-confirm", args=[pk]), payload, format="json"
        )

    def test_confirming_the_matched_catalog_entry_persists(self):
        item = self.make_scan_item()

        response = self.confirm(item.pk, {"catalog_id": REVIEW_CATALOG_ID})

        self.assertEqual(response.status_code, 201)
        book = ConfirmedBook.objects.get()
        self.assertEqual(book.catalog_id, REVIEW_CATALOG_ID)
        self.assertEqual(book.title, "Dune")
        self.assertEqual(response.data["scan_item_id"], item.pk)
        item.refresh_from_db()
        self.assertTrue(item.confirmed)

    def test_human_may_correct_the_match_to_a_different_catalog_entry(self):
        item = self.make_scan_item()

        response = self.confirm(item.pk, {"catalog_id": CORRECTION_CATALOG_ID})

        self.assertEqual(response.status_code, 201)
        book = ConfirmedBook.objects.get()
        self.assertEqual(book.catalog_id, CORRECTION_CATALOG_ID)
        self.assertEqual(book.author, "Toni Morrison")

        # The original prediction is preserved for auditability.
        item.refresh_from_db()
        self.assertEqual(item.matched_catalog_id, REVIEW_CATALOG_ID)
        self.assertEqual(item.matched_author, "Frank Herbert")
        self.assertEqual(item.confidence, 0.95)
        self.assertEqual(item.reasons, ["EDITION_AMBIGUITY"])

    def test_manual_confirmation_of_an_unmatched_item(self):
        item = self.make_scan_item(
            status=ScanItem.Status.UNMATCHED,
            matched_catalog_id=None,
            matched_title=None,
            matched_author=None,
            confidence=0.0,
            reasons=["NOT_LEGIBLE"],
            legible=False,
            raw_title=None,
            raw_author=None,
        )

        response = self.confirm(
            item.pk, {"title": "A Book Not In The Catalog", "author": "Someone Else"}
        )

        self.assertEqual(response.status_code, 201)
        book = ConfirmedBook.objects.get()
        self.assertIsNone(book.catalog_id)
        self.assertEqual(book.title, "A Book Not In The Catalog")
        self.assertEqual(book.author, "Someone Else")

    def test_manual_confirmation_without_author(self):
        item = self.make_scan_item()

        response = self.confirm(item.pk, {"title": "Title Only"})

        self.assertEqual(response.status_code, 201)
        self.assertIsNone(ConfirmedBook.objects.get().author)

    def test_auto_items_also_require_an_explicit_confirmation(self):
        item = self.make_scan_item(
            status=ScanItem.Status.AUTO,
            matched_catalog_id=AUTO_CATALOG_ID,
            matched_title="Home",
            matched_author="Harlan Coben",
            confidence=1.0,
            reasons=[],
        )
        self.assertEqual(ConfirmedBook.objects.count(), 0)

        response = self.confirm(item.pk, {"catalog_id": AUTO_CATALOG_ID})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(ConfirmedBook.objects.count(), 1)

    def test_ambiguous_and_invalid_payloads_are_400(self):
        item = self.make_scan_item()
        payloads = [
            {"catalog_id": REVIEW_CATALOG_ID, "title": "Dune"},
            {"catalog_id": REVIEW_CATALOG_ID, "author": "Frank Herbert"},
            {"author": "Frank Herbert"},
            {},
            {"catalog_id": ""},
            {"title": ""},
            {"catalog_id": "999999"},
        ]

        for payload in payloads:
            with self.subTest(payload=payload):
                response = self.confirm(item.pk, payload)
                self.assertEqual(response.status_code, 400)

        self.assertEqual(ConfirmedBook.objects.count(), 0)
        item.refresh_from_db()
        self.assertFalse(item.confirmed)

    def test_unknown_scan_item_is_404(self):
        response = self.confirm(999999, {"catalog_id": REVIEW_CATALOG_ID})

        self.assertEqual(response.status_code, 404)

    def test_unknown_scan_item_is_404_even_with_an_invalid_body(self):
        """Resource existence is the outer question, not body validity."""
        response = self.confirm(999999, {"catalog_id": "1", "title": "both"})

        self.assertEqual(response.status_code, 404)

    def test_double_confirmation_is_409(self):
        item = self.make_scan_item()
        first = self.confirm(item.pk, {"catalog_id": REVIEW_CATALOG_ID})
        self.assertEqual(first.status_code, 201)

        second = self.confirm(item.pk, {"catalog_id": CORRECTION_CATALOG_ID})

        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.data["error"]["code"], "already_confirmed")
        # The first confirmation stands, unmodified.
        self.assertEqual(ConfirmedBook.objects.count(), 1)
        self.assertEqual(ConfirmedBook.objects.get().catalog_id, REVIEW_CATALOG_ID)


# --------------------------------------------------------------------------
# GET /api/library/
# --------------------------------------------------------------------------


class LibraryTests(ApiTestCase):
    def test_empty_library(self):
        response = self.client.get(reverse("library-list"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, [])

    def test_library_contains_only_confirmed_books_newest_first(self):
        first = self.make_scan_item()
        second = self.make_scan_item(spine_index=1)
        unconfirmed = self.make_scan_item(spine_index=2)

        self.client.post(
            reverse("scan-item-confirm", args=[first.pk]),
            {"catalog_id": REVIEW_CATALOG_ID},
            format="json",
        )
        self.client.post(
            reverse("scan-item-confirm", args=[second.pk]),
            {"title": "Manually Added"},
            format="json",
        )

        response = self.client.get(reverse("library-list"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)
        # Newest first.
        self.assertEqual(response.data[0]["title"], "Manually Added")
        self.assertEqual(response.data[1]["title"], "Dune")
        self.assertNotIn(unconfirmed.pk, [row["scan_item_id"] for row in response.data])

    def test_confirmations_survive_later_requests(self):
        item = self.make_scan_item()
        self.client.post(
            reverse("scan-item-confirm", args=[item.pk]),
            {"catalog_id": REVIEW_CATALOG_ID},
            format="json",
        )

        # A second, unrelated scan must not disturb the library.
        self.post_scan(reads=[AUTO_READ])

        response = self.client.get(reverse("library-list"))
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["catalog_id"], REVIEW_CATALOG_ID)
