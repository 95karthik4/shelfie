"""Persistence for scans, their per-spine results, and confirmed books.

Three tables, no catalog table: catalog.csv is the canonical catalog and is
loaded in-process by matching.load_catalog(), so catalog identifiers are
stored here as plain strings rather than foreign keys. Mirroring the CSV into
the database would create two sources of truth for the same data.

Nothing in this module imports vision, vlm or matching. Models describe
storage; the pipeline that fills them lives in api/pipeline.py.

The human-in-the-loop boundary is expressed structurally: ScanItem.status may
be "auto", but a ConfirmedBook row only ever exists because a confirmation
request created it. Nothing in the scan path writes to that table.
"""

from django.db import models


class Scan(models.Model):
    """One uploaded shelf photo and the measurements from processing it."""

    created_at = models.DateTimeField(auto_now_add=True)
    original_filename = models.CharField(max_length=255, blank=True)
    # Kept so a scan can be re-run during a demo without re-uploading.
    image_path = models.CharField(max_length=500, blank=True)

    # Local detector stage: which path produced the boxes, how the quality
    # gate scored, and whether the OpenCV fallback took over.
    detector_source = models.CharField(max_length=32)
    detector_quality = models.FloatField(default=0.0)
    used_fallback = models.BooleanField(default=False)

    # Hosted VLM stage. Nullable because a scan with zero detected boxes
    # never calls the provider -- "no request made" is not a latency of 0.
    vlm_latency_ms = models.FloatField(null=True, blank=True)
    vlm_model = models.CharField(max_length=64, blank=True)
    vlm_requests_made = models.IntegerField(default=0)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return "Scan {} ({})".format(self.pk, self.original_filename or "unnamed")


class ScanItem(models.Model):
    """One detected spine: what was read off it, and what it matched."""

    class Status(models.TextChoices):
        AUTO = "auto", "Auto"
        REVIEW = "review", "Review"
        UNMATCHED = "unmatched", "Unmatched"

    scan = models.ForeignKey(Scan, related_name="items", on_delete=models.CASCADE)
    # Position in the crop list handed to the VLM. Preserved end to end so a
    # result can always be traced back to the spine it came from.
    spine_index = models.IntegerField()

    # What the VLM read. Null when the spine was illegible or the crop was
    # unusable -- deliberately distinct from an empty string.
    raw_title = models.CharField(max_length=300, null=True, blank=True)
    raw_author = models.CharField(max_length=300, null=True, blank=True)
    legible = models.BooleanField(default=False)

    # What the matcher returned. Plain strings, not FKs -- see module docstring.
    matched_catalog_id = models.CharField(max_length=32, null=True, blank=True)
    matched_title = models.CharField(max_length=300, null=True, blank=True)
    matched_author = models.CharField(max_length=300, null=True, blank=True)
    confidence = models.FloatField(default=0.0)

    # Defaults fail closed: an item whose status was never set explicitly is
    # unmatched, never auto. A bug should route a book to the human, not past
    # them.
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.UNMATCHED
    )
    # Matcher reason codes, e.g. ["DIFFERENT_WORK_AMBIGUITY"]. The review UI
    # explains itself from these, so they are stored verbatim.
    reasons = models.JSONField(default=list)

    # Mirrors "a ConfirmedBook exists for this item". Denormalised so the
    # review list can filter without a join.
    confirmed = models.BooleanField(default=False)

    class Meta:
        ordering = ("spine_index",)
        constraints = [
            models.UniqueConstraint(
                fields=("scan", "spine_index"), name="unique_spine_index_per_scan"
            )
        ]

    def __str__(self):
        return "ScanItem {} (spine {}, {})".format(
            self.pk, self.spine_index, self.status
        )


class ConfirmedBook(models.Model):
    """A book the user actually confirmed. The user's library is this table.

    Created only by the confirmation endpoint. A high-confidence match is
    reported as "auto" but is not written here until a human says so, which
    is what makes the review step a product feature rather than a formality.

    OneToOne, so confirming the same item twice is a database-level conflict
    (the view answers 409) rather than a silent duplicate.
    """

    scan_item = models.OneToOneField(
        ScanItem, related_name="confirmation", on_delete=models.CASCADE
    )
    # Null when the user typed the book in manually instead of accepting a
    # catalog match -- an unmatched spine still belongs in the library.
    catalog_id = models.CharField(max_length=32, null=True, blank=True)
    title = models.CharField(max_length=300)
    author = models.CharField(max_length=300, null=True, blank=True)
    confirmed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-confirmed_at",)

    def __str__(self):
        return "{} - {}".format(self.title, self.author or "unknown author")
