"""Request and response shapes for the API.

Serializers here validate and present; they never write. Every database
mutation happens in a view, inside an explicit transaction -- confirmation in
particular touches two tables (ConfirmedBook insert, ScanItem.confirmed
update) and must not be half-applied.

Everything a model or the AI produced is read-only on the way out. The only
client-supplied data this module accepts is an uploaded photo and a
confirmation choice.
"""

from django.conf import settings
from rest_framework import serializers

from .catalog import get_entry, normalized_id
from .models import ConfirmedBook, Scan, ScanItem


class ScanUploadSerializer(serializers.Serializer):
    """The multipart body of POST /api/scans/.

    ImageField (not FileField) so the upload is verified by actually decoding
    it with Pillow -- a .jpg extension on a text file fails here rather than
    somewhere inside the detector.

    Nothing is saved here. The view owns where the file lands and how long it
    lives; a serializer that wrote to disk would make validation failures
    leave orphans behind.
    """

    photo = serializers.ImageField(required=True)

    def validate_photo(self, value):
        # Note the ordering: DRF has already decoded the image by the time
        # this runs, so the size check is a guard against storing something
        # huge, not against parsing it. Django's own upload limits are the
        # first line of defence.
        if value.size > settings.MAX_UPLOAD_BYTES:
            raise serializers.ValidationError(
                "Photo is {:.1f} MB; the limit is {:.0f} MB.".format(
                    value.size / (1024 * 1024),
                    settings.MAX_UPLOAD_BYTES / (1024 * 1024),
                )
            )
        return value


class ScanItemSerializer(serializers.ModelSerializer):
    """One detected spine, as the app sees it. Output only.

    Two fields are renamed on the way out: spine_index -> index and
    matched_catalog_id -> catalog_id. The database names say where the value
    came from; the API names say what it means to the client.
    """

    index = serializers.IntegerField(source="spine_index", read_only=True)
    catalog_id = serializers.CharField(
        source="matched_catalog_id", read_only=True, allow_null=True
    )

    class Meta:
        model = ScanItem
        fields = (
            "id",
            "index",
            "raw_title",
            "raw_author",
            "legible",
            "catalog_id",
            "matched_title",
            "matched_author",
            "confidence",
            "status",
            "reasons",
            "confirmed",
        )
        # Every field is model- or AI-derived. A client cannot edit what the
        # pipeline decided; it can only confirm or correct it through the
        # confirmation endpoint.
        read_only_fields = fields


class ScanSerializer(serializers.ModelSerializer):
    """A whole scan and its items. Output only.

    detector and vlm are nested for the client's benefit while staying flat
    columns in the database -- the model is shaped for storage, not to mirror
    this JSON. Two SerializerMethodFields are the cheapest way to bridge that
    without a nested model or a second table.
    """

    scan_id = serializers.IntegerField(source="id", read_only=True)
    detector = serializers.SerializerMethodField()
    vlm = serializers.SerializerMethodField()
    items = ScanItemSerializer(many=True, read_only=True)

    class Meta:
        model = Scan
        fields = ("scan_id", "detector", "vlm", "items")
        read_only_fields = fields

    def get_detector(self, obj):
        return {
            "source": obj.detector_source,
            "quality": obj.detector_quality,
            "used_fallback": obj.used_fallback,
        }

    def get_vlm(self, obj):
        # latency_ms and model stay null when no hosted call was made (zero
        # crops). "No request" is not a latency of zero.
        return {
            "latency_ms": obj.vlm_latency_ms,
            "requests_made": obj.vlm_requests_made,
            "model": obj.vlm_model or None,
        }


class ConfirmSerializer(serializers.Serializer):
    """The body of POST /api/scan-items/<id>/confirm/. Input only.

    Exactly one of two modes, never both:

        catalog mode   {"catalog_id": "42"}
        manual mode    {"title": "Dune", "author": "Frank Herbert"}

    The submitted catalog_id deliberately does NOT have to equal the item's
    matched_catalog_id. Correcting the model to a *different* catalog entry
    is the entire point of the review step -- rejecting a mismatch would make
    the human able to agree with the model but never to overrule it.

    On success, validated_data carries:
        mode           "catalog" or "manual"
        catalog_id     canonical string id, or absent in manual mode
        catalog_entry  the resolved catalog row, in catalog mode
        title/author   in manual mode
    """

    # allow_blank stays False on both identity fields, so "" is rejected by
    # field validation before the mode logic below ever runs.
    catalog_id = serializers.CharField(required=False)
    title = serializers.CharField(required=False)
    author = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    def validate_title(self, value):
        # DRF's CharField trims whitespace by default, so allow_blank=False
        # already rejects both "" and "   " before this runs. Keeping the
        # strip here is defence-in-depth: it makes the guarantee explicit
        # rather than dependent on a DRF default that could be configured
        # away.
        title = value.strip()
        if not title:
            raise serializers.ValidationError("Title cannot be blank.")
        return title

    def validate_author(self, value):
        if value is None:
            return None
        author = value.strip()
        return author or None

    def validate(self, attrs):
        has_catalog = "catalog_id" in attrs
        has_title = "title" in attrs
        # Key presence, not truthiness: sending author alongside catalog_id
        # is a contradiction whatever its value.
        has_author = "author" in attrs

        if has_catalog and (has_title or has_author):
            raise serializers.ValidationError(
                "Send either catalog_id (to confirm a catalog match) or "
                "title/author (to add a book manually), not both."
            )

        if not has_catalog and not has_title:
            if has_author:
                raise serializers.ValidationError(
                    {"title": "A manual confirmation needs a title."}
                )
            raise serializers.ValidationError(
                "Send either catalog_id or title to confirm this book."
            )

        if has_catalog:
            entry = get_entry(attrs["catalog_id"])
            if entry is None:
                raise serializers.ValidationError(
                    {"catalog_id": "No catalog entry with this id."}
                )
            # Canonicalised so ConfirmedBook.catalog_id is stored in one
            # consistent form regardless of whether the client sent 42 or
            # "42" or " 42 ".
            attrs["catalog_id"] = normalized_id(entry["catalog_id"])
            attrs["catalog_entry"] = entry
            attrs["mode"] = "catalog"
        else:
            attrs["mode"] = "manual"
            attrs.setdefault("author", None)

        return attrs


class ConfirmedBookSerializer(serializers.ModelSerializer):
    """A persisted confirmation -- one row of the user's library. Output only."""

    scan_item_id = serializers.IntegerField(read_only=True)

    class Meta:
        model = ConfirmedBook
        fields = (
            "id",
            "scan_item_id",
            "catalog_id",
            "title",
            "author",
            "confirmed_at",
        )
        read_only_fields = fields
