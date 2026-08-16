"""Scan orchestration: photo in, per-spine results out.

The one place the three stages are wired together:

    detect_spines -> crop_spines -> read_spines_detailed -> match

Performs no database writes: it returns plain dicts and the view persists
them in a single transaction after this returns, so a VLM failure part-way
leaves no half-written scan. It does import ScanItem, but only for its
Status enum -- so that the status strings on the wire and the status choices
in the database come from one definition and cannot drift.

Errors are deliberately not caught here. detect_spines/crop_spines never
raise (documented in vision/detector.py); read_spines_detailed raises
structured VLMErrors, which the DRF exception handler maps onto HTTP
statuses in Group 4. Swallowing them here would be the "silently dropped"
failure the task explicitly warns against.
"""

import logging

from matching import match
from matching.matcher import (
    STATUS_AUTO_READY,
    STATUS_REVIEW,
    STATUS_UNMATCHED,
)
from vlm import CODE_INVALID_CROP, read_spines_detailed

from .catalog import build_index, get_catalog, normalized_id
from .models import ScanItem

logger = logging.getLogger(__name__)


# The matcher speaks its own vocabulary; the API contract speaks the
# ScanItem.Status values. Mapped in exactly one place, and sourced from the
# model's own enum so the wire format and the DB choices cannot drift apart.
STATUS_MAP = {
    STATUS_AUTO_READY: ScanItem.Status.AUTO.value,
    STATUS_REVIEW: ScanItem.Status.REVIEW.value,
    STATUS_UNMATCHED: ScanItem.Status.UNMATCHED.value,
}

# Added to the matcher's own reasons when the crop never reached the VLM.
# The matcher cannot know this -- it only sees legible=False.
REASON_INVALID_CROP = "INVALID_CROP"


def run_scan(image_path, catalog=None):
    """Run the full pipeline over one image.

    Args:
        image_path: path to the uploaded photo, on disk.
        catalog: optional catalog override; defaults to the cached real one.

    Returns:
        {"detector": {...}, "vlm": {...}, "items": [...]} -- see _build_item
        for the per-item shape. Item ids are absent; the view assigns them
        when it persists.

    Raises:
        VLMError subclasses, propagated from the hosted stage.
    """
    # Imported here, not at module scope: vision/detector.py loads YOLO
    # weights at import, which would otherwise cost seconds on every
    # manage.py command, autoreload and test run.
    from vision import detector

    detection = detector.detect_spines(image_path)
    boxes = detection["boxes"]

    # crop_spines skips any box that clamps to zero area, so len(crops) can
    # be < len(boxes). Indices below are positions in the crop list -- the
    # same list the VLM was shown -- which is what keeps spine_index
    # meaningful end to end.
    crops = detector.crop_spines(image_path, boxes)

    # No crops means no hosted call at all: read_spines_detailed returns
    # ([], metrics) without constructing a client. Zero books found is a
    # successful scan, not an error.
    reads, vlm_metrics = read_spines_detailed(crops)

    # One resolution point, one index built from whatever catalog that
    # produced. match() and the display lookup therefore always read the
    # same data -- there is no branch in which an injected catalog could be
    # matched against while display strings came from the cached real one.
    if catalog is None:
        catalog = get_catalog()
    index = build_index(catalog)

    items = [_build_item(read, catalog, index) for read in reads]

    logger.info(
        "scan complete: %d boxes, %d crops, %d items, detector=%s, cache_hit=%s",
        len(boxes),
        len(crops),
        len(items),
        detection["source"],
        vlm_metrics.get("cache_hit"),
    )

    return {
        "detector": {
            "source": detection["source"],
            "quality": detection["quality"],
            "used_fallback": detection["used_fallback"],
        },
        # Only these three go to the client. Token usage and cache state stay
        # in the log: useful to us, not to the app, and not something to
        # expose about the provider.
        "vlm": {
            "latency_ms": vlm_metrics.get("latency_ms"),
            "requests_made": vlm_metrics.get("requests_made", 0),
            "model": vlm_metrics.get("model"),
        },
        "items": items,
    }


def _build_item(read, catalog, index):
    """One VLM read -> one persisted-shape item dict.

    Every read becomes an item, including illegible spines and crops that
    never reached the provider. Nothing is filtered out: a spine the pipeline
    could not resolve is exactly what the review step exists for.
    """
    raw_title = read.get("title")
    raw_author = read.get("author")
    legible = bool(read.get("legible"))

    # Called even when legible is False: match() short-circuits to a
    # structured UNMATCHED/NOT_LEGIBLE result before scoring anything, so
    # the "unreadable spine" rule stays in the matcher where it is tested,
    # rather than being duplicated here.
    result = match(raw_title, raw_author, catalog, legible=legible)

    reasons = list(result["reasons"])
    if read.get("error") == CODE_INVALID_CROP:
        reasons.append(REASON_INVALID_CROP)

    status = STATUS_MAP.get(result["status"])
    if status is None:
        # Unknown matcher status: route to a human rather than guess. The
        # one direction this must never fail in is towards auto-accept.
        logger.warning(
            "unmapped matcher status %r; routing to review", result["status"]
        )
        status = ScanItem.Status.REVIEW.value

    entry = index.get(normalized_id(result["catalog_id"]))

    return {
        "index": read["index"],
        "raw_title": raw_title,
        "raw_author": raw_author,
        "legible": legible,
        "catalog_id": result["catalog_id"],
        "matched_title": entry.get("title") if entry else None,
        "matched_author": entry.get("author") if entry else None,
        "confidence": result["confidence"],
        "status": status,
        "reasons": reasons,
    }
