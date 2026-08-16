"""Spine detection: YOLOv8n-COCO ("book" class) primary path, OpenCV fallback
on low quality.

Per CLAUDE.md's frozen architecture: local detector -> quality gate ->
(OpenCV vertical-edge fallback if the gate fails) -> spine crops. This module
owns all three stages. It is pure Python + ultralytics + opencv-python, zero
Django imports, so it can be exercised and benchmarked standalone.

YOLOv8n-COCO, filtered to its stock "book" class, is the runtime detector --
chosen over the open-vocabulary YOLO-World ("book spine" prompt) after
benchmarking both on the committed test photos: YOLOv8n found far more boxes
per image at comparable CPU latency, and visual inspection of its overlays
showed those boxes were predominantly individual spines rather than
whole-shelf blobs. YOLO-World remains in benchmark.py as the historical
comparison point, but no longer runs in the request path.

Box schema used throughout this module (both YOLOv8n and the OpenCV
fallback produce this same shape, so callers never need to know which path
ran):

    {"x1": float, "y1": float, "x2": float, "y2": float, "confidence": float}

Coordinates are pixel offsets in the original image, top-left origin.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Tunable constants. Everything tunable lives here, nothing is inlined below.
# --------------------------------------------------------------------------

# Smallest YOLOv8 checkpoint ultralytics ships -- chosen for CPU latency.
# Off-the-shelf weights only, per CLAUDE.md; never fine-tuned. Closed-set
# COCO detector, filtered below to just its "book" class.
YOLOV8N_WEIGHTS = "yolov8n.pt"
COCO_BOOK_CLASS_NAME = "book"

DEVICE = "cpu"

# ultralytics' own default confidence threshold for this checkpoint.
YOLOV8N_CONF_THRESHOLD = 0.25

# --- Quality gate: 0.40*mean_confidence + 0.30*plausible_box_ratio + 0.30*coverage_score ---
MEAN_CONFIDENCE_WEIGHT = 0.40
PLAUSIBLE_BOX_RATIO_WEIGHT = 0.30
COVERAGE_WEIGHT = 0.30
QUALITY_THRESHOLD = 0.55

# The weighted quality_score() above can still score well on a shelf where
# only a handful of spines were found (e.g. one confident, well-formed box),
# because the score is an average/ratio, not a count. Requiring a minimum
# absolute number of detections closes that failure-open case: a scored-high
# but obviously-incomplete shelf still falls through to the OpenCV fallback.
MIN_PLAUSIBLE_DETECTIONS = 4

# A box counts as "plausible" (a real spine, not noise or a misfire spanning
# half the shelf) when its area and elongation fall in these bounds.
MIN_BOX_AREA_RATIO = 0.0008  # smaller than this is almost certainly noise
MAX_BOX_AREA_RATIO = 0.35  # bigger than this is not a single spine
MIN_ASPECT_RATIO = 1.15  # long-side / short-side; spines are elongated

SOURCE_YOLOV8N = "yolov8n_coco"
SOURCE_OPENCV_FALLBACK = "opencv_fallback"
SOURCE_NONE = "none"

# --- OpenCV vertical-edge fallback ---
# Fallback boxes carry no model confidence, so every fallback box gets this
# fixed placeholder -- it exists only so boxes from either path share one
# schema.
FALLBACK_CONFIDENCE = 0.50
# Column kept as a candidate spine boundary once its vertical-gradient energy
# reaches this fraction of the strongest column in the image.
FALLBACK_EDGE_THRESHOLD = 0.35
# Consecutive strong columns closer than this (px) are one edge, not two.
FALLBACK_EDGE_GROUP_GAP = 3
FALLBACK_MIN_SPINE_WIDTH_PX = 12
FALLBACK_MAX_SPINE_WIDTH_RATIO = 0.5  # wider than half the image isn't a spine


# --------------------------------------------------------------------------
# Model loading. Loaded once at import, reused for every call/image.
# --------------------------------------------------------------------------

_DETECTOR_MODEL = YOLO(YOLOV8N_WEIGHTS)


# --------------------------------------------------------------------------
# YOLOv8n-COCO detection, filtered to the "book" class
# --------------------------------------------------------------------------


def _read_image(image_path):
    """cv2.imread wrapper: never raises, returns None for anything unreadable."""
    try:
        return cv2.imread(str(image_path))
    except Exception:
        return None


def detect(image_path):
    """Run YOLOv8n-COCO over image_path, return spine boxes for its "book"
    class only.

    Returns [] for a missing/corrupt/unreadable image or zero "book"
    detections -- never raises.
    """
    image = _read_image(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    try:
        results = _DETECTOR_MODEL.predict(
            source=image,
            conf=YOLOV8N_CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
    except Exception:
        return []

    boxes = []
    for result in results:
        names = result.names
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None:
            continue
        for box in result_boxes:
            class_id = int(box.cls[0])
            if names.get(class_id) != COCO_BOOK_CLASS_NAME:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append(
                {
                    "x1": max(0.0, x1),
                    "y1": max(0.0, y1),
                    "x2": min(float(width), x2),
                    "y2": min(float(height), y2),
                    "confidence": float(box.conf[0]),
                }
            )
    return boxes


# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------


def _is_plausible_box(box, image_area):
    box_width = box["x2"] - box["x1"]
    box_height = box["y2"] - box["y1"]
    if box_width <= 0 or box_height <= 0 or image_area <= 0:
        return False

    area_ratio = (box_width * box_height) / image_area
    if not (MIN_BOX_AREA_RATIO <= area_ratio <= MAX_BOX_AREA_RATIO):
        return False

    long_side, short_side = max(box_width, box_height), min(box_width, box_height)
    if short_side <= 0:
        return False
    return (long_side / short_side) >= MIN_ASPECT_RATIO


def _coverage_score(boxes, image_width):
    """Fraction of the shelf's width spanned by detected spines.

    Uses horizontal coverage rather than area coverage: a bookshelf photo is
    a wide span of narrow spines, so "did we find spines across the shelf"
    is a better read of completeness than raw box area, which is skewed by
    a few tall or partially-cropped spines.
    """
    if not boxes or image_width <= 0:
        return 0.0
    intervals = sorted((box["x1"], box["x2"]) for box in boxes)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    covered = sum(end - start for start, end in merged)
    return min(1.0, covered / image_width)


def quality_score(boxes, image_size):
    """0.40*mean_confidence + 0.30*plausible_box_ratio + 0.30*coverage_score.

    image_size is (width, height) in pixels. Returns 0.0 for no boxes or a
    degenerate image size -- never raises.
    """
    if not boxes:
        return 0.0
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0.0

    image_area = width * height
    mean_confidence = sum(box["confidence"] for box in boxes) / len(boxes)
    plausible_box_ratio = sum(
        1 for box in boxes if _is_plausible_box(box, image_area)
    ) / len(boxes)
    coverage_score = _coverage_score(boxes, width)

    return (
        MEAN_CONFIDENCE_WEIGHT * mean_confidence
        + PLAUSIBLE_BOX_RATIO_WEIGHT * plausible_box_ratio
        + COVERAGE_WEIGHT * coverage_score
    )


# --------------------------------------------------------------------------
# OpenCV vertical-edge fallback
# --------------------------------------------------------------------------


def _group_edge_columns(columns):
    """Collapse a run of adjacent strong-edge columns into one boundary (its
    midpoint) per group, so a thick edge doesn't produce duplicate cuts."""
    groups = []
    group_start = columns[0]
    prev = columns[0]
    for col in columns[1:]:
        if col - prev > FALLBACK_EDGE_GROUP_GAP:
            groups.append((group_start + prev) // 2)
            group_start = col
        prev = col
    groups.append((group_start + prev) // 2)
    return groups


def _opencv_fallback_boxes(image):
    """Segment spines by vertical edges: a bookshelf photo is dominated by
    the near-vertical boundaries between adjacent spines, so a strong
    vertical-gradient column is a plausible spine boundary. Consecutive
    boundaries become full-height candidate spine boxes.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    column_energy = np.abs(sobel_x).sum(axis=0)

    peak = float(column_energy.max())
    if peak <= 0:
        return []

    normalized = column_energy / peak
    edge_columns = np.where(normalized >= FALLBACK_EDGE_THRESHOLD)[0]
    if edge_columns.size == 0:
        return []

    edges = _group_edge_columns(edge_columns.tolist())
    edges = sorted(set([0] + edges + [width]))

    boxes = []
    for x1, x2 in zip(edges[:-1], edges[1:]):
        spine_width = x2 - x1
        if spine_width < FALLBACK_MIN_SPINE_WIDTH_PX:
            continue
        if spine_width > width * FALLBACK_MAX_SPINE_WIDTH_RATIO:
            continue
        boxes.append(
            {
                "x1": float(x1),
                "y1": 0.0,
                "x2": float(x2),
                "y2": float(height),
                "confidence": FALLBACK_CONFIDENCE,
            }
        )
    return boxes


# --------------------------------------------------------------------------
# Combined pipeline: detect -> gate -> fallback
# --------------------------------------------------------------------------


def detect_spines(image_path):
    """Run the full local-detection stage: YOLOv8n-COCO ("book" class),
    gated by quality_score() AND a minimum detection count, falling back to
    OpenCV vertical-edge segmentation when either check fails.

    The count check exists alongside the weighted score because the score
    alone is a ratio/average and can pass on very few boxes (e.g. one
    confident, well-formed detection) that don't actually cover the shelf --
    see MIN_PLAUSIBLE_DETECTIONS above.

    Returns a dict: {"boxes": [...], "quality": float, "used_fallback": bool,
    "source": one of SOURCE_YOLOV8N / SOURCE_OPENCV_FALLBACK / SOURCE_NONE}.
    Never raises, including for a missing/corrupt image (boxes=[] in that case).
    """
    image = _read_image(image_path)
    if image is None:
        return {
            "boxes": [],
            "quality": 0.0,
            "used_fallback": False,
            "source": SOURCE_NONE,
        }

    height, width = image.shape[:2]
    book_boxes = detect(image_path)
    quality = quality_score(book_boxes, (width, height))

    if quality < QUALITY_THRESHOLD or len(book_boxes) < MIN_PLAUSIBLE_DETECTIONS:
        return {
            "boxes": _opencv_fallback_boxes(image),
            "quality": quality,
            "used_fallback": True,
            "source": SOURCE_OPENCV_FALLBACK,
        }

    return {
        "boxes": book_boxes,
        "quality": quality,
        "used_fallback": False,
        "source": SOURCE_YOLOV8N,
    }


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------


def crop_spines(image_path, boxes):
    """Crop each box out of image_path, in the order given.

    Boxes are clamped to the image bounds; a box that clamps to zero area is
    skipped rather than producing an empty crop. Returns [] for an
    unreadable image or an empty/None boxes list -- never raises.
    """
    if not boxes:
        return []
    image = _read_image(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    crops = []
    for box in boxes:
        x1 = int(max(0, min(box["x1"], width - 1)))
        y1 = int(max(0, min(box["y1"], height - 1)))
        x2 = int(max(x1 + 1, min(box["x2"], width)))
        y2 = int(max(y1 + 1, min(box["y2"], height)))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
    return crops
