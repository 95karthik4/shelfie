"""Benchmark YOLOv8n-COCO vs YOLO-World on the committed test photos, CPU only.

Answers the "local vs. hosted routing" question CLAUDE.md asks for on the
detector side: does a closed-set COCO "book" detector already do the job, or
does the open-vocabulary "book spine" prompt find more (and more relevant)
boxes? Run from backend/, with the project venv:

    venv/bin/python -m vision.benchmark

Every number printed here is measured on this machine, on the images
actually committed under test_photos/ -- nothing is invented or copied from
documentation. Per-image latency covers image decode + inference for both
detectors (an apples-to-apples read of real per-image wall-clock cost); model
construction and weight loading happen once, outside the timed per-image
loop, for both. An untimed warm-up inference also runs once per detector
before the measured loop, so first-call predictor initialization overhead
doesn't distort the first photo's latency or the average.

YOLOv8n-COCO won this comparison (more boxes/image at comparable latency,
overlays visually confirmed as individual spines) and became the runtime
detector in detector.py -- see that module's docstring. Because of that, this
benchmark deliberately does NOT call detector.detect() for the YOLO-World
column below: detector.detect() now runs YOLOv8n, and reusing it would
silently mislabel YOLOv8n numbers as YOLO-World. Both columns here load and
run their own model independently of detector.py, so this file stays a
correct historical comparison regardless of which model backs the runtime
detector.
"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from . import detector

TEST_PHOTOS_DIR = Path(__file__).resolve().parent.parent.parent / "test_photos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Standard closed-set COCO checkpoint -- same weights as the runtime detector,
# loaded independently here so this benchmark doesn't depend on detector.py's
# internals. Off-the-shelf weights only, never fine-tuned.
YOLOV8N_WEIGHTS = "yolov8n.pt"
YOLOV8N_CONF_THRESHOLD = 0.25  # ultralytics' own default for this checkpoint
COCO_BOOK_CLASS_NAME = "book"

# Open-vocabulary checkpoint -- historical comparison only, not part of the
# runtime pipeline. Off-the-shelf weights only, never fine-tuned.
YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"
YOLO_WORLD_CLASSES = ["book spine"]
# Open-vocabulary prompts like "book spine" tend to score lower than a
# closed-set COCO class even on a correct detection, so the confidence floor
# here is set well below ultralytics' 0.25 default to favor recall; the
# quality gate is what actually decides whether a result is trustworthy.
YOLO_WORLD_CONF_THRESHOLD = 0.10

DEVICE = "cpu"


def _test_photos(photos_dir):
    return sorted(
        path
        for path in Path(photos_dir).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _first_readable_photo(photos):
    for photo in photos:
        if cv2.imread(str(photo)) is not None:
            return photo
    return None


def _yolov8n_book_boxes(model, image_path):
    """Boxes YOLOv8n-COCO assigns to its stock 'book' class, converted to the
    same dict schema detector.quality_score() expects (x1, y1, x2, y2,
    confidence) -- this is the exact runtime detect() logic in detector.py,
    duplicated here so this benchmark stays independent of that module. Also
    returns wall-clock latency in milliseconds for decode + predict()
    together, matched to what _yolo_world_boxes() measures below."""
    start = time.perf_counter()
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    results = model.predict(
        source=image,
        conf=YOLOV8N_CONF_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000

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
    return boxes, latency_ms


def _yolo_world_boxes(model, image_path, image_size):
    """Run the real YOLO-World checkpoint (set_classes(["book spine"]))
    independently of detector.py, decode + predict timed together to match
    _yolov8n_book_boxes() above. Still scored with detector.quality_score(),
    since that gate logic is shared, model-agnostic, and unchanged."""
    start = time.perf_counter()
    image = cv2.imread(str(image_path))
    results = model.predict(
        source=image,
        conf=YOLO_WORLD_CONF_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    width, height = image_size
    boxes = []
    for result in results:
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None:
            continue
        for box in result_boxes:
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

    quality = detector.quality_score(boxes, image_size)
    return boxes, latency_ms, quality


def run_benchmark(photos_dir=TEST_PHOTOS_DIR):
    """Run both detectors over every image in photos_dir, print a table, and
    return the raw per-image rows."""
    photos = _test_photos(photos_dir)
    if not photos:
        print(f"No test photos found under {photos_dir}")
        return []

    yolov8n_model = YOLO(YOLOV8N_WEIGHTS)  # construction/weight load: untimed
    world_model = YOLO(YOLO_WORLD_WEIGHTS)  # construction/weight load: untimed
    world_model.set_classes(YOLO_WORLD_CLASSES)

    # Untimed warm-up: absorbs one-time predictor initialization (e.g. lazy
    # graph/backend setup on first predict() call) so it doesn't land inside
    # the first measured photo's latency, or skew the average.
    warmup_photo = _first_readable_photo(photos)
    if warmup_photo is not None:
        warmup_image = cv2.imread(str(warmup_photo))
        warmup_size = warmup_image.shape[1::-1]  # (width, height)
        _yolov8n_book_boxes(yolov8n_model, warmup_photo)
        _yolo_world_boxes(world_model, warmup_photo, warmup_size)

    rows = []
    for photo in photos:
        # Untimed readability/size check -- also warms the OS file cache
        # equally ahead of both timed reads below, so neither detector is
        # penalized for going first.
        probe_image = cv2.imread(str(photo))
        if probe_image is None:
            print(f"skipping unreadable image: {photo.name}")
            continue
        height, width = probe_image.shape[:2]

        v8n_boxes, v8n_latency_ms = _yolov8n_book_boxes(yolov8n_model, photo)
        world_boxes, world_latency_ms, world_quality = _yolo_world_boxes(
            world_model, str(photo), (width, height)
        )

        # The actual runtime gate from detector.detect_spines(): YOLOv8n is
        # the live detector now, so this -- not the YOLO-World quality above
        # -- is what decides whether a real request would hit the OpenCV
        # fallback.
        v8n_quality = detector.quality_score(v8n_boxes, (width, height))
        runtime_fallback = (
            v8n_quality < detector.QUALITY_THRESHOLD
            or len(v8n_boxes) < detector.MIN_PLAUSIBLE_DETECTIONS
        )

        rows.append(
            {
                "filename": photo.name,
                "yolov8n_boxes": len(v8n_boxes),
                "yolov8n_latency_ms": v8n_latency_ms,
                "yolov8n_quality": v8n_quality,
                "runtime_fallback": runtime_fallback,
                "yolo_world_boxes": len(world_boxes),
                "yolo_world_latency_ms": world_latency_ms,
                "yolo_world_quality": world_quality,
            }
        )

    _print_table(rows)
    return rows


def _print_table(rows):
    if not rows:
        return

    # v8n_* / runtime_fallback? = the actual runtime detector and gate.
    # world_* = historical comparison only (see module docstring); its
    # quality is never the runtime fallback decision.
    headers = [
        "filename",
        "v8n_boxes",
        "v8n_ms",
        "v8n_quality",
        "runtime_fallback?",
        "world_boxes",
        "world_ms",
        "world_quality",
    ]
    formatted = [headers]
    for row in rows:
        formatted.append(
            [
                row["filename"],
                str(row["yolov8n_boxes"]),
                f"{row['yolov8n_latency_ms']:.1f}",
                f"{row['yolov8n_quality']:.3f}",
                "yes" if row["runtime_fallback"] else "no",
                str(row["yolo_world_boxes"]),
                f"{row['yolo_world_latency_ms']:.1f}",
                f"{row['yolo_world_quality']:.3f}",
            ]
        )

    count = len(rows)
    avg_v8n_boxes = sum(r["yolov8n_boxes"] for r in rows) / count
    avg_v8n_ms = sum(r["yolov8n_latency_ms"] for r in rows) / count
    avg_v8n_quality = sum(r["yolov8n_quality"] for r in rows) / count
    avg_world_boxes = sum(r["yolo_world_boxes"] for r in rows) / count
    avg_world_ms = sum(r["yolo_world_latency_ms"] for r in rows) / count
    avg_world_quality = sum(r["yolo_world_quality"] for r in rows) / count
    formatted.append(
        [
            "AVERAGE",
            f"{avg_v8n_boxes:.1f}",
            f"{avg_v8n_ms:.1f}",
            f"{avg_v8n_quality:.3f}",
            "",
            f"{avg_world_boxes:.1f}",
            f"{avg_world_ms:.1f}",
            f"{avg_world_quality:.3f}",
        ]
    )

    widths = [max(len(row[i]) for row in formatted) for i in range(len(headers))]
    for i, row in enumerate(formatted):
        line = "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row))
        print(line)
        if i == 0:
            print("  ".join("-" * widths[col] for col in range(len(headers))))


if __name__ == "__main__":
    run_benchmark()
