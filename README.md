# Shelfie — bookshelf photo → structured library

> Setup/run steps, architecture description, and catalog notes are still to be
> written. The sections below record measurements taken so far.

## Measured performance

All figures below come from a single real request through the running Django
endpoint (`POST /api/scans/`) — real YOLOv8n detector on CPU, real crops, one
real Gemini call, real matcher, real SQLite writes. No mocks, no cache replay.

### End-to-end scan

| | |
|---|---|
| Test image | `test_photos/shelf_07_modern_readable.jpeg` |
| Detected spines | 16 |
| HTTP status | 201 |
| Wall-clock request latency | 21.26 s |
| Gemini VLM latency | 18.498 s |
| Hosted VLM requests | 1 |
| Detector | `yolov8n_coco` |
| Detector quality score | 0.757 |
| OpenCV fallback used | false |
| VLM cache hit | false |
| Results | 1 auto, 13 review, 2 unmatched |

The VLM call is ~87% of the wall clock. The remaining ~2.8 s covers the
one-time YOLO weight load, detection, cropping, matching against the
105-entry catalog, and persistence.

### Token usage (same scan)

| Category | Tokens |
|---|---|
| Input | 17,621 |
| Candidate/output | 695 |
| Thinking | 1,626 |
| **Billable output (candidates + thinking)** | **2,321** |
| **Total** | **19,942** |

The three scalar categories sum exactly to the provider-reported total, so
thinking tokens account for the entire gap between visible input/output and
the total. They are collected explicitly rather than derived — computing the
total as input + candidates would have undercounted this call by 1,626
tokens (8.2%).

Input dominates at ~1,101 tokens per crop, which is why every crop in a scan
goes out in **one** batched request rather than one request per book.

### Paid-equivalent cost estimate

Pricing source:
<https://ai.google.dev/gemini-api/docs/pricing>

Gemini 3.6 Flash, Standard paid tier, through 2026-12-31:

* $0.75 per 1M input tokens
* $3.75 per 1M output tokens, **including thinking tokens**

Applying those published rates to the measured usage above:

| Component | Cost |
|---|---|
| Input (17,621 tokens) | $0.013216 |
| Output + thinking (2,321 tokens) | $0.008704 |
| **Total per 16-spine image** | **$0.021920 USD** |

Approximately **2.2 cents per image**, or **$2.19 per 100 scans** and
**$21.92 per 1,000 scans**.

**This is a PAID-EQUIVALENT ESTIMATE, not verified money actually charged.**
It is arithmetic applying the provider's published paid rates to token counts
we measured. We did not independently verify which billing tier this API key
is on, and no billing statement was inspected. Treat the figure as an order-of-
magnitude guide to what this pipeline would cost at paid rates, not as an
invoice.

Two further caveats:

* Cost scales with crop count and crop size. A denser shelf produces more
  crops and more image tokens; a sparse one produces fewer.
* This is **one measurement of one photo**, not an average over the eight
  committed test photos.

## Known limitations and tradeoffs

### Low-scoring books are routed to REVIEW rather than UNMATCHED

Observed in the measured scan above: 13 of 16 items came back as `review`
with confidences between 0.03 and 0.28, every one carrying
`DIFFERENT_WORK_AMBIGUITY`. Those spines were read correctly by the VLM — they
are simply books that are **not in the catalog** at all.

The cause is rule precedence. The matcher runs its ambiguity scan before
status routing, and any detected ambiguity forces `REVIEW` regardless of
score. When a book is absent from the catalog, the top several candidates all
score near zero and therefore land within the 0.15 ambiguity margin of each
other, which trips the different-work penalty. The practical effect is that
the `UNMATCHED` route (score < 0.60) is close to unreachable for a legible
read, and a book that is genuinely absent is presented to the user as a
low-confidence *match* instead of as unmatched.

Nothing here is silently accepted or dropped — every spine still reaches the
user — but the review queue is longer and less informative than it should be.

**Possible improvement (not implemented):** apply ambiguity handling only
after the rank-1 score has reached the review threshold. Ambiguity between
two near-zero candidates is not meaningful ambiguity, and suppressing it
would let genuinely absent books fall through to `UNMATCHED` where they
belong.

The matcher was left unchanged after this observation so that the measured
numbers above and the committed test suite describe the same code.
