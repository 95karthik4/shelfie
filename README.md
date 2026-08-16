# Shelfie — bookshelf photo → structured library

Photograph a bookshelf; get a structured, human-confirmed list of books.

An Expo app sends a photo to a Django REST API. A local CPU detector finds the
spines, one hosted vision-language call reads them, a fuzzy matcher resolves
each read against a deliberately messy 105-entry catalog, and every book is
confirmed, corrected or discarded by a human before it enters the library.

---

## 1. Setup from a clean clone

Prerequisites: Python 3.13, Node 20+, a Gemini API key
([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

### Backend

```bash
cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt

cp .env.example .env          # then edit .env and set GEMINI_API_KEY
venv/bin/python manage.py migrate
venv/bin/python manage.py runserver 0.0.0.0:8000
```

`backend/.env` is gitignored and needs two values:

```
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-3.6-flash
```

The model id is **never** hardcoded — `vlm/gemini.py` reads it from the
environment. On first run, ultralytics downloads the YOLOv8n weights (~6 MB).

### Mobile

```bash
cd mobile
npm install

cp .env.example .env          # then set EXPO_PUBLIC_API_BASE_URL
npx expo start -c
```

`mobile/.env` needs the address of the API **as seen from the device running
the app**:

| Running on | Value |
|---|---|
| iOS Simulator | `http://127.0.0.1:8000` |
| Android emulator | `http://10.0.2.2:8000` |
| Physical phone, same Wi-Fi | `http://<your-mac-LAN-IP>:8000` (`ipconfig getifaddr en0`) |

`mobile/.env.example` also sets **`EXPO_PUBLIC_USE_RN_FETCH=1`, which is
required, not optional.** Expo SDK 57 installs its own WinterCG `fetch` as the
global `fetch` on iOS and Android, and that implementation accepts only
strings, `Blob`s, or objects with a `bytes()` method as form parts. React
Native's standard file part — `{ uri, name, type }` — therefore throws
`Unsupported FormDataPart implementation`, and the photo upload fails on the
device before any request is sent. The flag restores React Native's built-in
`fetch`, which handles that shape. Without it, `POST /api/scans/` cannot work.

`EXPO_PUBLIC_*` variables are inlined at bundle time, so restart with `-c`
after changing `.env`.

### Verify without a phone

```bash
cd backend
venv/bin/python -m pytest vlm/tests matching/tests -q   # 77 tests
venv/bin/python manage.py test api                      # 27 tests
venv/bin/python -m vision.benchmark                     # detector comparison, CPU
```

---

## 2. Architecture

```
Expo app  ──POST multipart photo──▶  Django REST (POST /api/scans/)
                                          │
                                          ▼
                      vision/detector.py  ── YOLOv8n-COCO, CPU, "book" class
                                          │
                                   quality gate (0.55)
                                          │
                              ┌───────────┴───────────┐
                         gate pass                gate fail
                              │                        │
                        YOLOv8n boxes      OpenCV vertical-edge fallback
                              └───────────┬───────────┘
                                          ▼
                              crop_spines() → N spine crops
                                          │
                                          ▼
                        vlm/gemini.py  ── ONE Gemini request, all N crops,
                                          indexed "SPINE 0…N", structured JSON
                                          │
                                          ▼
                       matching/matcher.py ── rapidfuzz scoring + ambiguity
                                          │
                                          ▼
                        {auto | review | unmatched} per spine → SQLite
                                          │
                                          ▼
                          Expo review screen → explicit human decision
                                          │
                                          ▼
                     POST /api/scan-items/<id>/confirm/ → ConfirmedBook
```

### Local vs hosted routing

| Work | Where | Why |
|---|---|---|
| Find spine bounding boxes | **Local** (YOLOv8n, CPU) | Object localisation is cheap locally and would otherwise mean sending the whole image and trusting the VLM to segment it. |
| Segment when detection is weak | **Local** (OpenCV edges) | A deterministic fallback beats an extra hosted call. |
| Read title/author off a spine | **Hosted** (Gemini) | Rotated, low-contrast, stylised spine text is exactly what a local OCR would fail on. |
| Match text to catalog | **Local** (rapidfuzz) | Deterministic, testable, free, and needs the catalog — no reason to pay a model for string comparison. |

The cost consequence: cropping locally means only spine pixels are uploaded,
and batching means the ~600-token instruction is sent **once** per photo
rather than once per book.

### Detector choice (measured, not assumed)

The original plan was YOLO-World with an open-vocabulary `"book spine"`
prompt. Benchmarking it against stock YOLOv8n-COCO on the eight committed
photos reversed that decision — see §3.

---

## 3. Measured numbers

Everything below was measured on this machine (Apple Silicon, CPU inference)
against the committed `test_photos/`. Nothing is copied from documentation.

### Detector benchmark — `venv/bin/python -m vision.benchmark`

| Photo | YOLOv8n boxes | YOLOv8n ms | quality | YOLO-World boxes | YOLO-World ms |
|---|---|---|---|---|---|
| shelf_01_antique_vertical | 11 | 100.0 | 0.810 | 5 | 119.1 |
| shelf_02_dense_antique | 16 | 86.1 | 0.835 | 4 | 118.9 |
| shelf_03_mixed_heights | 21 | 85.0 | 0.775 | 3 | 115.9 |
| shelf_04_horizontal_stacks | 23 | 117.8 | 0.733 | 0 | 178.2 |
| shelf_05_dense_vertical | 14 | 175.5 | 0.832 | 11 | 201.6 |
| shelf_06_rotated_horizontal | 9 | 138.5 | 0.658 | 1 | 163.8 |
| shelf_07_modern_readable | 16 | 77.6 | 0.757 | 0 | 104.4 |
| shelf_08_wide_mixed_layout | 34 | 101.3 | 0.773 | 2 | 148.4 |
| **Average** | **18.0** | **110.2** | **0.772** | **3.2** | **143.8** |

YOLOv8n found **5.6× more boxes at 77% of the latency**, and visual inspection
of the overlays confirmed its boxes were individual spines rather than
whole-shelf blobs. It is the shipped detector; YOLO-World remains in
`vision/benchmark.py` as the historical comparison only.

The OpenCV fallback did **not** trigger on any of the eight photos (all scored
above the 0.55 gate with ≥4 plausible boxes). It is therefore exercised by
unit tests rather than by these photos.

### End-to-end scan (real HTTP request, real Gemini call)

| | |
|---|---|
| Test image | `test_photos/shelf_07_modern_readable.jpeg` |
| Detected spines | 16 |
| HTTP status | 201 |
| **Wall-clock request latency** | **21.26 s** |
| **Gemini VLM latency** | **18.498 s** (~87% of the total) |
| Everything else (YOLO load + detect + crop + match + persist) | ~2.8 s |
| Hosted VLM requests | 1 |
| Detector / quality / fallback | `yolov8n_coco` / 0.757 / false |
| VLM cache hit | false |
| Results | 1 auto, 13 review, 2 unmatched |

### Token usage (same scan)

| Category | Tokens |
|---|---|
| Input | 17,621 (~1,101 per crop) |
| Candidate/output | 695 |
| Thinking | 1,626 |
| **Billable output (candidates + thinking)** | **2,321** |
| **Total** | **19,942** |

The three scalars sum exactly to the provider-reported total, so thinking
tokens account for the whole gap. They are read from the provider's
`usage_metadata` rather than derived — computing the total as input +
candidates would have undercounted this call by 1,626 tokens (8.2%).

### Paid-equivalent cost estimate

Pricing source: <https://ai.google.dev/gemini-api/docs/pricing> — Gemini 3.6
Flash, Standard paid tier, through 2026-12-31: **$0.75 / 1M input**,
**$3.75 / 1M output including thinking tokens**.

| Component | Cost |
|---|---|
| Input (17,621 tokens) | $0.013216 |
| Output + thinking (2,321 tokens) | $0.008704 |
| **Total per 16-spine image** | **$0.021920** |

≈ **2.2 ¢ per image**, ≈ **$2.19 per 100 scans**, ≈ **$21.92 per 1,000 scans**.

**This is a PAID-EQUIVALENT ESTIMATE, not verified money actually charged.**
It applies published paid rates to token counts we measured. We did not
independently verify which billing tier this API key is on, and no billing
statement was inspected.

Caveats: cost scales with crop count and crop size, and this is **one
measurement of one photo**, not an average over the eight test photos.

### Real-device run (physical Android, over the internet)

The whole flow was also run from a **physical Android phone on a different
network**, reaching this machine through temporary HTTPS tunnels — a real
camera photo of a real bookshelf, not a committed test image.

| | |
|---|---|
| Detected spines | 15 |
| `POST /api/scans/` | **201 Created**, **22.57 s** end to end |
| Gemini VLM latency | 19,059 ms, 1 hosted request, cache miss |
| Detector | `yolov8n_coco`, quality 0.763, no fallback |
| Results | **1 high-confidence, 13 review, 1 unmatched** |
| Confirm | `POST /api/scan-items/38/confirm/` → **201**, persisted *The Da Vinci Code* / *Dan Brown* from catalog row 51 |
| Discard | No request made, no `ConfirmedBook` row, item still `confirmed=False` |
| Library | `GET /api/library/` returned exactly that one confirmed book |

Re-running **the identical photo** hit the development VLM cache: same crops →
same cache key → **`cache_hit=True`, zero hosted requests, 1.19 s** for the
whole request instead of 22.57 s. The detector still ran; only the hosted call
was skipped.

Two things this run confirms beyond the numbers: the human-in-the-loop
boundary held under real use (15 books scanned, one confirmed by an explicit
tap, one discarded with no server trace, thirteen left waiting), and the
`REVIEW`-vs-`UNMATCHED` limitation in §9 reproduced independently on a
different shelf.

---

## 4. The catalog

`catalog.csv` — 105 entries, 104 distinct `work_id`s.

Schema: `catalog_id, work_id, title, author, alternate_titles,
author_aliases, edition, contains_work_ids, notes`. Multi-value columns are
pipe-separated. `work_id` is what makes "same book, different row" expressible
separately from "different book, same title".

Weighted towards books people actually own (classics, popular fiction and
non-fiction, well-known series) so that real shelves match something. The
deliberate traps:

| Ambiguity required | In the catalog |
|---|---|
| Two editions of one book | `Dune` ids 1 & 2 — same `work_id=dune`, editions *Ace 1990* / *Penguin 2010* |
| Same book, two titles | `Northern Lights` / *The Golden Compass*; `Harry Potter and the Philosopher's Stone` / *Sorcerer's Stone*; `1984` / *Nineteen Eighty-Four* (11 rows carry alternates) |
| Different books, same title | `Home` — Harlan Coben (id 5) vs Toni Morrison (id 6); `The Gift` — Danielle Steel (id 7) vs Hafiz (id 8) |
| Omnibus + its volumes | `The Lord of the Rings` (id 14) contains `fellowship_ring\|two_towers\|return_king`; `His Dark Materials` (id 17) contains its three volumes |
| Substring titles | `Dune` vs `Dune Messiah`; `The Final Empire` vs `Mistborn: The Final Empire` |
| Author name variants | 34 rows carry aliases — `J.K. Rowling` / `Rowling, J.K.` / `Joanne Rowling` / `J K Rowling`; accented and transliterated forms |

Built in about half an hour: LLM-generated candidate rows, then hand-checked
for the trap coverage above, with each trap recorded in the `notes` column so
the intent is auditable.

---

## 5. Matching, and where confidence comes from

Pure Python in `backend/matching/`, zero Django imports, so it is testable in
isolation.

**Normalisation is information-preserving**: Unicode NFKD, accent strip,
casefold, punctuation → spaces, `&` → `and`. No stopword removal. Leading
articles are handled in *scoring* (the query is compared both with and without
its leading article, best wins) rather than being destroyed up front. Authors
additionally get `"Last, First"` → `"first last"` and initial collapsing
(`J.K.` → `j k`).

**Score** per catalog row:

- `T` = best rapidfuzz `token_set_ratio` over {canonical title + alternates} × {query, query-minus-article}
- Substring guard: if one normalised title is a proper substring of the other, apply a **0.25** penalty unless an exact alias matched (this is what stops *Dune* scoring 1.0 against *Dune Messiah*)
- Author readable: `S = 0.75·T + 0.25·A`; author unreadable: `S = T`, and the final confidence is **capped at 0.84** — deliberately below the 0.85 auto threshold, so a title-only read can never auto-accept

**Ambiguity** — any rival within a **0.15** margin of rank-1, worst relation
wins, compared against the best candidate *of each relation type* rather than
blindly against rank-2:

| Relation | Penalty | Reason code |
|---|---|---|
| Different work | −0.30 | `DIFFERENT_WORK_AMBIGUITY` |
| Omnibus / contained | −0.15 | `OMNIBUS_AMBIGUITY` |
| Same work, different edition | −0.05 | `EDITION_AMBIGUITY` |

**Routing**: any ambiguity reason → `review`, regardless of score; else
≥0.85 → `auto`; else ≥0.60 → `review`; else `unmatched`.

Confidence is **an explainable decision score, not a calibrated probability**.
It exists to rank and to route, and every deduction is reported as a reason
code the UI shows in plain language.

---

## 6. Human in the loop

The persistence boundary is structural: a scan writes `Scan` and `ScanItem`
rows describing what the models *think*. A `ConfirmedBook` row — the user's
library — is only ever created by `POST /api/scan-items/<id>/confirm/`.

**Even `auto` items require an explicit tap.** High confidence changes the
label and the button text, never the requirement.

Every spine, whatever its status, offers three outcomes:

| Outcome | What happens |
|---|---|
| **Confirm** | Accept the suggested catalog entry → `ConfirmedBook` with that `catalog_id` |
| **Correct** | Accept a *different* catalog entry, or type a title/author freehand. The submitted `catalog_id` deliberately need not equal the matcher's — a human who can only agree isn't reviewing |
| **Discard** | Frontend-only. No request, no `ConfirmedBook`. The card stays visible, labelled "Discarded — not added to library", with Undo |

Nothing is silently accepted and nothing is silently dropped: illegible spines
and unusable crops still become items, and the AI's original read, suggestion,
confidence and reasons remain on screen after a decision, so what the model
said and what the human chose are both auditable.

---

## 7. Failure handling

| Failure | Behaviour |
|---|---|
| Missing/corrupt/oversized upload | 400 with DRF field errors; no file kept |
| Zero spines detected | **201** with `items: []` and no hosted call at all |
| Illegible spine / unusable crop | Normal item, `unmatched`, reason `NOT_LEGIBLE` / `INVALID_CROP` |
| Malformed VLM JSON, or an index set that doesn't match the crops exactly | One stricter retry, then `VLMResponseError` → **502** |
| Missing `GEMINI_API_KEY`/`GEMINI_MODEL` | `VLMConfigurationError` → **503** (never disguised as unreadable books) |
| 408/429/5xx | Bounded retry, full-jitter backoff, hard cap of 4 hosted requests → **503** + `Retry-After: 30` |
| 401/403/400/413 | No retry → **502** |
| Pipeline or DB write fails | Transaction rolls back, orphaned upload deleted, error propagates unchanged |
| Double confirmation | **409** (OneToOne is the DB-level backstop) |

Provider error text is **logged, never returned** — response bodies carry only
a server-owned message, a stable `code`, and `retryable`.

---

## 8. Tests

104 tests, all offline — no network, no Gemini, no YOLO weights loaded.

```bash
cd backend
venv/bin/python -m pytest vlm/tests matching/tests -q   # 77
venv/bin/python manage.py test api                      # 27
```

- **Matcher (18)** — the 13 required cases: exact match, OCR typo, US/UK alternate titles, `J.K. Rowling` vs `Rowling, J. K.`, accents, the `Dune`/`Dune Messiah` substring guard, shared title with unreadable author, shared title with correct author, two editions, omnibus vs volume, `legible:false`, garbage input, and the three-editions-plus-a-different-work case where the −0.30 tier must win.
- **VLM (59)** — index integrity (missing/duplicate/out-of-range/extra all force the one retry), exact response key set, retry bounds, systemic failures raising rather than degrading, cache round-trip and corruption.
- **API (27)** — upload validation, successful scan, all four VLM failure statuses, the confirm/correct/manual paths, 400/404/409, library persistence, and the orphan-upload cleanup boundary.

---

## 9. Known limitations

### Physical iOS runtime is unverified (Android is verified)

**Android: verified on real hardware** — see §3.4. Camera, capture, upload,
review and library persistence all ran on a physical Android phone.

**iOS: never run on a device.** The app type-checks, passes `expo-doctor`
21/21 and bundles for iOS (603 modules), but two blockers stopped a physical
iOS run:

1. The App Store build of Expo Go does not currently run **SDK 57**.
2. This Mac has **Command Line Tools only — no Xcode and no iOS SDK**, and a
   local development build needs both (plus Developer Mode on the phone). At
   ~40 GB and hours of setup, that did not fit the deadline.

Since both platforms share the same JavaScript and the same `expo-camera` /
`expo-image-picker` APIs, the Android run exercises the same code paths — but
iOS-specific behaviour (permission prompts, its multipart implementation) is
genuinely untested.

One path was **not** tested on hardware either: correcting a match to a
different catalog entry, and manual title/author entry. Only
accept-the-suggestion and discard were exercised on the device. Both
untested paths are covered by the API test suite (§8).

### Low-scoring books are routed to REVIEW rather than UNMATCHED

In the measured scan, 13 of 16 items came back `review` at 0.03–0.28
confidence, all carrying `DIFFERENT_WORK_AMBIGUITY`. Those spines were read
correctly — they are books genuinely **not in the catalog**.

The cause is rule precedence: the ambiguity scan runs before status routing,
and any ambiguity forces `review` regardless of score. For an absent book the
top candidates all score near zero and therefore sit inside the 0.15 margin of
each other, tripping the different-work penalty. The practical effect is that
`unmatched` is close to unreachable for a legible read.

Nothing is silently accepted or dropped, but the review queue is longer and
less informative than it should be. **Fix (not implemented):** apply ambiguity
handling only once rank-1 clears the review threshold — ambiguity between two
near-zero candidates is not meaningful ambiguity. Left unchanged so that the
measured numbers above and the committed test suite describe the same code.

### Other

- **Synchronous request.** ~21 s with no queue or progress endpoint; the client waits. Fine for a demo, wrong for production.
- **No auth**, per the brief. The library is global.
- **Uploads accumulate** in `backend/media/` with no cleanup.
- **Detector fallback is untested on real photos** — the OpenCV path never triggered on the eight committed images.
- **One cost measurement**, one photo.

### With another day

1. Fix the ambiguity/threshold interaction above and re-measure the auto/review/unmatched split across all eight photos.
2. Get the app onto real hardware (Xcode local build) and verify capture.
3. Accuracy measurement: hand-label the eight photos and report precision/recall per stage, rather than only latency and cost.
4. Show the runner-up candidate in the review UI — the matcher already returns it; the frozen response contract didn't include it.
5. Make the scan asynchronous (202 + poll) so the phone isn't holding a 21-second request.

---

## 10. Repo layout

```
backend/
  config/        Django project (settings, urls, wsgi/asgi)
  api/           models, serializers, views, pipeline, catalog adapter, error handler
  vision/        YOLOv8n detector, quality gate, OpenCV fallback, benchmark
  vlm/           Gemini client — read_spines() is the only public entry point
  matching/      normalisation + matcher (pure Python, no Django)
mobile/          Expo app (TypeScript, no navigation/state/UI libraries)
catalog.csv      105 canonical entries
test_photos/     8 committed shelf photos
```
