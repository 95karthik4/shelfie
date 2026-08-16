"""Hosted VLM stage: read title + author off spine crops with Google Gemini.

Per CLAUDE.md's frozen architecture this is the one place in the app that
talks to a hosted model. It sits between the local vision stage and the
matcher:

    vision.crop_spines() -> [numpy BGR crops] -> read_spines() -> [reads]
                                                                    |
                                                          matching.match()

Zero Django imports. Nothing here imports the matcher. The only public
entry point is read_spines(); every google.genai symbol stays behind it, so
swapping providers later touches this file and nothing else.

Design commitments (see CLAUDE.md, and the checkpoint contract):

  * ONE hosted request per batch of crops, never one call per book.
  * Every crop is labelled "SPINE <i>" with its *original* list index, and
    results are reconciled by that index -- output order is never trusted.
  * A successful return corresponds exactly to every input index. Missing,
    duplicated or out-of-range indices in the provider's answer are all
    treated as a malformed batch, not silently patched up.
  * Systemic failures raise a structured VLMError. Only an individually
    unusable crop degrades in-band, as an "invalid_crop" entry.
  * Retries are bounded on two axes and by one hard request cap; the loops
    cannot interact into an unbounded sequence.
  * Crops are sent at their original pixel dimensions. Downscaling would cut
    upload bytes and image tokens, but small spine text is the hard part of
    this task and we have not measured what a resize costs in transcription
    accuracy -- so it is a benchmarked optimisation for later, not an
    unmeasured default now.

Result schema, one dict per input crop, sorted by index:

    {"index": int, "title": str|None, "author": str|None, "legible": bool}

plus an extra {"error": "invalid_crop"} key on crops we could not encode.
"""

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

from .errors import (
    CODE_INVALID_CROP,
    VLMAPIError,
    VLMConfigurationError,
    VLMError,
    VLMResponseError,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Tunable constants. Everything tunable lives here, nothing is inlined below.
# --------------------------------------------------------------------------

# Config is read from the environment on every call, never at import and
# never hardcoded -- CLAUDE.md forbids a literal model ID in source, and
# per-call reads keep the module importable (and testable) without secrets.
ENV_API_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"

# --- Retry budget ---
# Two axes. Outer: validation attempts (a well-formed HTTP response whose
# *content* we reject). Inner: transport attempts (the request itself
# failed transiently). 2 x 3 = 6 in principle, but MAX_TOTAL_REQUESTS caps
# the product, so the true worst case is 4 hosted calls per read_spines().
MAX_VALIDATION_ATTEMPTS = 2
MAX_TRANSPORT_ATTEMPTS = 3
MAX_TOTAL_REQUESTS = 4

# Full-jitter backoff: sleep = uniform(0, min(base * 2**attempt, max)).
# Full jitter rather than fixed delay so concurrent uploads retrying off the
# same 429 don't re-collide in lockstep.
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 8.0

# Retried: request timeout, rate limit, and any server-side error. 429 is the
# one we actually expect on Gemini's free tier. Every other 4xx is a fault in
# our own request (bad key, bad model ID, payload too large) and retrying it
# would just repeat the mistake.
TRANSIENT_STATUS_CODES = frozenset({408, 429})
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 599

# Per-request ceiling. Generous: a dozen images in one multimodal request is
# a slow call, and a premature client timeout would burn retry budget on a
# request that was going to succeed.
REQUEST_TIMEOUT_MS = 90_000

# --- Image encoding ---
JPEG_QUALITY = 90

# --- Prompt / schema versioning ---
# Bumped by hand whenever the prompt or schema below changes meaning. It is
# part of the cache key, so a bump invalidates every cached batch rather than
# replaying answers produced under different instructions.
PROMPT_VERSION = "1"

# --- Development cache ---
# backend/.vlm_cache/ -- gitignored, dev-only. Module-level so tests can
# redirect it to a tmp dir.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".vlm_cache"

# Deterministic reads: this is transcription, not creative writing.
TEMPERATURE = 0.0

# The exact key set every response object must carry -- no more, no fewer.
# Enforced rather than assumed: a missing key would otherwise read as a null
# via .get(), and an extra key means the model answered a question we did not
# ask, which is a signal the batch shouldn't be trusted wholesale.
RESPONSE_ENTRY_KEYS = frozenset({"index", "title", "author", "legible"})

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "title": {"type": "STRING", "nullable": True},
            "author": {"type": "STRING", "nullable": True},
            "legible": {"type": "BOOLEAN"},
        },
        "required": ["index", "title", "author", "legible"],
        "propertyOrdering": ["index", "title", "author", "legible"],
    },
}

BASE_PROMPT = """You are transcribing book spines from a photograph of a bookshelf.

Each image below is a single cropped book spine, preceded by its label:
SPINE 0, SPINE 1, SPINE 2, and so on.

The indices may not be consecutive and may not start at 0. Crops that could
not be processed are excluded from this request, so gaps in the numbering are
expected. Use the exact index printed on each label; never renumber them.

For every spine, return exactly one object using the index from its label.

Rules:
- Transcribe ONLY text that is visibly printed on that spine.
- Do NOT infer, complete or correct a title or author from your general
  knowledge of books. A partially visible title stays partial.
- Spine text is often rotated 90 degrees; read it in whatever orientation it
  is printed.
- If you can read a title but no author, set author to null and keep
  legible true. Same the other way round.
- If the crop is blurred, cropped through, too dark, shows no text, or is not
  a book spine at all, set title and author to null and legible to false.
- Never invent a plausible-sounding book to fill a gap.

Return one object for every SPINE label given, no more and no fewer, using
each index exactly once."""

# Appended on the single retry after a malformed batch. Restates the index
# contract, which is what a malformed response has usually broken.
STRICT_RETRY_SUFFIX = """

IMPORTANT -- the previous response was rejected as malformed.
Return ONLY a JSON array. No prose, no markdown fence.
Every object must have exactly the keys: index (integer), title (string or
null), author (string or null), legible (boolean).
Return exactly one object per SPINE label listed above: indices {indices},
each appearing exactly once, with no other indices. These indices are not
necessarily consecutive -- reproduce them exactly as labelled."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _load_config():
    """Read (api_key, model) from the environment.

    Raises VLMConfigurationError -- never a KeyError, never a stack trace
    with a partial key in it -- if either is missing or blank. Called before
    any client is built, so a misconfigured deployment costs zero requests.
    """
    api_key = (os.environ.get(ENV_API_KEY) or "").strip()
    model = (os.environ.get(ENV_MODEL) or "").strip()

    missing = [
        name
        for name, value in ((ENV_API_KEY, api_key), (ENV_MODEL, model))
        if not value
    ]
    if missing:
        raise VLMConfigurationError(
            "Hosted VLM is not configured: {} not set. "
            "Set it in backend/.env (gitignored) or the process "
            "environment.".format(" and ".join(missing))
        )
    # Deliberately returns the key but never logs it; no caller logs it either.
    return api_key, model


def _redact(text, secret):
    """Strip the API key out of a message before it can be raised or logged.

    Belt and braces: provider errors normally echo the response body, not the
    credential, but "normally" is not a guarantee worth betting a leaked key
    on, and every message this module produces passes through here.
    """
    if not secret or not text:
        return text
    return text.replace(secret, "***")


def _build_client(api_key):
    """Construct the Gemini client.

    Split out as its own function purely so the offline tests can replace it
    without monkeypatching the SDK.

    The SDK's own retry layer is left at its default, which is
    stop_after_attempt(1) -- i.e. off. Our retry policy is then the only one
    in play, which is what makes MAX_TOTAL_REQUESTS an honest bound and the
    measured latency attributable.
    """
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _build_client_guarded(api_key):
    """_build_client() with the containment boundary applied.

    Client construction is local setup (argument validation, transport
    wiring), so a failure here is our fault and non-retryable -- but it is
    still an SDK exception, and no raw SDK exception may leave this package.
    """
    try:
        return _build_client(api_key)
    except VLMError:
        # Already structured -- e.g. a test double raising deliberately.
        # Rewrapping would bury its code and status.
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberate containment boundary
        message = _redact(str(exc) or type(exc).__name__, api_key)
        raise VLMAPIError(
            "Could not initialise the Gemini client: {}".format(message),
            retryable=False,
        ) from exc


# --------------------------------------------------------------------------
# Crop encoding
# --------------------------------------------------------------------------


def _encode_crop(crop):
    """BGR numpy crop -> in-memory JPEG bytes, or None if unusable.

    Never raises and never writes to disk. Returning None is how a single bad
    crop degrades to an "invalid_crop" entry instead of taking down the batch.

    The crop keeps its original dimensions; see the module docstring on why
    there is no resize here.
    """
    try:
        if crop is None or not isinstance(crop, np.ndarray):
            return None
        if crop.size == 0 or crop.ndim not in (2, 3):
            return None
        if crop.ndim == 3 and crop.shape[2] not in (1, 3, 4):
            return None

        if crop.dtype != np.uint8:
            # imencode only accepts 8-bit for JPEG; anything else raises.
            # A dtype fix, not a resize -- no pixels are discarded.
            crop = np.clip(crop, 0, 255).astype(np.uint8)

        # cv2 expects BGR, which is exactly what crop_spines() hands us, so
        # no colour conversion here -- adding one would swap red and blue.
        ok, buffer = cv2.imencode(
            ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return None
        return buffer.tobytes()
    except Exception:
        logger.debug(
            "crop failed to encode; preserving it as invalid_crop", exc_info=True
        )
        return None


def invalid_crop_entry(index):
    """The preserved result for a crop we could not encode."""
    return {
        "index": index,
        "title": None,
        "author": None,
        "legible": False,
        "error": CODE_INVALID_CROP,
    }


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def _build_contents(encoded, strict):
    """Interleave "SPINE <i>" labels with their images, in index order.

    The label is what lets us reconcile by index later: the model is told
    which index each image is, so a reordered or partially-answered response
    is detectable rather than quietly misattributed. Labels carry the crop's
    original position in the caller's list, so excluded invalid crops leave
    visible gaps rather than shifting every later index by one.
    """
    indices = [index for index, _ in encoded]
    prompt = BASE_PROMPT
    if strict:
        prompt += STRICT_RETRY_SUFFIX.format(
            indices=", ".join(str(index) for index in indices)
        )

    contents = [prompt]
    for index, jpeg in encoded:
        contents.append("SPINE {}".format(index))
        contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    return contents


def _request_config():
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=TEMPERATURE,
    )


# --------------------------------------------------------------------------
# Retry machinery
# --------------------------------------------------------------------------


class _RequestBudget:
    """Hard cap shared by both retry loops.

    Neither loop can outlive it, which is what stops the validation retry and
    the transport retry from multiplying into a longer sequence than either
    intends.
    """

    def __init__(self, total):
        self.total = total
        self.remaining = total

    def spend(self):
        """Claim one request. False when the cap is reached."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    @property
    def used(self):
        return self.total - self.remaining


def _sleep(seconds):
    """Indirection so tests can record backoff without actually waiting."""
    time.sleep(seconds)


def _backoff_seconds(attempt):
    """Full jitter: uniform(0, min(base * 2**attempt, max))."""
    ceiling = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
    return random.uniform(0, ceiling)


def _is_transient_status(status):
    """408, 429, and every 5xx are worth retrying. Other 4xx are not."""
    if status in TRANSIENT_STATUS_CODES:
        return True
    return SERVER_ERROR_MIN <= status <= SERVER_ERROR_MAX


def _as_vlm_api_error(exc, secret=None):
    """Map any provider/transport exception onto VLMAPIError.

    This is the containment boundary: nothing raw from google.genai or httpx
    gets past it. Unrecognised exception types are classified non-retryable
    on purpose -- retrying an error we don't understand just multiplies it.
    """
    if isinstance(exc, VLMError):
        # Already ours. Preserve its code, status and retryability.
        return exc

    status = getattr(exc, "code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = None

    if status is not None:
        retryable = _is_transient_status(status)
    else:
        # No HTTP status: timeouts, connection resets, DNS failures. httpx
        # is a hard dependency of google-genai, so this import is safe.
        import httpx

        retryable = isinstance(exc, httpx.TransportError)

    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    return VLMAPIError(
        "Gemini request failed: {}".format(_redact(message, secret)),
        status_code=status,
        retryable=retryable,
    )


def _generate_with_transport_retries(client, model, contents, budget, secret=None):
    """One logical hosted call, with bounded transient retries.

    Raises VLMAPIError; returns the raw SDK response on success. The caller
    is responsible for recording budget.used even on the raising path -- see
    the try/finally in _resolve_batch().
    """
    last_error = None
    for attempt in range(MAX_TRANSPORT_ATTEMPTS):
        if not budget.spend():
            # Cap reached mid-retry. Surface the failure that got us here
            # rather than pretending the batch succeeded.
            raise last_error or VLMAPIError(
                "Gemini request budget exhausted after {} requests".format(
                    budget.total
                ),
                retryable=True,
            )
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=_request_config()
            )
        except Exception as exc:  # noqa: BLE001 -- deliberate containment boundary
            error = _as_vlm_api_error(exc, secret)
            if not error.retryable:
                raise error from exc
            last_error = error
            logger.warning(
                "transient Gemini failure (status=%s), attempt %d/%d",
                error.status_code,
                attempt + 1,
                MAX_TRANSPORT_ATTEMPTS,
            )
            if attempt + 1 < MAX_TRANSPORT_ATTEMPTS and budget.remaining > 0:
                _sleep(_backoff_seconds(attempt))

    raise last_error


# --------------------------------------------------------------------------
# Response validation
#
# Nothing below trusts the provider, schema enforcement notwithstanding.
# --------------------------------------------------------------------------


def _parse_payload(response):
    """Pull the JSON array out of an SDK response object."""
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise VLMResponseError("Gemini returned an empty response body")
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise VLMResponseError("Gemini returned unparseable JSON") from exc


def _clean_text(value, field):
    """None / blank -> None; str -> stripped. Any other type is malformed."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise VLMResponseError(
            "field {!r} had type {}, expected string or null".format(
                field, type(value).__name__
            )
        )
    stripped = value.strip()
    return stripped or None


def _validate_entries(payload, expected_indices):
    """Validate and reconcile a batch payload against the crops we asked about.

    Returns {index: entry}. Raises VLMResponseError for anything untrustworthy.

    Each object must carry exactly RESPONSE_ENTRY_KEYS. A missing key is not
    read as a null and an extra key is not ignored -- both mean the model
    departed from the contract, and both earn the stricter retry.

    Index integrity is one comparison: the multiset of returned indices must
    equal the multiset of expected ones. That single check covers missing,
    duplicated and out-of-range indices, so there is no "first duplicate
    wins" rule and no silent dropping -- all three are malformed, all three
    earn the stricter retry.
    """
    if not isinstance(payload, list):
        raise VLMResponseError(
            "expected a JSON array, got {}".format(type(payload).__name__)
        )

    seen = []
    entries = {}
    for item in payload:
        if not isinstance(item, dict):
            raise VLMResponseError(
                "array element was {}, expected an object".format(type(item).__name__)
            )

        keys = set(item)
        if keys != RESPONSE_ENTRY_KEYS:
            raise VLMResponseError(
                "object had keys {}, expected exactly {} (missing: {}, unexpected: {})".format(
                    sorted(keys),
                    sorted(RESPONSE_ENTRY_KEYS),
                    sorted(RESPONSE_ENTRY_KEYS - keys),
                    sorted(keys - RESPONSE_ENTRY_KEYS),
                )
            )

        index = item["index"]
        # bool is an int subclass in Python; True must not pass as index 1.
        if isinstance(index, bool) or not isinstance(index, int):
            raise VLMResponseError(
                "field 'index' had type {}, expected integer".format(
                    type(index).__name__
                )
            )

        legible = item["legible"]
        if not isinstance(legible, bool):
            raise VLMResponseError(
                "field 'legible' had type {}, expected boolean".format(
                    type(legible).__name__
                )
            )

        title = _clean_text(item["title"], "title")
        author = _clean_text(item["author"], "author")

        # A legible spine with nothing read off it is a contradiction, but a
        # benign one: downgrade rather than reject the whole batch, since the
        # useful fact (nothing was readable) is intact either way.
        if legible and title is None and author is None:
            legible = False

        seen.append(index)
        entries[index] = {
            "index": index,
            "title": title,
            "author": author,
            "legible": legible,
        }

    if sorted(seen) != sorted(expected_indices):
        raise VLMResponseError(
            "index set mismatch: expected {}, got {}".format(
                sorted(expected_indices), sorted(seen)
            )
        )
    return entries


# --------------------------------------------------------------------------
# Development cache
#
# A dev-time convenience only: correctness never depends on a hit, and a hit
# is re-validated through the same path as a live response.
# --------------------------------------------------------------------------


def _cache_key(encoded, model):
    """SHA-256 over image bytes + indices + model + prompt/schema version.

    Includes enough request context that changing the model, the prompt or
    the schema misses rather than replaying an answer produced under
    different instructions. The API key is never part of the key.
    """
    material = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "schema": json.dumps(RESPONSE_SCHEMA, sort_keys=True),
        "crops": [
            {"index": index, "sha256": hashlib.sha256(jpeg).hexdigest()}
            for index, jpeg in encoded
        ],
    }
    blob = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key):
    return Path(CACHE_DIR) / "{}.json".format(key)


def _cache_read(key):
    """Cached document, or None. Corruption is a miss, never an exception."""
    try:
        path = _cache_path(key)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            return None
        return document
    except Exception:
        logger.debug("VLM cache read failed; treating as a miss", exc_info=True)
        return None


def _cache_write(key, payload, usage):
    """Best-effort write of a validated batch. Never raises, never blocks a
    successful call. Only the model's answer is stored -- no credentials."""
    try:
        directory = Path(CACHE_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        document = {
            "prompt_version": PROMPT_VERSION,
            "payload": payload,
            "usage": usage,
        }
        with _cache_path(key).open("w", encoding="utf-8") as handle:
            json.dump(document, handle)
    except Exception:
        logger.debug("VLM cache write failed; continuing", exc_info=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _usage_dict(response):
    """Provider token usage, when the SDK reports it. None otherwise.

    Raw counts only. No cost arithmetic happens in this module -- README
    numbers come from measured requests, not from constants invented here.

    total_token_count is preserved exactly as reported by the provider rather
    than recomputed from individual categories. Provider totals may include
    additional usage categories such as thoughts_token_count or
    tool_use_prompt_token_count.

    Scalar counts only; the *_tokens_details breakdowns are deliberately left
    out until something needs them.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
        "cached_content_token_count",
        "total_token_count",
    )
    collected = {
        field: getattr(usage, field, None)
        for field in fields
        if getattr(usage, field, None) is not None
    }
    return collected or None


def _new_metrics(crops_total):
    return {
        "latency_ms": 0.0,
        "cache_hit": False,
        "crops_total": crops_total,
        "crops_valid": 0,
        "requests_made": 0,
        "validation_attempts": 0,
        "model": None,
        "usage": None,
    }


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------


def read_spines(crops):
    """Read title + author off each spine crop. THE public entry point.

    Args:
        crops: list of BGR numpy arrays, as returned by vision.crop_spines().

    Returns:
        One dict per input crop, sorted by index:
        {"index": int, "title": str|None, "author": str|None, "legible": bool},
        with an extra "error": "invalid_crop" key on crops that could not be
        encoded. The index set always matches range(len(crops)) exactly.

    Raises:
        VLMConfigurationError, VLMAPIError, VLMResponseError -- all subclasses
        of VLMError. A systemic failure raises; it is never returned as a list
        of unreadable books.
    """
    results, _ = read_spines_detailed(crops)
    return results


def read_spines_detailed(crops):
    """read_spines() plus a metrics dict, for latency/cost measurement.

    Returns (results, metrics). On failure the raised VLMError carries the
    partial metrics on .metrics -- including the true number of hosted
    requests spent -- so a failed call is still measurable.
    """
    crops = list(crops or [])
    metrics = _new_metrics(len(crops))
    if not crops:
        # No crops -> no client, no config requirement, no request.
        return [], metrics

    encoded = []
    invalid_indices = []
    for index, crop in enumerate(crops):
        jpeg = _encode_crop(crop)
        if jpeg is None:
            invalid_indices.append(index)
        else:
            encoded.append((index, jpeg))
    metrics["crops_valid"] = len(encoded)

    if not encoded:
        # Every crop was unusable. Still no request -- but every index is
        # accounted for.
        return [invalid_crop_entry(index) for index in invalid_indices], metrics

    try:
        api_key, model = _load_config()
    except VLMError as exc:
        exc.metrics = metrics
        raise
    metrics["model"] = model
    expected_indices = [index for index, _ in encoded]

    started = time.perf_counter()
    try:
        entries = _resolve_batch(api_key, model, encoded, expected_indices, metrics)
    except VLMError as exc:
        metrics["latency_ms"] = (time.perf_counter() - started) * 1000.0
        exc.metrics = metrics
        raise
    metrics["latency_ms"] = (time.perf_counter() - started) * 1000.0

    for index in invalid_indices:
        entries[index] = invalid_crop_entry(index)
    return [entries[index] for index in sorted(entries)], metrics


def _resolve_batch(api_key, model, encoded, expected_indices, metrics):
    """Cache lookup, else the live request loop. Returns {index: entry}."""
    key = _cache_key(encoded, model)
    document = _cache_read(key)
    if document is not None:
        try:
            # A cached payload goes through the identical validation path as
            # a live one -- a stale or hand-edited cache file cannot inject
            # anything a live response couldn't.
            entries = _validate_entries(document.get("payload"), expected_indices)
        except VLMResponseError:
            logger.debug("cached VLM payload failed validation; treating as a miss")
        else:
            metrics["cache_hit"] = True
            metrics["usage"] = document.get("usage")
            return entries

    client = _build_client_guarded(api_key)
    budget = _RequestBudget(MAX_TOTAL_REQUESTS)
    last_response_error = None

    for attempt in range(MAX_VALIDATION_ATTEMPTS):
        metrics["validation_attempts"] = attempt + 1
        contents = _build_contents(encoded, strict=attempt > 0)

        # try/finally, not a trailing assignment: when transport retries are
        # exhausted this helper raises, and a failed call still has to report
        # the hosted calls it actually spent.
        try:
            response = _generate_with_transport_retries(
                client, model, contents, budget, api_key
            )
        finally:
            metrics["requests_made"] = budget.used

        payload = None
        try:
            payload = _parse_payload(response)
            entries = _validate_entries(payload, expected_indices)
        except VLMResponseError as exc:
            last_response_error = exc
            logger.warning(
                "malformed Gemini batch (%s), validation attempt %d/%d",
                exc.message,
                attempt + 1,
                MAX_VALIDATION_ATTEMPTS,
            )
            if budget.remaining <= 0:
                break
            continue

        metrics["usage"] = _usage_dict(response)
        _cache_write(key, payload, metrics["usage"])
        return entries

    raise last_response_error
