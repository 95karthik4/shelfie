"""Offline tests for the hosted VLM layer.

Every test in this file runs without network access and without credentials.
The Gemini client is replaced with a fake whose queued outcomes we control,
and the autouse fixture below makes constructing a *real* client an outright
test failure -- so "did this accidentally call the API?" is answered by the
suite, not by watching a quota dashboard.

What is covered: index integrity and response-schema integrity (the
malformed-batch path), the split between systemic failures that raise and
per-crop failures that degrade, retry bounds, and the development cache.
"""

import json

import numpy as np
import pytest
from google.genai import errors as genai_errors

from vlm import errors as vlm_errors
from vlm import gemini
from vlm.errors import VLMAPIError, VLMConfigurationError, VLMResponseError


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeResponse:
    """Stand-in for GenerateContentResponse: only .text and .usage_metadata
    are ever read from it."""

    def __init__(self, text, usage=None):
        self.text = text
        self.usage_metadata = usage


class _FakeUsage:
    """Stand-in for GenerateContentResponseUsageMetadata.

    The values are synthetic, and the total is deliberately NOT the sum of the
    visible scalar fields (100 + 20 + 30 + 5 = 155, not 177). Provider totals
    may include usage categories we don't enumerate, so anything that derived
    the total instead of preserving what the provider reported would fail
    these tests.
    """

    def __init__(
        self,
        prompt=100,
        candidates=20,
        thoughts=30,
        tool_use=5,
        cached=None,
        total=177,
    ):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.tool_use_prompt_token_count = tool_use
        self.cached_content_token_count = cached
        self.total_token_count = total


class _FakeModels:
    def __init__(self, outcomes, calls):
        self._outcomes = list(outcomes)
        self.calls = calls

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._outcomes:
            raise AssertionError(
                "hosted call {} was not expected by this test".format(len(self.calls))
            )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes, calls):
        self.models = _FakeModels(outcomes, calls)


def install_client(monkeypatch, *outcomes):
    """Queue hosted-call outcomes; returns the list every call is recorded in.

    An outcome is either an exception to raise or a _FakeResponse to return.
    """
    calls = []
    monkeypatch.setattr(
        gemini, "_build_client", lambda api_key: _FakeClient(outcomes, calls)
    )
    return calls


def response(entries, usage=None):
    return _FakeResponse(json.dumps(entries), usage=usage)


def entry(index, title="Dune", author="Frank Herbert", legible=True):
    return {"index": index, "title": title, "author": author, "legible": legible}


def api_error(status, message="boom"):
    """A real google-genai error, so we test against the SDK's actual shape."""
    payload = {"error": {"code": status, "message": message, "status": "ERROR"}}
    if status >= 500:
        return genai_errors.ServerError(status, payload)
    return genai_errors.ClientError(status, payload)


def crop(fill=128, height=40, width=12):
    """A small but genuinely encodable BGR crop."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """No real credentials, no real client, no real sleeping, no real cache.

    _build_client is pointed at a tripwire: any test that reaches client
    construction without installing a fake fails loudly rather than trying
    to talk to Google.
    """
    monkeypatch.setenv(gemini.ENV_API_KEY, "test-key-not-a-real-secret")
    monkeypatch.setenv(gemini.ENV_MODEL, "test-model")
    monkeypatch.setattr(gemini, "CACHE_DIR", tmp_path / "vlm_cache")

    def _forbidden(api_key):
        raise AssertionError("test attempted to construct a real Gemini client")

    monkeypatch.setattr(gemini, "_build_client", _forbidden)
    return tmp_path


@pytest.fixture
def slept(monkeypatch):
    """Records backoff durations instead of waiting."""
    recorded = []
    monkeypatch.setattr(gemini, "_sleep", recorded.append)
    return recorded


# --------------------------------------------------------------------------
# No-request paths
# --------------------------------------------------------------------------


def test_empty_input_returns_empty_and_never_builds_a_client():
    # The tripwire in the autouse fixture is the assertion here.
    assert gemini.read_spines([]) == []
    assert gemini.read_spines(None) == []


def test_empty_input_does_not_require_configuration(monkeypatch):
    monkeypatch.delenv(gemini.ENV_API_KEY, raising=False)
    monkeypatch.delenv(gemini.ENV_MODEL, raising=False)
    assert gemini.read_spines([]) == []


def test_all_crops_invalid_makes_no_call_and_preserves_every_index():
    results = gemini.read_spines([None, "not an array", np.zeros((0, 5, 3))])

    assert [r["index"] for r in results] == [0, 1, 2]
    assert all(r["legible"] is False for r in results)
    assert all(r["title"] is None and r["author"] is None for r in results)
    assert all(r["error"] == vlm_errors.CODE_INVALID_CROP for r in results)


# --------------------------------------------------------------------------
# Per-crop degradation vs. systemic failure
# --------------------------------------------------------------------------


def test_invalid_crop_among_valid_ones_is_preserved_and_others_are_read(monkeypatch):
    calls = install_client(
        monkeypatch,
        response([entry(0, "Dune"), entry(2, "Neuromancer", "William Gibson")]),
    )

    results = gemini.read_spines([crop(10), None, crop(30)])

    assert len(calls) == 1
    assert [r["index"] for r in results] == [0, 1, 2]
    assert results[0]["title"] == "Dune"
    assert results[1]["error"] == vlm_errors.CODE_INVALID_CROP
    assert results[2]["title"] == "Neuromancer"
    # The degradation marker belongs to the bad crop only.
    assert "error" not in results[0] and "error" not in results[2]


def test_labels_use_original_indices_so_gaps_are_visible(monkeypatch):
    calls = install_client(monkeypatch, response([entry(0), entry(2)]))

    gemini.read_spines([crop(10), None, crop(30)])

    labels = [
        c for c in calls[0]["contents"] if isinstance(c, str) and c.startswith("SPINE")
    ]
    # Index 1 was dropped before the request; the remaining crops keep their
    # original positions rather than being renumbered 0, 1.
    assert labels == ["SPINE 0", "SPINE 2"]


def test_results_are_reconciled_by_index_not_by_response_order(monkeypatch):
    install_client(
        monkeypatch,
        response(
            [
                entry(2, "Neuromancer", "William Gibson"),
                entry(0, "Dune", "Frank Herbert"),
                entry(1, "Emma", "Jane Austen"),
            ]
        ),
    )

    results = gemini.read_spines([crop(10), crop(20), crop(30)])

    assert [r["index"] for r in results] == [0, 1, 2]
    assert [r["title"] for r in results] == ["Dune", "Emma", "Neuromancer"]


def test_unreadable_spine_is_a_normal_result_not_an_error(monkeypatch):
    install_client(
        monkeypatch,
        response(
            [entry(0, None, None, legible=False), entry(1, "Emma", "Jane Austen")]
        ),
    )

    results = gemini.read_spines([crop(10), crop(20)])

    assert results[0] == {"index": 0, "title": None, "author": None, "legible": False}
    assert results[1]["legible"] is True


def test_legible_true_with_nothing_read_is_downgraded(monkeypatch):
    """A contradiction, but a benign one: don't reject the whole batch over it."""
    install_client(monkeypatch, response([entry(0, "   ", None, legible=True)]))

    results = gemini.read_spines([crop(10)])

    assert results[0]["legible"] is False
    assert results[0]["title"] is None


# --------------------------------------------------------------------------
# Index and schema integrity -> malformed batch -> one retry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_payload, description",
    [
        # --- index integrity ---
        ([entry(0)], "missing index 1"),
        ([entry(0), entry(0)], "duplicate index 0"),
        ([entry(0), entry(1), entry(7)], "out-of-range index 7"),
        ([entry(0), entry(1), entry(2)], "extra in-range index 2"),
        # --- exact key set ---
        ([entry(0), {"index": 1, "author": None, "legible": True}], "missing 'title'"),
        ([entry(0), {"index": 1, "title": "x", "legible": True}], "missing 'author'"),
        ([entry(0), {"legible": True}], "only one key present"),
        (
            [entry(0), dict(entry(1), confidence=0.9)],
            "unexpected extra key 'confidence'",
        ),
        ([entry(0), dict(entry(1), spine_index=1)], "unexpected extra key alongside index"),
        # --- field types ---
        (
            [entry(0), {"index": 1, "title": 5, "author": None, "legible": True}],
            "title not a string",
        ),
        (
            [entry(0), {"index": 1, "title": "x", "author": None, "legible": "yes"}],
            "legible not a bool",
        ),
        (
            [entry(0), {"index": "1", "title": "x", "author": None, "legible": True}],
            "index not an int",
        ),
        (
            [entry(0), {"index": True, "title": "x", "author": None, "legible": True}],
            "index is a bool masquerading as int 1",
        ),
        # --- shape ---
        ([entry(0), "not an object"], "element not an object"),
        ({"spines": []}, "payload not an array"),
    ],
)
def test_malformed_batch_retries_once_then_succeeds(
    monkeypatch, bad_payload, description
):
    calls = install_client(
        monkeypatch,
        response(bad_payload),
        response([entry(0, "Dune"), entry(1, "Emma", "Jane Austen")]),
    )

    results = gemini.read_spines([crop(10), crop(20)])

    assert len(calls) == 2, description
    assert [r["index"] for r in results] == [0, 1]
    assert results[1]["title"] == "Emma"


def test_retry_uses_the_stricter_prompt_listing_the_exact_indices(monkeypatch):
    calls = install_client(
        monkeypatch,
        response([entry(0)]),
        response([entry(0), entry(2)]),
    )

    gemini.read_spines([crop(10), None, crop(30)])

    first_prompt = calls[0]["contents"][0]
    retry_prompt = calls[1]["contents"][0]
    assert "rejected as malformed" not in first_prompt
    assert "rejected as malformed" in retry_prompt
    assert "0, 2" in retry_prompt


def test_second_malformed_response_raises_structured_error(monkeypatch):
    calls = install_client(
        monkeypatch,
        response([entry(0)]),
        response([entry(0)]),
    )

    with pytest.raises(VLMResponseError) as excinfo:
        gemini.read_spines([crop(10), crop(20)])

    error = excinfo.value
    assert error.code == vlm_errors.CODE_INVALID_RESPONSE
    assert error.retryable is False
    # Exactly one retry -- not a third attempt.
    assert len(calls) == 2
    assert error.metrics["requests_made"] == 2
    assert error.metrics["validation_attempts"] == 2


def test_unparseable_json_is_treated_as_malformed(monkeypatch):
    calls = install_client(
        monkeypatch,
        _FakeResponse("here you go: ```json [{"),
        response([entry(0)]),
    )

    results = gemini.read_spines([crop(10)])

    assert len(calls) == 2
    assert results[0]["title"] == "Dune"


def test_empty_response_body_is_treated_as_malformed(monkeypatch):
    calls = install_client(monkeypatch, _FakeResponse(None), response([entry(0)]))

    gemini.read_spines([crop(10)])

    assert len(calls) == 2


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [gemini.ENV_API_KEY, gemini.ENV_MODEL])
def test_missing_configuration_raises_before_any_request(monkeypatch, missing):
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(VLMConfigurationError) as excinfo:
        gemini.read_spines([crop(10)])

    error = excinfo.value
    assert error.code == vlm_errors.CODE_CONFIGURATION
    assert error.retryable is False
    assert missing in error.message
    assert error.metrics["requests_made"] == 0


def test_blank_configuration_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv(gemini.ENV_MODEL, "   ")

    with pytest.raises(VLMConfigurationError):
        gemini.read_spines([crop(10)])


def test_configuration_error_never_becomes_unreadable_books(monkeypatch):
    """The whole point of raising: three crops must not come back as three
    silently 'unreadable' spines when the real problem is a missing key."""
    monkeypatch.delenv(gemini.ENV_API_KEY, raising=False)

    with pytest.raises(VLMConfigurationError):
        gemini.read_spines([crop(10), crop(20), crop(30)])


def test_client_construction_failure_is_wrapped_and_redacted(monkeypatch):
    secret = "super-secret-key-value"
    monkeypatch.setenv(gemini.ENV_API_KEY, secret)

    def _explode(api_key):
        raise ValueError("bad credential {}".format(api_key))

    monkeypatch.setattr(gemini, "_build_client", _explode)

    with pytest.raises(VLMAPIError) as excinfo:
        gemini.read_spines([crop(10)])

    error = excinfo.value
    assert error.retryable is False
    assert secret not in error.message
    assert "***" in error.message


# --------------------------------------------------------------------------
# Transport failures and retry bounds
# --------------------------------------------------------------------------


def test_permanent_client_error_is_not_retried(monkeypatch, slept):
    calls = install_client(monkeypatch, api_error(401, "invalid api key"))

    with pytest.raises(VLMAPIError) as excinfo:
        gemini.read_spines([crop(10)])

    error = excinfo.value
    assert error.status_code == 401
    assert error.retryable is False
    assert len(calls) == 1
    assert slept == []
    assert error.metrics["requests_made"] == 1


def test_transient_error_is_retried_up_to_the_transport_limit(monkeypatch, slept):
    calls = install_client(monkeypatch, api_error(429), api_error(429), api_error(429))

    with pytest.raises(VLMAPIError) as excinfo:
        gemini.read_spines([crop(10)])

    error = excinfo.value
    assert error.status_code == 429
    assert error.retryable is True
    assert len(calls) == gemini.MAX_TRANSPORT_ATTEMPTS == 3
    # Backoff between attempts, not after the last one.
    assert len(slept) == 2
    assert all(0 <= s <= gemini.BACKOFF_MAX_SECONDS for s in slept)


def test_failed_call_reports_the_requests_it_actually_spent(monkeypatch, slept):
    """Regression guard: the metric must survive the raising path, or a failed
    call reports zero hosted calls and the cost numbers become fiction."""
    install_client(monkeypatch, api_error(503), api_error(503), api_error(503))

    with pytest.raises(VLMAPIError) as excinfo:
        gemini.read_spines([crop(10)])

    assert excinfo.value.metrics["requests_made"] == 3
    assert excinfo.value.metrics["latency_ms"] > 0


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 507, 599])
def test_timeouts_rate_limits_and_all_5xx_count_as_transient(monkeypatch, slept, status):
    calls = install_client(monkeypatch, api_error(status), response([entry(0)]))

    results = gemini.read_spines([crop(10)])

    assert len(calls) == 2
    assert results[0]["title"] == "Dune"


@pytest.mark.parametrize("status", [400, 403, 404, 413, 422])
def test_non_transient_4xx_is_not_retried(monkeypatch, slept, status):
    calls = install_client(monkeypatch, api_error(status))

    with pytest.raises(VLMAPIError):
        gemini.read_spines([crop(10)])

    assert len(calls) == 1
    assert slept == []


def test_transport_error_without_a_status_is_retried(monkeypatch, slept):
    import httpx

    calls = install_client(
        monkeypatch, httpx.ConnectTimeout("timed out"), response([entry(0)])
    )

    results = gemini.read_spines([crop(10)])

    assert len(calls) == 2
    assert results[0]["title"] == "Dune"


def test_unrecognised_exception_is_wrapped_and_not_retried(monkeypatch, slept):
    calls = install_client(monkeypatch, RuntimeError("something odd"))

    with pytest.raises(VLMAPIError) as excinfo:
        gemini.read_spines([crop(10)])

    assert excinfo.value.retryable is False
    assert len(calls) == 1


def test_transient_then_success(monkeypatch, slept):
    calls = install_client(monkeypatch, api_error(429), response([entry(0)]))

    results = gemini.read_spines([crop(10)])

    assert len(calls) == 2
    assert results[0]["title"] == "Dune"
    assert len(slept) == 1


def test_total_request_cap_bounds_the_two_retry_axes(monkeypatch, slept):
    """Malformed batch, then transient failures on the retry: the two loops
    must not multiply past MAX_TOTAL_REQUESTS."""
    calls = install_client(
        monkeypatch,
        response([entry(0)]),  # malformed: index 1 missing -> validation retry
        api_error(429),
        api_error(429),
        api_error(429),
        response([entry(0), entry(1)]),  # would succeed, must never be reached
    )

    with pytest.raises(VLMAPIError):
        gemini.read_spines([crop(10), crop(20)])

    assert len(calls) == gemini.MAX_TOTAL_REQUESTS == 4


# --------------------------------------------------------------------------
# Usage metrics
# --------------------------------------------------------------------------


def test_usage_metadata_preserves_every_reported_scalar_field(monkeypatch):
    install_client(monkeypatch, response([entry(0)], usage=_FakeUsage()))

    _, metrics = gemini.read_spines_detailed([crop(10)])

    # cached_content_token_count was None and is omitted rather than reported
    # as 0 -- "not reported" and "zero" are different facts.
    assert metrics["usage"] == {
        "prompt_token_count": 100,
        "candidates_token_count": 20,
        "thoughts_token_count": 30,
        "tool_use_prompt_token_count": 5,
        "total_token_count": 177,
    }


def test_total_token_count_is_provider_reported_not_derived(monkeypatch):
    """The total is whatever the provider says it is.

    Deriving it from the categories we happen to enumerate would silently
    undercount any category we don't -- and the token count is what the
    README's cost estimate is built on.
    """
    install_client(monkeypatch, response([entry(0)], usage=_FakeUsage()))

    _, metrics = gemini.read_spines_detailed([crop(10)])
    usage = metrics["usage"]

    assert usage["total_token_count"] == 177
    assert usage["total_token_count"] != (
        usage["prompt_token_count"] + usage["candidates_token_count"]
    )
    # Not the sum of every scalar we collect, either.
    assert usage["total_token_count"] != sum(
        value for key, value in usage.items() if key != "total_token_count"
    )


# --------------------------------------------------------------------------
# Development cache
# --------------------------------------------------------------------------


def test_cache_hit_replays_without_a_hosted_call(monkeypatch):
    crops = [crop(10), crop(20)]
    payload = [entry(0, "Dune"), entry(1, "Emma", "Jane Austen")]

    first_calls = install_client(monkeypatch, response(payload, usage=_FakeUsage()))
    first, first_metrics = gemini.read_spines_detailed(crops)
    assert len(first_calls) == 1
    assert first_metrics["cache_hit"] is False
    assert first_metrics["usage"]["total_token_count"] == 177

    # No outcomes queued: any hosted call now fails the test.
    second_calls = install_client(monkeypatch)
    second, second_metrics = gemini.read_spines_detailed(crops)

    assert second_calls == []
    assert second == first
    assert second_metrics["cache_hit"] is True
    assert second_metrics["requests_made"] == 0


def test_cache_key_changes_with_the_model(monkeypatch):
    crops = [crop(10)]
    install_client(monkeypatch, response([entry(0)]))
    gemini.read_spines(crops)

    monkeypatch.setenv(gemini.ENV_MODEL, "a-different-model")
    calls = install_client(monkeypatch, response([entry(0)]))
    gemini.read_spines(crops)

    assert len(calls) == 1, "a different model must not replay the cached answer"


def test_corrupt_cache_entry_is_a_miss_not_a_crash(monkeypatch, tmp_path):
    crops = [crop(10)]
    install_client(monkeypatch, response([entry(0)]))
    gemini.read_spines(crops)

    cached = list((tmp_path / "vlm_cache").glob("*.json"))
    assert cached, "expected the successful batch to be cached"
    cached[0].write_text("{ this is not json", encoding="utf-8")

    calls = install_client(monkeypatch, response([entry(0, "Dune")]))
    results = gemini.read_spines(crops)

    assert len(calls) == 1
    assert results[0]["title"] == "Dune"


def test_cached_payload_is_revalidated_against_the_current_crops(monkeypatch, tmp_path):
    """A cache file cannot inject an index set a live response couldn't."""
    crops = [crop(10)]
    install_client(monkeypatch, response([entry(0)]))
    gemini.read_spines(crops)

    cached = list((tmp_path / "vlm_cache").glob("*.json"))[0]
    document = json.loads(cached.read_text(encoding="utf-8"))
    document["payload"] = [entry(0), entry(9, "Ghost Book")]
    cached.write_text(json.dumps(document), encoding="utf-8")

    calls = install_client(monkeypatch, response([entry(0, "Dune")]))
    results = gemini.read_spines(crops)

    assert len(calls) == 1, "an invalid cached payload must fall through to a request"
    assert [r["index"] for r in results] == [0]


def test_failures_are_never_cached(monkeypatch, slept, tmp_path):
    install_client(monkeypatch, api_error(500), api_error(500), api_error(500))

    with pytest.raises(VLMAPIError):
        gemini.read_spines([crop(10)])

    assert list((tmp_path / "vlm_cache").glob("*.json")) == []
