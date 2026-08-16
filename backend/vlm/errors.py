"""Structured VLM error hierarchy.

Deliberately free of any provider SDK import, so the Django layer can import
these to build an HTTP error response without pulling google-genai into the
web process.

The distinction these types encode is the whole point of this module:

  * A *systemic* failure -- no credentials, provider outage, exhausted
    retries, a batch response that is still malformed after a retry -- raises.
    It must never be laundered into N ordinary `legible: false` book results,
    because "we could not reach the model" and "this spine is unreadable" are
    different facts and the user deserves different treatment for each.

  * A *per-crop* failure (one unusable crop in an otherwise fine batch) does
    not raise. It degrades to a preserved result entry carrying
    `"error": "invalid_crop"`. See gemini.invalid_crop_entry().

Every failure path in this package terminates in one of these types; no raw
google.genai exception escapes.
"""

CODE_CONFIGURATION = "configuration"
CODE_API_FAILURE = "api_failure"
CODE_INVALID_RESPONSE = "invalid_response"

# Per-crop degradation marker. Not an exception -- it is the value of the
# "error" key on a preserved result entry. Lives here so callers have one
# place to import every VLM error identifier from.
CODE_INVALID_CROP = "invalid_crop"


class VLMError(Exception):
    """Base for every error this package raises.

    Attributes:
        code:        stable machine-readable identifier, one of the CODE_*
                     constants above. Safe to branch on; safe to expose.
        message:     human-readable description. Never contains the API key.
        retryable:   whether retrying the same call could plausibly succeed.
                     Advice for the caller (and for the eventual HTTP status
                     choice), not something this package acts on -- our own
                     retries are already exhausted by the time we raise.
        status_code: provider HTTP status when there was one, else None.
        metrics:     partial measurements from the failed call, so a failure
                     is still measurable for the README latency numbers.
    """

    code = "vlm_error"
    retryable = False

    def __init__(self, message, status_code=None, retryable=None, metrics=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
        self.metrics = metrics

    def as_dict(self):
        """Serializable form, for the structured HTTP error Django will build."""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }

    def __repr__(self):
        return "{}(code={!r}, status_code={!r}, retryable={!r}, message={!r})".format(
            type(self).__name__,
            self.code,
            self.status_code,
            self.retryable,
            self.message,
        )


class VLMConfigurationError(VLMError):
    """Missing or empty GEMINI_API_KEY / GEMINI_MODEL.

    A deployment mistake, not a transient condition: never retried, and raised
    before any client is constructed or any request is sent.
    """

    code = CODE_CONFIGURATION
    retryable = False


class VLMAPIError(VLMError):
    """The hosted provider call failed.

    Permanent (401/403/400/404/413, other non-listed 4xx) -> raised on the
    first failure with retryable=False. Transient (408/429/5xx, transport
    timeouts) -> raised only once the bounded retry budget is spent, with
    retryable=True to record that the failure was of the retryable kind and
    we tried.
    """

    code = CODE_API_FAILURE


class VLMResponseError(VLMError):
    """The provider answered, but the batch could not be trusted.

    Covers unparseable JSON, wrong field types, and -- equally -- any index
    set that does not exactly match the crops we asked about (missing,
    duplicated, or out-of-range indices). Raised only after the stricter
    re-prompt also came back malformed.
    """

    code = CODE_INVALID_RESPONSE
    retryable = False
