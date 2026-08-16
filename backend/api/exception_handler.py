"""DRF exception handler: structured VLM failures -> deliberate HTTP statuses.

The boundary between "the hosted model went wrong" and "the client sees an
HTTP error". Three rules hold here:

  * The VLM exception classes are the interface. Nothing parses provider
    exception strings or sniffs for substrings -- vlm/ already classified
    the failure, and re-deriving that here would mean two disagreeing
    classifiers.

  * Response bodies carry server-owned prose only. exc.message can contain
    provider response text, so it never reaches a client. It is logged at
    DEBUG for operators, which is safe because vlm/gemini.py redacts the API
    key before any provider text enters the exception.

  * Only VLMErrors are handled. Everything else is delegated to DRF, which
    already turns ValidationError into 400, Http404 into 404, and so on.
    Catching everything here would flatten those into a generic 500.
"""

import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from vlm import (
    VLMAPIError,
    VLMConfigurationError,
    VLMError,
    VLMResponseError,
)

logger = logging.getLogger(__name__)


# Seconds. Sent only with the transient 503: the request may work later.
RETRY_AFTER_SECONDS = 30

# Server-owned prose, keyed by the structured code. Deliberately says what
# the user can do, not what the provider said.
SAFE_MESSAGES = {
    "configuration": (
        "The book reading service is not configured on this server. "
        "Nothing was lost -- ask the operator to check the server setup."
    ),
    "api_failure_retryable": (
        "The book reading service is temporarily unavailable. "
        "Your photo was not processed -- please try again in a moment."
    ),
    "api_failure_permanent": (
        "The book reading service rejected this request and retrying will "
        "not help. Your photo was not processed."
    ),
    "invalid_response": (
        "The book reading service returned a response we could not read. "
        "Your photo was not processed -- please try again."
    ),
}

DEFAULT_MESSAGE = "The book reading service failed. Your photo was not processed."


def _resolve(exc):
    """(http_status, message, send_retry_after) for a VLMError.

    Configuration failures are 503 without Retry-After: the request is fine
    and will work once an operator fixes the server, but no amount of client
    retrying changes that, so promising a retry window would be a lie.

    A retryable API failure is 503 *with* Retry-After -- our bounded retries
    were exhausted, but the same request may genuinely succeed later.

    A non-retryable API failure (bad key, bad model id, oversized payload)
    and an unusable response are both 502: the upstream is broken in a way
    the client cannot fix by waiting.
    """
    if isinstance(exc, VLMConfigurationError):
        return 503, SAFE_MESSAGES["configuration"], False
    if isinstance(exc, VLMResponseError):
        return 502, SAFE_MESSAGES["invalid_response"], False
    if isinstance(exc, VLMAPIError):
        if exc.retryable:
            return 503, SAFE_MESSAGES["api_failure_retryable"], True
        return 502, SAFE_MESSAGES["api_failure_permanent"], False
    # An unrecognised VLMError subclass. Treated as a broken upstream rather
    # than assumed retryable -- fail towards "don't hammer the provider".
    return 502, DEFAULT_MESSAGE, False


def shelfie_exception_handler(exc, context):
    """Entry point named by REST_FRAMEWORK["EXCEPTION_HANDLER"]."""
    if not isinstance(exc, VLMError):
        # Not ours: DRF's own mapping is correct and more complete than
        # anything we would write. It returns None for exceptions it doesn't
        # recognise, which lets Django produce the 500 it should.
        return drf_exception_handler(exc, context)

    http_status, message, send_retry_after = _resolve(exc)
    request = context.get("request")
    metrics = exc.metrics or {}

    # Structured fields only: what failed, how we classified it, and what it
    # cost. No provider prose at this level.
    logger.error(
        "VLM failure: error_class=%s code=%s retryable=%s provider_status=%s "
        "http_status=%s path=%s requests_made=%s crops=%s latency_ms=%s",
        type(exc).__name__,
        exc.code,
        exc.retryable,
        exc.status_code,
        http_status,
        getattr(request, "path", None),
        metrics.get("requests_made"),
        metrics.get("crops_total"),
        metrics.get("latency_ms"),
    )

    # Provider text, operators only. Safe to log: vlm/gemini.py's _redact()
    # strips the API key before it can reach exc.message. Never returned.
    logger.debug(
        "VLM failure detail (provider text, redacted upstream): %s",
        exc.message,
    )

    response = Response(
        {
            "error": {
                "code": exc.code,
                "message": message,
                # The client's cue for whether a retry button makes sense.
                "retryable": bool(exc.retryable),
            }
        },
        status=http_status,
    )

    if send_retry_after:
        response["Retry-After"] = str(RETRY_AFTER_SECONDS)

    return response
