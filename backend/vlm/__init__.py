"""Hosted VLM stage: spine crops in, {title, author, legible} reads out.

Pure Python + opencv-python + google-genai. Zero Django imports, so it can be
exercised standalone. The provider lives entirely behind read_spines().

    from vlm import read_spines, VLMError

    try:
        reads = read_spines(crops)
    except VLMError as exc:
        ...  # structured: exc.code, exc.message, exc.retryable, exc.status_code

Systemic failures (no config, provider down, retries exhausted, a batch still
malformed after one retry) raise. Only an individually unusable crop degrades
in-band, as a preserved entry carrying "error": "invalid_crop".
"""

from .errors import (
    CODE_API_FAILURE,
    CODE_CONFIGURATION,
    CODE_INVALID_CROP,
    CODE_INVALID_RESPONSE,
    VLMAPIError,
    VLMConfigurationError,
    VLMError,
    VLMResponseError,
)
from .gemini import read_spines, read_spines_detailed

__all__ = [
    "read_spines",
    "read_spines_detailed",
    "VLMError",
    "VLMConfigurationError",
    "VLMAPIError",
    "VLMResponseError",
    "CODE_CONFIGURATION",
    "CODE_API_FAILURE",
    "CODE_INVALID_RESPONSE",
    "CODE_INVALID_CROP",
]
