"""Text normalization for catalog matching.

Pure stdlib. No Django, no I/O, no global state.

The rule that governs everything here: normalization is *information
preserving*. We fold away noise that OCR/VLM reads introduce (case, accents,
punctuation, spacing) but we never delete words. In particular there is no
stopword removal -- dropping "the"/"of"/"a" would collapse genuinely different
titles onto each other. Leading articles are handled in scoring instead, by
comparing the query both with and without its article.
"""

import re
import unicodedata

# Anything that is not a word character or whitespace becomes a space, so
# "J.K." -> "j k " and "Mistborn: The Final Empire" -> "mistborn the final empire".
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_UNDERSCORE_RE = re.compile(r"_+")
_WHITESPACE_RE = re.compile(r"\s+")

LEADING_ARTICLES = ("the", "a", "an")


def _strip_accents(text):
    """NFKD decompose, then drop combining marks: 'Brontë' -> 'Bronte'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_title(s):
    """Normalize a title for comparison.

    NFKD -> strip accents -> casefold -> '&' to 'and' -> punctuation to
    spaces -> collapse whitespace. No words are removed.
    """
    if s is None:
        return ""
    text = _strip_accents(str(s)).casefold()
    text = text.replace("&", " and ")
    text = _UNDERSCORE_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_leading_article(s):
    """Drop a single leading article from an already-normalized string.

    Returns the input unchanged if there is no leading article, or if the
    article is the only token ("The" stays "the" rather than becoming "").
    """
    if not s:
        return ""
    tokens = s.split()
    if len(tokens) > 1 and tokens[0] in LEADING_ARTICLES:
        return " ".join(tokens[1:])
    return s


def normalize_author(s):
    """Normalize an author name for comparison.

    Same character pipeline as titles, plus two author-specific steps:
      * "Lastname, Firstname" is reordered to "firstname lastname" so that
        catalog aliases in either order compare equal.
      * Initials are separated by the punctuation rule, so "J.K." and
        "J. K." both become "j k".
    """
    if s is None:
        return ""
    text = str(s)
    head, sep, tail = text.partition(",")
    if sep and head.strip() and tail.strip():
        text = f"{tail.strip()} {head.strip()}"
    return normalize_title(text)
