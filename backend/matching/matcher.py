"""Catalog matcher: one VLM spine read -> one canonical catalog entry.

Pure Python. stdlib + rapidfuzz only, zero Django imports, no hardcoded paths;
the catalog path is always passed in by the caller.

The confidence returned here is an *explainable decision score*, not a
calibrated probability. It exists to route a read into one of three buckets
(auto-add / human review / unmatched) and to explain, via `reasons`, why it
landed there.
"""

import csv

from rapidfuzz import fuzz

from .normalize import normalize_author, normalize_title, strip_leading_article

# --------------------------------------------------------------------------
# Tunable constants. Everything tunable lives here, nothing is inlined below.
# Thresholds are provisional and tuned against the tests in tests/.
# --------------------------------------------------------------------------

TITLE_WEIGHT = 0.75
AUTHOR_WEIGHT = 0.25

# Applied to a title pair where one normalized title is a proper (word-aligned)
# substring of the other, e.g. "Dune" vs "Dune Messiah". token_set_ratio scores
# such pairs 100 because one token set contains the other, so without this guard
# every "Dune*" title is a perfect match for the query "Dune".
SUBSTRING_PENALTY = 0.25

# A rival candidate is "in contention" when it scores within this margin of
# rank 1. Ambiguity is judged inside that window only.
AMBIGUITY_MARGIN = 0.15

# Ambiguity penalties, worst-wins. A rival that is a different work is far more
# dangerous than a rival that is another edition of the same work.
DIFFERENT_WORK_PENALTY = 0.30
OMNIBUS_PENALTY = 0.15
EDITION_PENALTY = 0.05

AUTO_READY_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60

# A title-only read (author unreadable) can never auto-add: the cap sits just
# below AUTO_READY_THRESHOLD on purpose.
NO_AUTHOR_CONFIDENCE_CAP = 0.84

STATUS_AUTO_READY = "AUTO_READY"
STATUS_REVIEW = "REVIEW"
STATUS_UNMATCHED = "UNMATCHED"

REASON_NOT_LEGIBLE = "NOT_LEGIBLE"
REASON_NO_TITLE_TEXT = "NO_TITLE_TEXT"
REASON_EMPTY_CATALOG = "EMPTY_CATALOG"
REASON_AUTHOR_UNREADABLE = "AUTHOR_UNREADABLE"
REASON_TITLE_ONLY_CAP = "TITLE_ONLY_CONFIDENCE_CAP"
REASON_SUBSTRING_PENALTY = "SUBSTRING_PENALTY_APPLIED"
REASON_LOW_SIMILARITY = "LOW_SIMILARITY"
REASON_DIFFERENT_WORK = "DIFFERENT_WORK_AMBIGUITY"
REASON_OMNIBUS = "OMNIBUS_AMBIGUITY"
REASON_EDITION = "EDITION_AMBIGUITY"

RELATION_SAME_WORK = "SAME_WORK"
RELATION_OMNIBUS = "OMNIBUS"
RELATION_DIFFERENT_WORK = "DIFFERENT_WORK"

# Pipe-separated multi-value columns in catalog.csv.
LIST_FIELDS = ("alternate_titles", "author_aliases", "contains_work_ids")
LIST_SEPARATOR = "|"


# --------------------------------------------------------------------------
# Catalog loading
# --------------------------------------------------------------------------


def _split_list(value):
    if not value:
        return []
    return [part.strip() for part in value.split(LIST_SEPARATOR) if part.strip()]


def load_catalog(path):
    """Load catalog.csv into a list of entry dicts.

    Pipe-separated columns become lists. Normalized forms are precomputed once
    per entry and cached on the entry under "_norm".
    """
    entries = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            entry = {
                key: (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for field in LIST_FIELDS:
                entry[field] = _split_list(entry.get(field, ""))
            entries.append(_prepare_entry(entry))
    return entries


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return _split_list(str(value))


def _dedupe(values):
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _prepare_entry(entry):
    """Return a copy of a catalog entry with cached normalized title/author
    variants attached under "_norm".

    The input dict is never mutated: matching against a caller-supplied
    in-memory catalog must leave that catalog exactly as the caller built it.
    """
    titles = [entry.get("title")] + _as_list(entry.get("alternate_titles"))
    authors = [entry.get("author")] + _as_list(entry.get("author_aliases"))
    prepared = dict(entry)
    prepared["_norm"] = {
        "titles": _dedupe(normalize_title(t) for t in titles),
        "authors": _dedupe(normalize_author(a) for a in authors),
    }
    return prepared


def _entry_norm(entry):
    """Normalized forms for an entry, computing them if the caller built the
    entry by hand instead of via load_catalog().

    Entries from load_catalog() carry "_norm" already; a hand-built entry is
    normalized on the fly, per call, without writing back to the caller's dict.
    """
    norm = entry.get("_norm")
    if norm is None:
        norm = _prepare_entry(entry)["_norm"]
    return norm


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _is_proper_substring(a, b):
    """True when one normalized string is a proper, word-aligned substring of
    the other. Word alignment stops accidents like "it" inside "hobbit"."""
    if not a or not b or a == b:
        return False
    padded_a, padded_b = f" {a} ", f" {b} "
    return padded_a in padded_b or padded_b in padded_a


def _title_score(query_forms, entry_titles):
    """Best title similarity for one entry, plus whether the winning pair was
    substring-penalized.

    Returns (score in 0..1, penalty_applied bool).
    """
    if not entry_titles:
        return 0.0, False

    # An exact match against the canonical title or any alternate title is
    # trusted outright -- the substring guard is not applied to that entry.
    exact_alias_match = any(
        candidate == form for candidate in entry_titles for form in query_forms
    )

    best_score = 0.0
    best_penalized = False
    for candidate in entry_titles:
        # Decided once per candidate title, against every query form: otherwise
        # the article-stripped form ("lord of the rings the two towers") would
        # slip past a guard that the full form triggers.
        penalized = not exact_alias_match and any(
            _is_proper_substring(form, candidate) for form in query_forms
        )
        for form in query_forms:
            raw = fuzz.token_set_ratio(form, candidate) / 100.0
            score = max(0.0, raw - SUBSTRING_PENALTY) if penalized else raw
            if score > best_score:
                best_score, best_penalized = score, penalized
    return best_score, best_penalized


def _author_score(query_author, entry_authors):
    if not query_author or not entry_authors:
        return 0.0
    return max(
        fuzz.token_sort_ratio(query_author, candidate) for candidate in entry_authors
    ) / 100.0


def _sort_key(candidate):
    """Highest score first; ties broken by catalog_id so results are stable."""
    catalog_id = str(candidate["entry"].get("catalog_id", ""))
    numeric = (0, int(catalog_id)) if catalog_id.isdigit() else (1, 0)
    return (-candidate["score"], numeric, catalog_id)


def _relation(best_entry, other_entry):
    """How a rival candidate relates to the rank-1 candidate."""
    best_work = best_entry.get("work_id")
    other_work = other_entry.get("work_id")
    if best_work and other_work and best_work == other_work:
        return RELATION_SAME_WORK
    if other_work in _as_list(best_entry.get("contains_work_ids")):
        return RELATION_OMNIBUS
    if best_work in _as_list(other_entry.get("contains_work_ids")):
        return RELATION_OMNIBUS
    return RELATION_DIFFERENT_WORK


# --------------------------------------------------------------------------
# Result construction
# --------------------------------------------------------------------------


def _runner_up_view(candidate):
    if candidate is None:
        return None
    entry = candidate["entry"]
    return {
        "catalog_id": entry.get("catalog_id"),
        "work_id": entry.get("work_id"),
        "title": entry.get("title"),
        "author": entry.get("author"),
        "score": round(candidate["score"], 4),
    }


def _result(
    catalog_id,
    work_id,
    confidence,
    status,
    reasons,
    runner_up,
    raw_title,
    raw_author,
):
    return {
        "catalog_id": catalog_id,
        "work_id": work_id,
        "confidence": round(confidence, 4),
        "status": status,
        "reasons": reasons,
        "runner_up": runner_up,
        "raw_title": raw_title,
        "raw_author": raw_author,
    }


def _unmatched(reasons, raw_title, raw_author, runner_up=None, confidence=0.0):
    return _result(
        catalog_id=None,
        work_id=None,
        confidence=confidence,
        status=STATUS_UNMATCHED,
        reasons=reasons,
        runner_up=runner_up,
        raw_title=raw_title,
        raw_author=raw_author,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def match(raw_title, raw_author, catalog, legible=True):
    """Match one spine read against the catalog.

    Never raises on bad input: an unreadable spine, an empty title, garbage
    text or an empty catalog all return a structured UNMATCHED result.
    """
    query_title = normalize_title(raw_title)
    query_author = normalize_author(raw_author)

    # 1. Nothing usable to match on.
    if not legible:
        return _unmatched([REASON_NOT_LEGIBLE], raw_title, raw_author)
    if not query_title:
        return _unmatched([REASON_NO_TITLE_TEXT], raw_title, raw_author)
    if not catalog:
        return _unmatched([REASON_EMPTY_CATALOG], raw_title, raw_author)

    author_readable = bool(query_author)
    query_forms = _dedupe([query_title, strip_leading_article(query_title)])

    # 2. Score every catalog entry.
    scored = []
    for entry in catalog:
        norm = _entry_norm(entry)
        title_score, penalized = _title_score(query_forms, norm["titles"])
        author_sim = (
            _author_score(query_author, norm["authors"]) if author_readable else 0.0
        )
        if author_readable:
            score = TITLE_WEIGHT * title_score + AUTHOR_WEIGHT * author_sim
        else:
            score = title_score
        scored.append(
            {
                "entry": entry,
                "score": score,
                "title_score": title_score,
                "author_score": author_sim,
                "substring_penalized": penalized,
            }
        )

    # 3. Rank.
    scored.sort(key=_sort_key)
    best = scored[0]
    best_entry = best["entry"]

    reasons = []
    if best["substring_penalized"]:
        reasons.append(REASON_SUBSTRING_PENALTY)
    if not author_readable:
        reasons.append(REASON_AUTHOR_UNREADABLE)

    # 4. Ambiguity scan: inside the margin, take the best rival OF EACH relation
    #    type and apply only the worst applicable penalty.
    rivals = [
        candidate
        for candidate in scored[1:]
        if best["score"] - candidate["score"] < AMBIGUITY_MARGIN
    ]
    by_relation = {}
    for candidate in rivals:  # rivals stay score-ordered, so first wins per type
        relation = _relation(best_entry, candidate["entry"])
        by_relation.setdefault(relation, candidate)

    penalty = 0.0
    ambiguous_with = None
    for relation, reason, relation_penalty in (
        (RELATION_DIFFERENT_WORK, REASON_DIFFERENT_WORK, DIFFERENT_WORK_PENALTY),
        (RELATION_OMNIBUS, REASON_OMNIBUS, OMNIBUS_PENALTY),
        (RELATION_SAME_WORK, REASON_EDITION, EDITION_PENALTY),
    ):
        if relation in by_relation:
            penalty = relation_penalty
            reasons.append(reason)
            ambiguous_with = by_relation[relation]
            break

    ambiguous = ambiguous_with is not None
    runner_up = _runner_up_view(
        ambiguous_with if ambiguous else (scored[1] if len(scored) > 1 else None)
    )

    confidence = min(1.0, max(0.0, best["score"] - penalty))

    # 5. A title-only read can never reach AUTO_READY.
    if not author_readable and confidence > NO_AUTHOR_CONFIDENCE_CAP:
        confidence = NO_AUTHOR_CONFIDENCE_CAP
        reasons.append(REASON_TITLE_ONLY_CAP)

    # 6. Route. Ambiguity always forces review, whatever the score.
    if ambiguous:
        status = STATUS_REVIEW
    elif confidence >= AUTO_READY_THRESHOLD:
        status = STATUS_AUTO_READY
    elif confidence >= REVIEW_THRESHOLD:
        status = STATUS_REVIEW
    else:
        status = STATUS_UNMATCHED

    # 7. Output contract. Nothing below the review floor is silently attached
    #    to a catalog entry -- the best candidate is still handed back as
    #    runner_up so the review screen can offer it.
    if status == STATUS_UNMATCHED:
        reasons.append(REASON_LOW_SIMILARITY)
        return _unmatched(
            reasons,
            raw_title,
            raw_author,
            runner_up=_runner_up_view(best),
            confidence=confidence,
        )

    return _result(
        catalog_id=best_entry.get("catalog_id"),
        work_id=best_entry.get("work_id"),
        confidence=confidence,
        status=status,
        reasons=reasons,
        runner_up=runner_up,
        raw_title=raw_title,
        raw_author=raw_author,
    )
