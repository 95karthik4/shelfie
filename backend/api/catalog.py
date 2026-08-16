"""Catalog access for the API layer.

A thin adapter over matching.load_catalog(), not a second implementation.
All CSV parsing, pipe-splitting and normalisation stay in matching/; this
module adds exactly two things the web layer needs and the matcher does not:

  * a process-lifetime cache, so a 105-row CSV isn't re-read per request
  * a catalog_id -> row index, because match() returns a catalog_id but no
    display strings, and both the scan response (matched_title /
    matched_author) and the confirm endpoint (validating a submitted
    catalog_id) need to resolve one to a row.

Imports django.conf and matching only -- no vision, no vlm, no models.
"""

import logging

from django.conf import settings

from matching import load_catalog

logger = logging.getLogger(__name__)

# Module-level cache. Populated on first use and reused for the life of the
# process; reset_cache() clears it for tests that point CATALOG_PATH
# elsewhere.
_catalog = None
_by_id = None


def normalized_id(catalog_id):
    """Coerce a catalog id to its canonical string form, or None.

    catalog.csv ids arrive from csv.DictReader as strings ("1", "2"), but
    callers legitimately hold ints -- a client posting JSON {"catalog_id": 1}
    means the same row as {"catalog_id": "1"}. Comparing normalised strings
    makes both work without the caller having to know which the CSV used.
    """
    if catalog_id is None:
        return None
    text = str(catalog_id).strip()
    return text or None


def build_index(catalog):
    """catalog_id -> row, for any catalog list.

    Shared by the cached real catalog and by callers holding one in memory
    (the pipeline, tests), so the identity rules below are enforced in
    exactly one place and a lookup can never be built from a different
    catalog than the one being matched against.

    Both failure modes are refusals to index a catalog whose *identity*
    column is unsound. The deliberate ambiguity in this catalog lives in
    titles and authors; catalog_id is the key the confirmation endpoint
    validates against and the key ConfirmedBook persists, so it has to mean
    exactly one row.
    """
    by_id = {}
    for entry in catalog:
        key = normalized_id(entry.get("catalog_id"))
        if key is None:
            # Such a row would still be scored by the matcher but could never
            # be confirmed -- matchable and unconfirmable at once.
            raise ValueError(
                "catalog row missing catalog_id: {!r}".format(entry.get("title"))
            )
        if key in by_id:
            # Two rows answering to one id would put an unresolvable
            # reference in the user's library. Fail loudly instead of
            # silently picking one.
            raise ValueError("duplicate catalog_id in catalog: {}".format(key))
        by_id[key] = entry
    return by_id


def _load():
    """Read the catalog and build the id index. Called once, lazily."""
    catalog = load_catalog(settings.CATALOG_PATH)
    by_id = build_index(catalog)

    # Assigned together, after both are built, so a concurrent reader sees
    # either the uninitialised state or a complete pair -- never a half-built
    # index. Two threads racing here both do the same idempotent work.
    global _catalog, _by_id
    _catalog, _by_id = catalog, by_id
    logger.info("loaded %d catalog entries from %s", len(catalog), settings.CATALOG_PATH)


def get_catalog():
    """The catalog as matching.match() expects it: a list of row dicts.

    Returns the cached list itself, not a copy -- callers must treat it as
    read-only.
    """
    if _catalog is None:
        _load()
    return _catalog


def get_entry(catalog_id):
    """The catalog row for an id, or None if there is no such row.

    Returns None rather than raising: an unknown id is an ordinary outcome
    (a stale client, a hand-typed confirmation), and the confirm endpoint
    turns it into a 400 rather than a 500.
    """
    if _by_id is None:
        _load()
    key = normalized_id(catalog_id)
    if key is None:
        return None
    return _by_id.get(key)


def reset_cache():
    """Drop the cached catalog. For tests that repoint settings.CATALOG_PATH."""
    global _catalog, _by_id
    _catalog = None
    _by_id = None
