"""The 13 matcher tests required by CLAUDE.md.

They run against the real catalog.csv at the repo root, so they exercise the
ambiguity that was deliberately built into it. Test 13 is the one exception:
it needs three editions of one work and the real catalog has at most two, so
it builds a small in-memory catalog with the same schema.
"""

from pathlib import Path

import pytest

from matching.matcher import (
    AUTO_READY_THRESHOLD,
    REASON_AUTHOR_UNREADABLE,
    REASON_DIFFERENT_WORK,
    REASON_EDITION,
    REASON_NOT_LEGIBLE,
    REASON_OMNIBUS,
    STATUS_AUTO_READY,
    STATUS_REVIEW,
    STATUS_UNMATCHED,
    load_catalog,
    match,
)
from matching.normalize import normalize_author

CATALOG_PATH = Path(__file__).resolve().parents[3] / "catalog.csv"

OUTPUT_CONTRACT_KEYS = {
    "catalog_id",
    "work_id",
    "confidence",
    "status",
    "reasons",
    "runner_up",
    "raw_title",
    "raw_author",
}


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(CATALOG_PATH)


# 1 -------------------------------------------------------------------------
def test_exact_title_and_author_is_high_confidence_auto_ready(catalog):
    result = match("The Kite Runner", "Khaled Hosseini", catalog)

    assert result["work_id"] == "kite_runner"
    assert result["status"] == STATUS_AUTO_READY
    assert result["confidence"] >= AUTO_READY_THRESHOLD
    assert result["reasons"] == []


# 2 -------------------------------------------------------------------------
def test_minor_ocr_typo_in_title_still_matches(catalog):
    result = match("The Kite Runer", "Khaled Hosseini", catalog)

    assert result["work_id"] == "kite_runner"
    assert result["status"] == STATUS_AUTO_READY
    assert result["confidence"] >= AUTO_READY_THRESHOLD


# 3 -------------------------------------------------------------------------
def test_us_uk_alternate_title_resolves_to_same_work(catalog):
    uk = match("Northern Lights", "Philip Pullman", catalog)
    us = match("The Golden Compass", "Philip Pullman", catalog)

    assert uk["work_id"] == "northern_lights"
    assert us["work_id"] == "northern_lights"
    assert us["status"] == STATUS_AUTO_READY


# 4 -------------------------------------------------------------------------
def test_initials_and_lastname_first_author_forms_are_equivalent(catalog):
    assert normalize_author("J.K. Rowling") == normalize_author("Rowling, J. K.")

    plain = match("Harry Potter and the Chamber of Secrets", "J.K. Rowling", catalog)
    inverted = match(
        "Harry Potter and the Chamber of Secrets", "Rowling, J. K.", catalog
    )

    assert plain["work_id"] == inverted["work_id"] == "chamber_secrets"
    assert plain["confidence"] == inverted["confidence"]
    assert inverted["status"] == STATUS_AUTO_READY


# 5 -------------------------------------------------------------------------
def test_accented_and_unaccented_author_are_the_same_author(catalog):
    assert normalize_author("Gabriel García Márquez") == normalize_author(
        "Gabriel Garcia Marquez"
    )

    accented = match("Love in the Time of Cholera", "Gabriel García Márquez", catalog)
    plain = match("Love in the Time of Cholera", "Gabriel Garcia Marquez", catalog)

    assert accented["work_id"] == plain["work_id"] == "love_cholera"
    assert accented["confidence"] == plain["confidence"]
    assert plain["status"] == STATUS_AUTO_READY


# 6 -------------------------------------------------------------------------
def test_substring_guard_prevents_dune_overconfidence(catalog):
    # token_set_ratio scores "Dune" against "Dune Messiah" at 100 because one
    # token set contains the other. The guard is what keeps them apart.
    messiah = match("Dune Messiah", "Frank Herbert", catalog)

    assert messiah["work_id"] == "dune_messiah"
    assert messiah["status"] == STATUS_AUTO_READY
    # "Dune" is demoted far enough that it is not even in contention.
    assert messiah["runner_up"]["work_id"] == "dune"
    assert messiah["runner_up"]["score"] < AUTO_READY_THRESHOLD
    assert REASON_DIFFERENT_WORK not in messiah["reasons"]

    # And the short title does not drag the longer ones in as rivals.
    dune = match("Dune", "Frank Herbert", catalog)
    assert dune["work_id"] == "dune"
    assert REASON_DIFFERENT_WORK not in dune["reasons"]


# 7 -------------------------------------------------------------------------
def test_shared_title_with_unreadable_author_goes_to_review(catalog):
    result = match("Home", None, catalog)

    assert result["status"] == STATUS_REVIEW
    assert REASON_DIFFERENT_WORK in result["reasons"]
    assert REASON_AUTHOR_UNREADABLE in result["reasons"]
    # Both candidates are real books called "Home"; the user has to choose.
    assert {result["work_id"], result["runner_up"]["work_id"]} == {
        "home_coben",
        "home_morrison",
    }


# 8 -------------------------------------------------------------------------
def test_shared_title_with_readable_author_resolves_to_correct_work(catalog):
    morrison = match("Home", "Toni Morrison", catalog)
    coben = match("Home", "Harlan Coben", catalog)

    assert morrison["work_id"] == "home_morrison"
    assert morrison["status"] == STATUS_AUTO_READY
    assert morrison["confidence"] >= AUTO_READY_THRESHOLD

    assert coben["work_id"] == "home_coben"
    assert coben["status"] == STATUS_AUTO_READY


# 9 -------------------------------------------------------------------------
def test_two_editions_of_same_work_force_edition_review(catalog):
    result = match("Dune", "Frank Herbert", catalog)

    assert result["work_id"] == "dune"
    assert REASON_EDITION in result["reasons"]
    # Ambiguity forces review even though the score is very high.
    assert result["status"] == STATUS_REVIEW
    assert result["confidence"] >= AUTO_READY_THRESHOLD
    assert result["runner_up"]["work_id"] == "dune"
    assert result["runner_up"]["catalog_id"] != result["catalog_id"]


# 10 ------------------------------------------------------------------------
def test_omnibus_and_contained_volume_force_omnibus_review(catalog):
    # A real spine of the collected edition prints both the omnibus name and
    # the volume name, which puts the two entries neck and neck.
    result = match("His Dark Materials: Northern Lights", "Philip Pullman", catalog)

    assert REASON_OMNIBUS in result["reasons"]
    assert REASON_DIFFERENT_WORK not in result["reasons"]
    assert result["status"] == STATUS_REVIEW
    assert {result["work_id"], result["runner_up"]["work_id"]} == {
        "northern_lights",
        "hdm_omnibus",
    }


# 11 ------------------------------------------------------------------------
def test_illegible_spine_returns_unmatched_without_crash(catalog):
    result = match("smudged text", "unreadable", catalog, legible=False)

    assert result["status"] == STATUS_UNMATCHED
    assert result["catalog_id"] is None
    assert result["work_id"] is None
    assert result["reasons"] == [REASON_NOT_LEGIBLE]
    assert result["confidence"] == 0.0
    assert result["raw_title"] == "smudged text"


# 12 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw_title,raw_author,empty_after_normalization",
    [
        ("", "", True),
        (None, None, True),
        ("   ", "Frank Herbert", True),
        ("!!! ... ???", None, True),
        ("📚🔥", "🙂", False),
        ("qwzx plmk", "zzz", False),
    ],
)
def test_garbage_input_never_crashes_and_is_never_silently_accepted(
    catalog, raw_title, raw_author, empty_after_normalization
):
    result = match(raw_title, raw_author, catalog)

    # Contract holds, nothing raised.
    assert set(result) == OUTPUT_CONTRACT_KEYS
    # Never silently accepted: garbage can never be added without a human.
    assert result["status"] != STATUS_AUTO_READY
    assert result["status"] in {STATUS_REVIEW, STATUS_UNMATCHED}
    assert result["confidence"] < AUTO_READY_THRESHOLD
    # Never silently dropped either: there is always an explanation.
    assert result["reasons"]

    if empty_after_normalization:
        # Nothing to match on at all, so there is no candidate to attach.
        assert result["status"] == STATUS_UNMATCHED
        assert result["catalog_id"] is None
        assert result["work_id"] is None
        assert result["confidence"] == 0.0


# 13 ------------------------------------------------------------------------
def _edition_cluster_catalog():
    """Three editions of one work, plus a different work that shares the title
    and has a near-identical author name so it lands inside the margin.

    Built in memory because the real catalog has no work with three editions.
    """
    def entry(catalog_id, work_id, author, edition):
        return {
            "catalog_id": catalog_id,
            "work_id": work_id,
            "title": "The Silver Road",
            "author": author,
            "alternate_titles": [],
            "author_aliases": [],
            "edition": edition,
            "contains_work_ids": [],
        }

    return [
        entry("1", "silver_road_reed", "Alan Reed", "First 1998"),
        entry("2", "silver_road_reed", "Alan Reed", "Reissue 2007"),
        entry("3", "silver_road_reed", "Alan Reed", "Anniversary 2018"),
        entry("4", "silver_road_read", "Alan Read", "Harbour 2011"),
    ]


def test_different_work_ambiguity_outranks_edition_ambiguity():
    catalog = _edition_cluster_catalog()

    result = match("The Silver Road", "Alan Reed", catalog)

    # Three same-work editions tie at the top, but the rival that is a
    # different book is the dangerous one, so it sets the penalty.
    # Rank 2 here is legitimately another same-work edition; what matters is
    # that the different-work candidate inside the margin sets the tier.
    assert result["work_id"] == "silver_road_reed"
    assert REASON_DIFFERENT_WORK in result["reasons"]
    assert REASON_EDITION not in result["reasons"]
    assert result["status"] == STATUS_REVIEW
    assert result["confidence"] == pytest.approx(0.7, abs=0.05)
