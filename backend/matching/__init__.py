"""Catalog matching. Pure Python -- no Django imports live in this package."""

from .matcher import load_catalog, match
from .normalize import normalize_author, normalize_title, strip_leading_article

__all__ = [
    "load_catalog",
    "match",
    "normalize_author",
    "normalize_title",
    "strip_leading_article",
]
