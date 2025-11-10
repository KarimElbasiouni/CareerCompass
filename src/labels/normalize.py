"""Stub helpers for future label normalization work."""

from __future__ import annotations


def normalize_title(raw: str | None) -> str | None:
    """Placeholder: trim the raw title string."""
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def map_title_to_family(title: str | None) -> str | None:
    """Placeholder mapping from normalized title to coarser families."""
    _ = title  # TODO: implement mapping table
    return None
