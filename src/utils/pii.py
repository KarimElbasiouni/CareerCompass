"""Lightweight helpers for masking obvious PII patterns."""

from __future__ import annotations

import regex as re

EMAIL_PATTERN = re.compile(
    r"""
    (?P<email>
        [a-z0-9._%+\-]+      # local part
        @
        [a-z0-9.\-]+\.[a-z]{2,}  # domain
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

PHONE_PATTERN = re.compile(
    r"""
    (?P<phone>
        (?:(?:\+|00)\d{1,3}[\s\-.]*)?  # country code
        (?:\(?\d{3}\)?[\s\-.]*)?       # area code
        \d{3}[\s\-.]*\d{4}             # local number
    )
    """,
    re.VERBOSE,
)


def _coerce_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return text


def mask_email_phone(text: str | None) -> str:
    """Replace email and phone patterns with lightweight sentinels."""
    clean_text = _coerce_text(text)
    if not clean_text:
        return ""
    masked = EMAIL_PATTERN.sub("[EMAIL]", clean_text)
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    return masked


def mask_exact_name(text: str | None, name: str | None) -> str:
    """Replace the exact person name (if provided) with a sentinel."""
    clean_text = _coerce_text(text)
    if not clean_text:
        return ""
    if not name:
        return clean_text
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    return pattern.sub("[NAME]", clean_text)
