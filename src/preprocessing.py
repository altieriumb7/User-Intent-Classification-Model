"""Text preprocessing utilities for support-ticket intent classification."""

from __future__ import annotations

import re
import html


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s'$]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalize one user message into a compact, model-friendly string."""
    if text is None:
        return ""

    value = html.unescape(str(text)).lower().strip()
    value = _URL_RE.sub(" url ", value)
    value = _EMAIL_RE.sub(" email ", value)
    value = value.replace("&", " and ")
    value = _NON_TEXT_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value
