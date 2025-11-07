"""Utility helpers for preparing Indonesian texts before inference."""
from __future__ import annotations

import re
from typing import Dict

try:
    from unidecode import unidecode
except Exception:  # pragma: no cover - optional dependency
    unidecode = None  # type: ignore


_SLANG_MAP: Dict[str, str] = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "yg": "yang",
    "dr": "dari",
    "tp": "tapi",
}

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z0-9\s.,;:!?()'\-]")
_MULTI_SPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lightweight normalisation mirroring the preprocessing used in training."""

    if not isinstance(text, str):
        return ""

    cleaned = text
    if unidecode is not None:
        cleaned = unidecode(cleaned)
    cleaned = cleaned.lower()
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _MENTION_PATTERN.sub(" ", cleaned)
    cleaned = _NON_ALPHA_PATTERN.sub(" ", cleaned)
    cleaned = " ".join(_SLANG_MAP.get(word, word) for word in cleaned.split())
    return _MULTI_SPACE.sub(" ", cleaned).strip()
