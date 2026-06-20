from __future__ import annotations

import re
import unicodedata
from collections import Counter


NUMERIC_CHARS = set("0123456789.,-+/:%$₩€¥*()")
NUMERIC_PUNCTUATION = set(".,-+/:%$₩€¥*()")
CONFUSION_GROUPS = (
    ("0", "O"),
    ("1", "I"),
    ("1", "l"),
    ("2", "Z"),
    ("5", "S"),
    ("6", "G"),
    ("8", "B"),
    (".", ","),
    ("-", ""),
    ("₩", "W"),
)
OPTIONAL_NUMERIC_CORRECTIONS = str.maketrans(
    {
        "O": "0",
        "I": "1",
        "l": "1",
        "S": "5",
        "B": "8",
        "G": "6",
    }
)


def normalize_text(raw_text: str) -> str:
    """Normalize spacing without correcting OCR confusions or dropping punctuation."""
    text = unicodedata.normalize("NFKC", raw_text)
    return re.sub(r"\s+", " ", text).strip()


def corrected_numeric_text_optional(text: str) -> str:
    return normalize_text(text).translate(OPTIONAL_NUMERIC_CORRECTIONS)


def is_numeric_token(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    has_digit = any(ch.isdigit() for ch in normalized)
    allowed_or_space = all(ch in NUMERIC_CHARS or ch.isspace() for ch in normalized)
    return has_digit and allowed_or_space


def digit_sequence(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if ch.isdigit())


def numeric_char_sequence(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if ch.isdigit() or ch in NUMERIC_PUNCTUATION)


def punctuation_counter(text: str) -> Counter[str]:
    return Counter(ch for ch in normalize_text(text) if ch in NUMERIC_PUNCTUATION)


def decimal_count(text: str) -> int:
    return normalize_text(text).count(".")


def thousands_separator_count(text: str) -> int:
    return normalize_text(text).count(",")


def sign_counter(text: str) -> Counter[str]:
    text = normalize_text(text)
    return Counter(ch for ch in text if ch in "-+")


def currency_counter(text: str) -> Counter[str]:
    text = normalize_text(text)
    return Counter(ch for ch in text if ch in "$₩€¥")

