"""Small, dependency-free text utilities used by the public pipeline."""

from __future__ import annotations

import re


_ABBREVIATION_PATTERN = re.compile(r"\b(e\.g|i\.e|etc|vs|inc|ltd|approx)\.", re.IGNORECASE)
_DECIMAL_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")
_PROTECTED_SPAN_PATTERN = re.compile(r"(`[^`]*`|<<|>>|<=|>=)")


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""

    return re.sub(r"\s+", " ", text).strip()


def detokenize(text: str) -> str:
    """Repair the most common whitespace artifacts from fast tokenizers."""

    text = re.sub(r"\s+([.,;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    return normalize_space(text)


def split_sentences(text: str) -> list[str]:
    """Split already-present sentence boundaries with conservative rules."""

    protected = _DECIMAL_PATTERN.sub("\x00", text)
    protected = _ABBREVIATION_PATTERN.sub(
        lambda match: match.group(0)[:-1] + "\x00", protected
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", protected)
    return [part.replace("\x00", ".").strip() for part in parts if part.strip()]


def mask_protected_spans(text: str) -> tuple[str, dict[str, str]]:
    """Replace code-like spans and comparison operators with T5 sentinels."""

    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"<extra_id_{len(mapping)}>"
        mapping[token] = match.group(0)
        return token

    return _PROTECTED_SPAN_PATTERN.sub(replace, text), mapping


def restore_protected_spans(text: str, mapping: dict[str, str]) -> tuple[str, set[str]]:
    """Restore masked spans and report which sentinels were observed."""

    restored = text
    seen: set[str] = set()
    for token, original in mapping.items():
        if token in restored:
            seen.add(token)
            restored = restored.replace(token, original)
    return detokenize(restored), seen


def word_tokens(text: str) -> list[str]:
    """Return lowercase alphanumeric word tokens for lightweight checks."""

    return re.findall(r"[a-z0-9]+", text.lower())


def number_tokens(text: str) -> list[str]:
    """Return integer and dotted-version-like numeric tokens."""

    return re.findall(r"\d+(?:\.\d+)+|\d+", text)
