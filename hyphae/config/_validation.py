"""Shared internal primitives for strict configuration validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_core import PydanticCustomError

IDENTIFIER_MAX_CHARS = 128


def prefer_canonical_alias(values: Any, *, canonical: str, alias: str) -> Any:
    """Drop a compatibility alias when its canonical spelling is also present."""
    if isinstance(values, Mapping) and canonical in values and alias in values:
        normalized = dict(values)
        normalized.pop(alias)
        return normalized
    return values


def validate_nonblank_bounded(
    value: str,
    *,
    label: str,
    max_chars: int = IDENTIFIER_MAX_CHARS,
    allow_empty: bool = False,
) -> str:
    """Validate bounded operator-facing names without exposing their contents."""
    if allow_empty and value == "":
        return value
    if not value.strip():
        raise PydanticCustomError("nonblank_required", f"{label} must be nonblank")
    if len(value) > max_chars:
        raise PydanticCustomError(
            "length_limit_exceeded",
            f"{label} exceeds its configured length bound",
        )
    return value


def reject_bool_or_float_for_int(value: Any) -> Any:
    """Keep environment integer strings, but reject YAML booleans/floats."""
    if isinstance(value, (bool, float)):
        raise PydanticCustomError("integer_required", "value must be an integer")
    return value
