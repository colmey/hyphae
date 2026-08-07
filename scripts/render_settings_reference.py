#!/usr/bin/env python3
"""Render the environment-settings reference from Pydantic field metadata."""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import AliasChoices
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = REPOSITORY_ROOT / "docs" / "configuration.md"
START_MARKER = "<!-- BEGIN GENERATED SETTINGS -->"
END_MARKER = "<!-- END GENERATED SETTINGS -->"

# Running a file by path puts scripts/, rather than the repository root, first.
sys.path.insert(0, str(REPOSITORY_ROOT))

from config.settings import LLMSettings, Settings  # noqa: E402


def _escape_markdown_cell(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _code(value: str) -> str:
    return f"`{_escape_markdown_cell(value)}`"


def _render_default(field_name: str, default: object) -> str:
    if field_name.endswith("_api_key"):
        return "*unset*"
    if default is PydanticUndefined:
        raise ValueError(f"{field_name} has no static default")
    if default is None:
        return "*none*"
    if isinstance(default, str) and not default:
        return "*unset*"
    if isinstance(default, bool):
        return _code(str(default).lower())
    if isinstance(default, Path):
        rendered = default
        if default.is_absolute():
            try:
                rendered = default.relative_to(REPOSITORY_ROOT)
            except ValueError:
                pass
        return _code(rendered.as_posix())
    if isinstance(default, (int, float, str)):
        return _code(str(default))
    raise TypeError(f"unsupported default for {field_name}: {type(default).__name__}")


def _environment_names(
    field_name: str,
    field: FieldInfo,
    *,
    prefix: str = "",
) -> tuple[str, tuple[str, ...]]:
    validation_alias = field.validation_alias
    if isinstance(validation_alias, AliasChoices):
        raw_names = tuple(validation_alias.choices)
    elif isinstance(validation_alias, str):
        raw_names = (validation_alias,)
    elif validation_alias is None:
        raw_names = (field.alias or field_name,)
    else:
        raise TypeError(f"unsupported validation alias for {field_name}")

    if not raw_names or any(not isinstance(name, str) for name in raw_names):
        raise TypeError(f"environment aliases for {field_name} must be strings")

    names = tuple(dict.fromkeys(f"{prefix}{name}".upper() for name in raw_names))
    return names[0], names[1:]


def _render_field_row(field_name: str, field: FieldInfo, *, prefix: str = "") -> str:
    if field.default_factory is not None:
        raise ValueError(f"{field_name} uses a default factory and cannot be rendered")
    description = (field.description or "").strip()
    if not description:
        raise ValueError(f"{field_name} has no description")

    environment_name, aliases = _environment_names(field_name, field, prefix=prefix)
    alias_cell = ", ".join(_code(alias) for alias in aliases) if aliases else "—"
    return " | ".join(
        (
            f"| {_code(environment_name)}",
            _render_default(field_name, field.default),
            _escape_markdown_cell(description),
            f"{alias_cell} |",
        )
    )


def render_settings_table() -> str:
    """Render all declared settings without constructing either settings model."""
    rows = [
        "| Environment variable | Default | Description | Compatibility aliases |",
        "|---|---|---|---|",
    ]
    for field_name, field in Settings.model_fields.items():
        if field_name == "llm":
            rows.extend(
                _render_field_row(name, nested_field, prefix="LLM_")
                for name, nested_field in LLMSettings.model_fields.items()
            )
            continue
        rows.append(_render_field_row(field_name, field))
    return "\n".join(rows)


def replace_generated_region(document: str, generated: str) -> str:
    """Replace exactly one generated region while preserving surrounding prose."""
    if document.count(START_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("configuration document must contain exactly one marker pair")

    start = document.index(START_MARKER) + len(START_MARKER)
    end = document.index(END_MARKER)
    if end < start:
        raise ValueError("generated settings markers are out of order")
    return f"{document[:start]}\n\n{generated.rstrip()}\n\n{document[end:]}"


def process_document(path: Path, *, write: bool) -> bool:
    """Write or check one settings-reference document; return whether it was current."""
    current = path.read_text(encoding="utf-8")
    expected = replace_generated_region(current, render_settings_table())
    if current == expected:
        return True
    if write:
        path.write_text(expected, encoding="utf-8")
        return False

    try:
        display_path = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        display_path = path
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(display_path),
        tofile=f"{display_path} (generated)",
    )
    sys.stderr.writelines(diff)
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="update the marked region")
    mode.add_argument("--check", action="store_true", help="fail if the region is stale")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        current = process_document(DOCUMENT_PATH, write=arguments.write)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0 if arguments.write or current else 1


if __name__ == "__main__":
    raise SystemExit(main())
