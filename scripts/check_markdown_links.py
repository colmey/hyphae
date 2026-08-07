#!/usr/bin/env python3
"""Check relative file links in tracked Markdown documents."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INLINE_LINK_START = re.compile(r"!?\[[^\]\n]*\]\(")
_REFERENCE_DEFINITION_START = re.compile(
    r"^[ \t]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*",
    re.MULTILINE,
)

MissingLink = tuple[Path, int, str]
LinkTarget = tuple[int, str]


def _parse_destination(
    text: str,
    start: int,
    *,
    inline: bool,
) -> tuple[str, int] | None:
    cursor = start
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1

    if cursor < len(text) and text[cursor] == "<":
        destination_start = cursor + 1
        cursor = destination_start
        escaped = False
        while cursor < len(text):
            character = text[cursor]
            if character in "\r\n":
                return None
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == ">":
                return text[destination_start:cursor], cursor + 1
            cursor += 1
        return None

    destination_start = cursor
    depth = 0
    escaped = False
    while cursor < len(text):
        character = text[cursor]
        if character in "\r\n" or (character in " \t" and depth == 0):
            break
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")":
            if depth == 0:
                if inline:
                    break
                return None
            depth -= 1
        cursor += 1

    if cursor == destination_start or depth != 0:
        return None
    return text[destination_start:cursor], cursor


def _inline_link_targets(text: str) -> list[LinkTarget]:
    targets: list[LinkTarget] = []
    for match in _INLINE_LINK_START.finditer(text):
        parsed = _parse_destination(text, match.end(), inline=True)
        if parsed is None:
            continue
        target, cursor = parsed
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor < len(text) and text[cursor] == ")":
            targets.append((match.start(), target))
            continue

        if cursor >= len(text) or text[cursor] not in "\"'(":
            continue
        closing_title = {
            '"': '"',
            "'": "'",
            "(": ")",
        }[text[cursor]]
        cursor += 1
        escaped = False
        while cursor < len(text):
            character = text[cursor]
            if character in "\r\n":
                break
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == closing_title:
                cursor += 1
                while cursor < len(text) and text[cursor] in " \t":
                    cursor += 1
                if cursor < len(text) and text[cursor] == ")":
                    targets.append((match.start(), target))
                break
            cursor += 1
    return targets


def _reference_link_targets(text: str) -> list[LinkTarget]:
    targets: list[LinkTarget] = []
    for match in _REFERENCE_DEFINITION_START.finditer(text):
        parsed = _parse_destination(text, match.end(), inline=False)
        if parsed is not None:
            target, _ = parsed
            targets.append((match.start(), target))
    return targets


def _link_targets(text: str) -> list[LinkTarget]:
    return sorted(
        [*_inline_link_targets(text), *_reference_link_targets(text)],
        key=lambda item: item[0],
    )


def tracked_markdown_files(repository_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(detail or "git ls-files failed")
    return tuple(
        repository_root / os.fsdecode(name)
        for name in result.stdout.split(b"\0")
        if name
    )


def find_missing_links(
    documents: Iterable[Path],
    *,
    repository_root: Path,
) -> list[MissingLink]:
    missing: list[MissingLink] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for offset, target in _link_targets(text):
            if target.startswith("#"):
                continue

            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            decoded_path = unquote(parsed.path)
            if Path(decoded_path).is_absolute():
                continue

            resolved = (document.parent / decoded_path).resolve(strict=False)
            if not resolved.exists():
                line_number = text.count("\n", 0, offset) + 1
                missing.append(
                    (document.relative_to(repository_root), line_number, target)
                )
    return sorted(missing, key=lambda item: (item[0].as_posix(), item[1], item[2]))


def main() -> int:
    try:
        documents = tracked_markdown_files(REPOSITORY_ROOT)
        missing = find_missing_links(documents, repository_root=REPOSITORY_ROOT)
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if missing:
        print("Missing relative Markdown targets:", file=sys.stderr)
        for document, line_number, target in missing:
            print(f"  {document}:{line_number}: {target}", file=sys.stderr)
        return 1
    print(f"Checked {len(documents)} tracked Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
