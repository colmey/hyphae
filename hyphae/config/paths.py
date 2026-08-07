"""Repository-owned application path resolution."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CONFIG_DIR.parents[1]


def application_path(value: Path | str) -> Path:
    """Return an absolute application-owned path without touching the filesystem."""
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def resolve_application_path(value: Path | str) -> Path:
    """Resolve application-owned relative paths against the repository root."""
    return application_path(value).resolve(strict=False)
