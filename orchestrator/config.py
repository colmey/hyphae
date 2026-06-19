# orchestrator/config.py

"""
Config loaders for the orchestration layer.

Mirrors the pattern in harness_config.load_mcp_config: read a YAML/text
file from disk, validate, return a typed object. The loaders raise loudly
on malformed input -- main.py decides whether to abort startup or fall
back to legacy mode (see orchestration_enabled in Settings).

Kept separate from harness_config.py so that orchestration is a fully
optional subsystem -- nothing in harness_config imports from orchestrator/,
which means the orchestrator can be removed or stubbed without disturbing
the existing layers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schemas import ModelsConfig

logger = logging.getLogger(__name__)


def load_models_config(path: Path | str) -> ModelsConfig:
    """Load and validate models.yaml.

    Raises FileNotFoundError if the file is missing, or ValueError if it
    fails schema validation. Caller (main.py) decides whether to treat
    those as fatal or as a signal to disable orchestration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"models config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"models config root must be a mapping, got {type(raw).__name__}"
        )

    try:
        cfg = ModelsConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid models config at {path}:\n{e}") from e

    logger.info(
        "loaded models config: %d entries (default=%s)",
        len(cfg.models), cfg.default_id(),
    )
    return cfg


def load_orchestrator_prompt(path: Path | str) -> str:
    """Read the orchestrator system prompt from disk.

    Plain text file (not YAML) so it can be edited freely without YAML
    quoting headaches. Whitespace at file boundaries is stripped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"orchestrator prompt file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"orchestrator prompt file is empty: {path}")

    logger.info("loaded orchestrator prompt: %d chars from %s", len(text), path)
    return text