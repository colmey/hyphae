# config/loaders.py

"""File-config loaders: one shared YAML pipeline plus the prompt loader.

The loaders raise loudly on malformed input -- main.py decides whether to
abort startup or fall back to legacy mode (see orchestration_enabled in
Settings). Call them AFTER load_secrets(), since the MCP YAML may contain
${ENV_VAR} references.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .schemas import MCPConfig, ModelsConfig

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} in strings. Raises if a var is unset."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in os.environ:
                raise ValueError(
                    f"environment variable {var!r} referenced in config is not set"
                )
            return os.environ[var]
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _load_yaml_model(
    path: Path | str,
    model_cls: type[_ModelT],
    *,
    what: str,
    interpolate: bool = False,
) -> _ModelT:
    """Shared pipeline: exists-check -> YAML parse -> mapping guard -> validate.

    `what` names the config in error messages (e.g. "MCP config"), keeping
    them identical to the pre-consolidation loaders.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{what} root must be a mapping, got {type(raw).__name__}")

    if interpolate:
        raw = _interpolate_env(raw)

    try:
        return model_cls.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid {what} at {path}:\n{e}") from e


def load_mcp_config(path: Path | str) -> MCPConfig:
    """Load and validate the MCP config YAML file (${ENV_VAR} interpolated)."""
    cfg = _load_yaml_model(path, MCPConfig, what="MCP config", interpolate=True)
    logger.info(
        "loaded MCP config: %d servers (%d enabled), tool_policy=%s",
        len(cfg.mcp_servers), len(cfg.enabled_servers()), cfg.tool_policy.mode,
    )
    return cfg


def load_models_config(
    path: Path | str,
    *,
    known_providers: Collection[str] | None = None,
) -> ModelsConfig:
    """Load and validate models.yaml.

    Raises FileNotFoundError if the file is missing, or ValueError if it
    fails schema validation. Caller (main.py) decides whether to treat
    those as fatal or as a signal to disable orchestration.

    `known_providers`, when given, rejects entries naming a provider not in
    it (main.py passes llm.client.supported_providers()). When omitted, an
    unknown provider surfaces later, at client build time.
    """
    cfg = _load_yaml_model(path, ModelsConfig, what="models config")
    if known_providers is not None:
        for model_id, entry in cfg.models.items():
            if entry.provider not in known_providers:
                raise ValueError(
                    f"model {model_id!r} has unknown provider {entry.provider!r}; "
                    f"registered providers: {sorted(known_providers)}. "
                    "Add a builder to llm/client.py's _PROVIDERS to support it."
                )
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
