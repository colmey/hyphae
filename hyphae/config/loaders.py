"""Strict, path-stable, secret-safe configuration file loaders."""

from __future__ import annotations

import errno
import logging
import os
import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .errors import (
    ConfigLoadError,
    ConfigValidationError,
    safe_path_display,
    safe_validation_summary,
)
from .paths import application_path
from .schemas import MCPConfig, ModelsConfig

if TYPE_CHECKING:
    from yaml.error import Mark

    from .settings import Settings

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class _DuplicateKeyError(yaml.YAMLError):
    def __init__(self, first_mark: "Mark", duplicate_mark: "Mark") -> None:
        self.first_mark = first_mark
        self.duplicate_mark = duplicate_mark
        super().__init__("duplicate mapping key")


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML constructors plus duplicate-key rejection at every depth."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, got {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        seen: dict[object, Mark] = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                first_mark = seen.get(key)
            except TypeError:
                first_mark = None
            if first_mark is not None:
                raise _DuplicateKeyError(first_mark, key_node.start_mark)
            try:
                seen[key] = key_node.start_mark
            except TypeError:
                pass
        return super().construct_mapping(node, deep=deep)


def _interpolate_env(value: Any, resolve: Callable[[str], str]) -> Any:
    """Recursively replace ${ENV_VAR} in strings without mutating its source."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            variable = match.group(1)
            try:
                return resolve(variable)
            except KeyError:
                raise KeyError(variable) from None

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {key: _interpolate_env(item, resolve) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item, resolve) for item in value]
    return value


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _missing_file(path: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(errno.ENOENT, f"{what} file not found", path)


def _resolve_source(path: Path | str, what: str) -> Path:
    candidate = application_path(path)
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigLoadError(
            candidate,
            f"could not resolve {what} path",
            cause=exc,
        ) from None


def _load_yaml_model(
    path: Path | str,
    model_cls: type[_ModelT],
    *,
    what: str,
    interpolate: bool = False,
    interpolation_resolver: Callable[[str], str] | None = None,
) -> _ModelT:
    """Parse one strict YAML mapping and return an immutable typed snapshot."""
    source = _resolve_source(path, what)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=_StrictSafeLoader)
    except FileNotFoundError:
        raise _missing_file(source, what) from None
    except _DuplicateKeyError as exc:
        detail = (
            f"invalid {what}: duplicate mapping key; first defined at line "
            f"{exc.first_mark.line + 1}, "
            f"column {exc.first_mark.column + 1}; repeated at line "
            f"{exc.duplicate_mark.line + 1}, column {exc.duplicate_mark.column + 1}"
        )
        raise ConfigLoadError(source, detail, cause=exc) from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = (
            f" at line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else ""
        )
        raise ConfigLoadError(
            source,
            f"invalid {what}: malformed YAML{location}",
            cause=exc,
        ) from None
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigLoadError(source, f"could not read {what}", cause=exc) from None

    if not isinstance(raw, dict):
        raise ConfigValidationError(source, f"invalid {what}: root must be a mapping")

    if interpolate:
        resolve = interpolation_resolver or os.environ.__getitem__
        try:
            raw = _interpolate_env(raw, resolve)
        except KeyError as exc:
            variable = str(exc.args[0])
            raise ConfigLoadError(
                source,
                f"invalid {what}: environment variable {variable!r} is not set",
                cause=exc,
            ) from None

    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(
            source,
            f"invalid {what} ({safe_validation_summary(exc)})",
            cause=exc,
        ) from None


def load_mcp_config(
    path: Path | str,
    *,
    environment: Mapping[str, str] | None = None,
) -> MCPConfig:
    """Load strict MCP YAML with source-owned environment interpolation."""
    source_environment = environment if environment is not None else os.environ
    return _load_mcp_config(path, source_environment.__getitem__)


def _load_mcp_config(
    path: Path | str,
    interpolation_resolver: Callable[[str], str],
) -> MCPConfig:
    config = _load_yaml_model(
        path,
        MCPConfig,
        what="MCP config",
        interpolate=True,
        interpolation_resolver=interpolation_resolver,
    )
    logger.info(
        "loaded MCP config: %d servers (%d enabled), tool_policy=%s",
        len(config.mcp_servers),
        len(config.enabled_servers()),
        config.tool_policy.mode,
    )
    return config


def load_mcp_config_from_settings(settings: "Settings") -> MCPConfig:
    """Load the configured MCP file with Settings-owned interpolation values."""
    return _load_mcp_config(
        settings.mcp_config_path,
        settings.interpolation_value,
    )


def _validate_model_runtime_bounds(
    config: ModelsConfig,
    *,
    path: Path,
    default_max_tokens: int,
    default_context_window: int,
    safety_margin: int,
) -> None:
    for model_id, entry in config.models.items():
        max_tokens = entry.max_tokens or default_max_tokens
        context_window = entry.context_window or default_context_window
        if max_tokens + safety_margin >= context_window:
            raise ConfigValidationError(
                path,
                "invalid models config "
                f"(models.{model_id}.context_window: inconsistent_context_bounds)",
            )


def load_models_config(
    path: Path | str,
    *,
    known_providers: Collection[str] | None = None,
) -> ModelsConfig:
    """Load and validate a model registry without constructing providers."""
    source = _resolve_source(path, "models config")
    config = _load_yaml_model(source, ModelsConfig, what="models config")
    if known_providers is not None:
        for model_id, entry in config.models.items():
            if entry.provider not in known_providers:
                raise ConfigValidationError(
                    source,
                    "invalid models config "
                    f"(models.{model_id}.provider: unknown_provider)",
                )
    logger.info(
        "parsed models config: %d entries (default=%s)",
        len(config.models),
        config.default_id(),
    )
    return config


def load_models_config_from_settings(
    settings: "Settings",
    *,
    known_providers: Collection[str] | None = None,
) -> ModelsConfig:
    """Load models with Settings-owned effective context/output limits."""
    config = load_models_config(
        settings.models_config_path,
        known_providers=known_providers,
    )
    _validate_model_runtime_bounds(
        config,
        path=settings.models_config_path,
        default_max_tokens=settings.llm.max_tokens,
        default_context_window=settings.context_default_window_tokens,
        safety_margin=settings.context_safety_margin_tokens,
    )
    if (
        settings.orchestrator_model_id
        and settings.orchestrator_model_id not in config.models
    ):
        raise ConfigValidationError(
            settings.models_config_path,
            "invalid application settings "
            "(orchestrator_model_id: unknown_model_identifier)",
        )
    logger.info(
        "validated models config: %d entries (default=%s)",
        len(config.models),
        config.default_id(),
    )
    return config


def load_orchestrator_prompt(path: Path | str) -> str:
    """Load a nonblank orchestrator prompt with safe I/O diagnostics."""
    source = _resolve_source(path, "orchestrator prompt")
    try:
        text = source.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise _missing_file(source, "orchestrator prompt") from None
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigLoadError(
            source,
            "could not read orchestrator prompt",
            cause=exc,
        ) from None
    if not text:
        raise ConfigValidationError(
            source, "invalid orchestrator prompt: file is empty"
        )

    logger.info(
        "loaded orchestrator prompt: %d chars from %s",
        len(text),
        safe_path_display(source),
    )
    return text


def load_agent_prompt(path: Path | str) -> str:
    """Load a nonblank trusted agent prompt with safe I/O diagnostics."""
    source = _resolve_source(path, "agent prompt")
    try:
        text = source.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise _missing_file(source, "agent prompt") from None
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigLoadError(source, "could not read agent prompt", cause=exc) from None
    if not text:
        raise ConfigValidationError(source, "invalid agent prompt: file is empty")

    logger.info(
        "loaded agent prompt: %d chars from %s", len(text), safe_path_display(source)
    )
    return text
