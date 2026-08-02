"""Safe configuration exceptions and validation-error rendering."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

_DETAIL_MAX_CHARS = 1_000
_LOCATION_PART_MAX_CHARS = 128
_MAX_RENDERED_ERRORS = 8
_PATH_MAX_CHARS = 512
_SAFE_LOCATION_PARTS = frozenset(
    {
        "<root>",
        "allow",
        "anthropic_api_key",
        "args",
        "abort_after_consecutive_tool_failures",
        "command",
        "context_default_window_tokens",
        "context_recent_messages",
        "context_safety_margin_tokens",
        "context_strategy",
        "context_summary_max_tokens",
        "context_window",
        "default",
        "description",
        "disabled",
        "disabled_tools",
        "env",
        "gemini_api_key",
        "hyphae_api_key",
        "harness_api_key",
        "llm",
        "log_level",
        "loop_max_iterations",
        "max_retries",
        "max_loop_iterations",
        "max_run_seconds",
        "max_run_tokens",
        "max_tokens",
        "mcpServers",
        "mcp_catalog_ttl_seconds",
        "mcp_config_path",
        "mcp_connect_timeout_seconds",
        "mcp_servers",
        "mode",
        "model",
        "model_name",
        "models_config_path",
        "models",
        "openai_api_key",
        "openai_base_url",
        "openai_compat_base_url",
        "openai_compat_tool_activity",
        "openai_compat_tool_activity_max_chars",
        "openai_compat_tool_activity_mode",
        "openai_provider_base_url",
        "openai_tool_block_max_chars",
        "orchestration_enabled",
        "orchestrator_prompt_path",
        "orchestrator_model_id",
        "provider",
        "retry_base_delay",
        "run_max_seconds",
        "run_max_tokens",
        "sampling",
        "session_capacity",
        "session_max_count",
        "session_ttl_seconds",
        "supports_native_tools",
        "temperature",
        "thinking",
        "timeout_seconds",
        "tool_result_max_chars",
        "tool_timeout_seconds",
        "tool_policy",
        "top_k",
        "top_p",
        "transport",
        "trace_enabled",
        "trace_jsonl_path",
        "trace_path",
        "url",
    }
)


def _bounded_single_line(value: object, limit: int) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(value)
    )
    text = " ".join(printable.split()).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def safe_path_display(path: Path | str) -> str:
    """Render a source path safely for operator-facing messages and logs."""
    return _bounded_single_line(path, _PATH_MAX_CHARS)


class CredentialUnavailableError(RuntimeError):
    """A configured provider credential is unavailable at startup."""


class ConfigLoadError(ValueError):
    """A safely rendered configuration-loading failure."""

    def __init__(
        self,
        path: Path | str,
        detail: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        source = Path(path)
        try:
            self._path = source.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            self._path = source.absolute()
        self._cause = cause
        safe_detail = _bounded_single_line(detail, _DETAIL_MAX_CHARS)
        safe_path = safe_path_display(self._path)
        super().__init__(f"{safe_detail}: {safe_path}")

    @property
    def path(self) -> Path:
        """Return the full resolved source path without using it for rendering."""
        return self._path


class ConfigValidationError(ConfigLoadError):
    """A typed configuration-schema or cross-field validation failure."""


def safe_validation_summary(error: ValidationError) -> str:
    """Render bounded field paths and error codes without rejected inputs."""
    rendered: list[str] = []
    errors = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    for item in errors[:_MAX_RENDERED_ERRORS]:
        location = item.get("loc", ())
        safe_location: list[str] = []
        for part in location:
            if isinstance(part, int):
                safe_location.append(str(part))
            else:
                bounded = _bounded_single_line(part, _LOCATION_PART_MAX_CHARS)
                safe_location.append(
                    bounded if bounded in _SAFE_LOCATION_PARTS else "<entry>"
                )
        field = ".".join(safe_location)
        error_type = _bounded_single_line(item.get("type", "invalid"), 64)
        rendered.append(f"{field or '<root>'}: {error_type}")
    omitted = len(errors) - len(rendered)
    if omitted > 0:
        rendered.append(f"{omitted} additional error(s) omitted")
    return "; ".join(rendered) or "validation failed"
