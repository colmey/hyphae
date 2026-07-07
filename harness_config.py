# harness_config.py

"""Runtime settings and typed MCP config loading.

Settings are built lazily after bootstrap has loaded `.env`, so modules should
call `get_settings()` instead of importing a module-level settings object.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings; use `get_settings()` in application code."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Only the selected provider's key must be set.
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="",
        description=(
            "Base URL for the OpenAI-compatible endpoint (e.g. a local Ollama "
            "server's /v1). Empty targets real OpenAI."
        ),
    )

    # Provider validation lives in llm.client to keep this layer import-light.
    llm_provider: str = Field(default="gemini", description="LLM provider name")
    llm_model: str = Field(default="gemini-3-flash-preview", description="Model name to use")
    llm_max_tokens: int = Field(default=4096)

    # Agent-loop timeouts/retries; <= 0 disables the respective bound.
    llm_timeout_seconds: float = Field(
        default=120,
        description="Per-attempt cap on a single llm.complete() call. <=0 disables.",
    )
    tool_timeout_seconds: float = Field(
        default=60,
        description="Cap on a single mcp.call_tool() call. <=0 disables.",
    )
    llm_max_retries: int = Field(
        default=3,
        description=(
            "Retries (not attempts) on transient LLM failures / empty candidates. "
            "0 disables retrying."
        ),
    )
    llm_retry_base_delay: float = Field(
        default=0.5,
        description="Base seconds for jittered exponential backoff between LLM retries.",
    )
    tool_result_max_chars: int = Field(
        default=20000,
        description=(
            "Clip threshold for a single flattened tool result before it enters "
            "session history. <=0 disables clipping."
        ),
    )

    # Optional run-level stop conditions. Each exits with an explicit done_reason.
    max_run_tokens: int = Field(
        default=0,
        description=(
            "Hard ceiling on cumulative total_tokens for one run; ends the run "
            "budget_exceeded with the partial answer. <=0 disables. When a "
            "provider reports absent/all-zero usage, the local token estimator "
            "(agent/context.py) fills in, so the cap works against local "
            "OpenAI-compatible servers too."
        ),
    )
    max_run_seconds: float = Field(
        default=0,
        description=(
            "Hard wall-clock ceiling on one run, measured from the first "
            "iteration and enforced during LLM/tool calls; ends the run "
            "deadline_exceeded with the partial answer. <=0 disables."
        ),
    )
    abort_after_consecutive_tool_failures: int = Field(
        default=0,
        description=(
            "Abort the run no_progress after this many tool-call failures in a "
            "row (a success resets the count). Should be greater than the "
            "fixed at-3 consecutive-failure nudge so the model gets a chance to "
            "recover first. <=0 disables."
        ),
    )

    # Context assembly shapes the outgoing view only; session history is unchanged.
    context_strategy: str = Field(
        default="naive",
        description=(
            "Context assembly strategy: 'naive' (pass-through + budget warning) "
            "or 'compaction' (summarize over-budget middle history). Unknown "
            "values degrade to 'naive' with a warning."
        ),
    )
    context_default_window_tokens: int = Field(
        default=32768,
        description=(
            "Assumed context window (tokens) for models whose models.yaml entry "
            "has no context_window, and for legacy/no-orchestrator mode."
        ),
    )
    context_safety_margin_tokens: int = Field(
        default=1024,
        description=(
            "Headroom subtracted from the context window (with max output "
            "tokens) when computing the input budget; absorbs estimator error."
        ),
    )
    context_recent_messages: int = Field(
        default=6,
        description=(
            "Recent protocol-safe units (a user turn, a no-tool assistant turn, "
            "or an assistant tool call plus its results) kept verbatim under "
            "compaction. Shrinks automatically if the tail alone overflows."
        ),
    )
    context_summary_max_tokens: int = Field(
        default=512,
        description="Output cap for the one-call compaction summarizer.",
    )

    # Harness paths. All runtime config lives under config/ by convention.
    mcp_config_path: Path = Field(default=Path("config/mcp_config.yaml"))
    max_loop_iterations: int = Field(default=10) # was 25, find a good balance
    log_level: str = Field(default="INFO")

    # Optional API-key gate for /chat, /chat/stream, and /v1/*; /health stays open.
    harness_api_key: str = Field(
        default="",
        description=(
            "Optional API key protecting /chat, /chat/stream, /v1/*. Empty "
            "disables auth; set enforces it. Accepts X-API-Key or Bearer."
        ),
    )

    # JSONL traces include full prompts/tool data; protect the file accordingly.
    trace_enabled: bool = Field(
        default=False,
        description="Persist the loop's event stream as a JSONL trace.",
    )
    trace_path: Path = Field(
        default=Path("traces/harness.jsonl"),
        description="Append-only JSONL trace file. Parent dirs are created.",
    )

    # In-memory session bounds; <= 0 disables the respective dimension.
    session_ttl_seconds: int = Field(
        default=3600,
        description="Idle TTL (seconds) before an in-memory session is evicted.",
    )
    session_max_count: int = Field(
        default=1000,
        description="Max sessions retained in memory; oldest-updated evicted first.",
    )

    # When off, routes use the default LLM and full tool inventory directly.
    orchestration_enabled: bool = Field(
        default=True,
        description="Master toggle for the orchestration layer.",
    )
    models_config_path: Path = Field(
        default=Path("config/models.yaml"),
        description="Path to the model registry YAML consumed by the orchestrator.",
    )
    orchestrator_prompt_path: Path = Field(
        default=Path("config/orchestrator_prompt.md"),
        description="Path to the orchestrator's system prompt file (markdown or plain text).",
    )
    orchestrator_model_id: str = Field(
        default="",
        description=(
            "Override which model_id the orchestrator itself uses to make routing "
            "decisions. Empty = use the default entry from models.yaml."
        ),
    )

    def required_api_key(self) -> str:
        """Return the API key matching the configured provider, or raise.

        Thin wrapper over api_key_for_provider(self.llm_provider); preserved for
        backward compatibility with callers that predate the per-provider factory.
        """
        return self.api_key_for_provider(self.llm_provider)

    def api_key_for_provider(self, provider: str) -> str:
        """Return the API key for a provider, or raise if it's needed but missing.

        Unknown providers fall back to `<PROVIDER>_API_KEY`, letting new
        credentialed providers register without changing Settings.
        """
        typed = {
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
        }
        key = typed.get(provider)
        if key is None:
            key = os.environ.get(f"{provider.upper()}_API_KEY", "")
        if not key:
            raise RuntimeError(
                f"no API key found in env for provider {provider!r} "
                f"(expected {provider.upper()}_API_KEY). Is it set in your "
                ".env file, and did bootstrap.load_secrets() run first?"
            )
        return key


# Lazily populated after bootstrap has loaded environment variables.
_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance, building it on first call."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def reset_settings() -> None:
    """Clear the settings cache. For tests, or after late env var changes."""
    global _settings_cache
    _settings_cache = None


class _MCPServerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disabled: bool = False
    disabled_tools: list[str] = Field(default_factory=list)


class StreamableHTTPServer(_MCPServerBase):
    transport: Literal["streamable-http"]
    url: str  # not HttpUrl: we allow internal hostnames like *.local


class SSEServer(_MCPServerBase):
    transport: Literal["sse"]
    url: str


class StdioServer(_MCPServerBase):
    transport: Literal["stdio"]
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


# Pydantic picks the right server model based on `transport`.
MCPServerConfig = Annotated[
    Union[StreamableHTTPServer, SSEServer, StdioServer],
    Field(discriminator="transport"),
]


class ToolPolicyConfig(BaseModel):
    """Dispatch-time policy for what tools may execute."""

    # Typos in security controls must fail loud instead of falling back to allow_all.
    model_config = ConfigDict(extra="forbid")

    mode: Literal["allow_all", "allow_list"] = "allow_all"
    allow: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_allow_list(self) -> "ToolPolicyConfig":
        # An allow_list with no usable patterns would silently deny every tool.
        if self.mode == "allow_list":
            if not self.allow:
                raise ValueError("tool_policy.mode 'allow_list' requires a non-empty 'allow' list")
            if any(not p.strip() for p in self.allow):
                raise ValueError("tool_policy.allow patterns must be non-empty strings")
        return self


class MCPConfig(BaseModel):
    # Catch top-level typos such as `tool_polciy`, which would disable policy.
    model_config = ConfigDict(extra="forbid")

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")
    tool_policy: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)

    @model_validator(mode="after")
    def _validate_names(self) -> "MCPConfig":
        # Server names are embedded in `{server}__{tool}` namespaced tool IDs.
        for name in self.mcp_servers:
            if "__" in name:
                raise ValueError(
                    f"server name {name!r} cannot contain '__' (reserved for tool namespacing)"
                )
            if not name or not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError(
                    f"server name {name!r} must be alphanumeric (dashes/underscores allowed)"
                )
        return self

    def enabled_servers(self) -> dict[str, MCPServerConfig]:
        """Return only servers not marked disabled."""
        return {n: s for n, s in self.mcp_servers.items() if not s.disabled}


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


def load_mcp_config(path: Path | str) -> MCPConfig:
    """Load and validate the MCP config YAML file.

    Call this AFTER bootstrap, since the YAML may contain ${ENV_VAR} references.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"MCP config root must be a mapping, got {type(raw).__name__}")

    interpolated = _interpolate_env(raw)

    try:
        return MCPConfig.model_validate(interpolated)
    except ValidationError as e:
        raise ValueError(f"invalid MCP config at {path}:\n{e}") from e
