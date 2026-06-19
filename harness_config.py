# harness_config.py

"""
Configuration loading for the AI harness.

Two things live here:
  1. `Settings`  - runtime/env config (API keys, model, paths) read from os.environ.
  2. `MCPConfig` - typed parse of mcp_config.yaml.

Secrets convention:
  Environment variables live in a `.env` file at the project root.
  `bootstrap.load_secrets()` loads that file into `os.environ` (via
  python-dotenv) BEFORE any harness code reads settings. Real environment
  variables already set in the process take precedence over the `.env` file.

  The bootstrap step and the harness run in the same Python process, so
  import order matters. To avoid accidentally snapshotting an empty env at
  module load, this file:
    - never constructs Settings() at module level
    - exposes `get_settings()` which builds Settings on first call and caches it
    - exposes `reset_settings()` for tests and for forcing a re-read after
      bootstrap if needed

  Rule of thumb: nothing in the harness should `from config import settings`
  (a module-level instance). Always call `get_settings()` from within a
  function, after bootstrap has run.

Config file layout:
  All runtime YAML/text config lives under `config/`:
    config/mcp_config.yaml
    config/models.yaml
    config/orchestrator_prompt.md

  These paths can be overridden via the corresponding env vars
  (MCP_CONFIG_PATH, MODELS_CONFIG_PATH, ORCHESTRATOR_PROMPT_PATH), so
  custom deployments can point them elsewhere without code changes.

The YAML loader supports `${ENV_VAR}` interpolation in string values so
secrets can be injected into the MCP config without hardcoding. Missing env
vars raise a clear error rather than silently producing empty strings.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Runtime settings (env-driven, backed by a .env file)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Environment-driven settings.

    Reads from os.environ, which `bootstrap.load_secrets()` populates from the
    project's `.env` file before this is constructed. As a fallback, pydantic-
    settings also reads `.env` directly (see model_config). Use get_settings()
    rather than instantiating directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider keys. Only the one matching the configured provider needs to be set.
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

    # LLM config. `llm_provider` is a free string validated at build time
    # against the provider registry (llm.client.supported_providers()); an
    # unknown value fails loudly in build_llm_client with NotImplementedError.
    # We keep harness_config free of any `llm` import to preserve layering, so
    # the Literal-style check lives in the LLM layer, not here.
    llm_provider: str = Field(default="gemini", description="LLM provider name")
    llm_model: str = Field(default="gemini-3-flash-preview", description="Model name to use")
    llm_max_tokens: int = Field(default=4096)

    # --- Reliability hardening (agent loop) -------------------------------
    # These bound long, tool-heavy runs so a transient failure or a hung
    # server can't kill or stall a request. The agent loop reads these via
    # the route (run_agent params); direct callers (smoke tests) get the
    # loop's own defaults instead. <= 0 disables the respective dimension.
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

    # Harness paths. All runtime config lives under config/ by convention.
    mcp_config_path: Path = Field(default=Path("config/mcp_config.yaml"))
    max_loop_iterations: int = Field(default=25)
    log_level: str = Field(default="INFO")

    # --- Observability (run tracing) --------------------------------------
    # When enabled, the loop serializes its event stream to a JSONL trace, one
    # record per event tagged with run_id + step + timestamp (+ latency on LLM/
    # tool steps). Off by default so the hot path and smoke tests are untouched;
    # a failing sink degrades silently and never breaks a request. NOTE: the
    # trace captures full message text, tool args, and tool results — treat the
    # file as sensitive (there is no auth on the harness yet; roadmap #8).
    trace_enabled: bool = Field(
        default=False,
        description="Persist the loop's event stream as a JSONL trace.",
    )
    trace_path: Path = Field(
        default=Path("traces/harness.jsonl"),
        description="Append-only JSONL trace file. Parent dirs are created.",
    )

    # --- In-memory session store bounds -----------------------------------
    # Sessions live only in process memory (LibreChat holds the durable
    # context). These cap memory so the store can't grow without limit under
    # concurrent load. <= 0 disables the respective dimension.
    session_ttl_seconds: int = Field(
        default=3600,
        description="Idle TTL (seconds) before an in-memory session is evicted.",
    )
    session_max_count: int = Field(
        default=1000,
        description="Max sessions retained in memory; oldest-updated evicted first.",
    )

    # --- Orchestration layer ---------------------------------------------
    # Master toggle. When false, the route bypasses the orchestrator entirely
    # and falls back to the legacy behavior: default LLM client, all MCP
    # tools, request.system used as-is. Useful for dev environments without
    # models.yaml present.
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

        Known providers have typed Settings fields; any other provider falls
        back to the conventional `<PROVIDER>_API_KEY` environment variable, so a
        newly registered credentialed provider needs no change here. Providers
        that don't use an API key (e.g. a local Ollama server) simply never call
        this method from their builder.
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


# Module-level cache. Populated lazily on first get_settings() call so that
# bootstrap code running earlier in the same process gets its env vars picked up.
_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance, building it on first call.

    Call this AFTER your bootstrap script has populated os.environ. If something
    in the harness needs settings at import time (it shouldn't), that's a bug
    -- defer the read until a function actually runs.
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def reset_settings() -> None:
    """Clear the settings cache. For tests, or after late env var changes."""
    global _settings_cache
    _settings_cache = None


# ---------------------------------------------------------------------------
# MCP config schema
# ---------------------------------------------------------------------------

class _MCPServerBase(BaseModel):
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


# Discriminated union: Pydantic picks the right model based on `transport`.
MCPServerConfig = Annotated[
    Union[StreamableHTTPServer, SSEServer, StdioServer],
    Field(discriminator="transport"),
]


class MCPConfig(BaseModel):
    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")

    @model_validator(mode="after")
    def _validate_names(self) -> "MCPConfig":
        # Tool namespacing uses `{server}__{tool}` -- reject server names that
        # would break that scheme.
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


# ---------------------------------------------------------------------------
# YAML loading with ${ENV_VAR} interpolation
# ---------------------------------------------------------------------------

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