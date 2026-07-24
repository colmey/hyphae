# config/settings.py

"""Runtime settings, loaded lazily from the environment and project ``.env``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor defaults to real locations so the server boots from any CWD:
# this package directory holds the config data files; .env is at repo root.
_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_DIR.parent


class Settings(BaseSettings):
    """Environment-driven settings; use `get_settings()` in application code."""

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pydantic passes its merged settings-source input through model validators,
    # including undeclared dotenv values. Retain that raw input privately so
    # interpolation does not expose extras or stringify validated/default values.
    _interpolation_values: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _capture_interpolation_values(cls, values: Any, handler: Any) -> Self:
        settings = handler(values)
        if isinstance(values, Mapping):
            settings._interpolation_values = {
                str(name).upper(): str(value)
                for name, value in values.items()
                if value is not None
            }
        return settings

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
    llm_model: str = Field(
        default="gemini-3-flash-preview", description="Model name to use"
    )
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
    mcp_connect_timeout_seconds: float = Field(
        default=30,
        description=(
            "Cap on complete MCP startup or recovery connection setup, including "
            "transport, initialization, and tool discovery. <=0 disables."
        ),
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
    openai_tool_block_max_chars: int = Field(
        default=2000,
        description=(
            "Truncation for one rendered tool-result <details> block on the "
            "/v1 streaming surface. Presentation-only; distinct from "
            "tool_result_max_chars, which clips session history."
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
            "Hard wall-clock ceiling on one accepted turn, measured immediately "
            "after the session claim and enforced across routing, retries, LLM/tool "
            "calls, and backoff; ends the run deadline_exceeded with the partial "
            "answer. <=0 disables."
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
            "has no context_window, and for unorchestrated mode."
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
    mcp_config_path: Path = Field(default=_CONFIG_DIR / "mcp_config.yaml")
    max_loop_iterations: int = Field(default=10)
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
        default=_CONFIG_DIR / "models.yaml",
        description="Path to the model registry YAML consumed by the orchestrator.",
    )
    orchestrator_prompt_path: Path = Field(
        default=_CONFIG_DIR / "orchestrator_prompt.md",
        description="Path to the orchestrator's system prompt file (markdown or plain text).",
    )
    orchestrator_model_id: str = Field(
        default="",
        description=(
            "Override which model_id the orchestrator itself uses to make routing "
            "decisions. Empty = use the default entry from models.yaml."
        ),
    )

    def interpolation_environment(self) -> dict[str, str]:
        """Return Settings-owned values for ``${ENV_VAR}`` interpolation.

        Only values supplied by Settings sources are included; declared defaults
        are not synthesized. Real process variables are applied last, preserving
        their precedence without mutating ``os.environ``.
        """
        values = dict(self._interpolation_values)
        values.update(os.environ)
        return values

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
            key = self.interpolation_environment().get(
                f"{provider.upper()}_API_KEY", ""
            )
        if not key:
            raise RuntimeError(
                f"no API key found in env for provider {provider!r} "
                f"(expected {provider.upper()}_API_KEY). Set it in the process "
                "environment or project .env file."
            )
        return key


# Lazily populated so importing application modules never reads credentials.
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
