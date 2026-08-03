# config/settings.py

"""Runtime settings, loaded lazily from the environment and project ``.env``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from ._validation import (
    prefer_canonical_alias,
    reject_bool_or_float_for_int,
    validate_nonblank_bounded,
)
from .errors import (
    ConfigLoadError,
    ConfigValidationError,
    CredentialUnavailableError,
    safe_validation_summary,
)
from .paths import CONFIG_DIR, REPOSITORY_ROOT, resolve_application_path

def _validate_identifier(value: str, label: str, *, allow_blank: bool = False) -> str:
    return validate_nonblank_bounded(
        value,
        label=label,
        allow_empty=allow_blank,
    )


# Pydantic numeric constraints: ``gt=0`` requires a positive value, while
# ``ge=0`` allows zero. Fields documented as ``<= 0 disables`` omit both.
class LLMSettings(BaseModel):
    """Provider-neutral defaults and execution controls for LLM calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(default="gemini", description="LLM provider name")
    model: str = Field(
        default="gemini-3-flash-preview",
        description="Provider-native model name to use",
        validation_alias=AliasChoices("model", "model_name"),
    )
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(
        default=120,
        allow_inf_nan=False,
        description="Per-attempt cap on a single llm.complete() call. <=0 disables.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Retries (not attempts) on transient LLM failures / empty candidates. "
            "0 disables retrying."
        ),
    )
    retry_base_delay: float = Field(
        default=0.5,
        ge=0,
        allow_inf_nan=False,
        description="Base seconds for jittered exponential backoff between LLM retries.",
    )

    @model_validator(mode="before")
    @classmethod
    def _prefer_canonical_model(cls, values: Any) -> Any:
        """Discard the compatibility `model_name` alias when both are set."""
        return prefer_canonical_alias(values, canonical="model", alias="model_name")

    _strict_integers = field_validator("max_tokens", "max_retries", mode="before")(
        reject_bool_or_float_for_int
    )

    @field_validator("provider")
    @classmethod
    def _provider_is_valid(cls, value: str) -> str:
        return _validate_identifier(value, "provider identifier")

    @field_validator("model")
    @classmethod
    def _model_is_valid(cls, value: str) -> str:
        return _validate_identifier(value, "model identifier")


class Settings(BaseSettings):
    """Environment-driven settings; use `get_settings()` in application code."""

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        env_nested_max_split=1,
        extra="ignore",
        frozen=True,
    )

    # Pydantic passes its merged settings-source input through model validators,
    # including undeclared dotenv values. Retain that raw input privately so
    # interpolation does not expose extras or stringify validated/default values.
    _interpolation_values: Mapping[str, str] = PrivateAttr(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_interpolation_values":
            raise AttributeError("configuration source snapshot is immutable")
        super().__setattr__(name, value)

    @model_validator(mode="wrap")
    @classmethod
    def _capture_interpolation_values(cls, values: Any, handler: Any) -> Self:
        settings = handler(values)
        if isinstance(values, Mapping):
            interpolation_values: dict[str, str] = {}
            for name, value in values.items():
                if name == "llm":
                    nested_values = (
                        value
                        if isinstance(value, Mapping)
                        else value.model_dump()
                        if isinstance(value, LLMSettings)
                        else {}
                    )
                    interpolation_values.update(
                        {
                            f"LLM_{nested_name}".upper(): str(nested_value)
                            for nested_name, nested_value in nested_values.items()
                            if nested_value is not None
                        }
                    )
                elif value is not None:
                    interpolation_values[str(name).upper()] = str(value)
            # Capture arbitrary interpolation-only process variables as well as
            # declared settings. Process variables preserve their source precedence.
            interpolation_values.update(os.environ)
            # Assign through Pydantic's private store once, then prohibit replacement.
            settings.__pydantic_private__["_interpolation_values"] = MappingProxyType(
                interpolation_values
            )
        return settings

    # Only the selected provider's key must be set.
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_compat_base_url: str = Field(
        default="",
        description=(
            "Base URL for the OpenAI-compatible endpoint (e.g. a local Ollama "
            "server's /v1). Empty targets real OpenAI."
        ),
        validation_alias=AliasChoices(
            "openai_compat_base_url",
            "openai_provider_base_url",
            "openai_base_url",
        ),
    )

    # Provider validation lives in llm.client to keep this layer import-light.
    # The one-split environment mapping preserves LLM_PROVIDER,
    # LLM_MODEL, LLM_MAX_TOKENS, and the LLM execution-control names.
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Non-LLM execution timeouts; <= 0 disables the respective bound.
    tool_timeout_seconds: float = Field(
        default=60,
        allow_inf_nan=False,
        description="Cap on a single mcp.call_tool() call. <=0 disables.",
    )
    mcp_connect_timeout_seconds: float = Field(
        default=30,
        allow_inf_nan=False,
        description=(
            "Cap on complete MCP startup or recovery connection setup, including "
            "transport, initialization, and tool discovery. <=0 disables."
        ),
    )
    mcp_catalog_ttl_seconds: float = Field(
        default=300,
        allow_inf_nan=False,
        description=(
            "Age after which an accepted request refreshes an MCP catalog before "
            "routing. <=0 disables age-driven refresh; startup and lease discovery "
            "still run."
        ),
    )
    tool_result_max_chars: int = Field(
        default=20000,
        description=(
            "Clip threshold for a single flattened tool result before it enters "
            "session history. <=0 disables clipping."
        ),
    )
    openai_compat_tool_activity_max_chars: int = Field(
        default=2000,
        gt=0,
        description=(
            "Presentation threshold for displayed arguments or results in one "
            "/v1 tool-activity payload. Distinct from "
            "tool_result_max_chars, which clips session history."
        ),
        validation_alias=AliasChoices(
            "openai_compat_tool_activity_max_chars",
            "openai_tool_block_max_chars",
        ),
    )
    openai_compat_tool_activity_mode: Literal[
        "reasoning", "reasoning_full", "hidden"
    ] = Field(
        default="reasoning",
        description=(
            "OpenAI-compatible streaming activity: 'reasoning' emits model "
            "reasoning and compact tool progress through delta.reasoning_content; "
            "'reasoning_full' adds bounded tool arguments and results; 'hidden' "
            "omits that optional channel."
        ),
        validation_alias=AliasChoices(
            "openai_compat_tool_activity_mode",
            "openai_compat_tool_activity",
        ),
    )

    # Optional run-level stop conditions. Each exits with an explicit done_reason.
    run_max_tokens: int = Field(
        default=0,
        description=(
            "Hard ceiling on cumulative total_tokens for one run; ends the run "
            "budget_exceeded with the partial answer. <=0 disables. When a "
            "provider reports absent/all-zero usage, the local token estimator "
            "(agent/context.py) fills in, so the cap works against local "
            "OpenAI-compatible servers too."
        ),
        validation_alias=AliasChoices("run_max_tokens", "max_run_tokens"),
    )
    run_max_seconds: float = Field(
        default=0,
        allow_inf_nan=False,
        description=(
            "Hard wall-clock ceiling on one accepted turn, measured immediately "
            "after the session claim and enforced across routing, retries, LLM/tool "
            "calls, and backoff; ends the run deadline_exceeded with the partial "
            "answer. <=0 disables."
        ),
        validation_alias=AliasChoices("run_max_seconds", "max_run_seconds"),
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
    context_strategy: Literal["naive", "compaction"] = Field(
        default="naive",
        description=(
            "Context assembly strategy: 'naive' (pass-through + budget warning) "
            "or 'compaction' (summarize over-budget middle history)."
        ),
    )
    context_default_window_tokens: int = Field(
        default=32768,
        gt=0,
        description=(
            "Assumed context window (tokens) for models whose models.yaml entry "
            "has no context_window, and for unorchestrated mode."
        ),
    )
    context_safety_margin_tokens: int = Field(
        default=1024,
        ge=0,
        description=(
            "Headroom subtracted from the context window (with max output "
            "tokens) when computing the input budget; absorbs estimator error."
        ),
    )
    context_recent_messages: int = Field(
        default=6,
        gt=0,
        description=(
            "Recent protocol-safe units (a user turn, a no-tool assistant turn, "
            "or an assistant tool call plus its results) kept verbatim under "
            "compaction. Shrinks automatically if the tail alone overflows."
        ),
    )
    context_summary_max_tokens: int = Field(
        default=512,
        gt=0,
        description="Output cap for the one-call compaction summarizer.",
    )

    # Application paths. All runtime config lives under config/ by convention.
    mcp_config_path: Path = Field(default=CONFIG_DIR / "mcp_config.yaml")
    loop_max_iterations: int = Field(
        default=10,
        gt=0,
        validation_alias=AliasChoices(
            "loop_max_iterations",
            "max_loop_iterations",
        ),
    )
    log_level: str = Field(default="INFO")

    # Optional API-key gate for /chat, /chat/stream, and /v1/*; /health stays open.
    hyphae_api_key: str = Field(
        default="",
        description=(
            "Optional API key protecting /chat, /chat/stream, /v1/*. Empty "
            "disables auth; set enforces it. Accepts X-API-Key or Bearer."
        ),
        validation_alias=AliasChoices("hyphae_api_key", "harness_api_key"),
    )

    # JSONL traces include full prompts/tool data; protect the file accordingly.
    trace_enabled: bool = Field(
        default=False,
        description="Persist the loop's event stream as a JSONL trace.",
    )
    trace_jsonl_path: Path = Field(
        default=Path("traces/harness.jsonl"),
        description="Append-only JSONL trace file. Parent dirs are created.",
        validation_alias=AliasChoices("trace_jsonl_path", "trace_path"),
    )

    # In-memory session bounds; <= 0 disables the respective dimension.
    session_ttl_seconds: int = Field(
        default=3600,
        description="Idle TTL (seconds) before an in-memory session is evicted.",
    )
    session_capacity: int = Field(
        default=1000,
        description="Max sessions retained in memory; oldest-updated evicted first.",
        validation_alias=AliasChoices("session_capacity", "session_max_count"),
    )

    # When off, routes use the default LLM and full tool inventory directly.
    orchestration_enabled: bool = Field(
        default=True,
        description="Master toggle for the orchestration layer.",
    )
    models_config_path: Path = Field(
        default=CONFIG_DIR / "models.yaml",
        description="Path to the model registry YAML consumed by the orchestrator.",
    )
    orchestrator_prompt_path: Path = Field(
        default=CONFIG_DIR / "orchestrator_prompt.md",
        description="Path to the orchestrator's system prompt file (markdown or plain text).",
    )
    orchestrator_model_id: str = Field(
        default="",
        description=(
            "Override which model_id the orchestrator itself uses to make routing "
            "decisions. Empty = use the default entry from models.yaml."
        ),
    )

    @field_validator(
        "tool_result_max_chars",
        "openai_compat_tool_activity_max_chars",
        "run_max_tokens",
        "abort_after_consecutive_tool_failures",
        "context_default_window_tokens",
        "context_safety_margin_tokens",
        "context_recent_messages",
        "context_summary_max_tokens",
        "loop_max_iterations",
        "session_ttl_seconds",
        "session_capacity",
        mode="before",
    )
    @classmethod
    def _integers_are_not_booleans_or_floats(cls, value: Any) -> Any:
        return reject_bool_or_float_for_int(value)

    @field_validator(
        "mcp_config_path",
        "models_config_path",
        "orchestrator_prompt_path",
        "trace_jsonl_path",
    )
    @classmethod
    def _resolve_application_paths(cls, value: Path) -> Path:
        try:
            return resolve_application_path(value)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PydanticCustomError(
                "path_resolution",
                "application path could not be resolved",
            ) from exc

    @field_validator("orchestrator_model_id")
    @classmethod
    def _orchestrator_model_id_is_valid(cls, value: str) -> str:
        return _validate_identifier(
            value,
            "orchestrator model identifier",
            allow_blank=True,
        )

    @model_validator(mode="after")
    def _validate_default_context_budget(self) -> "Settings":
        if (
            self.llm.max_tokens + self.context_safety_margin_tokens
            >= self.context_default_window_tokens
        ):
            raise PydanticCustomError(
                "inconsistent_context_bounds",
                "default max_tokens plus safety margin must be smaller than context window",
            )
        return self

    def interpolation_environment(self) -> dict[str, str]:
        """Return Settings-owned values for ``${ENV_VAR}`` interpolation.

        Only values supplied by Settings sources are included; declared defaults
        are not synthesized. Real process variables are applied last, preserving
        their precedence without mutating ``os.environ``.
        """
        return dict(self._interpolation_values)

    def interpolation_value(self, name: str) -> str:
        """Resolve one explicitly requested interpolation variable."""
        return self._interpolation_values[name]

    def api_key_for_provider(self, provider: str) -> str:
        """Return the API key for a provider, or raise if it's needed but missing.

        Unknown providers fall back to `<PROVIDER>_API_KEY`, letting new
        credentialed providers register without changing Settings.
        """
        typed = {
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "openai_compatible": self.openai_api_key,
            "openai": self.openai_api_key,
        }
        expected_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(provider, f"{provider.upper()}_API_KEY")
        key = typed.get(provider)
        if key is None:
            key = self._interpolation_values.get(expected_env, "")
        if not key:
            raise CredentialUnavailableError(
                f"no API key found in env for provider {provider!r} "
                f"(expected {expected_env}). Set it in the process "
                "environment or project .env file."
            )
        return key


# Lazily populated so importing application modules never reads credentials.
_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance, building it on first call."""
    global _settings_cache
    if _settings_cache is None:
        try:
            _settings_cache = Settings()
        except ValidationError as exc:
            raise ConfigValidationError(
                REPOSITORY_ROOT / ".env",
                f"invalid application settings ({safe_validation_summary(exc)})",
                cause=exc,
            ) from None
        except (SettingsError, OSError, UnicodeError) as exc:
            raise ConfigLoadError(
                REPOSITORY_ROOT / ".env",
                "could not load application settings",
                cause=exc,
            ) from None
    return _settings_cache


def reset_settings() -> None:
    """Clear the settings cache. For tests, or after late env var changes."""
    global _settings_cache
    _settings_cache = None
