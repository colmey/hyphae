# orchestrator/schemas.py

"""Typed config, structured output, and in-process values for orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.client import supported_providers
from mcp_layer.manager import NAMESPACE_SEP


class SamplingParams(BaseModel):
    """Optional per-model sampling parameters passed through to providers."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)


class ModelEntry(BaseModel):
    """One model the orchestrator may route to."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        ...,
        description="LLM provider name; must be registered in llm.client._PROVIDERS.",
    )
    model: str = Field(..., description="Provider-specific model identifier.")

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        """Reject providers the harness can't build."""
        known = supported_providers()
        if v not in known:
            raise ValueError(
                f"unknown provider {v!r}; registered providers: {sorted(known)}. "
                "Add a builder to llm/client.py's _PROVIDERS to support it."
            )
        return v
    description: str = Field(..., description="What this model is best at.")
    max_tokens: int | None = Field(
        default=None,
        description="Per-model output cap. Falls back to Settings.llm_max_tokens when None.",
    )
    context_window: int | None = Field(
        default=None,
        description="Total context window in tokens, used for the agent loop's "
                    "context budget. Falls back to "
                    "Settings.context_default_window_tokens when None. "
                    "Provider-agnostic: a plain size, no vendor branching.",
    )
    default: bool = Field(
        default=False,
        description="At most one entry should be marked default. Used as the "
                    "registry-wide fallback and as the orchestrator's own model "
                    "unless overridden by Settings.orchestrator_model_id.",
    )
    supports_native_tools: bool = Field(
        default=True,
        description="Whether the served endpoint supports native tool/function calling. "
                    "Declared only in Phase 5; prompted-tool fallback is Phase 6.",
    )
    thinking: Literal["none", "hint-param", "think-tags"] = Field(
        default="none",
        description="How this model exposes a thinking control. 'none': no knob; "
                    "'hint-param': a request field like reasoning_effort; "
                    "'think-tags': self-emits <think> inline.",
    )
    sampling: SamplingParams | None = Field(
        default=None,
        description="Optional per-model sampling passed through to the provider request.",
    )

    def to_profile(self) -> "ModelProfile":
        from llm.schemas import ModelProfile

        s = self.sampling
        return ModelProfile(
            supports_native_tools=self.supports_native_tools,
            thinking=self.thinking,
            temperature=s.temperature if s else None,
            top_p=s.top_p if s else None,
            top_k=s.top_k if s else None,
        )


class ModelsConfig(BaseModel):
    """Typed parse of models.yaml. Maps model_id -> ModelEntry."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "ModelsConfig":
        if not self.models:
            raise ValueError("models.yaml must define at least one model")

        defaults = [mid for mid, m in self.models.items() if m.default]
        if len(defaults) > 1:
            raise ValueError(
                f"models.yaml has multiple default models: {defaults!r}. "
                "Mark exactly one entry default: true."
            )

        for mid in self.models:
            if not mid:
                raise ValueError("model_id may not be empty")
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
            bad = set(mid) - allowed
            if bad:
                raise ValueError(
                    f"model_id {mid!r} contains disallowed characters: {sorted(bad)!r}"
                )
        return self

    def default_id(self) -> str:
        """Return the model_id marked default, or the first one if none is."""
        for mid, m in self.models.items():
            if m.default:
                return mid
        return next(iter(self.models))


class OrchestrationResult(BaseModel):
    """The structured output the orchestrator LLM must produce.

    Deliberately leaves `extra="forbid"` off because Gemini rejects the
    resulting JSON Schema keyword. Runtime sanitization still validates model
    and tool IDs against live inventory.
    """

    selected_model_id: str = Field(
        ...,
        description="Must match one of the model IDs in models.yaml.",
    )
    selected_tools: list[str] = Field(
        default_factory=list,
        description="Namespaced tool names ({server}__{tool}) to expose to the agent. "
                    "Empty list = no tools.",
    )
    generated_system_prompt: str = Field(
        ...,
        description="The system instruction to run the downstream agent with.",
    )
    thinking_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="How much the downstream model should deliberate: 'low' for "
                    "lookups/single-tool calls, 'medium' for the typical case, "
                    "'high' for complex multi-step reasoning. Defaults to 'medium' "
                    "if the orchestrator omits it.",
    )

    @field_validator("thinking_level", mode="before")
    @classmethod
    def _coerce_thinking_level(cls, v: object) -> str:
        """Clamp stray values to medium rather than failing orchestration."""
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered in ("low", "medium", "high"):
                return lowered
        return "medium"


@dataclass
class OrchestrationDecision:
    """Orchestrator output plus fallback metadata."""

    result: OrchestrationResult
    fallback_used: bool = False
    fallback_reason: str | None = None


@dataclass
class ToolPreferences:
    """Normalized caller hints about which tools to prioritize.

    This is a soft hint, not a filter; other tools remain available.
    """

    preferred_tools: list[str] = field(default_factory=list)
    tool_arg_hints: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.preferred_tools)

    @classmethod
    def from_request(cls, prefs: Iterable) -> "ToolPreferences":
        """Normalize request `MCP` preferences into namespaced tool names."""
        preferred: list[str] = []
        arg_hints: dict[str, list[str]] = {}
        for server in prefs or []:
            for tool_name, args in (server.tools or {}).items():
                namespaced = f"{server.name}{NAMESPACE_SEP}{tool_name}"
                preferred.append(namespaced)
                arg_hints[namespaced] = list(args or [])
        return cls(preferred_tools=preferred, tool_arg_hints=arg_hints)
