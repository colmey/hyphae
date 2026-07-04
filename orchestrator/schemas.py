# orchestrator/schemas.py

"""
Typed schemas for the orchestration layer.

Three things live here:
  1. ModelEntry / ModelsConfig  -- the typed parse of models.yaml.
  2. OrchestrationResult        -- the structured output the orchestrator
                                   LLM is required to produce.
  3. OrchestrationDecision      -- the orchestrator's per-call return value:
                                   result + whether the fallback was used.

We use Pydantic (not dataclasses) for the first two because:
  - models.yaml is config and benefits from Pydantic's validation/discriminator
    machinery (same pattern as harness_config.MCPConfig).
  - OrchestrationResult is structured-output material -- we want to hand the
    class object to provider SDKs that accept Pydantic models as response
    schemas (Gemini's response_schema, etc.).

OrchestrationDecision is a plain dataclass because nothing serializes it to
JSON; it's an in-process return type for the route to consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.client import supported_providers
from mcp_layer.manager import NAMESPACE_SEP


# ---------------------------------------------------------------------------
# models.yaml schema
# ---------------------------------------------------------------------------

class ModelEntry(BaseModel):
    """One model the orchestrator may route to.

    `provider` flows into the existing LLM client factory, so any provider
    supported by llm/client.py works without changes here.

    `description` is what the orchestrator LLM sees when picking a model.
    It is NOT used by the runtime -- write it for an LLM audience, not a
    human one.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        ...,
        description="LLM provider name; must be registered in llm.client._PROVIDERS.",
    )
    model: str = Field(..., description="Provider-specific model identifier.")

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        """Reject providers the harness can't build.

        The provider registry in llm/client.py is the single source of truth
        for what providers exist; validating against it here means models.yaml
        can never name a provider with no builder (which would only surface as a
        runtime NotImplementedError on the first request).
        """
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

        # Reject model IDs that could clash with our other namespacing (just
        # alphanumeric + dash/underscore/dot). Keeps logs and routing keys clean.
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


# ---------------------------------------------------------------------------
# Orchestrator structured output
# ---------------------------------------------------------------------------

class OrchestrationResult(BaseModel):
    """The structured output the orchestrator LLM must produce.

    NOTE: deliberately NOT using `extra="forbid"`.

    Pydantic emits `additionalProperties: false` into the JSON Schema when
    that config is set. Provider structured-output APIs that consume the
    schema (notably Gemini's `response_schema`) restrict the schema dialect
    and reject `additionalProperties`, causing the LLM call to 400 out
    before the model even sees the request.

    Safety-wise this costs us nothing: the Orchestrator runs `_sanitize()`
    on the parsed result before returning it, which already drops any tool
    names not in the live MCP inventory and rewrites unknown model_ids to
    the registry default. Any "extra" fields the LLM might hallucinate are
    silently ignored by Pydantic's default `extra="ignore"` behavior.
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
        """Lenient coercion, same spirit as the orchestrator's _sanitize().

        The field is typed as a Literal so Gemini's structured output
        constrains it to the three tiers. But if a stray value still arrives
        (wrong casing, a hallucinated tier, a non-Gemini provider that ignores
        the schema), we clamp to 'medium' rather than raising -- a single odd
        field must never sink an otherwise-valid orchestration decision.
        """
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered in ("low", "medium", "high"):
                return lowered
        return "medium"


# ---------------------------------------------------------------------------
# Orchestrator return value (in-process only)
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationDecision:
    """The Orchestrator's per-call output.

    Wraps the structured result with metadata about whether orchestration
    actually ran or fell back to safe defaults. Callers (the route, smoke
    tests) read `fallback_used` to distinguish a real LLM-driven decision
    from a degraded one. `fallback_reason` is a short human-readable string
    explaining why (LLM call failed, output unparseable, etc.); None on
    success.

    Plain dataclass on purpose -- this type never crosses a serialization
    boundary. The route extracts `result` and `fallback_used` to build the
    JSON-facing OrchestrationInfo / OrchestrationDecisionEvent.
    """

    result: OrchestrationResult
    fallback_used: bool = False
    fallback_reason: str | None = None


# ---------------------------------------------------------------------------
# Tool priority hints (in-process only)
# ---------------------------------------------------------------------------

@dataclass
class ToolPreferences:
    """Normalized caller hints about which tools to prioritize.

    Built from the request's `MCP` field. `preferred_tools` are namespaced
    ({server}__{tool}) names; `tool_arg_hints` maps each namespaced tool to
    the argument names the caller intends to use (informational context only).

    A soft hint, not a filter: the orchestrator favors preferred tools and
    they are guaranteed to be exposed, but other tools remain available.

    Plain dataclass on purpose — this never crosses a serialization boundary;
    it's an in-process value the route hands to Orchestrator.decide().
    """

    preferred_tools: list[str] = field(default_factory=list)
    tool_arg_hints: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.preferred_tools)

    @classmethod
    def from_request(cls, prefs: Iterable) -> "ToolPreferences":
        """Normalize the request `MCP` list into namespaced preferences.

        `prefs` is an iterable of objects with `.name` (server) and `.tools`
        (dict of tool name -> arg names) — i.e. api.schemas.MCPServerPreference.
        Duck-typed to avoid an api -> orchestrator import dependency. Tool
        names are namespaced here with the same `{server}__{tool}` convention
        MCPManager uses, so they line up with the live inventory. Validity
        (does the tool actually exist?) is enforced later, in the route's
        filter against mcp.get_tools_for_llm().
        """
        preferred: list[str] = []
        arg_hints: dict[str, list[str]] = {}
        for server in prefs or []:
            for tool_name, args in (server.tools or {}).items():
                namespaced = f"{server.name}{NAMESPACE_SEP}{tool_name}"
                preferred.append(namespaced)
                arg_hints[namespaced] = list(args or [])
        return cls(preferred_tools=preferred, tool_arg_hints=arg_hints)