# config/schemas.py

"""Typed models for all file-based harness config.

MCP server registry + tool policy (mcp_config.yaml) and the orchestrator
model registry (models.yaml). Pure pydantic: this module imports nothing
from the rest of the harness, so any layer may depend on it.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
                raise ValueError(
                    "tool_policy.mode 'allow_list' requires a non-empty 'allow' list"
                )
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
            allowed = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            )
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
