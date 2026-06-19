# api/schemas.py

"""
Pydantic schemas for the HTTP API.

The chat endpoint is plain-text in / plain-text out (see api/routes.py), so it
has no request/response body schema. What remains here are the JSON shapes the
harness still uses:
  - HealthResponse: the GET /health body.
  - OrchestrationInfo / TokenUsage: internal value objects produced while
    running a turn (routing metadata, token cost) — surfaced in logs/traces,
    not in the /chat response.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token counts: what a request spent."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0


class OrchestrationInfo(BaseModel):
    """The orchestrator's routing decision for one request.

    Null when orchestration is disabled. Carried internally for logging/tracing.
    """
    model_id: str = Field(..., description="The model_id the orchestrator selected.")
    tools: list[str] = Field(
        default_factory=list,
        description="Namespaced tool names the orchestrator exposed to the agent.",
    )
    system_prompt: str = Field(
        ...,
        description="The system prompt the agent actually ran with (after override resolution).",
    )
    fallback_used: bool = Field(
        default=False,
        description="True if orchestration failed and a safe default was substituted.",
    )
    thinking_level: str | None = Field(
        default=None,
        description="Deliberation level (low/medium/high) the orchestrator chose. "
                    "Null in legacy/fallback when unset.",
    )


class HealthResponse(BaseModel):
    """Body for GET /health."""
    status: str
    provider: str
    model: str
    connected_servers: list[str]
    tool_count: int
    orchestration_enabled: bool = Field(
        default=False,
        description="True if the orchestration layer is active.",
    )
    available_model_ids: list[str] = Field(
        default_factory=list,
        description="Model IDs the orchestrator may route to. Empty when orchestration is disabled.",
    )
