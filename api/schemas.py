# api/schemas.py

"""
Pydantic schemas for the HTTP API.

The chat endpoint is plain-text in / plain-text out (see api/routes.py), so it
has no request/response body schema. What remains here are the JSON shapes the
harness still uses:
  - HealthResponse: the GET /health body.
  - TokenUsage: internal token-cost value surfaced by buffered turns.
  - OrchestrationInfo: legacy routing shape retained for the later boundary
    cleanup; active turns use api.turn.TurnMetadata instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mcp_layer import MCPServerState


class TokenUsage(BaseModel):
    """Token counts: what a request spent."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0


class OrchestrationInfo(BaseModel):
    """Legacy routing shape superseded by ``api.turn.TurnMetadata``.

    Retained until the planned boundary cleanup removes old API models.
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


class MCPServerHealth(BaseModel):
    """One configured enabled MCP server's startup status."""

    name: str
    state: MCPServerState
    last_error: str | None
    tool_count: int


class HealthResponse(BaseModel):
    """Body for GET /health."""

    status: Literal["ok", "degraded"]
    provider: str
    model: str
    connected_servers: list[str]
    tool_count: int
    mcp_servers: list[MCPServerHealth]
    orchestration_enabled: bool = Field(
        default=False,
        description="True if the orchestration layer is active.",
    )
    available_model_ids: list[str] = Field(
        default_factory=list,
        description="Model IDs the orchestrator may route to. Empty when orchestration is disabled.",
    )
