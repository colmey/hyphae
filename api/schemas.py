# api/schemas.py

"""
Pydantic schemas for the HTTP API.

The chat endpoint is plain-text in / plain-text out (see api/routes.py), so it
has no request/response body schema. What remains here are the JSON shapes the
harness still uses:
  - HealthResponse: the GET /health body.
  - TokenUsage: internal token-cost value surfaced by buffered turns.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field

from mcp_runtime import MCPServerState

if TYPE_CHECKING:
    from agent import DoneEvent


class TokenUsage(BaseModel):
    """Token counts: what a request spent."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0

    @classmethod
    def from_done_event(cls, event: "DoneEvent") -> Self:
        """Convert flattened terminal-event usage at the HTTP boundary."""
        return cls(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            total_tokens=event.total_tokens,
            thinking_tokens=event.thinking_tokens,
        )


class MCPServerHealth(BaseModel):
    """One configured enabled MCP server's current retained status."""

    name: str
    state: MCPServerState
    last_error: str | None
    tool_count: int
    catalog_revision: int
    last_discovered_at: datetime | None
    next_refresh_at: datetime | None
    active_leases: int


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
        description="Executable model IDs advertised by the active routing mode.",
    )
