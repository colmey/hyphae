# agent/events.py

"""
Events yielded by the agent loop.

The loop is an async generator that streams these events as the conversation
unfolds. Non-streaming callers (the current /chat endpoint) collect them all
before responding. Streaming callers (future SSE endpoint) forward them
directly to the client.

Events are dataclasses, not plain dicts, so callers get type checking and
can dispatch with isinstance. The JSON serialization happens at the API
boundary, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class TextEvent:
    """The model emitted a text block. Cumulative across an iteration if the
    model produced multiple text blocks in one turn."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ReasoningEvent:
    """The provider surfaced reasoning text for trace/debug consumers."""

    text: str
    type: Literal["reasoning"] = "reasoning"


@dataclass
class ToolCallEvent:
    """The model decided to invoke a tool. Emitted before the call runs."""

    id: str
    name: str
    input: dict
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResultEvent:
    """A tool call completed (successfully or not)."""

    id: str
    name: str
    content: str
    is_error: bool
    # Wall-clock duration of the mcp.call_tool() execution, in milliseconds.
    # None when the call was short-circuited (stall detection) rather than run.
    latency_ms: float | None = None
    type: Literal["tool_result"] = "tool_result"


@dataclass
class UsageEvent:
    """Token usage reported for one LLM completion (one loop iteration)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    thinking_tokens: int = 0
    cached_tokens: int = 0
    iteration: int = 0
    # Wall-clock duration of the llm.complete() call (incl. retries) in ms.
    latency_ms: float | None = None
    type: Literal["usage"] = "usage"


@dataclass
class OrchestrationDecisionEvent:
    """The orchestrator picked a model, tool subset, and system prompt.

    Emitted by TurnRunner before run_agent() runs, so that
    clients can see why a particular model was chosen. Carries the same
    fields as orchestrator.schemas.OrchestrationResult but lives here so
    the API layer doesn't have to import the orchestrator package just to
    type-check event serialization.
    """

    model_id: str
    tools: list[str]
    system_prompt: str
    fallback_used: bool = False
    thinking_level: str | None = None
    type: Literal["orchestration"] = "orchestration"


@dataclass
class DoneEvent:
    reason: str
    iterations: int
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    type: Literal["done"] = "done"


@dataclass
class ErrorEvent:
    """An unexpected failure inside the loop. Use for surfacing errors the
    model itself never sees (e.g. an LLM API error after retries). Tool
    failures don't go here — those become ToolResultEvent(is_error=True)
    so the model can react."""

    message: str
    type: Literal["error"] = "error"


Event = Union[
    TextEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    OrchestrationDecisionEvent,
    DoneEvent,
    ErrorEvent,
]
