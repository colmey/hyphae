"""Agent layer: session state, events, and the reasoning loop.

Public surface used by hyphae.main and hyphae.api.routes:
  - Session, SessionStore, InMemorySessionStore, SessionNotFoundError,
    SessionHistoryLimitExceeded
  - SessionGuard, SessionBusyError, SessionCapacityError
  - All event types
  - run_agent
"""

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    OrchestrationDecisionEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .loop import run_agent
from .runtime import RunContext, RunLimits
from .tool_policy import PolicyDecision, ToolPolicy, PolicyVerdict, build_tool_policy
from .session import (
    InMemorySessionStore,
    Session,
    SessionBusyError,
    SessionCapacityError,
    SessionGuard,
    SessionHistoryLimitExceeded,
    SessionNotFoundError,
    SessionStore,
)
from .tracing import JSONLTracer, NoOpTracer, Tracer, build_tracer, run_logger

__all__ = [
    "DoneEvent",
    "ErrorEvent",
    "Event",
    "InMemorySessionStore",
    "JSONLTracer",
    "NoOpTracer",
    "OrchestrationDecisionEvent",
    "PolicyDecision",
    "ReasoningEvent",
    "RunContext",
    "RunLimits",
    "Session",
    "SessionBusyError",
    "SessionCapacityError",
    "SessionGuard",
    "SessionHistoryLimitExceeded",
    "SessionNotFoundError",
    "SessionStore",
    "TextEvent",
    "ToolCallEvent",
    "ToolPolicy",
    "ToolResultEvent",
    "Tracer",
    "UsageEvent",
    "PolicyVerdict",
    "build_tool_policy",
    "build_tracer",
    "run_agent",
    "run_logger",
]
