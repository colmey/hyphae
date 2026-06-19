"""Agent layer: session state, events, and the reasoning loop.

Public surface used by main.py and api/routes.py:
  - Session, SessionStore, InMemorySessionStore, SessionNotFoundError
  - SessionGuard, SessionBusyError
  - All event types
  - run_agent
"""

from .events import (
    DoneEvent,
    ErrorEvent,
    Event,
    OrchestrationDecisionEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .loop import run_agent
from .session import (
    InMemorySessionStore,
    Session,
    SessionBusyError,
    SessionGuard,
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
    "Session",
    "SessionBusyError",
    "SessionGuard",
    "SessionNotFoundError",
    "SessionStore",
    "TextEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "Tracer",
    "UsageEvent",
    "build_tracer",
    "run_agent",
    "run_logger",
]