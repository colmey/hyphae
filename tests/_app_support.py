"""Small, explicit seams for hermetic ASGI adapter tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from agent import InMemorySessionStore, SessionGuard
from config import Settings


class EmptyMCP:
    connected_servers: list[str] = []

    def status_snapshot(self) -> tuple[Any, ...]:
        return ()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[Any]:
        return []


@contextmanager
def wired_app(
    llm: Any,
    *,
    mcp: Any | None = None,
    registry: Any | None = None,
) -> Iterator[tuple[Any, Settings]]:
    """Publish deterministic dependencies and restore global app state afterward."""
    from main import app

    previous = dict(app.state._state)
    settings = Settings(_env_file=None, orchestration_enabled=False)
    app.state.settings = settings
    app.state.llm = llm
    app.state.mcp = mcp if mcp is not None else EmptyMCP()
    app.state.store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_max_count,
    )
    app.state.guard = SessionGuard()
    app.state.registry = registry
    app.state.orchestrator = None
    app.state.policy = None
    app.state.tracer = None
    try:
        yield app, settings
    finally:
        app.state._state.clear()
        app.state._state.update(previous)
