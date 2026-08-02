"""Small, explicit seams for hermetic ASGI adapter tests."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any, Iterator

from agent import InMemorySessionStore, SessionGuard
from config import Settings
from tooling import ToolCallResult


class EmptyMCP:
    connected_servers: list[str] = []

    def status_snapshot(self) -> tuple[Any, ...]:
        return ()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[Any]:
        return []

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        yield self

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        raise AssertionError(f"unexpected tool dispatch: {name} {arguments!r}")


class _StaticTurnOwner:
    """Give an agent-level ToolRuntime the accepted-turn owner used by ASGI tests."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    @asynccontextmanager
    async def open_turn(self, *, timeout_seconds: float | None = None):
        yield self._runtime


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
    app.state.unorchestrated_llm = llm
    runtime = mcp if mcp is not None else EmptyMCP()
    app.state.mcp = (
        runtime if hasattr(runtime, "open_turn") else _StaticTurnOwner(runtime)
    )
    app.state.store = InMemorySessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_count=settings.session_capacity,
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
