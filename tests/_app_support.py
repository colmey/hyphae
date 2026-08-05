"""Small, explicit seams for hermetic ASGI adapter tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from typing import Any, Iterator

from fastapi import FastAPI

from agent import (
    InMemorySessionStore,
    RunLimits,
    SessionGuard,
    SessionStore,
    Tracer,
)
from application import (
    ApplicationMCP,
    ApplicationRuntime,
    OrchestratedRouting,
    RoutingRuntime,
    UnorchestratedRouting,
)
from config import Settings
from llm.client import LLMClient
from mcp_runtime import MCPServerStatus, Tool
from orchestrator.contracts import ModelRegistry, RoutingService
from tooling import ToolCallResult, ToolRuntime


class EmptyMCP:
    @property
    def connected_servers(self) -> list[str]:
        return []

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]:
        return ()

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        return []

    def list_tools(self) -> list[tuple[str, Tool]]:
        return []

    @asynccontextmanager
    async def open_turn(
        self, *, timeout_seconds: float | None = None
    ) -> AsyncIterator[ToolRuntime]:
        yield self

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raise AssertionError(f"unexpected tool dispatch: {name} {arguments!r}")


class _StaticApplicationMCP:
    """Adapt a fixed ToolRuntime to the empty-health application MCP seam."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._runtime = runtime

    @property
    def connected_servers(self) -> list[str]:
        return []

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]:
        return ()

    def list_tools(self) -> list[tuple[str, Tool]]:
        return []

    @asynccontextmanager
    async def open_turn(
        self, *, timeout_seconds: float | None = None
    ) -> AsyncIterator[ToolRuntime]:
        yield self._runtime


@contextmanager
def wired_app(
    llm: LLMClient,
    *,
    mcp: ApplicationMCP | ToolRuntime | None = None,
    registry: ModelRegistry | None = None,
    orchestrator: RoutingService | None = None,
    tracer: Tracer | None = None,
    store: SessionStore | None = None,
    settings_overrides: Mapping[str, object] | None = None,
) -> Iterator[tuple[FastAPI, Settings]]:
    """Publish deterministic dependencies and restore global app state afterward."""
    from main import app

    previous = dict(app.state._state)
    settings = Settings.model_validate(
        {
            "orchestration_enabled": False,
            **dict(settings_overrides or {}),
        }
    )
    tool_source = mcp if mcp is not None else EmptyMCP()
    application_mcp: ApplicationMCP = (
        tool_source
        if isinstance(tool_source, ApplicationMCP)
        else _StaticApplicationMCP(tool_source)
    )
    routing = (
        OrchestratedRouting(
            orchestrator=orchestrator,
            registry=registry,
            agent_system_prompt="trusted agent system",
        )
        if orchestrator is not None and registry is not None
        else UnorchestratedRouting(
            llm=llm,
            model_id=settings.llm.model,
        )
    )
    app.state.runtime = ApplicationRuntime(
        settings=settings,
        routing=routing,
        limits=RunLimits.from_settings(settings),
        mcp=application_mcp,
        store=(
            store
            if store is not None
            else InMemorySessionStore(
                ttl_seconds=settings.session_ttl_seconds,
                max_count=settings.session_capacity,
                session_history_max_chars=settings.session_history_max_chars,
            )
        ),
        guard=SessionGuard(),
        policy=None,
        tracer=tracer,
    )
    try:
        yield app, settings
    finally:
        app.state._state.clear()
        app.state._state.update(previous)


def runtime_of(app: FastAPI) -> ApplicationRuntime:
    """Return the typed runtime published by ``wired_app`` or lifespan."""
    runtime = app.state.runtime
    assert isinstance(runtime, ApplicationRuntime)
    return runtime


def replace_routing(app: FastAPI, routing: RoutingRuntime) -> ApplicationRuntime:
    """Atomically replace a hand-wired application's routing runtime."""
    runtime = replace(runtime_of(app), routing=routing)
    app.state.runtime = runtime
    return runtime
