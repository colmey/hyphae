"""Typed FastAPI application composition and dependency providers."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request

from agent import RunLimits, SessionGuard, SessionStore, ToolPolicy, Tracer
from config import Settings
from mcp_runtime import MCPServerStatus, Tool

from .turn import RoutingRuntime, TurnRunner, TurnToolProvider


@runtime_checkable
class ApplicationMCP(TurnToolProvider, Protocol):
    """MCP turn ownership and health views consumed by the HTTP application."""

    @property
    def connected_servers(self) -> list[str]: ...

    def status_snapshot(self) -> tuple[MCPServerStatus, ...]: ...

    def list_tools(self) -> list[tuple[str, Tool]]: ...


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Process-owned dependencies published atomically to the HTTP application."""

    settings: Settings
    routing: RoutingRuntime
    limits: RunLimits
    mcp: ApplicationMCP
    store: SessionStore
    guard: SessionGuard
    policy: ToolPolicy | None
    tracer: Tracer | None

    def turn_runner(self) -> TurnRunner:
        """Derive one coherent accepted-turn owner from this composition."""
        return TurnRunner(
            routing=self.routing,
            limits=self.limits,
            mcp=self.mcp,
            store=self.store,
            guard=self.guard,
            policy=self.policy,
            tracer=self.tracer,
        )


async def get_application_runtime(request: Request) -> ApplicationRuntime:
    """Narrow Starlette's dynamic state seam to the published runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, ApplicationRuntime):
        raise RuntimeError("application runtime is unavailable")
    return runtime


def _presented_api_key(request: Request) -> str | None:
    """Extract the caller's API key from either accepted header form.

    ``X-API-Key`` takes precedence over the bearer form used by OpenWebUI.
    """
    xkey = request.headers.get("X-API-Key")
    if xkey:
        return xkey
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip() or None
    return None


async def require_api_key(
    request: Request,
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> None:
    """Enforce the optional API key for protected HTTP routes."""
    configured = runtime.settings.hyphae_api_key
    if not configured:
        return
    presented = _presented_api_key(request)
    # Comparing bytes also turns a non-ASCII decoded header into a clean 401.
    if presented is None or not secrets.compare_digest(
        presented.encode("utf-8"), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
