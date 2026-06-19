# api/dependencies.py

"""
FastAPI dependency providers.

These pull pre-built objects off `app.state` (populated by main.py's
lifespan) and inject them into route handlers via Depends(). Going through
Depends() rather than reaching into the request directly keeps routes
clean and makes them trivially testable with overrides.

Orchestration deps may be None at runtime when settings.orchestration_enabled
is false (or when models.yaml is missing). Routes must handle the None case
and fall back to legacy behavior; see api/routes.py.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from agent import SessionGuard, SessionStore, Tracer
from llm.client import LLMClient
from mcp_layer import MCPManager
from orchestrator import LLMRegistry, Orchestrator


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_mcp(request: Request) -> MCPManager:
    return request.app.state.mcp


def get_store(request: Request) -> SessionStore:
    return request.app.state.store


def get_guard(request: Request) -> SessionGuard:
    return request.app.state.guard


def get_settings_obj(request: Request):
    return request.app.state.settings


def get_orchestrator(request: Request) -> Optional[Orchestrator]:
    """Return the orchestrator, or None when orchestration is disabled."""
    return getattr(request.app.state, "orchestrator", None)


def get_registry(request: Request) -> Optional[LLMRegistry]:
    """Return the LLM registry, or None when orchestration is disabled."""
    return getattr(request.app.state, "registry", None)


def get_tracer(request: Request) -> Optional[Tracer]:
    """Return the run tracer, or None when tracing is disabled."""
    return getattr(request.app.state, "tracer", None)