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

import secrets
from typing import Optional

from fastapi import HTTPException, Request

from agent import SessionGuard, SessionStore, ToolPolicy, Tracer
from mcp_layer import MCPManager
from orchestrator import LLMRegistry, Orchestrator

from .turn import TurnRunner


async def get_mcp(request: Request) -> MCPManager:
    return request.app.state.mcp


async def get_store(request: Request) -> SessionStore:
    return request.app.state.store


async def get_guard(request: Request) -> SessionGuard:
    return request.app.state.guard


async def get_settings_obj(request: Request):
    return request.app.state.settings


async def get_orchestrator(request: Request) -> Optional[Orchestrator]:
    """Return the orchestrator, or None when orchestration is disabled."""
    return getattr(request.app.state, "orchestrator", None)


async def get_registry(request: Request) -> Optional[LLMRegistry]:
    """Return the LLM registry, or None when orchestration is disabled."""
    return getattr(request.app.state, "registry", None)


async def get_tracer(request: Request) -> Optional[Tracer]:
    """Return the run tracer, or None when tracing is disabled."""
    return getattr(request.app.state, "tracer", None)


async def get_policy(request: Request) -> Optional[ToolPolicy]:
    """Return the tool policy, or None when none is configured.

    getattr-with-default so hand-wired smoke tests that don't set app.state.policy
    still route (the loop treats None as allow-all).
    """
    return getattr(request.app.state, "policy", None)


def _presented_api_key(request: Request) -> Optional[str]:
    """Extract the caller's API key from either accepted header form.

    `X-API-Key: <key>` takes precedence; otherwise `Authorization: Bearer <key>`
    (the form OpenWebUI sends for OpenAI connections). None when neither is present.
    """
    xkey = request.headers.get("X-API-Key")
    if xkey:
        return xkey
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip() or None
    return None


async def require_api_key(request: Request) -> None:
    """Route dependency enforcing the optional API key.

    No-op when `harness_api_key` is unset (single-operator dev default). When set,
    rejects any request without a matching key (constant-time compare) with 401.
    Applied only to the chat routes; /health stays open. Auth lives entirely at
    this route layer -- nothing auth-related crosses into agent/ or llm/.
    """
    configured = request.app.state.settings.harness_api_key
    if not configured:
        return
    presented = _presented_api_key(request)
    # Compare on bytes: Starlette decodes headers as latin-1, so a non-ASCII key is
    # a non-ASCII str and secrets.compare_digest(str, str) would raise TypeError
    # (-> 500). Encoding both sides yields a clean 401 and keeps the constant-time
    # comparison.
    if presented is None or not secrets.compare_digest(
        presented.encode("utf-8"), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


async def get_turn_runner(request: Request) -> TurnRunner:
    """Assemble the TurnRunner seam from the published app.state singletons.

    Renderers depend on this one object instead of wiring nine, and it is the
    single place a future out-of-process adapter would swap for an HTTP-backed
    runner. The TurnRunner is a cheap value object, so building it per request
    (rather than stashing one on app.state) keeps lifespan and the hand-wired
    smoke tests free of an extra field while staying override-friendly.
    """
    return TurnRunner(
        legacy_llm=request.app.state.legacy_llm,
        mcp=request.app.state.mcp,
        store=request.app.state.store,
        guard=request.app.state.guard,
        settings=request.app.state.settings,
        orchestrator=getattr(request.app.state, "orchestrator", None),
        registry=getattr(request.app.state, "registry", None),
        policy=getattr(request.app.state, "policy", None),
        tracer=getattr(request.app.state, "tracer", None),
    )
