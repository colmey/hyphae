"""Typed FastAPI application composition and dependency providers."""

from __future__ import annotations

import secrets
from fastapi import Depends, HTTPException, Request

from agent import SessionBusyError, SessionCapacityError
from application import (
    ApplicationRuntime,
    ExecutionProtocolError,
    InvalidModelError,
    ModelInventoryError,
    RuntimeConfigurationError,
)


async def get_application_runtime(request: Request) -> ApplicationRuntime:
    """Narrow Starlette's dynamic state seam to the published runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, ApplicationRuntime):
        raise RuntimeError("application runtime is unavailable")
    return runtime


def turn_http_exception(exc: Exception) -> HTTPException | None:
    """Map the small set of application errors owned by the HTTP adapter."""
    if isinstance(exc, InvalidModelError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ModelInventoryError):
        return HTTPException(status_code=500, detail=str(exc))
    if isinstance(exc, SessionBusyError):
        return HTTPException(
            status_code=409,
            detail=f"session {exc.args[0]!r} is processing another request",
        )
    if isinstance(exc, SessionCapacityError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (ExecutionProtocolError, RuntimeConfigurationError)):
        return HTTPException(status_code=500, detail=str(exc))
    return None


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
