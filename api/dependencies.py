"""Typed FastAPI application composition and dependency providers."""

from __future__ import annotations

import secrets
from fastapi import Depends, HTTPException, Request

from application import ApplicationRuntime
from .public_errors import (
    authentication_failed_error,
    public_error_from_exception,
)


async def get_application_runtime(request: Request) -> ApplicationRuntime:
    """Narrow Starlette's dynamic state seam to the published runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, ApplicationRuntime):
        raise RuntimeError("application runtime is unavailable")
    return runtime


def turn_http_exception(exc: Exception) -> HTTPException:
    """Adapt an application outcome to the shared public-error handler."""
    error = public_error_from_exception(exc)
    return HTTPException(status_code=error.status, detail=error)


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
        error = authentication_failed_error()
        raise HTTPException(status_code=error.status, detail=error)
