"""Bounded HTTP request-body materialization for API routes."""

from __future__ import annotations

from fastapi import HTTPException, Request

from .public_errors import request_too_large_error


MAX_REQUEST_BODY_BYTES = 1_048_576


def _declared_body_length(request: Request) -> int | None:
    """Return a usable declared length, if the client supplied one."""
    try:
        return int(request.headers.get("content-length", ""))
    except ValueError:
        return None


async def read_request_body(request: Request) -> bytes:
    """Read one request body without allowing it to exceed the fixed API cap."""
    declared_length = _declared_body_length(request)
    if declared_length is not None and declared_length > MAX_REQUEST_BODY_BYTES:
        error = request_too_large_error()
        raise HTTPException(status_code=error.status, detail=error)

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BODY_BYTES:
            error = request_too_large_error()
            raise HTTPException(status_code=error.status, detail=error)
        body.extend(chunk)
    return bytes(body)
