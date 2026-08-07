# hyphae/api/routes.py

"""Native /health, /chat, and /chat/stream routes."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from sse_starlette.sse import EventSourceResponse

from hyphae.agent import (
    ReasoningEvent,
    DoneEvent,
    ErrorEvent,
    Session,
    SessionCapacityError,
    SessionNotFoundError,
)
from hyphae.agent.tracing import event_record
from hyphae.mcp_runtime import MCPServerState

from .dependencies import (
    get_application_runtime,
    require_api_key,
    turn_http_exception,
)
from .request_body import read_request_body
from .public_errors import (
    invalid_request_error,
    native_error_body,
    native_sse_error_body,
    execution_protocol_error,
    public_error_from_done_reason,
    public_error_from_exception,
    unsupported_media_type_error,
)
from .schemas import HealthResponse, MCPServerHealth
from hyphae.application import (
    ApplicationRuntime,
    OrchestratedRouting,
    PersistencePolicy,
    TurnRequest,
    TurnRunner,
)

logger = logging.getLogger(__name__)
router = APIRouter()
type _SSEFrame = dict[str, str]


async def _prompt_from_body(request: Request) -> str:
    """Read the native plain-text prompt body, rejecting empty requests."""
    content_type = request.headers.get("content-type")
    if content_type is not None and content_type.split(";", 1)[0].strip().lower() != "text/plain":
        error = unsupported_media_type_error()
        raise HTTPException(status_code=error.status, detail=error)
    try:
        prompt = (await read_request_body(request)).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        error = invalid_request_error()
        raise HTTPException(status_code=error.status, detail=error) from exc
    if not prompt:
        error = invalid_request_error()
        raise HTTPException(status_code=error.status, detail=error)
    return prompt


async def _session_from_header(request: Request, runner: TurnRunner) -> Session:
    """Resolve or create the native session named by X-Session-Id."""
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        try:
            return await runner.create_session()
        except SessionCapacityError as exc:
            mapped = turn_http_exception(exc)
            raise mapped from exc
    try:
        return await runner.store.get(session_id)
    except SessionNotFoundError:
        error = public_error_from_exception(SessionNotFoundError(session_id))
        raise HTTPException(status_code=error.status, detail=error) from None


@router.get("/health", response_model=HealthResponse)
async def health(
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> HealthResponse:
    mcp = runtime.mcp
    settings = runtime.settings
    routing = runtime.routing
    if isinstance(routing, OrchestratedRouting):
        orchestration_enabled = True
        available_model_ids = routing.registry.model_ids
    elif routing.advertised_model_ids is not None:
        orchestration_enabled = False
        available_model_ids = list(routing.advertised_model_ids)
    else:
        orchestration_enabled = False
        available_model_ids = []
    server_statuses = mcp.status_snapshot()
    return HealthResponse(
        status=(
            "ok"
            if all(status.state is MCPServerState.HEALTHY for status in server_statuses)
            else "degraded"
        ),
        provider=settings.llm.provider,
        model=settings.llm.model,
        connected_servers=mcp.connected_servers,
        tool_count=len(mcp.list_tools()),
        mcp_servers=[
            MCPServerHealth(
                name=status.name,
                state=status.state,
                last_error=status.last_error,
                tool_count=status.tool_count,
                catalog_revision=status.catalog_revision,
                last_discovered_at=status.last_discovered_at,
                next_refresh_at=status.next_refresh_at,
                active_leases=status.active_leases,
            )
            for status in server_statuses
        ],
        orchestration_enabled=orchestration_enabled,
        available_model_ids=available_model_ids,
    )


@router.post(
    "/chat", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)]
)
async def chat(
    request: Request,
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> Response:
    # Resolve before routing so follow-up turns include conversation history.
    prompt = await _prompt_from_body(request)
    runner = runtime.turn_runner()
    session = await _session_from_header(request, runner)

    try:
        result = await runner.run(
            TurnRequest(
                prompt=prompt,
                session=session,
                persistence=PersistencePolicy.PERSISTENT,
            )
        )
    except Exception as exc:
        logger.exception("error during /chat")
        error = public_error_from_exception(exc)
        return JSONResponse(status_code=error.status, content=native_error_body(error))
    terminal_error = public_error_from_done_reason(result.done_reason)
    if terminal_error is not None:
        return JSONResponse(
            status_code=terminal_error.status,
            content=native_error_body(terminal_error),
        )
    return PlainTextResponse(
        result.answer,
        headers={
            "X-Session-Id": session.session_id,
            "X-Done-Reason": result.done_reason,
        },
    )


@router.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(
    request: Request,
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> EventSourceResponse:
    """Native live event stream for one turn — the activity feed.

    Same plain-text/session contract as `/chat`, but streams typed loop events
    over SSE. Frames use the same `event_record` mapping as tracing.
    """
    # Resolve up front so X-Session-Id is known before the stream opens.
    prompt = await _prompt_from_body(request)
    runner = runtime.turn_runner()
    session = await _session_from_header(request, runner)

    async def _events() -> AsyncIterator[_SSEFrame]:
        step = 0
        error_event_seen = False
        try:
            turn = TurnRequest(
                prompt=prompt,
                session=session,
                persistence=PersistencePolicy.PERSISTENT,
                stream=True,
            )
            async with runner.open(turn) as execution:
                async for event in execution.events:
                    if error_event_seen:
                        if isinstance(event, DoneEvent):
                            error = (
                                public_error_from_done_reason(event.reason)
                                or public_error_from_done_reason("provider_error")
                            )
                            assert error is not None
                            yield {"data": json.dumps(native_sse_error_body(error))}
                            return
                        continue
                    if isinstance(event, ReasoningEvent):
                        continue
                    if isinstance(event, ErrorEvent):
                        error_event_seen = True
                        continue
                    if isinstance(event, DoneEvent):
                        error = public_error_from_done_reason(event.reason)
                        if error is not None:
                            yield {"data": json.dumps(native_sse_error_body(error))}
                            return
                    step += 1
                    yield {
                        "data": json.dumps(
                            event_record(
                                event, run_id=execution.metadata.run_id, step=step
                            )
                        )
                    }
                if error_event_seen:
                    error = execution_protocol_error()
                    yield {"data": json.dumps(native_sse_error_body(error))}
        except Exception as exc:
            logger.exception("error during /chat/stream")
            error = public_error_from_exception(exc)
            yield {"data": json.dumps(native_sse_error_body(error))}

    return EventSourceResponse(_events(), headers={"X-Session-Id": session.session_id})
