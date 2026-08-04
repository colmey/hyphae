# api/routes.py

"""Native /health, /chat, and /chat/stream routes."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from agent import (
    ReasoningEvent,
    Session,
    SessionNotFoundError,
)
from agent.tracing import event_record
from mcp_runtime import MCPServerState

from .dependencies import (
    ApplicationRuntime,
    get_application_runtime,
    require_api_key,
)
from .schemas import HealthResponse, MCPServerHealth
from .turn import OrchestratedRouting, PersistencePolicy, TurnRequest, TurnRunner

logger = logging.getLogger(__name__)
router = APIRouter()
type _SSEFrame = dict[str, str]


async def _prompt_from_body(request: Request) -> str:
    """Read the native plain-text prompt body, rejecting empty requests."""
    prompt = (await request.body()).decode("utf-8").strip()
    if not prompt:
        raise HTTPException(
            status_code=400, detail="empty body; send the prompt as plain text"
        )
    return prompt


async def _session_from_header(request: Request, runner: TurnRunner) -> Session:
    """Resolve or create the native session named by X-Session-Id."""
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return await runner.store.create()
    try:
        return await runner.store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")


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
) -> PlainTextResponse:
    # Resolve before routing so follow-up turns include conversation history.
    prompt = await _prompt_from_body(request)
    runner = runtime.turn_runner()
    session = await _session_from_header(request, runner)

    result = await runner.run(
        TurnRequest(
            prompt=prompt,
            session=session,
            persistence=PersistencePolicy.PERSISTENT,
        )
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
        try:
            turn = TurnRequest(
                prompt=prompt,
                session=session,
                persistence=PersistencePolicy.PERSISTENT,
                stream=True,
            )
            async with runner.open(turn) as execution:
                async for event in execution.events:
                    if isinstance(event, ReasoningEvent):
                        continue
                    step += 1
                    yield {
                        "data": json.dumps(
                            event_record(
                                event, run_id=execution.metadata.run_id, step=step
                            )
                        )
                    }
        except HTTPException as exc:
            # The stream is already 200, so send guard failures as error frames.
            yield {"data": json.dumps({"type": "error", "message": str(exc.detail)})}
        except Exception as exc:  # noqa: BLE001 - stream is open; surface, don't crash.
            logger.exception("error during /chat/stream")
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(_events(), headers={"X-Session-Id": session.session_id})
