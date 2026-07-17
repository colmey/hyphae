# api/routes.py

"""Native /health, /chat, and /chat/stream routes."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from agent import (
    ReasoningEvent,
    SessionNotFoundError,
    SessionStore,
)
from agent.tracing import event_record
from mcp_layer import MCPManager
from orchestrator import LLMRegistry

from .dependencies import (
    get_mcp,
    get_registry,
    get_settings_obj,
    get_store,
    get_turn_runner,
    require_api_key,
)
from .schemas import HealthResponse
from .turn import PersistencePolicy, TurnRequest, TurnRunner

logger = logging.getLogger(__name__)
router = APIRouter()


async def _prompt_from_body(request: Request) -> str:
    """Read the native plain-text prompt body, rejecting empty requests."""
    prompt = (await request.body()).decode("utf-8").strip()
    if not prompt:
        raise HTTPException(
            status_code=400, detail="empty body; send the prompt as plain text"
        )
    return prompt


async def _session_from_header(request: Request, store: SessionStore):
    """Resolve or create the native session named by X-Session-Id."""
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return await store.create()
    try:
        return await store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")


@router.get("/health", response_model=HealthResponse)
async def health(
    mcp: MCPManager = Depends(get_mcp),
    settings=Depends(get_settings_obj),
    registry: Optional[LLMRegistry] = Depends(get_registry),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider=settings.llm_provider,
        model=settings.llm_model,
        connected_servers=mcp.connected_servers,
        tool_count=len(mcp.list_tools()),
        orchestration_enabled=registry is not None,
        available_model_ids=registry.model_ids if registry is not None else [],
    )


@router.post(
    "/chat", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)]
)
async def chat(
    request: Request,
    store: SessionStore = Depends(get_store),
    runner: TurnRunner = Depends(get_turn_runner),
) -> PlainTextResponse:
    # Resolve before routing so follow-up turns include conversation history.
    prompt = await _prompt_from_body(request)
    session = await _session_from_header(request, store)

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
    store: SessionStore = Depends(get_store),
    runner: TurnRunner = Depends(get_turn_runner),
) -> EventSourceResponse:
    """Native live event stream for one turn — the activity feed.

    Same plain-text/session contract as `/chat`, but streams typed loop events
    over SSE. Frames use the same `event_record` mapping as tracing.
    """
    # Resolve up front so X-Session-Id is known before the stream opens.
    prompt = await _prompt_from_body(request)
    session = await _session_from_header(request, store)

    async def _events() -> AsyncIterator[dict]:
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
