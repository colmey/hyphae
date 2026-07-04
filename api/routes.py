# api/routes.py

"""
Native HTTP routes for the harness.

Three endpoints:
  GET  /health       -- liveness + connected MCP server summary
  POST /chat         -- run the agent loop on a plain-text prompt, return the answer
  POST /chat/stream  -- the same turn as an SSE feed of the loop's typed events

`/chat` is a plain-text dumb pipe: the request body IS the prompt and the
response body IS the answer. Session continuation rides on the `X-Session-Id`
header (echoed back on the response, alongside `X-Done-Reason`). `/chat/stream`
shares that contract but forwards events live instead of collecting them.

These are thin renderers over the shared orchestrate->loop core in
`api/turn.py`, reached through the `TurnRunner` seam; the OpenAI-compatible
`/v1/chat/completions` adapter (`api/openai_compatible.py`) is a peer renderer
over the same core.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from agent import (
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
)
from .schemas import HealthResponse
from .turn import TurnRunner

logger = logging.getLogger(__name__)
router = APIRouter()


async def _prompt_from_body(request: Request) -> str:
    """Read the native plain-text prompt body, rejecting empty requests."""
    prompt = (await request.body()).decode("utf-8").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty body; send the prompt as plain text")
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
    settings = Depends(get_settings_obj),
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


@router.post("/chat", response_class=PlainTextResponse)
async def chat(
    request: Request,
    store: SessionStore = Depends(get_store),
    runner: TurnRunner = Depends(get_turn_runner),
) -> PlainTextResponse:
    # Resolve the session before routing so a follow-up turn is routed with the
    # conversation in view. The new prompt is appended later, inside the runner.
    prompt = await _prompt_from_body(request)
    session = await _session_from_header(request, store)

    # Dumb-pipe contract: no system override, model hint, or tool prefs on the wire.
    answer, done_reason, _usage = await runner.run(prompt=prompt, session=session)
    return PlainTextResponse(
        answer,
        headers={"X-Session-Id": session.session_id, "X-Done-Reason": done_reason},
    )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    store: SessionStore = Depends(get_store),
    runner: TurnRunner = Depends(get_turn_runner),
) -> EventSourceResponse:
    """Native live event stream for one turn — the activity feed.

    Same dumb-pipe contract as `/chat` (plain-text body = prompt, optional
    `X-Session-Id` continues a session), but instead of collecting the events
    into one answer it forwards the loop's typed events *as they happen* over
    SSE: `text`, `tool_call`, `tool_result`, `usage`, `done`, `error`. A client
    that wants to show "what the agent is doing" (tool calls live, then the
    answer) reads this; clients that just want the answer keep using `/chat` or
    `/v1`. The stream ends after the `done` event.

    This is a pure renderer over the shared `_turn_events` core — it adds no
    orchestration or loop logic. Each frame's `data:` is the JSON event produced
    by the same bytes-safe `event_record` mapping the tracer uses, so there is a
    single source of truth for event serialization. Tool calls are *not* token-
    streamed (a tool call must be fully assembled before it runs), but each
    `tool_call`/`tool_result` is emitted the instant the loop reaches it.
    """
    # Resolve the session up front (mirrors /chat) so the X-Session-Id response
    # header is known before the stream opens. The new prompt is appended inside
    # _turn_events, under the same-session guard.
    prompt = await _prompt_from_body(request)
    session = await _session_from_header(request, store)

    run_id = uuid.uuid4().hex

    async def _events() -> AsyncIterator[dict]:
        step = 0
        try:
            async for event in runner.events(prompt=prompt, session=session):
                step += 1
                yield {"data": json.dumps(event_record(event, run_id=run_id, step=step))}
        except HTTPException as exc:
            # The same-session 409 surfaces here (the guard claim is the first
            # thing the runner does). The SSE response is already 200, so we
            # report it as a terminal error frame rather than an HTTP status.
            yield {"data": json.dumps({"type": "error", "message": str(exc.detail)})}
        except Exception as exc:  # noqa: BLE001 - stream is open; surface, don't crash.
            logger.exception("error during /chat/stream")
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(_events(), headers={"X-Session-Id": session.session_id})
