# api/openai_compat.py

"""
OpenAI-compatible inbound adapter.

A thin translator that lets OpenWebUI / LibreChat (and any OpenAI client) talk to
the harness by pointing `base_url` at `/v1`. It owns only wire-format translation:
OpenAI JSON in -> internal Message list -> the shared core (`_turn_events` /
`_run_turn` in api/routes.py) -> an OpenAI `chat.completion` object, or an SSE
stream of `chat.completion.chunk` frames. No orchestration or loop logic lives
here -- both invariants are preserved by routing every turn through the one core.

Stateless by design: each request seeds a fresh ephemeral Session from
`messages[]` (the client re-feeds history), so there is no server-side
conversation state and the same-session 409 guard never fires.

Endpoints:
  POST /v1/chat/completions  -- non-stream JSON, or SSE when `stream: true`
  GET  /v1/models            -- the routable registry (the default model when
                                orchestration is off)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sse_starlette.sse import EventSourceResponse

from agent import (
    DoneEvent,
    SessionGuard,
    SessionStore,
    TextEvent,
    Tracer,
)
from llm.client import LLMClient
from llm.schemas import AssistantMessage, TextBlock
from mcp_layer import MCPManager
from orchestrator import LLMRegistry, Orchestrator

from .dependencies import (
    get_guard,
    get_llm,
    get_mcp,
    get_orchestrator,
    get_registry,
    get_settings_obj,
    get_store,
    get_tracer,
)
from .routes import _run_turn, _turn_events

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request shape. Tolerant by design: unknown fields (temperature, top_p, ...)
# are ignored rather than rejected, matching how OpenAI clients probe servers.
# ---------------------------------------------------------------------------

class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = None


class _ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[_ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    stream: bool = False


# Internal done_reason -> OpenAI finish_reason. Anything unmapped is a clean stop.
_FINISH_REASONS = {
    "end_turn": "stop",
    "truncated": "length",
    "max_tokens": "length",
    "max_iterations": "length",
}


def _finish_reason(done_reason: str) -> str:
    return _FINISH_REASONS.get(done_reason, "stop")


def _text_of(content: Any) -> str:
    """Flatten an OpenAI message `content` (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                text = p.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _prepare(messages: list[_ChatMessage]) -> tuple[str | None, list[tuple[str, str]], str | None]:
    """Map OpenAI `messages[]` to (system_override, history, prompt).

    `system` messages are concatenated into the system override. The final user
    message is the turn to run (`prompt`); every other user/assistant message
    becomes history seeded into the ephemeral session. `prompt` is None when no
    user message is present.
    """
    systems = [t for m in messages if m.role == "system" and (t := _text_of(m.content))]
    system_override = "\n\n".join(systems) if systems else None

    convo = [(m.role, _text_of(m.content)) for m in messages if m.role in ("user", "assistant")]
    last_user = next((i for i in range(len(convo) - 1, -1, -1) if convo[i][0] == "user"), None)
    if last_user is None:
        return system_override, convo, None

    prompt = convo[last_user][1]
    history = convo[:last_user] + convo[last_user + 1:]
    return system_override, history, prompt


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _error_response(message: str, *, status: int = 400, err_type: str = "invalid_request_error") -> JSONResponse:
    """OpenAI-style error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


def _default_model(settings, registry: Optional[LLMRegistry]) -> str:
    if registry is not None:
        try:
            return registry.default_id()
        except Exception:
            pass
    return settings.llm_model


def _completion_body(answer: str, model: str, usage, done_reason: str) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": _finish_reason(done_reason),
        }],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def _chunk(cid: str, created: int, model: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _stream(model: str, turn_kwargs: dict) -> AsyncIterator[dict]:
    """SSE generator: one role frame, a content delta per TextEvent, a final
    frame carrying finish_reason, then the `[DONE]` sentinel. Each TextEvent is
    mapped explicitly (never dataclasses.asdict -- provider_metadata holds bytes).
    """
    cid = _completion_id()
    created = int(time.time())
    yield {"data": json.dumps(_chunk(cid, created, model, {"role": "assistant"}, None))}

    done_reason = "end_turn"
    try:
        async for event in _turn_events(**turn_kwargs):
            if isinstance(event, TextEvent):
                yield {"data": json.dumps(_chunk(cid, created, model, {"content": event.text}, None))}
            elif isinstance(event, DoneEvent):
                done_reason = event.reason
    except Exception as e:  # noqa: BLE001 -- the stream is already open; surface, don't crash.
        logger.exception("error during /v1 stream")
        err = {"error": {"message": str(e), "type": "server_error", "param": None, "code": None}}
        yield {"data": json.dumps(err)}

    yield {"data": json.dumps(_chunk(cid, created, model, {}, _finish_reason(done_reason)))}
    yield {"data": "[DONE]"}


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    llm: LLMClient = Depends(get_llm),
    mcp: MCPManager = Depends(get_mcp),
    store: SessionStore = Depends(get_store),
    guard: SessionGuard = Depends(get_guard),
    settings = Depends(get_settings_obj),
    orchestrator: Optional[Orchestrator] = Depends(get_orchestrator),
    registry: Optional[LLMRegistry] = Depends(get_registry),
    tracer: Optional[Tracer] = Depends(get_tracer),
):
    try:
        payload = await request.json()
    except Exception:
        return _error_response("request body must be valid JSON")

    try:
        req = _ChatCompletionRequest.model_validate(payload)
    except ValidationError as e:
        return _error_response(f"invalid request: {e}")

    if not req.messages:
        return _error_response("'messages' must be a non-empty array")

    system_override, history, prompt = _prepare(req.messages)
    if not prompt:
        return _error_response("no user message found in 'messages'")

    reported_model = req.model or _default_model(settings, registry)

    # Fresh ephemeral session per request; the client owns the history.
    session = await store.create()
    for role, text in history:
        if role == "user":
            session.append_user(text)
        else:
            session.append_assistant(AssistantMessage(content=[TextBlock(text=text)]))

    turn_kwargs = dict(
        prompt=prompt,
        session=session,
        system_override=system_override,
        preferences=None,
        model_id=req.model,
        llm=llm,
        mcp=mcp,
        store=store,
        guard=guard,
        settings=settings,
        orchestrator=orchestrator,
        registry=registry,
        tracer=tracer,
    )

    if req.stream:
        return EventSourceResponse(_stream(reported_model, turn_kwargs))

    try:
        answer, done_reason, usage = await _run_turn(**turn_kwargs)
    except Exception as e:  # noqa: BLE001 -- never leak a stack trace to the client.
        logger.exception("error handling /v1/chat/completions")
        return _error_response(str(e), status=500, err_type="server_error")

    return JSONResponse(_completion_body(answer, reported_model, usage, done_reason))


@router.get("/v1/models")
async def list_models(
    settings = Depends(get_settings_obj),
    registry: Optional[LLMRegistry] = Depends(get_registry),
) -> JSONResponse:
    ids = registry.model_ids if registry is not None else [settings.llm_model]
    created = int(time.time())
    data = [
        {"id": mid, "object": "model", "created": created, "owned_by": "pyaiharness"}
        for mid in ids
    ]
    return JSONResponse({"object": "list", "data": data})
