# api/openai_compatible.py

"""OpenAI-compatible /v1 adapter over the shared TurnRunner core."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent import (
    DoneEvent,
    ErrorEvent,
    SessionStore,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from llm.schemas import AssistantMessage, TextBlock
from orchestrator import LLMRegistry

from .dependencies import (
    get_registry,
    get_settings_obj,
    get_store,
    get_turn_runner,
    require_api_key,
)
from .turn import TurnRunner

logger = logging.getLogger(__name__)
router = APIRouter()


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = None


class _ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[_ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    stream: bool = False


# Internal done_reason -> OpenAI finish_reason.
_FINISH_REASONS = {
    "end_turn": "stop",
    "empty": "stop",
    "truncated": "length",
    "max_tokens": "length",
    "max_iterations": "length",
    "budget_exceeded": "length",
    "deadline_exceeded": "length",
    "no_progress": "stop",
}


def _finish_reason(done_reason: str) -> str:
    try:
        return _FINISH_REASONS[done_reason]
    except KeyError as exc:
        raise ValueError(f"unsupported completion reason: {done_reason!r}") from exc


def _terminal_error_message(done_reason: str) -> str:
    if done_reason == "llm_error":
        return "LLM call failed"
    return f"unsupported completion reason: {done_reason!r}"


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


# Tool-call presentation for UIs that only render `delta.content`. Rendered
# blocks are stripped from replayed assistant history on the next request.

_TOOL_SUMMARY_MARK = "🔧 "

# Keyed on the marker so model-authored <details> blocks are left alone.
_TOOL_BLOCK_RE = re.compile(
    r"\n*<details>\s*<summary>" + _TOOL_SUMMARY_MARK + r".*?</details>\n*",
    re.DOTALL,
)


def _tool_details(event: ToolResultEvent, args: Any, max_chars: int) -> str:
    """Render one completed tool call as a self-contained collapsible block.

    A single delta avoids exposing unbalanced markdown to progressive renderers.
    """
    icon = "❌" if event.is_error else "✅"
    ms = f" · {event.latency_ms:.0f} ms" if event.latency_ms is not None else ""
    out = [f"\n\n<details>\n<summary>{_TOOL_SUMMARY_MARK}{event.name} {icon}{ms}</summary>\n"]
    if args:
        out.append(f"\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```\n")
    result = event.content
    if len(result) > max_chars:
        result = result[:max_chars] + "\n…[truncated]"
    # Defang embedded HTML so tool output cannot close the presentation block.
    result = result.replace("</details>", "<\u200b/details>")
    out.append(f"\n```\n{result}\n```\n\n</details>\n\n")
    return "".join(out)


def _strip_tool_blocks(text: str) -> str:
    """Remove rendered tool-call blocks from assistant text on the inbound path."""
    return _TOOL_BLOCK_RE.sub("", text)


def _prepare(messages: list[_ChatMessage]) -> tuple[str | None, list[tuple[str, str]], str | None]:
    """Map OpenAI `messages[]` to (system_override, history, prompt).

    The final user message is the active prompt; earlier user/assistant turns
    seed an ephemeral session. Rendered tool blocks are removed from history.
    """
    systems = [t for m in messages if m.role == "system" and (t := _text_of(m.content))]
    system_override = "\n\n".join(systems) if systems else None

    convo: list[tuple[str, str]] = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        text = _text_of(m.content)
        if m.role == "assistant":
            text = _strip_tool_blocks(text)
        convo.append((m.role, text))

    last_user = next((i for i in range(len(convo) - 1, -1, -1) if convo[i][0] == "user"), None)
    if last_user is None:
        return system_override, convo, None

    prompt = convo[last_user][1]
    history = convo[:last_user] + convo[last_user + 1:]
    return system_override, history, prompt


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _error_payload(message: str, *, err_type: str) -> dict[str, Any]:
    """Build the shared OpenAI error envelope for JSON and SSE responses."""
    return {"error": {"message": message, "type": err_type, "param": None, "code": None}}


def _error_response(message: str, *, status: int = 400, err_type: str = "invalid_request_error") -> JSONResponse:
    """Return an OpenAI-style error response."""
    return JSONResponse(
        status_code=status,
        content=_error_payload(message, err_type=err_type),
    )


async def openai_auth_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Reshape a /v1 401 into the OpenAI error envelope; delegate everything else.

    Registered app-wide but scoped to /v1 401s, leaving native /chat untouched.
    """
    if exc.status_code == 401 and request.url.path.startswith("/v1"):
        return _error_response(exc.detail, status=401, err_type="invalid_request_error")
    return await http_exception_handler(request, exc)


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


async def _stream(model: str, runner: TurnRunner, turn: dict, tool_block_max_chars: int) -> AsyncIterator[dict]:
    """Render core events as OpenAI SSE frames.

    Successful streams end with a finish-reason chunk. Failed streams instead
    emit one OpenAI error envelope. Both forms end with the `[DONE]` sentinel.
    """
    cid = _completion_id()
    created = int(time.time())
    yield {"data": json.dumps(_chunk(cid, created, model, {"role": "assistant"}, None))}

    done_reason: str | None = None
    failed = False
    pending_args: dict[str, Any] = {}  # tool_use_id -> input

    def server_error(message: str) -> dict[str, str]:
        return {"data": json.dumps(_error_payload(message, err_type="server_error"))}

    try:
        async for event in runner.events(**turn):
            if failed:
                continue
            if isinstance(event, TextEvent):
                yield {"data": json.dumps(_chunk(cid, created, model, {"content": event.text}, None))}
            elif isinstance(event, ToolCallEvent):
                pending_args[event.id] = event.input
            elif isinstance(event, ToolResultEvent):
                block = _tool_details(event, pending_args.pop(event.id, None), tool_block_max_chars)
                yield {"data": json.dumps(_chunk(cid, created, model, {"content": block}, None))}
            elif isinstance(event, ErrorEvent):
                failed = True
                yield server_error(event.message)
            elif isinstance(event, DoneEvent):
                done_reason = event.reason
                try:
                    _finish_reason(done_reason)
                except ValueError:
                    failed = True
                    yield server_error(_terminal_error_message(done_reason))
    except Exception as e:  # noqa: BLE001 -- the stream is already open; surface, don't crash.
        logger.exception("error during /v1 stream")
        if not failed:
            failed = True
            yield server_error(str(e) or "error during completion stream")

    if not failed:
        if done_reason is None:
            failed = True
            yield server_error("completion stream ended without a terminal event")
        else:
            yield {"data": json.dumps(_chunk(cid, created, model, {}, _finish_reason(done_reason)))}
    yield {"data": "[DONE]"}


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    request: Request,
    store: SessionStore = Depends(get_store),
    settings = Depends(get_settings_obj),
    registry: Optional[LLMRegistry] = Depends(get_registry),
    runner: TurnRunner = Depends(get_turn_runner),
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

    # Fresh ephemeral session per request; the client owns durable history.
    session = await store.create()
    for role, text in history:
        if role == "user":
            session.append_user(text)
        else:
            session.append_assistant(AssistantMessage(content=[TextBlock(text=text)]))

    # `model` is a routing hint; the runner owns singleton dependencies.
    turn = dict(
        prompt=prompt,
        session=session,
        system_override=system_override,
        model_id=req.model,
    )

    if req.stream:
        return EventSourceResponse(
            _stream(
                reported_model,
                runner,
                {**turn, "stream": True},
                settings.openai_tool_block_max_chars,
            )
        )

    try:
        answer, done_reason, usage = await runner.run(**turn)
    except Exception as e:  # noqa: BLE001 -- never leak a stack trace to the client.
        logger.exception("error handling /v1/chat/completions")
        return _error_response(str(e), status=500, err_type="server_error")

    try:
        _finish_reason(done_reason)
    except ValueError:
        return _error_response(
            _terminal_error_message(done_reason), status=500, err_type="server_error"
        )

    return JSONResponse(_completion_body(answer, reported_model, usage, done_reason))


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
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
