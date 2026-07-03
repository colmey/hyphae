# api/openai_compatible.py

"""
OpenAI-compatible inbound adapter.

A thin translator that lets OpenWebUI / LibreChat (and any OpenAI client) talk to
the harness by pointing `base_url` at `/v1`. It owns only wire-format translation:
OpenAI JSON in -> internal Message list -> the shared core (the `TurnRunner`
seam in api/turn.py) -> an OpenAI `chat.completion` object, or an SSE stream of
`chat.completion.chunk` frames. No orchestration or loop logic lives here --
both invariants are preserved by routing every turn through the one core.

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
import re
import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sse_starlette.sse import EventSourceResponse

from agent import (
    DoneEvent,
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
)
from .turn import TurnRunner

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
    "budget_exceeded": "length",
    "deadline_exceeded": "length",
    "no_progress": "stop",
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


# ---------------------------------------------------------------------------
# Tool-call presentation: outbound render <-> inbound strip.
#
# Chat UIs (OpenWebUI/LibreChat) only render `delta.content`, so the harness's
# server-side tool steps are surfaced by folding each completed call into a
# collapsible <details> block in the streamed content (`_tool_details`). The same
# blocks are stripped back out of assistant history on the inbound path
# (`_strip_tool_blocks`) so they never re-enter the agent's context when a
# stateless client re-feeds prior turns. The 🔧 summary marker is the seam.
# ---------------------------------------------------------------------------

_TOOL_SUMMARY_MARK = "🔧 "
_TOOL_RESULT_MAX = 2000

# One rendered block: <details><summary>🔧 …</summary> … </details>. Keyed on the
# marker so a model-authored <details> is never touched; non-greedy to the first
# close (blocks never nest, and embedded results have their closing tag defanged
# so a result containing "</details>" can't end the block early).
_TOOL_BLOCK_RE = re.compile(
    r"\n*<details>\s*<summary>" + _TOOL_SUMMARY_MARK + r".*?</details>\n*",
    re.DOTALL,
)


def _tool_details(event: ToolResultEvent, args: Any) -> str:
    """Render one completed tool call as a self-contained collapsible block.

    Opened and closed in a single delta so a UI's progressive markdown render
    never sees an unbalanced tag. `args` come from the matching ToolCallEvent
    (the result event doesn't carry them); None omits the args fence.
    """
    icon = "❌" if event.is_error else "✅"
    ms = f" · {event.latency_ms:.0f} ms" if event.latency_ms is not None else ""
    out = [f"\n\n<details>\n<summary>{_TOOL_SUMMARY_MARK}{event.name} {icon}{ms}</summary>\n"]
    if args:
        out.append(f"\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```\n")
    result = event.content
    if len(result) > _TOOL_RESULT_MAX:
        result = result[:_TOOL_RESULT_MAX] + "\n…[truncated]"
    # Defang a closing tag inside the result so it can't end the block early --
    # visually or for the inbound strip regex (zero-width space breaks the tag,
    # stays invisible). open-websearch results can carry raw HTML.
    result = result.replace("</details>", "<\u200b/details>")
    out.append(f"\n```\n{result}\n```\n\n</details>\n\n")
    return "".join(out)


def _strip_tool_blocks(text: str) -> str:
    """Remove rendered tool-call blocks from assistant text on the inbound path."""
    return _TOOL_BLOCK_RE.sub("", text)


def _prepare(messages: list[_ChatMessage]) -> tuple[str | None, list[tuple[str, str]], str | None]:
    """Map OpenAI `messages[]` to (system_override, history, prompt).

    `system` messages are concatenated into the system override. The final user
    message is the turn to run (`prompt`); every other user/assistant message
    becomes history seeded into the ephemeral session. Tool-call presentation
    blocks are stripped from assistant turns so the UI clutter never re-enters the
    agent's context. `prompt` is None when no user message is present.
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


async def _stream(model: str, runner: TurnRunner, turn: dict) -> AsyncIterator[dict]:
    """SSE generator: one role frame, a content delta per TextEvent, a final
    frame carrying finish_reason, then the `[DONE]` sentinel. Each TextEvent is
    mapped explicitly (never dataclasses.asdict -- provider_metadata holds bytes).
    """
    cid = _completion_id()
    created = int(time.time())
    yield {"data": json.dumps(_chunk(cid, created, model, {"role": "assistant"}, None))}

    done_reason = "end_turn"
    pending_args: dict[str, Any] = {}  # tool_use_id -> input; set on call, used on result
    try:
        async for event in runner.events(**turn):
            if isinstance(event, TextEvent):
                yield {"data": json.dumps(_chunk(cid, created, model, {"content": event.text}, None))}
            elif isinstance(event, ToolCallEvent):
                pending_args[event.id] = event.input
            elif isinstance(event, ToolResultEvent):
                block = _tool_details(event, pending_args.pop(event.id, None))
                yield {"data": json.dumps(_chunk(cid, created, model, {"content": block}, None))}
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

    # Fresh ephemeral session per request; the client owns the history.
    session = await store.create()
    for role, text in history:
        if role == "user":
            session.append_user(text)
        else:
            session.append_assistant(AssistantMessage(content=[TextBlock(text=text)]))

    # Only per-request primitives cross the seam; the runner carries the
    # singletons. `model` is a hint -- the orchestrator still picks tools/system.
    turn = dict(
        prompt=prompt,
        session=session,
        system_override=system_override,
        model_id=req.model,
    )

    if req.stream:
        return EventSourceResponse(_stream(reported_model, runner, turn))

    try:
        answer, done_reason, usage = await runner.run(**turn)
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
