# api/openai_compatible.py

"""OpenAI-compatible /v1 adapter over the shared TurnRunner core."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from agent import (
    DoneEvent,
    ErrorEvent,
    ReasoningEvent,
    Session,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .dependencies import (
    get_application_runtime,
    require_api_key,
    turn_http_exception,
)
from .request_body import read_request_body
from application import ApplicationRuntime, PersistencePolicy, TurnRequest, TurnRunner
from llm.schemas import CompletionUsage

logger = logging.getLogger(__name__)
router = APIRouter()
type _OpenAIPayload = dict[str, Any]
type _SSEFrame = dict[str, str]


class _InvalidChatRequest(ValueError):
    """Client-owned conversation content cannot form an executable turn."""


class _TextPart(BaseModel):
    """The sole OpenAI content-part shape this adapter can replay."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: StrictStr


type _MessageContent = StrictStr | Annotated[list[_TextPart], Field(min_length=1)]


class _SystemMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system"]
    content: _MessageContent


class _UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user"]
    content: _MessageContent


class _AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["assistant"]
    content: _MessageContent


type _ChatMessage = Annotated[
    _SystemMessage | _UserMessage | _AssistantMessage,
    Field(discriminator="role"),
]


class _ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[_ChatMessage] = Field(min_length=1)
    model: StrictStr | None = None
    stream: StrictBool = False


@dataclass(frozen=True, slots=True)
class _PreparedChat:
    """Validated OpenAI conversation inputs for one ephemeral turn."""

    system_override: str | None
    history: tuple[tuple[str, str], ...]
    prompt: str


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
    "content_filter": "content_filter",
    "refusal": "content_filter",
}


def _finish_reason(done_reason: str) -> str:
    try:
        return _FINISH_REASONS[done_reason]
    except KeyError as exc:
        raise ValueError(f"unsupported completion reason: {done_reason!r}") from exc


def _terminal_error_message(done_reason: str) -> str:
    if done_reason == "llm_error":
        return "LLM call failed"
    if done_reason == "provider_error":
        return "LLM provider terminated abnormally"
    if done_reason == "incomplete_stream":
        return "LLM stream ended without a terminal provider message"
    return f"unsupported completion reason: {done_reason!r}"


def _text_of(content: _MessageContent) -> str:
    """Flatten a validated OpenAI text message to its replayable text."""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


# Legacy tool-call presentation cleanup for conversations saved before tool
# activity moved out of `delta.content`.

_LEGACY_TOOL_SUMMARY_MARK = "🔧 "

# Keyed on the marker so model-authored <details> blocks are left alone.
_LEGACY_TOOL_BLOCK_RE = re.compile(
    r"\n*<details>\s*<summary>" + _LEGACY_TOOL_SUMMARY_MARK + r".*?</details>\n*",
    re.DOTALL,
)


def _bounded_activity_body(text: str, max_chars: int) -> str:
    """Apply the presentation threshold and retain the existing marker."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n…[truncated]"
    return text


def _fenced_activity_body(text: str, language: str) -> str:
    """Fence arbitrary text without allowing its contents to close the block."""
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", text)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _tool_display_name(name: str) -> str:
    """Make an MCP-qualified tool name easier to scan without changing its identity."""
    return name.replace("__", ".")


def _format_tool_call_activity(
    event: ToolCallEvent,
    max_chars: int,
    *,
    include_details: bool,
) -> str:
    """Render one server-owned tool call as deterministic Markdown."""
    heading = f"> **Tool** `{_tool_display_name(event.name)}` — running"
    if not include_details:
        return f"{heading}\n\n"

    args = json.dumps(
        event.input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ": "),
    )
    args = _bounded_activity_body(args, max_chars)
    return f"{heading}\n\n**Arguments**\n\n{_fenced_activity_body(args, 'json')}\n\n"


def _format_tool_result_activity(
    event: ToolResultEvent,
    max_chars: int,
    *,
    include_details: bool,
) -> str:
    """Render one server-owned tool result as deterministic Markdown."""
    if event.is_error:
        status = "failed"
        latency = (
            f" after {event.latency_ms:.0f} ms" if event.latency_ms is not None else ""
        )
    else:
        status = "completed"
        latency = (
            f" in {event.latency_ms:.0f} ms" if event.latency_ms is not None else ""
        )
    heading = f"> **Tool** `{_tool_display_name(event.name)}` — {status}{latency}"
    if not include_details:
        return f"{heading}\n\n"

    result = _bounded_activity_body(event.content, max_chars)
    return f"{heading}\n\n**Result**\n\n{_fenced_activity_body(result, 'text')}\n\n"


def _strip_legacy_tool_blocks(text: str) -> str:
    """Remove old Hyphae tool blocks from replayed assistant history."""
    return _LEGACY_TOOL_BLOCK_RE.sub("", text)


def _prepare_chat_request(
    messages: list[_ChatMessage],
) -> _PreparedChat:
    """Map OpenAI ``messages[]`` to a validated ephemeral chat value.

    The final user message is the active prompt; earlier user/assistant turns
    seed an ephemeral session. Rendered tool blocks are removed from history.
    """
    systems = [
        text
        for message in messages
        if message.role == "system" and (text := _text_of(message.content))
    ]
    system_override = "\n\n".join(systems) if systems else None

    convo: list[tuple[str, str]] = []
    for message in messages:
        if message.role == "system":
            continue
        convo.append((message.role, _text_of(message.content)))

    if not any(role == "user" for role, _text in convo):
        raise _InvalidChatRequest("no user message found in 'messages'")
    if convo[-1][0] != "user":
        raise _InvalidChatRequest(
            "the final conversational message must have role 'user'"
        )

    prompt = convo[-1][1]
    if not prompt:
        raise _InvalidChatRequest("no user message found in 'messages'")
    history = tuple(
        (
            role,
            _strip_legacy_tool_blocks(text) if role == "assistant" else text,
        )
        for role, text in convo[:-1]
    )
    return _PreparedChat(
        system_override=system_override,
        history=history,
        prompt=prompt,
    )


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _error_payload(message: str, *, err_type: str) -> dict[str, Any]:
    """Build the shared OpenAI error envelope for JSON and SSE responses."""
    return {
        "error": {"message": message, "type": err_type, "param": None, "code": None}
    }


def _error_response(
    message: str, *, status: int = 400, err_type: str = "invalid_request_error"
) -> JSONResponse:
    """Return an OpenAI-style error response."""
    return JSONResponse(
        status_code=status,
        content=_error_payload(message, err_type=err_type),
    )


def _http_error_response(exc: StarletteHTTPException) -> JSONResponse:
    """Preserve an HTTP exception in the OpenAI-compatible error envelope."""
    err_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
    return _error_response(str(exc.detail), status=exc.status_code, err_type=err_type)


async def openai_auth_exception_handler(request: Request, exc: Exception) -> Response:
    """Reshape a /v1 401 into the OpenAI error envelope; delegate everything else.

    Registered app-wide but scoped to /v1 401s, leaving native /chat untouched.
    """
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    if exc.status_code == 401 and request.url.path.startswith("/v1"):
        return _error_response(exc.detail, status=401, err_type="invalid_request_error")
    return await http_exception_handler(request, exc)


def _completion_body(
    answer: str, model: str, usage: CompletionUsage, done_reason: str
) -> _OpenAIPayload:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": _finish_reason(done_reason),
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def _chat_completion_chunk(
    cid: str,
    created: int,
    model: str,
    delta: _OpenAIPayload,
    finish_reason: str | None,
) -> _OpenAIPayload:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _stream_chat_completion(
    runner: TurnRunner,
    turn: TurnRequest,
    tool_activity_mode: Literal["reasoning", "reasoning_full", "hidden"],
    tool_activity_max_chars: int,
) -> AsyncIterator[_SSEFrame]:
    """Render core events as OpenAI SSE frames.

    Successful streams end with a finish-reason chunk. Failed streams instead
    emit one OpenAI error envelope. Both forms end with the `[DONE]` sentinel.
    """
    cid = _completion_id()
    created = int(time.time())

    done_reason: str | None = None
    failed = False
    model: str | None = None
    reasoning_tail = ""

    def server_error_frame(message: str) -> dict[str, str]:
        return {"data": json.dumps(_error_payload(message, err_type="server_error"))}

    def track_reasoning_text(text: str) -> None:
        nonlocal reasoning_tail
        reasoning_tail = (reasoning_tail + text)[-2:]

    def separate_tool_activity(text: str) -> str:
        """Add only the newlines needed to begin a distinct Markdown block."""
        if not reasoning_tail:
            prefix = ""
        elif reasoning_tail.endswith("\n\n"):
            prefix = ""
        elif reasoning_tail.endswith("\n"):
            prefix = "\n"
        else:
            prefix = "\n\n"
        separated = prefix + text
        track_reasoning_text(separated)
        return separated

    try:
        async with runner.open(turn) as execution:
            model = execution.metadata.model_id
            yield {
                "data": json.dumps(
                    _chat_completion_chunk(
                        cid, created, model, {"role": "assistant"}, None
                    )
                )
            }
            async for event in execution.events:
                if failed:
                    continue
                if isinstance(event, TextEvent):
                    yield {
                        "data": json.dumps(
                            _chat_completion_chunk(
                                cid, created, model, {"content": event.text}, None
                            )
                        )
                    }
                elif isinstance(event, ReasoningEvent):
                    if tool_activity_mode != "hidden" and event.text:
                        track_reasoning_text(event.text)
                        yield {
                            "data": json.dumps(
                                _chat_completion_chunk(
                                    cid,
                                    created,
                                    model,
                                    {"reasoning_content": event.text},
                                    None,
                                )
                            )
                        }
                elif isinstance(event, ToolCallEvent):
                    if tool_activity_mode != "hidden":
                        activity = separate_tool_activity(
                            _format_tool_call_activity(
                                event,
                                tool_activity_max_chars,
                                include_details=(
                                    tool_activity_mode == "reasoning_full"
                                ),
                            )
                        )
                        yield {
                            "data": json.dumps(
                                _chat_completion_chunk(
                                    cid,
                                    created,
                                    model,
                                    {"reasoning_content": activity},
                                    None,
                                )
                            )
                        }
                elif isinstance(event, ToolResultEvent):
                    if tool_activity_mode != "hidden":
                        activity = separate_tool_activity(
                            _format_tool_result_activity(
                                event,
                                tool_activity_max_chars,
                                include_details=(
                                    tool_activity_mode == "reasoning_full"
                                ),
                            )
                        )
                        yield {
                            "data": json.dumps(
                                _chat_completion_chunk(
                                    cid,
                                    created,
                                    model,
                                    {"reasoning_content": activity},
                                    None,
                                )
                            )
                        }
                elif isinstance(event, ErrorEvent):
                    failed = True
                    yield server_error_frame(event.message)
                elif isinstance(event, DoneEvent):
                    done_reason = event.reason
                    try:
                        _finish_reason(done_reason)
                    except ValueError:
                        failed = True
                        yield server_error_frame(_terminal_error_message(done_reason))
    except Exception as e:  # noqa: BLE001 -- the stream is already open; surface, don't crash.
        logger.exception("error during /v1 stream")
        if not failed:
            failed = True
            yield server_error_frame(str(e) or "error during completion stream")

    if not failed:
        if done_reason is None:
            failed = True
            yield server_error_frame("completion stream ended without a terminal event")
        else:
            assert model is not None
            yield {
                "data": json.dumps(
                    _chat_completion_chunk(
                        cid, created, model, {}, _finish_reason(done_reason)
                    )
                )
            }
    yield {"data": "[DONE]"}


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    request: Request,
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> Response:
    settings = runtime.settings
    runner = runtime.turn_runner()
    try:
        payload = json.loads((await read_request_body(request)).decode("utf-8"))
    except StarletteHTTPException as exc:
        return _http_error_response(exc)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error_response("request body must be valid JSON")

    try:
        req = _ChatCompletionRequest.model_validate(payload)
    except ValidationError as e:
        return _error_response(f"invalid request: {e}")

    try:
        prepared = _prepare_chat_request(req.messages)
    except _InvalidChatRequest as exc:
        return _error_response(str(exc))

    try:
        runner.validate_model_id(req.model)
    except Exception as exc:
        mapped = turn_http_exception(exc)
        if mapped is None:
            raise
        return _http_error_response(mapped)

    # Fresh ephemeral session per request; the client owns durable history.
    session = Session()
    for role, text in prepared.history:
        if role == "user":
            session.append_user(text)
        else:
            session.append_assistant_text(text)

    # The runner owns singleton dependencies and resolves the executing model.
    turn = TurnRequest(
        prompt=prepared.prompt,
        session=session,
        persistence=PersistencePolicy.EPHEMERAL,
        system_override=prepared.system_override,
        model_id=req.model,
        stream=req.stream,
    )

    if req.stream:
        return EventSourceResponse(
            _stream_chat_completion(
                runner,
                turn,
                settings.openai_compat_tool_activity_mode,
                settings.openai_compat_tool_activity_max_chars,
            )
        )

    try:
        result = await runner.run(turn)
    except Exception as exc:
        mapped = turn_http_exception(exc)
        if mapped is not None:
            return _http_error_response(mapped)
        logger.exception("error handling /v1/chat/completions")
        return _error_response(str(exc), status=500, err_type="server_error")

    try:
        _finish_reason(result.done_reason)
    except ValueError:
        return _error_response(
            _terminal_error_message(result.done_reason),
            status=500,
            err_type="server_error",
        )

    return JSONResponse(
        _completion_body(
            result.answer,
            result.metadata.model_id,
            result.usage,
            result.done_reason,
        )
    )


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(
    runtime: ApplicationRuntime = Depends(get_application_runtime),
) -> JSONResponse:
    runner = runtime.turn_runner()
    try:
        ids = runner.available_model_ids()
    except Exception as exc:
        mapped = turn_http_exception(exc)
        if mapped is None:
            raise
        return _http_error_response(mapped)
    created = int(time.time())
    data = [
        {"id": mid, "object": "model", "created": created, "owned_by": "hyphae"}
        for mid in ids
    ]
    return JSONResponse({"object": "list", "data": data})
